from __future__ import annotations

import asyncio
import fnmatch
import logging
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import BaseModel

from cyan.agent.trace.record import TraceRecord
from cyan.agent.trace.writer import TraceWriter
from cyan.service.protocol.envelope import EventPushEnvelope

logger = logging.getLogger(__name__)

_DEFAULT_QUEUE_SIZE = 256
_DROPPABLE_EVENT_TYPES = frozenset({"llm.token", "tool.call_finished"})


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class _QueuedEvent:
    payload: bytes
    event_type: str
    run_id: str | None


@dataclass
class _Subscription:
    sub_id: str
    writer: asyncio.StreamWriter
    topics: list[str]
    scope: str
    queue: asyncio.Queue[_QueuedEvent]
    write_lock: asyncio.Lock
    task: asyncio.Task[None] | None = None


class IpcEventBroadcaster:
    # 创建使用独立有界队列的事件广播器
    def __init__(
        self,
        trace: TraceWriter | None = None,
        queue_size: int = _DEFAULT_QUEUE_SIZE,
    ) -> None:
        if queue_size <= 0:
            raise ValueError("queue_size must be positive")
        self._subscriptions: list[_Subscription] = []
        self._trace = trace
        self._queue_size = queue_size

    # 注册一个客户端订阅，返回 subscription_id
    def subscribe(
        self,
        writer: asyncio.StreamWriter,
        topics: list[str],
        scope: str = "global",
    ) -> str:
        sub_id = f"sub-{uuid.uuid4().hex[:8]}"
        sub = _Subscription(
            sub_id=sub_id,
            writer=writer,
            topics=topics,
            scope=scope,
            queue=asyncio.Queue(maxsize=self._queue_size),
            write_lock=asyncio.Lock(),
        )
        sub.task = asyncio.create_task(
            self._write_loop(sub),
            name=f"ipc-event-writer-{sub_id}",
        )
        self._subscriptions.append(sub)
        return sub_id

    # 在实时事件之前按历史顺序写入订阅回放，避免同一 writer 并发交错
    async def replay(self, writer: asyncio.StreamWriter, events: list[dict[str, object]]) -> int:
        sub = next((item for item in self._subscriptions if item.writer is writer), None)
        if sub is None:
            return 0
        count = 0
        try:
            async with sub.write_lock:
                for event in events:
                    event_type = str(event.get("type", ""))
                    run_id = event.get("run_id")
                    item = _QueuedEvent(
                        payload=EventPushEnvelope(event=event).model_dump_json().encode() + b"\n",
                        event_type=event_type,
                        run_id=str(run_id) if run_id is not None else None,
                    )
                    writer.write(item.payload)
                    await writer.drain()
                    self._trace_push(sub, item)
                    count += 1
        except (ConnectionResetError, BrokenPipeError, OSError):
            self._remove_writer(writer, skip_task=asyncio.current_task())
            try:
                writer.close()
            except Exception:
                pass
        return count

    # 移除指定 writer 的所有订阅并取消各自写协程
    def unsubscribe(self, writer: asyncio.StreamWriter) -> None:
        self._remove_writer(writer)

    # 将事件非阻塞放入所有匹配订阅的独立有界队列
    async def handle(self, event: BaseModel) -> None:
        event_dict = event.model_dump()
        event_type: str = event_dict.get("type", "")
        run_id: str | None = event_dict.get("run_id")
        job_id: str | None = event_dict.get("job_id")
        queued_event: _QueuedEvent | None = None

        for sub in list(self._subscriptions):
            if not self._matches_topic(event_type, sub.topics):
                continue
            if not self._matches_scope(run_id, job_id, sub.scope):
                continue
            if queued_event is None:
                envelope = EventPushEnvelope(event=event_dict)
                queued_event = _QueuedEvent(
                    payload=envelope.model_dump_json().encode() + b"\n",
                    event_type=event_type,
                    run_id=run_id,
                )
            try:
                sub.queue.put_nowait(queued_event)
            except asyncio.QueueFull:
                if self._is_droppable(event_type):
                    logger.debug(
                        "dropping display event %s for slow sub %s",
                        event_type,
                        sub.sub_id,
                    )
                    continue
                if self._evict_droppable(sub):
                    sub.queue.put_nowait(queued_event)
                    continue
                logger.warning(
                    "disconnecting slow client for full state-event queue %s",
                    sub.sub_id,
                )
                self._disconnect(sub.writer)

    # 按队列顺序向单个订阅写事件，连接失败时清理该 writer 的全部订阅
    async def _write_loop(self, sub: _Subscription) -> None:
        try:
            while True:
                item = await sub.queue.get()
                try:
                    async with sub.write_lock:
                        sub.writer.write(item.payload)
                        await sub.writer.drain()
                        self._trace_push(sub, item)
                finally:
                    sub.queue.task_done()
        except asyncio.CancelledError:
            raise
        except (ConnectionResetError, BrokenPipeError, OSError):
            logger.debug("dead connection for sub %s, removing subscriptions", sub.sub_id)
            self._remove_writer(sub.writer, skip_task=asyncio.current_task())
            try:
                sub.writer.close()
            except Exception:
                pass

    # 从满队列中移除最早的可重建展示事件并保持其余顺序
    def _evict_droppable(self, sub: _Subscription) -> bool:
        retained: list[_QueuedEvent] = []
        evicted = False
        while True:
            try:
                item = sub.queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            sub.queue.task_done()
            if not evicted and self._is_droppable(item.event_type):
                evicted = True
            else:
                retained.append(item)
        for item in retained:
            sub.queue.put_nowait(item)
        return evicted

    # 移除一个 writer 的全部订阅，可跳过当前正在退出的写协程
    def _remove_writer(
        self,
        writer: asyncio.StreamWriter,
        skip_task: asyncio.Task[object] | None = None,
    ) -> None:
        removed = [sub for sub in self._subscriptions if sub.writer is writer]
        self._subscriptions = [sub for sub in self._subscriptions if sub.writer is not writer]
        for sub in removed:
            if sub.task is not None and sub.task is not skip_task:
                sub.task.cancel()

    # 移除并关闭无法跟上状态事件的慢客户端
    def _disconnect(self, writer: asyncio.StreamWriter) -> None:
        self._remove_writer(writer)
        try:
            writer.close()
        except Exception:
            pass

    # 记录成功写出的事件推送 trace
    def _trace_push(self, sub: _Subscription, item: _QueuedEvent) -> None:
        if self._trace is None:
            return
        client_id = str(sub.writer.get_extra_info("peername", "<unknown>"))
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE→CLIENT",
                layer="ipc",
                kind="push",
                run_id=item.run_id,
                client_id=client_id,
                data={"sub_id": sub.sub_id, "event_type": item.event_type},
            )
        )

    # 判断事件是否属于队列拥塞时允许丢弃的可重建展示更新
    @staticmethod
    def _is_droppable(event_type: str) -> bool:
        return event_type in _DROPPABLE_EVENT_TYPES

    # 检查事件类型是否匹配订阅的 topic 列表（支持 fnmatch glob 模式）
    @staticmethod
    def _matches_topic(event_type: str, topics: list[str]) -> bool:
        return any(fnmatch.fnmatch(event_type, pattern) for pattern in topics)

    # 检查事件标识是否匹配 global、run:<id> 或 job:<id> scope
    @staticmethod
    def _matches_scope(run_id: str | None, job_id: str | None, scope: str) -> bool:
        if scope == "global":
            return True
        if scope.startswith("run:"):
            return run_id == scope[4:]
        if scope.startswith("job:"):
            return job_id == scope[4:]
        return False
