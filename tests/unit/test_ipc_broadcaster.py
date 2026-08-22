from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import cast
from unittest.mock import AsyncMock, MagicMock

from cyan.service.protocol.events import (
    JobStartedEvent,
    LlmTokenEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepStartedEvent,
)
from cyan.service.transport.ipc_broadcaster import IpcEventBroadcaster


# 创建可控制 drain 行为的 StreamWriter mock
def _make_writer(
    *,
    drain: Callable[[], Awaitable[None]] | None = None,
    drain_raises: Exception | None = None,
) -> asyncio.StreamWriter:
    writer = MagicMock(spec=asyncio.StreamWriter)
    if drain is not None:
        writer.drain = AsyncMock(side_effect=drain)
    elif drain_raises is not None:
        writer.drain = AsyncMock(side_effect=drain_raises)
    else:
        writer.drain = AsyncMock()
    return cast(asyncio.StreamWriter, writer)


# 创建固定时间戳的 run.started 测试事件
def _run_started(run_id: str = "r1") -> RunStartedEvent:
    return RunStartedEvent(run_id=run_id, goal="test", ts="2026-01-01T00:00:00Z")


# 创建固定时间戳的 run.finished 测试事件
def _run_finished(run_id: str = "r1") -> RunFinishedEvent:
    return RunFinishedEvent(
        run_id=run_id,
        status="success",
        steps=1,
        ts="2026-01-01T00:00:00Z",
    )


# 让独立 writer task 获得运行机会并完成即时 drain
async def _flush_writer_tasks() -> None:
    await asyncio.sleep(0)
    await asyncio.sleep(0)


# 解析 writer 捕获的全部事件类型
def _written_event_types(writer: asyncio.StreamWriter) -> list[str]:
    calls = writer.write.call_args_list  # type: ignore[attr-defined]
    return [json.loads(call.args[0].rstrip(b"\n"))["event"]["type"] for call in calls]


# 功能：验证 subscribe 后匹配 topic 的事件由独立写协程写成合法 EventPushEnvelope
# 设计：捕获 writer 字节并在显式让出事件循环后反序列化，避免把入队误判为同步网络写入
async def test_subscriber_receives_matching_event() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()
    broadcaster.subscribe(writer, topics=["run.*"])

    await broadcaster.handle(_run_started())
    await _flush_writer_tasks()

    writer.write.assert_called_once()  # type: ignore[attr-defined]
    data = json.loads(writer.write.call_args[0][0].rstrip(b"\n"))  # type: ignore[attr-defined]
    assert data["kind"] == "event"
    assert data["event"]["type"] == "run.started"
    broadcaster.unsubscribe(writer)


# 功能：验证无订阅时 handle 不向任何 writer 写入数据
# 设计：创建 broadcaster 但不 subscribe，调用 handle 后断言 write 从未被调用以覆盖空 fan-out
async def test_no_subscription_no_write() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()

    await broadcaster.handle(_run_started())

    writer.write.assert_not_called()  # type: ignore[attr-defined]


# 功能：验证 topic glob "step.*" 匹配 step.started 但不匹配 run.started
# 设计：连续发布两种类型并等待队列消费，断言仅有 step 事件写出以锁定 fnmatch 语义
async def test_topic_glob_matches_step_not_run() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()
    broadcaster.subscribe(writer, topics=["step.*"])

    step_event = StepStartedEvent(run_id="r1", step=1, ts="2026-01-01T00:00:00Z")
    await broadcaster.handle(step_event)
    await broadcaster.handle(_run_started())
    await _flush_writer_tasks()

    assert _written_event_types(writer) == ["step.started"]
    broadcaster.unsubscribe(writer)


# 功能：验证 scope="global" 的订阅能收到任意 run_id 的事件
# 设计：发布两个不同 run_id 后检查两次写入，确认 global scope 不过滤 run_id
async def test_scope_global_receives_all_run_ids() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()
    broadcaster.subscribe(writer, topics=["run.*"], scope="global")

    await broadcaster.handle(_run_started("r1"))
    await broadcaster.handle(_run_started("r2"))
    await _flush_writer_tasks()

    assert writer.write.call_count == 2  # type: ignore[attr-defined]
    broadcaster.unsubscribe(writer)


# 功能：验证 scope="run:<id>" 只接收匹配 run_id 的事件
# 设计：发布匹配与不匹配事件并检查单次写入，直接覆盖 run-specific 过滤边界
async def test_scope_run_specific_filters_other_run_ids() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()
    broadcaster.subscribe(writer, topics=["run.*"], scope="run:abc")

    await broadcaster.handle(_run_started("abc"))
    await broadcaster.handle(_run_started("xyz"))
    await _flush_writer_tasks()

    assert writer.write.call_count == 1  # type: ignore[attr-defined]
    broadcaster.unsubscribe(writer)


# 功能：验证 scope="job:<id>" 只接收匹配 job_id 的事件
# 设计：用真实 JobStartedEvent 发布两个 job_id，避免用松散 mock 掩盖字段提取错误
async def test_scope_job_specific_filters_other_job_ids() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer()
    broadcaster.subscribe(writer, topics=["job.*"], scope="job:j1")
    base = {
        "seq": 1,
        "attempt_id": "a1",
        "argv": ["python", "train.py"],
        "workspace_root": "/tmp/project",
        "ts": "2026-01-01T00:00:00Z",
    }

    await broadcaster.handle(JobStartedEvent(job_id="j1", **base))
    await broadcaster.handle(JobStartedEvent(job_id="j2", **base))
    await _flush_writer_tasks()

    assert writer.write.call_count == 1  # type: ignore[attr-defined]
    broadcaster.unsubscribe(writer)


# 功能：验证 unsubscribe 后停止投递且正在等待 drain 的 writer task 被取消
# 设计：让 drain 永久等待并在 CancelledError 路径置位，直接证明清理的不只是订阅列表
async def test_unsubscribe_stops_delivery_and_cancels_writer_task() -> None:
    drain_started = asyncio.Event()
    drain_cancelled = asyncio.Event()
    blocker = asyncio.Event()

    # 模拟慢连接并显式观测写协程取消
    async def blocked_drain() -> None:
        drain_started.set()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            drain_cancelled.set()
            raise

    broadcaster = IpcEventBroadcaster()
    writer = _make_writer(drain=blocked_drain)
    broadcaster.subscribe(writer, topics=["run.*"])
    await broadcaster.handle(_run_started())
    await asyncio.wait_for(drain_started.wait(), timeout=1.0)

    broadcaster.unsubscribe(writer)
    await asyncio.wait_for(drain_cancelled.wait(), timeout=1.0)
    writer.write.reset_mock()  # type: ignore[attr-defined]
    await broadcaster.handle(_run_started())
    await _flush_writer_tasks()

    writer.write.assert_not_called()  # type: ignore[attr-defined]


# 功能：验证一个订阅的慢 drain 不阻塞 handle 或其他订阅的事件写出
# 设计：同时订阅阻塞 writer 与即时 writer，用 wait_for 约束 handle 并观察快 writer 独立推进
async def test_slow_subscriber_does_not_block_publisher_or_peer() -> None:
    blocker = asyncio.Event()

    # 模拟持续背压的客户端连接
    async def blocked_drain() -> None:
        await blocker.wait()

    broadcaster = IpcEventBroadcaster()
    slow_writer = _make_writer(drain=blocked_drain)
    fast_writer = _make_writer()
    broadcaster.subscribe(slow_writer, topics=["run.*"])
    broadcaster.subscribe(fast_writer, topics=["run.*"])

    await asyncio.wait_for(broadcaster.handle(_run_started()), timeout=0.1)
    await _flush_writer_tasks()

    fast_writer.write.assert_called_once()  # type: ignore[attr-defined]
    broadcaster.unsubscribe(slow_writer)
    broadcaster.unsubscribe(fast_writer)


# 功能：验证满队列会丢弃新到达的可重建 llm.token，而不影响已排队事件
# 设计：用容量一和阻塞 drain 固定队列状态，再释放连接检查仅写出在途状态与首个 token
async def test_full_queue_drops_reconstructable_display_event() -> None:
    drain_started = asyncio.Event()
    blocker = asyncio.Event()

    # 在首次写入后冻结消费端以稳定制造满队列
    async def blocked_drain() -> None:
        drain_started.set()
        await blocker.wait()

    broadcaster = IpcEventBroadcaster(queue_size=1)
    writer = _make_writer(drain=blocked_drain)
    broadcaster.subscribe(writer, topics=["*"])
    await broadcaster.handle(_run_started())
    await asyncio.wait_for(drain_started.wait(), timeout=1.0)

    await broadcaster.handle(
        LlmTokenEvent(run_id="r1", token="kept", ts="2026-01-01T00:00:00Z")
    )
    await broadcaster.handle(
        LlmTokenEvent(run_id="r1", token="dropped", ts="2026-01-01T00:00:00Z")
    )
    blocker.set()
    await _flush_writer_tasks()

    assert _written_event_types(writer) == ["run.started", "llm.token"]
    broadcaster.unsubscribe(writer)


# 功能：验证状态事件入队时优先淘汰队列中的可重建展示事件
# 设计：容量一队列先放 token 再放 run.finished，检查最终保留状态事件而非断开客户端
async def test_state_event_evicts_queued_display_event() -> None:
    drain_started = asyncio.Event()
    blocker = asyncio.Event()

    # 在首次状态事件的 drain 上施加背压
    async def blocked_drain() -> None:
        drain_started.set()
        await blocker.wait()

    broadcaster = IpcEventBroadcaster(queue_size=1)
    writer = _make_writer(drain=blocked_drain)
    broadcaster.subscribe(writer, topics=["*"])
    await broadcaster.handle(_run_started())
    await asyncio.wait_for(drain_started.wait(), timeout=1.0)

    await broadcaster.handle(
        LlmTokenEvent(run_id="r1", token="evict", ts="2026-01-01T00:00:00Z")
    )
    await broadcaster.handle(_run_finished())
    blocker.set()
    await _flush_writer_tasks()

    assert _written_event_types(writer) == ["run.started", "run.finished"]
    writer.close.assert_not_called()  # type: ignore[attr-defined]
    broadcaster.unsubscribe(writer)


# 功能：验证状态事件满队列且无展示事件可淘汰时断开慢客户端
# 设计：容量一队列制造一个在途状态和一个排队状态，再发布状态并断言关闭而非丢弃
async def test_full_state_only_queue_disconnects_slow_client() -> None:
    drain_started = asyncio.Event()
    drain_cancelled = asyncio.Event()
    blocker = asyncio.Event()

    # 阻塞消费并记录 broadcaster 发起的任务取消
    async def blocked_drain() -> None:
        drain_started.set()
        try:
            await blocker.wait()
        except asyncio.CancelledError:
            drain_cancelled.set()
            raise

    broadcaster = IpcEventBroadcaster(queue_size=1)
    writer = _make_writer(drain=blocked_drain)
    broadcaster.subscribe(writer, topics=["run.*"])
    await broadcaster.handle(_run_started("in-flight"))
    await asyncio.wait_for(drain_started.wait(), timeout=1.0)

    await broadcaster.handle(_run_started("queued"))
    await asyncio.wait_for(broadcaster.handle(_run_finished("overflow")), timeout=0.1)
    await asyncio.wait_for(drain_cancelled.wait(), timeout=1.0)

    writer.close.assert_called_once()  # type: ignore[attr-defined]
    writer.write.reset_mock()  # type: ignore[attr-defined]
    await broadcaster.handle(_run_started("after-disconnect"))
    await _flush_writer_tasks()
    writer.write.assert_not_called()  # type: ignore[attr-defined]


# 功能：验证写入失败后订阅自动移除且后续事件不再尝试写入
# 设计：让 drain 抛 ConnectionResetError 并让 writer task 运行，随后重发事件验证已清理
async def test_dead_connection_removed_after_failure() -> None:
    broadcaster = IpcEventBroadcaster()
    writer = _make_writer(drain_raises=ConnectionResetError())
    broadcaster.subscribe(writer, topics=["run.*"])

    await broadcaster.handle(_run_started())
    await _flush_writer_tasks()
    assert writer.write.call_count == 1  # type: ignore[attr-defined]

    writer.write.reset_mock()  # type: ignore[attr-defined]
    await broadcaster.handle(_run_started())
    await _flush_writer_tasks()
    writer.write.assert_not_called()  # type: ignore[attr-defined]
