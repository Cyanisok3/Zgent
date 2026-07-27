from __future__ import annotations

import asyncio
import datetime
import fcntl
import fnmatch
import json
import logging
import signal
import time
from collections.abc import Coroutine
from datetime import UTC
from pathlib import Path
from typing import IO, Any, cast

from pydantic import BaseModel

import cyan
from cyan.core.bus.commands import (
    AgentRunCommand,
    AgentRunResult,
    CoreShutdownCommand,
    CoreShutdownResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    IncidentDecideCommand,
    IncidentDecideResult,
    JobCancelCommand,
    JobCancelResult,
    JobGetCommand,
    JobGetResult,
    JobListCommand,
    JobListResult,
    JobReadLogCommand,
    JobReadLogResult,
    JobStartCommand,
    JobStartResult,
    PermissionRespondCommand,
    PermissionRespondResult,
    PongResult,
    SessionCloseCommand,
    SessionCloseResult,
    SessionCompactCommand,
    SessionCompactResult,
    SessionCreateCommand,
    SessionCreateResult,
    SessionGetHistoryCommand,
    SessionGetHistoryResult,
    SessionSendMessageCommand,
    SessionSendMessageResult,
)
from cyan.core.bus.envelope import EventPushEnvelope, HandlerError
from cyan.core.bus.events import JobFinishedEvent, JobStartedEvent
from cyan.core.config import CyanConfig, get_config
from cyan.core.events.bus import EventBus
from cyan.core.incidents.coordinator import IncidentCoordinator
from cyan.core.jobs import (
    AttemptRecord,
    FailureRecord,
    JobEvent,
    JobRecord,
    JobSpec,
    JobStore,
    JobSupervisor,
)
from cyan.core.jobs.models import JobEventType
from cyan.core.llm.provider import AnthropicProvider
from cyan.core.logging_setup import setup_logging
from cyan.core.mcp.server import McpServerManager
from cyan.core.permissions.manager import PermissionManager
from cyan.core.permissions.storage import load_policy_file
from cyan.core.runner import AgentRunner, RunProfile
from cyan.core.runs import events_file, new_run_id
from cyan.core.session import Session, SessionManager, SessionStore
from cyan.core.trace.record import TraceRecord
from cyan.core.trace.writer import TraceWriter
from cyan.core.transport.ipc_broadcaster import IpcEventBroadcaster
from cyan.core.transport.socket_server import SocketServer, get_connection_writer

logger = logging.getLogger(__name__)

JOB_NOT_FOUND = -32030
INCIDENT_DECISION_FAILED = -32031
CORE_SHUTTING_DOWN = -32032


def _now() -> str:
    return datetime.datetime.now(UTC).isoformat()


# 将一个文本字段原文替换为字符数和 UTF-8 字节数
def _summarize_trace_text(data: dict[str, Any], field: str) -> None:
    value = str(data.pop(field, ""))
    data[f"{field}_chars"] = len(value)
    data[f"{field}_bytes"] = len(value.encode("utf-8"))


# 将一个映射字段原文替换为键名、字符数和 UTF-8 字节数
def _summarize_trace_mapping(data: dict[str, Any], field: str) -> None:
    value = data.pop(field, {})
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    data[f"{field.removesuffix('s')}_keys"] = (
        sorted(value) if isinstance(value, dict) else []
    )
    data[f"{field}_chars"] = len(encoded)
    data[f"{field}_bytes"] = len(encoded.encode("utf-8"))


# 将事件中的内容型字段替换为尺寸摘要，避免 daemon trace 复制业务 payload
def _trace_event_data(event: BaseModel) -> dict[str, Any]:
    data = event.model_dump()
    event_type = data.get("type")
    if event_type == "tool.call_started":
        _summarize_trace_mapping(data, "params")
    elif event_type == "tool.call_finished":
        _summarize_trace_text(data, "output")
    elif event_type == "tool.call_failed":
        _summarize_trace_text(data, "error_message")
    elif event_type == "permission.requested":
        _summarize_trace_mapping(data, "params")
        _summarize_trace_text(data, "param_preview")
    else:
        content_field = {
            "run.started": "goal",
            "llm.token": "token",
            "log.line": "message",
            "session.message_received": "content",
            "subagent.started": "description",
            "skill.invoked": "arguments",
            "patch.proposed": "summary",
        }.get(str(event_type))
        if content_field is not None:
            _summarize_trace_text(data, content_field)
    return data


# 生成不含 MCP 环境、命令参数或其他凭据值的 daemon 配置日志摘要
def _config_log_data(config: CyanConfig) -> dict[str, Any]:
    return {
        "host": config.host,
        "port": config.port,
        "log_level": config.logging.level,
        "log_format": config.logging.format,
        "model": config.llm.default_model,
        "agent_max_steps": config.agent.max_steps,
        "trace_enabled": config.trace.enabled,
        "mcp_server_count": len(config.mcp.servers),
    }


class CoreApp:
    def __init__(self) -> None:
        self._start_time = time.monotonic()
        self._bus = EventBus()
        self._broadcaster: IpcEventBroadcaster | None = None
        self._trace: TraceWriter | None = None
        self._config: CyanConfig | None = None
        self._running_runs: set[asyncio.Task[Any]] = set()
        self._sessions: SessionManager | None = None
        self._permission_manager: PermissionManager | None = None
        self._mcp_manager: McpServerManager | None = None
        self._job_store: JobStore | None = None
        self._job_supervisor: JobSupervisor | None = None
        self._incidents: IncidentCoordinator | None = None
        self._daemon_lock: IO[str] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._server: SocketServer | None = None
        self._job_start_lock = asyncio.Lock()
        self._shutting_down = False

    # 获取进程级排他锁，阻止第二个 daemon 在端口检查前修改恢复状态
    def _acquire_daemon_lock(self) -> None:
        path = Path("~/.cyan/cyan-core.lock").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open("a+", encoding="utf-8")
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise SystemExit("cyan-core is already running") from None
        self._daemon_lock = lock

    # 处理 core.ping 请求，返回服务版本、运行时长和接收时间
    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        client = params.get("client", "unknown")
        logger.debug("ping from %s", client)
        return PongResult(
            server_version=cyan.__version__,
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
            received_at=datetime.datetime.now(datetime.UTC).isoformat(),
        )

    # 接受已连接本机客户端的停止请求，并在响应写回后唤醒统一清理流程
    async def _shutdown_handler(self, params: dict[str, Any]) -> CoreShutdownResult:
        CoreShutdownCommand.model_validate(params)
        if self._shutdown_event is None:
            raise HandlerError(-32603, "shutdown event is not initialized")
        self._shutting_down = True
        if self._job_supervisor is not None:
            self._job_supervisor.stop_starting()
        async with self._job_start_lock:
            if self._server is not None:
                await self._server.stop_accepting()
        asyncio.get_running_loop().call_later(0.05, self._shutdown_event.set)
        return CoreShutdownResult()

    # 将 EventBus 事件写入 trace（作为 EventBus 订阅者）
    async def _trace_event_handler(self, event: BaseModel) -> None:
        assert self._trace is not None
        event_dict = _trace_event_data(event)
        self._trace.emit(
            TraceRecord(
                ts=_now(),
                direction="CORE",
                layer="event",
                kind="event",
                run_id=event_dict.get("run_id"),
                data=event_dict,
            )
        )

    # 启动一次 agent run：异步创建 AgentRunner 并立即返回 run_id
    async def _agent_run_handler(self, params: dict[str, Any]) -> AgentRunResult:
        assert self._sessions is not None
        cmd = AgentRunCommand.model_validate(params)
        session = await self._sessions.create(mode="one_shot", title=cmd.goal[:40])
        run_id = new_run_id()
        run_task = asyncio.create_task(
            self._sessions.send_message(session.id, cmd.goal, run_id=run_id)
        )
        self._running_runs.add(run_task)
        run_task.add_done_callback(self._running_runs.discard)
        return AgentRunResult(run_id=run_id)

    # 创建 chat 或 one_shot session，并返回 session_id
    async def _session_create_handler(self, params: dict[str, Any]) -> SessionCreateResult:
        assert self._sessions is not None
        cmd = SessionCreateCommand.model_validate(params)
        if cmd.mode == "incident":
            raise HandlerError(-32602, "incident sessions are daemon-owned")
        session = await self._sessions.create(
            mode=cmd.mode,
            title=cmd.title,
            workspace_root=cmd.workspace_root,
        )
        return SessionCreateResult(session_id=session.id, status=session.status)

    # 将 JobSupervisor 的失败回调转交给 IncidentCoordinator
    async def _job_failure_handler(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        failure: FailureRecord,
    ) -> None:
        assert self._incidents is not None
        await self._incidents.handle_failure(job, attempt, failure)

    # 为 Incident session 注入只读工具和不可压缩系统约束
    def _run_profile(self, session: Session) -> RunProfile | None:
        if session.mode != "incident":
            return None
        assert self._incidents is not None
        return self._incidents.profile_for_session(session)

    # 创建真实后台进程 Job 并立即返回其标识
    async def _job_start_handler(self, params: dict[str, Any]) -> JobStartResult:
        assert self._job_supervisor is not None
        cmd = JobStartCommand.model_validate(params)
        root = Path(cmd.workspace_root).expanduser()
        if not root.is_absolute():
            raise HandlerError(-32602, "workspace_root must be absolute")
        async with self._job_start_lock:
            if self._shutting_down:
                raise HandlerError(CORE_SHUTTING_DOWN, "core is shutting down")
            try:
                job = await self._job_supervisor.start(
                    JobSpec(argv=cmd.argv, workspace_root=root, env=cmd.env)
                )
            except (OSError, ValueError) as exc:
                raise HandlerError(-32602, str(exc)) from exc
            assert job.current_attempt_id is not None
            if job.status == "running":
                await self._publish_latest_job_event(job, "job.started")
            task = asyncio.create_task(self._watch_job(job.id), name=f"job-watch-{job.id}")
            self._running_runs.add(task)
            task.add_done_callback(self._running_runs.discard)
            return JobStartResult(job_id=job.id)

    # 将指定 Attempt 最新的持久化状态转换后发布到实时事件流
    async def _publish_latest_job_event(
        self,
        job: JobRecord,
        expected_type: JobEventType,
    ) -> None:
        assert self._job_store is not None
        if job.current_attempt_id is None:
            return
        persisted = self._job_store.find_attempt_event(
            job.id,
            job.current_attempt_id,
            expected_type,
        )
        if persisted is None:
            logger.error(
                "missing persisted job event job_id=%s attempt_id=%s expected=%s",
                job.id,
                job.current_attempt_id,
                expected_type,
            )
            return
        canonical = self._canonical_job_event(persisted)
        if canonical is not None:
            await self._bus.publish(canonical)

    # 等待一次 Job Attempt 结束并广播最终退出状态
    async def _watch_job(self, job_id: str) -> None:
        assert self._job_supervisor is not None
        job = await self._job_supervisor.wait(job_id)
        await self._publish_latest_job_event(
            job,
            cast(JobEventType, f"job.{job.status}"),
        )

    # 返回所有 Job 的安全增强视图
    async def _job_list_handler(self, params: dict[str, Any]) -> JobListResult:
        JobListCommand.model_validate(params)
        assert self._incidents is not None
        return JobListResult(jobs=await self._incidents.list_jobs())

    # 返回单个 Job、当前 Attempt 与 Incident artifact
    async def _job_get_handler(self, params: dict[str, Any]) -> JobGetResult:
        cmd = JobGetCommand.model_validate(params)
        assert self._incidents is not None
        try:
            view = await self._incidents.job_view(cmd.job_id)
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HandlerError(JOB_NOT_FOUND, "job not found") from exc
        return JobGetResult.model_validate(view)

    # 用户明确请求时终止当前 Job 进程组
    async def _job_cancel_handler(self, params: dict[str, Any]) -> JobCancelResult:
        cmd = JobCancelCommand.model_validate(params)
        assert self._job_supervisor is not None
        try:
            job = await self._job_supervisor.cancel(cmd.job_id)
        except FileNotFoundError as exc:
            raise HandlerError(JOB_NOT_FOUND, "job not found") from exc
        return JobCancelResult(status=job.status)

    # 按字节游标返回一段原始 stdout 或 stderr
    async def _job_read_log_handler(self, params: dict[str, Any]) -> JobReadLogResult:
        cmd = JobReadLogCommand.model_validate(params)
        assert self._job_store is not None
        try:
            chunk = self._job_store.read_log(
                cmd.job_id,
                cmd.attempt_id,
                cmd.stream,
                cmd.offset,
                min(cmd.limit, 32 * 1024),
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HandlerError(JOB_NOT_FOUND, "job or attempt not found") from exc
        return JobReadLogResult(
            data=chunk.text,
            next_offset=chunk.end_offset,
            eof=chunk.eof,
        )

    # 执行一次 patch approve/reject 决策及可选 smoke
    async def _incident_decide_handler(
        self,
        params: dict[str, Any],
    ) -> IncidentDecideResult:
        cmd = IncidentDecideCommand.model_validate(params)
        assert self._incidents is not None
        try:
            status = await self._incidents.decide(
                cmd.incident_id,
                cmd.proposal_id,
                cmd.decision,
                run_smoke=cmd.run_smoke,
                smoke_config_fingerprint=cmd.smoke_config_fingerprint,
            )
        except (KeyError, OSError, ValueError) as exc:
            raise HandlerError(INCIDENT_DECISION_FAILED, str(exc)) from exc
        return IncidentDecideResult(status=status)

    # 向 session 发送一条用户消息，后台执行并立即返回可订阅的 run_id
    async def _session_send_handler(self, params: dict[str, Any]) -> SessionSendMessageResult:
        assert self._sessions is not None
        cmd = SessionSendMessageCommand.model_validate(params)
        run_id = new_run_id()
        coroutine: Coroutine[Any, Any, Any]
        if (
            self._incidents is not None
            and self._incidents.is_incident_session(cmd.session_id)
        ):
            try:
                await self._incidents.follow_up(
                    cmd.session_id,
                    cmd.content,
                    run_id,
                )
            except (KeyError, ValueError) as exc:
                raise HandlerError(INCIDENT_DECISION_FAILED, str(exc)) from exc
            return SessionSendMessageResult(run_id=run_id)
        else:
            coroutine = self._sessions.send_message(
                cmd.session_id,
                cmd.content,
                run_id=run_id,
            )
        task = asyncio.create_task(coroutine, name=f"session-run-{run_id}")
        self._running_runs.add(task)
        task.add_done_callback(self._running_runs.discard)
        return SessionSendMessageResult(run_id=run_id)

    # 返回 session 的完整 Anthropic messages 历史
    async def _session_history_handler(self, params: dict[str, Any]) -> SessionGetHistoryResult:
        assert self._sessions is not None
        cmd = SessionGetHistoryCommand.model_validate(params)
        messages = await self._sessions.get_history(cmd.session_id)
        return SessionGetHistoryResult(messages=messages)

    # 接收客户端权限审批响应，resolve 对应挂起的 Future
    async def _permission_respond_handler(self, params: dict[str, Any]) -> PermissionRespondResult:
        cmd = PermissionRespondCommand.model_validate(params)
        logger.info(
            "permission.respond received tool_use_id=%s decision=%s",
            cmd.tool_use_id, cmd.decision,
        )
        if self._permission_manager is None:
            logger.error("permission.respond: PermissionManager not initialized")
            return PermissionRespondResult()
        self._permission_manager.respond(cmd.tool_use_id, cmd.decision)
        return PermissionRespondResult()

    # 手动压缩 session thread，将摘要持久化写入 thread.jsonl
    async def _session_compact_handler(self, params: dict[str, Any]) -> SessionCompactResult:
        assert self._sessions is not None
        cmd = SessionCompactCommand.model_validate(params)
        result = await self._sessions.compact(cmd.session_id, cmd.focus)
        return result  # type: ignore[no-any-return]

    # 关闭 session 并返回 closed 状态
    async def _session_close_handler(self, params: dict[str, Any]) -> SessionCloseResult:
        assert self._sessions is not None
        cmd = SessionCloseCommand.model_validate(params)
        await self._sessions.close(cmd.session_id)
        return SessionCloseResult(status="closed")

    # 注册客户端事件订阅，可选先回放 events.jsonl 历史再接收实时流
    async def _subscribe_handler(self, params: dict[str, Any]) -> EventSubscribeResult:
        cmd = EventSubscribeCommand.model_validate(params)
        writer = get_connection_writer()

        assert self._broadcaster is not None
        sub_id = self._broadcaster.subscribe(writer, cmd.topics, cmd.scope)
        replayed_count = 0
        if cmd.scope.startswith("job:"):
            replayed_count = await self._replay_job_events(
                cmd.scope[4:],
                writer,
                cmd.topics,
                cmd.after_seq,
            )
        elif cmd.replay_from_run is not None:
            replayed_count = await self._replay_events(
                cmd.replay_from_run, writer, cmd.topics
            )

        return EventSubscribeResult(subscription_id=sub_id, replayed_count=replayed_count)

    # 将磁盘 Job 事件映射为实时总线唯一使用的 canonical schema
    def _canonical_job_event(
        self,
        event: JobEvent,
    ) -> JobStartedEvent | JobFinishedEvent | None:
        assert self._job_store is not None
        if event.attempt_id is None:
            return None
        if event.type == "job.started":
            spec = self._job_store.read_spec(event.job_id)
            return JobStartedEvent(
                seq=event.seq,
                job_id=event.job_id,
                attempt_id=event.attempt_id,
                argv=spec.argv,
                workspace_root=str(spec.workspace_root),
                ts=event.occurred_at,
            )
        attempt = self._job_store.read_attempt(event.job_id, event.attempt_id)
        return JobFinishedEvent(
            seq=event.seq,
            job_id=event.job_id,
            attempt_id=event.attempt_id,
            status=event.status,
            exit_code=attempt.returncode,
            signal=attempt.signal,
            ts=event.occurred_at,
        )

    # 从 Job 局部事件流回放 after_seq 之后的状态事件
    async def _replay_job_events(
        self,
        job_id: str,
        writer: asyncio.StreamWriter,
        topics: list[str],
        after_seq: int,
    ) -> int:
        assert self._job_store is not None
        try:
            events = self._job_store.read_events(job_id)
        except (FileNotFoundError, ValueError):
            return 0
        count = 0
        for event in events:
            if event.seq <= after_seq:
                continue
            try:
                canonical = self._canonical_job_event(event)
            except (FileNotFoundError, ValueError):
                logger.warning(
                    "skip incomplete persisted job event job_id=%s seq=%d",
                    job_id,
                    event.seq,
                )
                continue
            if canonical is None or not any(
                fnmatch.fnmatch(canonical.type, pattern) for pattern in topics
            ):
                continue
            writer.write(
                EventPushEnvelope(
                    event=canonical.model_dump(mode="json")
                ).model_dump_json().encode()
                + b"\n"
            )
            count += 1
        if count:
            await writer.drain()
        return count

    # 从 events.jsonl 向 writer 回放匹配 topic 的历史事件，返回已回放条数
    async def _replay_events(
        self,
        run_id: str,
        writer: asyncio.StreamWriter,
        topics: list[str],
    ) -> int:
        path = events_file(run_id)
        if not path.exists():
            for candidate in Path("~/.cyan/sessions").expanduser().glob(
                f"*/runs/{run_id}/events.jsonl"
            ):
                path = candidate
                break
        if not path.exists():
            return 0

        count = 0
        for line in path.read_text().splitlines():
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            event_type: str = event.get("type", "")
            if not any(fnmatch.fnmatch(event_type, p) for p in topics):
                continue
            envelope = EventPushEnvelope(event=event)
            writer.write(envelope.model_dump_json().encode() + b"\n")
            count += 1

        if count:
            await writer.drain()
        return count

    # 启动守护进程：加载配置、初始化日志、启动 trace、启动 TCP 服务器，并等待退出信号
    async def run(self) -> None:
        self._start_time = time.monotonic()
        self._config = get_config()
        setup_logging(self._config)
        self._acquire_daemon_lock()
        self._shutdown_event = asyncio.Event()

        if self._config.trace.enabled:
            trace_path = Path(self._config.trace.file).expanduser()
            self._trace = TraceWriter(trace_path)
            await self._trace.start()
            self._bus.subscribe(self._trace_event_handler)

        policy_file = Path("~/.cyan/policy.toml").expanduser()
        self._permission_manager = PermissionManager(
            policy_file=policy_file,
            timeout_s=self._config.permission.timeout_s,
        )
        logger.info(
            "permission manager: timeout_s=%.1f  persistent=%d entries",
            self._config.permission.timeout_s,
            len(load_policy_file(policy_file)),
        )

        self._broadcaster = IpcEventBroadcaster(trace=self._trace)
        self._bus.subscribe(self._broadcaster.handle)
        sessions_root = Path("~/.cyan/sessions").expanduser()
        jobs_root = Path("~/.cyan/jobs").expanduser()
        self._job_store = JobStore(jobs_root)
        self._job_supervisor = JobSupervisor(
            self._job_store,
            failure_callback=self._job_failure_handler,
        )
        interrupted = await self._job_supervisor.recover_interrupted()
        if interrupted:
            logger.info("recovered %d interrupted job(s)", len(interrupted))
        store = SessionStore(sessions_root)
        assert self._config is not None
        compact_provider = AnthropicProvider(self._config.llm.default_model)

        self._mcp_manager = McpServerManager()

        self._sessions = SessionManager(
            store,
            runner_factory=lambda: AgentRunner(
                self._config,  # type: ignore[arg-type]
                bus=self._bus,
                trace=self._trace,
                permission_manager=self._permission_manager,
                mcp_manager=self._mcp_manager,
                profile_factory=self._run_profile,
            ),
            bus=self._bus,
            provider=compact_provider,
        )
        self._incidents = IncidentCoordinator(
            self._job_store,
            self._sessions,
            self._job_supervisor,
            self._bus,
        )
        await self._incidents.recover()

        server = SocketServer(
            self._config.host,
            self._config.port,
            self._broadcaster,
            trace=self._trace,
        )
        self._server = server
        server.register("core.ping", self._ping_handler)
        server.register("core.shutdown", self._shutdown_handler)
        server.register("agent.run", self._agent_run_handler)
        server.register("event.subscribe", self._subscribe_handler)
        server.register("session.create", self._session_create_handler)
        server.register("session.send_message", self._session_send_handler)
        server.register("session.get_history", self._session_history_handler)
        server.register("session.close", self._session_close_handler)
        server.register("permission.respond", self._permission_respond_handler)
        server.register("session.compact", self._session_compact_handler)
        server.register("job.start", self._job_start_handler)
        server.register("job.list", self._job_list_handler)
        server.register("job.get", self._job_get_handler)
        server.register("job.cancel", self._job_cancel_handler)
        server.register("job.read_log", self._job_read_log_handler)
        server.register("incident.decide", self._incident_decide_handler)

        addr = await server.start()
        logger.info("cyan-core %s listening addr=%s", cyan.__version__, addr)
        logger.info("config: %s", _config_log_data(self._config))

        loop = asyncio.get_running_loop()
        assert self._shutdown_event is not None
        loop.add_signal_handler(signal.SIGINT, self._shutdown_event.set)
        loop.add_signal_handler(signal.SIGTERM, self._shutdown_event.set)

        await self._shutdown_event.wait()

        logger.info("shutting down")
        self._shutting_down = True
        if self._job_supervisor is not None:
            self._job_supervisor.stop_starting()
        async with self._job_start_lock:
            await server.stop_accepting()
        await server.stop()
        if self._incidents is not None:
            await self._incidents.close()
        if self._job_supervisor is not None and self._job_store is not None:
            active_jobs = [
                job.id
                for job in self._job_store.list_jobs()
                if job.status in ("starting", "running")
            ]
            if active_jobs:
                await asyncio.gather(
                    *(self._job_supervisor.cancel(job_id) for job_id in active_jobs),
                    return_exceptions=True,
                )
        for run_task in list(self._running_runs):
            run_task.cancel()
        if self._running_runs:
            await asyncio.gather(*self._running_runs, return_exceptions=True)
        if self._mcp_manager is not None:
            await self._mcp_manager.stop_all()
        if self._trace is not None:
            await self._trace.stop()
        if self._daemon_lock is not None:
            fcntl.flock(self._daemon_lock.fileno(), fcntl.LOCK_UN)
            self._daemon_lock.close()
            self._daemon_lock = None
        self._server = None


# 同步入口：启动 CoreApp 事件循环
def run() -> None:
    asyncio.run(CoreApp().run())
