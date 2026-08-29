from __future__ import annotations

import asyncio
import datetime
import fcntl
import fnmatch
import json
import logging
import os
import secrets
import signal
import time
from pathlib import Path
from typing import IO, Any, cast

from pydantic import BaseModel

import cyan
from cyan.agent.events.bus import EventBus
from cyan.agent.llm.base import LLMProvider
from cyan.agent.trace.record import TraceRecord
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig, get_config
from cyan.errors import HandlerError
from cyan.service.logging_setup import setup_logging
from cyan.service.protocol.commands import (
    WIRE_PROTOCOL_VERSION,
    CoreShutdownCommand,
    CoreShutdownResult,
    EventSubscribeCommand,
    EventSubscribeResult,
    IncidentDecideCommand,
    IncidentDecideResult,
    IncidentFollowUpCommand,
    IncidentFollowUpResult,
    IncidentReviewCommand,
    IncidentReviewResult,
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
    LaunchPreviewCommand,
    LaunchPreviewResult,
    LaunchStartCommand,
    LaunchStartResult,
    PingCommand,
    PongResult,
)
from cyan.service.protocol.envelope import EventPushEnvelope
from cyan.service.transport.ipc_broadcaster import IpcEventBroadcaster
from cyan.service.transport.socket_server import SocketServer, get_connection_writer
from cyan.training.events import JobFinishedEvent, JobStartedEvent
from cyan.training.incidents.coordinator import IncidentCoordinator
from cyan.training.jobs import (
    AttemptRecord,
    FailureRecord,
    JobEvent,
    JobRecord,
    JobSpec,
    JobStore,
    JobSupervisor,
)
from cyan.training.jobs.launch import (
    LaunchParseError,
    build_launch_environment,
    launch_fingerprint,
    parse_training_command,
)
from cyan.training.jobs.models import JobEventType

logger = logging.getLogger(__name__)

JOB_NOT_FOUND = -32030
INCIDENT_DECISION_FAILED = -32031
CORE_SHUTTING_DOWN = -32032
INCIDENT_REVIEW_FAILED = -32033
PROTOCOL_INCOMPATIBLE = -32034


# 返回当前 UTC 时间
def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


# 将 trace 中的文本替换为长度摘要
def _summarize_trace_text(data: dict[str, Any], field: str) -> None:
    value = str(data.pop(field, ""))
    data[f"{field}_chars"] = len(value)
    data[f"{field}_bytes"] = len(value.encode("utf-8"))


# 将 trace 中的映射替换为键名和长度摘要
def _summarize_trace_mapping(data: dict[str, Any], field: str) -> None:
    value = data.pop(field, {})
    encoded = json.dumps(value, ensure_ascii=False, default=str)
    data[f"{field}_keys"] = sorted(value) if isinstance(value, dict) else []
    data[f"{field}_chars"] = len(encoded)
    data[f"{field}_bytes"] = len(encoded.encode("utf-8"))


# 生成不复制日志、补丁、工具输出或用户文本的事件摘要
def _trace_event_data(event: BaseModel) -> dict[str, Any]:
    data = event.model_dump()
    event_type = str(data.get("type"))
    if event_type == "tool.call_started":
        _summarize_trace_mapping(data, "params")
    elif event_type == "tool.call_finished":
        _summarize_trace_text(data, "output")
    elif event_type == "tool.call_failed":
        _summarize_trace_text(data, "error_message")
    elif event_type == "run.started":
        _summarize_trace_text(data, "goal")
    elif event_type == "llm.token":
        _summarize_trace_text(data, "token")
    return data


# 生成不含模型扩展、密钥或进程参数的 daemon 配置摘要
def _config_log_data(config: CyanConfig) -> dict[str, Any]:
    return {
        "host": config.host,
        "port": config.port,
        "log_level": config.logging.level,
        "log_format": config.logging.format,
        "model": config.llm.default_model,
        "trace_enabled": config.trace.enabled,
    }


class CoreApp:
    # 初始化双进程 daemon 的训练和 Incident 所有者
    def __init__(self, *, provider: LLMProvider | None = None) -> None:
        self._start_time = time.monotonic()
        self._startup_workspace_root = Path.cwd().resolve()
        self._bus = EventBus()
        self._provider = provider
        self._broadcaster: IpcEventBroadcaster | None = None
        self._trace: TraceWriter | None = None
        self._config: CyanConfig | None = None
        self._running_tasks: set[asyncio.Task[Any]] = set()
        self._job_store: JobStore | None = None
        self._job_supervisor: JobSupervisor | None = None
        self._incidents: IncidentCoordinator | None = None
        self._daemon_lock: IO[str] | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._server: SocketServer | None = None
        self._job_start_lock = asyncio.Lock()
        self._shutting_down = False
        self._startup_ready = False

    # 获取进程级排他锁并写入 v2 daemon 发现信息
    def _acquire_daemon_lock(self) -> None:
        path = Path("~/.cyan/cyan-core.lock").expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        lock = path.open("a+", encoding="utf-8")
        os.fchmod(lock.fileno(), 0o600)
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            lock.close()
            raise SystemExit("cyan-core is already running") from None
        assert self._config is not None
        lock.seek(0)
        lock.truncate()
        lock.write(
            json.dumps(
                {
                    "host": self._config.host,
                    "port": self._config.port,
                    "workspace_root": str(self._startup_workspace_root),
                    "pid": os.getpid(),
                    "protocol_version": WIRE_PROTOCOL_VERSION,
                },
                separators=(",", ":"),
                sort_keys=True,
            )
        )
        lock.flush()
        os.fsync(lock.fileno())
        self._daemon_lock = lock

    # 处理 core.ping 并明确 v2 协议版本
    async def _ping_handler(self, params: dict[str, Any]) -> PongResult:
        command = PingCommand.model_validate(params)
        if (
            command.protocol_version is not None
            and command.protocol_version != WIRE_PROTOCOL_VERSION
        ):
            raise HandlerError(
                PROTOCOL_INCOMPATIBLE,
                f"protocol version {command.protocol_version} is incompatible; "
                f"server uses {WIRE_PROTOCOL_VERSION}",
            )
        return PongResult(
            server_version=cyan.__version__,
            protocol_version=WIRE_PROTOCOL_VERSION,
            startup_workspace_root=str(self._startup_workspace_root),
            uptime_ms=int((time.monotonic() - self._start_time) * 1000),
            received_at=_now(),
        )

    # 接受停止请求并唤醒统一清理流程
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

    # 将 Agent 事件写入脱敏 daemon trace
    async def _trace_event_handler(self, event: BaseModel) -> None:
        if self._trace is not None:
            self._trace.emit(
                TraceRecord(
                    ts=_now(),
                    direction="CORE",
                    layer="event",
                    kind="event",
                    run_id=event.model_dump().get("run_id"),
                    data=_trace_event_data(event),
                )
            )

    # 将 JobSupervisor 的失败回调交给 IncidentCoordinator
    async def _job_failure_handler(
        self, job: JobRecord, attempt: AttemptRecord, failure: FailureRecord
    ) -> None:
        assert self._incidents is not None
        await self._incidents.handle_failure(job, attempt, failure)

    # 在唯一启动锁内创建真实训练进程
    async def _start_job(self, cmd: JobStartCommand) -> JobStartResult:
        assert self._job_supervisor is not None
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
            if job.status == "running":
                await self._publish_latest_job_event(job, "job.started")
            task = asyncio.create_task(self._watch_job(job.id), name=f"job-watch-{job.id}")
            self._running_tasks.add(task)
            task.add_done_callback(self._running_tasks.discard)
            return JobStartResult(job_id=job.id)

    # 校验 job.start 参数并复用训练启动边界
    async def _job_start_handler(self, params: dict[str, Any]) -> JobStartResult:
        return await self._start_job(JobStartCommand.model_validate(params))

    # 解析训练命令并返回确定性预览
    async def _launch_preview_handler(self, params: dict[str, Any]) -> LaunchPreviewResult:
        cmd = LaunchPreviewCommand.model_validate(params)
        root = Path(cmd.workspace_root).expanduser()
        try:
            launch = parse_training_command(cmd.command, root, cmd.env)
            fingerprint = launch_fingerprint(launch, root, cmd.env)
        except (LaunchParseError, OSError, ValueError) as exc:
            raise HandlerError(-32602, str(exc)) from exc
        return LaunchPreviewResult(
            argv=list(launch.argv),
            cwd=str(root.resolve(strict=True)),
            env_overrides=launch.env_overrides,
            executable=launch.executable,
            config_paths=list(launch.config_paths),
            fingerprint=fingerprint,
        )

    # 重新解析并在预览指纹一致时启动训练
    async def _launch_start_handler(self, params: dict[str, Any]) -> LaunchStartResult:
        cmd = LaunchStartCommand.model_validate(params)
        root = Path(cmd.workspace_root).expanduser()
        try:
            launch = parse_training_command(cmd.command, root, cmd.env)
            fingerprint = launch_fingerprint(launch, root, cmd.env)
        except (LaunchParseError, OSError, ValueError) as exc:
            raise HandlerError(-32602, str(exc)) from exc
        if not secrets.compare_digest(fingerprint, cmd.preview_fingerprint):
            raise HandlerError(-32602, "launch preview is stale; preview the command again")
        result = await self._start_job(
            JobStartCommand(
                argv=list(launch.argv),
                workspace_root=str(root.resolve(strict=True)),
                env=build_launch_environment(launch.env_overrides, cmd.env),
            )
        )
        return LaunchStartResult(job_id=result.job_id)

    # 从 JobStore 读取并广播指定状态事件
    async def _publish_latest_job_event(self, job: JobRecord, expected_type: JobEventType) -> None:
        assert self._job_store is not None
        if job.current_attempt_id is None:
            return
        event = self._job_store.find_attempt_event(job.id, job.current_attempt_id, expected_type)
        if event is not None:
            canonical = self._canonical_job_event(event)
            if canonical is not None:
                await self._bus.publish(canonical)

    # 等待一次 Job Attempt 结束并广播最终状态
    async def _watch_job(self, job_id: str) -> None:
        assert self._job_supervisor is not None
        job = await self._job_supervisor.wait(job_id)
        await self._publish_latest_job_event(job, cast(JobEventType, f"job.{job.status}"))

    # 返回所有 Job 的安全增强视图
    async def _job_list_handler(self, params: dict[str, Any]) -> JobListResult:
        JobListCommand.model_validate(params)
        assert self._incidents is not None
        return JobListResult(jobs=await self._incidents.list_jobs())

    # 返回单个 Job、Attempt 与 Incident 当前快照
    async def _job_get_handler(self, params: dict[str, Any]) -> JobGetResult:
        cmd = JobGetCommand.model_validate(params)
        assert self._incidents is not None
        try:
            return JobGetResult.model_validate(await self._incidents.job_view(cmd.job_id))
        except (FileNotFoundError, KeyError, ValueError) as exc:
            raise HandlerError(JOB_NOT_FOUND, "job not found") from exc

    # 用户明确请求时终止 Job 进程组
    async def _job_cancel_handler(self, params: dict[str, Any]) -> JobCancelResult:
        cmd = JobCancelCommand.model_validate(params)
        assert self._job_supervisor is not None
        try:
            job = await self._job_supervisor.cancel(cmd.job_id)
        except FileNotFoundError as exc:
            raise HandlerError(JOB_NOT_FOUND, "job not found") from exc
        return JobCancelResult(status=job.status)

    # 按字节游标返回一段原始训练日志
    async def _job_read_log_handler(self, params: dict[str, Any]) -> JobReadLogResult:
        cmd = JobReadLogCommand.model_validate(params)
        assert self._job_store is not None
        try:
            chunk = self._job_store.read_log(
                cmd.job_id, cmd.attempt_id, cmd.stream, cmd.offset, min(cmd.limit, 32 * 1024)
            )
        except (FileNotFoundError, ValueError) as exc:
            raise HandlerError(JOB_NOT_FOUND, "job or attempt not found") from exc
        return JobReadLogResult(
            data=chunk.text,
            next_offset=chunk.end_offset,
            total_bytes=chunk.total_bytes,
            eof=chunk.eof,
        )

    # 执行 Incident 审批决策
    async def _incident_decide_handler(self, params: dict[str, Any]) -> IncidentDecideResult:
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

    # 返回当前 proposal 的只读前后文本
    async def _incident_review_handler(self, params: dict[str, Any]) -> IncidentReviewResult:
        cmd = IncidentReviewCommand.model_validate(params)
        assert self._incidents is not None
        try:
            path, before, after = self._incidents.review_proposal(
                cmd.job_id, cmd.incident_id, cmd.proposal_id
            )
        except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
            raise HandlerError(INCIDENT_REVIEW_FAILED, str(exc)) from exc
        return IncidentReviewResult(
            proposal_id=cmd.proposal_id, path=path, before_text=before, after_text=after
        )

    # 通过明确 Incident ID 创建一轮追问并先返回 run_id
    async def _incident_follow_up_handler(self, params: dict[str, Any]) -> IncidentFollowUpResult:
        cmd = IncidentFollowUpCommand.model_validate(params)
        assert self._incidents is not None
        try:
            run_id = await self._incidents.follow_up(cmd.incident_id, cmd.content)
        except (KeyError, ValueError) as exc:
            raise HandlerError(INCIDENT_DECISION_FAILED, str(exc)) from exc
        return IncidentFollowUpResult(run_id=run_id)

    # 注册客户端事件订阅并可选回放 Job/Run 摘要
    async def _subscribe_handler(self, params: dict[str, Any]) -> EventSubscribeResult:
        cmd = EventSubscribeCommand.model_validate(params)
        writer = get_connection_writer()
        assert self._broadcaster is not None
        replay_pending = cmd.scope.startswith("job:") or cmd.replay_from_run is not None
        subscription_id = self._broadcaster.subscribe(
            writer,
            cmd.topics,
            cmd.scope,
            replay_pending=replay_pending,
        )
        replayed = 0
        try:
            if cmd.scope.startswith("job:"):
                replayed = await self._replay_job_events(
                    cmd.scope[4:], subscription_id, writer, cmd.topics, cmd.after_seq
                )
            elif cmd.replay_from_run is not None:
                replayed = await self._replay_events(
                    cmd.replay_from_run, subscription_id, writer, cmd.topics
                )
        finally:
            self._broadcaster.activate(subscription_id)
        return EventSubscribeResult(subscription_id=subscription_id, replayed_count=replayed)

    # 将持久化 JobEvent 转换为总线事件
    def _canonical_job_event(self, event: JobEvent) -> JobStartedEvent | JobFinishedEvent | None:
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

    # 回放 Job 局部事件流
    async def _replay_job_events(
        self,
        job_id: str,
        subscription_id: str,
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
        replay: list[dict[str, object]] = []
        for event in events:
            if event.seq <= after_seq:
                continue
            try:
                canonical = self._canonical_job_event(event)
            except (FileNotFoundError, ValueError):
                continue
            if canonical is None or not any(
                fnmatch.fnmatch(canonical.type, pattern) for pattern in topics
            ):
                continue
            replay.append(canonical.model_dump(mode="json"))
            count += 1
        if self._broadcaster is not None:
            await self._broadcaster.replay(subscription_id, replay)
        elif count:
            for replay_event in replay:
                writer.write(
                    EventPushEnvelope(event=replay_event).model_dump_json().encode() + b"\n"
                )
            await writer.drain()
        return count

    # 在所有 Incident run 目录中寻找并回放摘要事件
    async def _replay_events(
        self,
        run_id: str,
        subscription_id: str,
        writer: asyncio.StreamWriter,
        topics: list[str],
    ) -> int:
        assert self._job_store is not None
        paths = self._job_store._root.glob(f"*/incidents/*/runs/{run_id}/events.jsonl")
        path = next(iter(paths), None)
        if path is None or not path.exists():
            return 0
        count = 0
        replay: list[dict[str, object]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if any(fnmatch.fnmatch(str(event.get("type", "")), pattern) for pattern in topics):
                replay.append(event)
                count += 1
        if self._broadcaster is not None:
            await self._broadcaster.replay(subscription_id, replay)
        elif count:
            for replay_event in replay:
                writer.write(
                    EventPushEnvelope(event=replay_event).model_dump_json().encode() + b"\n"
                )
            await writer.drain()
        return count

    # 启动 daemon、恢复训练和 Incident，并注册 v2 RPC
    async def run(self) -> None:
        self._start_time = time.monotonic()
        self._config = get_config()
        setup_logging(self._config)
        self._acquire_daemon_lock()
        self._shutdown_event = asyncio.Event()
        if self._config.trace.enabled:
            self._trace = TraceWriter(Path(self._config.trace.file).expanduser())
            await self._trace.start()
            self._bus.subscribe(self._trace_event_handler)
        self._broadcaster = IpcEventBroadcaster(trace=self._trace)
        self._bus.subscribe(self._broadcaster.handle)
        self._job_store = JobStore(Path("~/.cyan/jobs").expanduser())
        self._job_supervisor = JobSupervisor(
            self._job_store, failure_callback=self._job_failure_handler
        )
        await self._job_supervisor.recover_interrupted()
        self._incidents = IncidentCoordinator(
            self._job_store,
            self._job_supervisor,
            self._bus,
            self._config,
            provider=self._provider,
            trace=self._trace,
        )
        await self._incidents.recover()
        server = SocketServer(
            self._config.host, self._config.port, self._broadcaster, trace=self._trace
        )
        self._server = server
        for method, handler in {
            "core.ping": self._ping_handler,
            "core.shutdown": self._shutdown_handler,
            "event.subscribe": self._subscribe_handler,
            "launch.preview": self._launch_preview_handler,
            "launch.start": self._launch_start_handler,
            "job.start": self._job_start_handler,
            "job.list": self._job_list_handler,
            "job.get": self._job_get_handler,
            "job.cancel": self._job_cancel_handler,
            "job.read_log": self._job_read_log_handler,
            "incident.decide": self._incident_decide_handler,
            "incident.review": self._incident_review_handler,
            "incident.follow_up": self._incident_follow_up_handler,
        }.items():
            server.register(method, handler)
        addr = await server.start()
        self._startup_ready = True
        logger.info("cyan-core %s listening addr=%s", cyan.__version__, addr)
        logger.info("config: %s", _config_log_data(self._config))
        loop = asyncio.get_running_loop()
        loop.add_signal_handler(signal.SIGINT, self._shutdown_event.set)
        loop.add_signal_handler(signal.SIGTERM, self._shutdown_event.set)
        await self._shutdown_event.wait()
        self._shutting_down = True
        self._job_supervisor.stop_starting()
        async with self._job_start_lock:
            await server.stop_accepting()
        await server.stop()
        if self._incidents is not None:
            await self._incidents.close()
        if self._job_supervisor is not None:
            active = (
                [
                    job.id
                    for job in self._job_store.list_jobs()
                    if job.status in ("starting", "running")
                ]
                if self._job_store
                else []
            )
            await asyncio.gather(
                *(self._job_supervisor.cancel(job_id) for job_id in active), return_exceptions=True
            )
        for task in list(self._running_tasks):
            task.cancel()
        if self._running_tasks:
            await asyncio.gather(*self._running_tasks, return_exceptions=True)
        if self._trace is not None:
            await self._trace.stop()
        if self._daemon_lock is not None:
            self._daemon_lock.seek(0)
            self._daemon_lock.truncate()
            self._daemon_lock.flush()
            fcntl.flock(self._daemon_lock.fileno(), fcntl.LOCK_UN)
            self._daemon_lock.close()
            self._daemon_lock = None
        self._server = None


# 同步入口：运行 daemon 事件循环
def run() -> None:
    app = CoreApp()
    try:
        asyncio.run(app.run())
    except (Exception, SystemExit) as exc:
        if not app._startup_ready:
            logger.error("cyan-core startup failed: %s", exc)
        raise
