from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import UTC, datetime
from typing import Any, BinaryIO, Literal

from cyan.core.jobs.models import (
    AttemptRecord,
    FailureRecord,
    JobEventType,
    JobRecord,
    JobSpec,
)
from cyan.core.jobs.store import JobStore
from cyan.core.jobs.workflow import (
    ArtifactMetadata,
    WorkflowPhase,
    artifact_is_fresh,
    snapshot_artifact,
    workflow_contract_fingerprint,
)
from cyan.core.processes import read_process_identity, terminate_owned_process_group

logger = logging.getLogger(__name__)

FailureCallback = Callable[
    [JobRecord, AttemptRecord, FailureRecord],
    Awaitable[None] | None,
]
PhaseCallback = Callable[[JobRecord, AttemptRecord], Awaitable[None] | None]


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 生成简短且不可预测的 Job ID
def _new_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


@dataclass
class _RunningAttempt:
    process: asyncio.subprocess.Process | None = None
    monitor: asyncio.Task[None] | None = None
    cancel_requested: bool = False
    launch_failed: bool = False
    launch_ready: asyncio.Event = dataclass_field(default_factory=asyncio.Event)


@dataclass(frozen=True)
class _ProcessResult:
    returncode: int | None
    signal: int | None = None
    timed_out: bool = False
    launch_error: str | None = None
    persistence_error: str | None = None


class JobSupervisor:
    # 初始化后台进程监督器
    def __init__(
        self,
        store: JobStore,
        failure_callback: FailureCallback | None = None,
        phase_callback: PhaseCallback | None = None,
    ) -> None:
        self._store = store
        self._failure_callback = failure_callback
        self._phase_callback = phase_callback
        self._running: dict[str, _RunningAttempt] = {}
        self._lock = asyncio.Lock()
        self._accepting_starts = True

    # 关闭新进程入口，已运行任务仍由 cancel 负责收尾
    def stop_starting(self) -> None:
        self._accepting_starts = False

    # 创建 Job 并启动第一次 Attempt
    async def start(self, spec: JobSpec) -> JobRecord:
        workspace_root = spec.workspace_root.expanduser().resolve(strict=True)
        if not workspace_root.is_dir():
            raise NotADirectoryError(workspace_root)
        resolved_spec = spec.model_copy(
            update={
                "workspace_root": workspace_root,
                "env": dict(spec.env),
            }
        )
        now = _now()
        record = JobRecord(
            id=_new_job_id(),
            status="starting",
            created_at=now,
            updated_at=now,
        )
        async with self._lock:
            if not self._accepting_starts:
                raise RuntimeError("job supervisor is shutting down")
            self._store.create_job(record, resolved_spec)
            await self._launch(record, resolved_spec)
        return self._store.read_job(record.id)

    # 使用持久化启动信息创建下一次 Attempt
    async def retry(self, job_id: str) -> JobRecord:
        async with self._lock:
            if not self._accepting_starts:
                raise RuntimeError("job supervisor is shutting down")
            if job_id in self._running:
                raise RuntimeError(f"job is already running: {job_id}")
            record = self._store.read_job(job_id)
            spec = self._store.read_spec(job_id)
            await self._launch(record, spec)
        return self._store.read_job(job_id)

    # 等待当前 Attempt 结束并返回最新 Job 状态
    async def wait(self, job_id: str) -> JobRecord:
        running = self._running.get(job_id)
        if running is not None and running.monitor is not None:
            await asyncio.shield(running.monitor)
        return self._store.read_job(job_id)

    # 终止 daemon 崩溃后仍存活的原进程组，再封口磁盘中的活跃记录
    async def recover_interrupted(self) -> list[JobRecord]:
        for job in self._store.list_jobs():
            if job.status not in ("starting", "running") or job.current_attempt_id is None:
                continue
            try:
                attempt = self._store.read_attempt(job.id, job.current_attempt_id)
            except FileNotFoundError:
                continue
            await self._terminate_recovered_process(attempt)
        return self._store.recover_interrupted()

    # 请求终止当前进程组并等待状态持久化
    async def cancel(self, job_id: str, timeout: float = 5.0) -> JobRecord:
        running = self._running.get(job_id)
        if running is None:
            return self._store.read_job(job_id)
        if running.monitor is None:
            return self._store.read_job(job_id)
        if running.process is not None and running.process.returncode is not None:
            await asyncio.shield(running.monitor)
            return self._store.read_job(job_id)
        running.cancel_requested = True
        if running.process is not None:
            self._signal_process_group(running.process, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(running.monitor), timeout=timeout)
        except TimeoutError:
            if running.process is not None and running.process.returncode is None:
                self._signal_process_group(running.process, signal.SIGKILL)
            await asyncio.shield(running.monitor)
        return self._store.read_job(job_id)

    # 创建 Attempt 并安排完整 preflight/main/postflight 生命周期
    async def _launch(self, job: JobRecord, spec: JobSpec) -> None:
        attempt_id = f"attempt-{len(job.attempt_ids) + 1:04d}"
        now = _now()
        attempt = AttemptRecord(
            id=attempt_id,
            job_id=job.id,
            status="running",
            started_at=now,
            phase="preflight" if spec.workflow_contract is not None else "main",
        )
        job.status = "running"
        job.updated_at = now
        job.current_attempt_id = attempt_id
        job.attempt_ids.append(attempt_id)
        self._store.write_attempt(attempt)
        self._store.write_job(job)
        self._store.append_event(
            event_type="job.started",
            job_id=job.id,
            attempt_id=attempt.id,
            status=job.status,
            occurred_at=job.updated_at,
        )
        running = _RunningAttempt()
        self._running[job.id] = running
        running.monitor = asyncio.create_task(
            self._run_attempt(job.id, attempt.id, spec),
            name=f"job-monitor-{job.id}-{attempt.id}",
        )
        if spec.workflow_contract is None:
            await running.launch_ready.wait()
            if running.launch_failed and running.monitor is not None:
                await asyncio.shield(running.monitor)

    # 持续排空一个输出流；写盘失败后继续读 pipe 并通知当前执行终止
    async def _drain(
        self,
        reader: asyncio.StreamReader,
        file: BinaryIO,
        persistence_failed: asyncio.Event,
        persistence_errors: list[str],
    ) -> None:
        writable = True
        try:
            while chunk := await reader.read(64 * 1024):
                if not writable:
                    continue
                try:
                    file.write(chunk)
                    file.flush()
                except (OSError, ValueError) as exc:
                    writable = False
                    persistence_errors.append(str(exc))
                    persistence_failed.set()
        except (OSError, ValueError) as exc:
            persistence_errors.append(str(exc))
            persistence_failed.set()

    # 更新 Attempt 当前 phase 并通知可选实时回调
    async def _set_phase(
        self,
        job_id: str,
        attempt_id: str,
        phase: WorkflowPhase,
        check_id: str | None,
    ) -> None:
        attempt = self._store.read_attempt(job_id, attempt_id)
        attempt.phase = phase
        attempt.check_id = check_id
        self._store.write_attempt(attempt)
        if self._phase_callback is None:
            return
        job = self._store.read_job(job_id)
        result = self._phase_callback(job, attempt)
        if inspect.isawaitable(result):
            await result

    # 将确定性 phase boundary 追加到现有 Attempt 日志
    def _append_marker(
        self,
        job_id: str,
        attempt_id: str,
        phase: WorkflowPhase,
        check_id: str | None,
    ) -> None:
        suffix = f" check={check_id}" if check_id is not None else ""
        marker = f"\n[cyan phase={phase}{suffix}]\n".encode()
        for stream in ("stdout", "stderr"):
            path = self._store.log_path(job_id, attempt_id, stream)
            with path.open("ab") as handle:
                handle.write(marker)
                handle.flush()

    # 执行一个 trusted user-owned argv 并复用现有日志和进程组边界
    async def _run_process(
        self,
        job_id: str,
        attempt_id: str,
        spec: JobSpec,
        argv: list[str],
        *,
        phase: WorkflowPhase,
        check_id: str | None = None,
        timeout_s: float | None = None,
    ) -> _ProcessResult:
        if spec.workflow_contract is not None:
            await self._set_phase(job_id, attempt_id, phase, check_id)
            self._append_marker(job_id, attempt_id, phase, check_id)
        stdout_path = self._store.log_path(job_id, attempt_id, "stdout")
        stderr_path = self._store.log_path(job_id, attempt_id, "stderr")
        with stdout_path.open("ab") as stdout_file, stderr_path.open("ab") as stderr_file:
            try:
                process = await asyncio.create_subprocess_exec(
                    *argv,
                    cwd=spec.workspace_root,
                    env=spec.env,
                    stdin=asyncio.subprocess.DEVNULL,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    start_new_session=True,
                )
            except OSError as exc:
                running = self._running[job_id]
                running.launch_failed = True
                running.launch_ready.set()
                return _ProcessResult(returncode=None, launch_error=str(exc))

            running = self._running[job_id]
            running.process = process
            running.launch_ready.set()
            attempt = self._store.read_attempt(job_id, attempt_id)
            attempt.pid = process.pid
            attempt.process_identity = await self._read_process_identity(process.pid)
            self._store.write_attempt(attempt)
            assert process.stdout is not None
            assert process.stderr is not None
            persistence_failed = asyncio.Event()
            persistence_errors: list[str] = []
            stdout_task = asyncio.create_task(
                self._drain(
                    process.stdout,
                    stdout_file,
                    persistence_failed,
                    persistence_errors,
                )
            )
            stderr_task = asyncio.create_task(
                self._drain(
                    process.stderr,
                    stderr_file,
                    persistence_failed,
                    persistence_errors,
                )
            )
            process_task = asyncio.create_task(process.wait())
            failure_task = asyncio.create_task(persistence_failed.wait())
            timeout_task = (
                asyncio.create_task(asyncio.sleep(timeout_s)) if timeout_s is not None else None
            )
            watched: set[asyncio.Task[Any]] = {
                process_task,
                failure_task,
            }
            if timeout_task is not None:
                watched.add(timeout_task)
            done, _ = await asyncio.wait(watched, return_when=asyncio.FIRST_COMPLETED)
            timed_out = timeout_task is not None and timeout_task in done
            if (failure_task in done or timed_out) and process.returncode is None:
                self._signal_process_group(process, signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.shield(process_task), timeout=5.0)
                except TimeoutError:
                    self._signal_process_group(process, signal.SIGKILL)
            returncode = await process_task
            for task in (failure_task, timeout_task):
                if task is not None:
                    task.cancel()
            await asyncio.gather(
                *(task for task in (failure_task, timeout_task) if task is not None),
                return_exceptions=True,
            )
            drain_results = await asyncio.gather(
                stdout_task,
                stderr_task,
                return_exceptions=True,
            )
            for result in drain_results:
                if isinstance(result, BaseException):
                    persistence_errors.append(str(result))

        running.process = None
        attempt = self._store.read_attempt(job_id, attempt_id)
        attempt.pid = None
        attempt.process_identity = None
        self._store.write_attempt(attempt)
        return _ProcessResult(
            returncode=returncode,
            signal=-returncode if returncode < 0 else None,
            timed_out=timed_out,
            persistence_error=persistence_errors[0] if persistence_errors else None,
        )

    # 构造一个带 Contract 定位字段的确定性失败记录
    def _contract_failure(
        self,
        job_id: str,
        attempt_id: str,
        spec: JobSpec,
        *,
        phase: Literal["preflight", "postflight"],
        message: str,
        rule: str,
        check_id: str | None = None,
        artifact_path: str | None = None,
        returncode: int | None = None,
    ) -> FailureRecord:
        return FailureRecord(
            job_id=job_id,
            attempt_id=attempt_id,
            occurred_at=_now(),
            kind="contract_violation",
            returncode=returncode,
            message=message,
            phase=phase,
            check_id=check_id,
            artifact_path=artifact_path,
            contract_fingerprint=workflow_contract_fingerprint(spec.workflow_contract),
            violation_rule=rule,
        )

    # 校验一个 phase 对应的 artifact rules，返回首个确定性违约
    def _validate_artifacts(
        self,
        job_id: str,
        attempt_id: str,
        spec: JobSpec,
        phase: Literal["preflight", "postflight"],
    ) -> FailureRecord | None:
        contract = spec.workflow_contract
        if contract is None:
            return None
        attempt = self._store.read_attempt(job_id, attempt_id)
        baseline = {item.path: item for item in attempt.artifact_baseline}
        roles = {"input", "config"} if phase == "preflight" else {"output"}
        for artifact in contract.artifacts:
            if artifact.role not in roles:
                continue
            try:
                current = snapshot_artifact(spec.workspace_root, artifact)
            except (OSError, ValueError) as exc:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    artifact_path=artifact.path,
                    rule="artifact_stat",
                    message=f"could not inspect workflow artifact {artifact.path}: {exc}",
                )
            if artifact.required and not current.exists:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    artifact_path=artifact.path,
                    rule="required",
                    message=f"required {artifact.role} artifact is missing: {artifact.path}",
                )
            if current.exists and current.size is not None and current.size < artifact.min_bytes:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    artifact_path=artifact.path,
                    rule="min_bytes",
                    message=(
                        f"artifact {artifact.path} has {current.size} bytes; "
                        f"requires at least {artifact.min_bytes}"
                    ),
                )
            if artifact.fresh:
                before = baseline.get(
                    artifact.path,
                    ArtifactMetadata(path=artifact.path, exists=False),
                )
                if not artifact_is_fresh(before, current):
                    return self._contract_failure(
                        job_id,
                        attempt_id,
                        spec,
                        phase=phase,
                        artifact_path=artifact.path,
                        rule="fresh",
                        message=f"output artifact was not updated: {artifact.path}",
                    )
        return None

    # 执行指定 phase 的全部 custom checks，返回首个违约或内部错误
    async def _run_checks(
        self,
        job_id: str,
        attempt_id: str,
        spec: JobSpec,
        phase: Literal["preflight", "postflight"],
    ) -> FailureRecord | None:
        contract = spec.workflow_contract
        if contract is None:
            return None
        for check in (item for item in contract.checks if item.phase == phase):
            result = await self._run_process(
                job_id,
                attempt_id,
                spec,
                check.argv,
                phase=phase,
                check_id=check.id,
                timeout_s=check.timeout_s,
            )
            if result.persistence_error is not None:
                return FailureRecord(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    occurred_at=_now(),
                    kind="supervisor_error",
                    message=f"log persistence failed: {result.persistence_error}",
                    phase=phase,
                    check_id=check.id,
                )
            if result.launch_error is not None:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    check_id=check.id,
                    rule="check_launch",
                    message=f"workflow check {check.id} could not start: {result.launch_error}",
                )
            if result.timed_out:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    check_id=check.id,
                    rule="check_timeout",
                    message=f"workflow check {check.id} timed out after {check.timeout_s}s",
                )
            if result.returncode != 0:
                return self._contract_failure(
                    job_id,
                    attempt_id,
                    spec,
                    phase=phase,
                    check_id=check.id,
                    rule="check_exit",
                    returncode=result.returncode,
                    message=f"workflow check {check.id} exited with status {result.returncode}",
                )
            running = self._running.get(job_id)
            if running is not None and running.cancel_requested:
                return None
        return None

    # 按 frozen JobSpec 执行一次完整 workflow Attempt
    async def _run_attempt(self, job_id: str, attempt_id: str, spec: JobSpec) -> None:
        try:
            if spec.workflow_contract is not None:
                attempt = self._store.read_attempt(job_id, attempt_id)
                attempt.artifact_baseline = [
                    snapshot_artifact(spec.workspace_root, artifact)
                    for artifact in spec.workflow_contract.artifacts
                ]
                self._store.write_attempt(attempt)
                await self._set_phase(job_id, attempt_id, "preflight", None)
                self._append_marker(job_id, attempt_id, "preflight", None)
                failure = self._validate_artifacts(job_id, attempt_id, spec, "preflight")
                if failure is None:
                    failure = await self._run_checks(
                        job_id,
                        attempt_id,
                        spec,
                        "preflight",
                    )
                if failure is not None:
                    await self._finish_attempt(job_id, attempt_id, failure=failure)
                    return
                running = self._running.get(job_id)
                if running is not None and running.cancel_requested:
                    await self._finish_attempt(job_id, attempt_id, cancelled=True)
                    return

            main = await self._run_process(
                job_id,
                attempt_id,
                spec,
                spec.argv,
                phase="main",
            )
            running = self._running.get(job_id)
            if running is not None and running.cancel_requested:
                await self._finish_attempt(job_id, attempt_id, cancelled=True)
                return
            if main.persistence_error is not None:
                failure = FailureRecord(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    occurred_at=_now(),
                    kind="supervisor_error",
                    message=f"log persistence failed: {main.persistence_error}",
                    phase="main",
                )
                await self._finish_attempt(job_id, attempt_id, failure=failure)
                return
            if main.launch_error is not None:
                failure = FailureRecord(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    occurred_at=_now(),
                    kind="launch_error",
                    message=main.launch_error,
                    phase="main",
                )
                await self._finish_attempt(job_id, attempt_id, failure=failure)
                return
            if main.returncode != 0:
                failure = FailureRecord(
                    job_id=job_id,
                    attempt_id=attempt_id,
                    occurred_at=_now(),
                    kind="process_exit",
                    returncode=main.returncode,
                    signal=main.signal,
                    message=f"process exited with status {main.returncode}",
                    phase="main",
                    contract_fingerprint=workflow_contract_fingerprint(
                        spec.workflow_contract
                    ),
                )
                await self._finish_attempt(job_id, attempt_id, failure=failure)
                return

            if spec.workflow_contract is not None:
                await self._set_phase(job_id, attempt_id, "postflight", None)
                self._append_marker(job_id, attempt_id, "postflight", None)
                failure = self._validate_artifacts(job_id, attempt_id, spec, "postflight")
                if failure is None:
                    failure = await self._run_checks(
                        job_id,
                        attempt_id,
                        spec,
                        "postflight",
                    )
                if failure is not None:
                    await self._finish_attempt(job_id, attempt_id, failure=failure)
                    return
                running = self._running.get(job_id)
                if running is not None and running.cancel_requested:
                    await self._finish_attempt(job_id, attempt_id, cancelled=True)
                    return
            await self._finish_attempt(job_id, attempt_id)
        except Exception as exc:
            logger.exception("workflow attempt failed internally job_id=%s", job_id)
            failure = FailureRecord(
                job_id=job_id,
                attempt_id=attempt_id,
                occurred_at=_now(),
                kind="supervisor_error",
                message=str(exc),
                phase=self._store.read_attempt(job_id, attempt_id).phase,
            )
            await self._finish_attempt(job_id, attempt_id, failure=failure)

    # 持久化 Attempt 终态、Job event 和可诊断 Failure
    async def _finish_attempt(
        self,
        job_id: str,
        attempt_id: str,
        *,
        failure: FailureRecord | None = None,
        cancelled: bool = False,
    ) -> None:
        attempt = self._store.read_attempt(job_id, attempt_id)
        job = self._store.read_job(job_id)
        finished_at = _now()
        attempt.finished_at = finished_at
        attempt.pid = None
        attempt.process_identity = None
        job.updated_at = finished_at
        if cancelled:
            attempt.status = "cancelled"
            job.status = "cancelled"
            event_type: JobEventType = "job.cancelled"
        elif failure is not None:
            attempt.status = "failed"
            attempt.returncode = failure.returncode
            attempt.signal = failure.signal
            attempt.error = failure.message if failure.kind == "supervisor_error" else None
            job.status = "failed"
            event_type = "job.failed"
            self._store.write_failure(failure)
        else:
            attempt.status = "succeeded"
            attempt.returncode = 0
            attempt.signal = None
            job.status = "succeeded"
            event_type = "job.succeeded"
        self._store.write_attempt(attempt)
        self._store.write_job(job)
        self._store.append_event(
            event_type=event_type,
            job_id=job.id,
            attempt_id=attempt.id,
            status=job.status,
            occurred_at=finished_at,
        )
        self._running.pop(job_id, None)
        if failure is not None:
            await self._notify_failure(job, attempt, failure)

    # 调用可选失败回调且隔离回调异常
    async def _notify_failure(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        failure: FailureRecord,
    ) -> None:
        if self._failure_callback is None or failure.kind not in {
            "process_exit",
            "contract_violation",
        }:
            return
        try:
            result = self._failure_callback(job, attempt, failure)
            if inspect.isawaitable(result):
                await result
        except Exception:
            logger.exception("job failure callback raised")

    # 向子进程会话对应的整个进程组发送信号
    def _signal_process_group(
        self,
        process: asyncio.subprocess.Process,
        sig: signal.Signals,
    ) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            return

    # 读取不含 argv 的 OS 进程启动身份，用于 daemon 重启时规避 PID 复用
    async def _read_process_identity(self, pid: int) -> str | None:
        return await read_process_identity(pid)

    # 仅在 PID、session、进程组和启动身份全部匹配时终止遗留进程
    async def _terminate_recovered_process(self, attempt: AttemptRecord) -> None:
        if attempt.pid is None or attempt.process_identity is None:
            return
        await terminate_owned_process_group(
            attempt.pid,
            attempt.process_identity,
        )
