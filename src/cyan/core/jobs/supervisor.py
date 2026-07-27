from __future__ import annotations

import asyncio
import inspect
import logging
import os
import signal
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import BinaryIO

from cyan.core.jobs.models import (
    AttemptRecord,
    FailureRecord,
    JobEventType,
    JobRecord,
    JobSpec,
)
from cyan.core.jobs.store import JobStore
from cyan.core.processes import read_process_identity, terminate_owned_process_group

logger = logging.getLogger(__name__)

FailureCallback = Callable[
    [JobRecord, AttemptRecord, FailureRecord],
    Awaitable[None] | None,
]


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 生成简短且不可预测的 Job ID
def _new_job_id() -> str:
    return f"job-{uuid.uuid4().hex[:12]}"


@dataclass
class _RunningAttempt:
    process: asyncio.subprocess.Process
    monitor: asyncio.Task[None]
    cancel_requested: bool = False


class JobSupervisor:
    # 初始化后台进程监督器
    def __init__(
        self,
        store: JobStore,
        failure_callback: FailureCallback | None = None,
    ) -> None:
        self._store = store
        self._failure_callback = failure_callback
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
        if running is not None:
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
        if running.process.returncode is not None:
            await asyncio.shield(running.monitor)
            return self._store.read_job(job_id)
        running.cancel_requested = True
        self._signal_process_group(running.process, signal.SIGTERM)
        try:
            await asyncio.wait_for(asyncio.shield(running.monitor), timeout=timeout)
        except TimeoutError:
            if running.process.returncode is None:
                self._signal_process_group(running.process, signal.SIGKILL)
            await asyncio.shield(running.monitor)
        return self._store.read_job(job_id)

    # 启动子进程并安排后台日志收集与退出监控
    async def _launch(self, job: JobRecord, spec: JobSpec) -> None:
        attempt_id = f"attempt-{len(job.attempt_ids) + 1:04d}"
        now = _now()
        attempt = AttemptRecord(
            id=attempt_id,
            job_id=job.id,
            status="starting",
            started_at=now,
        )
        job.status = "starting"
        job.updated_at = now
        job.current_attempt_id = attempt_id
        job.attempt_ids.append(attempt_id)
        self._store.write_attempt(attempt)
        self._store.write_job(job)

        stdout_path = self._store.log_path(job.id, attempt_id, "stdout")
        stderr_path = self._store.log_path(job.id, attempt_id, "stderr")
        stdout_file = stdout_path.open("wb")
        stderr_file = stderr_path.open("wb")
        try:
            process = await asyncio.create_subprocess_exec(
                *spec.argv,
                cwd=spec.workspace_root,
                env=spec.env,
                stdin=asyncio.subprocess.DEVNULL,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
            )
        except OSError as error:
            stdout_file.close()
            stderr_file.close()
            await self._record_launch_failure(job, attempt, error)
            return

        attempt.status = "running"
        attempt.pid = process.pid
        attempt.process_identity = await self._read_process_identity(process.pid)
        job.status = "running"
        job.updated_at = _now()
        self._store.write_attempt(attempt)
        self._store.write_job(job)
        self._store.append_event(
            event_type="job.started",
            job_id=job.id,
            attempt_id=attempt.id,
            status=job.status,
            occurred_at=job.updated_at,
        )
        monitor = asyncio.create_task(
            self._monitor(job.id, attempt.id, process, stdout_file, stderr_file),
            name=f"job-monitor-{job.id}-{attempt.id}",
        )
        self._running[job.id] = _RunningAttempt(process=process, monitor=monitor)

    # 持续排空一个输出流；写盘失败后继续读 pipe 并通知 monitor 终止进程
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
        finally:
            try:
                file.close()
            except OSError as exc:
                persistence_errors.append(str(exc))
                persistence_failed.set()

    # 等待子进程退出并持久化最终状态
    async def _monitor(
        self,
        job_id: str,
        attempt_id: str,
        process: asyncio.subprocess.Process,
        stdout_file: BinaryIO,
        stderr_file: BinaryIO,
    ) -> None:
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
        done, _ = await asyncio.wait(
            {process_task, failure_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        if failure_task in done and process.returncode is None:
            self._signal_process_group(process, signal.SIGTERM)
            try:
                await asyncio.wait_for(asyncio.shield(process_task), timeout=5.0)
            except TimeoutError:
                self._signal_process_group(process, signal.SIGKILL)
        returncode = await process_task
        failure_task.cancel()
        await asyncio.gather(failure_task, return_exceptions=True)
        drain_results = await asyncio.gather(stdout_task, stderr_task, return_exceptions=True)
        for result in drain_results:
            if isinstance(result, BaseException):
                logger.error("failed to persist process output: %s", result)

        running = self._running.get(job_id)
        cancelled = running is not None and running.cancel_requested
        attempt = self._store.read_attempt(job_id, attempt_id)
        job = self._store.read_job(job_id)
        finished_at = _now()
        attempt.finished_at = finished_at
        attempt.returncode = returncode
        attempt.signal = -returncode if returncode < 0 else None
        job.updated_at = finished_at

        if cancelled:
            attempt.status = "cancelled"
            job.status = "cancelled"
            event_type: JobEventType = "job.cancelled"
        elif persistence_errors:
            attempt.status = "failed"
            attempt.error = f"log persistence failed: {persistence_errors[0]}"
            job.status = "failed"
            event_type = "job.failed"
        elif returncode == 0:
            attempt.status = "succeeded"
            job.status = "succeeded"
            event_type = "job.succeeded"
        else:
            attempt.status = "failed"
            job.status = "failed"
            event_type = "job.failed"

        if attempt.status == "failed":
            failure = FailureRecord(
                job_id=job.id,
                attempt_id=attempt.id,
                occurred_at=finished_at,
                kind="supervisor_error" if persistence_errors else "process_exit",
                returncode=returncode,
                signal=attempt.signal,
                message=(
                    attempt.error or "log persistence failed"
                    if persistence_errors
                    else f"process exited with status {returncode}"
                ),
            )
            self._store.write_failure(failure)

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

        if attempt.status == "failed" and failure.kind == "process_exit":
            await self._notify_failure(job, attempt, failure)

    # 持久化进程无法启动时的失败状态
    async def _record_launch_failure(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        error: OSError,
    ) -> None:
        finished_at = _now()
        attempt.status = "failed"
        attempt.finished_at = finished_at
        attempt.error = str(error)
        job.status = "failed"
        job.updated_at = finished_at
        failure = FailureRecord(
            job_id=job.id,
            attempt_id=attempt.id,
            occurred_at=finished_at,
            kind="launch_error",
            message=str(error),
        )
        self._store.write_failure(failure)
        self._store.write_attempt(attempt)
        self._store.write_job(job)
        self._store.append_event(
            event_type="job.failed",
            job_id=job.id,
            attempt_id=attempt.id,
            status=job.status,
            occurred_at=finished_at,
        )
        await self._notify_failure(job, attempt, failure)

    # 调用可选失败回调且隔离回调异常
    async def _notify_failure(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        failure: FailureRecord,
    ) -> None:
        if self._failure_callback is None or failure.kind != "process_exit":
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
