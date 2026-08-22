from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyan.training.jobs.models import (
    AttemptRecord,
    FailureRecord,
    JobEvent,
    JobEventType,
    JobRecord,
    JobSpec,
    JobStatus,
    LogChunk,
    LogStream,
)

MAX_LOG_READ_BYTES = 32 * 1024
logger = logging.getLogger(__name__)


# 返回当前 UTC 时间的 ISO 8601 字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class JobStore:
    # 初始化基于文件的 Job 存储
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser().resolve()
        self._root.mkdir(parents=True, exist_ok=True)
        self._event_seq: dict[str, int] = {}

    # 返回经过校验的单层路径组件
    def _component(self, value: str, label: str) -> str:
        if not value or Path(value).name != value or value in {".", ".."}:
            raise ValueError(f"invalid {label}: {value!r}")
        return value

    # 返回指定 Job 的目录
    def job_dir(self, job_id: str) -> Path:
        return self._root / self._component(job_id, "job_id")

    # 返回指定 Attempt 的目录
    def attempt_dir(self, job_id: str, attempt_id: str) -> Path:
        return self.job_dir(job_id) / "attempts" / self._component(attempt_id, "attempt_id")

    # 将 JSON 数据原子替换到目标文件
    def _write_json_atomic(self, path: Path, data: dict[str, Any], mode: int | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=2)
                file.write("\n")
                file.flush()
                os.fsync(file.fileno())
            if mode is not None:
                os.chmod(temp_path, mode)
            os.replace(temp_path, path)
            if mode is not None:
                os.chmod(path, mode)
        finally:
            temp_path.unlink(missing_ok=True)

    # 创建 Job 目录并持久化公开状态和私有启动信息
    def create_job(self, record: JobRecord, spec: JobSpec) -> None:
        path = self.job_dir(record.id)
        path.mkdir(parents=True, exist_ok=False)
        self.write_job(record)
        self._write_json_atomic(
            path / "launch.json",
            spec.model_dump(mode="json"),
            mode=0o600,
        )
        (path / "events.jsonl").touch()

    # 原子写入 Job 当前状态
    def write_job(self, record: JobRecord) -> None:
        self._write_json_atomic(
            self.job_dir(record.id) / "job.json",
            record.model_dump(mode="json"),
        )

    # 读取 Job 当前状态
    def read_job(self, job_id: str) -> JobRecord:
        return JobRecord.model_validate_json(
            (self.job_dir(job_id) / "job.json").read_text(encoding="utf-8")
        )

    # 按最近更新时间倒序列出全部 Job
    def list_jobs(self) -> list[JobRecord]:
        records = [
            JobRecord.model_validate_json(path.read_text(encoding="utf-8"))
            for path in self._root.glob("*/job.json")
        ]
        return sorted(records, key=lambda record: record.updated_at, reverse=True)

    # 将 daemon 重启前遗留的活跃 Job 封口为 interrupted
    def recover_interrupted(self) -> list[JobRecord]:
        recovered: list[JobRecord] = []
        for job in self.list_jobs():
            if job.status not in ("starting", "running"):
                continue
            now = _now()
            attempt_id = job.current_attempt_id
            if attempt_id is not None:
                attempt = self.read_attempt(job.id, attempt_id)
                failure_path = self.attempt_dir(job.id, attempt_id) / "failure.json"
                if failure_path.exists():
                    failure = self.read_failure(job.id, attempt_id)
                    attempt.status = "failed"
                    attempt.finished_at = failure.occurred_at
                    job.status = "failed"
                    job.updated_at = failure.occurred_at
                    self.write_attempt(attempt)
                    self.write_job(job)
                    events = self.read_events(job.id)
                    if not events or (
                        events[-1].type,
                        events[-1].attempt_id,
                    ) != ("job.failed", attempt_id):
                        self.append_event(
                            event_type="job.failed",
                            job_id=job.id,
                            attempt_id=attempt_id,
                            status=job.status,
                            occurred_at=failure.occurred_at,
                        )
                    continue
                if attempt.status in ("starting", "running"):
                    attempt.status = "interrupted"
                    attempt.finished_at = now
                    self.write_attempt(attempt)
            job.status = "interrupted"
            job.updated_at = now
            self.write_job(job)
            self.append_event(
                event_type="job.interrupted",
                job_id=job.id,
                attempt_id=attempt_id,
                status=job.status,
                occurred_at=now,
            )
            recovered.append(job)
        return recovered

    # 读取用于精确重跑的私有启动信息
    def read_spec(self, job_id: str) -> JobSpec:
        return JobSpec.model_validate_json(
            (self.job_dir(job_id) / "launch.json").read_text(encoding="utf-8")
        )

    # 原子写入一次 Attempt 的当前状态
    def write_attempt(self, record: AttemptRecord) -> None:
        self._write_json_atomic(
            self.attempt_dir(record.job_id, record.id) / "attempt.json",
            record.model_dump(mode="json"),
        )

    # 读取一次 Attempt 的当前状态
    def read_attempt(self, job_id: str, attempt_id: str) -> AttemptRecord:
        return AttemptRecord.model_validate_json(
            (self.attempt_dir(job_id, attempt_id) / "attempt.json").read_text(
                encoding="utf-8"
            )
        )

    # 原子写入一次失败快照
    def write_failure(self, record: FailureRecord) -> None:
        self._write_json_atomic(
            self.attempt_dir(record.job_id, record.attempt_id) / "failure.json",
            record.model_dump(mode="json"),
        )

    # 读取一次失败快照
    def read_failure(self, job_id: str, attempt_id: str) -> FailureRecord:
        return FailureRecord.model_validate_json(
            (self.attempt_dir(job_id, attempt_id) / "failure.json").read_text(
                encoding="utf-8"
            )
        )

    # 返回一次 Attempt 的日志文件路径
    def log_path(self, job_id: str, attempt_id: str, stream: LogStream) -> Path:
        if stream not in ("stdout", "stderr"):
            raise ValueError(f"invalid log stream: {stream!r}")
        return self.attempt_dir(job_id, attempt_id) / f"{stream}.log"

    # 读取有界日志字节并返回稳定偏移量
    def read_log(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int = 0,
        limit: int = MAX_LOG_READ_BYTES,
    ) -> LogChunk:
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if limit < 1 or limit > MAX_LOG_READ_BYTES:
            raise ValueError(f"limit must be between 1 and {MAX_LOG_READ_BYTES}")

        self.read_attempt(job_id, attempt_id)
        path = self.log_path(job_id, attempt_id, stream)
        if not path.exists():
            total = 0
            data = b""
            start = 0
        else:
            total = path.stat().st_size
            start = min(offset, total)
            with path.open("rb") as file:
                file.seek(start)
                data = file.read(limit)
        end = start + len(data)
        return LogChunk(
            stream=stream,
            start_offset=start,
            end_offset=end,
            total_bytes=total,
            eof=end >= total,
            text=data.decode("utf-8", errors="replace"),
        )

    # 读取有效事件；可在追加前截断崩溃留下的损坏尾部
    def _load_events(self, job_id: str, *, repair_tail: bool) -> list[JobEvent]:
        path = self.job_dir(job_id) / "events.jsonl"
        if not path.exists():
            return []
        events: list[JobEvent] = []
        mode = "rb+" if repair_tail else "rb"
        with path.open(mode) as file:
            while True:
                row_start = file.tell()
                raw = file.readline()
                if not raw:
                    break
                try:
                    event = JobEvent.model_validate_json(raw)
                except ValueError:
                    logger.warning("discard malformed job event suffix path=%s", path)
                    if repair_tail:
                        file.seek(row_start)
                        file.truncate()
                        file.flush()
                        os.fsync(file.fileno())
                    break
                events.append(event)
                if repair_tail and not raw.endswith(b"\n"):
                    file.write(b"\n")
                    file.flush()
                    os.fsync(file.fileno())
        return events

    # 返回 Job 事件文件中最后使用的序号
    def _last_event_seq(self, job_id: str) -> int:
        cached = self._event_seq.get(job_id)
        if cached is not None:
            return cached
        events = self._load_events(job_id, repair_tail=True)
        last = events[-1].seq if events else 0
        self._event_seq[job_id] = last
        return last

    # 追加一条不含原始日志的 Job 状态事件
    def append_event(
        self,
        *,
        event_type: JobEventType,
        job_id: str,
        attempt_id: str | None,
        status: JobStatus,
        occurred_at: str,
    ) -> JobEvent:
        seq = self._last_event_seq(job_id) + 1
        event = JobEvent(
            seq=seq,
            type=event_type,
            job_id=job_id,
            attempt_id=attempt_id,
            status=status,
            occurred_at=occurred_at,
        )
        path = self.job_dir(job_id) / "events.jsonl"
        payload = (event.model_dump_json() + "\n").encode()
        with path.open("ab", buffering=0) as file:
            file.write(payload)
            os.fsync(file.fileno())
        self._event_seq[job_id] = seq
        return event

    # 读取 Job 的全部状态事件
    def read_events(self, job_id: str) -> list[JobEvent]:
        return self._load_events(job_id, repair_tail=False)

    # 返回指定 Attempt 中匹配类型的已持久化状态事件
    def find_attempt_event(
        self,
        job_id: str,
        attempt_id: str,
        event_type: JobEventType,
    ) -> JobEvent | None:
        for event in reversed(self.read_events(job_id)):
            if event.attempt_id == attempt_id and event.type == event_type:
                return event
        return None
