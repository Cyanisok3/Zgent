from __future__ import annotations

import asyncio
import hashlib
import re
import sys
from datetime import datetime
from pathlib import Path

from cyan.training.incidents.log_tool import FileJobLogReader, JobLogReader, LogStream
from cyan.training.incidents.models import FailureCapsule, Incident, LogSnapshot
from cyan.training.incidents.store import IncidentStore
from cyan.training.jobs.models import FailureRecord
from cyan.training.jobs.store import JobStore

_CAPSULE_LOG_BYTES = 32 * 1024
_EVIDENCE_BUDGET_BYTES = 256 * 1024
_SAFE_ENV_NAMES = frozenset(
    {
        "CONDA_DEFAULT_ENV",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "MKL_NUM_THREADS",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "RANK",
        "VIRTUAL_ENV",
        "WORLD_SIZE",
    }
)
_SAFE_ENV_PREFIXES = ("CUDA_", "NCCL_", "TORCH_")
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "COOKIE", "KEY")
_WORKSPACE_REFERENCE = re.compile(
    r"^(?P<identity>.+@sha256:[0-9a-f]{64})#L"
    r"(?P<start>[1-9][0-9]*)(?:-L(?P<end>[1-9][0-9]*))?$"
)
_LOG_REFERENCE = re.compile(
    r"^(?P<identity>(?:stdout|stderr):[^/]+/[^@]+@bytes:)"
    r"(?P<start>[0-9]+)-(?P<end>[0-9]+)$"
)


# 将支持的 evidence reference 拆成稳定身份和闭区间
def _reference_range(reference: str) -> tuple[str, int, int] | None:
    for pattern in (_WORKSPACE_REFERENCE, _LOG_REFERENCE):
        match = pattern.fullmatch(reference)
        if match is None:
            continue
        start = int(match.group("start"))
        end_group = match.group("end")
        end = int(end_group) if end_group is not None else start
        if end < start:
            return None
        return match.group("identity"), start, end
    return None


# 接受已观察引用本身或其同源子范围，拒绝扩大和身份替换
def _reference_was_observed(reference: str, observed: set[str]) -> bool:
    if reference in observed:
        return True
    requested = _reference_range(reference)
    if requested is None:
        return False
    identity, start, end = requested
    for item in observed:
        registered = _reference_range(item)
        if registered is None:
            continue
        observed_identity, observed_start, observed_end = registered
        if (
            identity == observed_identity
            and observed_start <= start
            and end <= observed_end
        ):
            return True
    return False


# 计算文件 SHA-256；不存在时按空内容计算
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.exists():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


# 从文件尾部读取固定字节并构造可引用快照
def _snapshot_log(path: Path, limit: int) -> LogSnapshot:
    size = path.stat().st_size if path.exists() else 0
    start = max(0, size - max(0, limit))
    raw = b""
    if path.exists() and limit > 0:
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(limit)
    return LogSnapshot(
        size=size,
        sha256=_sha256_file(path),
        included_start=start,
        included_end=start + len(raw),
        tail=raw.decode("utf-8", errors="replace"),
    )


# 在线程中为两个已封口日志计算哈希和 stderr 优先的固定尾部
def _snapshot_failure_logs(
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[LogSnapshot, LogSnapshot]:
    stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
    stderr_limit = min(stderr_size, _CAPSULE_LOG_BYTES)
    stdout_limit = _CAPSULE_LOG_BYTES - stderr_limit
    return (
        _snapshot_log(stdout_path, stdout_limit),
        _snapshot_log(stderr_path, stderr_limit),
    )


# 从任意大小的日志尾部读取有界 UTF-8 文本，不把完整 smoke 输出载入内存
def _tail_text(path: Path, limit: int) -> str:
    if limit <= 0 or not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - limit))
        raw = handle.read(limit)
    return raw.decode("utf-8", errors="replace")


# 从完整启动环境提取有限且不含凭据的诊断摘要
def _safe_environment(env: dict[str, str]) -> dict[str, str]:
    summary = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    for key in sorted(env):
        upper = key.upper()
        if any(marker in upper for marker in _SECRET_MARKERS):
            continue
        if key not in _SAFE_ENV_NAMES and not key.startswith(_SAFE_ENV_PREFIXES):
            continue
        summary[key] = env[key][:512]
    return summary


class _BudgetedJobLogReader(JobLogReader):
    # 将日志 reader 绑定到单个失败 Attempt，并持久化累计读取字节数
    def __init__(
        self,
        jobs: JobStore,
        store: IncidentStore,
        incident: Incident,
        byte_limit: int = _EVIDENCE_BUDGET_BYTES,
    ) -> None:
        self._jobs = jobs
        self._store = store
        self._incident = incident
        self._byte_limit = byte_limit
        self._bytes_read = 0
        self._store.write_evidence_usage(
            incident.id,
            bytes_read=0,
            byte_limit=byte_limit,
        )

    # 校验 Agent 只能访问当前 Incident 对应日志
    def _validate(self, job_id: str, attempt_id: str) -> None:
        if job_id != self._incident.job_id or attempt_id != self._incident.attempt_id:
            raise ValueError("log access is limited to the current incident attempt")

    # 返回当前 Incident 日志流大小
    async def size(self, job_id: str, attempt_id: str, stream: LogStream) -> int:
        self._validate(job_id, attempt_id)
        path = self._jobs.log_path(job_id, attempt_id, stream)
        return path.stat().st_size if path.exists() else 0

    # 在总证据预算内读取日志并记录实际字节数
    async def read(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        limit: int,
    ) -> bytes:
        self._validate(job_id, attempt_id)
        remaining = self._byte_limit - self._bytes_read
        if remaining <= 0:
            raise ValueError("incident log evidence budget exhausted")
        path = self._jobs.log_path(job_id, attempt_id, stream)
        if not path.exists():
            return b""
        with path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(min(limit, remaining))
        self._bytes_read += len(content)
        self._store.write_evidence_usage(
            self._incident.id,
            bytes_read=self._bytes_read,
            byte_limit=self._byte_limit,
        )
        return content

    # 扫描当前 Attempt 的完整日志，扫描字节不计入返回证据预算
    async def search(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        query: bytes,
    ) -> int | None:
        self._validate(job_id, attempt_id)
        reader = FileJobLogReader(self._jobs.log_path)
        return await reader.search(job_id, attempt_id, stream, offset, query)


# 执行短 Git 查询；非 Git 或命令失败时返回 None
async def _git_output(root: Path, *args: str) -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "git",
            "-C",
            str(root),
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
    except (OSError, TimeoutError):
        return None
    if process.returncode != 0:
        return None
    return stdout.decode("utf-8", errors="replace").strip()


# 从已封口的日志、Git 和启动信息构造确定性失败胶囊
async def build_failure_capsule(jobs: JobStore, failure: FailureRecord) -> FailureCapsule:
    spec = jobs.read_spec(failure.job_id)
    root = spec.workspace_root
    stderr_path = jobs.log_path(failure.job_id, failure.attempt_id, "stderr")
    stdout_path = jobs.log_path(failure.job_id, failure.attempt_id, "stdout")
    git_head_result, dirty_output, snapshots = await asyncio.gather(
        _git_output(root, "rev-parse", "HEAD"),
        _git_output(
            root,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        ),
        asyncio.to_thread(
            _snapshot_failure_logs,
            stdout_path,
            stderr_path,
        ),
    )
    stdout_snapshot, stderr_snapshot = snapshots
    dirty_paths = (
        [line[3:] for line in dirty_output.splitlines() if len(line) >= 4][:200]
        if dirty_output is not None
        else []
    )
    return FailureCapsule(
        job_id=failure.job_id,
        attempt_id=failure.attempt_id,
        argv=spec.argv,
        cwd=str(root),
        occurred_at=datetime.fromisoformat(failure.occurred_at),
        failure_kind=failure.kind,
        returncode=failure.returncode,
        signal=failure.signal,
        git_head=git_head_result,
        dirty_paths=dirty_paths,
        environment=_safe_environment(spec.env),
        stdout=stdout_snapshot,
        stderr=stderr_snapshot,
    )
