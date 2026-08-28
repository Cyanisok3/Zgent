from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from cyan.training.incidents.models import FailureCapsule, LogSnapshot
from cyan.training.jobs.models import FailureRecord
from cyan.training.jobs.store import JobStore

_CAPSULE_LOG_BYTES = 32 * 1024
_SMOKE_EVIDENCE_BYTES = 16 * 1024
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


@dataclass(frozen=True)
class SmokeEvidenceBlock:
    source: Literal["stdout", "stderr"]
    reference: str
    description: str
    sha256: str
    start: int
    end: int
    byte_length: int
    content: str


@dataclass(frozen=True)
class SmokeEvidenceSelection:
    content: str
    blocks: tuple[SmokeEvidenceBlock, ...]
    stdout_sha256: str
    stderr_sha256: str

    # 返回本次 Smoke 证据正文的字节数
    @property
    def selected_bytes(self) -> int:
        return sum(block.byte_length for block in self.blocks)


# 从 Smoke 日志尾部构造带三字段 JSON 头的有限证据块
def _smoke_block(
    path: Path,
    source: Literal["stdout", "stderr"],
    incident_id: str,
    proposal_id: str,
    budget: int,
) -> SmokeEvidenceBlock | None:
    if budget <= 0 or not path.exists():
        return None
    size = path.stat().st_size
    if size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(max(0, size - min(size, budget)))
        raw = handle.read(min(size, budget))
    description = f"latest Smoke {source} output"
    while raw:
        start = size - len(raw)
        reference = f"{source}:smoke-{incident_id}/{proposal_id}@bytes:{start}-{size}"
        header = json.dumps(
            {
                "source": source,
                "reference": reference,
                "description": description,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
        encoded_size = len(header.encode("utf-8")) + 1 + len(raw)
        if encoded_size <= budget:
            return SmokeEvidenceBlock(
                source=source,
                reference=reference,
                description=description,
                sha256=_sha256_file(path),
                start=start,
                end=size,
                byte_length=len(raw),
                content=raw.decode("utf-8", errors="replace"),
            )
        raw = raw[max(1, encoded_size - budget) :]
    return None


# 以 stderr 优先、固定总预算选择 Proposal 对应的 Smoke 输出
def select_smoke_evidence(
    stdout_path: Path,
    stderr_path: Path,
    incident_id: str,
    proposal_id: str,
    max_bytes: int = _SMOKE_EVIDENCE_BYTES,
) -> SmokeEvidenceSelection:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    blocks: list[SmokeEvidenceBlock] = []
    remaining = max_bytes
    candidates: tuple[
        tuple[Literal["stderr", "stdout"], Path],
        tuple[Literal["stderr", "stdout"], Path],
    ] = (("stderr", stderr_path), ("stdout", stdout_path))
    for source, path in candidates:
        separator = 1 if blocks else 0
        block = _smoke_block(
            path,
            source,
            incident_id,
            proposal_id,
            remaining - separator,
        )
        if block is None:
            continue
        blocks.append(block)
        rendered = json.dumps(
            {
                "source": block.source,
                "reference": block.reference,
                "description": block.description,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ) + "\n" + block.content
        remaining -= separator + len(rendered.encode("utf-8"))
    return SmokeEvidenceSelection(
        content="\n".join(
            json.dumps(
                {
                    "source": block.source,
                    "reference": block.reference,
                    "description": block.description,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
            + block.content
            for block in blocks
        ),
        blocks=tuple(blocks),
        stdout_sha256=_sha256_file(stdout_path),
        stderr_sha256=_sha256_file(stderr_path),
    )


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
