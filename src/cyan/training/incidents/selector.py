from __future__ import annotations

import hashlib
import re
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Literal

from pydantic import BaseModel, ConfigDict, Field

from cyan.training.incidents.models import FailureCapsule


class EvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["stdout", "stderr"]
    start: int = Field(ge=0)
    end: int = Field(ge=0)
    kind: str
    selection_reason: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    # 转换为 Incident 工具可验证的稳定 byte-range 引用
    def as_reference(self, capsule: FailureCapsule) -> str:
        return f"{self.source}:{capsule.job_id}/{capsule.attempt_id}@bytes:{self.start}-{self.end}"


class EvidenceSelection(BaseModel):
    model_config = ConfigDict(extra="forbid")

    content: str
    references: list[EvidenceReference]
    scanned_bytes: int = Field(ge=0)
    selected_bytes: int = Field(ge=0)
    duplicates_removed: int = Field(ge=0)
    stdout_sha256: str
    stderr_sha256: str


@dataclass
class _Candidate:
    source: Literal["stdout", "stderr"]
    start: int
    end: int
    kind: str
    reason: str
    content: bytes
    priority: int


_TRACEBACK = b"Traceback (most recent call last):"
_ERROR_LINE = re.compile(rb"\b(?:Error|Exception|Fatal)\b")
_FRAME = re.compile(rb"File [\"\']([^\"\']+)[\"\']")
_TAIL_BYTES = 8 * 1024
_MAX_BLOCK_BYTES = 32 * 1024


# 扫描单个日志文件并只保留有限候选块、尾部和哈希
def _scan(
    path: Path, source: Literal["stdout", "stderr"], workspace: str
) -> tuple[list[_Candidate], int, str]:
    candidates: list[_Candidate] = []
    digest = hashlib.sha256()
    total = 0
    tail: deque[tuple[int, bytes]] = deque()
    traceback_start: int | None = None
    traceback_lines: list[tuple[int, bytes]] = []
    latest_traceback: _Candidate | None = None
    handle: BinaryIO = path.open("rb") if path.exists() else _empty_stream()
    with handle:
        while True:
            line_start = total
            line = handle.readline()
            if not line:
                break
            total += len(line)
            digest.update(line)
            tail.append((line_start, line))
            while sum(len(item[1]) for item in tail) > _TAIL_BYTES:
                tail.popleft()
            if _TRACEBACK in line:
                traceback_start = line_start
                traceback_lines = [(line_start, line)]
                continue
            if traceback_start is not None:
                if (
                    line.strip()
                    and sum(len(item[1]) for item in traceback_lines) < _MAX_BLOCK_BYTES
                ):
                    traceback_lines.append((line_start, line))
                if not line.strip() or total - line_start > _MAX_BLOCK_BYTES:
                    block = b"".join(item[1] for item in traceback_lines)
                    if block:
                        latest_traceback = _Candidate(
                            source,
                            traceback_start,
                            traceback_start + len(block),
                            "traceback",
                            "latest complete Python traceback",
                            block,
                            1,
                        )
                    traceback_start = None
                    traceback_lines = []
            if _ERROR_LINE.search(line):
                candidates.append(
                    _Candidate(
                        source, line_start, total, "error_line", "latest error/fatal line", line, 3
                    )
                )
            frame = _FRAME.search(line)
            if frame is not None and workspace.encode() in frame.group(1):
                candidates.append(
                    _Candidate(
                        source,
                        line_start,
                        total,
                        "workspace_frame",
                        "workspace traceback frame",
                        line,
                        2,
                    )
                )
    if traceback_start is not None and traceback_lines:
        block = b"".join(item[1] for item in traceback_lines)
        latest_traceback = _Candidate(
            source,
            traceback_start,
            traceback_start + len(block),
            "traceback",
            "latest complete Python traceback",
            block,
            1,
        )
    if latest_traceback is not None:
        candidates.append(latest_traceback)
    if tail:
        start = tail[0][0]
        candidates.append(
            _Candidate(
                source,
                start,
                total,
                "tail",
                "log tail fallback",
                b"".join(item[1] for item in tail),
                5,
            )
        )
    return candidates, total, digest.hexdigest()


# 提供一个空的二进制流以统一不存在日志的扫描分支
def _empty_stream() -> BinaryIO:
    from io import BytesIO

    return BytesIO()


# 对两个不可变日志做固定优先级、可复现的有限证据选择
def select_evidence(
    capsule: FailureCapsule,
    stdout_path: Path,
    stderr_path: Path,
    max_bytes: int = 32 * 1024,
) -> EvidenceSelection:
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    stdout_candidates, stdout_size, stdout_sha = _scan(stdout_path, "stdout", capsule.cwd)
    stderr_candidates, stderr_size, stderr_sha = _scan(stderr_path, "stderr", capsule.cwd)
    candidates = sorted(
        [*stderr_candidates, *stdout_candidates],
        key=lambda item: (item.priority, -item.start, item.source),
    )
    selected: list[_Candidate] = []
    rendered: list[str] = []
    seen: set[bytes] = set()
    selected_size = 0
    duplicates = 0
    for candidate in candidates:
        if candidate.content in seen:
            duplicates += 1
            continue
        if selected_size >= max_bytes:
            break
        content = candidate.content[: max_bytes - selected_size]
        while content:
            block = (
                f"[{candidate.source} bytes={candidate.start}-"
                f"{candidate.start + len(content)} kind={candidate.kind}]\n"
                f"{content.decode('utf-8', errors='replace').rstrip(chr(10))}"
            )
            if len("\n".join([*rendered, block]).encode("utf-8")) <= max_bytes:
                break
            overflow = len("\n".join([*rendered, block]).encode("utf-8")) - max_bytes
            content = content[: max(0, len(content) - max(1, overflow))]
        if not content:
            break
        seen.add(candidate.content)
        selected_candidate = _Candidate(
            candidate.source,
            candidate.start,
            candidate.start + len(content),
            candidate.kind,
            candidate.reason,
            content,
            candidate.priority,
        )
        selected.append(selected_candidate)
        rendered.append(
            f"[{candidate.source} bytes={candidate.start}-"
            f"{candidate.start + len(content)} kind={candidate.kind}]\n"
            f"{content.decode('utf-8', errors='replace').rstrip(chr(10))}"
        )
        selected_size += len(content)
    references = [
        EvidenceReference(
            source=item.source,
            start=item.start,
            end=item.end,
            kind=item.kind,
            selection_reason=item.reason,
            sha256=stderr_sha if item.source == "stderr" else stdout_sha,
        )
        for item in selected
    ]
    body = "\n".join(rendered)
    return EvidenceSelection(
        content=body,
        references=references,
        scanned_bytes=stdout_size + stderr_size,
        selected_bytes=selected_size,
        duplicates_removed=duplicates,
        stdout_sha256=stdout_sha,
        stderr_sha256=stderr_sha,
    )
