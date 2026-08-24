from __future__ import annotations

import hashlib
import re
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
_ERROR_LINE = re.compile(
    rb"(?:^|[^A-Za-z0-9_])(?:[A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)|Fatal)(?::|\b)"
)
_FRAME = re.compile(rb"File [\"\']([^\"\']+)[\"\']")
_TAIL_BYTES = 8 * 1024
_MAX_BLOCK_BYTES = 32 * 1024


# 从文件末尾按字节读取有界回退证据
def _tail_candidate(
    path: Path,
    source: Literal["stdout", "stderr"],
    total: int,
) -> _Candidate | None:
    if total <= 0 or not path.exists():
        return None
    start = max(0, total - _TAIL_BYTES)
    with path.open("rb") as handle:
        handle.seek(start)
        content = handle.read(_TAIL_BYTES)
    return _Candidate(
        source,
        start,
        start + len(content),
        "tail",
        "stderr tail" if source == "stderr" else "stdout tail fallback",
        content,
        5 if source == "stderr" else 6,
    )


# 扫描单个日志并只保留最新的结构化候选、字节数和哈希
def _scan(
    path: Path, source: Literal["stdout", "stderr"], workspace: str
) -> tuple[list[_Candidate], int, str]:
    digest = hashlib.sha256()
    total = 0
    traceback_start: int | None = None
    traceback_lines: list[tuple[int, bytes]] = []
    traceback_bytes = 0
    latest_traceback: _Candidate | None = None
    traceback_exception: _Candidate | None = None
    current_exception: _Candidate | None = None
    latest_error: _Candidate | None = None
    latest_frame: _Candidate | None = None

    # 将当前 traceback 收口为最新完整块和终止异常行
    def finish_traceback() -> None:
        nonlocal latest_traceback, traceback_exception
        nonlocal traceback_start, traceback_lines, traceback_bytes, current_exception
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
            traceback_exception = current_exception
        traceback_start = None
        traceback_lines = []
        traceback_bytes = 0
        current_exception = None

    handle: BinaryIO = path.open("rb") if path.exists() else _empty_stream()
    with handle:
        while True:
            line_start = total
            line = handle.readline()
            if not line:
                break
            total += len(line)
            digest.update(line)
            if _TRACEBACK in line:
                finish_traceback()
                traceback_start = line_start
                traceback_lines = [(line_start, line)]
                traceback_bytes = len(line)
                continue
            if traceback_start is not None:
                if line.strip() and traceback_bytes < _MAX_BLOCK_BYTES:
                    remaining = _MAX_BLOCK_BYTES - traceback_bytes
                    piece = line[:remaining]
                    traceback_lines.append((line_start, piece))
                    traceback_bytes += len(piece)
                if _ERROR_LINE.search(line):
                    current_exception = _Candidate(
                        source,
                        line_start,
                        total,
                        "traceback_exception",
                        "traceback final exception line",
                        line,
                        2,
                    )
                if not line.strip():
                    finish_traceback()
            if source == "stderr" and _ERROR_LINE.search(line):
                latest_error = _Candidate(
                    source,
                    line_start,
                    total,
                    "error_line",
                    "latest stderr error/exception/fatal line",
                    line,
                    4,
                )
            frame = _FRAME.search(line)
            if frame is not None and workspace.encode() in frame.group(1):
                latest_frame = _Candidate(
                    source,
                    line_start,
                    total,
                    "workspace_frame",
                    "workspace traceback frame",
                    line,
                    3,
                )
    finish_traceback()
    candidates = [
        item
        for item in (
            latest_traceback,
            traceback_exception,
            latest_frame,
            latest_error,
            _tail_candidate(path, source, total),
        )
        if item is not None
    ]
    return candidates, total, digest.hexdigest()


# 提供一个空的二进制流以统一不存在日志的扫描分支
def _empty_stream() -> BinaryIO:
    from io import BytesIO

    return BytesIO()


# 将候选渲染为带稳定字节范围的上下文块
def _render(candidate: _Candidate, content: bytes) -> str:
    return (
        f"[{candidate.source} bytes={candidate.start}-"
        f"{candidate.start + len(content)} kind={candidate.kind}]\n"
        f"{content.decode('utf-8', errors='replace').rstrip(chr(10))}"
    )


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
        reserved = 0
        if candidate.kind == "traceback":
            exception = next(
                (
                    item
                    for item in candidates
                    if item.kind == "traceback_exception" and item.source == candidate.source
                ),
                None,
            )
            if exception is not None and exception.content not in seen:
                reserved = len(_render(exception, exception.content).encode("utf-8")) + 1
        available = max_bytes - reserved
        if len("\n".join(rendered).encode("utf-8")) >= available:
            if candidate.kind == "traceback":
                continue
            break
        content = candidate.content
        content_start = candidate.start
        while content:
            render_candidate = _Candidate(
                candidate.source,
                content_start,
                content_start + len(content),
                candidate.kind,
                candidate.reason,
                content,
                candidate.priority,
            )
            block = _render(render_candidate, content)
            if len("\n".join([*rendered, block]).encode("utf-8")) <= available:
                break
            overflow = len("\n".join([*rendered, block]).encode("utf-8")) - available
            remove = min(len(content), max(1, overflow))
            if candidate.kind == "tail":
                content = content[remove:]
                content_start += remove
            else:
                content = content[: len(content) - remove]
        if not content:
            if candidate.kind == "traceback":
                continue
            break
        seen.add(candidate.content)
        selected_candidate = _Candidate(
            candidate.source,
            content_start,
            content_start + len(content),
            candidate.kind,
            candidate.reason,
            content,
            candidate.priority,
        )
        selected.append(selected_candidate)
        rendered.append(_render(selected_candidate, content))
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
