from __future__ import annotations

import json
import math
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from cyan.training.incidents.models import FailureCapsule, LogSnapshot
from cyan.training.incidents.selector import select_evidence

from cyan_bench.cases import LoadedCase
from cyan_bench.models import (
    Baseline,
    ProcessCapture,
    ResolvedAnchor,
    SelectionArtifact,
    SelectionReference,
)

_MAX_SELECTION_BYTES = 32 * 1024
_CHUNK_BYTES = 2 * 1024
_CHUNK_OVERLAP = 256
_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]*|[0-9]+")


@dataclass(frozen=True)
class _Chunk:
    source: Literal["stdout", "stderr"]
    start: int
    end: int
    content: bytes
    tokens: Counter[str]


# 将日志片段渲染为保留来源和 byte range 的文本块
def _render(source: str, start: int, end: int, content: bytes) -> bytes:
    header = f"[{source} bytes={start}-{end}]\n".encode()
    return header + content


# 在总上下文预算内追加一段原始日志并返回实际引用范围
def _append_block(
    output: bytearray,
    references: list[SelectionReference],
    *,
    source: Literal["stdout", "stderr"],
    start: int,
    content: bytes,
    kind: str,
    reason: str,
    max_bytes: int,
) -> None:
    separator = b"\n" if output else b""
    available = max_bytes - len(output) - len(separator)
    if available <= 0:
        return
    header = f"[{source} bytes={start}-{start + len(content)}]\n".encode()
    if len(header) >= available:
        return
    selected = content[: available - len(header)]
    if not selected:
        return
    end = start + len(selected)
    block = _render(source, start, end, selected)
    output.extend(separator)
    output.extend(block)
    references.append(
        SelectionReference(
            source=source,
            start=start,
            end=end,
            kind=kind,
            selection_reason=reason,
        )
    )


# 计算 selection references 覆盖的唯一原始日志字节数
def _unique_reference_bytes(references: list[SelectionReference]) -> int:
    total = 0
    for source in ("stdout", "stderr"):
        ranges = sorted(
            (item.start, item.end) for item in references if item.source == source
        )
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        total += sum(end - start for start, end in merged)
    return total


# 将日志文本切成固定字节窗口并保留稳定 offset
def _chunks(path: Path, source: Literal["stdout", "stderr"]) -> list[_Chunk]:
    data = path.read_bytes()
    chunks: list[_Chunk] = []
    step = _CHUNK_BYTES - _CHUNK_OVERLAP
    for start in range(0, len(data), step):
        content = data[start : start + _CHUNK_BYTES]
        if not content:
            break
        text = content.decode("utf-8", errors="replace").lower()
        tokens = Counter(_TOKEN.findall(text))
        chunks.append(_Chunk(source, start, start + len(content), content, tokens))
        if start + len(content) >= len(data):
            break
    return chunks


# 使用固定 k1 与 b 对全部日志 chunk 计算最小 BM25 排名
def _bm25_rank(chunks: list[_Chunk], query: str) -> list[_Chunk]:
    if not chunks:
        return []
    query_terms = set(_TOKEN.findall(query.lower()))
    average_length = sum(sum(chunk.tokens.values()) for chunk in chunks) / len(chunks)
    document_frequency: Counter[str] = Counter()
    for chunk in chunks:
        document_frequency.update(set(chunk.tokens))
    scored: list[tuple[float, _Chunk]] = []
    for chunk in chunks:
        length = sum(chunk.tokens.values())
        score = 0.0
        for term in query_terms:
            frequency = chunk.tokens.get(term, 0)
            if not frequency:
                continue
            documents = document_frequency[term]
            inverse = math.log(1 + (len(chunks) - documents + 0.5) / (documents + 0.5))
            denominator = frequency + 1.5 * (
                1 - 0.75 + 0.75 * length / max(1.0, average_length)
            )
            score += inverse * frequency * 2.5 / denominator
        scored.append((score, chunk))
    scored.sort(key=lambda item: (-item[0], item[1].source, item[1].start))
    return [item[1] for item in scored]


# 构造不含日志尾部和故障答案的公共 Failure Capsule 元数据
def capsule_metadata(case: LoadedCase, capture: ProcessCapture) -> dict[str, object]:
    return {
        "argv": capture.argv,
        "cwd": capture.cwd,
        "returncode": capture.returncode,
        "hardware": case.manifest.hardware,
        "framework": case.manifest.framework,
        "stdout_bytes": capture.stdout_bytes,
        "stderr_bytes": capture.stderr_bytes,
    }


# 将一次真实进程结果转换为 Cyan Selector 所需的无 tail Capsule
def _selector_capsule(case: LoadedCase, capture: ProcessCapture) -> FailureCapsule:
    return FailureCapsule(
        job_id=case.manifest.id,
        attempt_id=f"repeat-{capture.repeat}",
        argv=capture.argv,
        cwd=capture.cwd,
        occurred_at=capture.created_at,
        failure_kind="process_exit",
        returncode=capture.returncode,
        environment={"hardware": case.manifest.hardware},
        stdout=LogSnapshot(
            size=capture.stdout_bytes,
            sha256=capture.stdout_sha256,
            included_start=capture.stdout_bytes,
            included_end=capture.stdout_bytes,
            tail="",
        ),
        stderr=LogSnapshot(
            size=capture.stderr_bytes,
            sha256=capture.stderr_sha256,
            included_start=capture.stderr_bytes,
            included_end=capture.stderr_bytes,
            tail="",
        ),
    )


# 生成 FullNative、Tail、BM25、CyanSelector 或 Oracle 证据选择
def select_baseline(
    case: LoadedCase,
    capture: ProcessCapture,
    run_dir: Path,
    baseline: Baseline,
    output_dir: Path,
) -> SelectionArtifact:
    stdout_path = run_dir / capture.stdout_path
    stderr_path = run_dir / capture.stderr_path
    started = time.monotonic()
    output = bytearray()
    references: list[SelectionReference] = []
    scanned = capture.stdout_bytes + capture.stderr_bytes
    max_bytes = _MAX_SELECTION_BYTES
    if baseline == "full_native":
        max_bytes = scanned + 256
        full_streams: tuple[tuple[Literal["stdout", "stderr"], Path], ...] = (
            ("stdout", stdout_path),
            ("stderr", stderr_path),
        )
        for source, path in full_streams:
            _append_block(
                output,
                references,
                source=source,
                start=0,
                content=path.read_bytes(),
                kind="full_log",
                reason="complete stream in frozen stdout-then-stderr order",
                max_bytes=max_bytes,
            )
    elif baseline == "tail_32":
        tail_streams: tuple[tuple[Literal["stdout", "stderr"], Path], ...] = (
            ("stderr", stderr_path),
            ("stdout", stdout_path),
        )
        for source, path in tail_streams:
            data = path.read_bytes()
            header_reserve = 64
            remaining = max(0, max_bytes - len(output) - header_reserve)
            start = max(0, len(data) - remaining)
            _append_block(
                output,
                references,
                source=source,
                start=start,
                content=data[start:],
                kind="tail",
                reason="stderr-first fixed tail baseline",
                max_bytes=max_bytes,
            )
    elif baseline == "bm25_32":
        metadata = json.dumps(capsule_metadata(case, capture), ensure_ascii=False)
        stderr_tail = stderr_path.read_bytes()[-2048:].decode("utf-8", errors="replace")
        ranked = _bm25_rank(
            [*_chunks(stdout_path, "stdout"), *_chunks(stderr_path, "stderr")],
            f"{metadata}\n{stderr_tail}",
        )
        for chunk in ranked:
            _append_block(
                output,
                references,
                source=chunk.source,
                start=chunk.start,
                content=chunk.content,
                kind="bm25_chunk",
                reason="standard BM25 score with frozen chunking",
                max_bytes=max_bytes,
            )
    elif baseline == "cyan_selector_32":
        selection = select_evidence(
            _selector_capsule(case, capture),
            stdout_path,
            stderr_path,
            max_bytes=max_bytes,
        )
        output.extend(selection.content.encode("utf-8"))
        references = [
            SelectionReference(
                source=item.source,
                start=item.start,
                end=item.end,
                kind=item.kind,
                selection_reason=item.selection_reason,
            )
            for item in selection.references
        ]
    elif baseline == "oracle_32":
        ranges = [
            ResolvedAnchor.model_validate(item)
            for item in json.loads((run_dir / "gold-ranges.json").read_text(encoding="utf-8"))
        ]
        for item in ranges:
            path = stdout_path if item.source == "stdout" else stderr_path
            with path.open("rb") as handle:
                handle.seek(item.start)
                content = handle.read(item.end - item.start)
            _append_block(
                output,
                references,
                source=item.source,
                start=item.start,
                content=content,
                kind="oracle",
                reason=f"gold required group {item.group}",
                max_bytes=max_bytes,
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    content_path = output_dir / "selection.txt"
    content_path.write_bytes(bytes(output))
    artifact = SelectionArtifact(
        case_id=case.manifest.id,
        baseline=baseline,
        repeat=capture.repeat,
        content_path="selection.txt",
        selected_bytes=len(output),
        unique_selected_bytes=_unique_reference_bytes(references),
        scanned_bytes=scanned,
        latency_seconds=round(time.monotonic() - started, 6),
        references=references,
    )
    (output_dir / "selection.json").write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact
