from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field

from cyan.core.tools.base import BaseTool, ToolResult

_MAX_RESULT_BYTES = 32 * 1024
_SEARCH_CHUNK_BYTES = 64 * 1024

LogStream = Literal["stdout", "stderr"]
LogMode = Literal["tail", "range", "search"]
LogPathResolver = Callable[[str, str, LogStream], Path]


class JobLogReader(Protocol):
    # 返回指定日志流的当前字节数
    async def size(self, job_id: str, attempt_id: str, stream: LogStream) -> int: ...

    # 从稳定字节偏移读取不超过 limit 的原始日志
    async def read(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        limit: int,
    ) -> bytes: ...


class FileJobLogReader:
    # 使用路径解析回调初始化本地文件日志 reader
    def __init__(self, path_resolver: LogPathResolver) -> None:
        self._path_resolver = path_resolver

    # 返回路径回调所定位日志文件的字节数
    async def size(self, job_id: str, attempt_id: str, stream: LogStream) -> int:
        return self._path_resolver(job_id, attempt_id, stream).stat().st_size

    # 按字节偏移读取日志，保证不会超过请求上限
    async def read(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        limit: int,
    ) -> bytes:
        path = self._path_resolver(job_id, attempt_id, stream)
        with path.open("rb") as handle:
            handle.seek(offset)
            return handle.read(limit)


class ReadJobLogParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1, max_length=128)
    attempt_id: str = Field(min_length=1, max_length=128)
    stream: LogStream
    mode: LogMode = "tail"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=_MAX_RESULT_BYTES, ge=1, le=_MAX_RESULT_BYTES)
    query: str | None = Field(default=None, min_length=1, max_length=1024)


class LogSlice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    start: int
    end: int
    content: str


class LogReadResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str
    attempt_id: str
    stream: LogStream
    mode: LogMode
    size: int
    match_offset: int | None = None
    slice: LogSlice | None = None
    reference: str | None = None


class ReadJobLogTool(BaseTool):
    params_model = ReadJobLogParams
    name = "read_job_log"
    description = (
        "Read immutable training stdout or stderr by stable byte range. "
        "Supports tail, range, and first-match search; each result is limited to 32 KiB."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "attempt_id": {"type": "string"},
            "stream": {"type": "string", "enum": ["stdout", "stderr"]},
            "mode": {"type": "string", "enum": ["tail", "range", "search"]},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": _MAX_RESULT_BYTES,
            },
            "query": {"type": "string"},
        },
        "required": ["job_id", "attempt_id", "stream"],
    }

    # 使用注入的日志 reader 初始化只读工具
    def __init__(
        self,
        reader: JobLogReader,
        *,
        evidence_refs: set[str] | None = None,
    ) -> None:
        self._reader = reader
        self._evidence_refs = evidence_refs

    # 从 reader 获取并截断一个稳定字节区间
    async def _read_slice(
        self,
        params: ReadJobLogParams,
        start: int,
        limit: int,
    ) -> LogSlice:
        raw = await self._reader.read(
            params.job_id,
            params.attempt_id,
            params.stream,
            start,
            min(limit, _MAX_RESULT_BYTES),
        )
        raw = raw[: min(limit, _MAX_RESULT_BYTES)]
        return LogSlice(
            start=start,
            end=start + len(raw),
            content=raw.decode("utf-8", errors="replace"),
        )

    # 从指定偏移开始查找首个 UTF-8 查询串并返回其字节位置
    async def _find_first(
        self,
        params: ReadJobLogParams,
        size: int,
        query: bytes,
    ) -> int | None:
        position = min(params.offset, size)
        overlap = b""
        while position < size:
            raw = await self._reader.read(
                params.job_id,
                params.attempt_id,
                params.stream,
                position,
                min(_SEARCH_CHUNK_BYTES, size - position),
            )
            if not raw:
                break
            window = overlap + raw
            found = window.find(query)
            if found >= 0:
                return position - len(overlap) + found
            overlap_size = max(0, len(query) - 1)
            overlap = window[-overlap_size:] if overlap_size else b""
            position += len(raw)
        return None

    # 按 tail、range 或 search 模式返回有界日志及稳定 byte range
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = ReadJobLogParams.model_validate(params)
        size = await self._reader.size(
            parsed.job_id,
            parsed.attempt_id,
            parsed.stream,
        )
        match_offset: int | None = None
        log_slice: LogSlice | None

        if parsed.mode == "tail":
            start = max(0, size - parsed.limit)
            log_slice = await self._read_slice(parsed, start, parsed.limit)
        elif parsed.mode == "range":
            start = min(parsed.offset, size)
            log_slice = await self._read_slice(parsed, start, parsed.limit)
        else:
            if parsed.query is None:
                return ToolResult(
                    content="query is required when mode=search",
                    is_error=True,
                    error_type="schema_error",
                )
            match_offset = await self._find_first(
                parsed,
                size,
                parsed.query.encode("utf-8"),
            )
            if match_offset is None:
                log_slice = None
            else:
                start = max(parsed.offset, match_offset - parsed.limit // 2)
                start = min(start, max(0, size - parsed.limit))
                log_slice = await self._read_slice(parsed, start, parsed.limit)

        result = LogReadResult(
            job_id=parsed.job_id,
            attempt_id=parsed.attempt_id,
            stream=parsed.stream,
            mode=parsed.mode,
            size=size,
            match_offset=match_offset,
            slice=log_slice,
            reference=(
                f"{parsed.stream}:{parsed.job_id}/{parsed.attempt_id}"
                f"@bytes:{log_slice.start}-{log_slice.end}"
                if log_slice is not None
                else None
            ),
        )
        if self._evidence_refs is not None and result.reference is not None:
            self._evidence_refs.add(result.reference)
        return ToolResult(content=result.model_dump_json())
