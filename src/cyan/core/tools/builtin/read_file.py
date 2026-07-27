from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cyan.core.tools.base import BaseTool, ToolResult
from cyan.core.tools.workspace import (
    normalize_workspace_root,
    resolve_workspace_path,
    workspace_relative_path,
)

_MAX_BYTES = 512 * 1024  # 512 KB
_MAX_LINES = 1000


class ReadFileParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str
    start_line: int = Field(default=1, ge=1)
    line_count: int | None = Field(default=None, ge=1, le=_MAX_LINES)


class ReadFileTool(BaseTool):
    params_model = ReadFileParams
    name = "read_file"
    description = (
        "Read the text content of a file. "
        "Path must be relative to the bound workspace root. "
        "Supports a bounded 1-based line range and returns a stable source reference. "
        "Output larger than 512 KB is truncated."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the file (relative to workspace root).",
            },
            "start_line": {
                "type": "integer",
                "description": "First line to return, using 1-based numbering (default 1).",
            },
            "line_count": {
                "type": "integer",
                "description": f"Maximum lines to return (max {_MAX_LINES}).",
            },
        },
        "required": ["path"],
    }

    # 将读取范围绑定到不可变工作区根目录
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        evidence_refs: set[str] | None = None,
    ) -> None:
        self.root = normalize_workspace_root(root)
        self._evidence_refs = evidence_refs

    # 在线程中扫描文件并返回正文和稳定引用，避免大文件哈希阻塞 daemon event loop
    def _read_sync(self, parsed: ReadFileParams) -> tuple[str, str]:
        path = resolve_workspace_path(self.root, parsed.path)
        digest = hashlib.sha256()
        selected = bytearray()
        first_line: int | None = None
        last_line: int | None = None
        byte_truncated = False

        with path.open("rb") as file:
            for line_number, line in enumerate(file, start=1):
                digest.update(line)
                if line_number < parsed.start_line:
                    continue
                if (
                    parsed.line_count is not None
                    and line_number >= parsed.start_line + parsed.line_count
                ):
                    continue
                if byte_truncated:
                    continue

                remaining = _MAX_BYTES - len(selected)
                if remaining == 0:
                    byte_truncated = True
                    continue
                if len(line) > remaining:
                    selected.extend(line[:remaining])
                    first_line = first_line or line_number
                    last_line = line_number
                    byte_truncated = True
                    continue
                selected.extend(line)
                first_line = first_line or line_number
                last_line = line_number

        relative = workspace_relative_path(self.root, path)
        line_reference = (
            "empty"
            if first_line is None
            else f"L{first_line}-L{last_line}"
        )
        reference = f"{relative}@sha256:{digest.hexdigest()}#{line_reference}"
        header = f"[source {reference}]"
        text = selected.decode("utf-8", errors="replace")
        content = header if not text else f"{header}\n{text}"
        if byte_truncated:
            content += "\n[truncated]"
        return content, reference

    # 读取有界文件范围并返回可引用的路径、哈希和行号
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = ReadFileParams.model_validate(params)
        content, reference = await asyncio.to_thread(self._read_sync, parsed)
        if self._evidence_refs is not None:
            self._evidence_refs.add(reference)
        return ToolResult(content=content)
