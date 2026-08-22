from __future__ import annotations

import asyncio
import hashlib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cyan.agent.tools.base import BaseTool, ToolResult
from cyan.agent.tools.workspace import (
    normalize_workspace_root,
    resolve_workspace_path,
    workspace_relative_path,
)

_MAX_FILE_BYTES = 1024 * 1024
_MAX_FILES = 1000
_MAX_RESULTS = 100
_MAX_LINE_CHARS = 500


class SearchTextParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    query: str = Field(min_length=1, max_length=200)
    path: str = "."
    max_results: int = Field(default=50, ge=1, le=_MAX_RESULTS)


class SearchTextTool(BaseTool):
    params_model = SearchTextParams
    name = "search_text"
    description = (
        "Search text files under the workspace for a literal string. "
        f"Scans at most {_MAX_FILES} files and returns at most {_MAX_RESULTS} "
        "matching lines with stable path, hash, and line references."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Literal, case-sensitive text to search for.",
            },
            "path": {
                "type": "string",
                "description": "Relative file or directory path (default '.').",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum matching lines (default 50, max {_MAX_RESULTS}).",
            },
        },
        "required": ["query"],
    }

    # 将文本检索绑定到不可变工作区根目录
    def __init__(
        self,
        root: str | Path | None = None,
        *,
        evidence_refs: set[str] | None = None,
    ) -> None:
        self.root = normalize_workspace_root(root)
        self._evidence_refs = evidence_refs

    # 按稳定路径顺序生成工作区内的有界文件集合
    def _files(self, start: Path) -> list[Path]:
        if start.is_file():
            return [start]
        if not start.exists():
            raise FileNotFoundError(f"no such path: {start}")
        if not start.is_dir():
            raise OSError(f"unsupported path: {start}")

        files: list[Path] = []
        pending = [start]
        while pending and len(files) < _MAX_FILES:
            directory = pending.pop()
            entries = sorted(directory.iterdir(), key=lambda entry: entry.name, reverse=True)
            for entry in entries:
                if entry.is_symlink():
                    continue
                if entry.is_dir():
                    pending.append(entry)
                elif entry.is_file():
                    files.append(entry)
                    if len(files) >= _MAX_FILES:
                        break
        return sorted(files, key=lambda path: workspace_relative_path(self.root, path))

    # 在线程中扫描有界文件集合，返回展示文本及本轮实际观察到的证据引用
    def _search_sync(self, parsed: SearchTextParams) -> tuple[str, list[str]]:
        start = resolve_workspace_path(self.root, parsed.path)
        matches: list[str] = []
        references: list[str] = []
        files_scanned = 0
        truncated = False

        for path in self._files(start):
            if path.stat().st_size > _MAX_FILE_BYTES:
                continue
            raw = path.read_bytes()
            if b"\0" in raw:
                continue
            files_scanned += 1
            text = raw.decode("utf-8", errors="replace")
            digest = hashlib.sha256(raw).hexdigest()
            relative = workspace_relative_path(self.root, path)
            for line_number, line in enumerate(text.splitlines(), start=1):
                if parsed.query not in line:
                    continue
                if len(matches) >= parsed.max_results:
                    truncated = True
                    break
                excerpt = line[:_MAX_LINE_CHARS]
                reference = f"{relative}@sha256:{digest}#L{line_number}"
                references.append(reference)
                matches.append(f"{reference}: {excerpt}")
            if truncated:
                break

        header = (
            f"[search query={parsed.query!r} files_scanned={files_scanned} "
            f"matches={len(matches)} truncated={str(truncated).lower()}]"
        )
        content = "\n".join([header, *matches])
        return content, references

    # 在文本文件中执行字面量检索并限制扫描量与返回结果数
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = SearchTextParams.model_validate(params)
        content, references = await asyncio.to_thread(self._search_sync, parsed)
        if self._evidence_refs is not None:
            self._evidence_refs.update(references)
        return ToolResult(content=content)
