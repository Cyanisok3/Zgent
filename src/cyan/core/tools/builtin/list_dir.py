from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from cyan.core.tools.base import BaseTool, ToolResult
from cyan.core.tools.workspace import (
    normalize_workspace_root,
    resolve_workspace_path,
    workspace_relative_path,
)

_MAX_DEPTH = 4
_MAX_ENTRIES = 200


class ListDirParams(BaseModel):
    model_config = ConfigDict(extra="ignore")
    path: str = "."
    max_depth: int = Field(default=2, ge=1, le=_MAX_DEPTH)


class ListDirTool(BaseTool):
    params_model = ListDirParams
    name = "list_dir"
    description = (
        "List the contents of a directory as a tree. "
        "Path must be relative to the bound workspace root. "
        "Hidden entries (starting with .) are included. "
        f"Maximum depth is {_MAX_DEPTH}, maximum total entries is {_MAX_ENTRIES}."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Relative path to the workspace root (default '.').",
            },
            "max_depth": {
                "type": "integer",
                "description": f"How many levels deep to recurse (default 2, max {_MAX_DEPTH}).",
            },
        },
        "required": [],
    }

    # 将目录遍历绑定到不可变工作区根目录
    def __init__(self, root: str | Path | None = None) -> None:
        self.root = normalize_workspace_root(root)

    # 以树状格式列出目录内容，深度和条数有上限
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        p = ListDirParams.model_validate(params)
        path_str = p.path
        max_depth = p.max_depth
        root = resolve_workspace_path(self.root, path_str)
        if not root.exists():
            raise FileNotFoundError(f"no such directory: {path_str}")
        if not root.is_dir():
            raise NotADirectoryError(f"not a directory: {path_str}")

        display_root = workspace_relative_path(self.root, root)
        lines: list[str] = [display_root + "/"]
        count = 0

        # 递归生成有界目录树且不跟随符号链接目录
        def _walk(directory: Path, depth: int, prefix: str) -> None:
            nonlocal count
            if depth > max_depth or count >= _MAX_ENTRIES:
                return
            entries = sorted(directory.iterdir(), key=lambda e: (e.is_file(), e.name))
            for i, entry in enumerate(entries):
                if count >= _MAX_ENTRIES:
                    lines.append(f"{prefix}... (truncated)")
                    return
                connector = "└── " if i == len(entries) - 1 else "├── "
                is_directory = entry.is_dir()
                suffix = "/" if is_directory and not entry.is_symlink() else ""
                lines.append(f"{prefix}{connector}{entry.name}{suffix}")
                count += 1
                if is_directory and not entry.is_symlink() and depth < max_depth:
                    extension = "    " if i == len(entries) - 1 else "│   "
                    _walk(entry, depth + 1, prefix + extension)

        _walk(root, 1, "")
        return ToolResult(content="\n".join(lines))
