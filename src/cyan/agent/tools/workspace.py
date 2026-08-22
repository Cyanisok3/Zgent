from __future__ import annotations

from pathlib import Path


# 将工具工作区固定为已解析的现有目录
def normalize_workspace_root(root: str | Path | None = None) -> Path:
    workspace_root = Path.cwd() if root is None else Path(root)
    resolved = workspace_root.resolve(strict=True)
    if not resolved.is_dir():
        raise NotADirectoryError(f"workspace root is not a directory: {workspace_root}")
    return resolved


# 将相对路径解析到工作区内并拒绝绝对路径、目录遍历和符号链接越界
def resolve_workspace_path(root: Path, path: str) -> Path:
    relative = Path(path)
    if relative.is_absolute():
        raise PermissionError(f"absolute path not allowed: {path}")
    if ".." in relative.parts:
        raise PermissionError(f"path traversal not allowed: {path}")

    resolved = (root / relative).resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PermissionError(f"path escapes workspace: {path}") from exc
    return resolved


# 返回工作区内路径的稳定 POSIX 相对表示
def workspace_relative_path(root: Path, path: Path) -> str:
    relative = path.relative_to(root).as_posix()
    return relative or "."
