from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cyan.core.tools.builtin.read_file import ReadFileTool


# 功能：验证读取存在的文件时返回完整内容且 is_error 为 False
# 设计：写临时文件后读取，断言 content 和 is_error，覆盖正常路径（happy path）
async def test_read_existing_file(tmp_path: Path) -> None:
    f = tmp_path / "hello.txt"
    f.write_text("hello world", encoding="utf-8")
    result = await ReadFileTool(tmp_path).invoke({"path": "hello.txt"})
    assert not result.is_error
    digest = hashlib.sha256(b"hello world").hexdigest()
    assert result.content == (
        f"[source hello.txt@sha256:{digest}#L1-L1]\nhello world"
    )


# 功能：验证文件不存在时抛出 FileNotFoundError 而非返回错误 ToolResult
# 设计：传入不存在的路径，确认 ReadFileTool 不吞掉异常，让调用方（invoke_tool）负责错误分类和事件发布
async def test_file_not_found_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ReadFileTool(tmp_path).invoke({"path": "missing.txt"})


# 功能：验证包含 `..` 的路径被拒绝并抛出 PermissionError
# 设计：传入 `"../secret.txt"` 这种最典型的目录遍历形式，确认安全边界第一道防线有效
async def test_path_traversal_dotdot_raises() -> None:
    with pytest.raises(PermissionError):
        await ReadFileTool().invoke({"path": "../secret.txt"})


# 功能：验证多级路径中嵌入的 `..` 经过路径规范化后也被正确检测
# 设计：使用 `"subdir/../../etc/passwd"` 测试路径 resolve 后的深度遍历，确认单层 `..` 过滤不足以覆盖此情况
async def test_path_traversal_nested_raises() -> None:
    with pytest.raises(PermissionError):
        await ReadFileTool().invoke({"path": "subdir/../../etc/passwd"})


# 功能：验证超过 512KB 的文件被截断并在末尾追加 [truncated] 标记
# 设计：写 600KB 文件，断言内容以 x×512KB 开头、以 [truncated] 结尾，确认截断不破坏前缀内容
async def test_truncation_over_512kb(tmp_path: Path) -> None:
    f = tmp_path / "big.txt"
    f.write_bytes(b"x" * (600 * 1024))
    result = await ReadFileTool(tmp_path).invoke({"path": "big.txt"})
    assert not result.is_error
    assert result.content.endswith("[truncated]")
    body = result.content.split("\n", maxsplit=1)[1]
    assert body.startswith("x" * (512 * 1024))


# 功能：验证恰好等于 512KB 的文件不被截断（边界值：超过而非大于等于）
# 设计：boundary check，确认截断阈值为"严格超过 512KB"，防止 off-by-one 错误
async def test_exact_512kb_is_not_truncated(tmp_path: Path) -> None:
    f = tmp_path / "exact.txt"
    f.write_bytes(b"y" * (512 * 1024))
    result = await ReadFileTool(tmp_path).invoke({"path": "exact.txt"})
    assert not result.is_error
    assert not result.content.endswith("[truncated]")
    body = result.content.split("\n", maxsplit=1)[1]
    assert len(body) == 512 * 1024


# 功能：验证空文件返回稳定来源引用而非错误
# 设计：零字节文件没有有效行号，断言 empty 标记让 Agent 仍能引用该文件
async def test_empty_file_returns_source_reference(tmp_path: Path) -> None:
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    result = await ReadFileTool(tmp_path).invoke({"path": "empty.txt"})
    assert not result.is_error
    digest = hashlib.sha256(b"").hexdigest()
    assert result.content == f"[source empty.txt@sha256:{digest}#empty]"


# 功能：验证 read_file 只返回请求的 1-based 行范围
# 设计：从第二行读取两行并同时断言来源引用和正文，覆盖上下界语义
async def test_read_file_returns_bounded_line_range(tmp_path: Path) -> None:
    content = "one\ntwo\nthree\nfour\n"
    (tmp_path / "lines.txt").write_text(content, encoding="utf-8")

    result = await ReadFileTool(tmp_path).invoke(
        {"path": "lines.txt", "start_line": 2, "line_count": 2}
    )

    digest = hashlib.sha256(content.encode()).hexdigest()
    assert result.content == (
        f"[source lines.txt@sha256:{digest}#L2-L3]\ntwo\nthree\n"
    )


# 功能：验证 read_file 拒绝绝对路径，即使目标位于绑定工作区内
# 设计：绝对路径不能绕过 workspace-relative 协议，直接使用根内文件覆盖该边界
async def test_read_file_rejects_absolute_path(tmp_path: Path) -> None:
    target = tmp_path / "inside.txt"
    target.write_text("inside", encoding="utf-8")

    with pytest.raises(PermissionError):
        await ReadFileTool(tmp_path).invoke({"path": str(target)})


# 功能：验证 read_file 拒绝解析后指向工作区外的符号链接
# 设计：根内链接指向相邻文件，覆盖路径文本无 `..` 但 resolve 后越界的情况
async def test_read_file_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("secret", encoding="utf-8")
    (workspace / "link.txt").symlink_to(outside)

    with pytest.raises(PermissionError):
        await ReadFileTool(workspace).invoke({"path": "link.txt"})


# 功能：验证默认 workspace root 在工具构造时捕获而不随进程 cwd 漂移
# 设计：在第一个目录构造工具后切换到第二个目录，仍应读取第一个目录的同名文件
async def test_read_file_binds_default_root_at_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "same.txt").write_text("first", encoding="utf-8")
    (second / "same.txt").write_text("second", encoding="utf-8")
    monkeypatch.chdir(first)
    tool = ReadFileTool()
    monkeypatch.chdir(second)

    result = await tool.invoke({"path": "same.txt"})

    assert result.content.endswith("\nfirst")
