from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from cyan.core.tools.builtin.search_text import SearchTextTool


# 功能：验证 search_text 返回稳定路径、完整文件哈希和匹配行号
# 设计：在两个文件中仅放置一个命中，精确比较引用以覆盖结果格式与 literal 搜索
@pytest.mark.asyncio
async def test_search_text_returns_stable_reference(tmp_path: Path) -> None:
    content = "first\nneedle here\nlast\n"
    (tmp_path / "match.txt").write_text(content, encoding="utf-8")
    (tmp_path / "other.txt").write_text("nothing\n", encoding="utf-8")

    result = await SearchTextTool(tmp_path).invoke({"query": "needle"})

    digest = hashlib.sha256(content.encode()).hexdigest()
    assert not result.is_error
    assert "matches=1" in result.content
    assert (
        f"match.txt@sha256:{digest}#L2: needle here"
        in result.content
    )


# 功能：验证 search_text 在 max_results 后停止并显式标记截断
# 设计：创建三个命中但只请求两个，断言输出条目数和 truncated 元数据
@pytest.mark.asyncio
async def test_search_text_limits_results(tmp_path: Path) -> None:
    (tmp_path / "many.txt").write_text("hit\nhit\nhit\n", encoding="utf-8")

    result = await SearchTextTool(tmp_path).invoke(
        {"query": "hit", "max_results": 2}
    )

    assert "matches=2" in result.content
    assert "truncated=true" in result.content
    assert result.content.count("#L") == 2


# 功能：验证命中数恰好等于 max_results 时不会误报截断
# 设计：只创建两个命中并设置上限为二，区分“达到上限”和“确有额外结果”
@pytest.mark.asyncio
async def test_search_text_exact_limit_is_not_truncated(tmp_path: Path) -> None:
    (tmp_path / "exact.txt").write_text("hit\nhit\n", encoding="utf-8")

    result = await SearchTextTool(tmp_path).invoke(
        {"query": "hit", "max_results": 2}
    )

    assert "matches=2" in result.content
    assert "truncated=false" in result.content


# 功能：验证 search_text 跳过二进制文件而不把其内容返回给 Agent
# 设计：使用含 NUL 的文件模拟二进制，再断言零匹配和无文件引用
@pytest.mark.asyncio
async def test_search_text_skips_binary_files(tmp_path: Path) -> None:
    (tmp_path / "binary.bin").write_bytes(b"\0needle")

    result = await SearchTextTool(tmp_path).invoke({"query": "needle"})

    assert "matches=0" in result.content
    assert "#L" not in result.content


# 功能：验证 search_text 拒绝绝对路径和解析后越界的入口 symlink
# 设计：同一测试覆盖两种可绕过纯文本 `..` 检查的路径形式
@pytest.mark.asyncio
async def test_search_text_rejects_outside_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    tool = SearchTextTool(workspace)

    with pytest.raises(PermissionError):
        await tool.invoke({"query": "x", "path": str(workspace)})
    with pytest.raises(PermissionError):
        await tool.invoke({"query": "x", "path": "link"})
