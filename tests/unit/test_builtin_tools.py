from __future__ import annotations

from pathlib import Path

import pytest

from cyan.agent.tools.builtin.bash import BashTool
from cyan.agent.tools.builtin.list_dir import ListDirTool
from cyan.agent.tools.builtin.write_file import WriteFileTool

# ── bash ──────────────────────────────────────────────────────────────────────

# 功能：验证成功命令的 stdout 出现在 ToolResult.content 中，is_error 为 False
# 设计：用 echo 命令避免外部依赖，直接比较输出内容，无需 mock
@pytest.mark.asyncio
async def test_bash_success_stdout() -> None:
    result = await BashTool().invoke({"command": "echo hello"})
    assert not result.is_error
    assert "hello" in result.content


# 功能：验证非零退出码时 is_error=True 且 content 包含退出码标注
# 设计：`exit 2` 是最简单的非零退出；不依赖任何外部命令行为
@pytest.mark.asyncio
async def test_bash_nonzero_exit_is_error() -> None:
    result = await BashTool().invoke({"command": "exit 2"})
    assert result.is_error
    assert "[exit 2]" in result.content


# 功能：验证命令超时后 is_error=True，error_type 为 "timeout"
# 设计：timeout=1s 搭配 sleep 2 必然超时；验证 error_type 而非 content，避免超时消息格式耦合
@pytest.mark.asyncio
async def test_bash_timeout() -> None:
    result = await BashTool().invoke({"command": "sleep 5", "timeout": 1})
    assert result.is_error
    assert result.error_type == "timeout"


# 功能：验证 stderr 被合并到 stdout 输出中
# 设计：只写 stderr 的命令（>&2 echo），输出应该出现在合并后的 content 里
@pytest.mark.asyncio
async def test_bash_stderr_merged() -> None:
    result = await BashTool().invoke({"command": "echo err >&2"})
    assert not result.is_error
    assert "err" in result.content


# 功能：验证 bash 子进程始终从工具绑定的工作区根目录启动
# 设计：在临时目录执行 pwd 并比较解析后的绝对路径，直接覆盖 cwd 注入
@pytest.mark.asyncio
async def test_bash_uses_workspace_root(tmp_path: Path) -> None:
    result = await BashTool(tmp_path).invoke({"command": "pwd"})
    assert not result.is_error
    assert result.content.strip() == str(tmp_path.resolve())


# ── write_file ────────────────────────────────────────────────────────────────

# 功能：验证 write_file 写入文件后内容可以被读取，返回字节数
# 设计：写入临时目录，断言文件存在且内容一致；用 tmp_path fixture 自动清理
@pytest.mark.asyncio
async def test_write_file_creates_and_returns_size(tmp_path: Path) -> None:
    target = tmp_path / "out.txt"
    result = await WriteFileTool(tmp_path).invoke(
        {"path": "out.txt", "content": "hello world"}
    )
    assert not result.is_error
    assert "11" in result.content  # "hello world" = 11 bytes
    assert target.read_text() == "hello world"


# 功能：验证 write_file 自动创建不存在的父目录
# 设计：路径包含两层不存在的子目录，确认写入后目录结构被创建
@pytest.mark.asyncio
async def test_write_file_creates_parent_dirs(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "file.txt"
    result = await WriteFileTool(tmp_path).invoke(
        {"path": "a/b/file.txt", "content": "x"}
    )
    assert not result.is_error
    assert target.exists()


# 功能：验证 write_file 拒绝包含 .. 的路径并抛出 PermissionError
# 设计：.. 路径遍历与 read_file 遵循相同规则，用相同的断言模式保持一致性
@pytest.mark.asyncio
async def test_write_file_rejects_traversal() -> None:
    with pytest.raises(PermissionError):
        await WriteFileTool().invoke({"path": "../secret.txt", "content": "x"})


# ── list_dir ──────────────────────────────────────────────────────────────────

# 功能：验证 list_dir 输出包含目录中的文件名
# 设计：在 tmp_path 创建已知结构，断言文件名出现在 content 中；不约束格式细节
@pytest.mark.asyncio
async def test_list_dir_shows_files(tmp_path: Path) -> None:
    (tmp_path / "foo.py").write_text("x")
    (tmp_path / "bar.md").write_text("y")
    result = await ListDirTool(tmp_path).invoke({"path": "."})
    assert not result.is_error
    assert "foo.py" in result.content
    assert "bar.md" in result.content


# 功能：验证 list_dir 按 max_depth 限制递归深度（depth=1 时不展示孙级目录内容）
# 设计：创建 parent/child/grandchild 三层，depth=1 时 grandchild 不应出现在输出中
@pytest.mark.asyncio
async def test_list_dir_respects_max_depth(tmp_path: Path) -> None:
    child = tmp_path / "child"
    child.mkdir()
    grandchild = child / "grandchild"
    grandchild.mkdir()
    (grandchild / "deep.txt").write_text("x")

    result = await ListDirTool(tmp_path).invoke({"path": ".", "max_depth": 1})
    assert not result.is_error
    assert "child" in result.content
    assert "deep.txt" not in result.content


# 功能：验证对不存在的路径 list_dir 抛出 FileNotFoundError
# 设计：直接传入不存在的路径字符串，预期抛出标准异常（invocation.py 捕获后返回 error ToolResult）
@pytest.mark.asyncio
async def test_list_dir_missing_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        await ListDirTool(tmp_path).invoke({"path": "does-not-exist"})


# 功能：验证 list_dir 拒绝包含 .. 的路径
# 设计：与 read_file 和 write_file 保持一致的安全规则
@pytest.mark.asyncio
async def test_list_dir_rejects_traversal() -> None:
    with pytest.raises(PermissionError):
        await ListDirTool().invoke({"path": "../"})


# 功能：验证所有路径型工具拒绝绝对路径
# 设计：对 list_dir 与 write_file 使用同一个根内绝对路径，排除仅越界路径才被拒绝的误实现
@pytest.mark.asyncio
async def test_path_tools_reject_absolute_paths(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        await ListDirTool(tmp_path).invoke({"path": str(tmp_path)})
    with pytest.raises(PermissionError):
        await WriteFileTool(tmp_path).invoke(
            {"path": str(tmp_path / "out.txt"), "content": "x"}
        )


# 功能：验证 write_file 不会经由根内符号链接写到工作区外
# 设计：创建指向相邻目录的 symlink 父目录，确认 resolve 后边界检查发生在 mkdir/write 之前
@pytest.mark.asyncio
async def test_write_file_rejects_symlink_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    outside = tmp_path / "outside"
    workspace.mkdir()
    outside.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)

    with pytest.raises(PermissionError):
        await WriteFileTool(workspace).invoke(
            {"path": "link/escaped.txt", "content": "secret"}
        )
    assert not (outside / "escaped.txt").exists()
