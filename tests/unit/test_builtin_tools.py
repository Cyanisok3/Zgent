from pathlib import Path

import pytest

from cyan.agent.tools.builtin import ListDirTool, ReadFileTool, SearchTextTool


# 功能：验证三项基础工具只读取绑定工作区
# 设计：使用临时目录和真实文件，不引入模拟文件系统
@pytest.mark.asyncio
async def test_read_only_builtin_tools(tmp_path: Path) -> None:
    (tmp_path / "train.py").write_text("print('ok')\n", encoding="utf-8")
    read = await ReadFileTool(tmp_path).invoke({"path": "train.py"})
    listed = await ListDirTool(tmp_path).invoke({})
    searched = await SearchTextTool(tmp_path).invoke({"query": "print"})
    assert "print('ok')" in read.content
    assert "train.py" in listed.content
    assert "train.py@sha256:" in searched.content


# 功能：验证基础工具拒绝工作区外路径
# 设计：走真实路径校验边界，确保 Incident profile 不能越界
@pytest.mark.asyncio
async def test_builtin_tools_reject_path_escape(tmp_path: Path) -> None:
    with pytest.raises(PermissionError):
        await ReadFileTool(tmp_path).invoke({"path": "../outside.txt"})
