from pathlib import Path

import pytest
from pydantic import ValidationError

from cyan.agent.tools.builtin import ReadFileTool, SearchTextTool


# 功能：验证读取工具拒绝空路径参数
# 设计：通过工具自身 Pydantic schema 检查，不绕过调用边界
@pytest.mark.asyncio
async def test_read_file_requires_path(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        await ReadFileTool(tmp_path).invoke({})


# 功能：验证搜索工具拒绝空查询
# 设计：确认工具 schema 在 LLM 调用前阻止无效请求
@pytest.mark.asyncio
async def test_search_requires_query(tmp_path: Path) -> None:
    with pytest.raises(ValidationError):
        await SearchTextTool(tmp_path).invoke({})
