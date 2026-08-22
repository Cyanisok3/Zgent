from __future__ import annotations

import ast
from pathlib import Path


# 收集指定源码包下的所有绝对导入前缀
def _imports_under(package: str) -> set[str]:
    root = Path(__file__).resolve().parents[2] / "src" / "cyan" / package
    imports: set[str] = set()
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imports.add(node.module)
    return imports


# 功能：验证 Agent 和训练 Jobs 不反向依赖服务层或其他领域
# 设计：用 AST 检查源码导入前缀，避免运行时导入掩盖架构环路
def test_domain_import_boundaries() -> None:
    agent_imports = _imports_under("agent")
    jobs_imports = _imports_under("training/jobs")
    incidents_imports = _imports_under("training/incidents")

    assert not any(item.startswith("cyan.service") for item in agent_imports)
    assert not any(item.startswith("cyan.training") for item in agent_imports)
    assert not any(item.startswith("cyan.agent") for item in jobs_imports)
    assert not any(item.startswith("cyan.service") for item in jobs_imports)
    assert not any(item.startswith("cyan.service") for item in incidents_imports)
