from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from cyan.core.incidents.patch import (
    PatchError,
    PatchService,
    build_replacement_diff,
)
from cyan.core.incidents.store import IncidentStore
from cyan.core.incidents.tools import ProposePatchTool, SubmitDiagnosisTool


# 初始化仅供 git apply 使用的真实临时工作树
def _init_repo(path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(path)], check=True)


# 构造已保存 proposal 并返回对应 store
async def _proposal(tmp_path: Path, old: str, new: str) -> tuple[Path, IncidentStore]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _init_repo(workspace)
    target = workspace / "train.py"
    target.write_text(old + "\n", encoding="utf-8")
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    store = IncidentStore(tmp_path / "incidents")
    evidence = [
        {
            "source": "workspace",
            "reference": f"train.py@sha256:{digest}#L1-L1",
            "description": "Observed source line.",
        }
    ]
    diagnosis = await SubmitDiagnosisTool(store, "incident-1").invoke(
        {
            "category": "runtime",
            "summary": "observed crash",
            "root_cause": "the observed source causes the crash",
            "evidence": evidence,
            "confidence": 1.0,
        }
    )
    assert not diagnosis.is_error
    result = await ProposePatchTool(store, "incident-1", workspace).invoke(
        {
            "path": "train.py",
            "search": old,
            "replace": new,
            "evidence": evidence,
        }
    )
    assert not result.is_error
    return workspace, store


@pytest.mark.parametrize(
    ("original", "updated"),
    [
        (b"old\n", b"new\n"),
        (b"first\nold one\nold two\nlast\n", b"first\nnew one\nnew two\nlast\n"),
        (b"first\nremove\nlast\n", b"first\nlast\n"),
        (b"old", b"new"),
        (b"old\r\nnext\r\n", b"new\r\nnext\r\n"),
    ],
)
# 功能：验证本地 diff 构造覆盖单行、多行、删除、无末尾换行和 CRLF
# 设计：对每组真实字节运行 git apply --check 和 apply，并比较最终字节语义
def test_build_replacement_diff_is_git_applicable(
    tmp_path: Path,
    original: bytes,
    updated: bytes,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _init_repo(workspace)
    target = workspace / "train.py"
    target.write_bytes(original)
    patch = build_replacement_diff(
        "train.py",
        original.decode("utf-8"),
        updated.decode("utf-8"),
    )
    patch_path = tmp_path / "proposal.diff"
    patch_path.write_bytes(patch.encode("utf-8"))

    subprocess.run(
        ["git", "-C", str(workspace), "apply", "--check", str(patch_path)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "apply", str(patch_path)],
        check=True,
        capture_output=True,
    )

    assert target.read_bytes() == updated


# 功能：验证 PatchService 经 git apply --check 应用并可按 receipt 安全反向恢复
# 设计：使用真实 Git 工作树贯穿 proposal、apply、reverse，避免 mock 掩盖 Git patch 语义
async def test_patch_service_apply_and_safe_reverse(tmp_path: Path) -> None:
    workspace, store = await _proposal(tmp_path, "old", "new")
    proposal = store.read_proposal("incident-1")
    service = PatchService(workspace)

    receipt = await service.apply(proposal, store.patch_path(proposal))

    assert (workspace / "train.py").read_text(encoding="utf-8") == "new\n"
    await service.reverse(proposal, store.patch_path(proposal), receipt)
    assert (workspace / "train.py").read_text(encoding="utf-8") == "old\n"


# 功能：验证 apply 前目标文件哈希变化会被判定为 stale
# 设计：proposal 后人为修改文件并调用真实 Git 服务，确认 hash recheck 先于写入
async def test_patch_service_rejects_stale_base(tmp_path: Path) -> None:
    workspace, store = await _proposal(tmp_path, "old", "new")
    proposal = store.read_proposal("incident-1")
    (workspace / "train.py").write_text("user edit\n", encoding="utf-8")

    with pytest.raises(PatchError, match="stale target"):
        await PatchService(workspace).apply(proposal, store.patch_path(proposal))

    assert (workspace / "train.py").read_text(encoding="utf-8") == "user edit\n"


# 功能：验证 apply 后用户再次修改文件会阻止自动 reverse
# 设计：先真实应用再制造 smoke 期间变更，确保 receipt hash 门禁不会覆盖用户内容
async def test_patch_service_reverse_rejects_changed_applied_state(tmp_path: Path) -> None:
    workspace, store = await _proposal(tmp_path, "old", "new")
    proposal = store.read_proposal("incident-1")
    service = PatchService(workspace)
    receipt = await service.apply(proposal, store.patch_path(proposal))
    (workspace / "train.py").write_text("later edit\n", encoding="utf-8")

    with pytest.raises(PatchError, match="applied state changed"):
        await service.reverse(proposal, store.patch_path(proposal), receipt)

    assert (workspace / "train.py").read_text(encoding="utf-8") == "later edit\n"
