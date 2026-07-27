from __future__ import annotations

import json
from pathlib import Path

import pytest

from cyan.core.incidents.store import IncidentStore
from cyan.core.incidents.tools import ProposePatchTool, SubmitDiagnosisTool


# 生成修改单文件的最小 unified diff
def _modify_patch(path: str, old: str, new: str) -> str:
    return (
        f"diff --git a/{path} b/{path}\n"
        f"--- a/{path}\n"
        f"+++ b/{path}\n"
        "@@ -1 +1 @@\n"
        f"-{old}\n"
        f"+{new}\n"
    )


# 功能：验证 submit_diagnosis 按 ML taxonomy 保存结构化 evidence
# 设计：通过真实 store 往返并比较关键字段，覆盖工具模型和 artifact 文件边界
async def test_submit_diagnosis_writes_typed_artifact(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")
    tool = SubmitDiagnosisTool(store, "incident-1")

    result = await tool.invoke(
        {
            "category": "shape",
            "summary": "batch dimensions differ",
            "root_cause": "collate_fn returns an extra axis",
            "evidence": [
                {
                    "source": "stderr",
                    "reference": "bytes 120:180",
                    "description": "tensor size mismatch traceback",
                }
            ],
            "confidence": 0.9,
        }
    )

    payload = json.loads(result.content)
    diagnosis = store.read_diagnosis("incident-1")
    assert diagnosis.id == payload["id"]
    assert diagnosis.category == "shape"
    assert diagnosis.evidence[0].reference == "bytes 120:180"


# 功能：验证诊断工具拒绝 harness 未登记为已观察的证据引用
# 设计：注入只接受单一 byte range 的 validator，提交另一范围并断言零 artifact 写入
async def test_submit_diagnosis_rejects_unobserved_evidence(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")
    observed = {("stderr", "bytes 120:180")}

    # 根据测试中的已观察登记表返回具体校验错误
    def validate_evidence(evidence: object) -> str | None:
        source = getattr(evidence, "source")
        reference = getattr(evidence, "reference")
        if (source, reference) in observed:
            return None
        return f"unobserved evidence reference: {source}:{reference}"

    tool = SubmitDiagnosisTool(
        store,
        "incident-1",
        evidence_validator=validate_evidence,
    )

    result = await tool.invoke(
        {
            "category": "shape",
            "summary": "batch dimensions differ",
            "root_cause": "collate_fn returns an extra axis",
            "evidence": [
                {
                    "source": "stderr",
                    "reference": "bytes 900:950",
                    "description": "invented traceback location",
                }
            ],
            "confidence": 0.9,
        }
    )

    assert result.is_error
    assert result.error_type == "schema_error"
    assert "unobserved evidence reference" in result.content
    assert not (store.incident_dir("incident-1") / "diagnosis.json").exists()


# 功能：验证 propose_patch 只写 artifact 并记录目标文件原始 SHA-256
# 设计：调用前后比较 workspace 文件，同时读取 proposal.json 和 proposal.diff 排除隐式修改
async def test_propose_patch_does_not_modify_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("old\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)
    patch = _modify_patch("train.py", "old", "new")

    result = await tool.invoke({"patch": patch})

    assert not result.is_error
    proposal = store.read_proposal("incident-1")
    assert target.read_text(encoding="utf-8") == "old\n"
    assert proposal.files[0].base_sha256 is not None
    assert store.read_patch(proposal) == patch


@pytest.mark.parametrize(
    "patch",
    [
        _modify_patch("../outside.py", "old", "new"),
        _modify_patch("/absolute.py", "old", "new"),
        _modify_patch(".cyan/config.toml", "old", "new"),
        _modify_patch(".cyan/config.toml", "old", "new"),
        "GIT binary patch\nliteral 1\nA\n",
    ],
)
# 功能：验证 propose_patch 拒绝逃逸路径、verifier 配置和 binary patch
# 设计：参数化覆盖同一安全边界的五种输入，并断言不会产生 proposal artifact
async def test_propose_patch_rejects_unsafe_diff(tmp_path: Path, patch: str) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)

    result = await tool.invoke({"patch": patch})

    assert result.is_error
    assert result.error_type == "schema_error"
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()
