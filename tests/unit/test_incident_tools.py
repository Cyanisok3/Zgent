from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from cyan.training.incidents.patch import PatchService, sha256_bytes
from cyan.training.incidents.store import IncidentStore
from cyan.training.incidents.tools import ProposePatchTool, SubmitDiagnosisTool


# 为当前真实文件构造可校验的 workspace evidence
def _workspace_evidence(
    path: str,
    target: Path,
    *,
    start: int = 1,
    end: int = 1,
) -> dict[str, str]:
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    return {
        "source": "workspace",
        "reference": f"{path}@sha256:{digest}#L{start}-L{end}",
        "description": "Observed source used for the replacement.",
    }


# 调用结构化替换工具并默认引用目标文件首行
async def _propose(
    tool: ProposePatchTool,
    target: Path,
    *,
    path: str = "train.py",
    search: str,
    replace: str,
    evidence: list[dict[str, str]] | None = None,
) -> object:
    try:
        tool._store.read_diagnosis(tool._incident_id)
    except FileNotFoundError:
        await SubmitDiagnosisTool(tool._store, tool._incident_id).invoke(
            {
                "category": "runtime",
                "summary": "observed crash",
                "root_cause": "the observed source causes the crash",
                "evidence": [_workspace_evidence(path, target)],
                "confidence": 1.0,
            }
        )
    return await tool.invoke(
        {
            "path": path,
            "search": search,
            "replace": replace,
            "evidence": evidence or [_workspace_evidence(path, target)],
        }
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


# 功能：验证结构化替换只写 artifact 并记录真实文件哈希
# 设计：在真实 Git 工作树执行 preflight，比较调用前后文件并核对生成 diff 的摘要
async def test_propose_patch_generates_diff_without_workspace_write(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    subprocess.run(["git", "init", "-q", str(workspace)], check=True)
    target = workspace / "train.py"
    target.write_text("old\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(
        store,
        "incident-1",
        workspace,
        patch_service=PatchService(workspace),
    )

    result = await _propose(tool, target, search="old", replace="new")

    assert not getattr(result, "is_error")
    proposal = store.read_proposal("incident-1")
    patch = store.read_patch(proposal)
    assert target.read_text(encoding="utf-8") == "old\n"
    assert proposal.files[0].path == "train.py"
    assert proposal.files[0].change_type == "modify"
    assert proposal.files[0].base_sha256 == hashlib.sha256(b"old\n").hexdigest()
    assert proposal.patch_sha256 == sha256_bytes(patch.encode("utf-8"))
    assert proposal.diagnosis_id == store.read_diagnosis("incident-1").id
    assert "-old\n+new\n" in patch


# 功能：验证 propose_patch 由 harness 绑定当前 diagnosis 且不接受模型提供 ID
# 设计：检查公开 schema 并在缺少 diagnosis 时直接调用工具，断言不会生成 proposal
async def test_propose_patch_requires_harness_managed_diagnosis(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("old\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)

    result = await tool.invoke(
        {
            "path": "train.py",
            "search": "old",
            "replace": "new",
            "evidence": [_workspace_evidence("train.py", target)],
        }
    )

    assert "diagnosis_id" not in tool.input_schema["properties"]
    assert getattr(result, "is_error")
    assert "submit_diagnosis must succeed" in getattr(result, "content")
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()


# 功能：验证零匹配仅首次返回 evidence 范围内的有界真实源码
# 设计：同一工具连续失败两次，断言第二次明确停止且不再次泄露校正片段
async def test_propose_patch_zero_match_allows_one_bounded_correction(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("first\nactual source\nlast\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)
    evidence = [_workspace_evidence("train.py", target, start=2, end=2)]

    first = await _propose(
        tool,
        target,
        search="wrong source",
        replace="fixed",
        evidence=evidence,
    )
    second = await _propose(
        tool,
        target,
        search="still wrong",
        replace="fixed",
        evidence=evidence,
    )

    assert getattr(first, "is_error")
    assert "actual source" in getattr(first, "content")
    assert len(getattr(first, "content").encode("utf-8")) < 9 * 1024
    assert getattr(second, "is_error")
    assert "stop proposing a patch" in getattr(second, "content")
    assert "actual source" not in getattr(second, "content")
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()


# 功能：验证多匹配返回候选起始行并要求扩展 SEARCH
# 设计：构造两处相同代码，检查确定性行号反馈且工作区和 artifact 均不变
async def test_propose_patch_multiple_matches_reports_lines(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("same\nmiddle\nsame\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)

    result = await _propose(tool, target, search="same", replace="new")

    assert getattr(result, "is_error")
    assert "matches 2 locations" in getattr(result, "content")
    assert "[1, 3]" in getattr(result, "content")
    assert target.read_text(encoding="utf-8") == "same\nmiddle\nsame\n"
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()


@pytest.mark.parametrize("search", [" value = 1", "value  = 1", "value = 1\n"])
# 功能：验证缩进、空白和换行差异不会触发模糊替换
# 设计：为每种近似文本创建独立工具，断言全部按零精确匹配拒绝
async def test_propose_patch_rejects_inexact_search(
    tmp_path: Path,
    search: str,
) -> None:
    workspace = tmp_path / search.encode().hex()
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("value = 1", encoding="utf-8")
    store = IncidentStore(tmp_path / f"incidents-{search.encode().hex()}")
    tool = ProposePatchTool(store, "incident-1", workspace)

    result = await _propose(tool, target, search=search, replace="value = 2")

    assert getattr(result, "is_error")
    assert "no exact match" in getattr(result, "content")
    assert target.read_text(encoding="utf-8") == "value = 1"


# 功能：验证 no-op 和 stale workspace evidence 均被拒绝
# 设计：分别提交相同替换与旧哈希引用，断言不会创建 proposal artifact
async def test_propose_patch_rejects_noop_and_stale_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_text("old\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)

    noop = await _propose(tool, target, search="old", replace="old")
    stale = await _propose(
        tool,
        target,
        search="old",
        replace="new",
        evidence=[
            {
                "source": "workspace",
                "reference": f"train.py@sha256:{'0' * 64}#L1-L1",
                "description": "Stale source.",
            }
        ],
    )

    assert getattr(noop, "is_error")
    assert "does not change" in getattr(noop, "content")
    assert getattr(stale, "is_error")
    assert "current SHA-256" in getattr(stale, "content")
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"value\x00data", "binary"),
        (b"\xff\xfe", "UTF-8"),
        (b"x" * (1024 * 1024 + 1), "too large"),
    ],
)
# 功能：验证二进制、非 UTF-8 和超过 1 MiB 的文件不可提补丁
# 设计：直接写入三类真实字节内容并断言读取边界在 artifact 前生效
async def test_propose_patch_rejects_unsupported_files(
    tmp_path: Path,
    content: bytes,
    message: str,
) -> None:
    workspace = tmp_path / message
    workspace.mkdir()
    target = workspace / "train.py"
    target.write_bytes(content)
    store = IncidentStore(tmp_path / f"incidents-{message}")
    tool = ProposePatchTool(store, "incident-1", workspace)

    result = await _propose(tool, target, search="x", replace="y")

    assert getattr(result, "is_error")
    assert message in getattr(result, "content")
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()


# 功能：验证路径逃逸与受保护 verifier 配置不可修改
# 设计：使用真实外部文件和真实配置文件，断言两类路径均在生成 artifact 前拒绝
async def test_propose_patch_rejects_unsafe_and_protected_paths(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("old\n", encoding="utf-8")
    config = workspace / ".cyan" / "config.toml"
    config.parent.mkdir()
    config.write_text("old\n", encoding="utf-8")
    store = IncidentStore(tmp_path / "incidents")
    tool = ProposePatchTool(store, "incident-1", workspace)

    escaped = await _propose(
        tool,
        outside,
        path="../outside.py",
        search="old",
        replace="new",
    )
    protected = await _propose(
        tool,
        config,
        path=".cyan/config.toml",
        search="old",
        replace="new",
    )

    assert getattr(escaped, "is_error")
    assert "unsafe workspace path" in getattr(escaped, "content")
    assert getattr(protected, "is_error")
    assert "verifier config is protected" in getattr(protected, "content")
    assert outside.read_text(encoding="utf-8") == "old\n"
    assert config.read_text(encoding="utf-8") == "old\n"
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()
