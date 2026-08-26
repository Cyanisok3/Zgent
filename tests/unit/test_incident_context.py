from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cyan.training.incidents.context import (
    MAX_INITIAL_EVIDENCE_BYTES,
    MAX_INPUT_BYTES,
    build_incident_context,
)
from cyan.training.incidents.models import FailureCapsule, Incident, LogSnapshot
from cyan.training.incidents.profile import build_incident_profile
from cyan.training.incidents.selector import EvidenceSelection
from cyan.training.incidents.store import IncidentStore
from cyan.training.jobs.store import JobStore


# 构造当前 Incident context 测试使用的失败胶囊
def _capsule(tmp_path: Path) -> FailureCapsule:
    snapshot = LogSnapshot(
        size=3,
        sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        included_start=0,
        included_end=3,
        tail="err",
    )
    return FailureCapsule(
        job_id="job-1",
        attempt_id="attempt-1",
        argv=["python", "train.py"],
        cwd=str(tmp_path),
        occurred_at=datetime.now(UTC),
        failure_kind="process_exit",
        returncode=1,
        stdout=snapshot,
        stderr=snapshot,
    )


# 构造最小 Incident 快照
def _incident(tmp_path: Path) -> Incident:
    now = datetime.now(UTC)
    return Incident(
        id="incident-1",
        job_id="job-1",
        attempt_id="attempt-1",
        workspace_root=str(tmp_path),
        failure_path="failure.json",
        created_at=now,
        updated_at=now,
    )


# 功能：验证 context 只带当前指令、失败胶囊和 selector 证据
# 设计：检查固定 prompt 中没有历史会话或全局 context，并锁定两级预算常量
def test_incident_context_contains_only_bounded_inputs(tmp_path: Path) -> None:
    selection = EvidenceSelection(
        content=(
            '{"source":"stderr","reference":"stderr:job-1/attempt-1@bytes:0-3",'
            '"description":"stderr tail"}\nerr'
        ),
        references=[],
        scanned_bytes=3,
        selected_bytes=3,
        duplicates_removed=0,
        stdout_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        stderr_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )

    context = build_incident_context(
        _incident(tmp_path),
        _capsule(tmp_path),
        selection,
        "inspect",
        {"diagnosis": {"summary": "previous"}},
    )

    assert "Current instruction:\ninspect" in context.system_prompt
    assert '"summary": "previous"' in context.system_prompt
    assert "session" not in context.system_prompt.lower()
    assert "causal_support=direct or inferred" in context.system_prompt
    assert "Inference — not directly established by observed evidence:" in context.system_prompt
    assert "patch_recommended=true" in context.system_prompt
    assert MAX_INPUT_BYTES == 128 * 1024
    assert MAX_INITIAL_EVIDENCE_BYTES == 32 * 1024


# 功能：验证 Incident profile 拒绝仅格式合法但未被工具观测的源码引用
# 设计：直接调用 profile 中的诊断工具，用伪造 SHA 证明统一登记表是唯一准入
async def test_incident_profile_rejects_unobserved_workspace_reference(
    tmp_path: Path,
) -> None:
    incident = _incident(tmp_path)
    store = IncidentStore(tmp_path / "incidents")
    store.write_incident(incident)
    selection = EvidenceSelection(
        content="",
        references=[],
        scanned_bytes=0,
        selected_bytes=0,
        duplicates_removed=0,
        stdout_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        stderr_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    )
    profile = build_incident_profile(
        JobStore(tmp_path / "jobs"),
        store,
        incident,
        _capsule(tmp_path),
        selection,
        "inspect",
    )
    tool = next(item for item in profile.tools if item.name == "submit_diagnosis")

    result = await tool.invoke(
        {
            "category": "runtime",
            "summary": "invented evidence",
            "root_cause": "not actually observed",
            "evidence": [
                {
                    "source": "workspace",
                    "reference": f"train.py@sha256:{'a' * 64}#L1",
                    "description": "syntactically valid but invented",
                }
            ],
            "confidence": 0.5,
            "causal_support": "inferred",
            "patch_recommended": False,
        }
    )

    assert result.is_error
    assert "was not observed" in result.content
