from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cyan.training.incidents.context import (
    MAX_INITIAL_EVIDENCE_BYTES,
    MAX_INPUT_BYTES,
    build_incident_context,
)
from cyan.training.incidents.models import FailureCapsule, Incident, LogSnapshot
from cyan.training.incidents.selector import EvidenceSelection


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
        content="[stderr bytes=0-3 kind=tail]\nerr",
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
    assert MAX_INPUT_BYTES == 128 * 1024
    assert MAX_INITIAL_EVIDENCE_BYTES == 32 * 1024
