from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from cyan.training.incidents.models import Incident, Proposal, ProposalFile
from cyan.training.incidents.store import IncidentStore


# 构造最小 Incident 快照供存储测试使用
def _incident() -> Incident:
    now = datetime.now(UTC)
    return Incident(
        id="incident-1",
        job_id="job-1",
        attempt_id="attempt-1",
        workspace_root="/tmp/workspace",
        failure_path="attempts/attempt-1/failure.json",
        created_at=now,
        updated_at=now,
    )


# 功能：验证 run.json 在后台任务前落盘且不创建旧 sessions 目录
# 设计：直接调用 create_run 后读取文件，锁定 Incident-owned persistence 边界
def test_create_run_persists_prepared_record(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")
    store.write_incident(_incident())

    run = store.create_run(
        "incident-1",
        "run-1",
        "follow_up",
        "inspect again",
        attempt_id="attempt-1",
    )

    path = store.run_dir("incident-1", run.run_id)
    assert (path / "run.json").exists()
    assert store.read_run("incident-1", "run-1").status == "prepared"
    assert not (tmp_path / "sessions").exists()


# 功能：验证 Proposal diff 与元数据采用先文件后快照的合并写入
# 设计：通过真实读取确认只出现 proposal.diff 和嵌套 proposal，不生成 proposal.json
def test_proposal_is_nested_and_diff_is_single_artifact(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")
    store.write_incident(_incident())
    proposal = Proposal(
        id="proposal-1",
        incident_id="incident-1",
        patch_sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        files=[ProposalFile(path="train.py", change_type="modify")],
        created_at=datetime.now(UTC),
    )

    store.write_proposal(proposal, "diff --git a/train.py b/train.py\n")

    saved = store.read_incident("incident-1")
    assert saved.proposal is not None
    assert store.read_patch(proposal).startswith("diff --git")
    assert not (store.incident_dir("incident-1") / "proposal.json").exists()
    store.clear_proposal("incident-1")
    assert store.read_incident("incident-1").proposal is None
    assert not (store.incident_dir("incident-1") / "proposal.diff").exists()
