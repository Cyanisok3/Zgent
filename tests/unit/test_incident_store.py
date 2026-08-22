from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from cyan.training.incidents.models import Incident
from cyan.training.incidents.store import IncidentStore


# 功能：验证 Incident typed model 能通过文件 store 原子往返
# 设计：使用真实临时目录并比较完整模型，覆盖时间与绝对 workspace 路径的 JSON 序列化
def test_incident_store_roundtrip(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")
    now = datetime.now(UTC)
    incident = Incident(
        id="incident-1",
        job_id="job-1",
        attempt_id="attempt-1",
        workspace_root=str(tmp_path),
        failure_path="attempts/attempt-1/failure.json",
        created_at=now,
        updated_at=now,
    )

    store.write_incident(incident)

    assert store.read_incident("incident-1") == incident
    assert not list(store.incident_dir("incident-1").glob("*.tmp"))


# 功能：验证 artifact store 拒绝可逃逸根目录的 incident ID
# 设计：直接调用路径入口而非依赖模型校验，覆盖来自 RPC 或损坏磁盘数据的防御边界
def test_incident_store_rejects_unsafe_id(tmp_path: Path) -> None:
    store = IncidentStore(tmp_path / "incidents")

    with pytest.raises(ValueError, match="unsafe incident id"):
        store.incident_dir("../outside")


# 功能：验证 Incident 拒绝相对 workspace_root
# 设计：在模型边界构造相对路径，确保后续工具不可能回退到 daemon 当前工作目录
def test_incident_requires_absolute_workspace() -> None:
    now = datetime.now(UTC)

    with pytest.raises(ValueError, match="workspace_root must be absolute"):
        Incident(
            id="incident-1",
            job_id="job-1",
            attempt_id="attempt-1",
            workspace_root="relative/project",
            failure_path="failure.json",
            created_at=now,
            updated_at=now,
        )
