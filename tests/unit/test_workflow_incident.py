from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from cyan.core.events.bus import EventBus
from cyan.core.incidents.coordinator import IncidentCoordinator
from cyan.core.jobs import (
    JobSpec,
    JobStore,
    JobSupervisor,
    WorkflowArtifact,
    WorkflowContract,
)


class _Sessions:
    # 初始化只记录创建和 LLM 消息次数的最小 SessionManager 替身
    def __init__(self) -> None:
        self.created = 0
        self.messages = 0

    # 返回带稳定 ID 的 daemon-owned Incident session
    async def create(self, **kwargs: object) -> SimpleNamespace:
        self.created += 1
        return SimpleNamespace(id=f"session-{self.created}")

    # 记录会触发 Incident Agent 的消息调用
    async def send_message(self, *args: object, **kwargs: object) -> None:
        self.messages += 1


# 等待 Incident 达到目标状态并返回最新安全视图
async def _wait_status(
    coordinator: IncidentCoordinator,
    job_id: str,
    status: str,
) -> dict[str, Any]:
    for _ in range(300):
        view = await coordinator.job_view(job_id)
        incident = view.get("incident")
        if isinstance(incident, dict) and incident.get("status") == status:
            return view
        await asyncio.sleep(0.01)
    raise AssertionError(f"incident did not reach {status}")


# 功能：验证 preflight input 违约零 LLM 路由到 action_required 并按 frozen Contract 重检
# 设计：首轮缺文件，之后修改磁盘 Contract 再补原输入，证明 retry 只读 launch.json snapshot
async def test_deterministic_contract_failure_and_frozen_retry(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    sessions = _Sessions()
    bus = EventBus()
    holder: dict[str, IncidentCoordinator] = {}

    # 将真实 Supervisor failure 转交给当前 Coordinator
    async def on_failure(job: Any, attempt: Any, failure: Any) -> None:
        await holder["coordinator"].handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, failure_callback=on_failure)
    coordinator = IncidentCoordinator(jobs, sessions, supervisor, bus)  # type: ignore[arg-type]
    holder["coordinator"] = coordinator
    contract = WorkflowContract(
        artifacts=[WorkflowArtifact(path="data/train.csv", role="input", min_bytes=1)]
    )
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "print('ran')"],
            workspace_root=tmp_path,
            workflow_contract=contract,
        )
    )
    await supervisor.wait(job.id)

    view = await _wait_status(coordinator, job.id, "action_required")

    assert sessions.created == 1
    assert sessions.messages == 0
    assert view["proposal"] is None
    assert view["diagnosis"]["recovery"]["kind"] == "operator_action"
    assert view["diagnosis"]["evidence"][0]["source"] == "contract"
    launch = jobs.read_spec(job.id)
    assert launch.workflow_contract == contract

    disk_contract = tmp_path / ".cyan" / "workflow.toml"
    disk_contract.parent.mkdir()
    disk_contract.write_text(
        'version = 1\n[[artifacts]]\npath = "different.csv"\nrole = "input"\nrequired = true\n',
        encoding="utf-8",
    )
    data = tmp_path / "data"
    data.mkdir()
    (data / "train.csv").write_text("row\n", encoding="utf-8")
    incident_id = str(view["incident"]["id"])

    assert await coordinator.retry(incident_id) == "retry_running"
    resolved = await _wait_status(coordinator, job.id, "resolved")

    assert resolved["job"]["status"] == "succeeded"
    assert len(resolved["job"]["attempt_ids"]) == 2
    assert sessions.messages == 0
    await coordinator.close()


# 功能：验证 incident.retry 只接受 action_required 状态
# 设计：构造普通 postflight 违约进入 Agent 路径，再直接调用 retry 检查状态闸门
async def test_incident_retry_rejects_non_action_required(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    sessions = _Sessions()
    bus = EventBus()
    holder: dict[str, IncidentCoordinator] = {}

    # 将真实 workflow failure 转交 Coordinator
    async def on_failure(job: Any, attempt: Any, failure: Any) -> None:
        await holder["coordinator"].handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, failure_callback=on_failure)
    coordinator = IncidentCoordinator(jobs, sessions, supervisor, bus)  # type: ignore[arg-type]
    holder["coordinator"] = coordinator
    contract = WorkflowContract(
        artifacts=[WorkflowArtifact(path="model.pt", role="output", min_bytes=1)]
    )
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "pass"],
            workspace_root=tmp_path,
            workflow_contract=contract,
        )
    )
    await supervisor.wait(job.id)
    view = await _wait_status(coordinator, job.id, "unresolved")

    try:
        await coordinator.retry(str(view["incident"]["id"]))
    except ValueError as error:
        assert "operator action" in str(error)
    else:
        raise AssertionError("retry unexpectedly accepted unresolved Incident")
    assert sessions.messages == 1
    await coordinator.close()


# 功能：验证人工动作后的新失败仍复用同一 Incident 和只读 session
# 设计：先触发确定性缺输入，再补输入让 frozen main 非零退出，检查 Attempt 更新而不新建 Incident
async def test_operator_retry_failure_reuses_incident_session(tmp_path: Path) -> None:
    jobs = JobStore(tmp_path / "jobs")
    sessions = _Sessions()
    bus = EventBus()
    holder: dict[str, IncidentCoordinator] = {}

    # 将真实 workflow failure 转交 Coordinator
    async def on_failure(job: Any, attempt: Any, failure: Any) -> None:
        await holder["coordinator"].handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, failure_callback=on_failure)
    coordinator = IncidentCoordinator(jobs, sessions, supervisor, bus)  # type: ignore[arg-type]
    holder["coordinator"] = coordinator
    contract = WorkflowContract(
        artifacts=[WorkflowArtifact(path="input.csv", role="input", min_bytes=1)]
    )
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "raise SystemExit(9)"],
            workspace_root=tmp_path,
            workflow_contract=contract,
        )
    )
    await supervisor.wait(job.id)
    first = await _wait_status(coordinator, job.id, "action_required")
    incident_id = str(first["incident"]["id"])
    session_id = str(first["incident"]["session_id"])
    (tmp_path / "input.csv").write_text("ready\n", encoding="utf-8")

    assert await coordinator.retry(incident_id) == "retry_running"
    second = await _wait_status(coordinator, job.id, "unresolved")

    assert second["incident"]["id"] == incident_id
    assert second["incident"]["session_id"] == session_id
    assert second["incident"]["attempt_id"] == "attempt-0002"
    assert second["diagnosis"] is None
    assert sessions.created == 1
    assert sessions.messages == 1
    assert jobs.read_failure(job.id, "attempt-0002").contract_fingerprint is not None
    await coordinator.close()
