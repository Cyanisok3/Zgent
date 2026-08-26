from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

import cyan.training.incidents.runtime as runtime_module
from cyan.agent.events.bus import EventBus
from cyan.agent.runner import RunOutcome
from cyan.config import CyanConfig
from cyan.training.incidents.models import FailureCapsule, Incident, LogSnapshot
from cyan.training.incidents.runtime import IncidentRuntime
from cyan.training.incidents.store import IncidentStore
from cyan.training.jobs.models import (
    AttemptRecord,
    FailureRecord,
    JobRecord,
    JobSpec,
)
from cyan.training.jobs.store import JobStore


# 构造 Runtime 读取所需的失败 Job、Attempt、日志和 Incident
def _seed_runtime(tmp_path: Path) -> tuple[JobStore, IncidentStore, str, str]:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    jobs = JobStore(tmp_path / "jobs")
    now = datetime.now(UTC).isoformat()
    job = JobRecord(
        id="job-1",
        status="failed",
        created_at=now,
        updated_at=now,
        current_attempt_id="attempt-1",
        attempt_ids=["attempt-1"],
    )
    jobs.create_job(job, JobSpec(argv=["python", "train.py"], workspace_root=workspace))
    attempt = AttemptRecord(
        id="attempt-1",
        job_id="job-1",
        status="failed",
        started_at=now,
        finished_at=now,
        returncode=1,
    )
    jobs.write_attempt(attempt)
    stdout = b"normal output\n"
    stderr = b"RuntimeError: failed\n"
    stdout_path = jobs.log_path("job-1", "attempt-1", "stdout")
    stderr_path = jobs.log_path("job-1", "attempt-1", "stderr")
    stdout_path.write_bytes(stdout)
    stderr_path.write_bytes(stderr)
    capsule = FailureCapsule(
        job_id="job-1",
        attempt_id="attempt-1",
        argv=["python", "train.py"],
        cwd=str(workspace),
        occurred_at=datetime.now(UTC),
        failure_kind="process_exit",
        returncode=1,
        stdout=LogSnapshot(
            size=len(stdout),
            sha256=hashlib.sha256(stdout).hexdigest(),
            included_start=0,
            included_end=len(stdout),
            tail=stdout.decode(),
        ),
        stderr=LogSnapshot(
            size=len(stderr),
            sha256=hashlib.sha256(stderr).hexdigest(),
            included_start=0,
            included_end=len(stderr),
            tail=stderr.decode(),
        ),
    )
    jobs.write_failure(
        FailureRecord(
            job_id="job-1",
            attempt_id="attempt-1",
            occurred_at=now,
            kind="process_exit",
            returncode=1,
            message="failed",
            capsule=capsule.model_dump(mode="json"),
        )
    )
    incidents = IncidentStore(jobs.job_dir("job-1") / "incidents")
    incidents.write_incident(
        Incident(
            id="incident-1",
            job_id="job-1",
            attempt_id="attempt-1",
            workspace_root=str(workspace),
            failure_path=str(jobs.attempt_dir("job-1", "attempt-1") / "failure.json"),
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
        )
    )
    run = incidents.create_run("incident-1", "run-1", "initial", "inspect")
    return jobs, incidents, "incident-1", run.run_id


# 使用空 Provider 隔离 Runtime 后置条件，不触发真实模型调用
class _SuccessfulRunner:
    # 返回没有诊断 artifact 的伪成功结果
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    # 模拟 AgentRunner 成功但未写入 Diagnosis
    async def run_and_capture(self, *args: Any, **kwargs: Any) -> RunOutcome:
        del args, kwargs
        return RunOutcome(status="success", result="done", reason=None)


# 功能：验证 Runner 伪成功但缺少 Diagnosis 时 Runtime 改记 diagnosis_missing
# 设计：使用真实日志和持久化 Failure Capsule，确保缺失结果不会被统计成 abstain 或 success
@pytest.mark.asyncio
async def test_runtime_marks_missing_diagnosis_as_failed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    jobs, store, incident_id, run_id = _seed_runtime(tmp_path)
    monkeypatch.setattr(runtime_module, "AgentRunner", _SuccessfulRunner)

    outcome = await IncidentRuntime(
        jobs,
        store,
        CyanConfig(),
        bus=EventBus(),
    ).run(incident_id, run_id)

    assert outcome.status == "failed"
    assert outcome.reason == "diagnosis_missing"
    persisted = store.read_run(incident_id, run_id)
    assert persisted.status == "failed"
    assert persisted.reason == "diagnosis_missing"
