from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from cyan.core.incidents.coordinator import _BudgetedJobLogReader
from cyan.core.incidents.log_tool import ReadJobLogTool
from cyan.core.incidents.models import Incident
from cyan.core.incidents.store import IncidentStore
from cyan.core.jobs import JobSpec, JobStore, JobSupervisor


# 功能：验证真实长日志可全文搜索且扫描字节不消耗 Incident 证据预算
# 设计：在临时 Git 仓库运行真实失败子进程，把标记放到 256 KiB 之后并核对持久化用量
async def test_long_log_search_only_charges_returned_evidence(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    script = workspace / "train.py"
    script.write_text(
        "import sys\n"
        "sys.stderr.write('x' * (512 * 1024))\n"
        "sys.stderr.write('ROOT_CAUSE\\n')\n"
        "sys.stderr.write('z' * 1024)\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    subprocess.run(
        ["git", "-C", str(workspace), "init"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.email", "cyan@example.invalid"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "config", "user.name", "cyan test"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "add", "train.py"],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )

    jobs = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(jobs)
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "train.py"],
            workspace_root=workspace,
            env=dict(os.environ),
        )
    )
    await supervisor.wait(job.id)
    failed = jobs.read_job(job.id)
    assert failed.current_attempt_id is not None
    attempt_id = failed.current_attempt_id

    now = datetime.now(UTC)
    incident = Incident(
        id="inc-long-log",
        job_id=job.id,
        attempt_id=attempt_id,
        workspace_root=str(workspace),
        failure_path=str(jobs.attempt_dir(job.id, attempt_id) / "failure.json"),
        created_at=now,
        updated_at=now,
    )
    store = IncidentStore(jobs.job_dir(job.id) / "incidents")
    reader = _BudgetedJobLogReader(jobs, store, incident)
    result = await ReadJobLogTool(reader).invoke(
        {
            "job_id": job.id,
            "attempt_id": attempt_id,
            "stream": "stderr",
            "mode": "search",
            "query": "ROOT_CAUSE",
            "limit": 128,
        }
    )

    payload = json.loads(result.content)
    usage = json.loads(
        (store.incident_dir(incident.id) / "evidence_usage.json").read_text()
    )
    assert payload["match_offset"] > 256 * 1024
    assert "ROOT_CAUSE" in payload["slice"]["content"]
    assert usage == {"bytes_read": 128, "byte_limit": 256 * 1024}
