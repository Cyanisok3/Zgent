from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from cyan.training.incidents.evidence import build_failure_capsule
from cyan.training.incidents.selector import select_evidence
from cyan.training.jobs import JobSpec, JobStore, JobSupervisor


# 功能：验证超过 5 MiB 的真实训练日志仍可分块扫描并压缩到 32 KiB
# 设计：真实 Git 工作区和真实 Python 子进程产生长 stderr，selector 不通过 Agent 或工具预算
async def test_long_log_selector_is_bounded_and_relocatable(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    script = workspace / "train.py"
    frame = repr(f"  File {workspace / 'train.py'}, line 4, in <module>\n")
    script.write_text(
        "import sys\n"
        "for _ in range(1500):\n"
        "    sys.stderr.write('x' * 4096 + '\\n')\n"
        "sys.stderr.write('Traceback (most recent call last):\\n')\n"
        f"sys.stderr.write({frame})\n"
        "sys.stderr.write('RuntimeError: ROOT_CAUSE\\n')\n"
        "raise SystemExit(2)\n",
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(workspace), "init"], check=True, capture_output=True)
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
    subprocess.run(["git", "-C", str(workspace), "add", "train.py"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(workspace), "commit", "-m", "fixture"],
        check=True,
        capture_output=True,
    )

    jobs = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(jobs)
    job = await supervisor.start(
        JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace, env=dict(os.environ))
    )
    await supervisor.wait(job.id)
    failed = jobs.read_job(job.id)
    assert failed.current_attempt_id is not None
    failure = jobs.read_failure(job.id, failed.current_attempt_id)
    capsule = await build_failure_capsule(jobs, failure)
    stdout_path = jobs.log_path(job.id, failure.attempt_id, "stdout")
    stderr_path = jobs.log_path(job.id, failure.attempt_id, "stderr")

    selection = select_evidence(capsule, stdout_path, stderr_path)

    assert selection.scanned_bytes > 5 * 1024 * 1024
    assert selection.selected_bytes <= 32 * 1024
    assert any(item.kind == "traceback" for item in selection.references)
    assert "ROOT_CAUSE" in selection.content
    for item in selection.references:
        path = stderr_path if item.source == "stderr" else stdout_path
        raw = path.read_bytes()[item.start : item.end]
        assert raw
