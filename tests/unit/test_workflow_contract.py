from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from cyan.core.incidents.models import Diagnosis
from cyan.core.jobs import (
    AttemptRecord,
    FailureRecord,
    JobSpec,
    JobStore,
    JobSupervisor,
    WorkflowArtifact,
    WorkflowCheck,
    WorkflowContract,
    artifact_is_fresh,
    load_workflow_contract,
    snapshot_artifact,
    workflow_contract_fingerprint,
)
from cyan.core.jobs.launch import launch_fingerprint, parse_training_command


# 将给定 TOML 写入测试工作区的标准 Contract 路径
def _write_contract(root: Path, content: str) -> Path:
    path = root / ".cyan" / "workflow.toml"
    path.parent.mkdir(exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


# 等待真实子进程退出，避免只验证直接 check 进程而遗漏其进程组成员
async def _wait_pid_gone(pid: int, timeout_s: float = 3.0) -> None:
    deadline = asyncio.get_running_loop().time() + timeout_s
    while asyncio.get_running_loop().time() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        await asyncio.sleep(0.05)
    pytest.fail(f"process remained alive after check timeout: {pid}")


# 功能：验证旧持久化模型缺少 Workflow 与 recovery 字段时仍可读取
# 设计：直接用升级前最小 JSON 形状解析 JobSpec、Attempt、Failure 和 Diagnosis
def test_legacy_persisted_models_keep_defaults(tmp_path: Path) -> None:
    spec = JobSpec.model_validate(
        {"argv": ["true"], "workspace_root": str(tmp_path), "env": {}}
    )
    attempt = AttemptRecord.model_validate(
        {
            "id": "attempt-0001",
            "job_id": "job-legacy",
            "status": "failed",
            "started_at": "2026-01-01T00:00:00+00:00",
        }
    )
    failure = FailureRecord.model_validate(
        {
            "job_id": "job-legacy",
            "attempt_id": "attempt-0001",
            "occurred_at": "2026-01-01T00:00:01+00:00",
            "kind": "process_exit",
            "message": "process exited with status 1",
        }
    )
    diagnosis = Diagnosis.model_validate(
        {
            "id": "diagnosis-legacy",
            "incident_id": "incident-legacy",
            "category": "runtime",
            "summary": "legacy diagnosis",
            "root_cause": "legacy root cause",
            "evidence": [
                {
                    "source": "stderr",
                    "reference": "stderr:job-legacy/attempt-0001@bytes:0-1",
                    "description": "legacy evidence",
                }
            ],
            "confidence": 0.8,
            "created_at": "2026-01-01T00:00:02+00:00",
        }
    )

    assert spec.workflow_contract is None
    assert attempt.phase == "main"
    assert attempt.artifact_baseline == []
    assert failure.phase is None
    assert diagnosis.recovery is None


# 功能：验证 Contract 规范化路径、去重、未知字段和 freshness 约束
# 设计：从真实 TOML 读取合法样例，再对三个非法模型输入检查严格拒绝
def test_contract_parsing_and_strict_validation(tmp_path: Path) -> None:
    _write_contract(
        tmp_path,
        "version = 1\n"
        "[[artifacts]]\n"
        'path = "data//train.csv"\n'
        'role = "input"\n'
        "required = true\n"
        "min_bytes = 1\n"
        "[[checks]]\n"
        'id = "schema"\n'
        'phase = "preflight"\n'
        f'argv = ["{sys.executable}", "-c", "pass"]\n',
    )

    contract = load_workflow_contract(tmp_path)

    assert contract is not None
    assert contract.artifacts[0].path == "data/train.csv"
    with pytest.raises(ValidationError, match="Extra inputs"):
        WorkflowContract.model_validate({"version": 1, "kind": "training"})
    with pytest.raises(ValidationError, match="must be unique"):
        WorkflowContract(
            checks=[
                WorkflowCheck(id="same", phase="preflight", argv=["true"]),
                WorkflowCheck(id="same", phase="postflight", argv=["true"]),
            ]
        )
    with pytest.raises(ValidationError, match="required output"):
        WorkflowArtifact(path="model.pt", role="output", required=False, fresh=True)


# 功能：验证 Contract 文件本身拒绝 symlink，artifact 仅允许 workspace 内 symlink
# 设计：分别创建内部与外部真实目标，直接覆盖解析后的 realpath 边界
def test_contract_and_artifact_symlink_boundaries(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = tmp_path / "contract.toml"
    target.write_text("version = 1\n", encoding="utf-8")
    contract_path = workspace / ".cyan" / "workflow.toml"
    contract_path.parent.mkdir()
    contract_path.symlink_to(target)
    with pytest.raises(ValueError, match="may not be a symlink"):
        load_workflow_contract(workspace)

    contract_path.unlink()
    contract_path.write_text("version = 1\n", encoding="utf-8")
    internal = workspace / "real-data"
    internal.mkdir()
    (internal / "train.csv").write_text("x\n", encoding="utf-8")
    (workspace / "data").symlink_to(internal, target_is_directory=True)
    artifact = WorkflowArtifact(path="data/train.csv", role="input")
    assert snapshot_artifact(workspace, artifact).exists

    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "secret.csv").write_text("x\n", encoding="utf-8")
    (workspace / "external").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes workspace"):
        snapshot_artifact(
            workspace,
            WorkflowArtifact(path="external/secret.csv", role="input"),
        )


# 功能：验证 freshness 只比较轻量 metadata 并识别新建、替换与未变化
# 设计：在同一路径依次读取 baseline、原样、更新和替换后的 stat
def test_artifact_freshness_metadata(tmp_path: Path) -> None:
    artifact = WorkflowArtifact(
        path="model.pt",
        role="output",
        required=True,
        fresh=True,
    )
    before = snapshot_artifact(tmp_path, artifact)
    (tmp_path / "model.pt").write_bytes(b"first")
    created = snapshot_artifact(tmp_path, artifact)
    assert artifact_is_fresh(before, created)
    assert not artifact_is_fresh(created, snapshot_artifact(tmp_path, artifact))
    replacement = tmp_path / "replacement"
    replacement.write_bytes(b"second")
    os.replace(replacement, tmp_path / "model.pt")
    assert artifact_is_fresh(created, snapshot_artifact(tmp_path, artifact))


# 功能：验证 Launch fingerprint 纳入规范化 Contract snapshot
# 设计：固定 argv/cwd/env，仅改变一个 artifact rule 并比较摘要
def test_launch_fingerprint_includes_workflow_contract(tmp_path: Path) -> None:
    environment = {"PATH": os.environ["PATH"]}
    launch = parse_training_command(f"{sys.executable} train.py", tmp_path, environment)
    first = WorkflowContract(artifacts=[WorkflowArtifact(path="input.csv", role="input")])
    second = WorkflowContract(
        artifacts=[WorkflowArtifact(path="input.csv", role="input", min_bytes=2)]
    )

    assert workflow_contract_fingerprint(first) != workflow_contract_fingerprint(second)
    assert launch_fingerprint(launch, tmp_path, environment, first) != launch_fingerprint(
        launch,
        tmp_path,
        environment,
        second,
    )


# 功能：验证 Supervisor 在一个 Attempt 中顺序执行 Contract 生命周期
# 设计：使用真实 pre/post checks、main 子进程、frozen cwd/env、日志与 phase 回调
async def test_supervisor_runs_contract_lifecycle(tmp_path: Path) -> None:
    (tmp_path / "input.txt").write_text("payload", encoding="utf-8")
    phases: list[tuple[str, str | None]] = []

    # 收集实时 phase 增量，同时最终状态仍从 Attempt snapshot 读取
    def on_phase(job: object, attempt: object) -> None:
        phases.append((str(getattr(attempt, "phase")), getattr(attempt, "check_id")))

    contract = WorkflowContract(
        artifacts=[
            WorkflowArtifact(path="input.txt", role="input", min_bytes=1),
            WorkflowArtifact(path="output.txt", role="output", min_bytes=1, fresh=True),
        ],
        checks=[
            WorkflowCheck(
                id="pre",
                phase="preflight",
                argv=[
                    sys.executable,
                    "-c",
                    "import os; from pathlib import Path; "
                    "assert os.environ['WORKFLOW_VALUE'] == 'frozen'; "
                    "assert Path('input.txt').exists()",
                ],
            ),
            WorkflowCheck(
                id="post",
                phase="postflight",
                argv=[
                    sys.executable,
                    "-c",
                    "from pathlib import Path; assert Path('output.txt').read_text() == 'payload'",
                ],
            ),
        ],
    )
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store, phase_callback=on_phase)
    job = await supervisor.start(
        JobSpec(
            argv=[
                sys.executable,
                "-c",
                "from pathlib import Path; Path('output.txt').write_text(Path('input.txt').read_text())",
            ],
            workspace_root=tmp_path,
            env={"WORKFLOW_VALUE": "frozen"},
            workflow_contract=contract,
        )
    )

    finished = await supervisor.wait(job.id)
    attempt = store.read_attempt(job.id, "attempt-0001")

    assert finished.status == "succeeded"
    assert attempt.phase == "postflight"
    assert attempt.check_id == "post"
    assert [(phase, check) for phase, check in phases if check is not None] == [
        ("preflight", "pre"),
        ("postflight", "post"),
    ]
    assert "[cyan phase=main]" in store.read_log(job.id, attempt.id, "stdout").text


# 功能：验证旧 output 未更新时 fresh=true 产生 postflight Contract failure
# 设计：main 真实成功但不写产物，检查失败 taxonomy、phase、artifact 和回调
async def test_supervisor_rejects_stale_output(tmp_path: Path) -> None:
    (tmp_path / "model.pt").write_bytes(b"old")
    failures: list[object] = []
    contract = WorkflowContract(
        artifacts=[WorkflowArtifact(path="model.pt", role="output", min_bytes=1, fresh=True)]
    )
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(
        store,
        failure_callback=lambda job, attempt, failure: failures.append(failure),
    )
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "pass"],
            workspace_root=tmp_path,
            workflow_contract=contract,
        )
    )

    finished = await supervisor.wait(job.id)
    failure = store.read_failure(job.id, "attempt-0001")

    assert finished.status == "failed"
    assert failure.kind == "contract_violation"
    assert failure.phase == "postflight"
    assert failure.artifact_path == "model.pt"
    assert failure.violation_rule == "fresh"
    assert failures


# 功能：验证 custom check 超时会终止其完整进程组，而不遗留孙进程
# 设计：check 启动真实长时子进程并写出 PID，超时后同时验证 taxonomy 与 OS 存活状态
async def test_supervisor_check_timeout_terminates_process_group(tmp_path: Path) -> None:
    child_pid_path = tmp_path / "check-child.pid"
    check_code = (
        "import subprocess, sys, time; from pathlib import Path; "
        "child=subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)']); "
        "Path('check-child.pid').write_text(str(child.pid)); "
        "time.sleep(30)"
    )
    contract = WorkflowContract(
        checks=[
            WorkflowCheck(
                id="timeout",
                phase="preflight",
                argv=[sys.executable, "-c", check_code],
                timeout_s=0.2,
            )
        ]
    )
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store)
    job = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "pass"],
            workspace_root=tmp_path,
            workflow_contract=contract,
        )
    )

    finished = await supervisor.wait(job.id)
    failure = store.read_failure(job.id, "attempt-0001")
    child_pid = int(child_pid_path.read_text(encoding="utf-8"))

    assert finished.status == "failed"
    assert failure.kind == "contract_violation"
    assert failure.check_id == "timeout"
    assert failure.violation_rule == "check_timeout"
    await _wait_pid_gone(child_pid)
