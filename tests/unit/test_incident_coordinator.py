from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.types import LlmResponse, ToolCallBlock
from cyan.config import CyanConfig
from cyan.training.incidents.coordinator import IncidentCoordinator
from cyan.training.incidents.evidence import _reference_was_observed
from cyan.training.incidents.smoke import SmokeExecution
from cyan.training.incidents.store import IncidentStore
from cyan.training.jobs import JobSpec, JobStore, JobSupervisor
from cyan.training.processes import read_process_identity


# 功能：验证已观察的工作区和日志范围允许引用其中的稳定子范围
# 设计：直接覆盖单行、行区间和 byte 区间，固定 evidence 的包含关系
def test_evidence_reference_accepts_observed_subranges() -> None:
    digest = "a" * 64
    observed = {
        f"train.py@sha256:{digest}#L15-L23",
        "stderr:job-1/attempt-1@bytes:100-200",
    }

    assert _reference_was_observed(f"train.py@sha256:{digest}#L23", observed)
    assert _reference_was_observed(f"train.py@sha256:{digest}#L17-L18", observed)
    assert _reference_was_observed("stderr:job-1/attempt-1@bytes:120-180", observed)


# 功能：验证 evidence 子范围不能扩大边界或替换来源身份
# 设计：改变行范围、文件 hash、日志 Job 和 byte 范围，锁定安全边界
def test_evidence_reference_rejects_expansion_and_identity_changes() -> None:
    digest = "a" * 64
    observed = {
        f"train.py@sha256:{digest}#L15-L23",
        "stderr:job-1/attempt-1@bytes:100-200",
    }

    assert not _reference_was_observed(f"train.py@sha256:{digest}#L14-L23", observed)
    assert not _reference_was_observed(f"train.py@sha256:{'b' * 64}#L17", observed)
    assert not _reference_was_observed("stderr:job-2/attempt-1@bytes:120-180", observed)
    assert not _reference_was_observed("stderr:job-1/attempt-1@bytes:120-201", observed)


# 从 Anthropic messages 尾部取得最近一次工具结果 JSON
def _latest_tool_json(messages: list[dict[str, Any]]) -> dict[str, Any]:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return cast(dict[str, Any], json.loads(str(block["content"])))
    raise AssertionError("tool result missing")


# 从 Anthropic messages 尾部取得最近一次工具结果文本
def _latest_tool_content(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(content):
            if isinstance(block, dict) and block.get("type") == "tool_result":
                return str(block["content"])
    raise AssertionError("tool result missing")


class _IncidentProvider:
    # 初始化一个按日志、源码、诊断和 proposal 顺序调用工具的确定性 provider
    def __init__(
        self,
        *,
        abstain: bool = False,
        force_propose_after_abstain: bool = False,
    ) -> None:
        self._run_id: str | None = None
        self._evidence: list[dict[str, str]] = []
        self._abstain = abstain
        self._force_propose_after_abstain = force_propose_after_abstain
        self.chat_calls = 0

    # 根据当前步骤返回受控 Incident 工具调用
    async def chat(
        self,
        messages: list[dict[str, Any]],
        tool_schemas: list[dict[str, Any]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del bus
        self.chat_calls += 1
        if run_id != self._run_id:
            self._run_id = run_id
            self._evidence = []
        tool_names = {str(item.get("name")) for item in tool_schemas}
        if step >= 4 and not self._abstain:
            assert tool_names == {"propose_patch"}
        else:
            assert tool_names == {
                "read_file",
                "list_dir",
                "search_text",
                "read_job_log",
                "submit_diagnosis",
                "propose_patch",
            }
        if step == 1:
            assert system is not None
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-log",
                        name="read_job_log",
                        input={
                            "stream": "stderr",
                            "mode": "tail",
                        },
                    )
                ],
            )
        if step == 2:
            log_result = _latest_tool_json(messages)
            reference = str(log_result["reference"])
            self._evidence = [
                {
                    "source": "stderr",
                    "reference": reference,
                    "description": "The real process emitted the crash marker.",
                }
            ]
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-source",
                        name="read_file",
                        input={"path": "train.py"},
                    )
                ],
            )
        if step == 3:
            source_result = _latest_tool_content(messages)
            source_reference = re.search(r"\[source ([^\]]+)\]", source_result)
            assert source_reference is not None
            self._evidence.append(
                {
                    "source": "workspace",
                    "reference": source_reference.group(1),
                    "description": "The causal exit is present in train.py.",
                }
            )
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="diagnose",
                        name="submit_diagnosis",
                        input={
                            "category": "runtime",
                            "summary": "train.py exits deliberately",
                            "root_cause": "The script prints boom and exits with status 2.",
                            "evidence": self._evidence,
                            "confidence": 1.0,
                            "causal_support": "inferred" if self._abstain else "direct",
                            "patch_recommended": not self._abstain,
                        },
                    )
                ],
            )
        if step == 4:
            if self._abstain and not self._force_propose_after_abstain:
                return LlmResponse(
                    stop_reason="end_turn",
                    text="Diagnosis submitted without a safe patch.",
                )
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="propose",
                        name="propose_patch",
                        input={
                            "path": "train.py",
                            "search": 'print("boom", file=sys.stderr)\nsys.exit(2)',
                            "replace": 'print("recovered")',
                            "evidence": self._evidence,
                        },
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="Diagnosis and patch submitted.")


# 运行 Git 命令并要求成功
def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


# 创建带真实 HEAD 和确定性失败脚本的临时 Git 工作区
def _workspace(root: Path, *, smoke_exit: int | None = None) -> Path:
    root.mkdir()
    (root / "train.py").write_text(
        'import sys\nprint("boom", file=sys.stderr)\nsys.exit(2)\n',
        encoding="utf-8",
    )
    if smoke_exit is not None:
        config = root / ".cyan" / "config.toml"
        config.parent.mkdir()
        config.write_text(
            "[incident.smoke]\n"
            f'argv = ["{sys.executable}", "-c", "raise SystemExit({smoke_exit})"]\n'
            "timeout_s = 5\n",
            encoding="utf-8",
        )
    _git(root, "init")
    _git(root, "config", "user.email", "cyan@example.invalid")
    _git(root, "config", "user.name", "cyan test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root


# 组装真实 JobSupervisor 与确定性只读 Agent 的完整 Incident 测试系统
def _system(
    root: Path, *, provider: Any | None = None
) -> tuple[IncidentCoordinator, JobSupervisor, JobStore]:
    jobs = JobStore(root.parent / "jobs")
    bus = EventBus()
    holder: dict[str, IncidentCoordinator] = {}

    # 将真实进程失败交给测试系统中的 Coordinator
    async def failure_callback(job: Any, attempt: Any, failure: Any) -> None:
        await holder["coordinator"].handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, failure_callback=failure_callback)
    coordinator = IncidentCoordinator(
        jobs,
        supervisor,
        bus,
        CyanConfig(),
        provider=provider or _IncidentProvider(),
    )
    holder["coordinator"] = coordinator
    return coordinator, supervisor, jobs


# 等待 Job 视图达到指定 Incident 状态
async def _wait_incident(
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


# 等待指定 Incident run 进入给定终态原因
async def _wait_run_reason(
    store: IncidentStore,
    incident_id: str,
    run_id: str,
    reason: str,
) -> None:
    for _ in range(300):
        run = store.read_run(incident_id, run_id)
        if run.status == "failed" and run.reason == reason:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"run did not fail with {reason}")


# 功能：验证真实非零进程自动唤醒只读 Agent，审批前零写入，批准后真实重跑并 resolved
# 设计：使用临时 Git、真实 Python 子进程和真实 git apply，覆盖 harness 的主闭环
async def test_failure_to_approval_and_real_retry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    original = (workspace / "train.py").read_text(encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")

        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert awaiting["proposal"] is not None
        incident = awaiting["incident"]
        proposal = awaiting["proposal"]
        assert (jobs.job_dir(job.id) / "incidents" / incident["id"] / "proposal.diff").exists()

        result = await coordinator.decide(
            str(incident["id"]),
            str(proposal["id"]),
            "approve",
            run_smoke=False,
        )
        assert result == "retry_running"
        await _wait_incident(coordinator, job.id, "resolved")
        assert jobs.read_job(job.id).status == "succeeded"
        assert len(jobs.read_job(job.id).attempt_ids) == 2
        assert "recovered" in (workspace / "train.py").read_text(encoding="utf-8")
        persisted = IncidentStore(jobs.job_dir(job.id) / "incidents").read_incident(
            str(incident["id"])
        )
        assert persisted.apply_receipt is not None
    finally:
        await coordinator.close()


# 功能：验证 abstain 诊断不生成 proposal 且 Incident 收敛为 unresolved
# 设计：真实子进程触发故障，检查硬门禁前后工作区和 artifact 均保持只读
async def test_abstaining_diagnosis_stays_unresolved_without_proposal(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "repo")
    original = (workspace / "train.py").read_text(encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace, provider=_IncidentProvider(abstain=True))
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        unresolved = await _wait_incident(coordinator, job.id, "unresolved")

        assert unresolved["diagnosis"]["causal_support"] == "inferred"
        assert unresolved["diagnosis"]["patch_recommended"] is False
        assert unresolved["proposal"] is None
        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        incident = unresolved["incident"]
        assert not (
            jobs.job_dir(job.id)
            / "incidents"
            / str(incident["id"])
            / "proposal.diff"
        ).exists()
    finally:
        await coordinator.close()


# 功能：验证 abstain 后误调用 propose_patch 仍被硬门禁拒绝
# 设计：让确定性 Agent 故意继续提案，检查不产生 diff 且 Incident 仍保持 unresolved
async def test_abstention_gate_rejects_followup_proposal(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(
        workspace,
        provider=_IncidentProvider(
            abstain=True,
            force_propose_after_abstain=True,
        ),
    )
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        unresolved = await _wait_incident(coordinator, job.id, "unresolved")

        assert unresolved["proposal"] is None
        incident = unresolved["incident"]
        assert not (
            jobs.job_dir(job.id)
            / "incidents"
            / str(incident["id"])
            / "proposal.diff"
        ).exists()
    finally:
        await coordinator.close()


# 功能：验证 smoke 通过后使用最新状态启动原命令重跑
# 设计：使用真实 Git、真实 smoke 子进程和真实训练重跑覆盖旧快照回归
async def test_smoke_success_retries_original_command(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo", smoke_exit=0)
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")

        result = await coordinator.decide(
            str(awaiting["incident"]["id"]),
            str(awaiting["proposal"]["id"]),
            "approve",
            run_smoke=True,
            smoke_config_fingerprint=str(awaiting["smoke_config_fingerprint"]),
        )

        assert result == "retry_running"
        resolved = await _wait_incident(coordinator, job.id, "resolved")
        assert resolved["smoke_result"]["status"] == "passed"
        assert resolved["incident"]["apply_receipt"] is not None
        assert len(jobs.read_job(job.id).attempt_ids) == 2
    finally:
        await coordinator.close()


# 功能：验证非 Git 工作区的 proposal 只可审阅且批准无副作用
# 设计：在真实非 Git 目录中完成诊断，检查视图门禁、FSM 和文件内容
async def test_non_git_proposal_is_review_only(tmp_path: Path) -> None:
    workspace = tmp_path / "plain-workspace"
    workspace.mkdir()
    original = 'import sys\nprint("boom", file=sys.stderr)\nsys.exit(2)\n'
    (workspace / "train.py").write_text(original, encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")

        assert awaiting["can_apply"] is False
        with pytest.raises(ValueError, match="review-only"):
            await coordinator.decide(
                str(awaiting["incident"]["id"]),
                str(awaiting["proposal"]["id"]),
                "approve",
                run_smoke=False,
            )

        current = await coordinator.job_view(job.id)
        assert current["incident"]["status"] == "awaiting_approval"
        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert jobs.read_job(job.id).attempt_ids == ["attempt-0001"]
    finally:
        await coordinator.close()


# 功能：验证 smoke 失败会先回滚 proposal，不启动完整训练重跑
# 设计：使用真实配置和真实 smoke 子进程，检查工作区恢复及 incident 重新诊断
async def test_smoke_failure_rolls_back_before_retry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo", smoke_exit=7)
    original = (workspace / "train.py").read_text(encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident = awaiting["incident"]
        proposal = awaiting["proposal"]
        await coordinator.decide(
            str(incident["id"]),
            str(proposal["id"]),
            "approve",
            run_smoke=True,
            smoke_config_fingerprint=str(awaiting["smoke_config_fingerprint"]),
        )
        second = await _wait_incident(coordinator, job.id, "awaiting_approval")

        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert jobs.read_job(job.id).attempt_ids == ["attempt-0001"]
        assert second["smoke_result"]["status"] == "failed"
    finally:
        await coordinator.close()


# 功能：验证 daemon 恢复会终止具备身份校验的遗留 smoke 进程并收敛为 unresolved
# 设计：用真实独立进程写入 incident 快照，恢复流程必须先确认进程组归属
async def test_recover_terminates_orphaned_smoke_process(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    process: asyncio.subprocess.Process | None = None
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        incident = store.read_incident(str(awaiting["incident"]["id"]))
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import time; time.sleep(30)",
            start_new_session=True,
        )
        process_identity = await read_process_identity(process.pid)
        assert process_identity is not None
        store.write_smoke_execution(
            incident.id,
            SmokeExecution(
                status="running",
                pid=process.pid,
                process_identity=process_identity,
                started_at=datetime.now(UTC),
            ),
        )
        persisted = store.read_incident(incident.id)
        persisted.status = "smoke_running"
        store.write_incident(persisted)

        await coordinator.recover()
        for _ in range(100):
            if await read_process_identity(process.pid) is None:
                break
            await asyncio.sleep(0.05)
        recovered = store.read_incident(incident.id)
        assert recovered.status == "unresolved"
        assert recovered.smoke_execution is not None
        assert recovered.smoke_execution.status == "interrupted"
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        await coordinator.close()


# 功能：验证命令无法启动时只保留 Job 失败，不自动创建 Incident
# 设计：使用真实不存在的 executable，确保 Incident 只响应训练进程非零退出
async def test_launch_error_does_not_open_incident(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=["definitely-not-a-real-cyan-command"], workspace_root=workspace)
        )
        await coordinator.recover()
        assert jobs.read_failure(job.id, "attempt-0001").kind == "launch_error"
        assert (await coordinator.job_view(job.id))["incident"] is None
    finally:
        await coordinator.close()


# 功能：验证追问会创建新的 Incident-owned run，并携带上一轮有界结果
# 设计：在真实首轮 proposal 上调用显式 follow_up，检查新 run.json 和清理后的快照
async def test_follow_up_creates_bounded_run(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_id = str(awaiting["incident"]["id"])
        original_spawn = coordinator._spawn

        # 模拟后台启动前 daemon 崩溃，只验证 prepared run 和状态快照
        def discard_background(coroutine: Any, name: str) -> None:
            del name
            coroutine.close()

        coordinator._spawn = discard_background  # type: ignore[method-assign]
        try:
            run_id = await coordinator.follow_up(incident_id, "Re-check the evidence.")
        finally:
            coordinator._spawn = original_spawn  # type: ignore[method-assign]
        run_path = jobs.job_dir(job.id) / "incidents" / incident_id / "runs" / run_id
        assert (run_path / "run.json").exists()
        run = IncidentStore(jobs.job_dir(job.id) / "incidents").read_run(incident_id, run_id)
        assert run.previous_outcome_summary is not None
        assert "diagnosis" in run.previous_outcome_summary
        current = IncidentStore(jobs.job_dir(job.id) / "incidents").read_incident(incident_id)
        assert current.status == "diagnosing"
        assert current.active_run_id == run_id
    finally:
        await coordinator.close()


# 功能：验证已持久化 Capsule 后日志被替换会在 LLM 前拒绝运行
# 设计：首轮完成后修改真实 stderr，比较 provider 调用数并检查 log_changed
async def test_follow_up_rejects_changed_failure_log_before_llm(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    provider = _IncidentProvider()
    coordinator, supervisor, jobs = _system(workspace, provider=provider)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_id = str(awaiting["incident"]["id"])
        attempt_id = str(awaiting["incident"]["attempt_id"])
        calls_before = provider.chat_calls
        jobs.log_path(job.id, attempt_id, "stderr").write_text(
            "RuntimeError: replaced log\n", encoding="utf-8"
        )

        run_id = await coordinator.follow_up(incident_id, "Re-check this failure.")
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        await _wait_run_reason(store, incident_id, run_id, "log_changed")

        assert provider.chat_calls == calls_before
    finally:
        await coordinator.close()


# 功能：验证缺失持久化 Failure Capsule 时不会从当前日志重建
# 设计：清空 failure.json 中的 capsule 后发起追问，断言稳定失败原因且 LLM 未调用
async def test_follow_up_rejects_missing_failure_capsule(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    provider = _IncidentProvider()
    coordinator, supervisor, jobs = _system(workspace, provider=provider)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_id = str(awaiting["incident"]["id"])
        attempt_id = str(awaiting["incident"]["attempt_id"])
        failure = jobs.read_failure(job.id, attempt_id)
        jobs.write_failure(failure.model_copy(update={"capsule": None}))
        calls_before = provider.chat_calls

        run_id = await coordinator.follow_up(incident_id, "Re-check this failure.")
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        await _wait_run_reason(
            store, incident_id, run_id, "failure_capsule_unavailable"
        )

        assert provider.chat_calls == calls_before
    finally:
        await coordinator.close()


# 功能：验证 daemon 恢复会复用尚未启动的追问 run
# 设计：阻止两次后台调度，检查恢复不新建 run 且保留原始 instruction
async def test_recover_resumes_prepared_follow_up(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_id = str(awaiting["incident"]["id"])
        original_spawn = coordinator._spawn

        def discard_background(coroutine: Any, name: str) -> None:
            del name
            coroutine.close()

        coordinator._spawn = discard_background  # type: ignore[method-assign]
        try:
            run_id = await coordinator.follow_up(incident_id, "Keep the exact evidence range.")
            store = IncidentStore(jobs.job_dir(job.id) / "incidents")
            run_before = store.read_run(incident_id, run_id)
            await coordinator.recover()
        finally:
            coordinator._spawn = original_spawn  # type: ignore[method-assign]

        run_after = store.read_run(incident_id, run_id)
        current = store.read_incident(incident_id)
        assert run_after.run_id == run_before.run_id
        assert run_after.instruction == "Keep the exact evidence range."
        assert current.active_run_id == run_id
    finally:
        await coordinator.close()
