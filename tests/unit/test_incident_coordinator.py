from __future__ import annotations

import asyncio
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from cyan.core.config import CyanConfig
from cyan.core.events.bus import EventBus
from cyan.core.incidents.coordinator import (
    IncidentCoordinator,
    _reference_was_observed,
)
from cyan.core.incidents.smoke import SmokeExecution
from cyan.core.incidents.store import IncidentStore
from cyan.core.jobs import JobSpec, JobStore, JobSupervisor
from cyan.core.llm.types import LlmResponse, ToolCallBlock
from cyan.core.processes import read_process_identity
from cyan.core.runner import AgentRunner
from cyan.core.session import SessionManager, SessionStore


# 功能：验证已观察的工作区和日志范围允许引用其中的稳定子范围
# 设计：直接覆盖单行、行区间和 byte 区间，排除对 Agent 引用字符串完全相等的依赖
def test_evidence_reference_accepts_observed_subranges() -> None:
    digest = "a" * 64
    observed = {
        f"train.py@sha256:{digest}#L15-L23",
        "stderr:job-1/attempt-1@bytes:100-200",
    }

    assert _reference_was_observed(f"train.py@sha256:{digest}#L23", observed)
    assert _reference_was_observed(f"train.py@sha256:{digest}#L17-L18", observed)
    assert _reference_was_observed(
        "stderr:job-1/attempt-1@bytes:120-180",
        observed,
    )


# 功能：验证 evidence 子范围不能扩大边界或替换来源身份
# 设计：逐一改变行范围、文件 hash、日志 Job 和 byte 范围，锁定包含关系的安全边界
def test_evidence_reference_rejects_expansion_and_identity_changes() -> None:
    digest = "a" * 64
    observed = {
        f"train.py@sha256:{digest}#L15-L23",
        "stderr:job-1/attempt-1@bytes:100-200",
    }

    assert not _reference_was_observed(f"train.py@sha256:{digest}#L14-L23", observed)
    assert not _reference_was_observed(
        f"train.py@sha256:{'b' * 64}#L17",
        observed,
    )
    assert not _reference_was_observed(
        "stderr:job-2/attempt-1@bytes:120-180",
        observed,
    )
    assert not _reference_was_observed(
        "stderr:job-1/attempt-1@bytes:120-201",
        observed,
    )


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
    # 初始化一个按日志、诊断、proposal 顺序调用工具的确定性 provider
    def __init__(self) -> None:
        self._step = 0
        self._evidence: list[dict[str, str]] = []

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
        self._step += 1
        tool_names = {str(item.get("name")) for item in tool_schemas}
        assert tool_names == {
            "read_file",
            "list_dir",
            "search_text",
            "read_job_log",
            "submit_diagnosis",
            "propose_patch",
        }
        if self._step == 1:
            assert system is not None
            job_id = re.search(r'"job_id": "([^"]+)"', system)
            attempt_id = re.search(r'"attempt_id": "([^"]+)"', system)
            assert job_id is not None and attempt_id is not None
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-log",
                        name="read_job_log",
                        input={
                            "job_id": job_id.group(1),
                            "attempt_id": attempt_id.group(1),
                            "stream": "stderr",
                            "mode": "tail",
                        },
                    )
                ],
            )
        if self._step == 2:
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
                        input={
                            "path": "train.py",
                        },
                    )
                ],
            )
        if self._step == 3:
            source_result = _latest_tool_content(messages)
            source_reference = re.search(r"\[source ([^\]]+)\]", source_result)
            assert source_reference is not None
            workspace_evidence = {
                "source": "workspace",
                "reference": source_reference.group(1),
                "description": "The causal exit is present in train.py.",
            }
            self._evidence.append(workspace_evidence)
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
                        },
                    )
                ],
            )
        if self._step == 4:
            diagnosis = _latest_tool_json(messages)
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="propose",
                        name="propose_patch",
                        input={
                            "path": "train.py",
                            "search": (
                                'print("boom", file=sys.stderr)\n'
                                "sys.exit(2)"
                            ),
                            "replace": 'print("recovered")',
                            "diagnosis_id": str(diagnosis["id"]),
                            "evidence": self._evidence,
                        },
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="Diagnosis and patch submitted.")


# 运行 Git 命令并要求成功
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
    )


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
    root: Path,
) -> tuple[IncidentCoordinator, JobSupervisor, JobStore]:
    jobs = JobStore(root.parent / "jobs")
    sessions_store = SessionStore(root.parent / "sessions")
    bus = EventBus()
    holder: dict[str, IncidentCoordinator] = {}

    # 将真实进程失败交给测试系统中的 Coordinator
    async def failure_callback(job: Any, attempt: Any, failure: Any) -> None:
        await holder["coordinator"].handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, failure_callback=failure_callback)
    config = CyanConfig()

    # 每个 session run 使用一个新的确定性 provider，并动态读取 Incident profile
    def runner_factory() -> AgentRunner:
        return AgentRunner(
            config,
            bus=bus,
            provider=_IncidentProvider(),
            profile_factory=holder["coordinator"].profile_for_session,
        )

    sessions = SessionManager(
        sessions_store,
        runner_factory=runner_factory,
        bus=bus,
    )
    coordinator = IncidentCoordinator(jobs, sessions, supervisor, bus)
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


# 功能：验证真实非零进程自动唤醒只读 Agent，审批前零写入，批准后真实重跑并 resolved
# 设计：使用临时 Git、真实 Python 子进程和真实 git apply，provider 只替代外部 LLM，完整覆盖 harness 边界
async def test_failure_to_approval_and_real_retry(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    original = (workspace / "train.py").read_text(encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace)
    retry_starts: list[Any] = []

    # 收集批准后重跑的实时 started 事件，验证它和磁盘 seq 使用同一契约
    async def collect_retry_start(event: Any) -> None:
        if getattr(event, "type", "") == "job.started":
            retry_starts.append(event)

    coordinator._bus.subscribe(collect_retry_start)
    try:
        job = await supervisor.start(JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace))
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")

        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert awaiting["diagnosis"]["evidence"][0]["reference"].startswith("stderr:")
        assert awaiting["proposal"] is not None
        assert awaiting["can_apply"] is True

        incident = awaiting["incident"]
        proposal = awaiting["proposal"]
        result = await coordinator.decide(
            str(incident["id"]),
            str(proposal["id"]),
            "approve",
            run_smoke=False,
        )
        assert result == "retry_running"
        await _wait_incident(coordinator, job.id, "resolved")

        final_job = jobs.read_job(job.id)
        assert final_job.status == "succeeded"
        assert len(final_job.attempt_ids) == 2
        persisted_start = next(
            event
            for event in jobs.read_events(job.id)
            if event.type == "job.started" and event.attempt_id == "attempt-0002"
        )
        assert [(event.attempt_id, event.seq) for event in retry_starts] == [
            ("attempt-0002", persisted_start.seq)
        ]
        assert "recovered" in (workspace / "train.py").read_text(encoding="utf-8")
    finally:
        await coordinator.close()


# 功能：验证 daemon 重启时已成功的完整重跑会把 retry_running Incident 收敛为 resolved
# 设计：先走通真实修复，再把持久化 Incident 模拟回崩溃前状态并调用 recover，检查不被误报为 unresolved
async def test_recover_successful_retry_as_resolved(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_data = awaiting["incident"]
        proposal = awaiting["proposal"]

        await coordinator.decide(
            str(incident_data["id"]),
            str(proposal["id"]),
            "approve",
            run_smoke=False,
        )
        await _wait_incident(coordinator, job.id, "resolved")

        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        incident = store.read_incident(str(incident_data["id"]))
        incident.status = "retry_running"
        store.write_incident(incident)
        await coordinator.recover()

        recovered = store.read_incident(incident.id)
        assert jobs.read_job(job.id).status == "succeeded"
        assert recovered.status == "resolved"
    finally:
        await coordinator.close()


# 功能：验证 retry_running 已落盘但新 Attempt 尚未创建的崩溃窗口会恢复为 unresolved
# 设计：保持 Job 指向原失败 Attempt，仅提前写 Incident 状态并 recover，避免状态永久卡住
async def test_recover_retry_before_launch_as_unresolved(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        incident = store.read_incident(str(awaiting["incident"]["id"]))
        incident.status = "retry_running"
        store.write_incident(incident)

        await coordinator.recover()

        assert jobs.read_job(job.id).current_attempt_id == incident.attempt_id
        assert store.read_incident(incident.id).status == "unresolved"
    finally:
        await coordinator.close()


# 功能：验证 follow-up 崩溃窗口不会在重启后重新呈现上一轮 diagnosis 与 proposal
# 设计：阻止后台任务启动并检查制品已同步清除，再 recover 产出全新 proposal ID
async def test_follow_up_clears_old_artifacts_before_background_run(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_data = awaiting["incident"]
        old_proposal_id = str(awaiting["proposal"]["id"])
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        original_spawn = coordinator._spawn

        # 模拟 daemon 在同步状态转移完成后、后台 Agent 真正启动前崩溃
        def discard_background(coroutine: Any, name: str) -> None:
            coroutine.close()

        coordinator._spawn = discard_background  # type: ignore[method-assign]
        try:
            await coordinator.follow_up(
                str(incident_data["session_id"]),
                "Re-check the evidence.",
                "run-follow-up",
            )
        finally:
            coordinator._spawn = original_spawn  # type: ignore[method-assign]

        incident = store.read_incident(str(incident_data["id"]))
        assert incident.status == "diagnosing"
        assert not (store.incident_dir(incident.id) / "diagnosis.json").exists()
        assert not (store.incident_dir(incident.id) / "proposal.json").exists()

        await coordinator.recover()
        recovered = await _wait_incident(coordinator, job.id, "awaiting_approval")
        assert recovered["proposal"]["id"] != old_proposal_id
    finally:
        await coordinator.close()


# 功能：验证命令根本无法启动时只保留 TUI 可见失败，不自动创建训练 Incident
# 设计：真实执行不存在的 argv，并在显式 recover 后再次断言没有 LLM session 或 proposal
async def test_launch_error_does_not_open_incident(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(
                argv=["definitely-not-a-real-cyan-command"],
                workspace_root=workspace,
            )
        )
        await coordinator.recover()
        view = await coordinator.job_view(job.id)

        assert jobs.read_failure(job.id, "attempt-0001").kind == "launch_error"
        assert view["incident"] is None
        incident_root = jobs.job_dir(job.id) / "incidents"
        assert not list(incident_root.glob("*/incident.json"))
    finally:
        await coordinator.close()


# 功能：验证 daemon 重启会先终止有持久化身份的遗留 smoke 进程，再开放 Incident 追问
# 设计：用真实独立进程模拟硬崩溃遗留 verifier，写入 running artifact 后 recover 并检查进程和状态
async def test_recover_terminates_orphaned_smoke_process(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    coordinator, supervisor, jobs = _system(workspace)
    process: asyncio.subprocess.Process | None = None
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        incident_id = str(awaiting["incident"]["id"])
        store = IncidentStore(jobs.job_dir(job.id) / "incidents")
        incident = store.read_incident(incident_id)

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
        incident.status = "smoke_running"
        store.write_incident(incident)

        await coordinator.recover()
        await asyncio.wait_for(process.wait(), timeout=5)

        assert store.read_incident(incident.id).status == "unresolved"
        assert store.read_smoke_execution(incident.id).status == "interrupted"
        assert store.read_smoke_result(incident.id).status == "interrupted"
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        await coordinator.close()


# 功能：验证可选 smoke 非零退出时不启动完整重跑，且仅在哈希一致时自动反向应用补丁
# 设计：使用项目真实 .cyan 配置和真实 smoke 子进程，等待同一 session 重新产出 proposal 后检查原文件与 attempt 数
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
        assert second["smoke"]["status"] == "failed"
        assert second["smoke"]["returncode"] == 7
        assert second["incident"]["session_id"] == incident["session_id"]
    finally:
        await coordinator.close()


# 功能：验证审批前 smoke 配置变化会 fail-closed 且对工作区零写入
# 设计：展示旧指纹后替换真实 TOML，再批准 smoke，断言 stale、零 apply 和零重跑
async def test_smoke_config_change_before_approval_is_stale(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo", smoke_exit=0)
    original = (workspace / "train.py").read_text(encoding="utf-8")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[sys.executable, "train.py"], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        fingerprint = str(awaiting["smoke_config_fingerprint"])
        config = workspace / ".cyan" / "config.toml"
        config.write_text(
            '[incident.smoke]\nargv = ["/usr/bin/true"]\ntimeout_s = 5\n',
            encoding="utf-8",
        )

        result = await coordinator.decide(
            str(awaiting["incident"]["id"]),
            str(awaiting["proposal"]["id"]),
            "approve",
            run_smoke=True,
            smoke_config_fingerprint=fingerprint,
        )

        assert result == "stale"
        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert jobs.read_job(job.id).attempt_ids == ["attempt-0001"]
    finally:
        await coordinator.close()


# 功能：验证批准后的原命令无法再次启动时 Incident 不会悬挂在 retry_running
# 设计：首轮使用真实可执行 launcher 失败，审批前删除 launcher，断言第二次 launch_error 收敛为 unresolved
async def test_retry_launch_error_becomes_unresolved(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path / "repo")
    launcher = workspace / "run-training"
    launcher.write_text(
        f"#!{sys.executable}\n"
        "import runpy\n"
        "runpy.run_path('train.py', run_name='__main__')\n",
        encoding="utf-8",
    )
    launcher.chmod(0o755)
    _git(workspace, "add", "run-training")
    _git(workspace, "commit", "-m", "add launcher")
    coordinator, supervisor, jobs = _system(workspace)
    try:
        job = await supervisor.start(
            JobSpec(argv=[str(launcher)], workspace_root=workspace)
        )
        await supervisor.wait(job.id)
        awaiting = await _wait_incident(coordinator, job.id, "awaiting_approval")
        launcher.unlink()

        result = await coordinator.decide(
            str(awaiting["incident"]["id"]),
            str(awaiting["proposal"]["id"]),
            "approve",
            run_smoke=False,
        )
        unresolved = await _wait_incident(coordinator, job.id, "unresolved")

        assert result == "retry_running"
        assert unresolved["incident"]["status"] == "unresolved"
        assert jobs.read_job(job.id).status == "failed"
        assert jobs.read_job(job.id).attempt_ids == ["attempt-0001", "attempt-0002"]
        assert jobs.read_failure(job.id, "attempt-0002").kind == "launch_error"
    finally:
        await coordinator.close()
