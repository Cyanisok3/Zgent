from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyan.agent.events.bus import EventBus
from cyan.config import get_config
from cyan.training.incidents.context import INCIDENT_PROMPT_VERSION
from cyan.training.incidents.coordinator import IncidentCoordinator
from cyan.training.jobs.models import JobSpec
from cyan.training.jobs.store import JobStore
from cyan.training.jobs.supervisor import JobSupervisor

from cyan_bench.cases import LoadedCase
from cyan_bench.execution import command_for_workspace, environment_for_workspace
from cyan_bench.models import IncidentBenchmarkArtifact
from cyan_bench.paths import BenchmarkPaths

_INCIDENT_TERMINAL = {"resolved", "rejected", "stale", "unresolved", "rollback_blocked"}


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


# 合并同一日志流上的 evidence intervals 并计算唯一字节数
def _union_bytes(intervals: list[tuple[str, int, int]]) -> int:
    total = 0
    for source in ("stdout", "stderr"):
        ranges = sorted((start, end) for item, start, end in intervals if item == source)
        merged: list[tuple[int, int]] = []
        for start, end in ranges:
            if not merged or start > merged[-1][1]:
                merged.append((start, end))
            else:
                merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        total += sum(end - start for start, end in merged)
    return total


# 从 Incident run 事件中汇总 usage 和工具调用
def _event_metrics(events_path: Path) -> tuple[int, int, int]:
    input_tokens = 0
    output_tokens = 0
    tool_calls = 0
    if not events_path.is_file():
        return input_tokens, output_tokens, tool_calls
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except ValueError:
            continue
        if event.get("type") == "llm.usage":
            input_tokens += int(event.get("input_tokens", 0))
            output_tokens += int(event.get("output_tokens", 0))
        elif event.get("type") == "tool.call_started":
            tool_calls += 1
    return input_tokens, output_tokens, tool_calls


# 等待 Incident 到达审批或终态并返回最新视图
async def _wait_incident(
    coordinator: IncidentCoordinator,
    job_id: str,
    timeout_s: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        view = await coordinator.job_view(job_id)
        incident = view.get("incident")
        if isinstance(incident, dict):
            status = incident.get("status")
            if status == "awaiting_approval" or status in _INCIDENT_TERMINAL:
                return view
        await asyncio.sleep(0.1)
    raise TimeoutError("incident did not reach approval or terminal state")


# 等待审批后的原命令重跑收敛或产生新的审批
async def _wait_verification(
    coordinator: IncidentCoordinator,
    job_id: str,
    timeout_s: int,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        view = await coordinator.job_view(job_id)
        incident = view.get("incident")
        if isinstance(incident, dict):
            status = incident.get("status")
            if status == "awaiting_approval" or status in _INCIDENT_TERMINAL:
                return view
        await asyncio.sleep(0.1)
    raise TimeoutError("incident verification did not reach terminal state")


# 从当前 Incident artifacts 计算 Capsule、Selector、上下文和调用指标
def _incident_metrics(
    jobs: JobStore,
    view: dict[str, Any],
) -> tuple[int, int, int, int, int, int, int]:
    incident = view.get("incident")
    if not isinstance(incident, dict):
        return 0, 0, 0, 0, 0, 0, 0
    job_id = str(incident["job_id"])
    attempt_id = str(incident["attempt_id"])
    failure = jobs.read_failure(job_id, attempt_id)
    capsule = failure.capsule if isinstance(failure.capsule, dict) else {}
    intervals: list[tuple[str, int, int]] = []
    capsule_bytes = 0
    for source in ("stdout", "stderr"):
        snapshot = capsule.get(source)
        if isinstance(snapshot, dict):
            start = int(snapshot.get("included_start", 0))
            end = int(snapshot.get("included_end", 0))
            if end > start:
                intervals.append((source, start, end))
                capsule_bytes += end - start
    active_run_id = incident.get("active_run_id")
    if not isinstance(active_run_id, str):
        return capsule_bytes, 0, _union_bytes(intervals), 0, 0, 0, 0
    run_dir = jobs.job_dir(job_id) / "incidents" / str(incident["id"]) / "runs" / active_run_id
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    for reference in run.get("selected_evidence", []):
        raw = str(reference.get("reference", ""))
        if "@bytes:" not in raw:
            continue
        source = str(reference.get("source", ""))
        start_text, end_text = raw.rsplit("@bytes:", 1)[1].split("-", 1)
        intervals.append((source, int(start_text), int(end_text)))
    input_tokens, output_tokens, tool_calls = _event_metrics(run_dir / "events.jsonl")
    return (
        capsule_bytes,
        int(run.get("selected_bytes", 0)),
        _union_bytes(intervals),
        int(run.get("peak_input_bytes", 0)),
        input_tokens,
        output_tokens,
        tool_calls,
    )


# 读取当前 Incident run 的稳定失败原因
def _active_run_reason(jobs: JobStore, view: dict[str, Any]) -> str | None:
    incident = view.get("incident")
    if not isinstance(incident, dict):
        return None
    run_id = incident.get("active_run_id")
    if not isinstance(run_id, str):
        return None
    run_path = (
        jobs.job_dir(str(incident["job_id"]))
        / "incidents"
        / str(incident["id"])
        / "runs"
        / run_id
        / "run.json"
    )
    if not run_path.is_file():
        return None
    reason = json.loads(run_path.read_text(encoding="utf-8")).get("reason")
    return str(reason) if reason is not None else None


# 提取有界诊断摘要，避免 benchmark artifact 保存完整工具输出
def _diagnosis_summary(
    diagnosis: object,
) -> tuple[str | None, str | None, str | None, bool | None, list[dict[str, object]]]:
    if not isinstance(diagnosis, dict):
        return None, None, None, None, []
    category = diagnosis.get("category")
    root_cause = diagnosis.get("root_cause")
    support = diagnosis.get("causal_support")
    patch_recommended = diagnosis.get("patch_recommended")
    refs: list[dict[str, object]] = []
    raw_evidence = diagnosis.get("evidence")
    if isinstance(raw_evidence, list):
        for item in raw_evidence[:32]:
            if not isinstance(item, dict):
                continue
            source = item.get("source")
            reference = item.get("reference")
            if isinstance(source, str) and isinstance(reference, str):
                refs.append({"source": source, "reference": reference})
    return (
        category if isinstance(category, str) else None,
        root_cause if isinstance(root_cause, str) else None,
        support if isinstance(support, str) and support in {"direct", "inferred"} else None,
        patch_recommended if isinstance(patch_recommended, bool) else None,
        refs,
    )


# 在当前 Cyan 闭环中运行一次真实失败或无故障 Control
async def run_incident_track(
    case: LoadedCase,
    paths: BenchmarkPaths,
    workspace: Path,
    repeat: int,
    output_dir: Path,
    *,
    is_control: bool,
) -> IncidentBenchmarkArtifact:
    started = time.monotonic()
    jobs = JobStore(output_dir / "cyan-jobs")
    bus = EventBus()
    coordinator: IncidentCoordinator | None = None

    async def handle_failure(job: Any, attempt: Any, failure: Any) -> None:
        assert coordinator is not None
        await coordinator.handle_failure(job, attempt, failure)

    supervisor = JobSupervisor(jobs, handle_failure)
    coordinator = IncidentCoordinator(jobs, supervisor, bus, get_config())
    error: str | None = None
    view: dict[str, Any] = {}
    job_id = "unstarted"
    fallback_job_status = "unstarted"
    try:
        job = await supervisor.start(
            JobSpec(
                argv=command_for_workspace(case, paths, workspace),
                workspace_root=(workspace / case.manifest.cwd).resolve(),
                env=environment_for_workspace(case, workspace),
            )
        )
        job_id = job.id
        job = await supervisor.wait(job.id)
        fallback_job_status = job.status
        if is_control:
            view = await coordinator.job_view(job.id)
        else:
            view = await _wait_incident(coordinator, job.id, timeout_s=900)
            incident = view.get("incident")
            proposal = view.get("proposal")
            if (
                case.manifest.patchable
                and isinstance(incident, dict)
                and isinstance(proposal, dict)
                and incident.get("status") == "awaiting_approval"
                and bool(view.get("can_apply"))
            ):
                await coordinator.decide(
                    str(incident["id"]),
                    str(proposal["id"]),
                    "approve",
                    run_smoke=False,
                )
                view = await _wait_verification(
                    coordinator,
                    job.id,
                    timeout_s=case.manifest.timeout_s + 60,
                )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        await coordinator.close()
    incident = view.get("incident")
    proposal = view.get("proposal")
    diagnosis = view.get("diagnosis")
    final_job = view.get("job")
    job_id = str(final_job.get("id")) if isinstance(final_job, dict) else job_id
    proposal_valid = bool(proposal) and bool(view.get("can_apply"))
    (
        diagnosis_category,
        diagnosis_root_cause,
        diagnosis_causal_support,
        diagnosis_patch_recommended,
        diagnosis_evidence_refs,
    ) = _diagnosis_summary(diagnosis)
    correct_patch_abstention = (
        not case.manifest.patchable
        and diagnosis_patch_recommended is False
        and not isinstance(proposal, dict)
    )
    missed_patch_opportunity = (
        case.manifest.patchable
        and diagnosis_patch_recommended is False
    )
    abstention_gate_violated = isinstance(proposal, dict) and (
        diagnosis_patch_recommended is not True
        or diagnosis_causal_support != "direct"
    )
    metrics = _incident_metrics(jobs, view)
    if error is None and _active_run_reason(jobs, view) == "llm_error":
        error = "llm_error"
    artifact = IncidentBenchmarkArtifact(
        case_id=case.manifest.id,
        repeat=repeat,
        is_control=is_control,
        prompt_version=INCIDENT_PROMPT_VERSION,
        job_id=job_id,
        incident_id=str(incident.get("id")) if isinstance(incident, dict) else None,
        final_job_status=(
            str(final_job.get("status"))
            if isinstance(final_job, dict)
            else fallback_job_status
        ),
        final_incident_status=(
            str(incident.get("status")) if isinstance(incident, dict) else None
        ),
        spurious_incident=is_control and isinstance(incident, dict),
        diagnosis_present=isinstance(diagnosis, dict),
        diagnosis_category=diagnosis_category,
        diagnosis_root_cause=diagnosis_root_cause,
        diagnosis_causal_support=diagnosis_causal_support,
        diagnosis_patch_recommended=diagnosis_patch_recommended,
        diagnosis_evidence_refs=diagnosis_evidence_refs,
        proposal_present=isinstance(proposal, dict),
        proposal_valid=proposal_valid,
        unsafe_proposal=not case.manifest.patchable and isinstance(proposal, dict),
        correct_patch_abstention=correct_patch_abstention,
        missed_patch_opportunity=missed_patch_opportunity,
        abstention_gate_violated=abstention_gate_violated,
        resolved=isinstance(incident, dict) and incident.get("status") == "resolved",
        capsule_tail_bytes=metrics[0],
        selector_selected_bytes=metrics[1],
        unique_evidence_bytes=metrics[2],
        peak_input_bytes=metrics[3],
        input_tokens=metrics[4],
        output_tokens=metrics[5],
        tool_calls=metrics[6],
        duration_seconds=round(time.monotonic() - started, 6),
        error=error,
        created_at=_now(),
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "incident-benchmark.json").write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact
