from __future__ import annotations

from pathlib import Path

from cyan.agent.runner import RunProfile
from cyan.agent.session import Session
from cyan.agent.tools.builtin import ReadFileTool, SearchTextTool
from cyan.training.incidents.evidence import (
    _BudgetedJobLogReader,
    _reference_was_observed,
    _tail_text,
)
from cyan.training.incidents.log_tool import ReadJobLogTool
from cyan.training.incidents.models import EvidenceRef, FailureCapsule, Incident
from cyan.training.incidents.patch import PatchService
from cyan.training.incidents.store import IncidentStore
from cyan.training.incidents.tools import ProposePatchTool, SubmitDiagnosisTool
from cyan.training.jobs.store import JobStore


# 构造 Incident 元数据不可用时的最小只读 profile
def unavailable_incident_profile(root: Path) -> RunProfile:
    return RunProfile(
        workspace_root=root,
        system_prompt_override=(
            "This incident metadata is unavailable. Do not inspect or modify files. "
            "Explain that the incident cannot be resumed."
        ),
        tool_whitelist=[],
        max_steps=1,
        compact_threshold=0.0,
        summary_only_events=True,
        include_context=False,
    )


# 为 Incident session 构造固定六工具的不可压缩只读 profile
def build_incident_profile(
    session: Session,
    jobs: JobStore,
    store: IncidentStore,
    incident: Incident,
) -> RunProfile:
    root = Path(session.workspace_root).resolve()
    failure = jobs.read_failure(incident.job_id, incident.attempt_id)
    capsule = FailureCapsule.model_validate(failure.capsule)

    smoke_context = ""
    try:
        smoke = store.read_smoke_result(incident.id)
        smoke_stdout = _tail_text(smoke.stdout_path, 16 * 1024)
        smoke_stderr = _tail_text(smoke.stderr_path, 16 * 1024)
        smoke_context = (
            "\nLatest smoke verifier result:\n"
            f"{smoke.model_dump_json()}\n"
            f"smoke stdout tail:\n{smoke_stdout}\n"
            f"smoke stderr tail:\n{smoke_stderr}\n"
        )
    except (FileNotFoundError, OSError, ValueError):
        pass

    reader = _BudgetedJobLogReader(jobs, store, incident)
    evidence_refs: set[str] = set()
    stderr_reference = (
        f"stderr:{incident.job_id}/{incident.attempt_id}"
        f"@bytes:{capsule.stderr.included_start}-{capsule.stderr.included_end}"
    )
    stdout_reference = (
        f"stdout:{incident.job_id}/{incident.attempt_id}"
        f"@bytes:{capsule.stdout.included_start}-{capsule.stdout.included_end}"
    )
    if capsule.stderr.included_end > capsule.stderr.included_start:
        evidence_refs.add(stderr_reference)
    if capsule.stdout.included_end > capsule.stdout.included_start:
        evidence_refs.add(stdout_reference)

    # 只接受本轮工具实际返回或 failure capsule 明确给出的引用
    def validate_evidence(evidence: EvidenceRef) -> str | None:
        if not _reference_was_observed(evidence.reference, evidence_refs):
            return f"evidence reference was not observed: {evidence.reference}"
        if evidence.source in ("stdout", "stderr"):
            if not evidence.reference.startswith(f"{evidence.source}:"):
                return "log evidence source does not match its reference"
        elif "@sha256:" not in evidence.reference:
            return "workspace evidence must include a path and SHA-256"
        return None

    system_prompt = (
        "You are cyan's read-only Incident Agent for a crashed local ML training process. "
        "This is incident response, not metric optimization or experiment research.\n"
        "You may inspect only the supplied workspace and this incident's immutable logs. "
        "Never claim a command ran, never modify the workspace, and never optimize loss or "
        "accuracy. Start from the failure capsule and traceback, then use targeted log search "
        "and small source reads only as needed. As soon as the evidence is sufficient, call "
        "submit_diagnosis exactly once before considering a patch. Log evidence must cite "
        "references returned by read_job_log; workspace evidence must cite path@sha256#line "
        "references returned by read/search tools. If the crash is fully caused by an invalid "
        "launch argument, missing external path or data, or the host environment, stop after "
        "diagnosis and tell the user what to change; do not add validation or fallback code. "
        "Otherwise propose a patch only when one minimal source or config edit directly fixes "
        "the observed crash. Modify only the causal call site; do not harden similar sites, "
        "refactor, or make unrelated improvements. Then call propose_patch with one relative "
        "file path and one exact SEARCH/REPLACE pair. Copy the smallest "
        "unique contiguous SEARCH text verbatim from read_file or search_text output; do not "
        "write diff headers, hunks, or line numbers. If exact matching fails, use the returned "
        "real source feedback to correct the proposal once. If the corrected proposal still "
        "fails, keep the diagnosis and stop proposing a patch. "
        "A diagnosis without a patch is valid.\n"
        f"Failure capsule:\n{capsule.model_dump_json(indent=2)}\n"
        f"Default stderr reference: {stderr_reference}\n"
        f"Default stdout reference: {stdout_reference}\n"
        f"{smoke_context}"
    )
    return RunProfile(
        workspace_root=root,
        system_prompt_override=system_prompt,
        tool_whitelist=[
            "read_file",
            "list_dir",
            "search_text",
            "read_job_log",
            "submit_diagnosis",
            "propose_patch",
        ],
        extra_tools=[
            ReadFileTool(
                root,
                evidence_refs=evidence_refs,
                max_bytes=64 * 1024,
                text_only=True,
            ),
            SearchTextTool(root, evidence_refs=evidence_refs),
            ReadJobLogTool(reader, evidence_refs=evidence_refs),
            SubmitDiagnosisTool(
                store,
                incident.id,
                evidence_validator=validate_evidence,
            ),
            ProposePatchTool(
                store,
                incident.id,
                root,
                evidence_validator=validate_evidence,
                patch_service=PatchService(root) if capsule.git_head is not None else None,
            ),
        ],
        max_steps=12,
        compact_threshold=0.0,
        summary_only_events=True,
        include_context=False,
        evidence_refs=evidence_refs,
    )
