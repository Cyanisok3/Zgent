from __future__ import annotations

from pathlib import Path
from typing import Any

from cyan.agent.runner import RunProfile
from cyan.agent.tools.builtin import ListDirTool, ReadFileTool, SearchTextTool
from cyan.training.incidents.context import MAX_INPUT_BYTES, build_incident_context
from cyan.training.incidents.evidence import _reference_was_observed
from cyan.training.incidents.log_tool import FileJobLogReader, ReadJobLogTool
from cyan.training.incidents.models import EvidenceRef, FailureCapsule, Incident
from cyan.training.incidents.patch import PatchService
from cyan.training.incidents.selector import EvidenceSelection
from cyan.training.incidents.store import IncidentStore
from cyan.training.incidents.tools import ProposePatchTool, SubmitDiagnosisTool
from cyan.training.jobs.store import JobStore


# 为单轮 Incident 构造固定六工具 profile
def build_incident_profile(
    jobs: JobStore,
    store: IncidentStore,
    incident: Incident,
    capsule: FailureCapsule,
    selection: EvidenceSelection,
    instruction: str,
    previous_outcome_summary: dict[str, Any] | None = None,
) -> RunProfile:
    root = Path(incident.workspace_root).resolve(strict=True)
    context = build_incident_context(
        incident,
        capsule,
        selection,
        instruction,
        previous_outcome_summary,
    )
    observed_refs = set(context.evidence_refs)
    stderr_reference = (
        f"stderr:{incident.job_id}/{incident.attempt_id}"
        f"@bytes:{capsule.stderr.included_start}-{capsule.stderr.included_end}"
    )
    stdout_reference = (
        f"stdout:{incident.job_id}/{incident.attempt_id}"
        f"@bytes:{capsule.stdout.included_start}-{capsule.stdout.included_end}"
    )
    observed_refs.update({stderr_reference, stdout_reference})

    # 校验 Agent 提交的证据只来自初始选择或已经读取的结果
    def validate_evidence(evidence: EvidenceRef) -> str | None:
        if evidence.source in ("stdout", "stderr"):
            if not _reference_was_observed(evidence.reference, observed_refs):
                return f"evidence reference was not observed: {evidence.reference}"
            return None
        if "@sha256:" not in evidence.reference:
            return "workspace evidence must include a path and SHA-256"
        return None

    reader = FileJobLogReader(jobs.log_path)
    tools = [
        ReadFileTool(root, evidence_refs=observed_refs, max_bytes=64 * 1024, text_only=True),
        ListDirTool(root),
        SearchTextTool(root, evidence_refs=observed_refs),
        ReadJobLogTool(reader, evidence_refs=observed_refs),
        SubmitDiagnosisTool(store, incident.id, evidence_validator=validate_evidence),
        ProposePatchTool(
            store,
            incident.id,
            root,
            evidence_validator=validate_evidence,
            patch_service=PatchService(root) if capsule.git_head is not None else None,
        ),
    ]
    return RunProfile(
        workspace_root=root,
        system_prompt=context.system_prompt,
        tools=tools,
        max_steps=12,
        max_input_bytes=MAX_INPUT_BYTES,
        summary_only_events=True,
    )
