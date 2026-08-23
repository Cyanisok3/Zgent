from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cyan.training.incidents.models import FailureCapsule, Incident
from cyan.training.incidents.selector import EvidenceSelection

MAX_INPUT_BYTES = 128 * 1024
MAX_INITIAL_EVIDENCE_BYTES = 32 * 1024


@dataclass
class IncidentContext:
    system_prompt: str
    evidence_refs: set[str]
    selection: EvidenceSelection


# 组装单轮 Incident 的固定 prompt、有限证据和当前状态摘要
def build_incident_context(
    incident: Incident,
    capsule: FailureCapsule,
    selection: EvidenceSelection,
    instruction: str,
    previous_outcome_summary: dict[str, Any] | None = None,
) -> IncidentContext:
    evidence_refs = {item.as_reference(capsule) for item in selection.references}
    previous: dict[str, Any] = dict(previous_outcome_summary or {})
    if incident.diagnosis is not None:
        previous.setdefault("diagnosis", incident.diagnosis.model_dump(mode="json"))
    if incident.proposal is not None:
        previous.setdefault("proposal", incident.proposal.model_dump(mode="json"))
    if incident.smoke_result is not None:
        previous.setdefault("smoke_result", incident.smoke_result.model_dump(mode="json"))
    prompt = (
        "You are cyan's read-only Incident Agent for a crashed local ML training process.\n"
        "This is incident response, not metric optimization or experiment research.\n"
        "Inspect only the bound workspace and immutable logs. Never run commands or modify "
        "the workspace. Use read-only tools to verify causal evidence, call "
        "submit_diagnosis exactly once when enough evidence exists, and propose at most one "
        "exact single-file SEARCH/REPLACE only when it directly fixes the observed crash. "
        "A diagnosis without a patch is valid. Evidence must cite observed references.\n\n"
        f"Failure capsule:\n{capsule.model_dump_json(indent=2)}\n\n"
        f"Selected evidence:\n{selection.content}\n\n"
        f"Previous bounded outcome:\n{json.dumps(previous, ensure_ascii=False, default=str)}\n\n"
        f"Current instruction:\n{instruction}"
    )
    return IncidentContext(system_prompt=prompt, evidence_refs=evidence_refs, selection=selection)
