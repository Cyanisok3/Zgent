from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from cyan.training.incidents.models import FailureCapsule, Incident
from cyan.training.incidents.selector import EvidenceSelection

MAX_INPUT_BYTES = 128 * 1024
MAX_INITIAL_EVIDENCE_BYTES = 32 * 1024
INCIDENT_PROMPT_VERSION = "causal-support-abstention-v6"


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
        "the workspace. Use read-only tools to verify causal evidence, then call "
        "submit_diagnosis exactly once. The diagnosis must include causal_support="
        "direct or inferred and patch_recommended=true or false.\n"
        "Copy evidence items using exactly these three JSON fields; never add kind, "
        "selection_reason, or sha256:\n"
        '{"source":"stderr","reference":"stderr:job-1/attempt-1@bytes:0-42",'
        '"description":"latest complete Python traceback"}\n'
        "When causal_support is direct, root_cause must trace to the earliest evidence-backed "
        "upstream configuration, data contract, component, or producer, not only the traceback "
        "leaf. When support is inferred, root_cause must begin with "
        "'Inference — not directly established by observed evidence:' and must not invent an "
        "unobserved file, setting, variable, or dependency version.\n"
        "Set patch_recommended=true only for a user-workspace cause fixable by one exact "
        "replacement in one existing file and verifiable by rerunning the original command. "
        "A change that only removes the first observed exception is not safe when downstream "
        "producer-consumer contract assumptions remain unverified; in that case set it false. "
        "For external data, environment or framework limitations, dependency changes, "
        "multi-file fixes, or insufficient evidence, set it false and stop after diagnosis. "
        "Once the observed traceback and workspace producer establish the relevant contract, "
        "submit the diagnosis immediately; abstaining does not require exhaustively tracing "
        "third-party framework internals. Use read-only tools only when they add causal "
        "evidence, and do not draft a patch before submitting the diagnosis. After a "
        "successful direct diagnosis with patch_recommended=true, immediately call "
        "propose_patch using already observed source; otherwise stop. Preserve the input budget "
        "for submit_diagnosis and, when allowed, propose_patch. "
        "A diagnosis without a patch is valid. Evidence must cite observed references.\n\n"
        f"Failure capsule:\n{capsule.model_dump_json(indent=2)}\n\n"
        f"Selected evidence:\n{selection.content}\n\n"
        f"Previous bounded outcome:\n{json.dumps(previous, ensure_ascii=False, default=str)}\n\n"
        f"Current instruction:\n{instruction}"
    )
    return IncidentContext(system_prompt=prompt, evidence_refs=evidence_refs, selection=selection)
