from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cyan.training.incidents.smoke import SmokeExecution, SmokeResult

_ID_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"

IncidentStatus = Literal[
    "diagnosing",
    "awaiting_approval",
    "applying",
    "smoke_running",
    "smoke_passed",
    "smoke_skipped",
    "retry_running",
    "resolved",
    "rejected",
    "stale",
    "unresolved",
    "rollback_blocked",
]
DiagnosisCategory = Literal[
    "data",
    "config",
    "shape",
    "dtype",
    "device",
    "memory",
    "environment",
    "checkpoint",
    "io",
    "runtime",
    "unknown",
]
CausalSupport = Literal["direct", "inferred"]
EvidenceSource = Literal["stdout", "stderr", "workspace"]
ChangeType = Literal["create", "modify", "delete"]


class EvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: EvidenceSource
    reference: str = Field(min_length=1, max_length=512)
    description: str = Field(min_length=1, max_length=2000)


class Incident(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID_PATTERN)
    job_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    workspace_root: str
    failure_path: str
    status: IncidentStatus = "diagnosing"
    active_run_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    active_proposal_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    diagnosis: Diagnosis | None = None
    proposal: Proposal | None = None
    apply_receipt: PatchReceipt | None = None
    smoke_execution: SmokeExecution | None = None
    smoke_result: SmokeResult | None = None
    last_outcome: str | None = Field(default=None, max_length=4000)
    created_at: datetime
    updated_at: datetime

    # 保证工作区路径是绝对路径，避免调用方依赖 daemon 的当前目录
    @field_validator("workspace_root")
    @classmethod
    def _workspace_must_be_absolute(cls, value: str) -> str:
        from pathlib import Path

        if not Path(value).is_absolute():
            raise ValueError("workspace_root must be absolute")
        return value


class Diagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID_PATTERN)
    incident_id: str = Field(pattern=_ID_PATTERN)
    category: DiagnosisCategory
    summary: str = Field(min_length=1, max_length=4000)
    root_cause: str = Field(min_length=1, max_length=8000)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)
    causal_support: CausalSupport = "inferred"
    patch_recommended: bool = False
    created_at: datetime


class ProposalFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    change_type: ChangeType
    base_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=_ID_PATTERN)
    incident_id: str = Field(pattern=_ID_PATTERN)
    diagnosis_id: str | None = Field(default=None, pattern=_ID_PATTERN)
    patch_path: Literal["proposal.diff"] = "proposal.diff"
    patch_sha256: str = Field(pattern=_SHA256_PATTERN)
    files: list[ProposalFile] = Field(min_length=1)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=32)
    created_at: datetime


class AppliedFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1)
    applied_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)


class PatchReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    proposal_id: str = Field(pattern=_ID_PATTERN)
    files: list[AppliedFile] = Field(min_length=1)
    applied_at: datetime


class LogSnapshot(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    size: int = Field(ge=0)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    included_start: int = Field(ge=0)
    included_end: int = Field(ge=0)
    tail: str


class FailureCapsule(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    job_id: str = Field(pattern=_ID_PATTERN)
    attempt_id: str = Field(pattern=_ID_PATTERN)
    argv: list[str] = Field(min_length=1)
    cwd: str
    occurred_at: datetime
    failure_kind: Literal["launch_error", "process_exit", "supervisor_error"]
    returncode: int | None = None
    signal: int | None = None
    git_head: str | None = None
    dirty_paths: list[str] = Field(default_factory=list)
    environment: dict[str, str] = Field(default_factory=dict)
    stdout: LogSnapshot
    stderr: LogSnapshot
