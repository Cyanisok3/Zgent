from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from cyan.core.jobs.workflow import ArtifactMetadata, WorkflowContract, WorkflowPhase

JobStatus = Literal[
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
AttemptStatus = Literal[
    "starting",
    "running",
    "succeeded",
    "failed",
    "cancelled",
    "interrupted",
]
LogStream = Literal["stdout", "stderr"]
JobEventType = Literal[
    "job.started",
    "job.succeeded",
    "job.failed",
    "job.cancelled",
    "job.interrupted",
]


class JobSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    argv: list[str] = Field(min_length=1)
    workspace_root: Path
    env: dict[str, str] = Field(default_factory=dict)
    workflow_contract: WorkflowContract | None = None


class JobRecord(BaseModel):
    id: str
    status: JobStatus
    created_at: str
    updated_at: str
    current_attempt_id: str | None = None
    attempt_ids: list[str] = Field(default_factory=list)


class AttemptRecord(BaseModel):
    id: str
    job_id: str
    status: AttemptStatus
    started_at: str
    finished_at: str | None = None
    pid: int | None = None
    process_identity: str | None = None
    returncode: int | None = None
    signal: int | None = None
    error: str | None = None
    phase: WorkflowPhase = "main"
    check_id: str | None = None
    artifact_baseline: list[ArtifactMetadata] = Field(default_factory=list)


class FailureRecord(BaseModel):
    job_id: str
    attempt_id: str
    occurred_at: str
    kind: Literal[
        "launch_error",
        "process_exit",
        "supervisor_error",
        "contract_violation",
    ]
    returncode: int | None = None
    signal: int | None = None
    message: str
    capsule: dict[str, Any] | None = None
    phase: WorkflowPhase | None = None
    check_id: str | None = None
    artifact_path: str | None = None
    contract_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    violation_rule: str | None = None


class JobEvent(BaseModel):
    seq: int
    type: JobEventType
    job_id: str
    attempt_id: str | None
    status: JobStatus
    occurred_at: str


class LogChunk(BaseModel):
    stream: LogStream
    start_offset: int
    end_offset: int
    total_bytes: int
    eof: bool
    text: str
