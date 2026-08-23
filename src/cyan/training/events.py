from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class JobStartedEvent(BaseModel):
    type: Literal["job.started"] = "job.started"
    seq: int
    job_id: str
    attempt_id: str
    argv: list[str]
    workspace_root: str
    ts: str


class JobFinishedEvent(BaseModel):
    type: Literal["job.finished"] = "job.finished"
    seq: int
    job_id: str
    attempt_id: str
    status: str
    exit_code: int | None = None
    signal: int | None = None
    ts: str


class IncidentOpenedEvent(BaseModel):
    type: Literal["incident.opened"] = "incident.opened"
    job_id: str
    incident_id: str
    attempt_id: str
    run_id: str | None = None
    ts: str


class IncidentStatusChangedEvent(BaseModel):
    type: Literal["incident.status_changed"] = "incident.status_changed"
    job_id: str
    incident_id: str
    status: str
    ts: str


class PatchProposedEvent(BaseModel):
    type: Literal["patch.proposed"] = "patch.proposed"
    job_id: str
    incident_id: str
    proposal_id: str
    summary: str
    ts: str


class SmokeFinishedEvent(BaseModel):
    type: Literal["smoke.finished"] = "smoke.finished"
    job_id: str
    incident_id: str
    status: str
    exit_code: int | None = None
    ts: str
