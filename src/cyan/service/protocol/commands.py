from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field

WIRE_PROTOCOL_VERSION = 2

COMMAND_TYPES = (
    "core.ping",
    "core.shutdown",
    "event.subscribe",
    "launch.preview",
    "launch.start",
    "job.start",
    "job.list",
    "job.get",
    "job.cancel",
    "job.read_log",
    "incident.decide",
    "incident.review",
    "incident.follow_up",
)


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str
    protocol_version: int | None = None


class PongResult(BaseModel):
    server_version: str
    protocol_version: int
    startup_workspace_root: str
    uptime_ms: int
    received_at: str


class CoreShutdownCommand(BaseModel):
    type: Literal["core.shutdown"] = "core.shutdown"


class CoreShutdownResult(BaseModel):
    status: Literal["stopping"] = "stopping"


class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str]
    scope: str = "global"
    replay_from_run: str | None = None
    after_seq: int = Field(default=0, ge=0)


class EventSubscribeResult(BaseModel):
    subscription_id: str
    replayed_count: int = 0


class LaunchPreviewCommand(BaseModel):
    type: Literal["launch.preview"] = "launch.preview"
    command: str = Field(min_length=1, max_length=64 * 1024)
    workspace_root: str
    env: dict[str, str] = Field(default_factory=dict)


class LaunchPreviewResult(BaseModel):
    argv: list[str]
    cwd: str
    env_overrides: dict[str, str]
    executable: str
    config_paths: list[str]
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LaunchStartCommand(BaseModel):
    type: Literal["launch.start"] = "launch.start"
    command: str = Field(min_length=1, max_length=64 * 1024)
    workspace_root: str
    env: dict[str, str] = Field(default_factory=dict)
    preview_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")


class LaunchStartResult(BaseModel):
    job_id: str


class JobStartCommand(BaseModel):
    type: Literal["job.start"] = "job.start"
    argv: list[str]
    workspace_root: str
    env: dict[str, str] = Field(default_factory=dict)


class JobStartResult(BaseModel):
    job_id: str


class JobListCommand(BaseModel):
    type: Literal["job.list"] = "job.list"


class JobListResult(BaseModel):
    jobs: list[dict[str, Any]]


class JobGetCommand(BaseModel):
    type: Literal["job.get"] = "job.get"
    job_id: str


class JobGetResult(BaseModel):
    job: dict[str, Any]
    argv: list[str]
    workspace_root: str
    attempt: dict[str, Any] | None = None
    incident: dict[str, Any] | None = None
    diagnosis: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    patch: str | None = None
    smoke_config: dict[str, Any] | None = None
    smoke_config_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    smoke: dict[str, Any] | None = None
    can_apply: bool = False
    smoke_config_error: str | None = None


class JobCancelCommand(BaseModel):
    type: Literal["job.cancel"] = "job.cancel"
    job_id: str


class JobCancelResult(BaseModel):
    status: str


class JobReadLogCommand(BaseModel):
    type: Literal["job.read_log"] = "job.read_log"
    job_id: str
    attempt_id: str
    stream: Literal["stdout", "stderr"]
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=32 * 1024, ge=1, le=64 * 1024)


class JobReadLogResult(BaseModel):
    data: str
    next_offset: int
    total_bytes: int
    eof: bool


class IncidentDecideCommand(BaseModel):
    type: Literal["incident.decide"] = "incident.decide"
    incident_id: str
    proposal_id: str
    decision: Literal["approve", "reject"]
    run_smoke: bool = True
    smoke_config_fingerprint: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")


class IncidentDecideResult(BaseModel):
    status: str


class IncidentReviewCommand(BaseModel):
    type: Literal["incident.review"] = "incident.review"
    job_id: str
    incident_id: str
    proposal_id: str


class IncidentReviewResult(BaseModel):
    proposal_id: str
    path: str
    before_text: str
    after_text: str


class IncidentFollowUpCommand(BaseModel):
    type: Literal["incident.follow_up"] = "incident.follow_up"
    incident_id: str
    content: str = Field(min_length=1, max_length=64 * 1024)


class IncidentFollowUpResult(BaseModel):
    run_id: str


# 根据 type 字段决定当前 v2 命令类型
Command = Annotated[
    PingCommand
    | CoreShutdownCommand
    | EventSubscribeCommand
    | LaunchPreviewCommand
    | LaunchStartCommand
    | JobStartCommand
    | JobListCommand
    | JobGetCommand
    | JobCancelCommand
    | JobReadLogCommand
    | IncidentDecideCommand
    | IncidentReviewCommand
    | IncidentFollowUpCommand,
    Discriminator("type"),
]
