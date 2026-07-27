from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Discriminator, Field

from cyan.core.session.model import SessionMode, SessionStatus


class PingCommand(BaseModel):
    type: Literal["core.ping"] = "core.ping"
    client: str


class PongResult(BaseModel):
    server_version: str
    uptime_ms: int
    received_at: str  # ISO 8601


class CoreShutdownCommand(BaseModel):
    type: Literal["core.shutdown"] = "core.shutdown"


class CoreShutdownResult(BaseModel):
    status: Literal["stopping"] = "stopping"


class AgentRunCommand(BaseModel):
    type: Literal["agent.run"] = "agent.run"
    goal: str


class AgentRunResult(BaseModel):
    run_id: str


class EventSubscribeCommand(BaseModel):
    type: Literal["event.subscribe"] = "event.subscribe"
    topics: list[str]          # fnmatch 模式，如 ["step.*", "tool.*"]
    scope: str = "global"      # "global" | "run:<run_id>" | "job:<job_id>"
    replay_from_run: str | None = None  # 设置则先从 events.jsonl 回放历史再接实时流
    after_seq: int = Field(default=0, ge=0)


class EventSubscribeResult(BaseModel):
    subscription_id: str
    replayed_count: int = 0


class SessionCreateCommand(BaseModel):
    type: Literal["session.create"] = "session.create"
    mode: SessionMode = "chat"
    title: str = ""
    workspace_root: str = ""


class SessionCreateResult(BaseModel):
    session_id: str
    status: SessionStatus


class SessionSendMessageCommand(BaseModel):
    type: Literal["session.send_message"] = "session.send_message"
    session_id: str
    content: str


class SessionSendMessageResult(BaseModel):
    run_id: str


class SessionGetHistoryCommand(BaseModel):
    type: Literal["session.get_history"] = "session.get_history"
    session_id: str


class SessionGetHistoryResult(BaseModel):
    messages: list[dict[str, Any]]


class SessionCloseCommand(BaseModel):
    type: Literal["session.close"] = "session.close"
    session_id: str


class SessionCloseResult(BaseModel):
    status: SessionStatus


class PermissionRespondCommand(BaseModel):
    type: Literal["permission.respond"] = "permission.respond"
    tool_use_id: str
    # "allow_once" | "always_allow" | "deny_once" | "always_deny"
    decision: str


class PermissionRespondResult(BaseModel):
    ok: bool = True


class SessionCompactCommand(BaseModel):
    type: Literal["session.compact"] = "session.compact"
    session_id: str
    focus: str = ""


class SessionCompactResult(BaseModel):
    summary_tokens: int
    saved_tokens: int


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
    smoke_config_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
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
    eof: bool


class IncidentDecideCommand(BaseModel):
    type: Literal["incident.decide"] = "incident.decide"
    incident_id: str
    proposal_id: str
    decision: Literal["approve", "reject"]
    run_smoke: bool = True
    smoke_config_fingerprint: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )


class IncidentDecideResult(BaseModel):
    status: str


# 根据 type 字段决定命令类型的判别联合
Command = Annotated[
    PingCommand
    | CoreShutdownCommand
    | AgentRunCommand
    | EventSubscribeCommand
    | SessionCreateCommand
    | SessionSendMessageCommand
    | SessionGetHistoryCommand
    | SessionCloseCommand
    | PermissionRespondCommand
    | SessionCompactCommand
    | JobStartCommand
    | JobListCommand
    | JobGetCommand
    | JobCancelCommand
    | JobReadLogCommand
    | IncidentDecideCommand,
    Discriminator("type"),
]
