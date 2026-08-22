from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator

from cyan.agent.events.models import (
    ContextCompactedEvent,
    LlmModelSelectedEvent,
    LlmTokenEvent,
    LlmUsageEvent,
    LogLineEvent,
    PermissionDeniedEvent,
    PermissionGrantedEvent,
    PermissionRequestedEvent,
    RunFinishedEvent,
    RunStartedEvent,
    SessionClosedEvent,
    SessionCreatedEvent,
    SessionMessageReceivedEvent,
    SessionResumedEvent,
    SessionWaitingForInputEvent,
    SkillInvokedEvent,
    StepFinishedEvent,
    StepStartedEvent,
    SubagentFinishedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from cyan.training.events import (
    IncidentOpenedEvent,
    IncidentStatusChangedEvent,
    JobFinishedEvent,
    JobStartedEvent,
    PatchProposedEvent,
    SmokeFinishedEvent,
)


class CoreStartedEvent(BaseModel):
    type: Literal["core.started"] = "core.started"
    listen_addr: str
    version: str


# 根据 type 字段决定事件类型的判别联合
Event = Annotated[
    CoreStartedEvent
    | RunStartedEvent
    | RunFinishedEvent
    | StepStartedEvent
    | StepFinishedEvent
    | ToolCallStartedEvent
    | ToolCallFinishedEvent
    | ToolCallFailedEvent
    | LlmTokenEvent
    | LlmUsageEvent
    | LlmModelSelectedEvent
    | LogLineEvent
    | SessionCreatedEvent
    | SessionMessageReceivedEvent
    | SessionWaitingForInputEvent
    | SessionResumedEvent
    | SessionClosedEvent
    | ContextCompactedEvent
    | PermissionRequestedEvent
    | PermissionGrantedEvent
    | PermissionDeniedEvent
    | SubagentStartedEvent
    | SubagentFinishedEvent
    | SkillInvokedEvent
    | JobStartedEvent
    | JobFinishedEvent
    | IncidentOpenedEvent
    | IncidentStatusChangedEvent
    | PatchProposedEvent
    | SmokeFinishedEvent,
    Discriminator("type"),
]
