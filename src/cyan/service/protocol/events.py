from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Discriminator

from cyan.agent.events.models import (
    LlmModelSelectedEvent,
    LlmTokenEvent,
    LlmUsageEvent,
    RunFinishedEvent,
    RunStartedEvent,
    StepFinishedEvent,
    StepStartedEvent,
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

EVENT_TYPES = (
    "core.started",
    "run.started",
    "run.finished",
    "step.started",
    "step.finished",
    "tool.call_started",
    "tool.call_finished",
    "tool.call_failed",
    "llm.token",
    "llm.usage",
    "llm.model_selected",
    "job.started",
    "job.finished",
    "incident.opened",
    "incident.status_changed",
    "patch.proposed",
    "smoke.finished",
)


class CoreStartedEvent(BaseModel):
    type: Literal["core.started"] = "core.started"
    listen_addr: str
    version: str


# 组合 Agent、训练和服务层的完整 v2 事件联合类型
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
    | JobStartedEvent
    | JobFinishedEvent
    | IncidentOpenedEvent
    | IncidentStatusChangedEvent
    | PatchProposedEvent
    | SmokeFinishedEvent,
    Discriminator("type"),
]
