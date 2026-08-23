from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel


class RunStartedEvent(BaseModel):
    type: Literal["run.started"] = "run.started"
    run_id: str
    goal: str
    ts: str


class RunFinishedEvent(BaseModel):
    type: Literal["run.finished"] = "run.finished"
    run_id: str
    status: str
    reason: str | None = None
    steps: int
    ts: str


class StepStartedEvent(BaseModel):
    type: Literal["step.started"] = "step.started"
    run_id: str
    step: int
    ts: str


class StepFinishedEvent(BaseModel):
    type: Literal["step.finished"] = "step.finished"
    run_id: str
    step: int
    ts: str


class ToolCallStartedEvent(BaseModel):
    type: Literal["tool.call_started"] = "tool.call_started"
    run_id: str
    tool_use_id: str
    tool_name: str
    params: dict[str, Any]
    ts: str


class ToolCallFinishedEvent(BaseModel):
    type: Literal["tool.call_finished"] = "tool.call_finished"
    run_id: str
    tool_use_id: str
    tool_name: str
    elapsed_ms: int
    output: str = ""
    ts: str


class ToolCallFailedEvent(BaseModel):
    type: Literal["tool.call_failed"] = "tool.call_failed"
    run_id: str
    tool_use_id: str
    tool_name: str
    error_class: str
    error_message: str
    elapsed_ms: int
    attempt: int = 1
    ts: str


class LlmTokenEvent(BaseModel):
    type: Literal["llm.token"] = "llm.token"
    run_id: str
    token: str
    ts: str


class LlmUsageEvent(BaseModel):
    type: Literal["llm.usage"] = "llm.usage"
    run_id: str
    input_tokens: int
    output_tokens: int
    cache_read_input_tokens: int
    cache_creation_input_tokens: int
    context_pct: float = 0.0
    ts: str


class LlmModelSelectedEvent(BaseModel):
    type: Literal["llm.model_selected"] = "llm.model_selected"
    run_id: str
    model: str
    strategy: str
    ts: str
