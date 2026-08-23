from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime

from pydantic import ValidationError

from cyan.agent.events.bus import EventBus
from cyan.agent.events.models import (
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from cyan.agent.llm.types import ToolCallBlock
from cyan.agent.tools.base import ToolResult
from cyan.agent.tools.errors import RateLimitedError
from cyan.agent.tools.registry import ToolRegistry

_DEFAULT_TIMEOUT = 120.0
_MAX_RETRIES = 2
_RETRY_BASE_S = 2.0
_RETRYABLE = frozenset({"runtime_error", "rate_limited"})


# 返回当前 UTC 时间字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 发布失败摘要事件并转换为可继续观察的 ToolResult
async def _fail(
    bus: EventBus,
    run_id: str,
    tool_call: ToolCallBlock,
    error_class: str,
    error_message: str,
    elapsed_ms: int,
    *,
    attempt: int = 1,
) -> ToolResult:
    await bus.publish(
        ToolCallFailedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            error_class=error_class,
            error_message=error_message,
            elapsed_ms=elapsed_ms,
            attempt=attempt,
            ts=_now(),
        )
    )
    return ToolResult(content=error_message, is_error=True, error_type=error_class)


# 校验并限时调用只读工具，保留有限重试但不引入权限状态机
async def invoke_tool(
    registry: ToolRegistry,
    tool_call: ToolCallBlock,
    bus: EventBus,
    run_id: str,
    timeout: float = _DEFAULT_TIMEOUT,
) -> ToolResult:
    started = time.monotonic()
    await bus.publish(
        ToolCallStartedEvent(
            run_id=run_id,
            tool_use_id=tool_call.id,
            tool_name=tool_call.name,
            params=dict(tool_call.input),
            ts=_now(),
        )
    )

    # 计算本次工具调用已经消耗的毫秒数
    def elapsed() -> int:
        return int((time.monotonic() - started) * 1000)

    tool = registry.get(tool_call.name)
    if tool is None:
        return await _fail(
            bus, run_id, tool_call, "runtime_error", f"unknown tool: {tool_call.name}", elapsed()
        )
    if tool.params_model is not None:
        try:
            tool.params_model.model_validate(dict(tool_call.input))
        except ValidationError as exc:
            return await _fail(bus, run_id, tool_call, "schema_error", str(exc), elapsed())

    for attempt in range(1, _MAX_RETRIES + 2):
        error_class: str | None = None
        error_message: str | None = None
        try:
            result = await asyncio.wait_for(tool.invoke(dict(tool_call.input)), timeout=timeout)
            if not result.is_error:
                await bus.publish(
                    ToolCallFinishedEvent(
                        run_id=run_id,
                        tool_use_id=tool_call.id,
                        tool_name=tool_call.name,
                        elapsed_ms=elapsed(),
                        output=result.content,
                        ts=_now(),
                    )
                )
                return result
            error_class = result.error_type or "runtime_error"
            error_message = result.content
        except RateLimitedError as exc:
            error_class = "rate_limited"
            error_message = str(exc)
        except TimeoutError:
            return await _fail(
                bus,
                run_id,
                tool_call,
                "timeout",
                f"tool timed out after {timeout}s",
                elapsed(),
                attempt=attempt,
            )
        except Exception as exc:
            error_class = "runtime_error"
            error_message = str(exc)
        assert error_class is not None and error_message is not None
        if error_class in _RETRYABLE and attempt <= _MAX_RETRIES:
            await bus.publish(
                ToolCallFailedEvent(
                    run_id=run_id,
                    tool_use_id=tool_call.id,
                    tool_name=tool_call.name,
                    error_class=error_class,
                    error_message=error_message,
                    elapsed_ms=elapsed(),
                    attempt=attempt,
                    ts=_now(),
                )
            )
            await asyncio.sleep(_RETRY_BASE_S * (2 ** (attempt - 1)))
            continue
        return await _fail(
            bus, run_id, tool_call, error_class, error_message, elapsed(), attempt=attempt
        )
    return ToolResult(content="internal error", is_error=True, error_type="runtime_error")
