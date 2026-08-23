from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime

from cyan.agent.context import ExecutionContext
from cyan.agent.events.bus import EventBus
from cyan.agent.events.models import StepFinishedEvent, StepStartedEvent
from cyan.agent.llm.base import LLMProvider
from cyan.agent.tools.invocation import invoke_tool
from cyan.agent.tools.registry import ToolRegistry

log = logging.getLogger(__name__)


# 返回当前 UTC 时间字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentLoop:
    # 初始化极简 LLM、工具和事件循环
    def __init__(self, provider: LLMProvider, registry: ToolRegistry, bus: EventBus) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus

    # 执行 LLM → tool → observe 循环，直到模型结束或达到硬步数上限
    async def run(self, context: ExecutionContext) -> None:
        while not context.is_done():
            context.step += 1
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
            schemas = self._registry.tool_schemas()
            input_size = context.serialized_size(schemas)
            context.observe_input_size(input_size)
            if input_size > context.max_input_bytes:
                context.mark_failed("context_budget_exhausted")
                await self._bus.publish(
                    StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
                )
                break
            try:
                response = await self._provider.chat(
                    messages=context.messages,
                    tool_schemas=schemas,
                    bus=self._bus,
                    run_id=context.run_id,
                    step=context.step,
                    system=context.system_prompt(),
                )
            except asyncio.CancelledError:
                context.mark_failed("cancelled")
                raise
            except Exception:
                log.exception("LLM call failed run_id=%s step=%d", context.run_id, context.step)
                context.mark_failed("llm_error")
                break

            blocks: list[dict[str, object]] = list(response.thinking_blocks)
            if response.text:
                blocks.append({"type": "text", "text": response.text})
            for tool_call in response.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": tool_call.id,
                        "name": tool_call.name,
                        "input": tool_call.input,
                    }
                )
            context.add_assistant_message(blocks)
            context.observe_input_size(context.serialized_size(schemas))
            if response.stop_reason == "tool_use":
                for tool_call in response.tool_calls:
                    result = await invoke_tool(self._registry, tool_call, self._bus, context.run_id)
                    context.add_tool_result(
                        tool_call.id,
                        result.content,
                        result.is_error,
                        tool_schemas=schemas,
                    )
                    context.observe_input_size(context.serialized_size(schemas))
            elif response.stop_reason == "max_tokens" and response.tool_calls:
                for tool_call in response.tool_calls:
                    context.add_tool_result(
                        tool_call.id,
                        "output token limit reached before this tool call completed",
                        True,
                        tool_schemas=schemas,
                    )
                    context.observe_input_size(context.serialized_size(schemas))

            if response.stop_reason == "end_turn":
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                context.mark_failed("exceeded_max_steps")
            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
