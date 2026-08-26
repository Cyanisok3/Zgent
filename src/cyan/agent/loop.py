from __future__ import annotations

import asyncio
import json
import logging
from datetime import UTC, datetime

from cyan.agent.context import ExecutionContext
from cyan.agent.events.bus import EventBus
from cyan.agent.events.models import StepFinishedEvent, StepStartedEvent
from cyan.agent.llm.base import LLMProvider
from cyan.agent.tools.base import ToolResult
from cyan.agent.tools.invocation import invoke_tool
from cyan.agent.tools.registry import ToolRegistry

log = logging.getLogger(__name__)
_READ_ONLY_TOOLS = frozenset({"read_file", "list_dir", "search_text", "read_job_log"})


# 返回当前 UTC 时间字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


class AgentLoop:
    # 初始化极简 LLM、工具和事件循环
    def __init__(
        self,
        provider: LLMProvider,
        registry: ToolRegistry,
        bus: EventBus,
        *,
        finalization_sequence: tuple[str, ...] = (),
        finalization_reserve_bytes: int = 0,
        finalization_steps: int = 0,
    ) -> None:
        self._provider = provider
        self._registry = registry
        self._bus = bus
        self._finalization_sequence = finalization_sequence
        self._finalization_reserve_bytes = max(0, finalization_reserve_bytes)
        self._finalization_steps = max(0, finalization_steps)

    # 返回当前收尾阶段只允许调用的工具名称
    def _finalization_tool(
        self,
        finalization_active: bool,
        required_tool: str | None,
    ) -> str | None:
        if not finalization_active:
            return None
        return required_tool

    # 构造不触发真实工具执行的阶段错误结果
    def _stage_error(self, tool_name: str, required_tool: str | None) -> ToolResult:
        return ToolResult(
            content=json.dumps(
                {
                    "error": "finalization_tool_required",
                    "required_tool": required_tool,
                    "received_tool": tool_name,
                },
                separators=(",", ":"),
            ),
            is_error=True,
            error_type="stage_error",
        )

    # 让收尾阶段推进到序列中的下一个工具
    def _advance_finalization(
        self,
        tool_name: str,
    ) -> tuple[bool, str | None]:
        try:
            index = self._finalization_sequence.index(tool_name)
        except ValueError:
            return False, None
        next_index = index + 1
        if next_index >= len(self._finalization_sequence):
            return True, None
        return True, self._finalization_sequence[next_index]

    # 执行 LLM → tool → observe 循环，直到模型结束或达到硬步数上限
    async def run(self, context: ExecutionContext) -> None:
        finalization_active = False
        required_tool: str | None = None
        soft_max_input_bytes = (
            context.max_input_bytes - self._finalization_reserve_bytes
            if self._finalization_sequence and self._finalization_reserve_bytes > 0
            else None
        )
        while not context.is_done():
            if (
                not finalization_active
                and self._finalization_sequence
                and self._finalization_steps > 0
                and context.max_steps - context.step <= self._finalization_steps
            ):
                finalization_active = True
                required_tool = self._finalization_sequence[0]
            context.step += 1
            await self._bus.publish(
                StepStartedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
            finalization_tool = self._finalization_tool(finalization_active, required_tool)
            schemas = self._registry.tool_schemas(
                [finalization_tool] if finalization_tool is not None else None
            )
            input_size = context.serialized_size(schemas)
            if (
                not finalization_active
                and soft_max_input_bytes is not None
                and input_size > soft_max_input_bytes
            ):
                finalization_active = True
                required_tool = self._finalization_sequence[0]
                finalization_tool = self._finalization_tool(
                    finalization_active,
                    required_tool,
                )
                assert finalization_tool is not None
                schemas = self._registry.tool_schemas([finalization_tool])
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
            required_called = False
            required_succeeded = False
            if response.stop_reason == "tool_use":
                for tool_call in response.tool_calls:
                    if context.is_done():
                        break
                    if finalization_tool is not None and tool_call.name != finalization_tool:
                        result = self._stage_error(tool_call.name, finalization_tool)
                    else:
                        required_called = (
                            finalization_tool is not None
                            and tool_call.name == finalization_tool
                        )
                        result = await invoke_tool(
                            self._registry,
                            tool_call,
                            self._bus,
                            context.run_id,
                        )
                    reserve_hit = context.add_tool_result(
                        tool_call.id,
                        result.content,
                        result.is_error,
                        tool_schemas=schemas,
                        soft_max_input_bytes=(
                            soft_max_input_bytes
                            if tool_call.name in _READ_ONLY_TOOLS
                            else None
                        ),
                    )
                    context.observe_input_size(context.serialized_size(schemas))
                    if reserve_hit:
                        finalization_active = True
                        required_tool = (
                            self._finalization_sequence[0]
                            if self._finalization_sequence
                            else None
                        )
                        finalization_tool = self._finalization_tool(
                            finalization_active,
                            required_tool,
                        )
                        required_called = False
                        required_succeeded = False
                    if (
                        finalization_tool is not None
                        and tool_call.name == finalization_tool
                        and not result.is_error
                    ):
                        required_called = True
                        required_succeeded = True
                    if not result.is_error and result.stop_after_success:
                        context.result = result.content
                        context.mark_success()
                        break
                    if not result.is_error and tool_call.name == "submit_diagnosis":
                        advanced, next_tool = self._advance_finalization(tool_call.name)
                        if advanced and next_tool is not None:
                            finalization_active = True
                            required_tool = next_tool
                            finalization_tool = next_tool
                            required_called = False
                            required_succeeded = False
                        elif advanced:
                            finalization_active = True
                            required_tool = None
                            finalization_tool = None
            elif response.stop_reason == "max_tokens" and response.tool_calls:
                for tool_call in response.tool_calls:
                    if context.is_done():
                        break
                    reserve_hit = context.add_tool_result(
                        tool_call.id,
                        "output token limit reached before this tool call completed",
                        True,
                        tool_schemas=schemas,
                        soft_max_input_bytes=(
                            soft_max_input_bytes
                            if tool_call.name in _READ_ONLY_TOOLS
                            else None
                        ),
                    )
                    context.observe_input_size(context.serialized_size(schemas))
                    if reserve_hit:
                        finalization_active = True
                        required_tool = (
                            self._finalization_sequence[0]
                            if self._finalization_sequence
                            else None
                        )

            if context.is_done():
                pass
            elif response.stop_reason == "end_turn" and finalization_tool is not None:
                if context.step < context.max_steps:
                    context.add_user_message(
                        json.dumps(
                            {
                                "error": "finalization_tool_required",
                                "required_tool": finalization_tool,
                            },
                            separators=(",", ":"),
                        )
                    )
                else:
                    context.mark_failed("finalization_tool_missing")
            elif response.stop_reason == "end_turn":
                context.result = response.text or ""
                context.mark_success()
            elif context.step >= context.max_steps:
                if finalization_tool is not None and not required_succeeded:
                    context.mark_failed("finalization_tool_missing")
                else:
                    context.mark_failed("exceeded_max_steps")
            elif finalization_tool is not None and required_called and not required_succeeded:
                # 保持当前收尾阶段，给工具校验失败一次修正机会
                required_tool = finalization_tool
            await self._bus.publish(
                StepFinishedEvent(run_id=context.run_id, step=context.step, ts=_now())
            )
