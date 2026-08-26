from __future__ import annotations

import asyncio
import logging
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cyan.agent.context import ExecutionContext
from cyan.agent.events.bus import EventBus, EventHandler
from cyan.agent.events.models import RunFinishedEvent, RunStartedEvent
from cyan.agent.events.writer import EventWriter
from cyan.agent.llm.base import LLMProvider
from cyan.agent.llm.provider import AnthropicProvider
from cyan.agent.loop import AgentLoop
from cyan.agent.tools.base import BaseTool
from cyan.agent.tools.registry import ToolRegistry
from cyan.agent.trace.provider import TracingProvider
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig


# 返回当前 UTC 时间字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None
    initial_input_bytes: int = 0
    peak_input_bytes: int = 0
    budget_exhausted: bool = False


@dataclass
class RunProfile:
    workspace_root: Path
    system_prompt: str
    tools: list[BaseTool]
    max_steps: int
    max_input_bytes: int
    summary_only_events: bool = True
    # 由业务 profile 指定收尾工具顺序，空序列保持通用 Agent 原语义
    finalization_sequence: tuple[str, ...] = ()
    # 为诊断和提案结果保留的软输入预算，零表示不启用
    finalization_reserve_bytes: int = 0
    # 触发收尾阶段时允许使用的最后步数，零表示不启用
    finalization_steps: int = 0


class AgentRunner:
    # 只按显式 Incident profile 组装一次运行，不加载任何通用 Agent 扩展
    def __init__(
        self,
        config: CyanConfig,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: Iterable[EventHandler] = (),
        trace: TraceWriter | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._provider = provider
        self._extra_handlers = list(extra_handlers)
        self._trace = trace

    # 执行一轮受 profile 限制的 Agent 并将事件写入指定 run 目录
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str,
        profile: RunProfile,
        run_path: Path,
    ) -> RunOutcome:
        run_path.mkdir(parents=True, exist_ok=True)
        local_bus = EventBus()
        async with EventWriter(
            run_path / "events.jsonl",
            summary_only=profile.summary_only_events,
        ) as writer:
            writer.subscribe(local_bus)
            for handler in self._extra_handlers:
                local_bus.subscribe(handler)
            if self._bus is not None:
                local_bus.subscribe(self._bus.publish)
            await local_bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))
            context = ExecutionContext(
                run_id=run_id,
                goal=goal,
                max_steps=profile.max_steps,
                system_prompt_text=profile.system_prompt,
                max_input_bytes=profile.max_input_bytes,
            )
            registry = ToolRegistry()
            for tool in profile.tools:
                registry.register(tool)
            cancelled = False
            try:
                provider = self._provider or AnthropicProvider(self._config.llm.default_model)
                if self._trace is not None:
                    provider = TracingProvider(
                        provider,
                        self._trace,
                        include_payload=False,
                    )
                await AgentLoop(
                    provider,
                    registry,
                    local_bus,
                    finalization_sequence=profile.finalization_sequence,
                    finalization_reserve_bytes=profile.finalization_reserve_bytes,
                    finalization_steps=profile.finalization_steps,
                ).run(context)
            except asyncio.CancelledError:
                cancelled = True
                if not context.is_done():
                    context.mark_failed("cancelled")
            except Exception:
                logging.getLogger(__name__).exception(
                    "agent run failed run_id=%s step=%d", run_id, context.step
                )
                if not context.is_done():
                    context.mark_failed("llm_error")
            await local_bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )
        if cancelled:
            raise asyncio.CancelledError()
        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
            initial_input_bytes=context.initial_input_bytes,
            peak_input_bytes=context.peak_input_bytes,
            budget_exhausted=(
                context.budget_exhausted or context.reason == "context_budget_exhausted"
            ),
        )

    # 提供最小的同步语义包装，调用方必须显式传入 profile 和 run 目录
    async def run(
        self,
        goal: str,
        *,
        run_id: str,
        profile: RunProfile,
        run_path: Path,
    ) -> RunOutcome:
        return await self.run_and_capture(
            goal,
            run_id=run_id,
            profile=profile,
            run_path=run_path,
        )
