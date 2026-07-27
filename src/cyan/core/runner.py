from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from cyan.core.bus.events import RunFinishedEvent, RunStartedEvent
from cyan.core.compact.compactor import Compactor
from cyan.core.config import CyanConfig
from cyan.core.context import ExecutionContext
from cyan.core.events.bus import EventBus, EventHandler
from cyan.core.events.writer import EventWriter
from cyan.core.llm.base import LLMProvider
from cyan.core.llm.provider import AnthropicProvider
from cyan.core.loop import AgentLoop
from cyan.core.mcp.server import McpServerManager
from cyan.core.memory.loader import load_context_file
from cyan.core.permissions.manager import PermissionManager
from cyan.core.runs import RUNS_DIR, new_run_id
from cyan.core.session.model import Session
from cyan.core.session.store import SessionStore
from cyan.core.subagent.registry import BackgroundTaskRegistry
from cyan.core.subagent.tool import AgentResultTool, SpawnAgentTool
from cyan.core.task.manager import TaskManager
from cyan.core.tools.base import BaseTool
from cyan.core.tools.builtin import (
    BashTool,
    ListDirTool,
    NoteSaveTool,
    ReadFileTool,
    TaskCreateTool,
    TaskGetTool,
    TaskListTool,
    TaskUpdateTool,
    WriteFileTool,
)
from cyan.core.tools.registry import ToolRegistry
from cyan.core.trace.provider import TracingProvider
from cyan.core.trace.writer import TraceWriter


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class RunOutcome:
    status: str
    result: str
    reason: str | None


@dataclass
class RunProfile:
    workspace_root: Path
    system_prompt_override: str | None = None
    tool_whitelist: list[str] | None = None
    extra_tools: list[BaseTool] | None = None
    max_steps: int | None = None
    compact_threshold: float | None = None
    summary_only_events: bool = False
    include_context: bool = True
    evidence_refs: set[str] | None = None


type RunProfileFactory = Callable[[Session], RunProfile | None]


class AgentRunner:
    # 组装所有运行时依赖，准备执行一次完整的 agent run
    def __init__(
        self,
        config: CyanConfig,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        extra_handlers: list[EventHandler] | None = None,
        runs_dir: Path | None = None,
        trace: TraceWriter | None = None,
        permission_manager: PermissionManager | None = None,
        mcp_manager: McpServerManager | None = None,
        profile_factory: RunProfileFactory | None = None,
    ) -> None:
        self._config = config
        self._bus = bus
        self._provider = provider
        self._extra_handlers: list[EventHandler] = extra_handlers or []
        self._runs_dir = runs_dir or RUNS_DIR
        self._trace = trace
        self._permission_manager = permission_manager
        self._mcp_manager = mcp_manager
        self._profile_factory = profile_factory
        # 跨 run 共享的后台 subagent 任务注册表
        self._task_registry = BackgroundTaskRegistry()

    # 构建工具注册表，注入 TaskManager（任务工具共享同一实例）；可选注入 SpawnAgentTool
    def _build_registry(
        self,
        task_manager: TaskManager,
        *,
        session: Session | None = None,
        store: SessionStore | None = None,
        run_id: str | None = None,
        provider: LLMProvider | None = None,
        bus: EventBus | None = None,
        child_runs_dir: Path | None = None,
        session_id: str = "",
        tool_whitelist: list[str] | None = None,
        workspace_root: Path | None = None,
        extra_tools: list[BaseTool] | None = None,
        evidence_refs: set[str] | None = None,
    ) -> ToolRegistry:
        allowed: set[str] | None = (
            set(tool_whitelist) if tool_whitelist is not None else None
        )
        root = (workspace_root or Path.cwd()).resolve()

        def _ok(name: str) -> bool:
            return allowed is None or name in allowed

        registry = ToolRegistry()
        for t in [
            ReadFileTool(root, evidence_refs=evidence_refs),
            BashTool(root),
            WriteFileTool(root),
            ListDirTool(root),
        ]:
            if _ok(t.name):
                registry.register(t)
        for tool in extra_tools or []:
            if _ok(tool.name):
                registry.register(tool)
        for t in [
            TaskCreateTool(task_manager),
            TaskUpdateTool(task_manager),
            TaskListTool(task_manager),
            TaskGetTool(task_manager),
        ]:
            if _ok(t.name):
                registry.register(t)
        if session is not None and store is not None and run_id is not None:
            note_tool = NoteSaveTool(store, session.id, run_id)
            if _ok(note_tool.name):
                registry.register(note_tool)
        if provider is not None and bus is not None and run_id is not None:
            runs_dir = child_runs_dir or self._runs_dir
            if _ok("spawn_agent"):
                registry.register(
                    SpawnAgentTool(
                        provider=provider,
                        parent_bus=bus,
                        parent_run_id=run_id,
                        permission_manager=self._permission_manager,
                        max_steps=self._config.agent.max_steps,
                        task_registry=self._task_registry,
                        runs_dir=runs_dir,
                        session_id=session_id,
                        depth=0,
                    )
                )
            if _ok("agent_result"):
                registry.register(AgentResultTool(self._task_registry))
        if self._mcp_manager is not None:
            for mcp_tool in self._mcp_manager.get_tools():
                if _ok(mcp_tool.name):
                    registry.register(mcp_tool)
        return registry

    # 执行一次完整的 agent run（委托给 run_and_capture，忽略返回值）
    async def run(self, goal: str, *, run_id: str | None = None) -> None:
        await self.run_and_capture(goal, run_id=run_id)

    # 执行 agent run 并返回 RunOutcome（含最终文字结果）
    async def run_and_capture(
        self,
        goal: str,
        *,
        run_id: str | None = None,
        session: Session | None = None,
        store: SessionStore | None = None,
        system_prompt_override: str | None = None,
        tool_whitelist: list[str] | None = None,
        workspace_root: Path | None = None,
        extra_tools: list[BaseTool] | None = None,
        evidence_refs: set[str] | None = None,
        max_steps: int | None = None,
        compact_threshold: float | None = None,
        summary_only_events: bool = False,
        include_context: bool = True,
    ) -> RunOutcome:
        run_id = run_id or new_run_id()
        profile = (
            self._profile_factory(session)
            if session is not None and self._profile_factory is not None
            else None
        )
        if profile is not None:
            workspace_root = workspace_root or profile.workspace_root
            system_prompt_override = (
                system_prompt_override or profile.system_prompt_override
            )
            if tool_whitelist is None:
                tool_whitelist = profile.tool_whitelist
            if extra_tools is None:
                extra_tools = profile.extra_tools
            if evidence_refs is None:
                evidence_refs = profile.evidence_refs
            max_steps = max_steps or profile.max_steps
            if compact_threshold is None:
                compact_threshold = profile.compact_threshold
            summary_only_events = profile.summary_only_events
            include_context = profile.include_context
        if workspace_root is None and session is not None and session.workspace_root:
            workspace_root = Path(session.workspace_root)
        root = (workspace_root or Path.cwd()).resolve()
        if session is not None and store is not None:
            run_path = store.runs_dir(session.id) / run_id
            history = store.read_messages(session.id)
            notes = store.read_notes(session.id) if include_context else ""
        else:
            run_path = self._runs_dir / run_id
            history = [{"role": "user", "content": goal}]
            notes = ""
        run_path.mkdir(parents=True, exist_ok=True)

        global_ctx = (
            load_context_file(Path("~/.cyan/context.md").expanduser())
            if include_context
            else ""
        )
        project_ctx = load_context_file(root / ".cyan/context.md") if include_context else ""

        task_manager = TaskManager(run_path / ".tasks")

        bus = EventBus()

        context = ExecutionContext(
            run_id=run_id,
            goal=goal,
            max_steps=max_steps or self._config.agent.max_steps,
            prefill_messages=history,
            session_notes=notes,
            global_context=global_ctx,
            project_context=project_ctx,
            system_prompt_override=system_prompt_override,
        )
        prefill_len = len(history)

        async with EventWriter(
            run_path / "events.jsonl",
            summary_only=summary_only_events,
        ) as writer:
            writer.subscribe(bus)
            for h in self._extra_handlers:
                bus.subscribe(h)
            if self._bus is not None:
                bus.subscribe(self._bus.publish)
            await bus.publish(RunStartedEvent(run_id=run_id, goal=goal, ts=_now()))

            cancelled = False
            try:
                provider: LLMProvider = self._provider or AnthropicProvider(
                    self._config.llm.default_model
                )
                if self._trace is not None:
                    provider = TracingProvider(
                        provider,
                        self._trace,
                        include_payload=(
                            self._config.trace.include_llm_payload
                            and not summary_only_events
                        ),
                    )
                session_id_str = session.id if session is not None else ""
                child_runs_dir = (
                    store.runs_dir(session.id)
                    if session is not None and store is not None
                    else self._runs_dir
                )
                registry = self._build_registry(
                    task_manager,
                    session=session,
                    store=store,
                    run_id=run_id,
                    provider=provider,
                    bus=bus,
                    child_runs_dir=child_runs_dir,
                    session_id=session_id_str,
                    tool_whitelist=tool_whitelist,
                    workspace_root=root,
                    extra_tools=extra_tools,
                    evidence_refs=evidence_refs,
                )
                session_dir = (
                    store.session_dir(session.id)
                    if session is not None and store is not None
                    else run_path
                )
                compactor = Compactor(bus, session_dir, session_id_str)
                loop = AgentLoop(
                    provider, registry, bus,
                    permission_manager=self._permission_manager,
                    compactor=compactor,
                    compact_threshold=(
                        self._config.compaction.auto_threshold
                        if compact_threshold is None
                        else compact_threshold
                    ),
                    session_id=session_id_str,
                )
                await loop.run(context)
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

            await bus.publish(
                RunFinishedEvent(
                    run_id=run_id,
                    status=context.status,
                    reason=context.reason,
                    steps=context.step,
                    ts=_now(),
                )
            )

        if session is not None and store is not None:
            store.append_messages(session.id, context.messages[prefill_len:], run_id=run_id)

        if cancelled:
            raise asyncio.CancelledError()

        return RunOutcome(
            status=context.status,
            result=context.result,
            reason=context.reason,
        )
