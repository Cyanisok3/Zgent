from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.types import LlmResponse, ToolCallBlock
from cyan.agent.runner import AgentRunner, RunProfile
from cyan.config import CyanConfig


class _EndTurnProvider:
    # 返回固定 end_turn，隔离 Runner 对真实模型 SDK 的依赖
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, tool_schemas, bus, run_id, step, system
        return LlmResponse(stop_reason="end_turn", text="done")


class _LoopingProvider:
    # 始终请求不存在的工具，以覆盖 max_steps 硬上限
    def __init__(self) -> None:
        self.calls = 0

    # 返回不可注册工具的确定性调用
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del messages, tool_schemas, bus, run_id, step, system
        self.calls += 1
        return LlmResponse(
            stop_reason="tool_use",
            tool_calls=[ToolCallBlock(id=f"t{self.calls}", name="unknown_tool", input={})],
        )


# 构造显式 Incident profile，Runner 不会从配置隐式加载工具
def _profile(tmp_path: Path, *, max_steps: int = 5) -> RunProfile:
    return RunProfile(
        workspace_root=tmp_path,
        system_prompt="incident prompt",
        tools=[],
        max_steps=max_steps,
        max_input_bytes=128 * 1024,
    )


# 执行一轮最小 Runner 并收集外部事件
async def _run(
    tmp_path: Path,
    *,
    run_id: str = "run-1",
    provider: Any | None = None,
    max_steps: int = 5,
) -> list[BaseModel]:
    events: list[BaseModel] = []

    # 保存外部可观察事件，避免读取内部 EventBus
    async def collect(event: BaseModel) -> None:
        events.append(event)

    runner = AgentRunner(
        CyanConfig(),
        provider=provider or _EndTurnProvider(),
        extra_handlers=[collect],
    )
    await runner.run_and_capture(
        "test goal",
        run_id=run_id,
        profile=_profile(tmp_path, max_steps=max_steps),
        run_path=tmp_path / run_id,
    )
    return events


# 功能：验证 Runner 发布 run.started 和 run.finished，并写入摘要事件文件
# 设计：通过显式 profile 和临时 run 目录覆盖最短成功路径
async def test_runner_publishes_lifecycle_and_persists_events(tmp_path: Path) -> None:
    events = await _run(tmp_path)
    types = [str(event.model_dump()["type"]) for event in events]
    assert types[0] == "run.started"
    assert types[-1] == "run.finished"
    assert events[0].model_dump()["goal"] == "test goal"
    rows = [json.loads(line) for line in (tmp_path / "run-1" / "events.jsonl").read_text().splitlines()]
    assert rows[0]["type"] == "run.started"
    assert rows[-1]["type"] == "run.finished"


# 功能：验证无限工具调用在 profile max_steps 达到后失败且原因可审计
# 设计：使用未注册工具，不引入任何业务工具或权限模块
async def test_runner_honors_profile_max_steps(tmp_path: Path) -> None:
    provider = _LoopingProvider()
    events = await _run(tmp_path, provider=provider, max_steps=3)
    finished = events[-1].model_dump()
    assert finished["type"] == "run.finished"
    assert finished["status"] == "failed"
    assert finished["reason"] == "exceeded_max_steps"
    assert provider.calls == 3


# 功能：验证未传入工具时 Registry 保持为空
# 设计：让 provider 捕获 schemas，确保 Runner 不隐式注册 Bash、Write 或其他工具
async def test_runner_registers_only_explicit_tools(tmp_path: Path) -> None:
    seen: list[list[dict[str, object]]] = []

    class Provider(_EndTurnProvider):
        # 记录本轮传入的工具 schema
        async def chat(self, *args: Any, **kwargs: Any) -> LlmResponse:
            seen.append(list(kwargs.get("tool_schemas", [])))
            return await super().chat(*args, **kwargs)

    await _run(tmp_path, provider=Provider())
    assert seen == [[]]


# 功能：验证并发运行各自写入独立 run 目录且不串流事件
# 设计：共享 Runner 但传入两个 run_id，覆盖 Incident 订阅回放的文件边界
async def test_runner_keeps_concurrent_event_files_isolated(tmp_path: Path) -> None:
    runner = AgentRunner(CyanConfig(), provider=_EndTurnProvider())
    await asyncio.gather(
        runner.run_and_capture(
            "first",
            run_id="run-first",
            profile=_profile(tmp_path),
            run_path=tmp_path / "run-first",
        ),
        runner.run_and_capture(
            "second",
            run_id="run-second",
            profile=_profile(tmp_path),
            run_path=tmp_path / "run-second",
        ),
    )

    for run_id in ("run-first", "run-second"):
        rows = [
            json.loads(line)
            for line in (tmp_path / run_id / "events.jsonl").read_text().splitlines()
        ]
        assert {row["run_id"] for row in rows} == {run_id}
