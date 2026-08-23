import json
from pathlib import Path

import pytest

from cyan.agent.events.models import (
    LlmTokenEvent,
    RunStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)
from cyan.agent.trace.record import TraceRecord
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig
from cyan.service.app import _config_log_data, _trace_event_data


# 构造最小 trace 记录供写入测试复用
def _record(direction: str = "CORE", kind: str = "event") -> TraceRecord:
    return TraceRecord(
        ts="2026-01-01T00:00:00.000Z",
        direction=direction,  # type: ignore[arg-type]
        layer="event",
        kind=kind,
        data={"type": "run.started", "run_id": "r1"},
    )


# 功能：验证 emit 后 stop 能将 record 写入文件
# 设计：用临时目录隔离文件副作用，并等待 writer drain 完成
@pytest.mark.asyncio
async def test_emit_writes_record_to_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    await writer.start()
    writer.emit(_record())
    await writer.stop()

    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert parsed["direction"] == "CORE"
    assert parsed["kind"] == "event"


# 功能：验证多次重启以追加模式保留已有 trace
# 设计：两轮独立 writer 写入同一文件，确保诊断摘要不会被覆盖
@pytest.mark.asyncio
async def test_writer_restarts_in_append_mode(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    for _ in range(2):
        writer = TraceWriter(path)
        await writer.start()
        writer.emit(_record())
        await writer.stop()

    assert len(path.read_text(encoding="utf-8").splitlines()) == 2


# 功能：验证 Agent 事件摘要不会复制工具参数、工具输出或 token 原文
# 设计：用同一个敏感标记覆盖三类内容，断言只保留键名和长度
def test_trace_event_data_redacts_payloads() -> None:
    secret = "INCIDENT-RAW-EVIDENCE-秘密"
    events = [
        ToolCallStartedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_file",
            params={"path": secret},
            ts="2026-01-01T00:00:00Z",
        ),
        ToolCallFinishedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_file",
            elapsed_ms=1,
            output=secret,
            ts="2026-01-01T00:00:00Z",
        ),
        ToolCallFailedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_job_log",
            error_class="runtime_error",
            error_message=secret,
            elapsed_ms=1,
            ts="2026-01-01T00:00:00Z",
        ),
        LlmTokenEvent(run_id="run-1", token=secret, ts="2026-01-01T00:00:00Z"),
    ]

    traced = [_trace_event_data(event) for event in events]
    encoded = json.dumps(traced, ensure_ascii=False)

    assert secret not in encoded
    assert traced[0]["params_keys"] == ["path"]
    assert traced[1]["output_chars"] == len(secret)
    assert traced[2]["error_message_bytes"] == len(secret.encode("utf-8"))
    assert traced[3]["token_bytes"] == len(secret.encode("utf-8"))


# 功能：验证运行目标摘要也只记录长度而不记录用户文本
# 设计：检查 run.started 和普通配置摘要的最小字段集合
def test_trace_goal_and_config_summary_are_bounded() -> None:
    secret = "long incident instruction"
    traced = _trace_event_data(
        RunStartedEvent(run_id="run-1", goal=secret, ts="2026-01-01T00:00:00Z")
    )
    config = _config_log_data(CyanConfig())

    assert "goal" not in traced
    assert traced["goal_chars"] == len(secret)
    assert set(config) == {
        "host",
        "port",
        "log_level",
        "log_format",
        "model",
        "trace_enabled",
    }
