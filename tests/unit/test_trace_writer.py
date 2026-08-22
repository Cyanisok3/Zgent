import json
from pathlib import Path

import pytest

from cyan.agent.trace.record import TraceRecord
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig, McpServerConfig
from cyan.service.app import _config_log_data, _trace_event_data
from cyan.service.protocol.events import (
    LlmTokenEvent,
    LogLineEvent,
    PatchProposedEvent,
    PermissionRequestedEvent,
    RunStartedEvent,
    SessionMessageReceivedEvent,
    SkillInvokedEvent,
    SubagentStartedEvent,
    ToolCallFailedEvent,
    ToolCallFinishedEvent,
    ToolCallStartedEvent,
)


def _record(direction: str = "CORE", kind: str = "event") -> TraceRecord:
    return TraceRecord(
        ts="2026-01-01T00:00:00.000Z",
        direction=direction,  # type: ignore[arg-type]
        layer="event",
        kind=kind,
        data={"type": "run.started", "run_id": "r1"},
    )


# 功能：验证 emit 后 stop 能将 record 写入文件
# 设计：用临时目录避免污染；await stop() 保证 drain 完成后再读文件
@pytest.mark.asyncio
async def test_emit_writes_record_to_file(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    await writer.start()

    writer.emit(_record())
    await writer.stop()

    lines = path.read_text().splitlines()
    assert len(lines) == 1
    parsed = json.loads(lines[0])
    assert parsed["direction"] == "CORE"
    assert parsed["kind"] == "event"


# 功能：验证多条 record 按 emit 顺序写入文件
# 设计：emit 三条方向各异的 record，断言顺序与方向均保持一致
@pytest.mark.asyncio
async def test_emit_multiple_records_in_order(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    await writer.start()

    writer.emit(_record("CLIENT→CORE", "command"))
    writer.emit(_record("CORE", "event"))
    writer.emit(_record("LLM→CORE", "api_response"))
    await writer.stop()

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["direction"] == "CLIENT→CORE"
    assert json.loads(lines[1])["direction"] == "CORE"
    assert json.loads(lines[2])["direction"] == "LLM→CORE"


# 功能：验证 emit 是同步非阻塞的（不需要 await）
# 设计：在 start() 之前调用 emit 会放入队列而不抛异常，start 后正常 drain
@pytest.mark.asyncio
async def test_emit_is_nonblocking(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    writer = TraceWriter(path)
    await writer.start()

    # emit 是同步调用，不应阻塞事件循环
    for _ in range(10):
        writer.emit(_record())
    await writer.stop()

    assert len(path.read_text().splitlines()) == 10


# 功能：验证 TraceWriter 自动创建不存在的父目录
# 设计：指定一个深层嵌套路径，start() 后 emit 能正常写入
@pytest.mark.asyncio
async def test_start_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "a" / "b" / "c" / "trace.jsonl"
    writer = TraceWriter(path)
    await writer.start()
    writer.emit(_record())
    await writer.stop()

    assert path.exists()
    assert len(path.read_text().splitlines()) == 1


# 功能：验证 stop 后再次 start 可以追加写入（文件已存在时）
# 设计：两次 start/stop 循环，断言文件行数累加而非覆盖
@pytest.mark.asyncio
async def test_append_mode_on_restart(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"

    writer = TraceWriter(path)
    await writer.start()
    writer.emit(_record())
    await writer.stop()

    writer2 = TraceWriter(path)
    await writer2.start()
    writer2.emit(_record())
    await writer2.stop()

    assert len(path.read_text().splitlines()) == 2


# 功能：验证 daemon event trace 不复制 Incident 工具参数、输出或失败消息
# 设计：把唯一敏感标记分别放入三类工具事件，再断言 trace 仅保留键名、状态与尺寸摘要
def test_tool_event_trace_redacts_payloads() -> None:
    secret = "INCIDENT-RAW-EVIDENCE-秘密"
    started = _trace_event_data(
        ToolCallStartedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="propose_patch",
            params={"patch": secret},
            ts="2026-01-01T00:00:00Z",
        )
    )
    finished = _trace_event_data(
        ToolCallFinishedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_file",
            elapsed_ms=1,
            output=secret,
            ts="2026-01-01T00:00:00Z",
        )
    )
    failed = _trace_event_data(
        ToolCallFailedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_job_log",
            error_class="runtime_error",
            error_message=secret,
            elapsed_ms=1,
            ts="2026-01-01T00:00:00Z",
        )
    )

    encoded = json.dumps([started, finished, failed], ensure_ascii=False)
    assert secret not in encoded
    assert started["param_keys"] == ["patch"]
    assert started["params_chars"] > 0
    assert started["params_bytes"] >= started["params_chars"]
    assert finished["output_chars"] == len(secret)
    assert finished["output_bytes"] == len(secret.encode())
    assert failed["error_message_chars"] == len(secret)


# 功能：验证其他内容型 EventBus 事件进入 daemon trace 前也统一移除原文
# 设计：用同一敏感标记覆盖 token、goal、日志、消息、权限、subagent、skill 和 proposal 摘要
def test_content_event_trace_redacts_text_and_permission_params() -> None:
    secret = "HIGH-VOLUME-INCIDENT-CONTENT-秘密"
    timestamp = "2026-01-01T00:00:00Z"
    events = [
        RunStartedEvent(run_id="run-1", goal=secret, ts=timestamp),
        LlmTokenEvent(run_id="run-1", token=secret, ts=timestamp),
        LogLineEvent(
            run_id="run-1",
            level="INFO",
            source="incident",
            message=secret,
            ts=timestamp,
        ),
        SessionMessageReceivedEvent(session_id="session-1", content=secret, ts=timestamp),
        PermissionRequestedEvent(
            run_id="run-1",
            tool_use_id="tool-1",
            tool_name="read_file",
            params={"path": secret},
            param_preview=secret,
            session_id="session-1",
            ts=timestamp,
        ),
        SubagentStartedEvent(
            run_id="run-child",
            parent_run_id="run-1",
            description=secret,
            ts=timestamp,
        ),
        SkillInvokedEvent(
            skill_name="review",
            arguments=secret,
            run_id="run-1",
            ts=timestamp,
        ),
        PatchProposedEvent(
            job_id="job-1",
            incident_id="incident-1",
            proposal_id="proposal-1",
            summary=secret,
            ts=timestamp,
        ),
    ]

    traced = [_trace_event_data(event) for event in events]
    encoded = json.dumps(traced, ensure_ascii=False)

    assert secret not in encoded
    assert traced[0]["goal_chars"] == len(secret)
    assert traced[1]["token_bytes"] == len(secret.encode())
    assert traced[4]["param_keys"] == ["path"]
    assert traced[4]["params_bytes"] > len(secret)
    assert traced[4]["param_preview_chars"] == len(secret)
    assert traced[-1]["summary_chars"] == len(secret)


# 功能：验证 daemon 启动配置日志不会序列化 MCP 环境变量中的凭据
# 设计：向配置放入唯一 token，再检查安全摘要只保留 server 数量与非敏感运行字段
def test_config_log_summary_excludes_mcp_secrets() -> None:
    secret = "MCP-TOKEN-MUST-NOT-BE-LOGGED"
    config = CyanConfig()
    config.mcp.servers.append(
        McpServerConfig(
            name="private",
            command="secret-command",
            args=["--token", secret],
            env={"API_TOKEN": secret},
        )
    )

    summary = _config_log_data(config)

    assert secret not in json.dumps(summary)
    assert "secret-command" not in json.dumps(summary)
    assert summary["mcp_server_count"] == 1
