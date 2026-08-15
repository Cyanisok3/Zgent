from __future__ import annotations

import pytest
from pydantic import ValidationError

from cyan.core.bus.commands import (
    WIRE_PROTOCOL_VERSION,
    CoreShutdownCommand,
    CoreShutdownResult,
    IncidentRetryCommand,
    PingCommand,
    PongResult,
)
from cyan.core.bus.events import CoreStartedEvent, JobPhaseChangedEvent


# 功能：验证 PingCommand 序列化后再反序列化，client 和 type 字段完整保留
# 设计：JSON 往返测试确认 wire 协议的序列化正确性，type 字段是 discriminated union 的判别键
def test_ping_command_roundtrip() -> None:
    cmd = PingCommand(client="cli/0.0.1")
    cmd2 = PingCommand.model_validate_json(cmd.model_dump_json())
    assert cmd2.client == "cli/0.0.1"
    assert cmd2.type == "core.ping"


# 功能：验证 PingCommand 的 type 字段默认值为 "core.ping"
# 设计：Literal 默认值测试，type 是 Command union 的判别键，必须与 union 定义完全一致，否则反序列化时会路由到错误类型
def test_ping_command_default_type() -> None:
    cmd = PingCommand(client="x")
    assert cmd.type == "core.ping"


# 功能：验证缺少必填 client 字段时 pydantic 校验失败
# 设计：传入空 dict 触发校验，确认 client 是必填字段，防止 daemon 收到不完整的 ping 命令进入 handler
def test_ping_command_missing_client_raises() -> None:
    with pytest.raises(ValidationError):
        PingCommand.model_validate({})


# 功能：验证 PongResult 序列化往返后所有字段完整保留
# 设计：与 PingCommand 对称，测试命令-响应对的两端序列化，确认 int 和 str 字段类型在往返中不变
def test_pong_result_roundtrip() -> None:
    pong = PongResult(
        server_version="0.0.1",
        protocol_version=1,
        startup_workspace_root="/tmp/project",
        uptime_ms=42,
        received_at="2026-05-11T00:00:00Z",
    )
    pong2 = PongResult.model_validate(pong.model_dump())
    assert pong2.server_version == "0.0.1"
    assert pong2.protocol_version == 1
    assert pong2.startup_workspace_root == "/tmp/project"
    assert pong2.uptime_ms == 42


# 功能：验证 daemon 停止命令与响应使用无 PID 的结构化协议
# 设计：对空参数命令和 stopping 结果做模型往返，锁定 core.shutdown 的最小 wire contract
def test_core_shutdown_models_roundtrip() -> None:
    command = CoreShutdownCommand.model_validate({})
    result = CoreShutdownResult.model_validate({})

    assert command.type == "core.shutdown"
    assert result.status == "stopping"


# 功能：验证 CoreStartedEvent 序列化往返后 listen_addr 和 type 字段正确保留
# 设计：CoreStartedEvent 是 daemon 启动通知，往返测试确认 type 的 Literal 约束在反序列化后保持（不被字段名覆盖）
def test_core_started_event_roundtrip() -> None:
    evt = CoreStartedEvent(listen_addr="127.0.0.1:7437", version="0.0.1")
    evt2 = CoreStartedEvent.model_validate_json(evt.model_dump_json())
    assert evt2.listen_addr == "127.0.0.1:7437"
    assert evt2.type == "core.started"


# 功能：验证 v2 新增 retry 命令和 phase event 的稳定判别字段
# 设计：直接做 JSON 往返并锁定 Python wire 常量
def test_workflow_v2_wire_models_roundtrip() -> None:
    retry = IncidentRetryCommand(incident_id="incident-1")
    phase = JobPhaseChangedEvent(
        job_id="job-1",
        attempt_id="attempt-1",
        phase="preflight",
        check_id="schema",
        ts="2026-05-11T00:00:00Z",
    )

    assert WIRE_PROTOCOL_VERSION == 2
    assert IncidentRetryCommand.model_validate_json(retry.model_dump_json()).type == (
        "incident.retry"
    )
    restored = JobPhaseChangedEvent.model_validate_json(phase.model_dump_json())
    assert (restored.phase, restored.check_id) == ("preflight", "schema")
