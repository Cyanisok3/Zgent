from __future__ import annotations

from cyan.service.protocol.commands import (
    COMMAND_TYPES,
    WIRE_PROTOCOL_VERSION,
    IncidentFollowUpCommand,
    IncidentFollowUpResult,
)
from cyan.service.protocol.events import EVENT_TYPES


# 功能：验证协议版本和 Incident follow-up 联合类型已进入 v2
# 设计：模型往返与联合类型集合同时断言，避免只改 schema 未注册 handler
def test_protocol_v2_follow_up_is_registered() -> None:
    command = IncidentFollowUpCommand(incident_id="incident-1", content="inspect again")
    result = IncidentFollowUpResult(run_id="run-1")

    assert WIRE_PROTOCOL_VERSION == 2
    assert command.type == "incident.follow_up"
    assert result.model_dump()["run_id"] == "run-1"
    assert "incident.follow_up" in COMMAND_TYPES
    assert "session.create" not in COMMAND_TYPES
    assert "permission.requested" not in EVENT_TYPES


# 功能：验证旧通用 Agent/Session 命令和事件不再出现在协议表面
# 设计：用集合断言协议删除项，防止兼容模型继续被客户端发现
def test_protocol_v2_removes_generic_surface() -> None:
    for method in (
        "agent.run",
        "session.create",
        "session.send_message",
        "session.compact",
        "permission.respond",
    ):
        assert method not in COMMAND_TYPES
    for event in ("context.compacted", "skill.invoked", "subagent.started"):
        assert event not in EVENT_TYPES
