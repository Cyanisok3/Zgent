"""Incident 状态机的纯声明式定义。

本模块是 Incident 状态流转的单一事实源：只依赖标准库与 models.py 的
``IncidentStatus``，不 import store/bus/session，不产生任何 I/O。运行时校验、
mermaid 文档与单元测试都从同一张 ``TRANSITIONS`` 表派生，保证三者永不漂移。
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from cyan.core.incidents.models import IncidentStatus


# 触发状态迁移的事件；调用方只发事件，机器计算目标状态
class Event(StrEnum):
    INVESTIGATION_DONE = "INVESTIGATION_DONE"
    INVESTIGATION_FAILED = "INVESTIGATION_FAILED"
    APPROVE = "APPROVE"
    APPROVE_INVALIDATED = "APPROVE_INVALIDATED"
    REJECT = "REJECT"
    FOLLOW_UP = "FOLLOW_UP"
    APPLY_OK_SMOKE = "APPLY_OK_SMOKE"
    APPLY_OK_NO_SMOKE = "APPLY_OK_NO_SMOKE"
    APPLY_FAILED = "APPLY_FAILED"
    SMOKE_PASSED = "SMOKE_PASSED"
    SMOKE_FAILED_ROLLED_BACK = "SMOKE_FAILED_ROLLED_BACK"
    SMOKE_FAILED_ROLLBACK_BLOCKED = "SMOKE_FAILED_ROLLBACK_BLOCKED"
    RETRY_STARTED = "RETRY_STARTED"
    RETRY_SUCCEEDED = "RETRY_SUCCEEDED"
    RETRY_ABORTED = "RETRY_ABORTED"
    RETRY_FAILED = "RETRY_FAILED"


# 新建 Incident 的初始状态
INITIAL_STATE: IncidentStatus = "diagnosing"

# 不再接受任何运行时事件的终态
TERMINAL_STATES: frozenset[IncidentStatus] = frozenset({"resolved", "rejected"})

# 合法转移表：源状态 -> 事件 -> 目标状态。
# smoke 失败事件同时允许从 smoke_running 与 applying 出发，因为 smoke 可能
# 在启动阶段就失败（on_started 从未触发），此时 Incident 仍停留在 applying。
TRANSITIONS: dict[IncidentStatus, dict[Event, IncidentStatus]] = {
    "diagnosing": {
        Event.INVESTIGATION_DONE: "awaiting_approval",
        Event.INVESTIGATION_FAILED: "unresolved",
    },
    "awaiting_approval": {
        Event.APPROVE: "applying",
        Event.APPROVE_INVALIDATED: "stale",
        Event.REJECT: "rejected",
        Event.FOLLOW_UP: "diagnosing",
    },
    "applying": {
        Event.APPLY_OK_SMOKE: "smoke_running",
        Event.APPLY_OK_NO_SMOKE: "smoke_skipped",
        Event.APPLY_FAILED: "stale",
        Event.SMOKE_FAILED_ROLLED_BACK: "diagnosing",
        Event.SMOKE_FAILED_ROLLBACK_BLOCKED: "rollback_blocked",
    },
    "smoke_running": {
        Event.SMOKE_PASSED: "smoke_passed",
        Event.SMOKE_FAILED_ROLLED_BACK: "diagnosing",
        Event.SMOKE_FAILED_ROLLBACK_BLOCKED: "rollback_blocked",
    },
    "smoke_passed": {
        Event.RETRY_STARTED: "retry_running",
    },
    "smoke_skipped": {
        Event.RETRY_STARTED: "retry_running",
    },
    "retry_running": {
        Event.RETRY_SUCCEEDED: "resolved",
        Event.RETRY_ABORTED: "unresolved",
        Event.RETRY_FAILED: "diagnosing",
    },
    "stale": {
        Event.FOLLOW_UP: "diagnosing",
    },
    "unresolved": {
        Event.FOLLOW_UP: "diagnosing",
    },
    "rollback_blocked": {
        Event.FOLLOW_UP: "diagnosing",
    },
}

# daemon 重启后每个状态应采取的恢复动作
RecoveryAction = Literal["keep", "reconcile", "quarantine"]

RECOVERY: dict[IncidentStatus, RecoveryAction] = {
    "diagnosing": "reconcile",
    "awaiting_approval": "reconcile",
    "applying": "quarantine",
    "smoke_running": "quarantine",
    "smoke_passed": "quarantine",
    "smoke_skipped": "quarantine",
    "retry_running": "reconcile",
    "resolved": "keep",
    "rejected": "keep",
    "stale": "keep",
    "unresolved": "keep",
    "rollback_blocked": "keep",
}


class IllegalTransitionError(ValueError):
    """在状态不接受该事件时抛出。"""


# 判断状态是否接受该事件，不抛异常，供调用方在副作用前做前置校验
def accepts(state: IncidentStatus, event: Event) -> bool:
    return event in TRANSITIONS.get(state, {})


# 校验并返回事件在给定状态下的目标状态，非法则抛 IllegalTransitionError
def transition(state: IncidentStatus, event: Event) -> IncidentStatus:
    allowed = TRANSITIONS.get(state)
    if allowed is None or event not in allowed:
        raise IllegalTransitionError(
            f"illegal transition: {event.value} from {state}"
        )
    return allowed[event]


# 返回状态在 daemon 重启后的恢复动作
def recover_action(state: IncidentStatus) -> RecoveryAction:
    return RECOVERY[state]


# 从 TRANSITIONS 生成确定性 mermaid stateDiagram-v2 文本
def render_mermaid() -> str:
    lines = ["stateDiagram-v2", f"    [*] --> {INITIAL_STATE}"]
    for state in sorted(TRANSITIONS):
        for event in sorted(TRANSITIONS[state], key=lambda item: item.value):
            target = TRANSITIONS[state][event]
            lines.append(f"    {state} --> {target}: {event.value}")
    for state in sorted(TERMINAL_STATES):
        lines.append(f"    {state} --> [*]")
    return "\n".join(lines)
