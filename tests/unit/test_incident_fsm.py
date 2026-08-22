from __future__ import annotations

from typing import get_args

import pytest

from cyan.training.incidents.fsm import (
    RECOVERY,
    TERMINAL_STATES,
    TRANSITIONS,
    Event,
    IllegalTransitionError,
    accepts,
    recover_action,
    render_mermaid,
    transition,
)
from cyan.training.incidents.models import IncidentStatus

_ALL_STATES = set(get_args(IncidentStatus))
_ALL_EVENTS = list(Event)


# 功能：验证 TRANSITIONS 中每条声明边都返回正确的目标状态
# 设计：遍历声明表而非手写期望，保证表与 transition() 实现永不漂移
def test_transition_table_edges() -> None:
    for state, edges in TRANSITIONS.items():
        for event, target in edges.items():
            assert transition(state, event) == target


# 功能：验证表未声明的状态/事件组合会抛出 IllegalTransitionError
# 设计：对每个状态枚举其表外事件，确保非法转移 fail-loud 而非静默
def test_transition_rejects_unlisted_events() -> None:
    for state in _ALL_STATES:
        allowed = set(TRANSITIONS.get(state, {}))
        for event in _ALL_EVENTS:
            if event in allowed:
                continue
            with pytest.raises(IllegalTransitionError):
                transition(state, event)


# 功能：验证终态集合正确且对任意事件都拒绝转移
# 设计：显式断言终态集合，防止误把终态重新开放
def test_terminal_states_reject_all_events() -> None:
    assert TERMINAL_STATES == {"resolved", "rejected"}
    for state in TERMINAL_STATES:
        for event in _ALL_EVENTS:
            with pytest.raises(IllegalTransitionError):
                transition(state, event)


# 功能：验证每个状态要么有转移边、要么是终态，且无死状态
# 设计：源状态并集必须等于类型字面量全集，smoke_failed 应已从类型移除
def test_transitions_cover_all_states() -> None:
    assert set(TRANSITIONS) | TERMINAL_STATES == _ALL_STATES
    assert "smoke_failed" not in _ALL_STATES


# 功能：验证每个状态都有恢复策略且取值合法
# 设计：RECOVERY 键集合等于状态全集，避免新增状态时漏配恢复动作
def test_recovery_covers_all_states() -> None:
    assert set(RECOVERY) == _ALL_STATES
    assert set(recover_action(state) for state in _ALL_STATES) <= {
        "keep",
        "reconcile",
        "quarantine",
    }


# 功能：验证 mermaid 渲染结果包含全部状态与初始入口
# 设计：检查每个状态名与初始箭头，保证文档与表同步
def test_render_mermaid_includes_all_states() -> None:
    text = render_mermaid()
    for state in _ALL_STATES:
        assert state in text
    assert "[*] --> diagnosing" in text


# 功能：验证 accepts 与 transition 对所有组合保持一致
# 设计：accepts 为真当且仅当 transition 不抛异常，保证前置校验与校验语义同源
def test_accepts_matches_transition() -> None:
    for state in _ALL_STATES:
        for event in _ALL_EVENTS:
            assert accepts(state, event) == (
                event in TRANSITIONS.get(state, {})
            )
            if accepts(state, event):
                transition(state, event)
            else:
                with pytest.raises(IllegalTransitionError):
                    transition(state, event)
