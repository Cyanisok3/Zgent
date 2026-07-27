from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from cyan.tui.app import (
    CyanTuiApp,
    _actionable_jobs,
    _approval_hint,
    _diagnosis_body,
    _elapsed_label,
    _log_result,
)


# 功能：验证自动附着只考虑运行中或仍有待处理 Incident 的任务
# 设计：混合成功、运行和待审批记录并检查时间倒序，覆盖选择器出现前的筛选边界
def test_actionable_jobs_filters_and_orders() -> None:
    jobs = [
        {"id": "done", "status": "succeeded", "updated_at": "2026-01-03T00:00:00Z"},
        {"id": "old", "status": "running", "updated_at": "2026-01-01T00:00:00Z"},
        {
            "id": "incident",
            "status": "failed",
            "updated_at": "2026-01-02T00:00:00Z",
            "incident": {"status": "awaiting_approval"},
        },
        {
            "id": "followup",
            "status": "failed",
            "updated_at": "2026-01-04T00:00:00Z",
            "incident": {"status": "unresolved"},
        },
    ]

    assert [job["id"] for job in _actionable_jobs(jobs)] == [
        "followup",
        "incident",
        "old",
    ]


# 功能：验证只有声明 smoke verifier 时才展示真实命令、超时和带 smoke 的批准操作
# 设计：分别传入空配置和完整配置，锁定用户批准前看到的实际 verifier 契约
def test_approval_hint_only_shows_smoke_when_configured() -> None:
    assert "smoke" not in _approval_hint(None).lower()
    hint = _approval_hint({"argv": ["python", "smoke.py", "--quick"], "timeout_s": 45.0})
    assert "smoke" in hint.lower()
    assert "$ python smoke.py --quick" in hint
    assert "timeout=45.0s" in hint
    assert "without smoke" in hint.lower()
    assert "reject" in hint.lower()


# 功能：验证 TUI 的 attempt 时长与诊断正文保留演示所需的真实信息
# 设计：固定起止时间并同时提供 summary/root_cause，断言格式稳定且二者均可见
def test_elapsed_and_diagnosis_display_helpers() -> None:
    assert _elapsed_label(
        "2026-01-01T00:00:00+00:00",
        None,
        now=datetime(2026, 1, 1, 1, 2, 3, tzinfo=UTC),
    ) == "1:02:03"
    body = _diagnosis_body(
        {
            "summary": "Shape mismatch",
            "root_cause": "The dataset emits 3 channels while the model expects 1.",
        }
    )
    assert "Shape mismatch" in body
    assert "Root cause:" in body
    assert "dataset emits 3 channels" in body


# 功能：验证非 Git 工作区只展示审阅和拒绝，不暴露任何批准入口
# 设计：显式传入 can_apply=false 并检查 A/R 文案缺失，锁定后端安全门在前端的对应反馈
def test_approval_hint_disables_apply_for_non_git_workspace() -> None:
    hint = _approval_hint({"argv": ["python", "smoke.py"]}, can_apply=False)

    assert "review-only" in hint
    assert "[X]" in hint
    assert "[A]" not in hint
    assert "[R]" not in hint


# 功能：验证日志读取兼容 RPC 的 data 字段并推进稳定字节 offset
# 设计：以含 EOF 的单个响应检查文本、next_offset 和终止标记，隔离 UI 轮询之外的协议解析
def test_log_result_reads_rpc_shape() -> None:
    assert _log_result({"data": "boom\n", "next_offset": 17, "eof": True}) == (
        "boom\n",
        17,
        True,
    )


# 功能：验证 Ctrl+Q 仅退出客户端且不会发送 job.cancel
# 设计：给应用注入记录命令的假客户端并替换 exit，直接调用 action 覆盖 detach 语义
async def test_quit_only_detaches() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录潜在命令，若实现误发 cancel 测试即可发现
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    exited: list[bool] = []
    app._client = client  # type: ignore[assignment]
    app.exit = lambda *args, **kwargs: exited.append(True)  # type: ignore[method-assign]

    await app.action_quit()

    assert exited == [True]
    assert client.commands == []


# 功能：验证从快照发现自动诊断 run 后只订阅一次对应事件流
# 设计：连续处理同一 active_run_id 两次，以假客户端记录 RPC 并排除轮询导致的重复订阅
async def test_active_incident_run_is_subscribed_once() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录订阅命令
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"subscription_id": "sub-1"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._snapshot = {"incident": {"active_run_id": "run-auto"}}

    await app._subscribe_active_run()
    await app._subscribe_active_run()

    assert client.commands == [
        (
            "event.subscribe",
            {
                "topics": ["run.*", "tool.*", "llm.*"],
                "scope": "run:run-auto",
                "replay_from_run": "run-auto",
            },
        )
    ]


# 功能：验证 follow-up worker 与快照轮询并发时仍只订阅一次同一 Agent run
# 设计：并发执行两个真实 TUI 订阅入口并主动让出调度，锁定 llm.token 不会被双重推送
async def test_followup_and_snapshot_share_one_run_subscription() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 返回固定 follow-up run，并让订阅 RPC 暴露并发窗口
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            await asyncio.sleep(0)
            if method == "session.send_message":
                return {"run_id": "run-race"}
            return {"subscription_id": "sub-run"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._session_id = "session-1"
    app._snapshot = {"incident": {"active_run_id": "run-race"}}

    await asyncio.gather(
        app._subscribe_active_run(),
        app._send_followup("check again"),
    )

    subscriptions = [
        command
        for command in client.commands
        if command[0] == "event.subscribe"
    ]
    assert subscriptions == [
        (
            "event.subscribe",
            {
                "topics": ["run.*", "tool.*", "llm.*"],
                "scope": "run:run-race",
                "replay_from_run": "run-race",
            },
        )
    ]


# 功能：验证 Job 事件订阅在 daemon 重连后从最近已见序号继续
# 设计：注入记录命令的假客户端并预置 cursor，直接断言 after_seq 没有回退到零
async def test_job_subscription_uses_persisted_cursor() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录 Job 事件订阅参数
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"subscription_id": "sub-job"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._job_event_seq = 7

    await app._subscribe_job_events()

    assert client.commands == [
        (
            "event.subscribe",
            {
                "topics": ["job.*", "incident.*"],
                "scope": "job:job-1",
                "after_seq": 7,
            },
        )
    ]


# 功能：验证 TUI 忽略重复 Job 事件并在真正切换任务时清零 cursor
# 设计：依次投递旧序号和新序号，再选择相同与不同 job_id，覆盖重连去重和切换边界
async def test_job_event_cursor_deduplicates_and_resets_on_switch() -> None:
    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    app._job_event_seq = 4

    await app._handle_event({"type": "job.finished", "seq": 4})
    await app._handle_event({"type": "job.started", "seq": 3})
    assert app._job_event_seq == 4

    await app._handle_event({"type": "job.finished", "seq": 6})
    assert app._job_event_seq == 6

    app._select_job("job-1")
    assert app._job_event_seq == 6
    app._select_job("job-2")
    assert app._job_event_seq == 0


# 功能：验证 Agent 历史回放不会在每次 TUI 重连后重复追加工具和结束状态
# 设计：把相同 tool/run 事件各投递两次并记录文本，锁定跨重连保留的稳定事件键去重
async def test_agent_replay_events_are_rendered_once() -> None:
    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    rendered: list[str] = []
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered.append(content)
    )
    tool_event = {
        "type": "tool.call_started",
        "run_id": "run-1",
        "tool_use_id": "tool-1",
        "tool_name": "read_job_log",
        "ts": "2026-01-01T00:00:00Z",
    }
    run_event = {
        "type": "run.finished",
        "run_id": "run-1",
        "status": "success",
        "ts": "2026-01-01T00:00:01Z",
    }

    await app._handle_event(tool_event)
    await app._handle_event(tool_event)
    await app._handle_event(run_event)
    await app._handle_event(run_event)

    assert rendered == ["tool: read_job_log", "Agent run: success"]


# 功能：验证 can_apply=false 时快捷键路径也无法绕过非 Git 审批限制
# 设计：直接调用统一决策方法，先断言 approve 零 RPC，再断言 reject 仍可发送
async def test_non_git_snapshot_blocks_approve_but_allows_reject() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录 Incident 决策
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"status": "rejected"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._incident_id = "incident-1"
    app._proposal_id = "proposal-1"
    app._awaiting_approval = True
    app._can_apply = False
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

    await app._decide("approve", run_smoke=True)
    assert client.commands == []

    await app._decide("reject", run_smoke=False)
    assert client.commands == [
        (
            "incident.decide",
            {
                "incident_id": "incident-1",
                "proposal_id": "proposal-1",
                "decision": "reject",
                "run_smoke": False,
            },
        )
    ]


# 功能：验证带 smoke 的批准请求绑定 TUI 当前展示的配置指纹
# 设计：直接调用决策路径并检查 RPC，防止 daemon 执行显示后被替换的 verifier
async def test_smoke_approval_sends_displayed_config_fingerprint() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录 Incident 决策
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"status": "smoke_running"}

    fingerprint = "a" * 64
    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._incident_id = "incident-1"
    app._proposal_id = "proposal-1"
    app._awaiting_approval = True
    app._can_apply = True
    app._smoke_config_fingerprint = fingerprint
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

    await app._decide("approve", run_smoke=True)

    assert client.commands == [
        (
            "incident.decide",
            {
                "incident_id": "incident-1",
                "proposal_id": "proposal-1",
                "decision": "approve",
                "run_smoke": True,
                "smoke_config_fingerprint": fingerprint,
            },
        )
    ]
