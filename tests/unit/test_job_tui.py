from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

from cyan.tui.app import (
    _LOCAL_SLASH_COMMANDS,
    ChatTextArea,
    CyanTuiApp,
    _actionable_jobs,
    _diagnosis_body,
    _diagnosis_disposition,
    _incident_action_choices,
    _incident_outcome_text,
    _log_result,
)


# 功能：验证可恢复任务按状态和更新时间筛选排序
# 设计：混合运行、待审批、未解决和已完成任务，锁定 /jobs 的选择边界
def test_actionable_jobs_filters_and_orders() -> None:
    jobs = [
        {"job": {"id": "done", "status": "succeeded", "updated_at": "2026-01-03"}},
        {"job": {"id": "running", "status": "running", "updated_at": "2026-01-01"}},
        {
            "job": {"id": "incident", "status": "failed", "updated_at": "2026-01-02"},
            "incident": {"status": "awaiting_approval"},
        },
        {
            "job": {"id": "followup", "status": "failed", "updated_at": "2026-01-04"},
            "incident": {"status": "unresolved"},
        },
    ]

    assert [entry["job"]["id"] for entry in _actionable_jobs(jobs)] == [
        "followup",
        "incident",
        "running",
    ]


# 功能：验证 TUI 只公开 Incident 业务所需的本地命令
# 设计：直接检查命令集合，防止普通 Chat、权限和旧审批命令回归
def test_local_slash_commands_are_incident_only() -> None:
    assert [name for name, _description in _LOCAL_SLASH_COMMANDS] == [
        "monitor",
        "start",
        "jobs",
        "incident",
        "help",
    ]


# 功能：验证审批选项只由 smoke 配置和后端 apply 能力决定
# 设计：覆盖无 smoke、有 smoke 与 review-only 三种上下文
def test_incident_action_choices_are_contextual() -> None:
    assert [item[0] for item in _incident_action_choices(None)] == ["approve", "reject"]
    assert [item[0] for item in _incident_action_choices({"argv": ["true"]})] == [
        "approve-smoke",
        "approve",
        "reject",
    ]
    assert _incident_action_choices(None, can_apply=False) == [
        ("reject", "Reject review-only patch")
    ]


# 功能：验证日志和诊断显示辅助函数保持稳定的协议语义
# 设计：不启动 Textual，只检查边界解析与用户可读正文
def test_log_and_diagnosis_helpers() -> None:
    assert _log_result(
        {"data": "boom\n", "next_offset": 5, "total_bytes": 5, "eof": True}
    ) == ("boom\n", 5, 5, True)
    assert _diagnosis_body({"summary": "Shape mismatch", "root_cause": "bad batch"}) == (
        "Shape mismatch\nRoot cause: bad batch"
    )
    assert "inferred" in _diagnosis_disposition(
        {"causal_support": "inferred", "patch_recommended": False},
        {},
        "unresolved",
    )
    assert "proposal is ready" in _diagnosis_disposition(
        {"causal_support": "direct", "patch_recommended": True},
        {"id": "proposal-1"},
        "awaiting_approval",
    )
    assert "no Smoke output" in _incident_outcome_text("smoke_evidence_unavailable")


# 功能：验证普通文本不会创建 Agent run 或调用任何 IPC 命令
# 设计：注入假客户端并收集提示，锁定默认状态必须先进入 /monitor 或显式 /incident
async def test_plain_text_is_rejected_without_rpc() -> None:
    class FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录不应发生的调用
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {}

    app = CyanTuiApp("127.0.0.1", 7437)
    client = FakeClient()
    notes: list[str] = []
    app._client = client  # type: ignore[assignment]
    app._append_text = lambda content, style="": notes.append(content)  # type: ignore[method-assign]

    await app._handle_submission("ordinary text")

    assert client.commands == []
    assert any("不会发送" in note for note in notes)


# 功能：验证显式 /incident 使用 v2 follow_up RPC 并订阅返回 run
# 设计：客户端只记录一条业务请求，订阅由注入的函数收集，排除旧 session 入口
async def test_incident_follow_up_uses_v2_rpc() -> None:
    class FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 返回 daemon 分配的 run_id
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"run_id": "run-2"}

    app = CyanTuiApp("127.0.0.1", 7437)
    client = FakeClient()
    subscriptions: list[str] = []
    app._client = client  # type: ignore[assignment]
    app._incident_id = "incident-1"
    async def subscribe(run_id: str) -> None:
        subscriptions.append(run_id)

    app._subscribe_run = subscribe  # type: ignore[method-assign]
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

    await app._handle_submission("/incident inspect again")

    assert client.commands == [
        (
            "incident.follow_up",
            {"incident_id": "incident-1", "content": "inspect again"},
        )
    ]
    assert subscriptions == ["run-2"]


# 功能：验证同一 run 只订阅一次且主题不包含已删除的通用事件
# 设计：重复调用发现逻辑并比较完整 RPC 参数，锁定事件回放竞态边界
async def test_active_run_subscription_is_idempotent() -> None:
    class FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录订阅请求
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"subscription_id": "sub-1"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = FakeClient()
    app._client = client  # type: ignore[assignment]
    app._snapshot = {"incident": {"active_run_id": "run-1"}}

    await app._subscribe_active_run()
    await app._subscribe_active_run()

    assert client.commands == [
        (
            "event.subscribe",
            {
                "topics": ["run.*", "step.*", "tool.*", "llm.*"],
                "scope": "run:run-1",
                "replay_from_run": "run-1",
            },
        )
    ]


# 功能：验证 TUI 忽略旧 Job 和旧 Run 的迟到事件
# 设计：注入当前标识后依次发送错误和正确事件，只允许当前 Run 渲染
async def test_tui_ignores_late_events_from_previous_job_and_run() -> None:
    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-current")
    app._active_run_id = "run-current"
    rendered: list[str] = []
    app._append_text = lambda content, style="": rendered.append(content)  # type: ignore[method-assign]

    await app._handle_event(
        {
            "type": "job.started",
            "job_id": "job-old",
            "seq": 10,
        }
    )
    await app._handle_event(
        {
            "type": "run.started",
            "run_id": "run-old",
        }
    )
    await app._handle_event(
        {
            "type": "run.started",
            "run_id": "run-current",
        }
    )

    assert app._job_event_seq == 0
    assert rendered == ["Agent run: run-current"]
    app._select_job("job-next")
    assert app._active_run_id is None


# 功能：验证 /monitor → 命令预览 → /start 复用 job.start
# 设计：以假 RPC 记录真实解析后的 argv、workspace 和环境覆盖项
async def test_monitor_preview_and_start(tmp_path: Path) -> None:
    class FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 返回新 Job ID 并接受事件订阅
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"job_id": "job-new"} if method == "job.start" else {"subscription_id": "s"}

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)
    client = FakeClient()
    rendered: list[str] = []
    app._client = client  # type: ignore[assignment]
    app._append_text = lambda content, style="": rendered.append(content)  # type: ignore[method-assign]
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    await app._handle_submission("/monitor")
    await app._handle_submission(f"CYAN_TEST_MODE=quick {sys.executable} train.py")
    await app._handle_submission("/start")

    method, params = client.commands[0]
    assert method == "job.start"
    assert params["argv"] == [sys.executable, "train.py"]
    assert params["workspace_root"] == str(tmp_path)
    assert params["env"]["CYAN_TEST_MODE"] == "quick"
    assert app._job_id == "job-new"
    assert any("Training launch preview" in text for text in rendered)


# 功能：验证上下文控件审批请求仍走 incident.decide 且 review-only 不能 apply
# 设计：分别测试拒绝和被 can_apply 门禁拦截的批准路径
async def test_incident_decision_is_contextual() -> None:
    class FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 返回结构化状态
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"status": "rejected"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = FakeClient()
    app._client = client  # type: ignore[assignment]
    app._incident_id = "incident-1"
    app._proposal_id = "proposal-1"
    app._awaiting_approval = True
    app._can_apply = False
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

    await app._decide("approve", run_smoke=True)
    assert client.commands == []
    await app._decide("reject", run_smoke=False)
    assert client.commands[0][0] == "incident.decide"
    assert client.commands[0][1]["decision"] == "reject"


# 功能：验证 Textual 生命周期中输入框始终挂载且默认焦点在命令输入
# 设计：替换连接 worker 为永不完成协程，隔离网络并检查真实 DOM
async def test_prompt_mounts_as_command_input(tmp_path: Path) -> None:
    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 保持 worker 存活但不访问网络
    async def idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = idle_connection  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", ChatTextArea)
        assert prompt.has_focus
        assert prompt.border_title == "Command or /command"


# 功能：验证 Ctrl+Q 仅退出 TUI，不发送 job.cancel
# 设计：注入空客户端和退出替身，锁定 detach 语义
async def test_quit_only_detaches() -> None:
    class FakeClient:
        # 记录潜在 RPC
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录调用
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = FakeClient()
    exited: list[bool] = []
    app._client = client  # type: ignore[assignment]
    app.exit = lambda *args, **kwargs: exited.append(True)  # type: ignore[method-assign]

    await app.action_quit()

    assert exited == [True]
    assert client.commands == []
