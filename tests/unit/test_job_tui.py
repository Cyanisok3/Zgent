from __future__ import annotations

import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from cyan.tui.app import (
    _LOCAL_SLASH_COMMANDS,
    ChatTextArea,
    ContextActionSelect,
    CyanTuiApp,
    PermissionSelect,
    SlashCompleteWidget,
    _actionable_jobs,
    _diagnosis_body,
    _elapsed_label,
    _incident_action_choices,
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


# 功能：验证本地 slash command 只保留监视、任务选择、Incident 追问和帮助入口
# 设计：直接检查稳定命令名集合，防止审批或取消动作重新泄漏到全局输入空间
def test_local_slash_commands_exclude_context_actions() -> None:
    assert [name for name, _description in _LOCAL_SLASH_COMMANDS] == [
        "monitor",
        "start",
        "jobs",
        "incident",
        "help",
    ]


# 功能：验证 smoke 配置只在上下文审批选项中增加带 smoke 的明确动作
# 设计：比较无配置和完整配置的稳定 action id，锁定 RPC 映射不再依赖 slash command
def test_incident_action_choices_only_add_smoke_when_configured() -> None:
    plain = _incident_action_choices(None)
    smoke = _incident_action_choices(
        {"argv": ["python", "smoke.py", "--quick"], "timeout_s": 45.0}
    )

    assert [action for action, _label in plain] == ["approve", "reject"]
    assert [action for action, _label in smoke] == [
        "approve-smoke",
        "approve",
        "reject",
    ]
    assert "python smoke.py --quick" in smoke[0][1]
    assert "timeout=45.0s" in smoke[0][1]


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


# 功能：验证非 Git 工作区的上下文控件只允许拒绝 review-only proposal
# 设计：显式传入 can_apply=false 并检查唯一 action id，锁定前端不能绕过后端安全门
def test_incident_action_choices_disable_apply_for_non_git_workspace() -> None:
    choices = _incident_action_choices(
        {"argv": ["python", "smoke.py"]},
        can_apply=False,
    )

    assert choices == [("reject", "Reject review-only patch")]


# 功能：验证日志读取解析文本、游标、总字节数和 EOF
# 设计：以单个响应检查四个稳定字段，隔离 UI 轮询之外的协议解析
def test_log_result_reads_rpc_shape() -> None:
    assert _log_result(
        {
            "data": "boom\n",
            "next_offset": 17,
            "total_bytes": 17,
            "eof": True,
        }
    ) == (
        "boom\n",
        17,
        17,
        True,
    )


# 功能：验证附着已有长日志时跳过历史前缀并从有界尾部继续
# 设计：用 5 MiB 总量响应驱动两次 stdout 读取，断言只渲染尾部且游标落在文件末端
async def test_attached_attempt_reads_persisted_log_tail() -> None:
    total_bytes = 5 * 1024 * 1024
    tail_offset = total_bytes - 32 * 1024

    class _FakeClient:
        # 初始化日志请求记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 按流和偏移返回长 stdout 或空 stderr
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            if params["stream"] == "stderr":
                return {
                    "data": "",
                    "next_offset": 0,
                    "total_bytes": 0,
                    "eof": True,
                }
            if params["offset"] == 0:
                return {
                    "data": "old-prefix",
                    "next_offset": 32 * 1024,
                    "total_bytes": total_bytes,
                    "eof": False,
                }
            assert params["offset"] == tail_offset
            return {
                "data": "recent-failure\n",
                "next_offset": total_bytes,
                "total_bytes": total_bytes,
                "eof": True,
            }

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-existing")
    client = _FakeClient()
    rendered_logs: list[tuple[str, str]] = []
    rendered_notes: list[str] = []
    app._client = client  # type: ignore[assignment]
    app._snapshot = {
        "job": {"id": "job-existing", "status": "failed"},
        "attempt": {"id": "attempt-1", "status": "failed"},
    }
    app._append_log = (  # type: ignore[method-assign]
        lambda stream, data: rendered_logs.append((stream, data))
    )
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered_notes.append(content)
    )

    await app._read_logs()

    stdout_offsets = [
        params["offset"]
        for method, params in client.commands
        if method == "job.read_log" and params["stream"] == "stdout"
    ]
    assert stdout_offsets == [0, tail_offset]
    assert rendered_logs == [("stdout", "recent-failure\n")]
    assert app._offsets["stdout"] == total_bytes
    assert any("showing the last 32768" in note for note in rendered_notes)


# 功能：验证由当前 TUI 新启动的任务仍从日志 byte 0 连续显示
# 设计：显式关闭首次 tail 模式并返回大总量，断言每个流只读取一次且保留 stdout 首块
async def test_new_attempt_reads_log_from_start() -> None:
    class _FakeClient:
        # 初始化日志请求记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 为 stdout 返回首块并让 stderr 保持为空
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            data = "training-start\n" if params["stream"] == "stdout" else ""
            return {
                "data": data,
                "next_offset": len(data.encode()),
                "total_bytes": 5 * 1024 * 1024 if data else 0,
                "eof": not data,
            }

    app = CyanTuiApp("127.0.0.1", 7437)
    client = _FakeClient()
    rendered_logs: list[tuple[str, str]] = []
    app._client = client  # type: ignore[assignment]
    app._select_job("job-new", tail_logs=False)
    app._snapshot = {
        "job": {"id": "job-new", "status": "running"},
        "attempt": {"id": "attempt-1", "status": "running"},
    }
    app._append_log = (  # type: ignore[method-assign]
        lambda stream, data: rendered_logs.append((stream, data))
    )
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

    await app._read_logs()

    assert [params["offset"] for _method, params in client.commands] == [0, 0]
    assert rendered_logs == [("stdout", "training-start\n")]


# 功能：验证统一多行输入框能在真实 Textual 生命周期中挂载并获得焦点
# 设计：用永不完成的无副作用连接协程替换网络 worker，只检查实际 DOM 组合
async def test_unified_prompt_mounts_in_textual_app(tmp_path: Path) -> None:
    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 保持挂载 worker 存活但不访问网络
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.pause()
        prompt = app.query_one("#prompt", ChatTextArea)
        assert prompt.has_focus
        assert prompt.border_title == "Message or /command"


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
                "topics": [
                    "run.*",
                    "step.*",
                    "tool.*",
                    "llm.*",
                    "permission.*",
                    "context.*",
                    "skill.*",
                    "log.*",
                ],
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
    app._incident_session_id = "session-1"
    app._snapshot = {"incident": {"active_run_id": "run-race"}}
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]

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
                "topics": [
                    "run.*",
                    "step.*",
                    "tool.*",
                    "llm.*",
                    "permission.*",
                    "context.*",
                    "skill.*",
                    "log.*",
                ],
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
    appended: list[Any] = []
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered.append(content)
    )
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]
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

    assert len(appended) == 1
    assert rendered == ["Agent run: success  steps=0"]


# 功能：验证普通聊天与 Incident 并发流式输出不会写入同一个文本块
# 设计：交错投递两个 run 的 token，并检查各自累积文本保持隔离
async def test_concurrent_agent_streams_use_separate_blocks() -> None:
    app = CyanTuiApp("127.0.0.1", 7437)
    appended: list[Any] = []
    app._append = lambda widget: appended.append(widget)  # type: ignore[method-assign]

    await app._handle_event({"type": "llm.token", "run_id": "chat", "token": "A"})
    await app._handle_event({"type": "llm.token", "run_id": "incident", "token": "X"})
    await app._handle_event({"type": "llm.token", "run_id": "chat", "token": "B"})

    assert len(appended) == 2
    assert app._agent_texts == {"chat": "AB", "incident": "X"}


# 功能：验证当前普通聊天 run 结束后统一输入框立即恢复可用
# 设计：直接投递匹配 run.finished，锁定无需全局 session 事件订阅的最小状态流
async def test_chat_run_finished_releases_prompt() -> None:
    app = CyanTuiApp("127.0.0.1", 7437)
    app._chat_run_id = "run-chat"
    app._chat_busy = True
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    await app._handle_event(
        {
            "type": "run.finished",
            "run_id": "run-chat",
            "status": "success",
        }
    )

    assert not app._chat_busy
    assert app._chat_run_id is None


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


# 功能：验证待审批 proposal 自动展示聚焦选择器并用 Enter 提交带 smoke 的决定
# 设计：在真实 Textual DOM 渲染 snapshot，按下 Enter 后检查 proposal 和配置指纹绑定
async def test_incident_action_select_submits_smoke_approval(tmp_path: Path) -> None:
    class _FakeClient:
        # 初始化 Incident 命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录上下文选择器发出的审批请求
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"status": "smoke_running"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1", workspace_root=tmp_path)

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]
    fingerprint = "a" * 64
    client = _FakeClient()

    async with app.run_test() as pilot:
        app._client = client  # type: ignore[assignment]
        app._snapshot = {
            "incident": {
                "id": "incident-1",
                "status": "awaiting_approval",
                "session_id": "session-1",
            },
            "diagnosis": {
                "id": "diagnosis-1",
                "summary": "shape mismatch",
                "root_cause": "wrong classifier width",
            },
            "proposal": {"id": "proposal-1", "files": []},
            "patch": "--- a/train.py\n+++ b/train.py\n",
            "smoke_config": {"argv": ["python", "smoke.py"], "timeout_s": 30},
            "smoke_config_fingerprint": fingerprint,
            "can_apply": True,
        }

        app._render_incident()
        await pilot.pause()
        actions = app.query_one("#incident-actions", ContextActionSelect)
        assert actions.has_focus
        assert actions.parent is app.query_one("#context-actions")

        await pilot.press("enter")
        await pilot.pause()

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


# 功能：验证运行期取消控件不抢聊天焦点并可通过鼠标发送 job.cancel
# 设计：渲染 running snapshot 后点击唯一动作，检查焦点和现有取消 RPC
async def test_running_job_action_clicks_cancel_without_stealing_focus(
    tmp_path: Path,
) -> None:
    class _FakeClient:
        # 初始化 Job 命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录上下文控件发出的取消请求
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"status": "cancelling"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1", workspace_root=tmp_path)

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]
    client = _FakeClient()

    async with app.run_test() as pilot:
        app._client = client  # type: ignore[assignment]
        app._snapshot = {"job": {"id": "job-1", "status": "running"}}
        app._render_job_actions()
        await pilot.pause()

        assert app.query_one("#prompt", ChatTextArea).has_focus
        assert (
            app.query_one("#job-actions", ContextActionSelect).parent
            is app.query_one("#context-actions")
        )
        assert await pilot.click("#job-actions", offset=(2, 0))
        await pilot.pause()

        assert client.commands == [
            ("job.cancel", {"job_id": "job-1"}),
        ]
        app._snapshot = {"job": {"id": "job-1", "status": "cancelled"}}
        app._render_job_actions()
        await pilot.pause()
        assert not list(app.query("#job-actions"))


# 功能：验证统一输入框中的单独 A 只会发送普通聊天，不会触发 Incident 审批
# 设计：同时预置有效提案和聊天会话，以命令记录锁定历史焦点冲突不会复发
async def test_plain_a_routes_to_chat_instead_of_incident_decision() -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 为聊天发送返回 run_id，并接受随后产生的事件订阅
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            if method == "session.send_message":
                return {"run_id": "run-chat"}
            return {"subscription_id": "sub-chat"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._chat_session_id = "chat-1"
    app._incident_id = "incident-1"
    app._proposal_id = "proposal-1"
    app._awaiting_approval = True
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    await app._handle_submission("A")

    assert client.commands[0] == (
        "session.send_message",
        {"session_id": "chat-1", "content": "A"},
    )
    assert all(method != "incident.decide" for method, _params in client.commands)
    assert app._awaiting_approval


@pytest.mark.parametrize(
    "legacy_command",
    ["/approve", "/approve-no-smoke", "/reject", "/cancel-job"],
)
# 功能：验证已删除的旧命令只进入普通聊天且不能修改 Job 或 Incident
# 设计：逐个提交四个旧 slash command，排除隐藏兼容审批或取消入口
async def test_removed_local_commands_cannot_change_state(legacy_command: str) -> None:
    class _FakeClient:
        # 初始化聊天和 Incident 命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 为普通聊天返回 run id 并接受订阅
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            if method == "session.send_message":
                return {"run_id": "run-chat"}
            return {"subscription_id": "sub-chat"}

    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-1")
    client = _FakeClient()
    app._client = client  # type: ignore[assignment]
    app._chat_session_id = "chat-1"
    app._incident_id = "incident-1"
    app._proposal_id = "proposal-1"
    app._awaiting_approval = True
    app._can_apply = True
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    await app._handle_submission(legacy_command)

    assert client.commands[0] == (
        "session.send_message",
        {"session_id": "chat-1", "content": legacy_command},
    )
    assert all(method != "incident.decide" for method, _params in client.commands)
    assert all(method != "job.cancel" for method, _params in client.commands)


# 功能：验证 /monitor 预览确认后复用现有 job.start 且只展示环境覆盖项
# 设计：用真实解释器解析并以假 RPC 检查 argv、cwd、合并环境和新 Job 附着状态
async def test_monitor_preview_and_start_use_existing_job_rpc(tmp_path: Path) -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 为 job.start 返回固定 Job，并接受 Job 事件订阅
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            if method == "job.start":
                return {"job_id": "job-new"}
            return {"subscription_id": "sub-job"}

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)
    client = _FakeClient()
    rendered: list[str] = []
    app._client = client  # type: ignore[assignment]
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered.append(content)
    )
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    await app._handle_submission("/monitor")
    await app._handle_submission(f"CYAN_TEST_MODE=quick {sys.executable} train.py")
    await app._handle_submission("/start")

    start_method, start_params = client.commands[0]
    assert start_method == "job.start"
    assert start_params["argv"] == [sys.executable, "train.py"]
    assert start_params["workspace_root"] == str(tmp_path)
    assert start_params["env"]["CYAN_TEST_MODE"] == "quick"
    preview = next(text for text in rendered if text.startswith("Training launch preview"))
    assert "CYAN_TEST_MODE=quick" in preview
    assert "PATH=" not in preview
    assert app._job_id == "job-new"


# 功能：验证多个历史任务不会在启动时挂载阻塞输入的选择器
# 设计：以假 job.list 返回两个 unresolved 任务，检查提示、空附着状态和输入框焦点
async def test_multiple_historical_jobs_keep_prompt_available(tmp_path: Path) -> None:
    class _FakeClient:
        # 返回两个仍可恢复的历史任务
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "job.list"
            return {
                "jobs": [
                    {
                        "job": {
                            "id": "job-1",
                            "status": "failed",
                            "updated_at": "2026-01-02T00:00:00Z",
                        },
                        "incident": {"status": "unresolved"},
                    },
                    {
                        "job": {
                            "id": "job-2",
                            "status": "failed",
                            "updated_at": "2026-01-01T00:00:00Z",
                        },
                        "incident": {"status": "awaiting_approval"},
                    },
                ]
            }

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        app._client = _FakeClient()  # type: ignore[assignment]

        await app._choose_job()
        await pilot.pause()

        assert app._job_id is None
        assert app._skip_job_restore
        assert not app._selection_in_progress
        assert not list(app.query("#job-picker"))
        assert app.query_one("#prompt", ChatTextArea).has_focus


# 功能：验证自动恢复不会附着其他工作区的唯一运行任务
# 设计：返回一个带外部 workspace 的可恢复 Job，检查当前项目保持空闲且提示显式 /jobs
async def test_single_foreign_job_does_not_auto_attach(tmp_path: Path) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir()

    class _FakeClient:
        # 返回另一个工作区中唯一的运行任务
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            assert method == "job.list"
            return {
                "jobs": [
                    {
                        "job": {
                            "id": "job-foreign",
                            "status": "running",
                            "updated_at": "2026-01-02T00:00:00Z",
                        },
                        "workspace_root": str(foreign),
                        "argv": [sys.executable, "train.py"],
                    }
                ]
            }

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)
    rendered: list[str] = []
    app._client = _FakeClient()  # type: ignore[assignment]
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered.append(content)
    )

    await app._choose_job()

    assert app._job_id is None
    assert app._skip_job_restore
    assert rendered == [
        "No active or pending job in this workspace. "
        "Type /monitor to start one or /jobs to attach another workspace."
    ]


# 功能：验证 /jobs 显式打开历史任务选择器并在确认后附着所选 Job
# 设计：让命令处理协程等待真实 OptionList 选择，再检查订阅和快照刷新链路
async def test_jobs_command_attaches_selected_historical_job(tmp_path: Path) -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 返回两个历史任务并支持选中后的订阅和快照读取
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            if method == "job.list":
                return {
                    "jobs": [
                        {
                            "job": {
                                "id": "job-newer",
                                "status": "failed",
                                "updated_at": "2026-01-02T00:00:00Z",
                            },
                            "incident": {"status": "unresolved"},
                        },
                        {
                            "job": {
                                "id": "job-older",
                                "status": "failed",
                                "updated_at": "2026-01-01T00:00:00Z",
                            },
                            "incident": {"status": "unresolved"},
                        },
                    ]
                }
            if method == "job.get":
                return {
                    "job": {"id": "job-newer", "status": "failed"},
                    "incident": {"status": "unresolved"},
                }
            return {"subscription_id": "sub-job"}

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]
    client = _FakeClient()

    async with app.run_test() as pilot:
        app._client = client  # type: ignore[assignment]
        selection = asyncio.create_task(app._handle_submission("/jobs"))
        await pilot.pause()

        assert app.query_one("#job-picker", ContextActionSelect).has_focus
        await pilot.press("enter")
        await selection
        await pilot.pause()

        assert app._job_id == "job-newer"
        assert any(method == "event.subscribe" for method, _params in client.commands)
        assert any(method == "job.get" for method, _params in client.commands)


# 功能：验证真实运行中的训练仍会阻止同一 TUI 启动第二个前台 Job
# 设计：预置 running 快照并直接调用本地命令，检查模式不变和精确提示
async def test_monitor_still_blocks_running_training() -> None:
    app = CyanTuiApp("127.0.0.1", 7437, job_id="job-running")
    rendered: list[str] = []
    app._snapshot = {"job": {"id": "job-running", "status": "running"}}
    app._append_text = (  # type: ignore[method-assign]
        lambda content, style="": rendered.append(content)
    )

    await app._handle_submission("/monitor")

    assert app._input_mode == "chat"
    assert rendered == [
        "The attached training process is still running. "
        "Use the Cancel training process action before starting another."
    ]


# 功能：验证输入斜杠会显示补全窗口，并可用方向键和 Enter 选择命令
# 设计：在真实 Textual 生命周期中固定两个候选，模拟键盘操作并检查回填文本
async def test_slash_popup_supports_keyboard_completion(tmp_path: Path) -> None:
    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 返回稳定候选，隔离本机 skills
    def _fixed_slash_items() -> list[tuple[str, str]]:
        return [("monitor", "monitor training"), ("help", "show help")]

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._build_slash_items = _fixed_slash_items  # type: ignore[method-assign]
    app._connection_loop = _idle_connection  # type: ignore[method-assign]

    async with app.run_test() as pilot:
        await pilot.press("/")
        await pilot.pause()
        assert app.query_one(SlashCompleteWidget).has_selection()

        await pilot.press("down", "enter")
        await pilot.pause()

        assert app.query_one("#prompt", ChatTextArea).text == "/help "
        assert not list(app.query(SlashCompleteWidget))


# 功能：验证 permission.requested 会显示焦点控件并通过 IPC 返回用户选择
# 设计：在真实 Textual DOM 中投递事件和按键，断言 permission.respond 的完整参数
async def test_permission_request_roundtrips_through_focused_control(
    tmp_path: Path,
) -> None:
    class _FakeClient:
        # 初始化命令记录
        def __init__(self) -> None:
            self.commands: list[tuple[str, dict[str, Any]]] = []

        # 记录权限响应
        async def send_command(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
            self.commands.append((method, params))
            return {"ok": True}

    app = CyanTuiApp("127.0.0.1", 7437, workspace_root=tmp_path)

    # 保持 TUI worker 存活但不连接 daemon
    async def _idle_connection() -> None:
        await asyncio.Event().wait()

    app._connection_loop = _idle_connection  # type: ignore[method-assign]
    client = _FakeClient()

    async with app.run_test() as pilot:
        app._client = client  # type: ignore[assignment]
        app._chat_busy = True
        app._update_prompt()
        await app._handle_event(
            {
                "type": "permission.requested",
                "run_id": "run-chat",
                "session_id": "chat-1",
                "tool_use_id": "tool-1",
                "tool_name": "write_file",
                "param_preview": "path='model.py'",
            }
        )
        await pilot.pause()

        assert app.query_one(PermissionSelect).has_focus
        await pilot.press("y")
        await pilot.pause()

        assert client.commands == [
            (
                "permission.respond",
                {"tool_use_id": "tool-1", "decision": "allow_once"},
            )
        ]
        assert not app._pending_permission_blocks
        assert not app._permission_selects
