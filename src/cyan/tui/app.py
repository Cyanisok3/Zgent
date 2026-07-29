from __future__ import annotations

import asyncio
import json
import os
import shlex
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual import events
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.css.query import NoMatches
from textual.message import Message
from textual.widgets import Label, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from cyan.core.skills.loader import SkillLoader
from cyan.core.transport.socket_client import IpcError, SocketClient
from cyan.tui.launch import (
    LaunchParseError,
    ParsedLaunch,
    format_launch_preview,
    parse_training_command,
)

_LOCAL_SLASH_COMMANDS = (
    ("monitor", "enter ML training monitor mode"),
    ("start", "launch the reviewed training command"),
    ("jobs", "choose a running or pending training job"),
    ("incident", "send a follow-up to the read-only Incident Agent"),
    ("help", "show local TUI commands"),
)
_LOG_READ_BYTES = 32 * 1024
_ACTIVE_JOB_STATUSES = {"starting", "running"}
_ACTIVE_INCIDENT_STATUSES = {
    "diagnosing",
    "awaiting_approval",
    "applying",
    "smoke_running",
    "smoke_passed",
    "smoke_skipped",
    "retry_running",
}
_FOLLOWUP_INCIDENT_STATUSES = {
    "awaiting_approval",
    "stale",
    "unresolved",
    "rollback_blocked",
}
_STATUS_LABELS = {
    "diagnosing": "Agent is investigating the failure",
    "awaiting_approval": "Patch is ready for review",
    "applying": "Applying the approved patch",
    "smoke_running": "Running the optional smoke verifier",
    "smoke_passed": "Smoke verifier passed",
    "smoke_skipped": "Smoke verifier skipped",
    "smoke_failed": "Smoke verifier failed; patch rollback requested",
    "retry_running": "Re-running the original command",
    "resolved": "Original command completed successfully",
    "rejected": "Patch rejected",
    "stale": "Workspace or smoke verifier changed; patch was not applied",
    "unresolved": "Incident remains unresolved",
    "rollback_blocked": "Workspace changed; automatic rollback was blocked",
}


# 从 job.list 的条目中取得实际任务记录
def _job_record(entry: dict[str, Any]) -> dict[str, Any]:
    nested = entry.get("job")
    return nested if isinstance(nested, dict) else entry


# 从任务条目中取得 Incident 记录
def _incident_record(entry: dict[str, Any]) -> dict[str, Any]:
    incident = entry.get("incident")
    return incident if isinstance(incident, dict) else {}


# 筛选可继续处理的任务并按最近更新时间排序
def _actionable_jobs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    actionable = [
        entry
        for entry in jobs
        if str(_job_record(entry).get("status", "")) in _ACTIVE_JOB_STATUSES
        or str(_incident_record(entry).get("status", "")) in _ACTIVE_INCIDENT_STATUSES
        or str(_incident_record(entry).get("status", ""))
        in _FOLLOWUP_INCIDENT_STATUSES
        or (
            str(_job_record(entry).get("status", "")) == "failed"
            and not _incident_record(entry)
        )
    ]
    return sorted(
        actionable,
        key=lambda entry: str(_job_record(entry).get("updated_at", "")),
        reverse=True,
    )


# 根据 proposal 能力和 smoke 配置生成当前可执行的审批选项
def _incident_action_choices(
    smoke_config: dict[str, Any] | None,
    *,
    can_apply: bool = True,
) -> list[tuple[str, str]]:
    if not can_apply:
        return [("reject", "Reject review-only patch")]
    if smoke_config:
        argv = smoke_config.get("argv", [])
        command = (
            shlex.join([str(value) for value in argv])
            if isinstance(argv, list)
            else str(argv)
        )
        timeout = smoke_config.get("timeout_s", 300)
        return [
            (
                "approve-smoke",
                f"Approve and run smoke: {command} (timeout={timeout}s)",
            ),
            ("approve", "Approve without smoke"),
            ("reject", "Reject patch"),
        ]
    return [
        ("approve", "Approve and rerun original command"),
        ("reject", "Reject patch"),
    ]


# 读取日志 RPC 的文本、下一字节位置、总字节数和 EOF
def _log_result(result: dict[str, Any]) -> tuple[str, int, int, bool]:
    data = str(result.get("data", result.get("text", "")))
    next_offset = int(result.get("next_offset", result.get("end_offset", 0)))
    total_bytes = int(result["total_bytes"])
    return data, next_offset, total_bytes, bool(result.get("eof", False))


# 将 Attempt 起止时间转换为稳定的短运行时长标签
def _elapsed_label(
    started_at: object,
    finished_at: object,
    *,
    now: datetime | None = None,
) -> str:
    if not started_at:
        return ""
    try:
        started = datetime.fromisoformat(str(started_at))
        finished = (
            datetime.fromisoformat(str(finished_at))
            if finished_at
            else (now or datetime.now(UTC))
        )
    except ValueError:
        return ""
    total_seconds = max(0, int((finished - started).total_seconds()))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return (
        f"{hours:d}:{minutes:02d}:{seconds:02d}"
        if hours
        else f"{minutes:02d}:{seconds:02d}"
    )


# 同时呈现诊断摘要和根因正文，避免摘要存在时遮蔽关键解释
def _diagnosis_body(diagnosis: dict[str, Any]) -> str:
    summary = str(diagnosis.get("summary") or "").strip()
    root_cause = str(diagnosis.get("root_cause") or "").strip()
    parts = [summary] if summary else []
    if root_cause and root_cause != summary:
        parts.append(f"Root cause: {root_cause}")
    return "\n".join(parts)


# 从 job.get 响应中读取指定字典字段
def _dict_field(snapshot: dict[str, Any], name: str) -> dict[str, Any]:
    value = snapshot.get(name)
    return value if isinstance(value, dict) else {}


# 从任务快照中提取启动参数和工作区
def _launch_fields(snapshot: dict[str, Any]) -> tuple[list[str], str]:
    job = _dict_field(snapshot, "job")
    launch = _dict_field(snapshot, "launch")
    spec_value = snapshot.get("spec")
    if not isinstance(spec_value, dict):
        spec_value = job.get("spec")
    spec: dict[str, Any] = spec_value if isinstance(spec_value, dict) else {}
    argv_value = snapshot.get("argv", launch.get("argv", spec.get("argv", job.get("argv", []))))
    argv = [str(value) for value in argv_value] if isinstance(argv_value, list) else []
    workspace = snapshot.get(
        "workspace_root",
        launch.get("workspace_root", spec.get("workspace_root", job.get("workspace_root", ""))),
    )
    return argv, str(workspace)


# 生成任务选择器中的单行摘要
def _job_label(entry: dict[str, Any]) -> str:
    record = _job_record(entry)
    argv, workspace = _launch_fields(entry)
    command = shlex.join(argv) if argv else "(command unavailable)"
    job_id = str(record.get("id", ""))[:8]
    status = str(record.get("status", "unknown"))
    cwd = f"  {workspace}" if workspace else ""
    return f"{status:10}  {job_id}  {command}{cwd}"


# 判断任务启动目录是否与当前 TUI 工作区一致
def _job_matches_workspace(entry: dict[str, Any], workspace_root: Path) -> bool:
    _argv, workspace = _launch_fields(entry)
    return bool(workspace) and Path(workspace).expanduser().resolve() == workspace_root


class SlashCompleteWidget(Static):
    """显示本地命令和普通 Agent skill 的斜杠补全。"""

    can_focus = False

    DEFAULT_CSS = """
    SlashCompleteWidget {
        height: auto;
        max-height: 14;
        padding: 0 1;
        margin: 0 1;
        background: $surface;
        border: round $surface-lighten-2;
    }
    """

    # 发布用户选中的命令名称
    class Selected(Message):
        # 保存不带斜杠的命令名称
        def __init__(self, name: str) -> None:
            self.name = name
            super().__init__()

    # 初始化候选列表和当前光标
    def __init__(self, items: list[tuple[str, str]]) -> None:
        super().__init__("")
        self._items = items
        self._filtered = list(items)
        self._cursor = 0

    # 按名称子串筛选候选并保持光标有效
    def set_query(self, query: str) -> None:
        lowered = query.lower()
        self._filtered = [
            item for item in self._items if not lowered or lowered in item[0].lower()
        ]
        self._cursor = min(self._cursor, max(0, len(self._filtered) - 1))
        self._redraw()

    # 将光标循环上移一项
    def move_up(self) -> None:
        if self._filtered:
            self._cursor = (self._cursor - 1) % len(self._filtered)
            self._redraw()

    # 将光标循环下移一项
    def move_down(self) -> None:
        if self._filtered:
            self._cursor = (self._cursor + 1) % len(self._filtered)
            self._redraw()

    # 选择当前候选并发布消息
    def select_current(self) -> None:
        if self._filtered:
            self.post_message(self.Selected(self._filtered[self._cursor][0]))

    # 返回当前是否存在可选候选
    def has_selection(self) -> bool:
        return bool(self._filtered)

    # 挂载后绘制初始候选
    def on_mount(self) -> None:
        self._redraw()

    # 绘制候选、描述和键盘提示
    def _redraw(self) -> None:
        if not self._filtered:
            self.update("[dim]  no matching commands[/dim]")
            return
        lines: list[str] = []
        for index, (name, description) in enumerate(self._filtered):
            marker = "❯" if index == self._cursor else " "
            style = "bold cyan" if index == self._cursor else "cyan"
            suffix = f"  [dim]{escape(description)}[/dim]" if description else ""
            lines.append(f"  [{style}]{marker} /{escape(name)}[/{style}]{suffix}")
        lines.append("[dim]  ↑↓ navigate   tab/enter select   esc dismiss[/dim]")
        self.update("\n".join(lines))


class ToolCallBlock(Static):
    """在单个块中展示工具开始、完成或失败状态。"""

    # 初始化工具标识、参数摘要和运行状态
    def __init__(self, tool_name: str, params: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._params = params
        super().__init__(self._render_pending())

    # 渲染等待完成的工具摘要
    def _render_pending(self) -> Text:
        text = Text()
        text.append("tool ", style="dim")
        text.append(self._tool_name, style="bold")
        if self._params:
            text.append(f"  {json.dumps(self._params, ensure_ascii=False)}", style="dim")
        return text

    # 用真实耗时和结果状态更新工具块
    def set_result(
        self,
        elapsed_ms: int,
        *,
        error_message: str = "",
    ) -> None:
        text = Text()
        if error_message:
            text.append("✗ ", style="bold red")
            text.append(f"{self._tool_name} failed", style="bold")
            text.append(f"  {error_message}", style="red")
        else:
            text.append("✓ ", style="bold green")
            text.append(f"{self._tool_name} completed", style="bold")
        text.append(f"  {elapsed_ms}ms", style="dim")
        self.update(text)


class PermissionBlock(Static):
    """展示工具权限请求及其最终决策。"""

    _LABELS = {
        "allow_once": "allowed once",
        "always_allow": "always allowed",
        "deny_once": "denied",
        "always_deny": "always denied",
        "auto_allow": "auto allowed",
        "auto_deny": "auto denied",
    }

    # 初始化待决权限请求
    def __init__(self, tool_name: str, param_preview: str) -> None:
        self._tool_name = tool_name
        self._param_preview = param_preview
        suffix = f"  [dim]{escape(param_preview)}[/dim]" if param_preview else ""
        super().__init__(
            f"[bold yellow]? permission[/bold yellow]  "
            f"[bold]{escape(tool_name)}[/bold]{suffix}"
        )

    # 将待决状态更新为允许或拒绝摘要
    def resolve(self, decision: str) -> None:
        allowed = decision in {"allow_once", "always_allow", "auto_allow"}
        icon = "[bold green]✓[/bold green]" if allowed else "[bold red]✗[/bold red]"
        label = self._LABELS.get(decision, decision)
        suffix = (
            f"  [dim]{escape(self._param_preview)}[/dim]"
            if self._param_preview
            else ""
        )
        self.update(
            f"{icon} permission  [bold]{escape(self._tool_name)}[/bold]"
            f"{suffix}  [dim]{escape(label)}[/dim]"
        )


class PermissionSelect(Static):
    """通过焦点和键盘选择一次性或持久权限决策。"""

    can_focus = True

    DEFAULT_CSS = """
    PermissionSelect {
        height: auto;
        padding: 0 1;
        margin: 0 1;
        border: round $warning;
    }
    """

    _CHOICES = (
        ("allow_once", "Allow once", "y / 1"),
        ("always_allow", "Always allow", "a / 2"),
        ("deny_once", "Deny", "n / 3"),
        ("always_deny", "Always deny", "d / 4"),
    )
    _KEYS = {
        "y": "allow_once",
        "1": "allow_once",
        "a": "always_allow",
        "2": "always_allow",
        "n": "deny_once",
        "3": "deny_once",
        "d": "always_deny",
        "4": "always_deny",
    }

    # 发布工具标识和用户决策
    class Decided(Message):
        # 保存来源控件、工具标识和决策
        def __init__(
            self,
            widget: PermissionSelect,
            tool_use_id: str,
            decision: str,
        ) -> None:
            self.widget = widget
            self.tool_use_id = tool_use_id
            self.decision = decision
            super().__init__()

    # 初始化工具标识和选择光标
    def __init__(self, tool_use_id: str) -> None:
        super().__init__("")
        self._tool_use_id = tool_use_id
        self._cursor = 0

    # 挂载后绘制并获取键盘焦点
    def on_mount(self) -> None:
        self.update(self._render_choices())
        self.focus()

    # 处理方向键、Enter 和直接决策快捷键
    def on_key(self, event: events.Key) -> None:
        if event.key in {"up", "k"}:
            event.stop()
            self._cursor = (self._cursor - 1) % len(self._CHOICES)
            self.update(self._render_choices())
            return
        if event.key in {"down", "j"}:
            event.stop()
            self._cursor = (self._cursor + 1) % len(self._CHOICES)
            self.update(self._render_choices())
            return
        if event.key == "enter":
            event.stop()
            self._pick(self._CHOICES[self._cursor][0])
            return
        decision = self._KEYS.get(event.key)
        if decision is not None:
            event.stop()
            self._pick(decision)

    # 渲染当前权限选项和光标
    def _render_choices(self) -> str:
        lines = []
        for index, (_decision, label, hint) in enumerate(self._CHOICES):
            marker = "❯" if index == self._cursor else " "
            style = "bold cyan" if index == self._cursor else ""
            if style:
                lines.append(f"  [{style}]{marker} {label}[/{style}]  [dim]{hint}[/dim]")
            else:
                lines.append(f"  {marker} {label}  [dim]{hint}[/dim]")
        lines.append("[dim]  ↑↓ navigate   enter confirm[/dim]")
        return "\n".join(lines)

    # 发布当前权限决策
    def _pick(self, decision: str) -> None:
        self.post_message(self.Decided(self, self._tool_use_id, decision))


class ContextActionSelect(OptionList):
    """只在当前 Job 或 Incident 状态有效时展示的上下文动作选择器。"""

    # 发布取消选择事件并保留由应用决定的控件生命周期
    class Cancelled(Message):
        # 保存被取消的选择器
        def __init__(self, widget: ContextActionSelect) -> None:
            self.widget = widget
            super().__init__()

    # 使用稳定动作标识构造可点击的纵向选择器
    def __init__(
        self,
        choices: list[tuple[str, str]],
        *,
        id: str,
        auto_focus: bool,
    ) -> None:
        super().__init__(
            *[Option(escape(label), id=action) for action, label in choices],
            id=id,
            compact=True,
        )
        self.action_choices = tuple(choices)
        self._auto_focus = auto_focus

    # 仅审批和显式 Job 选择器挂载后主动取得焦点
    def on_mount(self) -> None:
        if self._auto_focus:
            self.focus()

    # Esc 只返回聊天输入，不隐式作出任何决定
    def on_key(self, event: events.Key) -> None:
        if event.key == "escape":
            event.stop()
            event.prevent_default()
            self.post_message(self.Cancelled(self))


class ChatTextArea(TextArea):
    """支持 Enter 提交和修饰键换行的统一多行输入框。"""

    # 发布输入框当前完整文本
    class Submitted(Message):
        # 保存输入框引用和提交值
        def __init__(self, area: ChatTextArea) -> None:
            self.text_area = area
            self.value = area.text
            super().__init__()

    # 发布当前斜杠命令查询，None 表示关闭补全
    class SlashChanged(Message):
        # 保存斜杠后的过滤文本
        def __init__(self, query: str | None) -> None:
            self.query = query
            super().__init__()

    # 文本变化时检测无空白的斜杠前缀
    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.text
        query = text[1:] if text.startswith("/") and not any(c.isspace() for c in text) else None
        self.post_message(self.SlashChanged(query))

    # Enter 提交或选择补全，Shift/Alt/Cmd+Enter 或 Ctrl+J 插入换行
    async def _on_key(self, event: events.Key) -> None:
        popup: SlashCompleteWidget | None
        try:
            popup = self.app.query_one(SlashCompleteWidget)
        except NoMatches:
            popup = None
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            if popup is not None and popup.has_selection():
                popup.select_current()
                return
            if self.text.strip():
                self.post_message(self.Submitted(self))
            return
        if event.key in ("alt+enter", "shift+enter", "ctrl+j", "super+enter"):
            event.stop()
            event.prevent_default()
            if not self.read_only:
                self.insert("\n")
            return
        if popup is not None:
            if event.key == "up":
                event.stop()
                event.prevent_default()
                popup.move_up()
                return
            if event.key == "down":
                event.stop()
                event.prevent_default()
                popup.move_down()
                return
            if event.key == "tab":
                event.stop()
                event.prevent_default()
                popup.select_current()
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                self.post_message(self.SlashChanged(None))
                return
        await super()._on_key(event)


class CyanTuiApp(App[None]):
    """普通对话、真实训练进程和 Incident 调查共用的项目级 TUI。"""

    TITLE = "cyan"
    BINDINGS = [
        Binding("ctrl+q", "quit", "detach"),
    ]
    CSS = """
    Screen { background: $background; }
    #job-header {
        height: auto;
        min-height: 3;
        padding: 0 1;
        background: $surface;
    }
    #timeline {
        height: 1fr;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    #timeline Static { height: auto; padding: 0 0 1 0; }
    #context-actions {
        height: auto;
        max-height: 18;
        padding: 0 1;
    }
    #job-picker, #incident-actions, #job-actions {
        height: auto;
        max-height: 16;
        margin: 1 0;
    }
    #prompt {
        height: auto;
        min-height: 3;
        max-height: 12;
        margin: 0 1;
        border: round $surface-lighten-2;
        padding: 0 1;
        scrollbar-size-vertical: 1;
    }
    #prompt:focus {
        border: round $accent;
    }
    """

    # 初始化连接、项目工作区、聊天会话、附着目标和增量日志游标
    def __init__(
        self,
        host: str,
        port: int,
        job_id: str | None = None,
        workspace_root: Path | None = None,
    ) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._workspace_root = (workspace_root or Path.cwd()).resolve()
        self._job_id = job_id
        self._job_event_seq = 0
        self._client: SocketClient | None = None
        self._selection_ready = asyncio.Event()
        self._selection_in_progress = False
        self._skip_job_restore = False
        self._attempt_id: str | None = None
        self._offsets = {"stdout": 0, "stderr": 0}
        self._tail_logs_on_next_attempt = job_id is not None
        self._snapshot: dict[str, Any] = {}
        self._incident_id: str | None = None
        self._proposal_id: str | None = None
        self._incident_session_id: str | None = None
        self._chat_session_id: str | None = None
        self._chat_run_id: str | None = None
        self._chat_busy = False
        self._input_mode = "chat"
        self._pending_launch: ParsedLaunch | None = None
        self._slash_items: list[tuple[str, str]] = []
        self._pending_tool_blocks: dict[str, ToolCallBlock] = {}
        self._pending_permission_blocks: dict[str, PermissionBlock] = {}
        self._permission_selects: dict[str, PermissionSelect] = {}
        self._smoke_config: dict[str, Any] | None = None
        self._smoke_config_fingerprint: str | None = None
        self._can_apply = True
        self._awaiting_approval = False
        self._cancel_in_flight = False
        self._subscribed_run_ids: set[str] = set()
        self._visible_subagent_run_ids: set[str] = set()
        self._run_subscription_lock = asyncio.Lock()
        self._shown_diagnosis_id: str | None = None
        self._shown_proposal_id: str | None = None
        self._shown_smoke: str | None = None
        self._shown_incident_status: str | None = None
        self._seen_agent_events: set[tuple[str, str, str]] = set()
        self._agent_blocks: dict[str, Static] = {}
        self._agent_texts: dict[str, str] = {}

    # 组合固定头部、统一时间线和始终可用的多行输入框
    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]cyan[/bold cyan]  [dim]connecting...[/dim]", id="job-header")
        yield VerticalScroll(
            Static(
                "[dim]Type /monitor to supervise training, /help for commands. "
                "Ctrl+Q detaches without stopping a job.[/dim]",
                id="detach-note",
            ),
            id="timeline",
        )
        yield Vertical(id="context-actions")
        yield ChatTextArea("", id="prompt")

    # 挂载后启动唯一的连接与轮询 worker
    def on_mount(self) -> None:
        self._slash_items = self._build_slash_items()
        self._render_idle_header()
        self._update_prompt()
        self.query_one("#prompt", ChatTextArea).focus()
        self.run_worker(self._connection_loop(), exclusive=True, name="job-connection")

    # Ctrl+Q 只退出 TUI，不向 daemon 发送取消或关闭会话命令
    async def action_quit(self) -> None:
        self.exit()

    # 连接 daemon、创建聊天会话、恢复任务并持续增量读取状态与日志
    async def _connection_loop(self) -> None:
        reconnecting = False
        while True:
            client = SocketClient(self._host, self._port)
            event_task: asyncio.Task[None] | None = None
            try:
                await client.connect()
                self._client = client
                self._subscribed_run_ids.clear()
                self._visible_subagent_run_ids.clear()
                client.on_event(self._handle_event)
                event_task = asyncio.create_task(client.run_event_loop())
                if reconnecting:
                    self._append_text("Reconnected to daemon.", style="bold cyan")
                await self._prepare_chat()
                if self._job_id is None and not self._skip_job_restore:
                    await self._choose_job()
                if self._job_id is not None:
                    await self._subscribe_job_events()
                while not event_task.done():
                    if self._job_id is not None:
                        await self._refresh_snapshot()
                        await self._read_logs()
                    await asyncio.sleep(0.4)
                raise ConnectionError("daemon connection closed")
            except asyncio.CancelledError:
                raise
            except (IpcError, RuntimeError, OSError) as error:
                message = "Cannot connect" if not reconnecting else "Connection lost"
                self._append_text(
                    f"{message}: {error}; retrying...",
                    style="bold red",
                )
                reconnecting = True
            finally:
                if event_task is not None:
                    event_task.cancel()
                    with suppress(asyncio.CancelledError):
                        await event_task
                await client.close()
                self._client = None
            await asyncio.sleep(0.5)

    # 创建本次 TUI 的普通 chat session，并在重连时复用当前会话
    async def _prepare_chat(self) -> None:
        if self._client is None:
            return
        await self._client.send_command(
            "event.subscribe",
            {
                "topics": ["session.*", "subagent.*"],
                "scope": "global",
            },
        )
        if self._chat_session_id is None:
            result = await self._client.send_command(
                "session.create",
                {
                    "mode": "chat",
                    "title": self._workspace_root.name,
                    "workspace_root": str(self._workspace_root),
                },
            )
            self._chat_session_id = str(result["session_id"])
            self._append_text(
                f"Chat ready in {self._workspace_root}",
                style="bold cyan",
            )
        if self._chat_run_id is not None:
            await self._subscribe_run(self._chat_run_id)

    # 合并本地 TUI 命令与普通 Agent skills，保持本地命令优先
    def _build_slash_items(self) -> list[tuple[str, str]]:
        items = dict(_LOCAL_SLASH_COMMANDS)
        for skill in SkillLoader().list_all_skills():
            if skill.name not in items:
                description = skill.description.splitlines()[0] if skill.description else ""
                items[skill.name] = description[:60]
        return list(items.items())

    # 使用当前持久化序号订阅 Job 流，重连时仅回放未见事件
    async def _subscribe_job_events(self) -> None:
        if self._client is None or self._job_id is None:
            return
        await self._client.send_command(
            "event.subscribe",
            {
                "topics": ["job.*", "incident.*"],
                "scope": f"job:{self._job_id}",
                "after_seq": self._job_event_seq,
            },
        )

    # 切换附着目标时重置仅属于旧 Job 的事件和展示状态
    def _select_job(self, job_id: str, *, tail_logs: bool = True) -> None:
        if job_id != self._job_id:
            self._job_event_seq = 0
            self._attempt_id = None
            self._offsets = {"stdout": 0, "stderr": 0}
            self._snapshot = {}
            self._incident_id = None
            self._proposal_id = None
            self._incident_session_id = None
            self._shown_diagnosis_id = None
            self._shown_proposal_id = None
            self._shown_smoke = None
            self._shown_incident_status = None
            self._tail_logs_on_next_attempt = tail_logs
        self._job_id = job_id
        self._skip_job_restore = False

    # 自动附着唯一任务，多个任务仅在用户显式请求时打开选择器
    async def _choose_job(self, *, interactive: bool = False) -> None:
        if self._client is None:
            return
        result = await self._client.send_command("job.list", {})
        raw_jobs = result.get("jobs", [])
        all_jobs = _actionable_jobs(
            [entry for entry in raw_jobs if isinstance(entry, dict)]
            if isinstance(raw_jobs, list)
            else []
        )
        jobs = (
            all_jobs
            if interactive
            else [
                entry
                for entry in all_jobs
                if _job_matches_workspace(entry, self._workspace_root)
            ]
        )
        if not jobs:
            if all_jobs and not interactive:
                self._skip_job_restore = True
                self._append_text(
                    "No active or pending job in this workspace. "
                    "Type /monitor to start one or /jobs to attach another workspace.",
                    style="dim cyan",
                )
            else:
                self._append_text("No active or pending job. Type /monitor to start one.")
            return
        if len(jobs) == 1:
            self._select_job(str(_job_record(jobs[0]).get("id", "")))
            return
        if not interactive:
            self._skip_job_restore = True
            self._append_text(
                f"{len(jobs)} recoverable jobs found. Type /jobs to choose one.",
                style="dim cyan",
            )
            self.query_one("#prompt", ChatTextArea).focus()
            return
        options = [
            (str(_job_record(entry).get("id", "")), _job_label(entry))
            for entry in jobs
        ]
        self._selection_ready.clear()
        self._selection_in_progress = True
        self._append_text("Choose a job to attach:")
        self._mount_context_actions(
            "job-picker",
            options,
            auto_focus=True,
        )
        await self._selection_ready.wait()
        self._selection_in_progress = False

    # 接收 Job、Incident 和取消动作选择并路由到现有 RPC
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        widget_id = event.option_list.id
        if widget_id == "job-picker" and event.option_id:
            self._select_job(event.option_id)
            self._selection_ready.set()
            event.option_list.remove()
            self.query_one("#prompt", ChatTextArea).focus()
        elif widget_id == "incident-actions" and event.option_id:
            event.option_list.disabled = True
            action = event.option_id
            decision = "reject" if action == "reject" else "approve"
            self.run_worker(
                self._decide(decision, run_smoke=action == "approve-smoke"),
                name="incident-decision",
                exclusive=False,
            )
        elif widget_id == "job-actions" and event.option_id == "cancel":
            event.option_list.disabled = True
            self.run_worker(
                self._cancel_job(),
                name="job-cancel",
                exclusive=False,
            )

    # Esc 取消 Job 选择或把上下文操作焦点交还聊天输入框
    def on_context_action_select_cancelled(
        self,
        event: ContextActionSelect.Cancelled,
    ) -> None:
        if event.widget.id == "job-picker":
            event.widget.remove()
            self._selection_in_progress = False
            self._selection_ready.set()
        self.query_one("#prompt", ChatTextArea).focus()

    # 在日志区外挂载或更新单个稳定 ID 的上下文选择器
    def _mount_context_actions(
        self,
        widget_id: str,
        choices: list[tuple[str, str]],
        *,
        auto_focus: bool,
    ) -> None:
        with suppress(NoMatches):
            current = self.query_one(f"#{widget_id}", ContextActionSelect)
            if current.action_choices == tuple(choices):
                return
            current.remove()
        action_host = self.query_one("#context-actions", Vertical)
        action_host.mount(
            ContextActionSelect(
                choices,
                id=widget_id,
                auto_focus=auto_focus,
            )
        )

    # 根据输入框斜杠前缀挂载、更新或移除自动补全
    def on_chat_text_area_slash_changed(self, event: ChatTextArea.SlashChanged) -> None:
        if event.query is None or self._input_mode == "monitor":
            self._remove_slash_popup()
            return
        try:
            popup = self.query_one(SlashCompleteWidget)
        except NoMatches:
            popup = SlashCompleteWidget(self._slash_items)
            self.mount(popup, before="#prompt")
        popup.set_query(event.query)

    # 将选中的斜杠命令写回输入框并移动到末尾
    def on_slash_complete_widget_selected(
        self,
        event: SlashCompleteWidget.Selected,
    ) -> None:
        prompt = self.query_one("#prompt", ChatTextArea)
        prompt.text = f"/{event.name} "
        prompt.move_cursor(prompt.document.end)
        self._remove_slash_popup()

    # 移除当前斜杠补全弹窗
    def _remove_slash_popup(self) -> None:
        with suppress(NoMatches):
            self.query_one(SlashCompleteWidget).remove()

    # 拉取最新结构化状态并仅渲染尚未展示的 Incident 制品
    async def _refresh_snapshot(self) -> None:
        if self._client is None or self._job_id is None:
            return
        self._snapshot = await self._client.send_command("job.get", {"job_id": self._job_id})
        await self._subscribe_active_run()
        self._render_header()
        self._render_job_actions()
        self._render_incident()

    # 发现 Incident 新 run 后订阅其 Agent 事件，避免自动诊断流不可见
    async def _subscribe_active_run(self) -> None:
        incident = _dict_field(self._snapshot, "incident")
        run_id = str(incident.get("active_run_id") or "")
        await self._subscribe_run(run_id)

    # 串行化同一 Agent run 的订阅，并在连接切换时避免把旧连接标记为已订阅
    async def _subscribe_run(self, run_id: str) -> None:
        if not run_id:
            return
        async with self._run_subscription_lock:
            client = self._client
            if client is None or run_id in self._subscribed_run_ids:
                return
            await client.send_command(
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
                    "scope": f"run:{run_id}",
                    "replay_from_run": run_id,
                },
            )
            if client is self._client:
                self._subscribed_run_ids.add(run_id)

    # 在没有附着任务时展示项目级聊天和监视入口
    def _render_idle_header(self) -> None:
        header = Text()
        header.append("cyan", style="bold cyan")
        header.append("  chat · idle\n", style="dim")
        header.append(str(self._workspace_root), style="dim")
        self.query_one("#job-header", Label).update(header)

    # 更新固定头部中的真实命令、工作区和状态
    def _render_header(self) -> None:
        job = _dict_field(self._snapshot, "job")
        attempt = _dict_field(self._snapshot, "attempt")
        incident = _dict_field(self._snapshot, "incident")
        argv, workspace = _launch_fields(self._snapshot)
        command = shlex.join(argv) if argv else "(command unavailable)"
        job_status = str(job.get("status", "unknown"))
        attempt_status = str(attempt.get("status", ""))
        incident_status = str(incident.get("status", ""))
        status_parts = [job_status]
        if attempt_status and attempt_status != job_status:
            status_parts.append(f"attempt:{attempt_status}")
        if attempt.get("returncode") is not None:
            status_parts.append(f"exit={attempt['returncode']}")
        if attempt.get("signal") is not None:
            status_parts.append(f"signal={attempt['signal']}")
        elapsed = _elapsed_label(
            attempt.get("started_at"),
            attempt.get("finished_at"),
        )
        if elapsed:
            status_parts.append(f"elapsed={elapsed}")
        if incident_status:
            status_parts.append(f"incident:{incident_status}")
        header = Text()
        header.append("cyan", style="bold cyan")
        header.append(f"  {' · '.join(status_parts)}\n", style="dim")
        header.append("$ ", style="bold")
        header.append(command)
        header.append(f"\n{workspace}", style="dim")
        if attempt.get("error"):
            header.append(f"\nerror: {attempt['error']}", style="bold red")
        self.query_one("#job-header", Label).update(header)

    # 增量读取当前 attempt 的 stdout 与 stderr 并推进字节游标
    async def _read_logs(self) -> None:
        if self._client is None or self._job_id is None:
            return
        job = _dict_field(self._snapshot, "job")
        attempt = _dict_field(self._snapshot, "attempt")
        attempt_id = str(
            attempt.get("id")
            or job.get("current_attempt_id")
            or self._snapshot.get("attempt_id")
            or ""
        )
        if not attempt_id:
            return
        if attempt_id != self._attempt_id:
            self._attempt_id = attempt_id
            self._offsets = {"stdout": 0, "stderr": 0}
            self._append_text(f"attempt {attempt_id}", style="bold")
            tail_logs = self._tail_logs_on_next_attempt
            self._tail_logs_on_next_attempt = False
        else:
            tail_logs = False
        for stream in ("stdout", "stderr"):
            result = await self._client.send_command(
                "job.read_log",
                {
                    "job_id": self._job_id,
                    "attempt_id": attempt_id,
                    "stream": stream,
                    "offset": self._offsets[stream],
                    "limit": _LOG_READ_BYTES,
                },
            )
            data, next_offset, total_bytes, _eof = _log_result(result)
            if tail_logs and total_bytes > _LOG_READ_BYTES:
                tail_offset = total_bytes - _LOG_READ_BYTES
                result = await self._client.send_command(
                    "job.read_log",
                    {
                        "job_id": self._job_id,
                        "attempt_id": attempt_id,
                        "stream": stream,
                        "offset": tail_offset,
                        "limit": _LOG_READ_BYTES,
                    },
                )
                data, next_offset, _total_bytes, _eof = _log_result(result)
                self._append_text(
                    f"{stream}: showing the last {_LOG_READ_BYTES} of "
                    f"{total_bytes} persisted bytes",
                    style="dim",
                )
            self._offsets[stream] = next_offset
            if data:
                self._append_log(stream, data)

    # 将原始 stdout 或 stderr 块按流着色后追加到时间线
    def _append_log(self, stream: str, data: str) -> None:
        text = Text()
        text.append(f"{stream}\n", style="dim cyan" if stream == "stdout" else "dim red")
        text.append(data)
        self._append(Static(text))

    # 展示新的诊断、证据、补丁、验证状态和审批操作
    def _render_incident(self) -> None:
        incident = _dict_field(self._snapshot, "incident")
        diagnosis = _dict_field(self._snapshot, "diagnosis")
        proposal = _dict_field(self._snapshot, "proposal")
        smoke = _dict_field(self._snapshot, "smoke")
        smoke_config = _dict_field(self._snapshot, "smoke_config")
        self._incident_id = str(incident.get("id") or "") or None
        self._proposal_id = str(
            proposal.get("id") or incident.get("active_proposal_id") or ""
        ) or None
        self._incident_session_id = str(incident.get("session_id") or "") or None
        self._smoke_config = smoke_config or None
        self._smoke_config_fingerprint = (
            str(self._snapshot.get("smoke_config_fingerprint") or "") or None
        )
        self._can_apply = bool(self._snapshot.get("can_apply", True))

        status = str(incident.get("status", ""))
        if status and status != self._shown_incident_status:
            self._shown_incident_status = status
            self._append_text(_STATUS_LABELS.get(status, status), style="bold cyan")

        diagnosis_id = str(diagnosis.get("id") or "")
        if diagnosis and diagnosis_id != self._shown_diagnosis_id:
            self._shown_diagnosis_id = diagnosis_id
            self._append_diagnosis(diagnosis)

        proposal_id = self._proposal_id or ""
        if proposal and proposal_id != self._shown_proposal_id:
            self._shown_proposal_id = proposal_id
            patch = self._snapshot.get("patch", self._snapshot.get("proposal_diff", ""))
            self._append_proposal(proposal, str(patch))

        smoke_key = json.dumps(smoke, sort_keys=True, ensure_ascii=False) if smoke else ""
        if smoke_key and smoke_key != self._shown_smoke:
            self._shown_smoke = smoke_key
            self._append_smoke(smoke)

        self._awaiting_approval = (
            status == "awaiting_approval"
            and self._incident_id is not None
            and self._proposal_id is not None
        )
        if self._awaiting_approval:
            self._append_or_update_actions()
        else:
            with suppress(NoMatches):
                self.query_one("#incident-actions", ContextActionSelect).remove()

    # 将结构化根因和稳定证据引用追加到时间线
    def _append_diagnosis(self, diagnosis: dict[str, Any]) -> None:
        text = Text()
        text.append("Diagnosis\n", style="bold cyan")
        text.append(_diagnosis_body(diagnosis))
        evidence = diagnosis.get("evidence")
        if isinstance(evidence, list):
            for item in evidence:
                if not isinstance(item, dict):
                    continue
                reference = str(item.get("reference", ""))
                description = str(item.get("description", ""))
                text.append(f"\n• {reference}", style="bold")
                if description:
                    text.append(f"  {description}", style="dim")
        self._append(Static(text))

    # 将完整 proposed diff 作为纯文本追加，避免日志内容被解释成 Rich markup
    def _append_proposal(self, proposal: dict[str, Any], patch: str) -> None:
        text = Text()
        text.append("Proposed patch", style="bold cyan")
        files = proposal.get("files")
        if isinstance(files, list):
            paths = [str(item.get("path", "")) for item in files if isinstance(item, dict)]
            if paths:
                text.append(f"  {', '.join(paths)}", style="dim")
        text.append("\n")
        text.append(patch or "(diff unavailable)")
        self._append(Static(text))

    # 展示 smoke verifier 的命令、状态和真实退出码
    def _append_smoke(self, smoke: dict[str, Any]) -> None:
        argv = smoke.get("argv", smoke.get("command", []))
        command = (
            shlex.join([str(value) for value in argv])
            if isinstance(argv, list)
            else str(argv)
        )
        status = str(smoke.get("status", "unknown"))
        returncode = smoke.get("returncode")
        suffix = f"  exit={returncode}" if returncode is not None else ""
        command_line = f"\n$ {command}" if command else ""
        self._append_text(f"smoke: {status}{suffix}{command_line}", style="bold")

    # 根据真实 Job 状态创建或移除非抢焦点的取消训练动作
    def _render_job_actions(self) -> None:
        job = _dict_field(self._snapshot, "job")
        active = str(job.get("status", "")) in _ACTIVE_JOB_STATUSES
        if active:
            self._mount_context_actions(
                "job-actions",
                [("cancel", "Cancel training process")],
                auto_focus=False,
            )
            return
        self._cancel_in_flight = False
        with suppress(NoMatches):
            self.query_one("#job-actions", ContextActionSelect).remove()

    # 创建或更新唯一的审批选择器，避免轮询产生重复控件
    def _append_or_update_actions(self) -> None:
        choices = _incident_action_choices(
            self._smoke_config,
            can_apply=self._can_apply,
        )
        self._mount_context_actions(
            "incident-actions",
            choices,
            auto_focus=True,
        )

    # 发送一次 Incident 决策并等待后续真实状态由轮询刷新
    async def _decide(self, decision: str, *, run_smoke: bool) -> None:
        if (
            not self._awaiting_approval
            or self._client is None
            or self._incident_id is None
            or self._proposal_id is None
            or (decision == "approve" and not self._can_apply)
        ):
            return
        self._awaiting_approval = False
        payload: dict[str, Any] = {
            "incident_id": self._incident_id,
            "proposal_id": self._proposal_id,
            "decision": decision,
            "run_smoke": run_smoke,
        }
        if run_smoke:
            payload["smoke_config_fingerprint"] = self._smoke_config_fingerprint
        try:
            await self._client.send_command(
                "incident.decide",
                payload,
            )
        except (IpcError, RuntimeError, OSError) as error:
            self._append_text(f"Incident decision failed: {error}", style="red")
            self._awaiting_approval = True
            with suppress(NoMatches):
                self.query_one("#incident-actions", ContextActionSelect).remove()
            self._append_or_update_actions()
            return
        smoke_note = " with smoke" if decision == "approve" and run_smoke else ""
        self._append_text(f"{decision} submitted{smoke_note}", style="bold")

    # 向 daemon 发送用户明确触发的 job.cancel
    async def _cancel_job(self) -> None:
        if (
            self._client is None
            or self._job_id is None
            or not self._has_running_job()
            or self._cancel_in_flight
        ):
            return
        self._cancel_in_flight = True
        try:
            await self._client.send_command("job.cancel", {"job_id": self._job_id})
        except (IpcError, RuntimeError, OSError) as error:
            self._cancel_in_flight = False
            self._append_text(f"Cancellation failed: {error}", style="red")
            with suppress(NoMatches):
                self.query_one("#job-actions", ContextActionSelect).remove()
            self._render_job_actions()
            return
        self._append_text("Cancellation requested", style="yellow")

    # 统一接收多行输入，并按当前本地模式路由到命令解析或普通聊天
    def on_chat_text_area_submitted(self, event: ChatTextArea.Submitted) -> None:
        content = event.value.strip()
        if not content:
            return
        event.text_area.text = ""
        self.run_worker(
            self._handle_submission(content),
            name="prompt-submission",
            exclusive=False,
        )

    # 训练命令输入模式只做本地解析，其他输入先匹配本地斜杠命令
    async def _handle_submission(self, content: str) -> None:
        if self._input_mode == "monitor":
            if content == "/cancel":
                self._input_mode = "chat"
                self._pending_launch = None
                self._append_text("Training command entry cancelled.", style="dim")
                self._update_prompt()
                return
            self._preview_launch(content)
            return
        handled = await self._handle_local_command(content)
        if not handled:
            await self._send_chat(content)

    # 处理仅属于 TUI harness 的命令，未知斜杠输入继续交给普通 Agent skill
    async def _handle_local_command(self, content: str) -> bool:
        command, _, argument = content.partition(" ")
        if command == "/help":
            self._append_text(
                "/monitor · /start · /jobs · /incident <text>",
                style="dim",
            )
            return True
        if command == "/monitor":
            if self._job_id is not None and not self._snapshot:
                await self._refresh_snapshot()
            if self._has_running_job():
                self._append_text(
                    "The attached training process is still running. "
                    "Use the Cancel training process action before starting another.",
                    style="yellow",
                )
                return True
            self._input_mode = "monitor"
            self._pending_launch = None
            self._append_text(
                "Paste one training command. Enter submits; Shift+Enter inserts a newline. "
                "Type /cancel to leave.",
                style="bold cyan",
            )
            self._update_prompt()
            return True
        if command == "/start":
            await self._start_pending_launch()
            return True
        if command == "/jobs":
            if self._selection_in_progress:
                return True
            previous_job_id = self._job_id
            await self._choose_job(interactive=True)
            if self._job_id is not None and (
                self._job_id != previous_job_id or not self._snapshot
            ):
                await self._subscribe_job_events()
                await self._refresh_snapshot()
            return True
        if command == "/incident":
            if not argument.strip():
                self._append_text("Usage: /incident <follow-up>", style="yellow")
            else:
                await self._send_followup(argument.strip())
            return True
        return False

    # 仅判断当前附着任务的真实训练进程是否仍在运行
    def _has_running_job(self) -> bool:
        if self._job_id is None:
            return False
        job = _dict_field(self._snapshot, "job")
        return str(job.get("status", "")) in _ACTIVE_JOB_STATUSES

    # 将训练命令解析为确定性预览，解析失败时保持命令输入模式
    def _preview_launch(self, content: str) -> None:
        try:
            launch = parse_training_command(
                content,
                self._workspace_root,
                os.environ,
            )
        except LaunchParseError as error:
            self._append_text(f"Invalid training command: {error}", style="bold red")
            return
        self._pending_launch = launch
        self._input_mode = "chat"
        self._append_text(
            format_launch_preview(launch, self._workspace_root),
            style="bold",
        )
        self._update_prompt()

    # 用户显式确认后通过现有 job.start 启动真实训练并附着返回的 Job
    async def _start_pending_launch(self) -> None:
        if self._pending_launch is None:
            self._append_text("No launch preview. Type /monitor first.", style="yellow")
            return
        if self._client is None:
            self._append_text("Daemon is not connected.", style="bold red")
            return
        if self._job_id is not None and not self._snapshot:
            await self._refresh_snapshot()
        if self._has_running_job():
            self._append_text(
                "The attached training process is still running.",
                style="yellow",
            )
            return
        launch = self._pending_launch
        environment = dict(os.environ)
        environment.update(launch.env_overrides)
        result = await self._client.send_command(
            "job.start",
            {
                "argv": list(launch.argv),
                "workspace_root": str(self._workspace_root),
                "env": environment,
            },
        )
        job_id = str(result["job_id"])
        self._pending_launch = None
        self._select_job(job_id, tail_logs=False)
        await self._subscribe_job_events()
        self._append_text(f"Training started: {job_id}", style="bold cyan")

    # 发送普通聊天消息并订阅本次 Agent run
    async def _send_chat(self, content: str) -> None:
        if self._client is None or self._chat_session_id is None:
            self._append_text("Chat is not connected.", style="bold red")
            return
        if self._chat_busy:
            self._append_text("Agent is still working.", style="yellow")
            return
        self._append_text(f"> {content}", style="bold")
        self._chat_busy = True
        self._update_prompt()
        try:
            result = await self._client.send_command(
                "session.send_message",
                {"session_id": self._chat_session_id, "content": content},
            )
        except (IpcError, RuntimeError, OSError) as error:
            self._chat_busy = False
            self._update_prompt()
            self._append_text(f"Chat failed: {error}", style="bold red")
            return
        run_id = str(result.get("run_id", ""))
        self._chat_run_id = run_id or None
        await self._subscribe_run(run_id)
        if not self._chat_busy:
            self._chat_run_id = None

    # 复用 daemon 拥有的只读 Incident session 发送显式追问
    async def _send_followup(self, content: str) -> None:
        if self._client is None or self._incident_session_id is None:
            self._append_text("No incident is available for follow-up.", style="yellow")
            return
        self._append_text(f"incident> {content}", style="bold")
        result = await self._client.send_command(
            "session.send_message",
            {"session_id": self._incident_session_id, "content": content},
        )
        run_id = str(result.get("run_id", ""))
        await self._subscribe_run(run_id)

    # 将焦点审批结果通过现有 permission.respond IPC 返回 core
    async def on_permission_select_decided(
        self,
        message: PermissionSelect.Decided,
    ) -> None:
        if self._client is None:
            self._append_text("Permission response failed: daemon disconnected.", style="red")
            return
        try:
            await self._client.send_command(
                "permission.respond",
                {
                    "tool_use_id": message.tool_use_id,
                    "decision": message.decision,
                },
            )
        except (IpcError, RuntimeError, OSError) as error:
            self._append_text(f"Permission response failed: {error}", style="red")
            return
        self._resolve_permission(message.tool_use_id, message.decision)

    # 收起指定权限控件并恢复统一输入框状态
    def _resolve_permission(self, tool_use_id: str, decision: str) -> None:
        select = self._permission_selects.pop(tool_use_id, None)
        if select is not None:
            select.remove()
        block = self._pending_permission_blocks.pop(tool_use_id, None)
        if block is not None:
            block.resolve(decision)
        if (select is not None or block is not None) and not self._permission_selects:
            self._update_prompt()
            self.query_one("#prompt", ChatTextArea).focus()

    # 根据聊天或训练命令输入状态刷新统一输入框提示
    def _update_prompt(self) -> None:
        prompt = self.query_one("#prompt", ChatTextArea)
        if self._input_mode == "monitor":
            prompt.border_title = "Training command"
            prompt.read_only = False
        elif self._chat_busy:
            prompt.border_title = "Agent is working"
            prompt.read_only = True
        else:
            prompt.border_title = "Message or /command"
            prompt.read_only = False

    # 将聊天与 Incident 的流式文本、工具和权限事件追加到统一时间线
    async def _handle_event(self, event: dict[str, Any]) -> None:
        event_type = str(event.get("type", ""))
        if event_type.startswith("job."):
            seq = event.get("seq")
            if (
                not isinstance(seq, int)
                or isinstance(seq, bool)
                or seq <= self._job_event_seq
            ):
                return
            self._job_event_seq = seq
            return
        if event_type.startswith("session."):
            session_id = str(event.get("session_id") or "")
            if session_id not in {self._chat_session_id, self._incident_session_id}:
                return
            if (
                event_type == "session.waiting_for_input"
                and session_id == self._chat_session_id
            ):
                self._chat_busy = False
                self._chat_run_id = None
                self._update_prompt()
            elif event_type == "session.closed" and session_id == self._chat_session_id:
                self._chat_busy = False
                self._chat_run_id = None
                self._append_text("Chat session closed.", style="dim")
                prompt = self.query_one("#prompt", ChatTextArea)
                prompt.read_only = True
                prompt.border_title = "Chat session closed"
            return
        if event_type == "subagent.started":
            parent_run_id = str(event.get("parent_run_id") or "")
            if parent_run_id not in self._subscribed_run_ids:
                return
            run_id = str(event.get("run_id") or "")
            self._visible_subagent_run_ids.add(run_id)
            self._append_text(
                f"subagent: {event.get('description', '')}  {run_id}",
                style="dim cyan",
            )
            return
        if event_type == "subagent.finished":
            run_id = str(event.get("run_id") or "")
            parent_run_id = str(event.get("parent_run_id") or "")
            if (
                run_id not in self._visible_subagent_run_ids
                and parent_run_id not in self._subscribed_run_ids
            ):
                return
            self._visible_subagent_run_ids.discard(run_id)
            self._append_text(
                f"subagent: {run_id}  {event.get('status', '')}",
                style="dim",
            )
            return
        if event_type == "run.started":
            run_id = str(event.get("run_id") or "")
            event_key = (event_type, run_id, "")
            if event_key not in self._seen_agent_events:
                self._seen_agent_events.add(event_key)
                self._append_text(f"Agent run: {run_id}", style="dim cyan")
            return
        if event_type == "tool.call_started":
            tool_use_id = str(event.get("tool_use_id") or "")
            event_key = (event_type, str(event.get("run_id") or ""), tool_use_id)
            if not tool_use_id or event_key in self._seen_agent_events:
                return
            self._seen_agent_events.add(event_key)
            params = event.get("params")
            block = ToolCallBlock(
                str(event.get("tool_name") or ""),
                params if isinstance(params, dict) else {},
            )
            self._pending_tool_blocks[tool_use_id] = block
            self._append(block)
            return
        if event_type in {"tool.call_finished", "tool.call_failed"}:
            tool_use_id = str(event.get("tool_use_id") or "")
            identity = (
                tool_use_id
                or str(event.get("ts") or "")
            )
            event_key = (event_type, str(event.get("run_id") or ""), identity)
            if event_key in self._seen_agent_events:
                return
            self._seen_agent_events.add(event_key)
            run_id = str(event.get("run_id") or "")
            self._agent_blocks.pop(run_id, None)
            self._agent_texts.pop(run_id, None)
            tool_block = self._pending_tool_blocks.get(tool_use_id)
            if tool_block is not None:
                self._pending_tool_blocks.pop(tool_use_id)
                tool_block.set_result(
                    int(event.get("elapsed_ms") or 0),
                    error_message=(
                        str(event.get("error_message") or "")
                        if event_type == "tool.call_failed"
                        else ""
                    ),
                )
            return
        if event_type == "run.finished":
            run_id = str(event.get("run_id") or "")
            event_key = (event_type, run_id, str(event.get("status") or ""))
            if event_key in self._seen_agent_events:
                return
            self._seen_agent_events.add(event_key)
            self._agent_blocks.pop(run_id, None)
            self._agent_texts.pop(run_id, None)
            if run_id == self._chat_run_id:
                self._chat_busy = False
                self._chat_run_id = None
                self._update_prompt()
            self._append_text(
                f"Agent run: {event.get('status', '')}  "
                f"steps={event.get('steps', 0)}",
                style=(
                    "dim green"
                    if event.get("status") == "success"
                    else "dim red"
                ),
            )
            return
        if event_type == "permission.requested":
            tool_use_id = str(event.get("tool_use_id") or "")
            if not tool_use_id or tool_use_id in self._pending_permission_blocks:
                return
            tool_name = str(event.get("tool_name") or "")
            preview = str(event.get("param_preview") or "")
            permission_block = PermissionBlock(tool_name, preview)
            select = PermissionSelect(tool_use_id)
            self._pending_permission_blocks[tool_use_id] = permission_block
            self._permission_selects[tool_use_id] = select
            self._append(permission_block)
            prompt = self.query_one("#prompt", ChatTextArea)
            prompt.read_only = True
            prompt.border_title = f"Permission required: {tool_name}"
            self.mount(select, before="#prompt")
            return
        if event_type in {"permission.granted", "permission.denied"}:
            tool_use_id = str(event.get("tool_use_id") or "")
            self._resolve_permission(
                tool_use_id,
                str(event.get("decision") or event_type.rsplit(".", 1)[1]),
            )
            return
        if event_type == "llm.token":
            run_id = str(event.get("run_id") or "")
            token = str(event.get("token", ""))
            stream_block = self._agent_blocks.get(run_id)
            if stream_block is None:
                stream_block = Static("")
                self._agent_blocks[run_id] = stream_block
                self._agent_texts[run_id] = ""
                self._append(stream_block)
            text = self._agent_texts.get(run_id, "") + token
            self._agent_texts[run_id] = text
            stream_block.update(text)
            return
        if event_type == "llm.model_selected":
            self._append_text(
                f"model: {event.get('model', '')}  "
                f"strategy={event.get('strategy', '')}",
                style="dim",
            )
            return
        if event_type == "llm.usage":
            self._append_text(
                f"tokens: in={event.get('input_tokens', 0)} "
                f"out={event.get('output_tokens', 0)} "
                f"cache={event.get('cache_read_input_tokens', 0)}",
                style="dim",
            )
            return
        if event_type == "step.started":
            self._append_text(f"step {event.get('step', '')}", style="dim")
            return
        if event_type == "context.compacted":
            self._append_text(
                f"Context compacted: {event.get('original_tokens', 0)} → "
                f"{event.get('summary_tokens', 0)} tokens",
                style="dim cyan",
            )
            return
        if event_type == "skill.invoked":
            self._append_text(
                f"/{event.get('skill_name', '')}  {event.get('arguments', '')}",
                style="dim cyan",
            )
            return
        if event_type == "log.line":
            self._append_text(
                f"{event.get('level', 'INFO')} "
                f"{event.get('source', '')}: {event.get('message', '')}",
                style="dim",
            )

    # 将带样式的纯文本块追加到时间线
    def _append_text(self, content: str, *, style: str = "") -> None:
        self._append(Static(Text(content, style=style)))

    # 追加组件并将单滚屏滚动到底部
    def _append(self, widget: Static) -> None:
        timeline = self.query_one("#timeline", VerticalScroll)
        timeline.mount(widget)
        timeline.scroll_end(animate=False)


# 启动 Job TUI，可指定 job_id 或由应用自动恢复
def run(host: str, port: int, job_id: str | None = None) -> None:
    CyanTuiApp(host, port, job_id=job_id).run()
