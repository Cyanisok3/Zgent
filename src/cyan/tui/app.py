from __future__ import annotations

import asyncio
import json
import shlex
from contextlib import suppress
from datetime import UTC, datetime
from typing import Any

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll
from textual.css.query import NoMatches
from textual.widgets import Input, Label, OptionList, Static
from textual.widgets.option_list import Option

from cyan.core.transport.socket_client import IpcError, SocketClient

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


# 根据已冻结的 smoke verifier 配置展示实际命令、超时和审批选项
def _approval_hint(
    smoke_config: dict[str, Any] | None,
    *,
    can_apply: bool = True,
) -> str:
    if not can_apply:
        return "Non-Git workspace: patch is review-only. [X] reject"
    if smoke_config:
        argv = smoke_config.get("argv", [])
        command = (
            shlex.join([str(value) for value in argv])
            if isinstance(argv, list)
            else str(argv)
        )
        timeout = smoke_config.get("timeout_s", 300)
        return (
            f"Smoke verifier before full retry:\n$ {command}\n"
            f"timeout={timeout}s\n"
            "[A] approve + smoke    [R] approve without smoke    [X] reject"
        )
    return "[A] approve + retry    [X] reject"


# 兼容日志 RPC 字段并返回文本、下一字节位置和 EOF
def _log_result(result: dict[str, Any]) -> tuple[str, int, bool]:
    data = str(result.get("data", result.get("text", "")))
    next_offset = int(result.get("next_offset", result.get("end_offset", 0)))
    return data, next_offset, bool(result.get("eof", False))


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


class CyanTuiApp(App[None]):
    """真实训练进程、Incident 调查和验证共用的单滚屏界面。"""

    TITLE = "cyan Incident"
    BINDINGS = [
        Binding("ctrl+q", "quit", "detach"),
        Binding("a", "approve_smoke", show=False),
        Binding("r", "approve_without_smoke", show=False),
        Binding("x", "reject", show=False),
        Binding("c", "cancel_job", show=False),
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
    #job-picker { height: auto; max-height: 16; margin: 1 0; }
    #followup {
        display: none;
        height: 3;
        margin: 0 1;
        border: round $surface-lighten-2;
    }
    """

    # 初始化连接、附着目标和增量日志游标
    def __init__(self, host: str, port: int, job_id: str | None = None) -> None:
        super().__init__()
        self._host = host
        self._port = port
        self._job_id = job_id
        self._job_event_seq = 0
        self._client: SocketClient | None = None
        self._selection_ready = asyncio.Event()
        self._attempt_id: str | None = None
        self._offsets = {"stdout": 0, "stderr": 0}
        self._snapshot: dict[str, Any] = {}
        self._incident_id: str | None = None
        self._proposal_id: str | None = None
        self._session_id: str | None = None
        self._smoke_config: dict[str, Any] | None = None
        self._smoke_config_fingerprint: str | None = None
        self._can_apply = True
        self._awaiting_approval = False
        self._subscribed_run_ids: set[str] = set()
        self._run_subscription_lock = asyncio.Lock()
        self._shown_diagnosis_id: str | None = None
        self._shown_proposal_id: str | None = None
        self._shown_smoke: str | None = None
        self._shown_incident_status: str | None = None
        self._seen_agent_events: set[tuple[str, str, str]] = set()
        self._agent_block: Static | None = None
        self._agent_text = ""

    # 组合固定头部、单滚屏时间线和按需出现的追问输入框
    def compose(self) -> ComposeResult:
        yield Label("[bold cyan]cyan[/bold cyan]  [dim]connecting...[/dim]", id="job-header")
        yield VerticalScroll(
            Static(
                "[dim]Ctrl+Q detaches; the training process keeps running.[/dim]",
                id="detach-note",
            ),
            id="timeline",
        )
        yield Input(placeholder="Ask a follow-up about this incident", id="followup")

    # 挂载后启动唯一的连接与轮询 worker
    def on_mount(self) -> None:
        self.run_worker(self._connection_loop(), exclusive=True, name="job-connection")

    # Ctrl+Q 只退出 TUI，不向 daemon 发送取消或关闭会话命令
    async def action_quit(self) -> None:
        self.exit()

    # 连接 daemon、选择任务、订阅事件并持续增量读取状态与日志
    async def _connection_loop(self) -> None:
        reconnecting = False
        while True:
            client = SocketClient(self._host, self._port)
            event_task: asyncio.Task[None] | None = None
            try:
                await client.connect()
                self._client = client
                self._subscribed_run_ids.clear()
                client.on_event(self._handle_event)
                event_task = asyncio.create_task(client.run_event_loop())
                if reconnecting:
                    self._append_text("Reconnected to daemon.", style="bold cyan")
                if self._job_id is None:
                    await self._choose_job()
                if self._job_id is None:
                    return
                await self._subscribe_job_events()
                while not event_task.done():
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

    # 切换附着目标时重置仅属于旧 Job 的事件游标
    def _select_job(self, job_id: str) -> None:
        if job_id != self._job_id:
            self._job_event_seq = 0
        self._job_id = job_id

    # 自动附着唯一任务，存在多个任务时显示极简选择器
    async def _choose_job(self) -> None:
        if self._client is None:
            return
        result = await self._client.send_command("job.list", {})
        raw_jobs = result.get("jobs", [])
        jobs = _actionable_jobs(
            [entry for entry in raw_jobs if isinstance(entry, dict)]
            if isinstance(raw_jobs, list)
            else []
        )
        if not jobs:
            self._append_text("No active or pending job. Run `cyan watch -- <command>`.")
            return
        if len(jobs) == 1:
            self._select_job(str(_job_record(jobs[0]).get("id", "")))
            return
        options = [
            Option(escape(_job_label(entry)), id=str(_job_record(entry).get("id", "")))
            for entry in jobs
        ]
        self._append_text("Choose a job to attach:")
        self.query_one("#timeline", VerticalScroll).mount(
            OptionList(*options, id="job-picker", compact=True)
        )
        await self._selection_ready.wait()

    # 接收选择器结果并唤醒等待附着的 worker
    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_id:
            self._select_job(event.option_id)
            self._selection_ready.set()
        event.option_list.remove()

    # 拉取最新结构化状态并仅渲染尚未展示的 Incident 制品
    async def _refresh_snapshot(self) -> None:
        if self._client is None or self._job_id is None:
            return
        self._snapshot = await self._client.send_command("job.get", {"job_id": self._job_id})
        await self._subscribe_active_run()
        self._render_header()
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
                    "topics": ["run.*", "tool.*", "llm.*"],
                    "scope": f"run:{run_id}",
                    "replay_from_run": run_id,
                },
            )
            if client is self._client:
                self._subscribed_run_ids.add(run_id)

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
        for stream in ("stdout", "stderr"):
            result = await self._client.send_command(
                "job.read_log",
                {
                    "job_id": self._job_id,
                    "attempt_id": attempt_id,
                    "stream": stream,
                    "offset": self._offsets[stream],
                    "limit": 32 * 1024,
                },
            )
            data, next_offset, _eof = _log_result(result)
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
        self._session_id = str(incident.get("session_id") or "") or None
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
                self.query_one("#incident-actions", Static).remove()
        self._set_followup_visible(
            self._session_id is not None and status in _FOLLOWUP_INCIDENT_STATUSES
        )

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

    # 创建或更新唯一的审批提示，避免轮询产生重复块
    def _append_or_update_actions(self) -> None:
        hint = _approval_hint(self._smoke_config, can_apply=self._can_apply)
        try:
            self.query_one("#incident-actions", Static).update(Text(hint))
        except NoMatches:
            self._append(Static(Text(hint), id="incident-actions"))

    # 根据 Incident session 是否存在显隐追问输入框
    def _set_followup_visible(self, visible: bool) -> None:
        prompt = self.query_one("#followup", Input)
        prompt.styles.display = "block" if visible else "none"

    # A 键批准补丁；有 smoke 配置时先 smoke，否则直接完整重跑
    def action_approve_smoke(self) -> None:
        self.run_worker(
            self._decide("approve", run_smoke=self._smoke_config is not None),
            name="incident-decision",
            exclusive=False,
        )

    # R 键明确跳过可选 smoke 后批准补丁
    def action_approve_without_smoke(self) -> None:
        self.run_worker(
            self._decide("approve", run_smoke=False),
            name="incident-decision",
            exclusive=False,
        )

    # X 键拒绝当前补丁
    def action_reject(self) -> None:
        self.run_worker(
            self._decide("reject", run_smoke=False),
            name="incident-decision",
            exclusive=False,
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
        await self._client.send_command(
            "incident.decide",
            payload,
        )
        smoke_note = " with smoke" if decision == "approve" and run_smoke else ""
        self._append_text(f"{decision} submitted{smoke_note}", style="bold")

    # C 键显式取消当前训练任务，和 Ctrl+Q 的 detach 语义严格分离
    def action_cancel_job(self) -> None:
        self.run_worker(self._cancel_job(), name="job-cancel", exclusive=False)

    # 向 daemon 发送用户明确触发的 job.cancel
    async def _cancel_job(self) -> None:
        if self._client is None or self._job_id is None:
            return
        await self._client.send_command("job.cancel", {"job_id": self._job_id})
        self._append_text("Cancellation requested", style="yellow")

    # 提交追问后复用同一只读 Incident session 并订阅对应 run
    def on_input_submitted(self, event: Input.Submitted) -> None:
        content = event.value.strip()
        if not content or self._client is None or self._session_id is None:
            return
        event.input.value = ""
        self._append_text(f"> {content}", style="bold")
        self.run_worker(
            self._send_followup(content),
            name="incident-followup",
            exclusive=False,
        )

    # 在独立 worker 中发送追问，避免 Agent 运行期间冻结 TUI 消息泵
    async def _send_followup(self, content: str) -> None:
        if self._client is None or self._session_id is None:
            return
        result = await self._client.send_command(
            "session.send_message",
            {"session_id": self._session_id, "content": content},
        )
        run_id = str(result.get("run_id", ""))
        await self._subscribe_run(run_id)

    # 将 Agent 流式文本和工具摘要追加到同一 Incident 时间线
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
        elif event_type in {"tool.call_started", "run.finished"}:
            identity = (
                str(event.get("tool_use_id") or event.get("ts") or "")
                if event_type == "tool.call_started"
                else str(event.get("status") or "")
            )
            event_key = (event_type, str(event.get("run_id") or ""), identity)
            if event_key in self._seen_agent_events:
                return
            self._seen_agent_events.add(event_key)
            if event_type == "tool.call_started":
                self._agent_block = None
                self._append_text(f"tool: {event.get('tool_name', '')}", style="dim")
            else:
                self._agent_block = None
                self._append_text(f"Agent run: {event.get('status', '')}", style="dim")
        elif event_type == "llm.token":
            token = str(event.get("token", ""))
            if self._agent_block is None:
                self._agent_text = ""
                self._agent_block = Static("")
                self._append(self._agent_block)
            self._agent_text += token
            self._agent_block.update(self._agent_text)

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
