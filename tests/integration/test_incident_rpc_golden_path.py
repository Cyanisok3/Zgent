from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any, cast

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.types import LlmResponse, ToolCallBlock
from cyan.service.app import CoreApp
from cyan.service.transport.socket_client import SocketClient


# 从消息尾部读取最近一次工具返回的 JSON
def _latest_tool_json(messages: list[dict[str, object]]) -> dict[str, Any]:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(cast(list[dict[str, object]], content)):
            if block.get("type") == "tool_result":
                return cast(dict[str, Any], json.loads(str(block.get("content", "{}"))))
    raise AssertionError("tool result JSON missing")


# 从消息尾部读取最近一次工具返回的文本
def _latest_tool_text(messages: list[dict[str, object]]) -> str:
    for message in reversed(messages):
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for block in reversed(cast(list[dict[str, object]], content)):
            if block.get("type") == "tool_result":
                return str(block.get("content", ""))
    raise AssertionError("tool result text missing")


class _GoldenProvider:
    # 初始化一个只用于集成测试的确定性 Incident Provider
    def __init__(self) -> None:
        self._run_id: str | None = None
        self._evidence: list[dict[str, str]] = []
        self.chat_calls = 0

    # 按真实工具返回结果推进诊断和提案顺序
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del bus, system
        self.chat_calls += 1
        if run_id != self._run_id:
            self._run_id = run_id
            self._evidence = []

        names = {str(schema.get("name")) for schema in tool_schemas}
        if step < 4:
            assert names == {
                "read_file",
                "list_dir",
                "search_text",
                "read_job_log",
                "submit_diagnosis",
                "propose_patch",
            }
        else:
            assert names == {"propose_patch"}

        if step == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-stderr",
                        name="read_job_log",
                        input={"stream": "stderr", "mode": "tail"},
                    )
                ],
            )
        if step == 2:
            log_result = _latest_tool_json(messages)
            reference = str(log_result["reference"])
            self._evidence = [
                {
                    "source": "stderr",
                    "reference": reference,
                    "description": "The real process emitted the crash marker.",
                }
            ]
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-source",
                        name="read_file",
                        input={"path": "train.py"},
                    )
                ],
            )
        if step == 3:
            source = _latest_tool_text(messages)
            match = re.search(r"\[source ([^\]]+)\]", source)
            assert match is not None
            self._evidence.append(
                {
                    "source": "workspace",
                    "reference": match.group(1),
                    "description": "The source contains the failing exit.",
                }
            )
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="submit-diagnosis",
                        name="submit_diagnosis",
                        input={
                            "category": "runtime",
                            "summary": "The training script exits with status 2.",
                            "root_cause": (
                                "train.py prints the failure marker and exits with status 2."
                            ),
                            "evidence": self._evidence,
                            "confidence": 1.0,
                            "causal_support": "direct",
                            "patch_recommended": True,
                        },
                    )
                ],
            )
        if step == 4:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="propose-patch",
                        name="propose_patch",
                        input={
                            "path": "train.py",
                            "search": 'print("boom", file=sys.stderr)\nsys.exit(2)',
                            "replace": 'print("recovered")',
                            "evidence": self._evidence,
                        },
                    )
                ],
            )
        return LlmResponse(stop_reason="end_turn", text="completed")


# 执行 Git 命令并要求成功
def _git(root: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(root), *args],
        check=True,
        capture_output=True,
        text=True,
    )


# 创建失败训练脚本、Smoke 配置和真实临时 Git 工作区
def _make_workspace(root: Path, smoke_executable: str) -> Path:
    root.mkdir()
    (root / "train.py").write_text(
        'import sys\nprint("boom", file=sys.stderr)\nsys.exit(2)\n',
        encoding="utf-8",
    )
    config = root / ".cyan" / "config.toml"
    config.parent.mkdir()
    config.write_text(
        "[incident.smoke]\n"
        f"argv = {json.dumps([smoke_executable, '-q', 'recovered', 'train.py'])}\n"
        "timeout_s = 5\n",
        encoding="utf-8",
    )
    _git(root, "init")
    _git(root, "config", "user.email", "cyan@example.invalid")
    _git(root, "config", "user.name", "cyan test")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "fixture")
    return root


# 等待真实 daemon 开始监听测试端口
async def _wait_for_server(app_task: asyncio.Task[None], port: int) -> None:
    for _ in range(300):
        if app_task.done():
            await app_task
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", port)
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(0.01)
            continue
        del reader
        writer.close()
        await writer.wait_closed()
        return
    raise AssertionError("CoreApp did not start within 3 seconds")


# 轮询公开 job.get 直到 Incident 达到目标状态
async def _wait_for_incident(
    client: SocketClient,
    job_id: str,
    status: str,
) -> dict[str, Any]:
    for _ in range(500):
        view = await client.send_command("job.get", {"job_id": job_id})
        incident = view.get("incident")
        if isinstance(incident, dict) and incident.get("status") == status:
            return view
        await asyncio.sleep(0.02)
    raise AssertionError(f"Incident did not reach {status}")


# 轮询公开 job.get 直到 Job 与 Incident 都完成闭环
async def _wait_for_resolved(
    client: SocketClient,
    job_id: str,
) -> dict[str, Any]:
    for _ in range(500):
        view = await client.send_command("job.get", {"job_id": job_id})
        job = view.get("job")
        incident = view.get("incident")
        if (
            isinstance(job, dict)
            and job.get("status") == "succeeded"
            and isinstance(incident, dict)
            and incident.get("status") == "resolved"
        ):
            return view
        await asyncio.sleep(0.02)
    raise AssertionError("Job and Incident did not resolve")


# 功能：验证公开 TCP RPC 能驱动真实训练失败、审批、Smoke 和原命令重跑
# 设计：使用临时 HOME、真实 Git/子进程与确定性 Provider，不调用外部模型
async def test_public_rpc_incident_recovery_golden_path(
    tmp_path: Path,
    free_port: int,
    monkeypatch: Any,
) -> None:
    home = tmp_path / "home"
    home.mkdir()
    config_path = tmp_path / "cyan-config.toml"
    config_path.write_text(
        "[core]\n"
        "host = \"127.0.0.1\"\n"
        f"port = {free_port}\n"
        "[logging]\n"
        "level = \"WARNING\"\n"
        "file = \"\"\n"
        "[trace]\n"
        "enabled = false\n"
        "[llm]\n"
        "default_model = \"test-provider\"\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("CYAN_CONFIG", str(config_path))
    monkeypatch.setenv("CYAN_HOST", "127.0.0.1")
    monkeypatch.setenv("CYAN_PORT", str(free_port))
    monkeypatch.setenv("CYAN_LOG_LEVEL", "WARNING")
    monkeypatch.setenv("CYAN_LOG_FILE", "")
    monkeypatch.setenv("CYAN_LLM_DEFAULT_MODEL", "test-provider")
    monkeypatch.setenv("CYAN_TRACE_ENABLED", "false")
    smoke_executable = shutil.which("grep")
    assert smoke_executable is not None
    workspace = _make_workspace(tmp_path / "workspace", smoke_executable)
    original = (workspace / "train.py").read_text(encoding="utf-8")
    environment = {
        "PATH": os.environ["PATH"],
        "HOME": str(home),
        "PYTHONIOENCODING": "utf-8",
    }

    provider = _GoldenProvider()
    app = CoreApp(provider=provider)
    app_task = asyncio.create_task(app.run())
    client: SocketClient | None = None
    event_task: asyncio.Task[None] | None = None
    events: list[dict[str, Any]] = []

    # 收集公共事件摘要，确认生命周期事件经过同一 TCP 连接
    async def on_event(event: dict[str, Any]) -> None:
        events.append(event)

    try:
        await _wait_for_server(app_task, free_port)
        client = SocketClient("127.0.0.1", free_port)
        await client.connect()
        client.on_event(on_event)
        event_task = asyncio.create_task(client.run_event_loop())
        await client.send_command(
            "event.subscribe",
            {
                "topics": [
                    "job.*",
                    "incident.*",
                    "run.*",
                    "tool.*",
                    "patch.*",
                    "smoke.*",
                ],
                "scope": "global",
            },
        )

        command = f"{sys.executable} train.py"
        preview = await client.send_command(
            "launch.preview",
            {
                "command": command,
                "workspace_root": str(workspace),
                "env": environment,
            },
        )
        assert preview["argv"] == [sys.executable, "train.py"]
        started = await client.send_command(
            "launch.start",
            {
                "command": command,
                "workspace_root": str(workspace),
                "env": environment,
                "preview_fingerprint": preview["fingerprint"],
            },
        )
        job_id = str(started["job_id"])

        awaiting = await _wait_for_incident(client, job_id, "awaiting_approval")
        incident = cast(dict[str, Any], awaiting["incident"])
        proposal = cast(dict[str, Any], awaiting["proposal"])
        assert cast(dict[str, Any], awaiting["job"])["status"] == "failed"
        assert cast(dict[str, Any], awaiting["attempt"])["returncode"] == 2
        assert (workspace / "train.py").read_text(encoding="utf-8") == original
        assert proposal["id"] == incident["active_proposal_id"]
        proposal_path = home / ".cyan" / "jobs" / job_id / "incidents" / incident["id"] / "proposal.diff"
        assert proposal_path.exists()

        review = await client.send_command(
            "incident.review",
            {
                "job_id": job_id,
                "incident_id": incident["id"],
                "proposal_id": proposal["id"],
            },
        )
        assert review["path"] == "train.py"
        assert review["before_text"] == original
        assert "recovered" in review["after_text"]

        decision = await client.send_command(
            "incident.decide",
            {
                "incident_id": incident["id"],
                "proposal_id": proposal["id"],
                "decision": "approve",
                "run_smoke": True,
                "smoke_config_fingerprint": awaiting["smoke_config_fingerprint"],
            },
        )
        assert decision["status"] == "retry_running"

        resolved = await _wait_for_resolved(client, job_id)
        final_job = cast(dict[str, Any], resolved["job"])
        final_attempt = cast(dict[str, Any], resolved["attempt"])
        final_incident = cast(dict[str, Any], resolved["incident"])
        assert final_job["attempt_ids"] == ["attempt-0001", "attempt-0002"]
        assert final_attempt["status"] == "succeeded"
        assert final_attempt["returncode"] == 0
        assert final_incident["apply_receipt"] is not None
        assert final_incident["smoke_result"]["status"] == "passed"
        if final_incident["smoke_execution"] is not None:
            assert final_incident["smoke_execution"]["status"] == "passed"
        assert "recovered" in (workspace / "train.py").read_text(encoding="utf-8")
        retry_stdout = (
            home
            / ".cyan"
            / "jobs"
            / job_id
            / "attempts"
            / "attempt-0002"
            / "stdout.log"
        )
        assert "recovered" in retry_stdout.read_text(encoding="utf-8")
        assert provider.chat_calls > 0
        event_types = {str(event.get("type")) for event in events}
        assert {"job.started", "incident.opened", "run.started", "patch.proposed"} <= event_types
        assert "smoke.finished" in event_types
    finally:
        if client is not None:
            with suppress(Exception):
                await client.send_command("core.shutdown", {})
            if event_task is not None:
                event_task.cancel()
                with suppress(asyncio.CancelledError):
                    await event_task
            await client.close()
        if not app_task.done():
            with suppress(Exception):
                await asyncio.wait_for(app_task, timeout=5)
