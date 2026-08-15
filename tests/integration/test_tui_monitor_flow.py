from __future__ import annotations

import asyncio
import os
import stat
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyan.core.transport.socket_client import SocketClient
from cyan.tui.app import ChatTextArea, CyanTuiApp


# 功能：验证 TUI 预览确认通过真实 RPC 启动子进程并保持 launch.json 私有
# 设计：连接真实 daemon、运行临时训练脚本并轮询 JobStore 结果，不伪造前端成功状态
async def test_monitor_start_runs_real_job_and_keeps_launch_private(
    running_daemon: Any,
    free_port: int,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    script = workspace / "train.py"
    script.write_text("print('training-ok')\n")

    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    app = CyanTuiApp("127.0.0.1", free_port, workspace_root=workspace)
    app._client = client
    app._append_text = lambda content, style="": None  # type: ignore[method-assign]
    app._update_prompt = lambda: None  # type: ignore[method-assign]

    try:
        await app._preview_launch(
            f"CYAN_TUI_TEST=private {sys.executable} train.py"
        )
        await app._start_pending_launch()
        assert app._job_id is not None
        deadline = asyncio.get_running_loop().time() + 3.0
        while True:
            snapshot = await client.send_command("job.get", {"job_id": app._job_id})
            status = str(snapshot["job"]["status"])
            if status not in {"starting", "running"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("training job did not finish")
            await asyncio.sleep(0.05)

        assert status == "succeeded"
        assert "env" not in snapshot
        launch_path = tmp_path / ".cyan" / "jobs" / app._job_id / "launch.json"
        assert stat.S_IMODE(launch_path.stat().st_mode) == 0o600
        assert "CYAN_TUI_TEST" in launch_path.read_text()
    finally:
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
        await client.close()


# 功能：验证 TUI 附着真实长日志任务时只展示尾部并从末端游标继续
# 设计：经真实 daemon 运行输出 5 MiB 的子进程，再由 TUI 通过真实 RPC 读取并检查尾部标记
async def test_attached_tui_reads_real_long_log_tail(
    running_daemon: Any,
    free_port: int,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "tail-workspace"
    workspace.mkdir()
    script = workspace / "long_log.py"
    script.write_text(
        "import sys\n"
        "sys.stdout.write('PREFIX-MARKER\\n')\n"
        "sys.stdout.write('x' * (5 * 1024 * 1024))\n"
        "sys.stdout.write('\\nTAIL-MARKER\\n')\n",
        encoding="utf-8",
    )
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())

    try:
        started = await client.send_command(
            "job.start",
            {
                "argv": [sys.executable, "long_log.py"],
                "workspace_root": str(workspace),
                "env": dict(os.environ),
            },
        )
        job_id = str(started["job_id"])
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            snapshot = await client.send_command("job.get", {"job_id": job_id})
            if snapshot["job"]["status"] == "succeeded":
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("long-log training job did not finish")
            await asyncio.sleep(0.05)

        app = CyanTuiApp("127.0.0.1", free_port, job_id=job_id)
        rendered_logs: list[str] = []
        rendered_notes: list[str] = []
        app._client = client
        app._snapshot = snapshot
        app._append_log = (  # type: ignore[method-assign]
            lambda stream, data: rendered_logs.append(data)
        )
        app._append_text = (  # type: ignore[method-assign]
            lambda content, style="": rendered_notes.append(content)
        )

        await app._read_logs()

        stdout = "".join(rendered_logs)
        assert "TAIL-MARKER" in stdout
        assert "PREFIX-MARKER" not in stdout
        assert len(stdout.encode()) <= 32 * 1024
        assert any("showing the last 32768" in note for note in rendered_notes)
    finally:
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
        await client.close()


# 功能：验证多个真实运行中 Job 不阻塞键盘输入的 /monitor 新任务流程
# 设计：通过真实 daemon 先启动两个长任务，再在 Textual DOM 输入命令并确认第三个子进程成功
async def test_multiple_jobs_keep_real_tui_monitor_flow_available(
    running_daemon: Any,
    free_port: int,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "multi-workspace"
    workspace.mkdir()
    (workspace / "sleep.py").write_text(
        "import time\ntime.sleep(30)\n",
        encoding="utf-8",
    )
    (workspace / "train.py").write_text("print('training-ok')\n", encoding="utf-8")
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    historical_job_ids: list[str] = []

    try:
        for _index in range(2):
            result = await client.send_command(
                "job.start",
                {
                    "argv": [sys.executable, "sleep.py"],
                    "workspace_root": str(workspace),
                    "env": dict(os.environ),
                },
            )
            historical_job_ids.append(str(result["job_id"]))

        app = CyanTuiApp("127.0.0.1", free_port, workspace_root=workspace)
        async with app.run_test() as pilot:
            deadline = asyncio.get_running_loop().time() + 3.0
            while app._client is None or not app._skip_job_restore:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("TUI did not reach non-blocking multi-job state")
                await asyncio.sleep(0.05)

            prompt = app.query_one("#prompt", ChatTextArea)
            assert prompt.has_focus
            await pilot.press(*list("/monitor"), "enter", "enter")
            await pilot.pause()
            assert app._input_mode == "monitor"

            prompt.text = f"{sys.executable} train.py"
            await pilot.press("enter")
            deadline = asyncio.get_running_loop().time() + 2.0
            while app._pending_launch is None:
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError("training command preview was not created")
                await asyncio.sleep(0.05)
            prompt.text = "/start"
            await pilot.press("enter", "enter")
            await pilot.pause()

            deadline = asyncio.get_running_loop().time() + 5.0
            while True:
                if app._job_id is not None and app._job_id not in historical_job_ids:
                    snapshot = await client.send_command(
                        "job.get",
                        {"job_id": app._job_id},
                    )
                    if snapshot["job"]["status"] == "succeeded":
                        break
                if asyncio.get_running_loop().time() >= deadline:
                    raise AssertionError(
                        "TUI-started training did not succeed "
                        f"(job_id={app._job_id}, pending={app._pending_launch is not None})"
                    )
                await asyncio.sleep(0.05)
    finally:
        for job_id in historical_job_ids:
            with suppress(Exception):
                await client.send_command("job.cancel", {"job_id": job_id})
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
        await client.close()
