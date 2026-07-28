from __future__ import annotations

import asyncio
import os
import stat
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyan.core.transport.socket_client import SocketClient
from cyan.tui.app import CyanTuiApp
from cyan.tui.launch import parse_training_command


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
    app._pending_launch = parse_training_command(
        f"CYAN_TUI_TEST=private {sys.executable} train.py",
        workspace,
        os.environ,
    )

    try:
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
