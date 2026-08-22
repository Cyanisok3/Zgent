from __future__ import annotations

import asyncio
import os
import sys
from contextlib import suppress
from pathlib import Path
from typing import Any

from cyan.service.transport.socket_client import SocketClient


# 功能：验证 VS Code 使用的 preview/start/log 协议贯穿真实 daemon 和子进程
# 设计：在临时工作区运行真实训练脚本并轮询持久化 Job，不伪造前端成功结果
async def test_vscode_launch_protocol_runs_real_training(
    running_daemon: Any,
    free_port: int,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "vscode-workspace"
    workspace.mkdir()
    (workspace / "train.py").write_text(
        "import os\nprint('vscode-training-' + os.environ['MODE'])\n",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    environment["PRIVATE_TOKEN"] = "must-not-return"
    command = f"MODE=ok {sys.executable} train.py"
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())

    try:
        preview = await client.send_command(
            "launch.preview",
            {
                "command": command,
                "workspace_root": str(workspace),
                "env": environment,
            },
        )
        assert preview["env_overrides"] == {"MODE": "ok"}
        assert "PRIVATE_TOKEN" not in str(preview)

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
        deadline = asyncio.get_running_loop().time() + 5.0
        while True:
            snapshot = await client.send_command("job.get", {"job_id": job_id})
            if snapshot["job"]["status"] not in {"starting", "running"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("VS Code protocol training did not finish")
            await asyncio.sleep(0.05)

        assert snapshot["job"]["status"] == "succeeded"
        attempt_id = str(snapshot["attempt"]["id"])
        log = await client.send_command(
            "job.read_log",
            {
                "job_id": job_id,
                "attempt_id": attempt_id,
                "stream": "stdout",
                "offset": 0,
                "limit": 32 * 1024,
            },
        )
        assert "vscode-training-ok" in log["data"]
        assert "env" not in snapshot
    finally:
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
        await client.close()


# 功能：验证 VS Code 启动请求不能绕过已确认的 LaunchSpec 指纹
# 设计：通过真实 daemon 预览后改变环境并检查启动前结构化拒绝
async def test_vscode_launch_protocol_rejects_stale_preview(
    running_daemon: Any,
    free_port: int,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "stale-workspace"
    workspace.mkdir()
    command = f"{sys.executable} -c \"print('ok')\""
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    first_env = {"PATH": os.environ["PATH"], "VALUE": "first"}

    try:
        preview = await client.send_command(
            "launch.preview",
            {
                "command": command,
                "workspace_root": str(workspace),
                "env": first_env,
            },
        )
        try:
            await client.send_command(
                "launch.start",
                {
                    "command": command,
                    "workspace_root": str(workspace),
                    "env": {"PATH": os.environ["PATH"], "VALUE": "second"},
                    "preview_fingerprint": preview["fingerprint"],
                },
            )
        except Exception as error:
            assert "preview is stale" in str(error)
        else:
            raise AssertionError("stale preview unexpectedly started a Job")
    finally:
        event_task.cancel()
        with suppress(asyncio.CancelledError):
            await event_task
        await client.close()
