from __future__ import annotations

import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any

from cyan.service.transport.socket_client import SocketClient


# 功能：验证 VS Code/ TUI 共用的 launch.start 会启动真实 Job 并广播生命周期事件
# 设计：使用真实子进程和双进程 TCP daemon，不再调用已删除的 agent.run RPC
async def test_launch_start_returns_job_and_emits_events(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    del running_daemon
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    events: list[dict[str, Any]] = []

    # 收集 Job 生命周期事件
    async def on_event(event: dict[str, Any]) -> None:
        if str(event.get("type", "")).startswith("job."):
            events.append(event)

    client.on_event(on_event)
    event_task = asyncio.create_task(client.run_event_loop())
    try:
        preview = await client.send_command(
            "launch.preview",
            {
                "command": f"{sys.executable} -c \"print('ok')\"",
                "workspace_root": str(workspace),
            },
        )
        started = await client.send_command(
            "launch.start",
            {
                "command": f"{sys.executable} -c \"print('ok')\"",
                "workspace_root": str(workspace),
                "preview_fingerprint": preview["fingerprint"],
            },
        )
        job_id = str(started["job_id"])
        await client.send_command(
            "event.subscribe",
            {"topics": ["job.*"], "scope": f"job:{job_id}"},
        )
        deadline = asyncio.get_running_loop().time() + 5
        while True:
            snapshot = await client.send_command("job.get", {"job_id": job_id})
            if snapshot["job"]["status"] not in {"starting", "running"}:
                break
            if asyncio.get_running_loop().time() >= deadline:
                raise AssertionError("job did not finish")
            await asyncio.sleep(0.05)
        assert snapshot["job"]["status"] == "succeeded"
        assert any(event.get("type") == "job.finished" for event in events)
    finally:
        event_task.cancel()
        await asyncio.gather(event_task, return_exceptions=True)
        await client.close()


# 功能：验证两个独立客户端都能收到同一 Job 的 daemon 事件扇出
# 设计：在两个连接上订阅后启动真实命令，覆盖 IPC broadcaster 的双进程边界
async def test_two_clients_receive_job_broadcast(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
    tmp_path: Path,
) -> None:
    del running_daemon
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    client1 = SocketClient("127.0.0.1", free_port)
    client2 = SocketClient("127.0.0.1", free_port)
    await client1.connect()
    await client2.connect()
    received = [asyncio.Event(), asyncio.Event()]

    # 为每个客户端设置独立完成事件
    async def on_event1(event: dict[str, Any]) -> None:
        if event.get("type") == "job.finished":
            received[0].set()

    # 收集第二个客户端的完成事件
    async def on_event2(event: dict[str, Any]) -> None:
        if event.get("type") == "job.finished":
            received[1].set()

    client1.on_event(on_event1)
    client2.on_event(on_event2)
    loop1 = asyncio.create_task(client1.run_event_loop())
    loop2 = asyncio.create_task(client2.run_event_loop())
    try:
        preview = await client1.send_command(
            "launch.preview",
            {
                "command": f"{sys.executable} -c \"print('broadcast')\"",
                "workspace_root": str(workspace),
            },
        )
        started = await client1.send_command(
            "launch.start",
            {
                "command": f"{sys.executable} -c \"print('broadcast')\"",
                "workspace_root": str(workspace),
                "preview_fingerprint": preview["fingerprint"],
            },
        )
        job_id = str(started["job_id"])
        scope = {"topics": ["job.*"], "scope": f"job:{job_id}"}
        await client1.send_command("event.subscribe", scope)
        await client2.send_command("event.subscribe", scope)
        await asyncio.wait_for(asyncio.gather(*(event.wait() for event in received)), timeout=5)
    finally:
        loop1.cancel()
        loop2.cancel()
        await asyncio.gather(loop1, loop2, return_exceptions=True)
        await client1.close()
        await client2.close()
