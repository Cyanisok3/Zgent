from __future__ import annotations

import asyncio
import os
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

from cyan.cli.commands.core import _ping_check, cmd_core_start
from cyan.core.bus.commands import WIRE_PROTOCOL_VERSION
from cyan.core.config import CyanConfig
from cyan.core.transport.socket_client import SocketClient
from cyan.tui.app import CyanTuiApp

_CORE_START_TIMEOUT_SECONDS = 5.0


# 等待后台 daemon 接受连接，超时则报告启动失败
async def _wait_for_core(config: CyanConfig) -> None:
    deadline = time.monotonic() + _CORE_START_TIMEOUT_SECONDS
    while True:
        try:
            await _ping_check(config)
            return
        except (ConnectionRefusedError, OSError):
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"core did not start at {config.host}:{config.port}"
                ) from None
            await asyncio.sleep(0.05)


# 确保 daemon 已运行，未运行时自动后台启动并等待就绪
def ensure_core(config: CyanConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
        return
    except (ConnectionRefusedError, OSError):
        cmd_core_start(config)
    asyncio.run(_wait_for_core(config))


# 向 daemon 创建任务并在收到 job_id 后关闭一次性连接
async def _start_job(
    config: CyanConfig,
    argv: Sequence[str],
    workspace_root: Path,
    env: Mapping[str, str],
) -> str:
    client = SocketClient(config.host, config.port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    try:
        pong = await client.send_command("core.ping", {"client": "cyan-watch"})
        observed = pong.get("protocol_version")
        if observed != WIRE_PROTOCOL_VERSION:
            raise RuntimeError(
                "wire protocol mismatch: "
                f"daemon={observed} client={WIRE_PROTOCOL_VERSION}"
            )
        result = await client.send_command(
            "job.start",
            {
                "argv": list(argv),
                "workspace_root": str(workspace_root.resolve()),
                "env": dict(env),
            },
        )
        return str(result["job_id"])
    finally:
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        await client.close()


# 启动真实长任务并立即进入对应 Job TUI
def cmd_watch(argv: Sequence[str], config: CyanConfig) -> None:
    if not argv:
        raise ValueError("watch requires a command after --")
    ensure_core(config)
    job_id = asyncio.run(_start_job(config, argv, Path.cwd(), os.environ))
    CyanTuiApp(config.host, config.port, job_id=job_id).run()


# 确保 daemon 运行并启动自动恢复最近任务的 Job TUI
def cmd_job_tui(config: CyanConfig) -> None:
    ensure_core(config)
    CyanTuiApp(config.host, config.port).run()
