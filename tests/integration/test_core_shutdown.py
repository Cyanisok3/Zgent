from __future__ import annotations

import asyncio
import subprocess

from cyan.service.transport.socket_client import SocketClient


# 功能：验证停止 RPC 会命中真实监听的 daemon，并等待统一清理流程正常退出
# 设计：连接测试 daemon 发送 core.shutdown，断言响应后进程以零状态结束而非依赖 PID 信号
async def test_core_shutdown_roundtrip(
    running_daemon: subprocess.Popen[bytes],
    free_port: int,
) -> None:
    client = SocketClient("127.0.0.1", free_port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    try:
        result = await client.send_command("core.shutdown", {})
        assert result == {"status": "stopping"}
    finally:
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        await client.close()

    returncode = await asyncio.to_thread(running_daemon.wait, 5)
    assert returncode == 0
