from __future__ import annotations

import asyncio
import fcntl
import subprocess
import sys
import time
from pathlib import Path

from cyan.core.config import CyanConfig
from cyan.core.transport.socket_client import SocketClient

_CORE_LIFECYCLE_TIMEOUT_SECONDS = 15.0
_DAEMON_LOCK = Path("~/.cyan/cyan-core.lock").expanduser()


# 尝试连接 daemon，成功则正常返回，失败则抛出 ConnectionRefusedError/OSError
async def _ping_check(config: CyanConfig) -> None:
    _r, w = await asyncio.open_connection(config.host, config.port)
    w.close()
    await w.wait_closed()


# 尝试短暂获取 daemon 锁；成功后立即释放，只用于判断当前是否可启动
def _daemon_lock_is_free() -> bool:
    _DAEMON_LOCK.parent.mkdir(parents=True, exist_ok=True)
    lock = _DAEMON_LOCK.open("a+", encoding="utf-8")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return True
    finally:
        lock.close()


# 等待新 daemon 就绪，并区分当前子进程与并发启动成功的另一个 daemon
async def _wait_for_started(
    config: CyanConfig,
    process: subprocess.Popen[bytes],
) -> bool:
    deadline = time.monotonic() + _CORE_LIFECYCLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        returncode = process.poll()
        try:
            await _ping_check(config)
        except (ConnectionRefusedError, OSError):
            if returncode is not None and _daemon_lock_is_free():
                raise RuntimeError(
                    f"core exited before becoming ready (status {returncode})"
                ) from None
            await asyncio.sleep(0.05)
            continue
        await asyncio.sleep(0.05)
        return process.poll() is None
    raise RuntimeError(f"core did not start at {config.host}:{config.port}")


# 通过实际 daemon 连接发送结构化停止请求，避免依赖可陈旧或复用的 PID
async def _request_shutdown(config: CyanConfig) -> None:
    client = SocketClient(config.host, config.port)
    await client.connect()
    event_task = asyncio.create_task(client.run_event_loop())
    try:
        await asyncio.wait_for(
            client.send_command("core.shutdown", {}),
            timeout=3.0,
        )
    finally:
        event_task.cancel()
        try:
            await event_task
        except asyncio.CancelledError:
            pass
        await client.close()


# 等待 daemon 完全释放监听端口，防止紧接着 start 时误判旧进程仍可用
async def _wait_for_stopped(config: CyanConfig) -> None:
    deadline = time.monotonic() + _CORE_LIFECYCLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            await _ping_check(config)
        except (ConnectionRefusedError, OSError):
            return
        await asyncio.sleep(0.05)
    raise RuntimeError(f"core did not stop at {config.host}:{config.port}")


# 等待可启动槽位，并在另一个并发启动者先就绪时直接复用它
async def _wait_for_start_slot(config: CyanConfig) -> bool:
    deadline = time.monotonic() + _CORE_LIFECYCLE_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            await _ping_check(config)
        except (ConnectionRefusedError, OSError):
            if _daemon_lock_is_free():
                return True
        else:
            return False
        await asyncio.sleep(0.05)
    raise RuntimeError("core did not become available")


# 打印 daemon 当前状态（running / not running）
def cmd_core_status(config: CyanConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
        print(f"running  ({config.host}:{config.port})")
    except (ConnectionRefusedError, OSError):
        print("not running")


# 在后台启动 daemon，若已在运行则提示并退出
def cmd_core_start(config: CyanConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
        print(f"already running  ({config.host}:{config.port})")
        return
    except (ConnectionRefusedError, OSError):
        pass

    try:
        may_start = asyncio.run(_wait_for_start_slot(config))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if not may_start:
        print(f"already running  ({config.host}:{config.port})")
        return
    proc = subprocess.Popen(
        [sys.executable, "-m", "cyan.core"],
        start_new_session=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        started_here = asyncio.run(_wait_for_started(config, proc))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    if started_here:
        print(f"started  pid={proc.pid}  ({config.host}:{config.port})")
    else:
        print(f"already running  ({config.host}:{config.port})")


# 通过 RPC 停止实际监听的 daemon，并确认其完成清理和端口释放
def cmd_core_stop(config: CyanConfig) -> None:
    try:
        asyncio.run(_request_shutdown(config))
    except (ConnectionRefusedError, ConnectionError, OSError):
        print("not running")
        return
    try:
        asyncio.run(_wait_for_stopped(config))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print(f"stopped  ({config.host}:{config.port})")
