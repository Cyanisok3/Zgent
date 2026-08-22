from __future__ import annotations

import asyncio
import logging
import os
import signal

logger = logging.getLogger(__name__)


# 检查 PID 是否仍存在；权限不足时返回 None 表示无法确认
def _pid_exists(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    return True


# 读取不含 argv 的 OS 进程启动身份，用于持久化记录规避 PID 复用
async def read_process_identity(pid: int) -> str | None:
    try:
        process = await asyncio.create_subprocess_exec(
            "ps",
            "-o",
            "lstart=",
            "-p",
            str(pid),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        stdout, _ = await asyncio.wait_for(process.communicate(), timeout=2.0)
    except (OSError, TimeoutError):
        return None
    if process.returncode != 0:
        return None
    identity = stdout.decode("utf-8", errors="replace").strip()
    return identity or None


# 仅在 PID、启动身份、session leader 和进程组 leader 全匹配时终止遗留进程
async def terminate_owned_process_group(
    pid: int,
    process_identity: str,
    *,
    grace_s: float = 2.0,
) -> bool:
    if grace_s <= 0:
        raise ValueError("grace_s must be positive")
    observed_identity = await read_process_identity(pid)
    if observed_identity != process_identity:
        if observed_identity is not None:
            return True
        return _pid_exists(pid) is False
    try:
        if os.getsid(pid) != pid or os.getpgid(pid) != pid:
            logger.warning("refusing to signal recovered non-leader pid=%s", pid)
            return False
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    deadline = asyncio.get_running_loop().time() + grace_s
    while asyncio.get_running_loop().time() < deadline:
        observed_identity = await read_process_identity(pid)
        if observed_identity != process_identity:
            if observed_identity is not None or _pid_exists(pid) is False:
                return True
        await asyncio.sleep(0.05)
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        return True
    except PermissionError:
        return False
    kill_deadline = asyncio.get_running_loop().time() + grace_s
    while asyncio.get_running_loop().time() < kill_deadline:
        observed_identity = await read_process_identity(pid)
        if observed_identity != process_identity:
            if observed_identity is not None or _pid_exists(pid) is False:
                return True
        await asyncio.sleep(0.05)
    return False
