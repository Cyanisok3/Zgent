from __future__ import annotations

import subprocess
from unittest.mock import AsyncMock, Mock

from cyan.cli.commands import core
from cyan.core.config import CyanConfig


# 功能：验证后台 daemon 启动时不再继承调用终端的 stdin
# 设计：替换网络探测和 Popen，只检查最小启动参数包含 DEVNULL 输入
def test_core_start_detaches_stdin(monkeypatch) -> None:
    process = Mock(pid=12345)
    popen = Mock(return_value=process)
    monkeypatch.setattr(
        core,
        "_ping_check",
        AsyncMock(side_effect=ConnectionRefusedError),
    )
    monkeypatch.setattr(
        core,
        "_wait_for_start_slot",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(
        core,
        "_wait_for_started",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(core.subprocess, "Popen", popen)

    core.cmd_core_start(CyanConfig())

    assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL
