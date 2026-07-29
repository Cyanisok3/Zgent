from __future__ import annotations

import json
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
        "_discover_running_daemon",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        core,
        "_wait_for_started",
        AsyncMock(return_value=True),
    )
    monkeypatch.setattr(core.subprocess, "Popen", popen)

    core.cmd_core_start(CyanConfig())

    assert popen.call_args.kwargs["stdin"] == subprocess.DEVNULL


# 功能：验证 core start 的 JSON 模式返回插件所需的连接信息
# 设计：复用启动 mock 并只解析单行 stdout，避免依赖真实后台 daemon
def test_core_start_json_output(monkeypatch, capsys) -> None:
    process = Mock(pid=12345)
    monkeypatch.setattr(
        core,
        "_ping_check",
        AsyncMock(side_effect=ConnectionRefusedError),
    )
    monkeypatch.setattr(
        core,
        "_discover_running_daemon",
        AsyncMock(return_value=None),
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
    monkeypatch.setattr(core.subprocess, "Popen", Mock(return_value=process))

    core.cmd_core_start(CyanConfig(), json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "started"
    assert payload["host"] == "127.0.0.1"
    assert payload["port"] == 7437
    assert payload["pid"] == 12345


# 功能：验证 JSON 启动命令能发现由另一工作区配置启动的活动 daemon
# 设计：让当前端口探测失败并返回锁文件元数据，确认不会尝试第二次 Popen
def test_core_start_json_reports_discovered_workspace(monkeypatch, capsys) -> None:
    popen = Mock()
    monkeypatch.setattr(
        core,
        "_ping_check",
        AsyncMock(side_effect=ConnectionRefusedError),
    )
    monkeypatch.setattr(
        core,
        "_discover_running_daemon",
        AsyncMock(
            return_value={
                "host": "127.0.0.1",
                "port": 8123,
                "workspace_root": "/tmp/other-project",
                "pid": 456,
                "protocol_version": 1,
            }
        ),
    )
    monkeypatch.setattr(core.subprocess, "Popen", popen)

    core.cmd_core_start(CyanConfig(), json_output=True)

    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "already_running"
    assert payload["port"] == 8123
    assert payload["workspace_root"] == "/tmp/other-project"
    assert payload["protocol_version"] == 1
    popen.assert_not_called()
