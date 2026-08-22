from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from cyan.errors import HandlerError
from cyan.service.app import CoreApp
from cyan.service.protocol.commands import JobStartResult


# 功能：验证 launch.preview 只返回结构化预览和指纹，不回显继承环境
# 设计：直接调用真实 handler 与共享解析器，避免另造协议假实现
async def test_launch_preview_hides_inherited_environment(tmp_path: Path) -> None:
    app = CoreApp()
    result = await app._launch_preview_handler(
        {
            "command": f"MODE=test {sys.executable} train.py",
            "workspace_root": str(tmp_path),
            "env": {"PATH": os.environ["PATH"], "PRIVATE_TOKEN": "secret"},
        }
    )

    data = result.model_dump()
    assert data["argv"] == [sys.executable, "train.py"]
    assert data["env_overrides"] == {"MODE": "test"}
    assert "PRIVATE_TOKEN" not in str(data)


# 功能：验证 launch.start 仅在预览指纹一致时复用唯一 Job 启动边界
# 设计：只替换最终进程启动函数，真实执行两次解析和环境合并
async def test_launch_start_reuses_job_start_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = CoreApp()
    environment = {"PATH": os.environ["PATH"], "BASE": "value"}
    preview = await app._launch_preview_handler(
        {
            "command": f"MODE=test {sys.executable} train.py",
            "workspace_root": str(tmp_path),
            "env": environment,
        }
    )
    start = AsyncMock(return_value=JobStartResult(job_id="job-1"))
    monkeypatch.setattr(app, "_start_job", start)

    result = await app._launch_start_handler(
        {
            "command": f"MODE=test {sys.executable} train.py",
            "workspace_root": str(tmp_path),
            "env": environment,
            "preview_fingerprint": preview.fingerprint,
        }
    )

    assert result.job_id == "job-1"
    command = start.call_args.args[0]
    assert command.env["BASE"] == "value"
    assert command.env["MODE"] == "test"


# 功能：验证环境变化会令已确认的 launch preview 失效
# 设计：预览后只改变一个环境变量并检查 handler 在启动前拒绝
async def test_launch_start_rejects_stale_preview(tmp_path: Path) -> None:
    app = CoreApp()
    preview = await app._launch_preview_handler(
        {
            "command": f"{sys.executable} train.py",
            "workspace_root": str(tmp_path),
            "env": {"PATH": os.environ["PATH"], "VALUE": "first"},
        }
    )

    with pytest.raises(HandlerError, match="preview is stale"):
        await app._launch_start_handler(
            {
                "command": f"{sys.executable} train.py",
                "workspace_root": str(tmp_path),
                "env": {"PATH": os.environ["PATH"], "VALUE": "second"},
                "preview_fingerprint": preview.fingerprint,
            }
        )
