from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from cyan.cli import main as cli_main
from cyan.cli.commands import watch


class _FakeClient:
    # 初始化假的 IPC 客户端并记录命令
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        self.commands: list[tuple[str, dict[str, object]]] = []
        self.closed = False

    # 模拟连接成功
    async def connect(self) -> None:
        return None

    # 保持事件循环存活，直到测试中的调用方取消任务
    async def run_event_loop(self) -> None:
        import asyncio

        await asyncio.Event().wait()

    # 记录命令并返回固定 job id
    async def send_command(self, method: str, params: dict[str, object]) -> dict[str, object]:
        self.commands.append((method, params))
        if method == "core.ping":
            return {"protocol_version": 2}
        return {"job_id": "job-1"}

    # 记录连接关闭
    async def close(self) -> None:
        self.closed = True


# 功能：验证 watch 使用当前项目根目录和完整 argv 创建真实任务
# 设计：替换 SocketClient 后直接调用异步边界，精确检查 RPC 参数且不启动 daemon 或 TUI
async def test_start_job_sends_workspace_and_argv(monkeypatch, tmp_path: Path) -> None:
    clients: list[_FakeClient] = []

    # 为每次构造保存客户端引用，便于检查命令
    def make_client(host: str, port: int) -> _FakeClient:
        client = _FakeClient(host, port)
        clients.append(client)
        return client

    monkeypatch.setattr(watch, "SocketClient", make_client)
    config = SimpleNamespace(host="127.0.0.1", port=7437)

    job_id = await watch._start_job(
        config,
        ["python", "train.py", "--epochs", "2"],
        tmp_path,
        {"CUDA_VISIBLE_DEVICES": "0"},
    )

    assert job_id == "job-1"
    assert clients[0].commands == [
        ("core.ping", {"client": "cyan-watch"}),
        (
            "job.start",
            {
                "argv": ["python", "train.py", "--epochs", "2"],
                "workspace_root": str(tmp_path.resolve()),
                "env": {"CUDA_VISIBLE_DEVICES": "0"},
            },
        )
    ]
    assert clients[0].closed


# 功能：验证 watch 在提交 job.start 前明确拒绝旧 wire protocol
# 设计：让假 daemon 返回 v1，检查连接关闭且命令序列中没有任何状态变更请求
async def test_start_job_rejects_protocol_mismatch(monkeypatch, tmp_path: Path) -> None:
    client = _FakeClient("127.0.0.1", 7437)

    # 仅覆盖 ping 返回值，保留命令记录和关闭行为
    async def send_legacy(method: str, params: dict[str, object]) -> dict[str, object]:
        client.commands.append((method, params))
        return {"protocol_version": 1}

    client.send_command = send_legacy  # type: ignore[method-assign]
    monkeypatch.setattr(watch, "SocketClient", lambda host, port: client)
    config = SimpleNamespace(host="127.0.0.1", port=7437)

    with pytest.raises(RuntimeError, match="wire protocol mismatch"):
        await watch._start_job(config, ["python", "train.py"], tmp_path, {})

    assert client.commands == [("core.ping", {"client": "cyan-watch"})]
    assert client.closed


# 功能：验证 `cyan watch -- ...` 保留分隔符后的原始命令参数
# 设计：替换配置、日志和命令处理器后驱动 argparse 主入口，避免启动真实进程
def test_main_dispatches_watch_argv(monkeypatch) -> None:
    calls: list[list[str]] = []
    config = SimpleNamespace()
    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(cli_main, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli_main, "cmd_watch", lambda argv, _config: calls.append(argv))
    monkeypatch.setattr(
        "sys.argv",
        ["cyan", "watch", "--", "python", "train.py", "--epochs", "2"],
    )

    cli_main.main()

    assert calls == [["python", "train.py", "--epochs", "2"]]


# 功能：验证无子命令时直接进入自动附着的 Job TUI
# 设计：替换产品入口并检查调用次数，锁定 `cyan` 的零参数主流程而不挂载 Textual
def test_main_without_command_opens_job_tui(monkeypatch) -> None:
    calls: list[object] = []
    config = SimpleNamespace()
    monkeypatch.setattr(cli_main, "get_config", lambda: config)
    monkeypatch.setattr(cli_main, "setup_logging", lambda _config: None)
    monkeypatch.setattr(cli_main, "cmd_job_tui", lambda _config: calls.append(_config))
    monkeypatch.setattr("sys.argv", ["cyan"])

    cli_main.main()

    assert calls == [config]
