from __future__ import annotations

import argparse
import asyncio
import time

from cyan.cli.commands.core import _ping_check, cmd_core_start, cmd_core_stop
from cyan.cli.commands.version import cmd_version
from cyan.config import CyanConfig, get_config
from cyan.service.logging_setup import setup_logging
from cyan.tui.app import CyanTuiApp

_CORE_START_TIMEOUT_SECONDS = 15.0


# 等待 daemon 接受连接，超时则报告启动失败
async def _wait_for_core(config: CyanConfig) -> None:
    deadline = time.monotonic() + _CORE_START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            await _ping_check(config)
            return
        except (ConnectionRefusedError, OSError):
            await asyncio.sleep(0.05)
    raise RuntimeError(f"core did not start at {config.host}:{config.port}")


# 确保 daemon 运行后进入唯一 TUI 产品入口
def _run_tui(config: CyanConfig) -> None:
    try:
        asyncio.run(_ping_check(config))
    except (ConnectionRefusedError, OSError):
        cmd_core_start(config)
    asyncio.run(_wait_for_core(config))
    CyanTuiApp(config.host, config.port).run()


# 解析 cyan 的唯一用户入口和两个 daemon 生命周期命令
def main() -> None:
    parser = argparse.ArgumentParser(
        prog="cyan",
        description="Local ML training Incident Agent",
        epilog="Run cyan in an ML project, then use /monitor to supervise training.",
    )
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="{core}")
    core_parser = subparsers.add_parser("core", help="Manage the local daemon")
    core_sub = core_parser.add_subparsers(dest="core_command")
    start = core_sub.add_parser("start", help="Start the daemon in the background")
    start.add_argument("--json", action="store_true", help="Print machine-readable connection data")
    core_sub.add_parser("stop", help="Stop the running daemon")
    args = parser.parse_args()
    if args.version:
        cmd_version()
        return
    config = get_config()
    setup_logging(config)
    if args.command == "core":
        if args.core_command == "start":
            cmd_core_start(config, json_output=args.json)
            return
        if args.core_command == "stop":
            cmd_core_stop(config)
            return
        core_parser.print_help()
        raise SystemExit(1)
    _run_tui(config)


if __name__ == "__main__":
    main()
