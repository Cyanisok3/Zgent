from __future__ import annotations

import argparse
import sys

from cyan.cli.commands.chat import cmd_chat
from cyan.cli.commands.core import cmd_core_start, cmd_core_status, cmd_core_stop
from cyan.cli.commands.ping import cmd_ping
from cyan.cli.commands.run import cmd_run
from cyan.cli.commands.trace import cmd_trace
from cyan.cli.commands.version import cmd_version
from cyan.cli.commands.watch import cmd_job_tui, cmd_watch
from cyan.core.config import get_config
from cyan.core.logging_setup import setup_logging


# CLI 主入口：解析命令行参数并分发到对应子命令
def main() -> None:
    parser = argparse.ArgumentParser(prog="cyan", description="cyan CLI")
    parser.add_argument("--version", action="store_true", help="Print version and exit")
    subparsers = parser.add_subparsers(dest="command", metavar="{watch}")

    subparsers.add_parser("ping")
    subparsers.add_parser("chat")

    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--goal", required=True, help="Goal for the agent to accomplish")

    watch_parser = subparsers.add_parser("watch", help="Watch a training command")
    watch_parser.add_argument("argv", nargs=argparse.REMAINDER, help="Command after --")

    core_parser = subparsers.add_parser("core")
    core_sub = core_parser.add_subparsers(dest="core_command")
    core_sub.add_parser("start", help="Start the daemon in the background")
    core_sub.add_parser("stop", help="Stop the running daemon")
    core_sub.add_parser("status", help="Show daemon status")

    trace_parser = subparsers.add_parser("trace")
    trace_parser.add_argument("run_id", nargs="?", default=None, help="Filter by run ID")
    trace_parser.add_argument("--layer", choices=["ipc", "event", "llm"], help="Filter by layer")
    trace_parser.add_argument("--direction", help="Filter by direction (e.g. CORE→LLM)")
    trace_parser.add_argument("--raw", action="store_true", help="Output raw NDJSON")
    trace_parser.add_argument("--follow", "-f", action="store_true", help="Follow new records")

    args = parser.parse_args()

    if args.version:
        cmd_version()
        return

    config = get_config()
    setup_logging(config)

    if args.command == "ping":
        cmd_ping(config)
    elif args.command == "chat":
        cmd_chat(config)
    elif args.command == "run":
        cmd_run(args.goal, config)
    elif args.command == "watch":
        argv = list(args.argv)
        if argv[:1] == ["--"]:
            argv = argv[1:]
        if not argv:
            watch_parser.error("expected a command after --")
        cmd_watch(argv, config)
    elif args.command == "core":
        if args.core_command == "start":
            cmd_core_start(config)
        elif args.core_command == "stop":
            cmd_core_stop(config)
        elif args.core_command == "status":
            cmd_core_status(config)
        else:
            core_parser.print_help()
            sys.exit(1)
    elif args.command == "trace":
        cmd_trace(
            args.run_id,
            config,
            layer=args.layer,
            direction=args.direction,
            raw=args.raw,
            follow=args.follow,
        )
    else:
        cmd_job_tui(config)
