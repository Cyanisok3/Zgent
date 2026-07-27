from __future__ import annotations

import argparse
import logging
import logging.handlers
import os
from pathlib import Path

from cyan.core.config import get_config
from cyan.tui.app import CyanTuiApp

_DEFAULT_TUI_LOG = "~/.cyan/logs/tui.log"


# TUI 文件日志初始化：不写 stderr（避免干扰 Textual 渲染），只写滚动文件
def _setup_logging(level: str) -> None:
    log_path = Path(os.environ.get("CYAN_TUI_LOG_FILE", _DEFAULT_TUI_LOG)).expanduser()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    handler.setFormatter(
        logging.Formatter(
            'level=%(levelname)s ts=%(asctime)s source=%(name)s msg="%(message)s"',
            datefmt="%Y-%m-%dT%H:%M:%S",
        )
    )
    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    root.handlers.clear()
    root.addHandler(handler)


# cyan-tui 入口：可附着指定任务，否则自动恢复最近的可处理任务
def main() -> None:
    parser = argparse.ArgumentParser(prog="cyan-tui", description="cyan TUI")
    parser.add_argument(
        "--job",
        metavar="JOB_ID",
        help="Attach to one job instead of selecting the most recent active job",
    )
    args = parser.parse_args()

    config = get_config()
    _setup_logging(config.logging.level)
    app = CyanTuiApp(config.host, config.port, job_id=args.job)
    app.run()


if __name__ == "__main__":
    main()
