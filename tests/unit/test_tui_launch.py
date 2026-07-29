from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cyan.core.jobs.launch import (
    LaunchParseError,
    format_launch_preview,
    launch_fingerprint,
    parse_training_command,
)


# 功能：验证引号、反斜杠续行、环境覆盖和配置路径被解析为确定性 argv
# 设计：使用当前解释器的真实绝对路径，避免测试依赖机器上的 python 别名
def test_parse_training_command_builds_deterministic_preview(tmp_path: Path) -> None:
    command = (
        f'MODE="quick run" {sys.executable} train.py \\\n'
        '  --yaml "configs/cyan test.yaml" --epochs=1'
    )

    launch = parse_training_command(command, tmp_path, {"PATH": os.environ["PATH"]})
    preview = format_launch_preview(launch, tmp_path)

    assert launch.argv == (
        sys.executable,
        "train.py",
        "--yaml",
        "configs/cyan test.yaml",
        "--epochs=1",
    )
    assert launch.env_overrides == {"MODE": "quick run"}
    assert launch.executable == str(Path(sys.executable).resolve())
    assert launch.config_paths == (
        str((tmp_path / "configs/cyan test.yaml").resolve()),
    )
    assert "MODE=quick run" in preview
    assert "PATH=" not in preview


# 功能：验证等号参数只有位于命令开头且名称合法时才会被解释为环境覆盖
# 设计：在可执行文件后放入模型参数，锁定解析器不会吞掉普通 argv
def test_parse_training_command_keeps_later_equals_arguments(tmp_path: Path) -> None:
    launch = parse_training_command(
        f"{sys.executable} train.py model=small",
        tmp_path,
        {"PATH": os.environ["PATH"]},
    )

    assert launch.env_overrides == {}
    assert launch.argv[-1] == "model=small"


@pytest.mark.parametrize(
    "command",
    [
        f"{sys.executable} train.py | tee out.log",
        f"{sys.executable} train.py && echo done",
        f"{sys.executable} train.py || echo failed",
        f"{sys.executable} train.py > out.log",
        f"{sys.executable} train.py >> out.log",
        f"{sys.executable} train.py < input.txt",
        f"{sys.executable} train.py 2>&1",
        f"{sys.executable} train.py &",
        f"{sys.executable} train.py; echo done",
        f"{sys.executable} train.py $(whoami)",
        f"{sys.executable} train.py `whoami`",
    ],
)
# 功能：验证所有需要 shell 解释的运算符和命令替换都会被拒绝
# 设计：参数化覆盖管道、条件、重定向、后台、分号和两类命令替换
def test_parse_training_command_rejects_shell_syntax(
    command: str,
    tmp_path: Path,
) -> None:
    with pytest.raises(LaunchParseError):
        parse_training_command(command, tmp_path, {"PATH": os.environ["PATH"]})


# 功能：验证未使用反斜杠续行的多命令粘贴不会被误合并为单个 argv
# 设计：传入两个物理命令行并检查用户被引导改用脚本或续行
def test_parse_training_command_rejects_uncontinued_newline(tmp_path: Path) -> None:
    with pytest.raises(LaunchParseError, match="multiple commands"):
        parse_training_command(
            f"{sys.executable} train.py\n{sys.executable} eval.py",
            tmp_path,
            {"PATH": os.environ["PATH"]},
        )


# 功能：验证不可解析的可执行文件在确认预览前即被拒绝
# 设计：提供空 PATH 和稳定的虚构命令，避免意外命中宿主程序
def test_parse_training_command_rejects_missing_executable(tmp_path: Path) -> None:
    with pytest.raises(LaunchParseError, match="executable not found"):
        parse_training_command(
            "definitely-not-a-real-cyan-training-command train.py",
            tmp_path,
            {"PATH": ""},
        )


# 功能：验证相对可执行文件以训练工作区而非 daemon 当前目录为基准解析
# 设计：创建真实可执行脚本并从不同进程目录调用纯解析器
def test_parse_training_command_resolves_workspace_executable(tmp_path: Path) -> None:
    executable = tmp_path / "train"
    executable.write_text("#!/bin/sh\n", encoding="utf-8")
    executable.chmod(0o755)

    launch = parse_training_command("./train --steps 1", tmp_path, {"PATH": ""})

    assert launch.executable == str(executable.resolve())


# 功能：验证 LaunchSpec 指纹覆盖完整环境但预览仍不暴露继承环境
# 设计：只改变一个继承变量并比较稳定哈希，锁定确认与启动一致性
def test_launch_fingerprint_detects_environment_change(tmp_path: Path) -> None:
    first_env = {"PATH": os.environ["PATH"], "PRIVATE_TOKEN": "first"}
    second_env = {"PATH": os.environ["PATH"], "PRIVATE_TOKEN": "second"}
    launch = parse_training_command(f"{sys.executable} train.py", tmp_path, first_env)

    first = launch_fingerprint(launch, tmp_path, first_env)
    second = launch_fingerprint(launch, tmp_path, second_env)

    assert first != second
    assert "PRIVATE_TOKEN" not in format_launch_preview(launch, tmp_path)
