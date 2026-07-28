from __future__ import annotations

import re
import shlex
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_CONFIG_FLAGS = frozenset(
    {
        "--cfg",
        "--config",
        "--config-file",
        "--config_path",
        "--yaml",
    }
)
_SHELL_OPERATOR_CHARS = frozenset("|&><;")


class LaunchParseError(ValueError):
    pass


@dataclass(frozen=True)
class ParsedLaunch:
    argv: tuple[str, ...]
    env_overrides: dict[str, str]
    executable: str
    config_paths: tuple[str, ...]


# 将反斜杠续行折叠为空格，并拒绝多个未续行命令
def _normalize_lines(command: str) -> str:
    normalized = command.replace("\\\r\n", " ").replace("\\\n", " ")
    if "\n" in normalized or "\r" in normalized:
        raise LaunchParseError(
            "multiple commands are not supported; use backslash line continuations "
            "or a script"
        )
    return normalized


# 使用 shlex 仅做词法拆分，并拒绝所有需要 shell 解释的语法
def _split_command(command: str) -> list[str]:
    normalized = _normalize_lines(command).strip()
    if not normalized:
        raise LaunchParseError("training command is empty")
    if "$(" in normalized or "`" in normalized:
        raise LaunchParseError("command substitution is not supported; use a script")
    lexer = shlex.shlex(normalized, posix=True, punctuation_chars="|&><;")
    lexer.whitespace_split = True
    lexer.commenters = ""
    try:
        tokens = list(lexer)
    except ValueError as exc:
        raise LaunchParseError(str(exc)) from exc
    operator = next(
        (
            token
            for token in tokens
            if token and all(character in _SHELL_OPERATOR_CHARS for character in token)
        ),
        None,
    )
    if operator is not None:
        raise LaunchParseError(
            f"shell operator {operator!r} is not supported; use a script"
        )
    return tokens


# 从命令开头提取环境变量覆盖，后续等号参数仍作为普通 argv
def _extract_env(tokens: list[str]) -> tuple[dict[str, str], list[str]]:
    overrides: dict[str, str] = {}
    index = 0
    for token in tokens:
        if "=" not in token:
            break
        name, value = token.split("=", 1)
        if _ENV_NAME.fullmatch(name) is None:
            break
        overrides[name] = value
        index += 1
    argv = tokens[index:]
    if not argv:
        raise LaunchParseError("training command must include an executable")
    return overrides, argv


# 从常见显式配置参数中提取路径，仅用于确定性预览
def _config_paths(argv: list[str], workspace_root: Path) -> tuple[str, ...]:
    paths: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        value: str | None = None
        if token in _CONFIG_FLAGS and index + 1 < len(argv):
            value = argv[index + 1]
            index += 1
        else:
            for flag in _CONFIG_FLAGS:
                prefix = f"{flag}="
                if token.startswith(prefix):
                    value = token[len(prefix) :]
                    break
        if value:
            path = Path(value).expanduser()
            if not path.is_absolute():
                path = workspace_root / path
            paths.append(str(path.resolve(strict=False)))
        index += 1
    return tuple(paths)


# 解析训练命令并在合并后的 PATH 中解析真实可执行文件
def parse_training_command(
    command: str,
    workspace_root: Path,
    base_env: Mapping[str, str],
) -> ParsedLaunch:
    tokens = _split_command(command)
    overrides, argv = _extract_env(tokens)
    environment = dict(base_env)
    environment.update(overrides)
    executable = shutil.which(argv[0], path=environment.get("PATH"))
    if executable is None:
        raise LaunchParseError(f"executable not found: {argv[0]}")
    return ParsedLaunch(
        argv=tuple(argv),
        env_overrides=overrides,
        executable=str(Path(executable).resolve(strict=False)),
        config_paths=_config_paths(argv, workspace_root),
    )


# 生成不包含继承环境的启动预览文本
def format_launch_preview(launch: ParsedLaunch, workspace_root: Path) -> str:
    lines = [
        "Training launch preview",
        f"cwd: {workspace_root}",
        f"executable: {launch.executable}",
        f"argv: {shlex.join(launch.argv)}",
    ]
    if launch.env_overrides:
        lines.append("environment overrides:")
        lines.extend(f"  {name}={value}" for name, value in launch.env_overrides.items())
    else:
        lines.append("environment overrides: (none)")
    if launch.config_paths:
        lines.append("config paths:")
        lines.extend(f"  {path}" for path in launch.config_paths)
    else:
        lines.append("config paths: (none detected)")
    lines.append("Type /start to launch, or /monitor to replace this command.")
    return "\n".join(lines)
