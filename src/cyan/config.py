from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 7437
_DEFAULT_LOG_LEVEL = "INFO"
_DEFAULT_LOG_FILE = "~/.cyan/logs/core.log"
_DEFAULT_LOG_FORMAT = "text"
_DEFAULT_CONFIG_PATH = "~/.cyan/config.toml"
_DEFAULT_MODEL = "claude-sonnet-4-6"
_DEFAULT_TRACE_FILE = "~/.cyan/traces/daemon.jsonl"


@dataclass
class LoggingConfig:
    level: str = _DEFAULT_LOG_LEVEL
    file: str = _DEFAULT_LOG_FILE
    format: str = _DEFAULT_LOG_FORMAT


@dataclass
class LlmConfig:
    default_model: str = _DEFAULT_MODEL


@dataclass
class TraceConfig:
    enabled: bool = True
    file: str = _DEFAULT_TRACE_FILE


@dataclass
class CyanConfig:
    host: str = _DEFAULT_HOST
    port: int = _DEFAULT_PORT
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    trace: TraceConfig = field(default_factory=TraceConfig)


# 构建运行时配置，只保留 daemon、LLM 和脱敏 trace 所需的字段
def get_config() -> CyanConfig:
    config = CyanConfig()
    load_dotenv(".env", override=False)
    explicit = os.environ.get("CYAN_CONFIG")
    paths = (
        [Path(explicit).expanduser()]
        if explicit
        else [Path(_DEFAULT_CONFIG_PATH).expanduser(), Path(".cyan/config.toml")]
    )
    for path in paths:
        if path.exists():
            try:
                data = tomllib.loads(path.read_text(encoding="utf-8"))
            except (OSError, tomllib.TOMLDecodeError) as exc:
                raise SystemExit(f"Config parse error ({path}): {exc}") from exc
            _apply_toml(config, data)
    _apply_env(config)
    if config.host not in {"127.0.0.1", "::1"}:
        raise SystemExit("Config error: core.host must be a loopback IP address")
    if not 1 <= config.port <= 65535:
        raise SystemExit("Config error: core.port must be between 1 and 65535")
    return config


# 将允许的 TOML 小节写入配置，旧扩展配置直接拒绝以免产生隐式行为
def _apply_toml(config: CyanConfig, data: dict[str, Any]) -> None:
    known = {"core", "logging", "llm", "trace", "incident"}
    unknown = set(data) - known
    if unknown:
        raise SystemExit(f"Unknown top-level config keys: {', '.join(sorted(unknown))}")

    core = data.get("core", {})
    if not isinstance(core, dict) or set(core) - {"host", "port"}:
        raise SystemExit("Config error: [core] accepts only host and port")
    if "host" in core:
        if not isinstance(core["host"], str):
            raise SystemExit("Config error: core.host must be a string")
        config.host = core["host"]
    if "port" in core:
        if not isinstance(core["port"], int):
            raise SystemExit("Config error: core.port must be an integer")
        config.port = core["port"]

    logging_data = data.get("logging", {})
    if not isinstance(logging_data, dict) or set(logging_data) - {"level", "file", "format"}:
        raise SystemExit("Config error: [logging] accepts level, file and format")
    for key in ("level", "file", "format"):
        if key in logging_data:
            value = logging_data[key]
            if not isinstance(value, str):
                raise SystemExit(f"Config error: logging.{key} must be a string")
            setattr(config.logging, key, value)

    llm = data.get("llm", {})
    if not isinstance(llm, dict) or set(llm) - {"default_model"}:
        raise SystemExit("Config error: [llm] accepts only default_model")
    if "default_model" in llm:
        if not isinstance(llm["default_model"], str) or not llm["default_model"]:
            raise SystemExit("Config error: llm.default_model must be a non-empty string")
        config.llm.default_model = llm["default_model"]

    trace = data.get("trace", {})
    if not isinstance(trace, dict) or set(trace) - {"enabled", "file"}:
        raise SystemExit("Config error: [trace] accepts enabled and file")
    for key in ("enabled",):
        if key in trace:
            if not isinstance(trace[key], bool):
                raise SystemExit(f"Config error: trace.{key} must be a boolean")
            setattr(config.trace, key, trace[key])
    if "file" in trace:
        if not isinstance(trace["file"], str):
            raise SystemExit("Config error: trace.file must be a string")
        config.trace.file = trace["file"]

    incident = data.get("incident", {})
    if not isinstance(incident, dict) or set(incident) - {"smoke"}:
        raise SystemExit("Config error: [incident] accepts only smoke")


# 用当前支持的 CYAN_* 变量覆盖配置，删除旧 Agent 扩展变量
def _apply_env(config: CyanConfig) -> None:
    if (value := os.environ.get("CYAN_HOST")) is not None:
        config.host = value
    if (value := os.environ.get("CYAN_PORT")) is not None:
        try:
            config.port = int(value)
        except ValueError as exc:
            raise SystemExit(f"Config error: CYAN_PORT must be an integer, got: {value!r}") from exc
    for env_name, attr in (
        ("CYAN_LOG_LEVEL", "level"),
        ("CYAN_LOG_FILE", "file"),
        ("CYAN_LOG_FORMAT", "format"),
    ):
        if (value := os.environ.get(env_name)) is not None:
            setattr(config.logging, attr, value)
    if (value := os.environ.get("CYAN_LLM_DEFAULT_MODEL")) is not None:
        config.llm.default_model = value
    if (value := os.environ.get("CYAN_TRACE_ENABLED")) is not None:
        config.trace.enabled = value.lower() not in {"0", "false", "no"}
    if (value := os.environ.get("CYAN_TRACE_FILE")) is not None:
        config.trace.file = value
