from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import signal
import time
import tomllib
from collections.abc import Awaitable, Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator

from cyan.training.processes import read_process_identity

SmokeStartedCallback = Callable[[int, str], Awaitable[None] | None]


class SmokeVerifierConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    argv: list[str] = Field(min_length=1)
    timeout_s: float = Field(default=300.0, gt=0, le=3600)

    # 拒绝空参数，确保执行器能直接传给 create_subprocess_exec
    @field_validator("argv")
    @classmethod
    def _argv_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("smoke argv entries must not be empty")
        return value


class SmokeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["passed", "failed", "timed_out", "interrupted"]
    returncode: int | None
    elapsed_ms: int = Field(ge=0)
    stdout_path: Path
    stderr_path: Path


class SmokeExecution(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    status: Literal["running", "passed", "failed", "timed_out", "interrupted"]
    pid: int = Field(gt=0)
    process_identity: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None


class SmokeExecutor(Protocol):
    # 在真实 workspace 和环境中执行已冻结的 smoke verifier
    async def run(
        self,
        config: SmokeVerifierConfig,
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
        on_started: SmokeStartedCallback | None = None,
    ) -> SmokeResult: ...


# 生成只绑定可执行 argv 与 timeout 的稳定配置指纹
def smoke_verifier_fingerprint(config: SmokeVerifierConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class SubprocessSmokeExecutor:
    # 初始化真实 smoke 执行器并设置进程组优雅终止时限
    def __init__(self, terminate_grace_s: float = 2.0) -> None:
        if terminate_grace_s <= 0:
            raise ValueError("terminate_grace_s must be positive")
        self._terminate_grace_s = terminate_grace_s

    # 终止 smoke 的独立进程组，超时后升级为 SIGKILL
    async def _terminate_group(self, process: asyncio.subprocess.Process) -> None:
        if process.returncode is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            await asyncio.wait_for(process.wait(), timeout=self._terminate_grace_s)
        except TimeoutError:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            await process.wait()

    # 直接执行 argv，将 stdout/stderr 无损写入文件并按退出码返回结果
    async def run(
        self,
        config: SmokeVerifierConfig,
        *,
        cwd: Path,
        env: Mapping[str, str],
        stdout_path: Path,
        stderr_path: Path,
        on_started: SmokeStartedCallback | None = None,
    ) -> SmokeResult:
        resolved_cwd = cwd.resolve(strict=True)
        if stdout_path.resolve() == stderr_path.resolve():
            raise ValueError("stdout_path and stderr_path must differ")
        stdout_path.parent.mkdir(parents=True, exist_ok=True)
        stderr_path.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        timed_out = False
        with stdout_path.open("wb") as stdout_file, stderr_path.open("wb") as stderr_file:
            process = await asyncio.create_subprocess_exec(
                *config.argv,
                cwd=resolved_cwd,
                env=dict(env),
                stdin=asyncio.subprocess.DEVNULL,
                stdout=stdout_file,
                stderr=stderr_file,
                start_new_session=True,
            )
            process_identity = await read_process_identity(process.pid)
            if process_identity is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=0.05)
                except TimeoutError:
                    await self._terminate_group(process)
                    raise OSError("could not persist smoke process identity") from None
            else:
                if on_started is not None:
                    try:
                        callback_result = on_started(process.pid, process_identity)
                        if inspect.isawaitable(callback_result):
                            await callback_result
                    except BaseException:
                        await self._terminate_group(process)
                        raise
                try:
                    await asyncio.wait_for(process.wait(), timeout=config.timeout_s)
                except TimeoutError:
                    timed_out = True
                    await self._terminate_group(process)
                except asyncio.CancelledError:
                    await self._terminate_group(process)
                    raise

        elapsed_ms = int((time.monotonic() - started) * 1000)
        if timed_out:
            return SmokeResult(
                status="timed_out",
                returncode=None,
                elapsed_ms=elapsed_ms,
                stdout_path=stdout_path,
                stderr_path=stderr_path,
            )
        return SmokeResult(
            status="passed" if process.returncode == 0 else "failed",
            returncode=process.returncode,
            elapsed_ms=elapsed_ms,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
        )


# 从 .cyan/config.toml 读取可选 incident.smoke 配置
def load_smoke_verifier(workspace_root: Path) -> SmokeVerifierConfig | None:
    config_path = workspace_root / ".cyan" / "config.toml"
    if not config_path.exists():
        return None
    if config_path.is_symlink():
        raise ValueError("smoke config may not be a symlink")
    raw = tomllib.loads(config_path.read_text(encoding="utf-8"))
    incident = raw.get("incident")
    if incident is None:
        return None
    if not isinstance(incident, dict):
        raise ValueError("incident config must be a table")
    smoke = incident.get("smoke")
    if smoke is None:
        return None
    if not isinstance(smoke, dict):
        raise ValueError("incident.smoke config must be a table")
    return SmokeVerifierConfig.model_validate(smoke)
