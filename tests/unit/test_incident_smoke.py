from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from cyan.core.incidents.smoke import (
    SmokeVerifierConfig,
    SubprocessSmokeExecutor,
    load_smoke_verifier,
    smoke_verifier_fingerprint,
)


# 功能：验证项目未声明 smoke verifier 时返回 None
# 设计：使用空 workspace 体现零配置路径，避免把缺失配置误判为产品错误
def test_load_smoke_verifier_is_optional(tmp_path: Path) -> None:
    assert load_smoke_verifier(tmp_path) is None


# 功能：验证只加载 incident.smoke 下的 argv 和 timeout
# 设计：写入真实 TOML 并断言 frozen typed model，覆盖用户配置到执行契约的边界
def test_load_smoke_verifier_from_project_config(tmp_path: Path) -> None:
    config_dir = tmp_path / ".cyan"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[incident.smoke]\nargv = ["python", "train.py", "--smoke"]\ntimeout_s = 45\n',
        encoding="utf-8",
    )

    config = load_smoke_verifier(tmp_path)

    assert config == SmokeVerifierConfig(
        argv=["python", "train.py", "--smoke"],
        timeout_s=45,
    )
    assert config is not None
    with pytest.raises(ValidationError):
        config.timeout_s = 1


# 功能：验证 smoke 配置拒绝未知字段
# 设计：在 TOML 边界加入 Agent 可滥用的额外命令字段，确认最小契约采用 fail-closed
def test_load_smoke_verifier_rejects_unknown_keys(tmp_path: Path) -> None:
    config_dir = tmp_path / ".cyan"
    config_dir.mkdir()
    (config_dir / "config.toml").write_text(
        '[incident.smoke]\nargv = ["python", "train.py"]\nshell = true\n',
        encoding="utf-8",
    )

    with pytest.raises(ValidationError):
        load_smoke_verifier(tmp_path)


# 功能：验证 smoke 配置指纹只由规范化 argv 与 timeout 决定
# 设计：比较等价模型和变更命令，锁定审批 RPC 可稳定绑定的 SHA-256 契约
def test_smoke_verifier_fingerprint_is_stable_and_content_bound() -> None:
    first = SmokeVerifierConfig(argv=["python", "smoke.py"], timeout_s=45)
    equivalent = SmokeVerifierConfig(argv=["python", "smoke.py"], timeout_s=45.0)
    changed = SmokeVerifierConfig(argv=["python", "other.py"], timeout_s=45)

    fingerprint = smoke_verifier_fingerprint(first)

    assert len(fingerprint) == 64
    assert smoke_verifier_fingerprint(equivalent) == fingerprint
    assert smoke_verifier_fingerprint(changed) != fingerprint


# 功能：验证真实 smoke 执行器按退出码判定成功并无损保存两个输出流
# 设计：启动当前 Python 写入不同二进制内容，直接比较文件字节以排除文本解码和 EventBus 截断
async def test_subprocess_smoke_executor_preserves_output(tmp_path: Path) -> None:
    stdout_path = tmp_path / "smoke.stdout.log"
    stderr_path = tmp_path / "smoke.stderr.log"
    config = SmokeVerifierConfig(
        argv=[
            sys.executable,
            "-c",
            "import os; assert os.read(0, 1) == b''; "
            "os.write(1, b'out\\x00bytes'); os.write(2, b'err\\xffbytes')",
        ],
        timeout_s=5,
    )

    result = await SubprocessSmokeExecutor().run(
        config,
        cwd=tmp_path,
        env=os.environ,
        stdout_path=stdout_path,
        stderr_path=stderr_path,
    )

    assert result.status == "passed"
    assert result.returncode == 0
    assert stdout_path.read_bytes() == b"out\x00bytes"
    assert stderr_path.read_bytes() == b"err\xffbytes"


# 功能：验证 smoke 启动后立即向 harness 报告可持久化的 PID 启动身份
# 设计：通过真实短进程收集 on_started 参数，断言身份非空且在执行器成功返回前完成回调
async def test_subprocess_smoke_reports_process_identity(tmp_path: Path) -> None:
    observed: list[tuple[int, str]] = []

    # 收集 harness 将原子持久化的 smoke 进程身份
    async def on_started(pid: int, process_identity: str) -> None:
        observed.append((pid, process_identity))

    result = await SubprocessSmokeExecutor().run(
        SmokeVerifierConfig(
            argv=[sys.executable, "-c", "import time; time.sleep(0.2)"],
            timeout_s=5,
        ),
        cwd=tmp_path,
        env=os.environ,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
        on_started=on_started,
    )

    assert result.status == "passed"
    assert len(observed) == 1
    assert observed[0][0] > 0
    assert observed[0][1]


# 功能：验证合法的极短 smoke 在身份查询前退出时仍按真实退出码判定成功
# 设计：运行系统 true 命令覆盖 ps 已查不到 PID 的确定性竞态，确保不会误报 launch failure
async def test_subprocess_smoke_accepts_process_that_already_exited(
    tmp_path: Path,
) -> None:
    true_command = shutil.which("true")
    assert true_command is not None

    result = await SubprocessSmokeExecutor().run(
        SmokeVerifierConfig(argv=[true_command], timeout_s=5),
        cwd=tmp_path,
        env=os.environ,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.status == "passed"
    assert result.returncode == 0


# 功能：验证真实 smoke 超时会终止独立进程组并返回 timed_out
# 设计：运行长睡眠脚本并使用极短 timeout，断言快速返回且不把信号退出码误报为普通失败
async def test_subprocess_smoke_executor_times_out(tmp_path: Path) -> None:
    config = SmokeVerifierConfig(
        argv=[sys.executable, "-c", "import time; time.sleep(30)"],
        timeout_s=0.05,
    )

    result = await SubprocessSmokeExecutor(terminate_grace_s=0.1).run(
        config,
        cwd=tmp_path,
        env=os.environ,
        stdout_path=tmp_path / "stdout.log",
        stderr_path=tmp_path / "stderr.log",
    )

    assert result.status == "timed_out"
    assert result.returncode is None
    assert result.elapsed_ms < 2000
