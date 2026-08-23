from __future__ import annotations

from pathlib import Path

import pytest

from cyan.config import get_config


# 写入测试专用的环境变量文件
def _write_env(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


# 功能：验证 .env 文件中的值被正确加载并覆盖内建默认值
# 设计：写 .env 到临时目录并 chdir 进去，清除同名系统环境变量排除干扰，确认 .env 加载路径有效
def test_dotenv_base_loaded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "CYAN_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CYAN_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 9999


# 功能：验证系统环境变量的优先级高于 .env 文件中的值
# 设计：.env 写 9999，系统环境变量写 8888，确认最终值为 8888，对应四级优先链的顶层约束
def test_system_env_overrides_dotenv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env_file = tmp_path / ".env"
    _write_env(env_file, "CYAN_PORT=9999\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CYAN_PORT", "8888")

    cfg = get_config()

    assert cfg.port == 8888


# 功能：验证 .env 文件不存在时静默跳过，使用内建默认值（不抛异常）
# 设计：chdir 到空目录，清除系统环境变量，确认 get_config() 不因 .env 缺失而崩溃，默认端口为 7437
def test_missing_env_file_silent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CYAN_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 7437


# 功能：验证 .env 中设置的 CYAN_CONFIG 能正确影响 TOML 配置文件的加载路径
# 设计：.env 指向自定义 TOML 文件，TOML 中写入不同端口，确认 .env 在 TOML 加载前被读取（优先级链的正确顺序）
def test_dotenv_before_toml_cyan_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    toml_path = tmp_path / "custom.toml"
    toml_path.write_bytes(b'[core]\nport = 5555\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, f"CYAN_CONFIG={toml_path}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CYAN_CONFIG", raising=False)
    monkeypatch.delenv("CYAN_PORT", raising=False)

    cfg = get_config()

    assert cfg.port == 5555


# 功能：验证同一变量经过完整四级优先链后，最终值为最高优先级来源（系统环境变量）
# 设计：同时设置默认值(7437)/TOML(6000)/.env(7000)/系统环境变量(8000)，确认最终值为 8000，是优先级链的综合正确性验证
def test_priority_chain_full(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # 默认值：7437
    # TOML：6000
    # .env：7000
    # 系统环境变量：8000（最高）
    toml_path = tmp_path / "cyan.toml"
    toml_path.write_bytes(b'[core]\nport = 6000\n')

    env_file = tmp_path / ".env"
    _write_env(env_file, "CYAN_PORT=7000\n")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CYAN_CONFIG", str(toml_path))
    monkeypatch.setenv("CYAN_PORT", "8000")

    cfg = get_config()

    assert cfg.port == 8000


# 功能：验证项目配置中的 incident.smoke 表不会被通用配置加载器误判为未知字段
# 设计：显式指定只含 smoke 声明的 TOML，隔离用户全局配置并断言仍返回默认核心配置
def test_project_incident_smoke_config_is_accepted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text(
        '[incident.smoke]\nargv = ["python", "smoke.py"]\ntimeout_s = 45\n',
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CYAN_CONFIG", str(config_path))
    monkeypatch.delenv("CYAN_HOST", raising=False)
    monkeypatch.delenv("CYAN_PORT", raising=False)

    cfg = get_config()

    assert cfg.host == "127.0.0.1"
    assert cfg.port == 7437


# 功能：验证 daemon 配置拒绝监听非 loopback 地址
# 设计：通过最高优先级 CYAN_HOST 注入 0.0.0.0，断言加载阶段即失败而非启动后暴露端口
def test_non_loopback_host_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CYAN_CONFIG", str(tmp_path / "missing.toml"))
    monkeypatch.setenv("CYAN_HOST", "0.0.0.0")
    monkeypatch.delenv("CYAN_PORT", raising=False)

    with pytest.raises(SystemExit, match="loopback"):
        get_config()


# 功能：验证 incident 配置中的拼写错误会在启动阶段 fail-closed
# 设计：写入 somke 这一真实易错键，断言通用配置加载器明确拒绝而非静默禁用 verifier
def test_unknown_incident_key_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text("[incident.somke]\nargv = ['true']\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CYAN_CONFIG", str(config_path))

    with pytest.raises(SystemExit, match=r"\[incident\] accepts only smoke"):
        get_config()
