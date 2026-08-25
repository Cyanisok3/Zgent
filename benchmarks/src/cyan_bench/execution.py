from __future__ import annotations

import hashlib
import os
import shutil
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

from cyan_bench.cases import LoadedCase, case_fingerprint
from cyan_bench.models import ProcessCapture, Variant
from cyan_bench.paths import BenchmarkPaths


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


# 计算不可变 artifact 的 SHA-256
def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 运行基础设施命令并在失败时保留可操作的 stderr
def _checked(argv: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(
        argv,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        message = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"command failed ({result.returncode}): {argv!r}: {message}")


# 运行可能较久的准备命令并直接显示下载进度
def _checked_visible(argv: list[str], *, cwd: Path | None = None) -> None:
    result = subprocess.run(argv, cwd=cwd, stdin=subprocess.DEVNULL, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {argv!r}")


# 用仓库 URL 生成共享 Git 对象缓存目录名
def _repository_remote_path(paths: BenchmarkPaths, url: str) -> Path:
    return paths.cache / "repos" / hashlib.sha256(url.encode()).hexdigest()[:20]


# 用仓库 URL 与提交哈希生成不可变工作树缓存目录名
def _repository_cache_path(paths: BenchmarkPaths, url: str, revision: str) -> Path:
    identity = f"{url}\0{revision}".encode()
    return paths.cache / "repos" / hashlib.sha256(identity).hexdigest()[:20]


# 判断缓存目录是否为可读取的 Git 仓库
def _is_git_repository(path: Path) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "--git-dir"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


# 判断共享对象库是否已经包含目标提交
def _has_revision(path: Path, revision: str) -> bool:
    result = subprocess.run(
        ["git", "-C", str(path), "cat-file", "-e", f"{revision}^{{commit}}"],
        capture_output=True,
        check=False,
    )
    return result.returncode == 0


# 仅删除 benchmarks 缓存边界内可重建的半成品目录
def _discard_repository_cache(path: Path, paths: BenchmarkPaths) -> None:
    path.relative_to(paths.cache / "repos")
    shutil.rmtree(path)


# 下载并固定案例所需的上游提交
def prepare_repository(case: LoadedCase, paths: BenchmarkPaths) -> Path:
    remote = _repository_remote_path(paths, case.manifest.repo_url)
    snapshot = _repository_cache_path(paths, case.manifest.repo_url, case.manifest.repo_sha)
    remote.parent.mkdir(parents=True, exist_ok=True)
    if remote.exists() and not _is_git_repository(remote):
        _discard_repository_cache(remote, paths)
    if not remote.exists():
        _checked(
            [
                "git",
                "clone",
                "--filter=blob:none",
                "--no-checkout",
                case.manifest.repo_url,
                str(remote),
            ]
        )
    if not _has_revision(remote, case.manifest.repo_sha):
        _checked(
            [
                "git",
                "-C",
                str(remote),
                "fetch",
                "--depth=1",
                "origin",
                case.manifest.repo_sha,
            ]
        )
    if snapshot.exists() and not _is_git_repository(snapshot):
        _discard_repository_cache(snapshot, paths)
    if not snapshot.exists():
        _checked(
            [
                "git",
                "-C",
                str(remote),
                "worktree",
                "add",
                "--detach",
                str(snapshot),
                case.manifest.repo_sha,
            ]
        )
    _checked(["git", "-C", str(snapshot), "checkout", "--detach", case.manifest.repo_sha])
    return snapshot


# 以 frozen lock 创建或刷新独立训练环境
def prepare_environment(case: LoadedCase, paths: BenchmarkPaths) -> Path:
    environment = paths.environments / case.manifest.env_id
    lock = environment / "uv.lock"
    if not (environment / "pyproject.toml").is_file() or not lock.is_file():
        raise FileNotFoundError(f"missing locked environment: {environment}")
    _checked_visible(["uv", "sync", "--project", str(environment), "--frozen"])
    return environment


# 在无历史的新 Git 仓库中构造 control、buggy 或 fixed 工作区
def prepare_workspace(
    case: LoadedCase,
    paths: BenchmarkPaths,
    variant: Variant,
) -> Path:
    cache = _repository_cache_path(paths, case.manifest.repo_url, case.manifest.repo_sha)
    if not cache.exists():
        raise FileNotFoundError(f"repository not prepared: {cache}")
    head = subprocess.run(
        ["git", "-C", str(cache), "rev-parse", "HEAD"],
        capture_output=True,
        check=False,
        text=True,
    )
    expected = subprocess.run(
        ["git", "-C", str(cache), "rev-parse", f"{case.manifest.repo_sha}^{{commit}}"],
        capture_output=True,
        check=False,
        text=True,
    )
    if (
        head.returncode != 0
        or expected.returncode != 0
        or head.stdout.strip() != expected.stdout.strip()
    ):
        raise RuntimeError(f"repository cache revision mismatch: {cache}")
    workspace_root = paths.artifacts / "workspaces"
    workspace_root.mkdir(parents=True, exist_ok=True)
    workspace = workspace_root / f"{case.manifest.id}-{variant}-{time.time_ns()}"
    shutil.copytree(cache, workspace, ignore=shutil.ignore_patterns(".git"))
    try:
        overlay = case.root / "overlay"
        if overlay.is_dir():
            shutil.copytree(overlay, workspace, dirs_exist_ok=True)
        _checked(["git", "init", "-q"], cwd=workspace)
        _checked(["git", "config", "user.name", "cyan-bench"], cwd=workspace)
        _checked(["git", "config", "user.email", "cyan-bench@invalid"], cwd=workspace)
        workload = case.root / "workload.patch"
        if workload.is_file() and workload.stat().st_size:
            _checked(["git", "apply", "--check", str(workload)], cwd=workspace)
            _checked(["git", "apply", str(workload)], cwd=workspace)
        if variant in {"buggy", "fixed"}:
            fault = case.root / "fault.patch"
            _checked(["git", "apply", "--check", str(fault)], cwd=workspace)
            _checked(["git", "apply", str(fault)], cwd=workspace)
        if variant == "fixed":
            fix = case.root / "fix.patch"
            _checked(["git", "apply", "--check", str(fix)], cwd=workspace)
            _checked(["git", "apply", str(fix)], cwd=workspace)
        _checked(["git", "add", "-A"], cwd=workspace)
        _checked(
            ["git", "commit", "-q", "-m", f"{case.manifest.id} {variant} baseline"],
            cwd=workspace,
        )
    except Exception:
        shutil.rmtree(workspace)
        raise
    return workspace


# 将 manifest 中的占位符解析到本轮工作区与案例目录
def _expand(value: str, case: LoadedCase, workspace: Path) -> str:
    benchmark_cache = case.root.parents[1] / ".cache"
    return (
        value.replace("{workspace}", str(workspace))
        .replace("{case_dir}", str(case.root))
        .replace("{benchmark_cache}", str(benchmark_cache))
    )


# 将 python 命令绑定到案例的锁定虚拟环境
def command_for_workspace(
    case: LoadedCase,
    paths: BenchmarkPaths,
    workspace: Path,
) -> list[str]:
    argv = [_expand(item, case, workspace) for item in case.manifest.argv]
    if argv[0] == "python":
        python = paths.environments / case.manifest.env_id / ".venv" / "bin" / "python"
        if not python.is_file():
            raise FileNotFoundError(f"environment is not synced: {python}")
        argv[0] = str(python)
    return argv


# 构造与准入和完整 Incident track 共用的离线训练环境
def environment_for_workspace(case: LoadedCase, workspace: Path) -> dict[str, str]:
    env = os.environ.copy()
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV"):
        env.pop(key, None)
    env.update(
        {
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
            "TOKENIZERS_PARALLELISM": "false",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    env.update(
        {
            key: _expand(value, case, workspace)
            for key, value in case.manifest.environment.items()
        }
    )
    return env


# 将 Pydantic artifact 原子写入磁盘
def _write_capture(path: Path, capture: ProcessCapture) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(capture.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


# 在独立进程组中运行一次真实训练命令并持久化原始日志
def run_capture(
    case: LoadedCase,
    paths: BenchmarkPaths,
    workspace: Path,
    variant: Variant,
    repeat: int,
) -> tuple[ProcessCapture, Path]:
    run_dir = paths.artifacts / "captures" / case.manifest.id / variant / str(repeat)
    if run_dir.exists():
        raise FileExistsError(f"capture already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    stdout_path = run_dir / "stdout.log"
    stderr_path = run_dir / "stderr.log"
    argv = command_for_workspace(case, paths, workspace)
    cwd = (workspace / case.manifest.cwd).resolve()
    if not cwd.is_relative_to(workspace.resolve()):
        raise ValueError("resolved cwd escaped workspace")
    env = environment_for_workspace(case, workspace)
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=stdout,
            stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=case.manifest.timeout_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                returncode = process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                returncode = process.wait()
    capture = ProcessCapture(
        case_id=case.manifest.id,
        case_fingerprint=case_fingerprint(case),
        environment_lock_sha256=_sha256(
            paths.environments / case.manifest.env_id / "uv.lock"
        ),
        variant=variant,
        repeat=repeat,
        argv=argv,
        cwd=str(cwd),
        returncode=returncode,
        timed_out=timed_out,
        duration_seconds=round(time.monotonic() - started, 6),
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        stdout_bytes=stdout_path.stat().st_size,
        stderr_bytes=stderr_path.stat().st_size,
        stdout_sha256=_sha256(stdout_path),
        stderr_sha256=_sha256(stderr_path),
        created_at=_now(),
    )
    _write_capture(run_dir / "process.json", capture)
    return capture, run_dir


# 删除只用于生成日志的临时工作区
def discard_workspace(workspace: Path, paths: BenchmarkPaths) -> None:
    root = (paths.artifacts / "workspaces").resolve()
    target = workspace.resolve()
    if target.parent != root:
        raise ValueError("refusing to remove a non-benchmark workspace")
    shutil.rmtree(target)
