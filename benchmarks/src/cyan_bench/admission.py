from __future__ import annotations

import hashlib
import shutil
from datetime import UTC, datetime
from pathlib import Path

from cyan_bench.cases import (
    LoadedCase,
    case_fingerprint,
    resolve_anchors,
    write_resolved_anchors,
)
from cyan_bench.execution import discard_workspace, prepare_workspace, run_capture
from cyan_bench.models import AdmissionArtifact, ProcessCapture
from cyan_bench.paths import BenchmarkPaths


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


# 计算当前案例训练环境 lock 的内容哈希
def _environment_lock_sha(case: LoadedCase, paths: BenchmarkPaths) -> str:
    lock = paths.environments / case.manifest.env_id / "uv.lock"
    with lock.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


# 检查阶段里程碑是否确实出现在失败之前的真实日志中
def _milestone_seen(case: LoadedCase, run_dir: Path) -> bool:
    anchor = case.manifest.milestone_anchor
    if anchor is None:
        return True
    needle = anchor.encode("utf-8")
    return any(needle in (run_dir / name).read_bytes() for name in ("stdout.log", "stderr.log"))


# 仅移除可由 harness 重建的单轮过期 artifact
def _discard_generated_capture(run_dir: Path, paths: BenchmarkPaths) -> None:
    run_dir.relative_to(paths.artifacts / "captures")
    shutil.rmtree(run_dir)


# 对一个案例执行 control、buggy、fixed 各三次准入验证
def admit_case(case: LoadedCase, paths: BenchmarkPaths, repeats: int = 3) -> AdmissionArtifact:
    reasons: list[str] = []
    captures: list[str] = []
    expected_fingerprint = case_fingerprint(case)
    expected_lock_sha = _environment_lock_sha(case, paths)
    for variant in ("control", "buggy", "fixed"):
        for repeat in range(1, repeats + 1):
            run_dir = paths.artifacts / "captures" / case.manifest.id / variant / str(repeat)
            process_path = run_dir / "process.json"
            if process_path.is_file():
                capture = ProcessCapture.model_validate_json(
                    process_path.read_text(encoding="utf-8")
                )
                if capture.variant != variant or capture.repeat != repeat:
                    raise ValueError(f"capture identity mismatch: {process_path}")
                stale = (
                    capture.case_fingerprint != expected_fingerprint
                    or capture.environment_lock_sha256 != expected_lock_sha
                )
                if stale:
                    _discard_generated_capture(run_dir, paths)
            if not process_path.is_file():
                if run_dir.exists():
                    _discard_generated_capture(run_dir, paths)
                workspace = prepare_workspace(case, paths, variant)
                try:
                    capture, run_dir = run_capture(
                        case, paths, workspace, variant, repeat
                    )
                finally:
                    discard_workspace(workspace, paths)
            captures.append(str(run_dir.relative_to(paths.artifacts)))
            if capture.timed_out:
                reasons.append(f"{variant}/{repeat}: timed out")
            if variant == "buggy":
                if capture.returncode in {None, 0}:
                    reasons.append(f"buggy/{repeat}: expected non-zero exit")
                if not _milestone_seen(case, run_dir):
                    reasons.append(f"buggy/{repeat}: milestone anchor missing")
                try:
                    anchors = resolve_anchors(
                        case.expected,
                        run_dir / "stdout.log",
                        run_dir / "stderr.log",
                    )
                except ValueError as exc:
                    reasons.append(f"buggy/{repeat}: {exc}")
                else:
                    write_resolved_anchors(run_dir / "gold-ranges.json", anchors)
            elif capture.returncode != 0:
                reasons.append(f"{variant}/{repeat}: expected exit 0, got {capture.returncode}")
    artifact = AdmissionArtifact(
        case_id=case.manifest.id,
        admitted=not reasons,
        reasons=reasons,
        captures=captures,
        created_at=_now(),
    )
    output = paths.artifacts / "admissions" / case.manifest.id / "admission.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return artifact
