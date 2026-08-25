from __future__ import annotations

import hashlib
import json
from collections import Counter

from cyan_bench.cases import LoadedCase, case_fingerprint, discover_cases
from cyan_bench.models import DiagnosisRunArtifact, ProcessCapture, ResolvedAnchor
from cyan_bench.paths import BenchmarkPaths

_LONG_LOG_BYTES = 40 * 1024
_CONTROL_ROLES = {"short_quiet", "long_clean", "warning_heavy"}


# 读取已完成三轮准入的状态
def _admitted_three_times(case: LoadedCase, paths: BenchmarkPaths) -> bool:
    admission_path = paths.artifacts / "admissions" / case.manifest.id / "admission.json"
    if not admission_path.is_file():
        return False
    payload = json.loads(admission_path.read_text(encoding="utf-8"))
    if not payload.get("admitted"):
        return False
    lock_path = paths.environments / case.manifest.env_id / "uv.lock"
    if not lock_path.is_file():
        return False
    lock_sha = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    fingerprint = case_fingerprint(case)
    capture_paths = [
        (
            paths.artifacts
            / "captures"
            / case.manifest.id
            / variant
            / str(repeat)
            / "process.json"
        )
        for variant in ("control", "buggy", "fixed")
        for repeat in range(1, 4)
    ]
    if not all(path.is_file() for path in capture_paths):
        return False
    captures = [
        ProcessCapture.model_validate_json(path.read_text(encoding="utf-8"))
        for path in capture_paths
    ]
    return all(
        capture.case_fingerprint == fingerprint
        and capture.environment_lock_sha256 == lock_sha
        for capture in captures
    )


# 判断案例是否已有一轮真实日志达到 40 KiB 门槛
def _has_observed_long_log(case: LoadedCase, paths: BenchmarkPaths) -> bool:
    for repeat in range(1, 4):
        capture_path = (
            paths.artifacts
            / "captures"
            / case.manifest.id
            / "buggy"
            / str(repeat)
            / "process.json"
        )
        if not capture_path.is_file():
            continue
        capture = ProcessCapture.model_validate_json(capture_path.read_text(encoding="utf-8"))
        if capture.stdout_bytes + capture.stderr_bytes >= _LONG_LOG_BYTES:
            return True
    return False


# 判断必要证据是否在同一日志流真实跨越 32 KiB
def _has_wide_evidence_span(case: LoadedCase, paths: BenchmarkPaths) -> bool:
    for repeat in range(1, 4):
        ranges_path = (
            paths.artifacts
            / "captures"
            / case.manifest.id
            / "buggy"
            / str(repeat)
            / "gold-ranges.json"
        )
        if not ranges_path.is_file():
            continue
        ranges = [
            ResolvedAnchor.model_validate(item)
            for item in json.loads(ranges_path.read_text(encoding="utf-8"))
        ]
        for source in ("stdout", "stderr"):
            offsets = [item.start for item in ranges if item.source == source]
            if offsets and max(offsets) - min(offsets) > 32 * 1024:
                return True
    return False


# 从真实 FullNative API usage 中确认输入超过一万 token
def _has_observed_ten_k_tokens(case: LoadedCase, paths: BenchmarkPaths) -> bool:
    pattern = f"run-sets/*/diagnosis/{case.manifest.id}/buggy/*/full_native/diagnosis.json"
    for artifact_path in paths.artifacts.glob(pattern):
        artifact = DiagnosisRunArtifact.model_validate_json(
            artifact_path.read_text(encoding="utf-8")
        )
        if artifact.input_tokens is not None and artifact.input_tokens > 10_000:
            return True
    return False


# 审计 v1 数据集配额与已落盘的真实性证据
def audit_dataset(paths: BenchmarkPaths) -> dict[str, object]:
    cases = discover_cases(paths.cases)
    split_counts = Counter(case.manifest.split for case in cases)
    stage_counts = Counter(case.manifest.failure_stage for case in cases)
    mechanism_counts = Counter(case.manifest.mechanism_id for case in cases)
    control_counts = Counter(
        case.manifest.control_role for case in cases if case.manifest.control_role is not None
    )
    admitted = [case.manifest.id for case in cases if _admitted_three_times(case, paths)]
    long_bytes = [case.manifest.id for case in cases if _has_observed_long_log(case, paths)]
    long_tokens = [case.manifest.id for case in cases if _has_observed_ten_k_tokens(case, paths)]
    wide_evidence = [case.manifest.id for case in cases if _has_wide_evidence_span(case, paths)]
    reasons: list[str] = []
    if split_counts != Counter({"dev": 6, "test": 9}):
        reasons.append(f"split quota is {dict(split_counts)}, expected dev=6/test=9")
    control_roles_valid = set(control_counts) == _CONTROL_ROLES and all(
        value == 1 for value in control_counts.values()
    )
    if not control_roles_valid:
        reasons.append("control roles must contain short_quiet, long_clean and warning_heavy once")
    llm_cases = sum(case.manifest.training_domain == "llm" for case in cases)
    if llm_cases < 10:
        reasons.append("fewer than 10 LLM training cases")
    stage_quota_valid = (
        stage_counts["startup"] <= 4
        and stage_counts["mid_run"] >= 6
        and stage_counts["finalization"] >= 4
    )
    if not stage_quota_valid:
        reasons.append(f"failure-stage quota is {dict(stage_counts)}")
    no_safe_patch = sum(not case.manifest.patchable for case in cases)
    if no_safe_patch < 3:
        reasons.append("fewer than 3 no-safe-single-file-patch cases")
    duplicates = sorted(name for name, count in mechanism_counts.items() if count > 2)
    if duplicates:
        reasons.append(f"mechanisms repeated more than twice: {duplicates}")
    if len(admitted) != 15:
        reasons.append(f"only {len(admitted)}/15 cases passed three-repeat admission")
    if len(long_bytes) < 4:
        reasons.append(f"only {len(long_bytes)}/4 cases have observed logs >=40 KiB")
    if len(long_tokens) < 4:
        reasons.append(f"only {len(long_tokens)}/4 cases have API-verified input >10k tokens")
    if len(wide_evidence) < 2:
        reasons.append(f"only {len(wide_evidence)}/2 cases have required evidence span >32 KiB")
    return {
        "schema_version": 1,
        "ready": not reasons,
        "reasons": reasons,
        "case_count": len(cases),
        "split_counts": dict(split_counts),
        "stage_counts": dict(stage_counts),
        "llm_cases": llm_cases,
        "no_safe_patch_cases": no_safe_patch,
        "control_roles": dict(control_counts),
        "admitted_case_ids": admitted,
        "observed_long_log_case_ids": long_bytes,
        "api_verified_ten_k_token_case_ids": long_tokens,
        "wide_evidence_case_ids": wide_evidence,
    }
