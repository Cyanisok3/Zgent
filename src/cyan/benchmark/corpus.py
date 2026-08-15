from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from cyan.benchmark.models import (
    CaseManifest,
    EvidenceStream,
    GoldFact,
    LogArtifact,
    ReplayReceipt,
    SourceProvenance,
)
from cyan.core.incidents.models import FailureCapsule, LogSnapshot

CAPSULE_BYTES = 32 * 1024
LOGDX_VERSION = "v1.2"
LOGDX_URL = "https://github.com/eyuansu62/LogDx/archive/refs/tags/v1.2.tar.gz"
LOGDX_ARCHIVE_SHA256 = "c1b1da8fa604be65bcebfbee6d8a50875c739b928374591df0467584f3bd9184"


# 计算文件 SHA-256，供 manifest 和下载校验复用
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 对两个日志字节串计算稳定的联合摘要
def _combined_log_sha256(stdout: bytes, stderr: bytes) -> str:
    digest = hashlib.sha256()
    digest.update(len(stdout).to_bytes(8, "big"))
    digest.update(stdout)
    digest.update(stderr)
    return digest.hexdigest()


# 将对象以稳定 JSON 格式写入语料目录
def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 返回日志尾部快照并保留原始 byte offset
def _snapshot(path: Path, limit: int) -> LogSnapshot:
    size = path.stat().st_size
    start = max(0, size - limit)
    with path.open("rb") as handle:
        handle.seek(start)
        raw = handle.read(limit)
    return LogSnapshot(
        size=size,
        sha256=sha256_file(path),
        included_start=start,
        included_end=start + len(raw),
        tail=raw.decode("utf-8", errors="replace"),
    )


# 按 stderr 优先规则构造与生产 Failure Capsule 一致的 32 KiB 初始证据
def _capsule_snapshots(stdout_path: Path, stderr_path: Path) -> tuple[LogSnapshot, LogSnapshot]:
    stderr_limit = min(stderr_path.stat().st_size, CAPSULE_BYTES)
    stdout_limit = CAPSULE_BYTES - stderr_limit
    return _snapshot(stdout_path, stdout_limit), _snapshot(stderr_path, stderr_limit)


# 把日志行号区间转换为稳定 UTF-8 byte 半开区间
def _line_range_to_bytes(raw: bytes, start_line: int, end_line: int) -> tuple[int, int]:
    if start_line < 1 or end_line < start_line:
        raise ValueError("invalid one-based line range")
    starts = [0]
    for index, value in enumerate(raw):
        if value == 10:
            starts.append(index + 1)
    if start_line > len(starts):
        raise ValueError("line range starts beyond log")
    start = starts[start_line - 1]
    end = starts[end_line] if end_line < len(starts) else len(raw)
    return start, end


# 安全解析 tar，拒绝绝对路径、父目录和符号链接
def _safe_extract(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    with tarfile.open(archive, "r:gz") as bundle:
        for member in bundle.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"archive member escapes destination: {member.name}")
            if member.issym() or member.islnk():
                raise ValueError(f"archive links are not accepted: {member.name}")
        bundle.extractall(destination, filter="data")


class Corpus:
    # 绑定一个语料根目录并提供严格的 manifest/log 读取
    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    # 返回按 case_id 排序的全部 manifest
    def cases(self) -> list[CaseManifest]:
        manifests: list[CaseManifest] = []
        for path in sorted((self.root / "cases").glob("*/manifest.json")):
            manifests.append(CaseManifest.model_validate_json(path.read_text(encoding="utf-8")))
        return manifests

    # 解析并约束一个 Case 的日志路径
    def log_path(self, case: CaseManifest, stream: str) -> Path:
        artifact = case.logs[stream]  # type: ignore[index]
        path = (self.root / artifact.path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError("log path escapes corpus root")
        return path

    # 验证日志哈希、范围、split 泄漏和可选数量门槛
    def validate(self, *, require_complete_core: bool = False) -> dict[str, object]:
        cases = self.cases()
        errors: list[str] = []
        templates: dict[tuple[str, str], set[str]] = {}
        hashes: dict[str, set[str]] = {}
        for case in cases:
            fixed_valid = True
            for stream, artifact in case.logs.items():
                path = self.log_path(case, stream)
                if not path.is_file():
                    errors.append(f"{case.case_id}: missing {stream} log")
                    continue
                if path.stat().st_size != artifact.size:
                    errors.append(f"{case.case_id}: {stream} size mismatch")
                if sha256_file(path) != artifact.sha256:
                    errors.append(f"{case.case_id}: {stream} SHA-256 mismatch")
            for stream, artifact in case.fixed_logs.items():
                path = (self.root / artifact.path).resolve()
                if not path.is_relative_to(self.root) or not path.is_file():
                    errors.append(f"{case.case_id}: missing fixed {stream} log")
                    fixed_valid = False
                    continue
                if path.stat().st_size != artifact.size:
                    errors.append(f"{case.case_id}: fixed {stream} size mismatch")
                if sha256_file(path) != artifact.sha256:
                    errors.append(f"{case.case_id}: fixed {stream} SHA-256 mismatch")
            if (
                case.replay is not None
                and set(case.fixed_logs) == {"stdout", "stderr"}
                and fixed_valid
            ):
                failing_digest = _combined_log_sha256(
                    self.log_path(case, "stdout").read_bytes(),
                    self.log_path(case, "stderr").read_bytes(),
                )
                fixed_digest = _combined_log_sha256(
                    (self.root / case.fixed_logs["stdout"].path).resolve().read_bytes(),
                    (self.root / case.fixed_logs["stderr"].path).resolve().read_bytes(),
                )
                if failing_digest != case.replay.failing_log_sha256:
                    errors.append(f"{case.case_id}: failing replay digest mismatch")
                if fixed_digest != case.replay.fixed_log_sha256:
                    errors.append(f"{case.case_id}: fixed replay digest mismatch")
            combined = hashlib.sha256(
                "".join(
                    case.logs[stream].sha256 for stream in ("stdout", "stderr")
                ).encode("ascii")
            ).hexdigest()
            hashes.setdefault(combined, set()).add(case.split)
            templates.setdefault((case.workload, case.template_id), set()).add(case.split)
        for key, splits in templates.items():
            if len(splits) > 1 and key[0] != "logdx-ci":
                errors.append(f"template leakage {key}: {sorted(splits)}")
        for digest, splits in hashes.items():
            if len(splits) > 1:
                errors.append(f"log hash leakage {digest}: {sorted(splits)}")
        counts = Counter((case.tier, case.split) for case in cases)
        core_count = sum(case.tier == "cyan_core" for case in cases)
        historical_cases = [
            case
            for case in cases
            if case.tier == "cyan_core" and case.source.historical
        ]
        historical_count = len(historical_cases)
        for case in historical_cases:
            commit_pair = bool(case.source.failing_commit and case.source.fixing_commit)
            revision_pair = bool(
                case.source.issue_url
                and case.source.failing_revision
                and case.source.fixing_revision
            )
            if not (commit_pair or revision_pair):
                errors.append(f"{case.case_id}: incomplete historical provenance")
        if require_complete_core:
            for case in cases:
                if case.tier != "cyan_core" or case.split != "test":
                    continue
                if case.gold_review_status != "approved":
                    errors.append(f"{case.case_id}: test gold is not approved")
                if any(fact.review_passes < 2 for fact in case.gold_facts):
                    errors.append(f"{case.case_id}: test gold lacks two review passes")
        if require_complete_core and (core_count != 60 or historical_count != 12):
            errors.append(
                f"complete core requires 60 cases including 12 historical replays; "
                f"found {core_count} and {historical_count}"
            )
        return {
            "valid": not errors,
            "cases": len(cases),
            "counts": {f"{tier}/{split}": count for (tier, split), count in sorted(counts.items())},
            "historical_core_cases": historical_count,
            "errors": errors,
            "fingerprint": self.fingerprint(cases),
        }

    # 根据排序后的规范化 manifest 计算数据集 fingerprint
    def fingerprint(self, cases: Iterable[CaseManifest] | None = None) -> str:
        digest = hashlib.sha256()
        for case in sorted(cases or self.cases(), key=lambda item: item.case_id):
            payload = case.model_dump(mode="json", exclude_none=True)
            source = payload["source"]
            assert isinstance(source, dict)
            for field in (
                "issue_url",
                "failing_revision",
                "fixing_revision",
                "runtime_image",
            ):
                if source.get(field) is None:
                    source.pop(field, None)
            if source.get("historical") is False:
                source.pop("historical", None)
            digest.update(
                json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
                    "utf-8"
                )
            )
            digest.update(b"\n")
        return digest.hexdigest()


# 运行一个真实子进程并返回原始 stdout/stderr
def _run(argv: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[bytes]:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else f"{source_root}{os.pathsep}{existing}"
    )
    return subprocess.run(argv, cwd=cwd, env=environment, check=False, capture_output=True)


# 构造一段可重复的 Data/ML fixture 程序
def _fixture_program(
    workload: str,
    template: str,
    variant: int,
    *,
    fixed: bool,
) -> str:
    noise_lines = 1200 + variant * 600
    prefix = {
        "data_processing": "rows_read=128 shard=",
        "feature_transform": "feature_batch=64 transform=normalize step=",
        "cpu_training": "epoch=1 loss=0.812 step=",
    }[workload]
    lines = [
        "import sys",
        f"noise_lines = {noise_lines}",
        f"prefix = {prefix!r}",
        "for i in range(noise_lines):",
        "    print(f'{prefix}{i}')",
    ]
    if fixed:
        lines.extend(["print('workflow verification completed')", "raise SystemExit(0)"])
        return "\n".join(lines)
    if template == "main_nonzero":
        lines.extend(
            [
                "print('CYAN_EVIDENCE shape mismatch: expected 32 features, got 31', "
                "file=sys.stderr)",
                "print('Traceback: ValueError in train_batch', file=sys.stderr)",
                "raise SystemExit(2)",
            ]
        )
    elif template == "check_failed":
        lines.extend(
            [
                "print('CYAN_EVIDENCE schema check failed: column label is missing', "
                "file=sys.stderr)",
                "print('check_id=schema-check returncode=4', file=sys.stderr)",
                "raise SystemExit(4)",
            ]
        )
    elif template == "postflight_violation":
        lines.extend(
            [
                "print('CYAN_EVIDENCE output checkpoint skipped after empty training split')",
                "print('workflow command returned zero but output is stale', file=sys.stderr)",
                "raise SystemExit(5)",
            ]
        )
    else:
        lines.extend(
            [
                "print('required input data/train.csv is missing', file=sys.stderr)",
                "raise SystemExit(3)",
            ]
        )
    if template in {"main_nonzero", "check_failed", "postflight_violation"}:
        lines[-1:-1] = [
            "for j in range(2200):",
            "    print(f'cleanup telemetry step={j}')",
            "    print(f'cleanup telemetry step={j}', file=sys.stderr)",
            "print('secondary shutdown completed', file=sys.stderr)",
        ]
    return "\n".join(lines)


# 从一次真实失败和成功运行写出一个 CI fixture Case
def _write_ci_fixture(
    root: Path,
    workload: str,
    template: str,
    variant: int,
    split: str,
) -> CaseManifest:
    case_id = f"core-{workload}-{template}-{variant}"
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    failing_argv = [
        sys.executable,
        "-c",
        _fixture_program(workload, template, variant, fixed=False),
    ]
    fixed_argv = [
        sys.executable,
        "-c",
        _fixture_program(workload, template, variant, fixed=True),
    ]
    failed = _run(failing_argv)
    fixed = _run(fixed_argv)
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    fixed_stdout_path = case_dir / "fixed.stdout.log"
    fixed_stderr_path = case_dir / "fixed.stderr.log"
    stdout_path.write_bytes(failed.stdout)
    stderr_path.write_bytes(failed.stderr)
    fixed_stdout_path.write_bytes(fixed.stdout)
    fixed_stderr_path.write_bytes(fixed.stderr)
    stdout_snapshot, stderr_snapshot = _capsule_snapshots(stdout_path, stderr_path)
    phase = "main"
    failure_kind = "process_exit"
    recovery_kind = "patch"
    check_id = None
    artifact_path = None
    violation_rule = None
    if template == "check_failed":
        phase = "preflight"
        failure_kind = "contract_violation"
        check_id = "schema-check"
    elif template == "postflight_violation":
        phase = "postflight"
        failure_kind = "contract_violation"
        artifact_path = "artifacts/model.bin"
        violation_rule = "fresh"
    elif template == "deterministic_input":
        phase = "preflight"
        failure_kind = "contract_violation"
        recovery_kind = "operator_action"
        artifact_path = "data/train.csv"
        violation_rule = "required"
    capsule = FailureCapsule(
        job_id=case_id,
        attempt_id=f"attempt-{variant}",
        argv=failing_argv,
        cwd="/benchmark/workspace",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        failure_kind=failure_kind,  # type: ignore[arg-type]
        returncode=failed.returncode,
        phase=phase,  # type: ignore[arg-type]
        check_id=check_id,
        artifact_path=artifact_path,
        violation_rule=violation_rule,
        stdout=stdout_snapshot,
        stderr=stderr_snapshot,
    )
    gold: list[GoldFact] = []
    if template != "deterministic_input":
        marker = b"CYAN_EVIDENCE"
        stream: EvidenceStream = (
            "stdout" if template == "postflight_violation" else "stderr"
        )
        raw = stdout_path.read_bytes() if stream == "stdout" else stderr_path.read_bytes()
        start = raw.index(marker)
        end = raw.find(b"\n", start)
        end = len(raw) if end < 0 else end + 1
        gold.append(
            GoldFact(
                fact_id=f"{case_id}-cause",
                importance="essential",
                stream=stream,
                byte_start=start,
                byte_end=end,
                description="The emitted line identifies the injected causal failure.",
                provenance="injected",
                review_passes=2 if split == "test" else 1,
            )
        )
    logs: dict[EvidenceStream, LogArtifact] = {
        "stdout": LogArtifact(
            path=str(stdout_path.relative_to(root)),
            sha256=sha256_file(stdout_path),
            size=stdout_path.stat().st_size,
        ),
        "stderr": LogArtifact(
            path=str(stderr_path.relative_to(root)),
            sha256=sha256_file(stderr_path),
            size=stderr_path.stat().st_size,
        ),
    }
    fixed_logs: dict[EvidenceStream, LogArtifact] = {
        "stdout": LogArtifact(
            path=str(fixed_stdout_path.relative_to(root)),
            sha256=sha256_file(fixed_stdout_path),
            size=fixed_stdout_path.stat().st_size,
        ),
        "stderr": LogArtifact(
            path=str(fixed_stderr_path.relative_to(root)),
            sha256=sha256_file(fixed_stderr_path),
            size=fixed_stderr_path.stat().st_size,
        ),
    }
    manifest = CaseManifest(
        case_id=case_id,
        tier="cyan_core",
        workload=workload,
        split=split,  # type: ignore[arg-type]
        template_id=f"{template}-v{variant}",
        failure_kind=failure_kind,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        expected_recovery_kind=recovery_kind,  # type: ignore[arg-type]
        capsule=capsule,
        logs=logs,
        fixed_logs=fixed_logs,
        source=SourceProvenance(repository="cyan-controlled-fixtures", license="MIT"),
        replay=ReplayReceipt(
            failing_argv=failing_argv,
            fixed_argv=fixed_argv,
            failing_returncode=failed.returncode,
            fixed_returncode=fixed.returncode,
            failing_log_sha256=_combined_log_sha256(failed.stdout, failed.stderr),
            fixed_log_sha256=_combined_log_sha256(fixed.stdout, fixed.stderr),
        ),
        gold_facts=gold,
        expected_diagnosis_terms=(
            []
            if template == "deterministic_input"
            else {
                "main_nonzero": ["shape", "32", "31"],
                "check_failed": ["schema", "label"],
                "postflight_violation": ["checkpoint", "empty", "split"],
            }[template]
        ),
        annotation_notes=(
            "CI-only deterministic subprocess fixture; it is not part of the released Core corpus."
        ),
    )
    _write_json(case_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


# 生成八个不依赖 Data/ML 包的 CI 小型 Case
def prepare_ci_core(root: Path, *, limit: int = 8) -> list[CaseManifest]:
    root = root.expanduser().resolve()
    workloads = ["data_processing", "feature_transform", "cpu_training"]
    templates = [
        "main_nonzero",
        "check_failed",
        "postflight_violation",
        "deterministic_input",
    ]
    specs = [
        (workload, template, variant)
        for variant in range(4)
        for workload in workloads
        for template in templates
    ][:limit]
    cases: list[CaseManifest] = []
    for index, (workload, template, variant) in enumerate(specs):
        split = "train" if variant < 3 else ("dev" if index % 3 != 0 else "test")
        cases.append(_write_ci_fixture(root, workload, template, variant, split))
    return cases


# 返回真实异常最后一行在 stderr 中的稳定字节区间
def _exception_range(raw: bytes) -> tuple[int, int, str]:
    boundary = raw.find(b"cleanup domain=")
    traceback_raw = raw if boundary < 0 else raw[:boundary]
    lines = traceback_raw.splitlines(keepends=True)
    evidence = next((line for line in reversed(lines) if line.strip()), b"")
    if not evidence:
        raise ValueError("real workload failure produced no exception summary")
    start = raw.find(evidence)
    return start, start + len(evidence), evidence.decode("utf-8", errors="replace").strip()


# 返回真实 workload Case 的失败阶段和恢复语义
def _failure_metadata(template: str) -> tuple[str, str, str, str | None, str | None, str | None]:
    if template == "check_failed":
        return "contract_violation", "preflight", "patch", "quality-check", None, None
    if template == "postflight_violation":
        return (
            "contract_violation",
            "postflight",
            "patch",
            None,
            "artifacts/result.bin",
            "fresh",
        )
    if template == "deterministic_input":
        return (
            "contract_violation",
            "preflight",
            "operator_action",
            None,
            "data/input.csv",
            "required",
        )
    return "process_exit", "main", "patch", None, None, None


# 构造一个语料内日志文件的校验描述
def _log_artifact(root: Path, path: Path) -> LogArtifact:
    return LogArtifact(
        path=str(path.relative_to(root)),
        sha256=sha256_file(path),
        size=path.stat().st_size,
    )


# 运行一个真实 adapter 的失败和修复版本并写入 Core Case
def _write_real_core_case(
    root: Path,
    domain: str,
    adapter: int,
    template: str,
    split: str,
) -> CaseManifest:
    case_id = f"core-{domain}-{template}-{adapter}"
    case_dir = root / "cases" / case_id
    if case_dir.exists():
        shutil.rmtree(case_dir)
    workspace = case_dir / "workspace"
    workspace.mkdir(parents=True)
    base_argv = [
        sys.executable,
        "-m",
        "cyan.benchmark.workload",
        "--domain",
        domain,
        "--adapter",
        str(adapter),
        "--failure",
        template,
    ]
    failed = _run(base_argv, cwd=workspace)
    shutil.rmtree(workspace)
    workspace.mkdir()
    fixed_argv = [*base_argv, "--fixed"]
    fixed = _run(fixed_argv, cwd=workspace)
    if failed.returncode == 0 or fixed.returncode != 0:
        raise RuntimeError(
            f"{case_id}: replay gate failed; failing={failed.returncode}, "
            f"fixed={fixed.returncode}; "
            f"failing stderr={failed.stderr[-1000:]!r}; fixed stderr={fixed.stderr[-1000:]!r}"
        )
    stdout_path = case_dir / "stdout.log"
    stderr_path = case_dir / "stderr.log"
    fixed_stdout_path = case_dir / "fixed.stdout.log"
    fixed_stderr_path = case_dir / "fixed.stderr.log"
    stdout_path.write_bytes(failed.stdout)
    stderr_path.write_bytes(failed.stderr)
    fixed_stdout_path.write_bytes(fixed.stdout)
    fixed_stderr_path.write_bytes(fixed.stderr)
    stdout_snapshot, stderr_snapshot = _capsule_snapshots(stdout_path, stderr_path)
    failure_kind, phase, recovery, check_id, artifact_path, rule = _failure_metadata(template)
    gold: list[GoldFact] = []
    rubric: list[str] = (
        ["required workflow input is missing"] if template == "deterministic_input" else []
    )
    if template != "deterministic_input":
        start, end, summary = _exception_range(failed.stderr)
        gold.append(
            GoldFact(
                fact_id=f"{case_id}-cause",
                importance="essential",
                stream="stderr",
                byte_start=start,
                byte_end=end,
                description=summary,
                provenance="injected",
                review_passes=0 if split == "test" else 1,
            )
        )
        rubric = [summary]
    manifest = CaseManifest(
        case_id=case_id,
        tier="cyan_core",
        workload=f"{domain}-adapter-{adapter}",
        split=split,  # type: ignore[arg-type]
        template_id=f"{domain}-adapter-{adapter}-{template}",
        failure_kind=failure_kind,  # type: ignore[arg-type]
        phase=phase,  # type: ignore[arg-type]
        expected_recovery_kind=recovery,  # type: ignore[arg-type]
        capsule=FailureCapsule(
            job_id=case_id,
            attempt_id=f"attempt-{adapter}",
            argv=base_argv,
            cwd="/benchmark/workspace",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            failure_kind=failure_kind,  # type: ignore[arg-type]
            returncode=failed.returncode,
            phase=phase,  # type: ignore[arg-type]
            check_id=check_id,
            artifact_path=artifact_path,
            violation_rule=rule,
            stdout=stdout_snapshot,
            stderr=stderr_snapshot,
        ),
        logs={
            "stdout": _log_artifact(root, stdout_path),
            "stderr": _log_artifact(root, stderr_path),
        },
        fixed_logs={
            "stdout": _log_artifact(root, fixed_stdout_path),
            "stderr": _log_artifact(root, fixed_stderr_path),
        },
        source=SourceProvenance(
            repository={
                "data_processing": "https://github.com/pandas-dev/pandas",
                "feature_transform": "https://github.com/scikit-learn/scikit-learn",
                "cpu_training": "https://github.com/pytorch/examples",
            }[domain],
            license="BSD-3-Clause",
            upstream_case_id=f"{domain}-adapter-{adapter}",
        ),
        replay=ReplayReceipt(
            failing_argv=base_argv,
            fixed_argv=fixed_argv,
            failing_returncode=failed.returncode,
            fixed_returncode=fixed.returncode,
            failing_log_sha256=_combined_log_sha256(failed.stdout, failed.stderr),
            fixed_log_sha256=_combined_log_sha256(fixed.stdout, fixed.stderr),
        ),
        gold_facts=gold,
        expected_diagnosis_terms=rubric,
        root_cause_rubric=rubric,
        gold_review_status="draft" if split == "test" else "approved",
        annotation_notes=(
            "Actual Pandas/scikit-learn/PyTorch CPU operation with a controlled fault; "
            "it does not represent production fault frequency."
        ),
    )
    _write_json(case_dir / "manifest.json", manifest.model_dump(mode="json"))
    shutil.rmtree(workspace)
    return manifest


# 生成四十八个真实 Data/ML 运算的受控 Cyan Core Case
def prepare_controlled_core(root: Path, *, limit: int = 48) -> list[CaseManifest]:
    root = root.expanduser().resolve()
    templates = ["main_nonzero", "check_failed", "postflight_violation", "deterministic_input"]
    adapters = [
        (domain, adapter)
        for domain in ("data_processing", "feature_transform", "cpu_training")
        for adapter in range(4)
    ]
    split_by_adapter = {
        **{identity: "train" for identity in adapters[:9]},
        adapters[9]: "dev",
        adapters[10]: "dev",
        adapters[11]: "test",
    }
    specs = [
        (domain, adapter, template)
        for domain, adapter in adapters
        for template in templates
    ][:limit]
    return [
        _write_real_core_case(root, domain, adapter, template, split_by_adapter[(domain, adapter)])
        for domain, adapter, template in specs
    ]


# 从已审核 bundle 导入历史故障，保留其失败和修复提交凭证
def import_historical_bundles(source_root: Path, root: Path) -> list[CaseManifest]:
    imported: list[CaseManifest] = []
    for manifest_path in sorted(source_root.glob("*/manifest.json")):
        manifest = CaseManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
        if manifest.tier != "cyan_core" or not manifest.source.historical:
            raise ValueError(f"{manifest.case_id}: expected a historical cyan_core case")
        target_dir = root / "cases" / manifest.case_id
        target_dir.mkdir(parents=True, exist_ok=True)
        for stream, artifact in manifest.logs.items():
            source_log = (source_root / artifact.path).resolve()
            if not source_log.is_relative_to(source_root.resolve()):
                raise ValueError(f"{manifest.case_id}: source log escapes bundle root")
            target_log = target_dir / f"{stream}.log"
            shutil.copyfile(source_log, target_log)
        for stream, artifact in manifest.fixed_logs.items():
            source_log = (source_root / artifact.path).resolve()
            if not source_log.is_relative_to(source_root.resolve()):
                raise ValueError(f"{manifest.case_id}: fixed source log escapes bundle root")
            target_log = target_dir / f"fixed.{stream}.log"
            shutil.copyfile(source_log, target_log)
        rewritten = manifest.model_copy(
            update={
                "logs": {
                    stream: artifact.model_copy(
                        update={"path": str((target_dir / f"{stream}.log").relative_to(root))}
                    )
                    for stream, artifact in manifest.logs.items()
                },
                "fixed_logs": {
                    stream: artifact.model_copy(
                        update={
                            "path": str(
                                (target_dir / f"fixed.{stream}.log").relative_to(root)
                            )
                        }
                    )
                    for stream, artifact in manifest.fixed_logs.items()
                },
            }
        )
        _write_json(target_dir / "manifest.json", rewritten.model_dump(mode="json"))
        imported.append(rewritten)
    return imported


# 下载并校验 LogDx-CI v1.2 源码语料包
def fetch_logdx(destination: Path) -> Path:
    destination.mkdir(parents=True, exist_ok=True)
    archive = destination / "logdx-v1.2.tar.gz"
    if not archive.exists():
        urllib.request.urlretrieve(LOGDX_URL, archive)
    actual = sha256_file(archive)
    if actual != LOGDX_ARCHIVE_SHA256:
        raise ValueError(f"LogDx archive SHA-256 mismatch: {actual}")
    extracted = destination / "LogDx-1.2"
    if not extracted.exists():
        _safe_extract(archive, destination)
    return extracted


# 将 LogDx 行级标注转换为统一 byte-range Case
def import_logdx(source_root: Path, root: Path) -> list[CaseManifest]:
    cases_root = source_root / "cases"
    imported: list[CaseManifest] = []
    for case_json_path in sorted(cases_root.rglob("case.json")):
        upstream_dir = case_json_path.parent
        case_data = json.loads(case_json_path.read_text(encoding="utf-8"))
        truth = json.loads((upstream_dir / "ground_truth.json").read_text(encoding="utf-8"))
        raw = (upstream_dir / case_data["raw_log_path"]).read_bytes()
        case_id = f"logdx-{case_data['case_id']}"
        case_dir = root / "cases" / case_id
        case_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = case_dir / "stdout.log"
        stderr_path = case_dir / "stderr.log"
        stdout_path.write_bytes(b"")
        stderr_path.write_bytes(raw)
        stdout_snapshot, stderr_snapshot = _capsule_snapshots(stdout_path, stderr_path)
        capsule = FailureCapsule(
            job_id=case_id,
            attempt_id="attempt-logdx",
            argv=["github-actions", str(case_data.get("workflow_name", "workflow"))],
            cwd="/external/logdx",
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            failure_kind="process_exit",
            returncode=1,
            phase="main",
            stderr=stderr_snapshot,
            stdout=stdout_snapshot,
        )
        gold: list[GoldFact] = []
        for index, signal in enumerate(truth.get("required_signals", [])):
            evidence_lines = signal.get("evidence_lines") or []
            if not evidence_lines:
                continue
            start, end = _line_range_to_bytes(raw, evidence_lines[0][0], evidence_lines[0][1])
            gold.append(
                GoldFact(
                    fact_id=f"signal-{index}",
                    importance=(
                        "essential" if signal.get("importance") == "critical" else "supporting"
                    ),
                    stream="stderr",
                    byte_start=start,
                    byte_end=end,
                    description=str(signal.get("value") or signal.get("type") or "required signal"),
                    provenance="human_confirmed",
                    review_passes=1,
                )
            )
        category = str(case_data.get("failure_category", "unknown"))
        operator_categories = {
            "permission_or_secret",
            "dependency_install",
            "timeout_or_oom",
            "infrastructure",
        }
        manifest = CaseManifest(
            case_id=case_id,
            tier="external_generalization",
            workload="logdx-ci",
            split="external",
            template_id=f"logdx-{case_data['case_id']}",
            failure_kind="process_exit",
            phase="main",
            expected_recovery_kind=(
                "operator_action" if category in operator_categories else "patch"
            ),
            capsule=capsule,
            logs={
                "stdout": LogArtifact(
                    path=str(stdout_path.relative_to(root)),
                    sha256=sha256_file(stdout_path),
                    size=0,
                ),
                "stderr": LogArtifact(
                    path=str(stderr_path.relative_to(root)),
                    sha256=sha256_file(stderr_path),
                    size=len(raw),
                ),
            },
            source=SourceProvenance(
                repository=str(case_data.get("repo", "")) or None,
                license="CC-BY-4.0",
                upstream_case_id=str(case_data["case_id"]),
                artifact_sha256=LOGDX_ARCHIVE_SHA256,
            ),
            gold_facts=gold,
            expected_diagnosis_terms=[
                str(value)
                for value in truth.get("expected_diagnosis", {}).get("must_mention", [])
            ],
            annotation_notes=(
                "Imported from LogDx-CI v1.2; upstream ground truth is AI-drafted and "
                "single-author verified."
            ),
        )
        _write_json(case_dir / "manifest.json", manifest.model_dump(mode="json"))
        imported.append(manifest)
    if len(imported) != 35:
        raise ValueError(f"expected 35 LogDx-CI cases, found {len(imported)}")
    return imported


# 从一个真实背景日志构建指定大小和证据位置的压力 Case
def _write_stress_case(
    root: Path,
    source_path: Path,
    source_name: str,
    size_bytes: int,
    position: str,
) -> CaseManifest:
    case_id = f"stress-{source_name}-{size_bytes}-{position}"
    case_dir = root / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    output = case_dir / "stderr.log"
    stdout = case_dir / "stdout.log"
    stdout.write_bytes(b"")
    source = source_path.read_bytes()
    if not source:
        raise ValueError(f"stress source is empty: {source_path}")
    marker = b"CYAN_STRESS_EVIDENCE root-cause-anchor\n"
    marker_at = {
        "front": size_bytes // 20,
        "middle": size_bytes // 2,
        "tail": size_bytes - size_bytes // 20,
    }[position]
    with output.open("wb") as handle:
        written = 0
        inserted = False
        while written < size_bytes:
            if not inserted and written >= marker_at:
                chunk = marker[: max(0, size_bytes - written)]
                handle.write(chunk)
                written += len(chunk)
                inserted = True
                continue
            limit = size_bytes - written
            if not inserted:
                limit = min(limit, marker_at - written)
            chunk = source[: min(len(source), limit)]
            if not chunk:
                continue
            handle.write(chunk)
            written += len(chunk)
    raw = output.read_bytes()
    start = raw.index(marker.rstrip(b"\n"))
    end = min(len(raw), start + len(marker))
    stdout_snapshot, stderr_snapshot = _capsule_snapshots(stdout, output)
    capsule = FailureCapsule(
        job_id=case_id,
        attempt_id="attempt-stress",
        argv=["stress-workload"],
        cwd="/benchmark/stress",
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        failure_kind="process_exit",
        returncode=2,
        phase="main",
        stdout=stdout_snapshot,
        stderr=stderr_snapshot,
    )
    manifest = CaseManifest(
        case_id=case_id,
        tier="scale_stress",
        workload=source_name,
        split="stress",
        template_id=f"{source_name}-{size_bytes}-{position}",
        failure_kind="process_exit",
        phase="main",
        expected_recovery_kind="none",
        capsule=capsule,
        logs={
            "stdout": LogArtifact(
                path=str(stdout.relative_to(root)), sha256=sha256_file(stdout), size=0
            ),
            "stderr": LogArtifact(
                path=str(output.relative_to(root)),
                sha256=sha256_file(output),
                size=output.stat().st_size,
            ),
        },
        source=SourceProvenance(
            repository="logpai/loghub",
            license="research-use",
            upstream_case_id=source_path.name,
            artifact_sha256=sha256_file(source_path),
        ),
        gold_facts=[
            GoldFact(
                fact_id=f"{case_id}-anchor",
                importance="essential",
                stream="stderr",
                byte_start=start,
                byte_end=end,
                description="Known retrieval anchor embedded in real LogHub background noise.",
                provenance="injected",
                review_passes=1,
            )
        ],
        stress_position=position,  # type: ignore[arg-type]
        annotation_notes="Scale-only stress case; it does not measure Data/ML diagnosis quality.",
    )
    _write_json(case_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


# 使用 Hadoop、Spark、BGL 三个显式来源生成九个规模压力 Case
def prepare_stress(
    root: Path,
    sources: dict[str, Path],
    *,
    sizes: tuple[int, int, int] = (5 * 1024**2, 50 * 1024**2, 500 * 1024**2),
) -> list[CaseManifest]:
    required = {"hadoop", "spark", "bgl"}
    if set(sources) != required:
        raise ValueError(f"stress sources must be exactly {sorted(required)}")
    positions = ("front", "middle", "tail")
    return [
        _write_stress_case(root, sources[name], name, size, position)
        for name, size, position in zip(sorted(required), sizes, positions, strict=True)
        for position in positions
    ][:9]


# 在临时目录获取 LogDx 并导入指定语料根目录
def fetch_and_import_logdx(root: Path, cache_dir: Path | None = None) -> list[CaseManifest]:
    if cache_dir is not None:
        return import_logdx(fetch_logdx(cache_dir), root)
    with tempfile.TemporaryDirectory(prefix="cyan-logdx-") as directory:
        return import_logdx(fetch_logdx(Path(directory)), root)
