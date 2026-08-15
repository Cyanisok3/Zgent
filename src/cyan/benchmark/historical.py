from __future__ import annotations

import configparser
import json
import shutil
import stat
import subprocess
import zipfile
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from cyan.benchmark.corpus import (
    _capsule_snapshots,
    _combined_log_sha256,
    _write_json,
    sha256_file,
)
from cyan.benchmark.models import (
    CaseManifest,
    GoldFact,
    LogArtifact,
    ReplayReceipt,
    SourceProvenance,
)
from cyan.core.incidents.models import FailureCapsule

DEFECTS4ML_SHA256 = "ff491cdd7a7d4dc428eede086a21402a93f053caf6bb9c31c75d4ac44e96ef68"
DEFECTS4ML_RECORD = "https://zenodo.org/records/8376824"


@dataclass(frozen=True, slots=True)
class HistoricalCandidate:
    case_id: str
    image: str
    timeout_s: int = 180


@dataclass(frozen=True, slots=True)
class ReplayOutcome:
    candidate: HistoricalCandidate
    buggy: subprocess.CompletedProcess[bytes]
    fixed: subprocess.CompletedProcess[bytes]
    qualified: bool
    reason: str


DEFECTS4ML_CANDIDATES = (
    HistoricalCandidate("076", "cyan/defects4ml-tf20:v1"),
    HistoricalCandidate("094", "cyan/defects4ml-tf113:v1"),
    HistoricalCandidate("093", "cyan/defects4ml-tf21:v1"),
    HistoricalCandidate("082", "cyan/defects4ml-tf113:v1"),
    HistoricalCandidate("087", "cyan/defects4ml-tf21:v1", timeout_s=600),
    HistoricalCandidate("079", "cyan/defects4ml-tf113:v1"),
    HistoricalCandidate("080", "cyan/defects4ml-tf20:v1", timeout_s=600),
    HistoricalCandidate("081", "cyan/defects4ml-tf20:v1", timeout_s=600),
    HistoricalCandidate("085", "cyan/defects4ml-tf112:v1", timeout_s=600),
    HistoricalCandidate("090", "cyan/defects4ml-tf21:v1", timeout_s=600),
    HistoricalCandidate("097", "cyan/defects4ml-tf21:v1", timeout_s=600),
    HistoricalCandidate("054", "cyan/defects4ml-tf20:v1", timeout_s=600),
    HistoricalCandidate("096", "cyan/defects4ml-tf20:v1"),
    HistoricalCandidate("071", "cyan/defects4ml-tf20:v1"),
    HistoricalCandidate("053", "cyan/defects4ml-tf20:v1", timeout_s=600),
)


# 只释放候选 Case，拒绝 Zip Slip 和归档符号链接
def _extract_candidates(archive: Path, destination: Path) -> None:
    destination = destination.resolve()
    wanted = {candidate.case_id for candidate in DEFECTS4ML_CANDIDATES}
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            path = PurePosixPath(member.filename)
            if len(path.parts) < 3 or path.parts[0] != "bugs" or path.parts[1] not in wanted:
                continue
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"archive member escapes destination: {member.filename}")
            mode = member.external_attr >> 16
            if stat.S_ISLNK(mode):
                raise ValueError(f"archive links are not accepted: {member.filename}")
            target = (destination / Path(*path.parts)).resolve()
            if not target.is_relative_to(destination):
                raise ValueError(f"archive member escapes destination: {member.filename}")
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(member) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)


# 返回本地镜像 ID，使运行凭证不只依赖可变 tag
def _image_identity(image: str) -> str:
    result = subprocess.run(
        ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"missing replay image {image}: {result.stderr.strip()}")
    return f"{image}#{result.stdout.strip()}"


# 构造固定容器、只读源码和断网数据缓存的重放命令
def _variant_argv(
    candidate: HistoricalCandidate,
    case_root: Path,
    cache_root: Path,
    variant: str,
) -> list[str]:
    cache_root.mkdir(parents=True, exist_ok=True)
    return [
        "docker",
        "run",
        "--rm",
        "--platform",
        "linux/amd64",
        "--network",
        "none",
        "-e",
        "OMP_NUM_THREADS=1",
        "-e",
        "PYTHONHASHSEED=0",
        "-v",
        f"{case_root.resolve()}:/case:ro",
        "-v",
        f"{cache_root.resolve()}:/root/.keras:rw",
        "-w",
        "/case",
        candidate.image,
        "timeout",
        str(candidate.timeout_s),
        "python",
        f"{variant}/script.py",
    ]


# 在固定 x86 镜像中执行一个原始 buggy 或 fixed 入口
def _run_variant(
    candidate: HistoricalCandidate,
    case_root: Path,
    cache_root: Path,
    variant: str,
) -> subprocess.CompletedProcess[bytes]:
    argv = _variant_argv(candidate, case_root, cache_root, variant)
    try:
        return subprocess.run(
            argv,
            check=False,
            capture_output=True,
            timeout=candidate.timeout_s + 30,
        )
    except subprocess.TimeoutExpired as error:
        return subprocess.CompletedProcess(
            argv,
            124,
            stdout=error.stdout or b"",
            stderr=(error.stderr or b"") + b"\ncyan host timeout\n",
        )


# 从上一轮合格日志恢复结果，避免重复执行昂贵训练
def _resume_qualified(
    candidate: HistoricalCandidate,
    source_root: Path,
    cache_root: Path,
    replay_root: Path,
    previous: dict[str, object],
) -> ReplayOutcome | None:
    if previous.get("qualified") is not True:
        return None
    if previous.get("image") != _image_identity(candidate.image):
        return None
    case_root = source_root / "bugs" / candidate.case_id
    log_root = replay_root / candidate.case_id
    required = [
        log_root / "buggy.stdout.log",
        log_root / "buggy.stderr.log",
        log_root / "fixed.stdout.log",
        log_root / "fixed.stderr.log",
    ]
    if not all(path.is_file() for path in required):
        return None
    buggy_returncode = previous.get("buggy_returncode")
    fixed_returncode = previous.get("fixed_returncode")
    if not isinstance(buggy_returncode, int) or not isinstance(fixed_returncode, int):
        return None
    buggy = subprocess.CompletedProcess(
        _variant_argv(candidate, case_root, cache_root, "buggy"),
        buggy_returncode,
        required[0].read_bytes(),
        required[1].read_bytes(),
    )
    fixed = subprocess.CompletedProcess(
        _variant_argv(candidate, case_root, cache_root, "fixed"),
        fixed_returncode,
        required[2].read_bytes(),
        required[3].read_bytes(),
    )
    return ReplayOutcome(candidate, buggy, fixed, True, "qualified (resumed)")


# 判定候选是否满足真实失败和同入口修复成功
def _replay_candidate(
    candidate: HistoricalCandidate,
    source_root: Path,
    cache_root: Path,
) -> ReplayOutcome:
    case_root = source_root / "bugs" / candidate.case_id
    if not (case_root / "buggy" / "script.py").is_file():
        empty = subprocess.CompletedProcess([], 127, b"", b"missing buggy/script.py")
        return ReplayOutcome(candidate, empty, empty, False, "missing entry point")
    if not (case_root / "fixed" / "script.py").is_file():
        empty = subprocess.CompletedProcess([], 127, b"", b"missing fixed/script.py")
        return ReplayOutcome(candidate, empty, empty, False, "missing entry point")
    buggy = _run_variant(candidate, case_root, cache_root, "buggy")
    fixed = _run_variant(candidate, case_root, cache_root, "fixed")
    if buggy.returncode in {0, 124, 137}:
        return ReplayOutcome(candidate, buggy, fixed, False, "buggy did not fail diagnostically")
    if fixed.returncode != 0:
        return ReplayOutcome(candidate, buggy, fixed, False, "fixed replay did not succeed")
    return ReplayOutcome(candidate, buggy, fixed, True, "qualified")


# 从最后一个非空错误行生成待人工复核的稳定 byte 区间
def _candidate_gold(stderr: bytes, stdout: bytes, case_id: str) -> tuple[str, GoldFact]:
    stream = "stderr" if stderr.strip() else "stdout"
    raw = stderr if stream == "stderr" else stdout
    lines = raw.splitlines(keepends=True)
    last = next((line for line in reversed(lines) if line.strip()), raw)
    start = raw.rfind(last)
    end = start + len(last)
    description = last.decode("utf-8", errors="replace").strip()[:2000]
    return stream, GoldFact(
        fact_id=f"{case_id}-cause",
        importance="essential",
        stream=stream,  # type: ignore[arg-type]
        byte_start=start,
        byte_end=end,
        description=description,
        provenance="automatic_candidate",
    )


# 将一个合格重放封装为可导入的历史 Case bundle
def _write_bundle(
    outcome: ReplayOutcome,
    source_root: Path,
    bundle_root: Path,
    archive_sha256: str,
    split: str,
) -> CaseManifest:
    upstream_id = outcome.candidate.case_id
    case_id = f"core-history-defects4ml-{upstream_id}"
    case_dir = bundle_root / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "stdout": case_dir / "stdout.log",
        "stderr": case_dir / "stderr.log",
        "fixed.stdout": case_dir / "fixed.stdout.log",
        "fixed.stderr": case_dir / "fixed.stderr.log",
    }
    paths["stdout"].write_bytes(outcome.buggy.stdout)
    paths["stderr"].write_bytes(outcome.buggy.stderr)
    paths["fixed.stdout"].write_bytes(outcome.fixed.stdout)
    paths["fixed.stderr"].write_bytes(outcome.fixed.stderr)
    config = configparser.ConfigParser()
    config.read(source_root / "bugs" / upstream_id / "conf.ini", encoding="utf-8")
    metadata = config["DEFAULT"]
    stdout_snapshot, stderr_snapshot = _capsule_snapshots(paths["stdout"], paths["stderr"])
    _, gold = _candidate_gold(outcome.buggy.stderr, outcome.buggy.stdout, case_id)
    failing_argv = list(outcome.buggy.args)
    fixed_argv = list(outcome.fixed.args)
    logs = {
        "stdout": LogArtifact(
            path=str(paths["stdout"].relative_to(bundle_root)),
            sha256=sha256_file(paths["stdout"]),
            size=paths["stdout"].stat().st_size,
        ),
        "stderr": LogArtifact(
            path=str(paths["stderr"].relative_to(bundle_root)),
            sha256=sha256_file(paths["stderr"]),
            size=paths["stderr"].stat().st_size,
        ),
    }
    fixed_logs = {
        "stdout": LogArtifact(
            path=str(paths["fixed.stdout"].relative_to(bundle_root)),
            sha256=sha256_file(paths["fixed.stdout"]),
            size=paths["fixed.stdout"].stat().st_size,
        ),
        "stderr": LogArtifact(
            path=str(paths["fixed.stderr"].relative_to(bundle_root)),
            sha256=sha256_file(paths["fixed.stderr"]),
            size=paths["fixed.stderr"].stat().st_size,
        ),
    }
    root_cause = metadata.get("root_cause", "historical ML failure")
    manifest = CaseManifest(
        case_id=case_id,
        tier="cyan_core",
        workload=f"historical_{metadata.get('framework', 'ml')}",
        split=split,  # type: ignore[arg-type]
        template_id=f"defects4ml-{upstream_id}",
        failure_kind="process_exit",
        phase="main",
        expected_recovery_kind="patch",
        capsule=FailureCapsule(
            job_id=case_id,
            attempt_id=f"attempt-{upstream_id}",
            argv=failing_argv,
            cwd="/case",
            occurred_at=datetime(2023, 9, 25, tzinfo=UTC),
            failure_kind="process_exit",
            returncode=outcome.buggy.returncode,
            phase="main",
            stdout=stdout_snapshot,
            stderr=stderr_snapshot,
        ),
        logs=logs,  # type: ignore[arg-type]
        fixed_logs=fixed_logs,  # type: ignore[arg-type]
        source=SourceProvenance(
            repository=DEFECTS4ML_RECORD,
            issue_url=metadata.get("url"),
            failing_revision=f"defects4ml-v1:{upstream_id}:buggy",
            fixing_revision=f"defects4ml-v1:{upstream_id}:fixed",
            license="CC-BY-4.0",
            upstream_case_id=upstream_id,
            artifact_sha256=archive_sha256,
            runtime_image=_image_identity(outcome.candidate.image),
            historical=True,
        ),
        replay=ReplayReceipt(
            failing_argv=failing_argv,
            fixed_argv=fixed_argv,
            failing_returncode=outcome.buggy.returncode,
            fixed_returncode=outcome.fixed.returncode,
            failing_log_sha256=_combined_log_sha256(
                outcome.buggy.stdout, outcome.buggy.stderr
            ),
            fixed_log_sha256=_combined_log_sha256(
                outcome.fixed.stdout, outcome.fixed.stderr
            ),
        ),
        gold_facts=[gold],
        expected_diagnosis_terms=[root_cause.lower()],
        root_cause_rubric=[root_cause, metadata.get("description", root_cause)],
        annotation_notes=(
            "Automatically pre-annotated from a real Defects4ML replay; "
            "test Gold remains draft."
        ),
    )
    _write_json(case_dir / "manifest.json", manifest.model_dump(mode="json"))
    return manifest


# 保存每个候选的原始输出，便于审计淘汰原因
def _persist_outcome(outcome: ReplayOutcome, replay_root: Path) -> dict[str, object]:
    case_root = replay_root / outcome.candidate.case_id
    case_root.mkdir(parents=True, exist_ok=True)
    files = {
        "buggy.stdout": outcome.buggy.stdout,
        "buggy.stderr": outcome.buggy.stderr,
        "fixed.stdout": outcome.fixed.stdout,
        "fixed.stderr": outcome.fixed.stderr,
    }
    hashes: dict[str, str] = {}
    for name, content in files.items():
        path = case_root / f"{name}.log"
        path.write_bytes(content)
        hashes[name] = sha256_file(path)
    return {
        "upstream_case_id": outcome.candidate.case_id,
        "image": _image_identity(outcome.candidate.image),
        "buggy_returncode": outcome.buggy.returncode,
        "fixed_returncode": outcome.fixed.returncode,
        "qualified": outcome.qualified,
        "reason": outcome.reason,
        "log_sha256": hashes,
        "buggy_error_tail": outcome.buggy.stderr.decode(
            "utf-8", errors="replace"
        )[-1000:],
        "fixed_error_tail": outcome.fixed.stderr.decode(
            "utf-8", errors="replace"
        )[-1000:],
    }


# 重放候选并仅在凑满十二个时发布完整历史 bundle
def prepare_defects4ml_history(
    archive: Path,
    output_root: Path,
    work_root: Path,
    *,
    required: int = 12,
) -> dict[str, object]:
    archive = archive.expanduser().resolve()
    output_root = output_root.expanduser().resolve()
    work_root = work_root.expanduser().resolve()
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != DEFECTS4ML_SHA256:
        raise ValueError(f"Defects4ML archive SHA-256 mismatch: {actual_sha256}")
    source_root = work_root / "sources"
    cache_root = work_root / "keras-cache"
    replay_root = work_root / "replays"
    _extract_candidates(archive, source_root)
    previous_path = work_root / "replay-audit.json"
    previous_by_id: dict[str, dict[str, object]] = {}
    if previous_path.is_file():
        previous_payload = json.loads(previous_path.read_text(encoding="utf-8"))
        previous_by_id = {
            item["upstream_case_id"]: item for item in previous_payload["candidates"]
        }
    outcomes = []
    for candidate in DEFECTS4ML_CANDIDATES:
        resumed = _resume_qualified(
            candidate,
            source_root,
            cache_root,
            replay_root,
            previous_by_id.get(candidate.case_id, {}),
        )
        outcomes.append(
            resumed or _replay_candidate(candidate, source_root, cache_root)
        )
    qualified = [outcome for outcome in outcomes if outcome.qualified]
    candidate_audits = [
        _persist_outcome(outcome, replay_root) for outcome in outcomes
    ]
    audit = {
        "schema_version": 1,
        "archive_sha256": actual_sha256,
        "required": required,
        "qualified": len(qualified),
        "published": len(qualified) >= required,
        "candidates": candidate_audits,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "replay-audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if len(qualified) < required:
        return audit
    for index, outcome in enumerate(qualified[:required]):
        split = "dev" if index < 4 else "test"
        _write_bundle(outcome, source_root, output_root, actual_sha256, split)
    return audit
