from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from cyan_bench.admission import admit_case
from cyan_bench.baselines import select_baseline
from cyan_bench.cases import case_fingerprint, discover_cases, load_case, resolve_anchors
from cyan_bench.cli import _incident_complete, _parser
from cyan_bench.diagnosis import DIAGNOSIS_PROMPT_VERSION
from cyan_bench.execution import discard_workspace, prepare_workspace
from cyan_bench.incident_track import (
    _diagnosis_summary,
    _wait_verification,
    run_incident_track,
)
from cyan_bench.models import IncidentBenchmarkArtifact, ProcessCapture
from cyan_bench.paths import BenchmarkPaths, benchmark_paths
from cyan_bench.reporting import build_report
from cyan_bench.scoring import score_selection


class _AwaitingApprovalCoordinator:
    # 返回重跑失败后产生的新审批视图
    async def job_view(self, job_id: str) -> dict[str, object]:
        del job_id
        return {"incident": {"status": "awaiting_approval"}}


# 创建测试使用的最小 Python 训练仓库与双向补丁
def _fixture_case(tmp_path: Path) -> tuple[Path, BenchmarkPaths]:
    case_dir = tmp_path / "cases" / "fixture-fault"
    case_dir.mkdir(parents=True)
    repo_url = "https://github.com/example/trainer"
    cache = tmp_path / "cache"
    staging = cache / "repos" / "staging"
    staging.mkdir(parents=True)
    (staging / "train.py").write_text(
        'print("step=1")\nitems = [1]\nprint(items[0])\n', encoding="utf-8"
    )
    subprocess.run(["git", "init", "-q"], cwd=staging, check=True)
    subprocess.run(["git", "config", "user.name", "cyan-bench"], cwd=staging, check=True)
    subprocess.run(
        ["git", "config", "user.email", "cyan-bench@invalid"], cwd=staging, check=True
    )
    subprocess.run(["git", "add", "train.py"], cwd=staging, check=True)
    subprocess.run(["git", "commit", "-q", "-m", "fixture"], cwd=staging, check=True)
    repo_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=staging,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    identity = f"{repo_url}\0{repo_sha}".encode()
    cache_repo = cache / "repos" / hashlib.sha256(identity).hexdigest()[:20]
    staging.rename(cache_repo)
    (case_dir / "case.toml").write_text(
        "\n".join(
            [
                "schema_version = 1",
                'id = "fixture-fault"',
                'title = "Fixture fault"',
                'split = "dev"',
                'framework = "fixture"',
                'training_domain = "general_ml"',
                'fault_family = "data"',
                'mechanism_id = "fixture-failure"',
                'failure_stage = "mid_run"',
                'origin_kind = "provenance_preserving_port"',
                "patchable = true",
                f'repo_url = "{repo_url}"',
                f'repo_sha = "{repo_sha}"',
                'issue_url = "https://github.com/example/trainer/issues/1"',
                'fix_url = "https://github.com/example/trainer/pull/2"',
                'license = "MIT"',
                'env_id = "fixture-env"',
                'argv = ["python", "train.py"]',
                'cwd = "."',
                "timeout_s = 10",
                'hardware = "cpu"',
                'milestone_anchor = "step=1"',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (case_dir / "expected.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "diagnosis": {
                    "category": "data",
                    "culprit": ["items"],
                    "causal_mechanism": ["out of range"],
                },
                "required_groups": [
                    [{"source": "stderr", "literal": "IndexError: list index out of range"}]
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    fault = """diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
 print("step=1")
 items = [1]
-print(items[0])
+print(items[2])
"""
    fix = """diff --git a/train.py b/train.py
--- a/train.py
+++ b/train.py
@@ -1,3 +1,3 @@
 print("step=1")
 items = [1]
-print(items[2])
+print(items[0])
"""
    (case_dir / "fault.patch").write_text(fault, encoding="utf-8")
    (case_dir / "fix.patch").write_text(fix, encoding="utf-8")
    env_python = tmp_path / "envs" / "fixture-env" / ".venv" / "bin" / "python"
    env_python.parent.mkdir(parents=True)
    os.symlink(sys.executable, env_python)
    (tmp_path / "envs" / "fixture-env" / "uv.lock").write_text(
        "fixture-lock\n", encoding="utf-8"
    )
    paths = BenchmarkPaths(
        root=tmp_path,
        cases=tmp_path / "cases",
        environments=tmp_path / "envs",
        cache=cache,
        artifacts=tmp_path / "artifacts",
    )
    return case_dir, paths


# 功能：验证真实临时 Git 与真实子进程能够区分 control、buggy 和 fixed
# 设计：用一行来源明确的反向补丁制造运行期失败，并检查动态 gold range
def test_admission_uses_real_git_and_processes(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    case = load_case(case_dir)

    artifact = admit_case(case, paths, repeats=1)

    assert artifact.admitted is True
    buggy = ProcessCapture.model_validate_json(
        (paths.artifacts / "captures/fixture-fault/buggy/1/process.json").read_text()
    )
    assert buggy.returncode != 0
    assert (paths.artifacts / "captures/fixture-fault/buggy/1/gold-ranges.json").is_file()
    assert not any((paths.artifacts / "workspaces").iterdir())


# 功能：验证 Incident artifact 只保存有限诊断摘要和证据引用
# 设计：传入多余文本与错误引用类型，确认摘要不携带工具输出或未定义字段
def test_incident_diagnosis_summary_is_bounded() -> None:
    summary = _diagnosis_summary(
        {
            "category": "dtype",
            "root_cause": "direct cause",
            "causal_support": "direct",
            "patch_recommended": True,
            "evidence": [
                {"source": "stderr", "reference": "bytes:1-2", "description": "large"},
                {"source": "stderr", "reference": 3},
            ],
        }
    )

    assert summary[:4] == ("dtype", "direct cause", "direct", True)
    assert summary[4] == [{"source": "stderr", "reference": "bytes:1-2"}]
    assert DIAGNOSIS_PROMPT_VERSION == "causal-support-abstention-v1"


# 功能：验证 Tail、BM25 与 Cyan Selector 均产生可回溯且有界的 evidence
# 设计：复用真实失败日志并用 required-group overlap 统一评分
def test_baselines_share_dynamic_evidence_scoring(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    case = load_case(case_dir)
    admit_case(case, paths, repeats=1)
    run_dir = paths.artifacts / "captures/fixture-fault/buggy/1"
    capture = ProcessCapture.model_validate_json((run_dir / "process.json").read_text())

    for baseline in ("tail_32", "bm25_32", "cyan_selector_32"):
        selection = select_baseline(
            case,
            capture,
            run_dir,
            baseline,
            paths.artifacts / "selections" / baseline,
        )
        score = score_selection(selection, run_dir / "gold-ranges.json")
        assert selection.selected_bytes <= 32 * 1024
        assert score["gold_evidence_hit"] is True


# 功能：验证 gold anchors 每次从新日志解析而不是绑定一次运行的 offset
# 设计：在错误前加入动态长度前缀并确认解析后的 byte range 随之移动
def test_gold_anchors_resolve_against_each_log(tmp_path: Path) -> None:
    case_dir, _paths = _fixture_case(tmp_path)
    case = load_case(case_dir)
    stdout = tmp_path / "stdout.log"
    stderr = tmp_path / "stderr.log"
    stdout.write_text("", encoding="utf-8")
    stderr.write_text("pid=123\nIndexError: list index out of range\n", encoding="utf-8")
    first = resolve_anchors(case.expected, stdout, stderr)
    stderr.write_text(
        "pid=123 timestamp=dynamic-value\nIndexError: list index out of range\n",
        encoding="utf-8",
    )

    second = resolve_anchors(case.expected, stdout, stderr)

    assert second[0].start > first[0].start


# 功能：验证无故障 Control 走真实 JobSupervisor 时不会创建 Incident
# 设计：使用干净单提交工作区和真实子进程，不注入 LLM provider
async def test_control_product_path_has_no_spurious_incident(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    case = load_case(case_dir)
    workspace = prepare_workspace(case, paths, "control")
    try:
        artifact = await run_incident_track(
            case,
            paths,
            workspace,
            1,
            paths.artifacts / "incident-control",
            is_control=True,
        )
    finally:
        discard_workspace(workspace, paths)

    assert artifact.final_job_status == "succeeded"
    assert artifact.incident_id is None
    assert artifact.spurious_incident is False
    assert artifact.input_tokens == 0


# 功能：验证 annotated tag 对象 SHA 能解析到工作树 commit
# 设计：把真实临时仓库改用 tag object 作为 manifest SHA，再构造 control 工作区
def test_workspace_accepts_annotated_tag_object_sha(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    original = load_case(case_dir)
    original_cache = paths.cache / "repos" / hashlib.sha256(
        f"{original.manifest.repo_url}\0{original.manifest.repo_sha}".encode()
    ).hexdigest()[:20]
    subprocess.run(
        ["git", "tag", "-a", "fixture-v1", "-m", "fixture tag"],
        cwd=original_cache,
        check=True,
    )
    tag_sha = subprocess.run(
        ["git", "rev-parse", "fixture-v1"],
        cwd=original_cache,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tagged_cache = paths.cache / "repos" / hashlib.sha256(
        f"{original.manifest.repo_url}\0{tag_sha}".encode()
    ).hexdigest()[:20]
    original_cache.rename(tagged_cache)
    manifest_path = case_dir / "case.toml"
    manifest_path.write_text(
        manifest_path.read_text(encoding="utf-8").replace(
            f'repo_sha = "{original.manifest.repo_sha}"', f'repo_sha = "{tag_sha}"'
        ),
        encoding="utf-8",
    )

    case = load_case(case_dir)
    workspace = prepare_workspace(case, paths, "control")
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=workspace,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        assert len(head) == 40
    finally:
        discard_workspace(workspace, paths)


# 功能：验证提交的 15 个 manifest 静态满足 split、阶段与真实性配额
# 设计：只读取版本化案例定义，不依赖本机缓存、训练日志或付费模型 artifact
def test_versioned_dataset_meets_static_quotas() -> None:
    cases = discover_cases(benchmark_paths().cases)

    assert len(cases) == 15
    assert Counter(case.manifest.split for case in cases) == Counter({"dev": 6, "test": 9})
    stages = Counter(case.manifest.failure_stage for case in cases)
    assert stages["startup"] <= 4
    assert stages["mid_run"] >= 6
    assert stages["finalization"] >= 4
    assert sum(case.manifest.training_domain == "llm" for case in cases) >= 10
    assert sum(not case.manifest.patchable for case in cases) >= 3
    mechanisms = Counter(case.manifest.mechanism_id for case in cases)
    assert max(mechanisms.values()) <= 2
    roles = Counter(
        case.manifest.control_role
        for case in cases
        if case.manifest.control_role is not None
    )
    assert roles == Counter({"short_quiet": 1, "long_clean": 1, "warning_heavy": 1})


# 功能：验证 5 MiB 日志上的 BM25 与 Cyan Selector 输出有界且确定
# 设计：合成数据仅测试 harness，不进入真实案例或主榜评分
def test_large_log_selectors_are_deterministic_and_bounded(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    case = load_case(case_dir)
    run_dir = paths.artifacts / "captures/fixture-fault/buggy/1"
    run_dir.mkdir(parents=True)
    stdout = (b"training step completed loss=1.0\n" * 180_000)[: 5 * 1024 * 1024]
    stderr = (
        b"Traceback (most recent call last):\n"
        b'  File "train.py", line 3, in <module>\n'
        b"IndexError: list index out of range\n"
    )
    (run_dir / "stdout.log").write_bytes(stdout)
    (run_dir / "stderr.log").write_bytes(stderr)
    capture = ProcessCapture(
        case_id=case.manifest.id,
        case_fingerprint=case_fingerprint(case),
        environment_lock_sha256="0" * 64,
        variant="buggy",
        repeat=1,
        argv=[sys.executable, "train.py"],
        cwd=str(tmp_path),
        returncode=1,
        timed_out=False,
        duration_seconds=1.0,
        stdout_path="stdout.log",
        stderr_path="stderr.log",
        stdout_bytes=len(stdout),
        stderr_bytes=len(stderr),
        stdout_sha256=hashlib.sha256(stdout).hexdigest(),
        stderr_sha256=hashlib.sha256(stderr).hexdigest(),
        created_at=datetime.now(UTC),
    )

    for baseline in ("bm25_32", "cyan_selector_32"):
        first = select_baseline(
            case, capture, run_dir, baseline, paths.artifacts / "first" / baseline
        )
        second = select_baseline(
            case, capture, run_dir, baseline, paths.artifacts / "second" / baseline
        )
        assert first.selected_bytes <= 32 * 1024
        assert first.references == second.references
        assert (paths.artifacts / "first" / baseline / "selection.txt").read_bytes() == (
            paths.artifacts / "second" / baseline / "selection.txt"
        ).read_bytes()


# 功能：验证报告保留阶段、框架、故障族及 Control 原始分母
# 设计：用真实临时 Git/子进程生成一个 test selection 和一个正常产品观察
async def test_report_keeps_grouping_and_control_denominator(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    manifest = case_dir / "case.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('split = "dev"', 'split = "test"'),
        encoding="utf-8",
    )
    case = load_case(case_dir)
    admit_case(case, paths, repeats=1)
    run_dir = paths.artifacts / "captures/fixture-fault/buggy/1"
    capture = ProcessCapture.model_validate_json((run_dir / "process.json").read_text())
    selection_dir = (
        paths.artifacts
        / "run-sets/report-test/diagnosis/fixture-fault/buggy/1/tail_32"
    )
    select_baseline(case, capture, run_dir, "tail_32", selection_dir)
    workspace = prepare_workspace(case, paths, "control")
    try:
        await run_incident_track(
            case,
            paths,
            workspace,
            1,
            paths.artifacts / "run-sets/report-test/incident/fixture-fault/control/1",
            is_control=True,
        )
    finally:
        discard_workspace(workspace, paths)

    report = build_report("report-test", paths)

    by_stage = report["selection_test_by_failure_stage"]
    by_framework = report["selection_test_by_framework"]
    by_family = report["selection_test_by_fault_family"]
    controls = report["product_controls"]
    assert isinstance(by_stage, dict)
    assert isinstance(by_framework, dict)
    assert isinstance(by_family, dict)
    assert isinstance(controls, dict)
    assert "mid_run" in by_stage
    assert "fixture" in by_framework
    assert "data" in by_family
    assert controls["observations"] == 1
    assert controls["spurious_incidents"] == 0


# 功能：验证基础设施失败不会进入 Incident 能力指标或 token 用量
# 设计：写入一条有效结果和一条显式错误结果，检查报告仅计有效记录
def test_report_excludes_incident_infrastructure_errors(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    manifest = case_dir / "case.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('split = "dev"', 'split = "test"'),
        encoding="utf-8",
    )
    output_root = paths.artifacts / "run-sets/report-errors/incident/fixture-fault/buggy"
    for repeat, error in ((1, None), (2, "llm_error")):
        output = output_root / str(repeat) / "incident-benchmark.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            IncidentBenchmarkArtifact(
                case_id="fixture-fault",
                repeat=repeat,
                is_control=False,
                job_id=f"job-{repeat}",
                final_job_status="failed",
                spurious_incident=False,
                diagnosis_present=error is None,
                proposal_present=False,
                proposal_valid=False,
                unsafe_proposal=False,
                resolved=False,
                capsule_tail_bytes=100,
                selector_selected_bytes=100,
                unique_evidence_bytes=100,
                peak_input_bytes=100,
                input_tokens=100,
                output_tokens=50,
                tool_calls=1,
                duration_seconds=1.0,
                error=error,
                created_at=datetime.now(UTC),
            ).model_dump_json(),
            encoding="utf-8",
        )

    report = build_report("report-errors", paths)
    completeness = report["incident_completeness"]
    usage = report["usage"]
    assert isinstance(completeness, dict)
    assert isinstance(usage, dict)
    assert completeness["test_valid"] == 1
    assert completeness["fault_infrastructure_errors"] == 1
    assert usage["incident_test"]["input_tokens"] == 100


# 功能：验证重跑产生新 Proposal 时等待逻辑立即返回
# 设计：用最小 Coordinator 视图复现 awaiting_approval，避免真实等待超时
async def test_verification_returns_for_new_approval() -> None:
    view = await _wait_verification(_AwaitingApprovalCoordinator(), "job-1", timeout_s=1)  # type: ignore[arg-type]

    assert view["incident"] == {"status": "awaiting_approval"}


# 功能：验证模型连接失败的 Incident artifact 可被断点续跑识别
# 设计：写入 error 为空但 run reason 为 llm_error 的旧格式结果
def test_incident_resume_retries_llm_error(tmp_path: Path) -> None:
    artifact_path = tmp_path / "incident-benchmark.json"
    artifact_path.write_text(
        IncidentBenchmarkArtifact(
            case_id="fixture-fault",
            repeat=1,
            is_control=False,
            job_id="job-1",
            final_job_status="failed",
            spurious_incident=False,
            diagnosis_present=False,
            proposal_present=False,
            proposal_valid=False,
            unsafe_proposal=False,
            resolved=False,
            capsule_tail_bytes=0,
            selector_selected_bytes=0,
            unique_evidence_bytes=0,
            peak_input_bytes=0,
            input_tokens=0,
            output_tokens=0,
            tool_calls=0,
            duration_seconds=1.0,
            created_at=datetime.now(UTC),
        ).model_dump_json(),
        encoding="utf-8",
    )
    run_dir = tmp_path / "cyan-jobs/job-1/incidents/inc-1/runs/run-1"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text('{"reason":"llm_error"}', encoding="utf-8")

    assert _incident_complete(artifact_path) is False


# 功能：验证正式运行可显式限制案例、baseline 并启用断点续跑
# 设计：只解析 CLI，不启动模型或训练进程
def test_cli_supports_bounded_resumable_runs() -> None:
    args = _parser().parse_args(
        [
            "run",
            "--split",
            "test",
            "--track",
            "diagnosis",
            "--run-set",
            "pilot",
            "--case",
            "hf-11102-late-token",
            "--baseline",
            "full_native",
            "--repeat",
            "1",
            "--resume",
        ]
    )

    assert args.case == ["hf-11102-late-token"]
    assert args.baseline == ["full_native"]
    assert args.repeat == [1]
    assert args.resume is True
