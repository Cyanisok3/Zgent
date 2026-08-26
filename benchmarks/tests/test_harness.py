from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.types import LlmResponse, ToolCallBlock, UsageStats

from cyan_bench.admission import admit_case
from cyan_bench.audit import audit_dataset
from cyan_bench.baselines import select_baseline
from cyan_bench.cases import case_fingerprint, discover_cases, load_case, resolve_anchors
from cyan_bench.cli import _freeze_run_set, _incident_complete, _parser
from cyan_bench.diagnosis import DIAGNOSIS_PROMPT_VERSION
from cyan_bench.execution import discard_workspace, prepare_workspace
from cyan_bench.incident_track import (
    _diagnosis_summary,
    _wait_verification,
    run_incident_track,
)
from cyan_bench.models import (
    DiagnosisAnswerV2,
    IncidentBenchmarkArtifact,
    ProcessCapture,
)
from cyan_bench.paths import BenchmarkPaths, benchmark_paths
from cyan_bench.reporting import _incident_macro, build_report, write_report
from cyan_bench.scoring import score_selection


class _AwaitingApprovalCoordinator:
    # 返回重跑失败后产生的新审批视图
    async def job_view(self, job_id: str) -> dict[str, object]:
        del job_id
        return {"incident": {"status": "awaiting_approval"}}


class _ApprovalProvider:
    # 按固定顺序读取源码、提交 direct diagnosis 并提出唯一替换
    def __init__(self) -> None:
        self.calls = 0

    # 从当前 Incident prompt 中取出已观察的证据引用
    def _references(self, system: str | None) -> tuple[str, str]:
        text = system or ""
        workspace = re.search(
            r"train\.py@sha256:[0-9a-f]{64}#L[0-9]+-L[0-9]+", text
        )
        job = re.search(r'"job_id":\s*"([^"]+)"', text)
        attempt = re.search(r'"attempt_id":\s*"([^"]+)"', text)
        stderr_range = re.search(
            r'"stderr":\s*\{.*?"included_start":\s*(\d+),.*?'
            r'"included_end":\s*(\d+)',
            text,
            re.DOTALL,
        )
        if workspace is None or job is None or attempt is None or stderr_range is None:
            raise AssertionError("Incident prompt did not expose expected evidence refs")
        stderr = (
            f"stderr:{job.group(1)}/{attempt.group(1)}@bytes:"
            f"{stderr_range.group(1)}-{stderr_range.group(2)}"
        )
        return workspace.group(0), stderr

    # 返回一次假 Provider 响应，避免测试调用外部模型
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del tool_schemas, bus, run_id, step
        self.calls += 1
        if self.calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-1",
                        name="read_file",
                        input={"path": "train.py", "start_line": 1, "line_count": 3},
                    )
                ],
                usage=UsageStats(1, 1),
            )
        workspace_ref, stderr_ref = self._references(
            f"{system or ''}\n{json.dumps(messages, ensure_ascii=False)}"
        )
        if self.calls == 2:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="diagnosis-1",
                        name="submit_diagnosis",
                        input={
                            "category": "data",
                            "summary": "The list access exceeds the available items.",
                            "root_cause": "The workspace training code indexes items[2] although only one item exists.",
                            "evidence": [
                                {
                                    "source": "workspace",
                                    "reference": workspace_ref,
                                    "description": "Observed training source range.",
                                },
                                {
                                    "source": "stderr",
                                    "reference": stderr_ref,
                                    "description": "Observed failure traceback.",
                                },
                            ],
                            "confidence": 1.0,
                            "causal_support": "direct",
                            "patch_recommended": True,
                        },
                    )
                ],
                usage=UsageStats(1, 1),
            )
        if self.calls == 3:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="proposal-1",
                        name="propose_patch",
                        input={
                            "path": "train.py",
                            "search": "print(items[2])",
                            "replace": "print(items[0])",
                            "evidence": [
                                {
                                    "source": "workspace",
                                    "reference": workspace_ref,
                                    "description": "Observed exact source block.",
                                }
                            ],
                        },
                    )
                ],
                usage=UsageStats(1, 1),
            )
        return LlmResponse(stop_reason="end_turn", text="done", usage=UsageStats(1, 1))


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
    if env_python.is_symlink() or env_python.exists():
        env_python.unlink()
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
    assert DIAGNOSIS_PROMPT_VERSION == "causal-support-abstention-v3"


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


# 功能：验证真实 Incident track 在 resolved 后保留审批前 Proposal 合法性
# 设计：使用本地假 Provider 走 read、diagnosis、proposal、审批和原命令重跑闭环
async def test_incident_track_preserves_preapproval_proposal_after_resolved(
    tmp_path: Path,
) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    case = load_case(case_dir)
    workspace = prepare_workspace(case, paths, "buggy")
    try:
        artifact = await run_incident_track(
            case,
            paths,
            workspace,
            1,
            paths.artifacts / "incident-approved",
            is_control=False,
            provider=_ApprovalProvider(),
        )
    finally:
        discard_workspace(workspace, paths)

    assert artifact.final_incident_status == "resolved"
    assert artifact.proposal_present is True
    assert artifact.proposal_valid is True
    assert artifact.resolved is True
    assert artifact.abstention_gate_violated is False


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
    cases = discover_cases(benchmark_paths().cases, dataset_version="formal-v1")

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


# 功能：验证 CLI 支持有界、可续跑的运行
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
    assert not hasattr(args, "model")

    audit_args = _parser().parse_args(
        ["audit", "--dataset", "formal-v2", "--scope", "dev"]
    )
    assert audit_args.scope == "dev"


# 把 fixture 复制为指定数据集版本与 ID 的 v2 案例（fixture 不存在时才创建）
def _fixture_case_v2(
    tmp_path: Path,
    case_id: str,
    gold_support: str | None = "direct",
    gold_patch: bool | None = True,
    split: str = "dev",
    patchable: bool = False,
) -> tuple[Path, BenchmarkPaths]:
    case_dir = tmp_path / "cases" / "fixture-fault"
    if not (case_dir / "case.toml").is_file():
        _fixture_case(tmp_path)
    paths = _paths_of(tmp_path)
    v2_dir = tmp_path / "cases" / case_id
    v2_dir.mkdir(parents=True, exist_ok=True)
    for name in ("fault.patch", "fix.patch", "case.toml", "expected.json"):
        shutil.copy2(case_dir / name, v2_dir / name)
    manifest = (v2_dir / "case.toml").read_text(encoding="utf-8")
    manifest = manifest.replace('id = "fixture-fault"', f'id = "{case_id}"')
    manifest = manifest.replace('split = "dev"', f'split = "{split}"')
    manifest = manifest.replace("patchable = true", f"patchable = {str(patchable).lower()}")
    manifest = manifest.replace(
        'schema_version = 1', 'schema_version = 1\ndataset_version = "formal-v2"'
    )
    (v2_dir / "case.toml").write_text(manifest, encoding="utf-8")
    expected = json.loads((v2_dir / "expected.json").read_text(encoding="utf-8"))
    if gold_support is not None:
        expected["causal_support"] = gold_support
    if gold_patch is not None:
        expected["patch_recommended"] = gold_patch
    (v2_dir / "expected.json").write_text(
        json.dumps(expected, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return v2_dir, paths


# 功能：验证 formal-v1 与 formal-v2 案例发现互不污染且 rejected 不入集
# 设计：同目录放置 v1 fixture 与 v2 副本及一个 REJECTED.md 目录
def test_dataset_versions_are_isolated(tmp_path: Path) -> None:
    _fixture_case_v2(tmp_path, "v2-abstain-case", split="dev", patchable=False)
    rejected = tmp_path / "cases" / "rejected-case"
    rejected.mkdir()
    (rejected / "REJECTED.md").write_text("rejected", encoding="utf-8")
    paths = _paths_of(tmp_path)

    v1 = discover_cases(paths.cases, dataset_version="formal-v1")
    v2 = discover_cases(paths.cases, dataset_version="formal-v2")

    assert [case.manifest.id for case in v1] == ["fixture-fault"]
    assert [case.manifest.id for case in v2] == ["v2-abstain-case"]
    assert not any(case.manifest.id == "rejected-case" for case in v1 + v2)


# 功能：验证 formal-v2 审计要求 Gold 同时提供 causal_support 与 patch_recommended
# 设计：v2 案例缺 Gold 字段时 audit 必须列出该案例且 ready 为 False
def test_audit_v2_requires_gold_fields(tmp_path: Path) -> None:
    _fixture_case_v2(
        tmp_path,
        "v2-missing-gold",
        gold_support=None,
        gold_patch=None,
        split="test",
        patchable=True,
    )

    result = audit_dataset(
        _paths_of(tmp_path), dataset_version="formal-v2"
    )

    assert any(
        "v2-missing-gold" in reason and "missing causal_support/patch_recommended" in reason
        for reason in result["reasons"]
    )


# 从 fixture 路径构造 BenchmarkPaths（复用 fixture 目录结构）
def _paths_of(tmp_path: Path) -> BenchmarkPaths:
    return BenchmarkPaths(
        root=tmp_path,
        cases=tmp_path / "cases",
        environments=tmp_path / "envs",
        cache=tmp_path / "cache",
        artifacts=tmp_path / "artifacts",
    )


# 功能：验证 run-set.json 配置不一致或向旧 run-set 写 v2 时拒绝续跑
# 设计：先冻结再改模型/版本，以及旧 run-set 无配置文件但已有 artifact
def test_run_set_config_mismatch_rejects_resume(tmp_path: Path) -> None:
    _fixture_case(tmp_path)
    paths = _paths_of(tmp_path)
    run_root = paths.artifacts / "run-sets" / "dev-v2"

    _freeze_run_set(
        run_root,
        "formal-v2",
        "dev",
        ("fixture-fault",),
        "deepseek-v4-flash",
    )
    _freeze_run_set(
        run_root,
        "formal-v2",
        "dev",
        ("fixture-fault",),
        "deepseek-v4-flash",
    )

    try:
        _freeze_run_set(
            run_root,
            "formal-v2",
            "dev",
            ("fixture-fault",),
            "another-model",
        )
        raise AssertionError("model mismatch was not rejected")
    except SystemExit:
        pass
    try:
        _freeze_run_set(
            run_root,
            "formal-v1",
            "dev",
            ("fixture-fault",),
            "deepseek-v4-flash",
        )
        raise AssertionError("dataset mismatch was not rejected")
    except SystemExit:
        pass

    legacy = paths.artifacts / "run-sets" / "legacy-v1"
    (legacy / "diagnosis/x/buggy/1/tail_32").mkdir(parents=True)
    (legacy / "diagnosis/x/buggy/1/tail_32/diagnosis.json").write_text(
        "{}", encoding="utf-8"
    )
    # 旧 run-set 无 run-set.json 时只允许作为历史结果读取，任何运行写入都被拒绝
    for dataset in ("formal-v1", "formal-v2"):
        try:
            _freeze_run_set(
                legacy,
                dataset,
                "dev",
                ("fixture-fault",),
                "deepseek-v4-flash",
            )
            raise AssertionError(f"legacy run-set accepted a {dataset} write")
        except SystemExit:
            pass


# 功能：验证 run-set 元数据保存实际模型、案例范围和正常 ISO 时间
# 设计：冻结同一配置两次并重新解析 JSON，确认没有额外字符串编码
def test_run_set_metadata_is_frozen(tmp_path: Path) -> None:
    _fixture_case(tmp_path)
    paths = _paths_of(tmp_path)
    run_root = paths.artifacts / "run-sets" / "metadata"

    _freeze_run_set(
        run_root,
        "formal-v1",
        "test",
        ("fixture-fault",),
        "model-from-config",
    )
    payload = json.loads((run_root / "run-set.json").read_text(encoding="utf-8"))

    assert payload["dataset_version"] == "formal-v1"
    assert payload["split"] == "test"
    assert payload["selected_case_ids"] == ["fixture-fault"]
    assert payload["requested_model"] == "model-from-config"
    assert payload["created_at"].endswith("+00:00")
    datetime.fromisoformat(payload["created_at"])


# 功能：验证缺少 Incident 运行结果时 Markdown 报告仍能输出 N/A
# 设计：使用已冻结但没有 artifact 的临时 run-set，覆盖空阶段与不完整案例
def test_incomplete_report_renders_empty_stages(tmp_path: Path) -> None:
    case_dir, _ = _fixture_case(tmp_path)
    manifest = case_dir / "case.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('split = "dev"', 'split = "test"'),
        encoding="utf-8",
    )
    paths = _paths_of(tmp_path)
    _freeze_run_set(
        paths.artifacts / "run-sets" / "partial",
        "formal-v1",
        "test",
        ("fixture-fault",),
        "model-from-config",
    )

    _json_path, markdown_path = write_report("partial", paths)
    markdown = markdown_path.read_text(encoding="utf-8")

    assert "| startup | 0 | None | None |" in markdown
    assert "| Case | Patchable | Stage | Framework |" in markdown
    assert "| fixture-fault | yes | mid_run | fixture | N/A | N/A | N/A |" in markdown


# 功能：验证 Track A 严格模型对缺失 causal_support/evidence/patch_recommended 直接失败
# 设计：三种非法响应都必须抛 ValueError，合法响应可解析
def test_track_a_strict_model_rejects_missing_fields() -> None:
    valid = {
        "verdict": "fault",
        "diagnosis": {
            "category": "data",
            "culprit": "items",
            "causal_mechanism": "out of range",
            "causal_support": "direct",
            "evidence": [{"source": "stderr", "start": 0, "end": 1}],
        },
        "patch_recommended": True,
    }
    assert isinstance(
        DiagnosisAnswerV2.model_validate_json(json.dumps(valid)), DiagnosisAnswerV2
    )
    missing_support = json.loads(json.dumps(valid))
    del missing_support["diagnosis"]["causal_support"]
    try:
        DiagnosisAnswerV2.model_validate_json(json.dumps(missing_support))
        raise AssertionError("missing causal_support accepted")
    except ValueError:
        pass
    missing_evidence = json.loads(json.dumps(valid))
    missing_evidence["diagnosis"]["evidence"] = []
    try:
        DiagnosisAnswerV2.model_validate_json(json.dumps(missing_evidence))
        raise AssertionError("missing evidence accepted")
    except ValueError:
        pass
    missing_intent = json.loads(json.dumps(valid))
    del missing_intent["patch_recommended"]
    try:
        DiagnosisAnswerV2.model_validate_json(json.dumps(missing_intent))
        raise AssertionError("missing patch_recommended accepted")
    except ValueError:
        pass


# 功能：验证 resolved 后仍保留审批前 Proposal 合法性
# 设计：同一案例写两条 resolved 结果（proposal_valid 一真一假），proposal_valid 与 resolved 独立
def test_proposal_valid_preserved_after_resolved(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    manifest = case_dir / "case.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('split = "dev"', 'split = "test"'),
        encoding="utf-8",
    )
    output_root = paths.artifacts / "run-sets/props-valid/incident/fixture-fault/buggy"
    for repeat, proposal_valid in ((1, True), (2, False)):
        output = output_root / str(repeat) / "incident-benchmark.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            IncidentBenchmarkArtifact(
                case_id="fixture-fault",
                repeat=repeat,
                is_control=False,
                job_id=f"job-{repeat}",
                final_job_status="succeeded",
                final_incident_status="resolved",
                spurious_incident=False,
                diagnosis_present=True,
                proposal_present=proposal_valid,
                proposal_valid=proposal_valid,
                unsafe_proposal=False,
                resolved=True,
                capsule_tail_bytes=100,
                selector_selected_bytes=100,
                unique_evidence_bytes=100,
                peak_input_bytes=100,
                input_tokens=100,
                output_tokens=50,
                tool_calls=1,
                duration_seconds=1.0,
                created_at=datetime.now(UTC),
            ).model_dump_json(),
            encoding="utf-8",
        )

    report = build_report("props-valid", paths)
    incident = report["incident_test"]
    assert isinstance(incident, dict)
    assert incident["resolved_rate"]["macro_mean"] == 1.0
    assert incident["proposal_valid_rate"]["macro_mean"] == 0.5
    assert incident["proposal_valid_rate"]["cases"] == 1


# 功能：验证四个 abstention 指标使用各自适用分母
# 设计：构造 patchable 与非 patchable 混合记录，检查 macro 只在对应人群内平均
def test_abstention_metrics_use_correct_denominators() -> None:
    records = [
        {
            "case_id": "patch-a",
            "patchable": True,
            "resolved": True,
            "proposal_valid": True,
            "unsafe_proposal": False,
            "correct_patch_abstention": False,
            "missed_patch_opportunity": False,
            "patchable_resolved": True,
            "abstention_gate_violated": False,
            "abstention_metrics_available": True,
        },
        {
            "case_id": "patch-b",
            "patchable": True,
            "resolved": False,
            "proposal_valid": False,
            "unsafe_proposal": False,
            "correct_patch_abstention": False,
            "missed_patch_opportunity": True,
            "patchable_resolved": False,
            "abstention_gate_violated": False,
            "abstention_metrics_available": True,
        },
        {
            "case_id": "nopatch-a",
            "patchable": False,
            "resolved": False,
            "proposal_valid": False,
            "unsafe_proposal": True,
            "correct_patch_abstention": False,
            "missed_patch_opportunity": True,
            "abstention_gate_violated": True,
            "abstention_metrics_available": True,
        },
        {
            "case_id": "nopatch-b",
            "patchable": False,
            "resolved": False,
            "proposal_valid": False,
            "unsafe_proposal": False,
            "correct_patch_abstention": True,
            "missed_patch_opportunity": True,
            "abstention_gate_violated": False,
            "abstention_metrics_available": True,
        },
    ]

    assert _incident_macro(records, "unsafe_proposal")["macro_mean"] == 0.5
    assert _incident_macro(records, "correct_patch_abstention")["macro_mean"] == 0.5
    assert _incident_macro(records, "missed_patch_opportunity")["macro_mean"] == 0.5
    assert _incident_macro(records, "patchable_resolved")["macro_mean"] == 0.5
    assert _incident_macro(records, "abstention_gate_violated")["macro_mean"] == 0.25
    assert _incident_macro(records, "resolved")["macro_mean"] == 0.25
    assert _incident_macro(records, "unsafe_proposal")["cases"] == 2
    assert _incident_macro(records, "missed_patch_opportunity")["cases"] == 2
    assert _incident_macro(records, "abstention_gate_violated")["cases"] == 4


# 功能：验证历史 formal-v1 报告仍可读取且旧 run-set 只读为 formal-v1 历史
# 设计：无 run-set.json 的 run-set 构建报告默认 formal-v1；v2 数据集写入被拒绝
def test_legacy_run_set_report_still_readable(tmp_path: Path) -> None:
    case_dir, paths = _fixture_case(tmp_path)
    manifest = case_dir / "case.toml"
    manifest.write_text(
        manifest.read_text(encoding="utf-8").replace('split = "dev"', 'split = "test"'),
        encoding="utf-8",
    )
    output = (
        paths.artifacts
        / "run-sets/legacy-v1/incident/fixture-fault/buggy/1/incident-benchmark.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
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

    report = build_report("legacy-v1", paths)

    assert report["dataset_version"] == "formal-v1"
    assert report["run_set_has_config"] is False
    assert report["incident_completeness"]["test_expected"] == 3
    try:
        _freeze_run_set(
            paths.artifacts / "run-sets" / "legacy-v1",
            "formal-v2",
            "dev",
            ("fixture-fault",),
            "deepseek-v4-flash",
        )
        raise AssertionError("legacy run-set accepted a v2 write")
    except SystemExit:
        pass
