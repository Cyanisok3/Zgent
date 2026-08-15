from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from cyan.benchmark.agent_track import (
    BenchmarkEvidenceReader,
    ReadBenchmarkLogParams,
    SubmitBenchmarkDiagnosisTool,
    run_agent_strategy,
)
from cyan.benchmark.cli import _agent_run_async, build_parser
from cyan.benchmark.corpus import (
    Corpus,
    import_logdx,
    prepare_ci_core,
    prepare_stress,
)
from cyan.benchmark.models import CaseManifest
from cyan.benchmark.review import (
    compare_agent_scores,
    export_agent_review,
    export_gold_review,
    import_gold_reviews,
    score_agent_reviews,
)
from cyan.benchmark.runner import export_features, run_retrieval, score_retrieval_run
from cyan.benchmark.scoring import compare_scores
from cyan.core.llm.types import LlmResponse, ToolCallBlock, UsageStats


class _BenchmarkProvider:
    # 初始化一个会真实调用日志工具和提交工具的确定性 Provider
    def __init__(self) -> None:
        self.calls = 0

    # 根据上一轮工具结果生成下一步固定 tool call
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: object,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        del tool_schemas, bus, run_id, step, system
        self.calls += 1
        usage = UsageStats(input_tokens=20, output_tokens=10)
        if self.calls == 1:
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="read-1",
                        name="read_benchmark_log",
                        input={
                            "stream": "stderr",
                            "mode": "search",
                            "query": "CYAN_EVIDENCE",
                            "limit": 256,
                        },
                    )
                ],
                usage=usage,
            )
        if self.calls == 2:
            content = messages[-1]["content"]
            assert isinstance(content, list)
            result = json.loads(content[0]["content"])
            return LlmResponse(
                stop_reason="tool_use",
                tool_calls=[
                    ToolCallBlock(
                        id="submit-1",
                        name="submit_benchmark_diagnosis",
                        input={
                            "root_cause": "shape mismatch expected 32 features but got 31",
                            "recovery_kind": "patch",
                            "evidence_references": [result["reference"]],
                        },
                    )
                ],
                usage=usage,
            )
        return LlmResponse(stop_reason="end_turn", text="done", usage=usage)


# 写出一个最小 LogDx 上游 Case
def _write_logdx_case(root: Path, index: int) -> None:
    case_id = f"case-{index:02d}"
    directory = root / "cases" / "dev" / case_id
    directory.mkdir(parents=True)
    raw = b"setup\nRuntimeError: root cause\nProcess completed with exit code 1\n"
    (directory / "raw.log").write_bytes(raw)
    (directory / "case.json").write_text(
        json.dumps(
            {
                "case_id": case_id,
                "repo": "example/project",
                "raw_log_path": "raw.log",
                "failure_category": "test_assertion",
                "workflow_name": "tests",
            }
        ),
        encoding="utf-8",
    )
    (directory / "ground_truth.json").write_text(
        json.dumps(
            {
                "required_signals": [
                    {
                        "type": "exception",
                        "value": "RuntimeError: root cause",
                        "importance": "critical",
                        "evidence_lines": [[2, 2]],
                    }
                ],
                "expected_diagnosis": {"must_mention": ["RuntimeError", "root cause"]},
            }
        ),
        encoding="utf-8",
    )


# 功能：验证 CI 小型语料来自失败/成功子进程且可确定性重建
# 设计：连续生成两次八个 Case，比较 fingerprint、replay return code 与语料审计
def test_prepare_ci_corpus_is_reproducible(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    first = prepare_ci_core(root, limit=8)
    first_fingerprint = Corpus(root).fingerprint()
    manifest_path = root / "cases" / first[0].case_id / "manifest.json"
    legacy_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    for field in (
        "issue_url",
        "failing_revision",
        "fixing_revision",
        "runtime_image",
        "historical",
    ):
        legacy_payload["source"].pop(field, None)
    manifest_path.write_text(json.dumps(legacy_payload), encoding="utf-8")
    assert Corpus(root).fingerprint() == first_fingerprint
    second = prepare_ci_core(root, limit=8)

    assert len(first) == len(second) == 8
    assert first_fingerprint == Corpus(root).fingerprint()
    assert all(case.replay is not None for case in first)
    assert all(case.replay and case.replay.failing_returncode != 0 for case in first)
    assert all(case.replay and case.replay.fixed_returncode == 0 for case in first)
    assert Corpus(root).validate()["valid"] is True


# 功能：验证 Oracle 上限、检索预算和长尾根因能被评分器区分
# 设计：运行 Tail、Hybrid 与 Oracle，断言 Tail 漏召回且 Oracle 全召回
def test_retrieval_track_enforces_budget_and_oracle_ceiling(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    prepare_ci_core(corpus.root, limit=8)
    tail = score_retrieval_run(corpus, run_retrieval(corpus, "capsule_tail"))
    hybrid_run = run_retrieval(corpus, "heuristic_hybrid")
    hybrid = score_retrieval_run(corpus, hybrid_run)
    oracle = score_retrieval_run(corpus, run_retrieval(corpus, "oracle"))

    assert any(row.metrics.essential_recall_at_256k < 1.0 for row in tail)
    assert all(bundle.returned_bytes <= 256 * 1024 for bundle in hybrid_run.bundles)
    assert all(bundle.peak_rss_bytes > 0 for bundle in hybrid_run.bundles)
    assert all(item.cost_bytes <= 32 * 1024 for bundle in hybrid_run.bundles for item in bundle.items)
    assert all(row.metrics.essential_recall_at_256k == 1.0 for row in oracle)
    assert sum(row.metrics.essential_recall_at_256k for row in hybrid) >= sum(
        row.metrics.essential_recall_at_256k for row in tail
    )


# 功能：验证 LogDx 的行级 required signal 被转换为精确 byte range
# 设计：构造 35 个最小上游目录，导入后核对数量、许可、文本区间和外部 split
def test_import_logdx_converts_line_labels_to_byte_ranges(tmp_path: Path) -> None:
    source = tmp_path / "LogDx-1.2"
    for index in range(35):
        _write_logdx_case(source, index)
    root = tmp_path / "corpus"

    imported = import_logdx(source, root)

    assert len(imported) == 35
    first = imported[0]
    assert first.split == "external"
    assert first.source.license == "CC-BY-4.0"
    fact = first.gold_facts[0]
    raw = Corpus(root).log_path(first, "stderr").read_bytes()
    assert raw[fact.byte_start : fact.byte_end] == b"RuntimeError: root cause\n"
    assert Corpus(root).validate()["valid"] is True


# 功能：验证压力语料按三个来源、三个证据位置生成九个独立 Case
# 设计：用小尺寸真实文本替代大下载，核对文件大小、marker 范围和 stress 分层
def test_prepare_stress_builds_nine_scale_only_cases(tmp_path: Path) -> None:
    sources: dict[str, Path] = {}
    for name in ("hadoop", "spark", "bgl"):
        path = tmp_path / f"{name}.log"
        path.write_text(f"{name} INFO worker heartbeat\n" * 20, encoding="utf-8")
        sources[name] = path

    cases = prepare_stress(
        tmp_path / "corpus",
        sources,
        sizes=(8 * 1024, 16 * 1024, 32 * 1024),
    )

    assert len(cases) == 9
    for case in cases:
        assert case.tier == "scale_stress"
        fact = case.gold_facts[0]
        raw = Corpus(tmp_path / "corpus").log_path(case, "stderr").read_bytes()
        assert b"CYAN_STRESS_EVIDENCE" in raw[fact.byte_start : fact.byte_end]


# 功能：验证 Candidate 特征默认不会导出 test gold
# 设计：生成完整 48 个受控 Case，导出 JSONL 后确认只有 train/dev case_id
def test_feature_export_excludes_test_by_default(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    prepare_ci_core(corpus.root)
    output = tmp_path / "features.jsonl"

    rows = export_features(corpus, output)
    exported = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    test_ids = {case.case_id for case in corpus.cases() if case.split == "test"}

    assert rows == len(exported)
    assert not ({row["case_id"] for row in exported} & test_ids)


# 功能：验证比较报告严格按 tier 分层且不生成统一总分
# 设计：对 Core 和 Stress 的 Oracle 结果共同汇总，检查两个分区和空 combined_score
def test_compare_never_combines_tier_scores(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    prepare_ci_core(corpus.root, limit=2)
    sources: dict[str, Path] = {}
    for name in ("hadoop", "spark", "bgl"):
        path = tmp_path / f"{name}.log"
        path.write_text("INFO worker heartbeat\n" * 20, encoding="utf-8")
        sources[name] = path
    prepare_stress(corpus.root, sources, sizes=(2048, 4096, 8192))
    scores = [
        *score_retrieval_run(corpus, run_retrieval(corpus, "capsule_tail")),
        *score_retrieval_run(corpus, run_retrieval(corpus, "oracle")),
    ]

    report = compare_scores(scores)

    assert set(report["tiers"]) == {"cyan_core", "scale_stress"}
    assert report["combined_score"] is None
    core_oracle = report["tiers"]["cyan_core"]["oracle"]
    assert "essential_recall_at_64k" in core_oracle["metrics"]
    assert "essential_recall_at_256k" in core_oracle["paired_vs_capsule_tail"]


# 功能：验证 test gold 拒绝单轮审核或未确认 LLM 标注
# 设计：复制一个生成 Case 修改 split 与事实 provenance，断言 Pydantic 拒绝
def test_test_gold_requires_two_human_review_passes(tmp_path: Path) -> None:
    case = prepare_ci_core(tmp_path / "corpus", limit=1)[0]
    payload = case.model_dump(mode="json")
    payload["split"] = "test"
    payload["gold_review_status"] = "approved"
    payload["gold_facts"][0]["review_passes"] = 1

    with pytest.raises(ValidationError, match="two review passes"):
        CaseManifest.model_validate(payload)


# 功能：验证自动预标注即使已有两次计数也不能冒充人工确认 Gold
# 设计：构造 approved test Case 并保留 automatic_candidate，断言模型拒绝
def test_approved_test_gold_must_be_human_confirmed(tmp_path: Path) -> None:
    case = prepare_ci_core(tmp_path / "corpus", limit=1)[0]
    payload = case.model_dump(mode="json")
    payload["split"] = "test"
    payload["gold_review_status"] = "approved"
    payload["gold_facts"][0]["review_passes"] = 2
    payload["gold_facts"][0]["provenance"] = "automatic_candidate"

    with pytest.raises(ValidationError, match="human-confirmed"):
        CaseManifest.model_validate(payload)


# 功能：验证历史来源可由可追溯 issue/revision 对计数而不伪造 Git commit
# 设计：改写一个 Core manifest 的来源字段，审计 historical 数量和 provenance
def test_historical_issue_revisions_are_counted(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    original = prepare_ci_core(root, limit=1)[0]
    payload = original.model_dump(mode="json")
    payload["source"] = {
        "repository": "https://zenodo.org/records/8376824",
        "issue_url": "https://stackoverflow.com/questions/55142951",
        "failing_revision": "defects4ml-v1:076:buggy",
        "fixing_revision": "defects4ml-v1:076:fixed",
        "license": "CC-BY-4.0",
        "upstream_case_id": "076",
        "historical": True,
    }
    manifest = root / "cases" / original.case_id / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    audit = Corpus(root).validate()

    assert audit["valid"] is True
    assert audit["historical_core_cases"] == 1


# 功能：验证正式 Agent Track 在调用 Provider 前拒绝不完整 Core
# 设计：只生成一个 CI Case 并解析正式命令，断言缺少历史与双审时立即退出
async def test_agent_cli_requires_complete_core_before_provider(tmp_path: Path) -> None:
    corpus_root = tmp_path / "corpus"
    prepare_ci_core(corpus_root, limit=1)
    args = build_parser().parse_args(
        [
            "agent-run",
            "--corpus",
            str(corpus_root),
            "--strategy",
            "current_agent",
            "--model",
            "unconfigured-model",
            "--output",
            str(tmp_path / "agent.json"),
            "--runs-dir",
            str(tmp_path / "runs"),
        ]
    )

    with pytest.raises(SystemExit, match="requires complete Core"):
        await _agent_run_async(args)


# 功能：验证 Agent 日志工具共享预算且提交工具拒绝伪造引用
# 设计：读取一个小切片后分别提交未知和已观察 reference，检查两层结果
async def test_agent_track_accepts_only_observed_evidence(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    case = prepare_ci_core(corpus.root, limit=1)[0]
    reader = BenchmarkEvidenceReader(corpus, case, byte_budget=128)
    result = reader.read(
        ReadBenchmarkLogParams(stream="stderr", mode="tail", limit=64)
    )
    tool = SubmitBenchmarkDiagnosisTool(reader.observed)

    rejected = await tool.invoke(
        {
            "root_cause": "shape mismatch",
            "recovery_kind": "patch",
            "evidence_references": ["stderr@bytes:1-2"],
        }
    )
    accepted = await tool.invoke(
        {
            "root_cause": "shape mismatch",
            "recovery_kind": "patch",
            "evidence_references": [result["reference"]],
        }
    )

    assert rejected.is_error is True
    assert accepted.is_error is False
    assert reader.remaining() == 64


# 功能：验证 Agent 策略真实经过只读日志工具并生成可评分结果
# 设计：用确定性 Provider 发起 search 和 submit 两次调用，核对证据、token 与恢复类型
async def test_agent_strategy_runs_with_read_only_benchmark_tools(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    original = prepare_ci_core(corpus.root, limit=1)[0]
    payload = original.model_dump(mode="json")
    payload["split"] = "test"
    payload["gold_review_status"] = "approved"
    payload["gold_facts"][0]["review_passes"] = 2
    payload["gold_facts"][0]["provenance"] = "human_confirmed"
    case = CaseManifest.model_validate(payload)
    provider = _BenchmarkProvider()

    result = await run_agent_strategy(
        corpus,
        case,
        "retrieval_skill",
        provider,
        model="fake-model",
        runs_dir=tmp_path / "runs",
        byte_budget=1024,
        token_budget=1000,
    )

    assert result.diagnosis_submitted is True
    assert result.recovery_kind_correct is True
    assert result.essential_evidence_recall == 1.0
    assert result.tool_calls == 2
    assert result.input_tokens == 60
    assert result.output_tokens == 30


# 功能：验证两份一致的 Gold 盲审能够批准 test manifest
# 设计：导出一个 draft Case，分别填写两名审核者后导入并检查审核状态与次数
def test_gold_review_requires_two_matching_reviewers(tmp_path: Path) -> None:
    root = tmp_path / "corpus"
    original = prepare_ci_core(root, limit=1)[0]
    payload = original.model_dump(mode="json")
    payload["split"] = "test"
    payload["gold_review_status"] = "draft"
    payload["gold_facts"][0]["review_passes"] = 0
    manifest = root / "cases" / original.case_id / "manifest.json"
    manifest.write_text(json.dumps(payload), encoding="utf-8")
    template = tmp_path / "gold.json"
    assert export_gold_review(Corpus(root), template) == 1
    first = json.loads(template.read_text(encoding="utf-8"))
    second = json.loads(template.read_text(encoding="utf-8"))
    first["reviewer_id"] = "reviewer-a"
    second["reviewer_id"] = "reviewer-b"
    first["cases"][0]["approved"] = True
    second["cases"][0]["approved"] = True
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"
    first_path.write_text(json.dumps(first), encoding="utf-8")
    second_path.write_text(json.dumps(second), encoding="utf-8")

    report = import_gold_reviews(Corpus(root), first_path, second_path)

    approved = Corpus(root).cases()[0]
    assert report["approved_cases"] == 1
    assert approved.gold_review_status == "approved"
    assert approved.gold_facts[0].review_passes == 2
    assert approved.gold_facts[0].provenance == "human_confirmed"


# 功能：验证 Agent 盲审隐藏策略并生成双人一致率和策略汇总
# 设计：复用确定性 Agent 结果，填写两份相同 CSV 后检查 kappa 与正确率
async def test_agent_review_export_score_and_compare(tmp_path: Path) -> None:
    corpus = Corpus(tmp_path / "corpus")
    original = prepare_ci_core(corpus.root, limit=1)[0]
    payload = original.model_dump(mode="json")
    payload["split"] = "test"
    payload["gold_review_status"] = "approved"
    payload["gold_facts"][0]["review_passes"] = 2
    payload["gold_facts"][0]["provenance"] = "human_confirmed"
    case = CaseManifest.model_validate(payload)
    result = await run_agent_strategy(
        corpus,
        case,
        "retrieval_skill",
        _BenchmarkProvider(),
        model="fake-model",
        runs_dir=tmp_path / "runs",
        byte_budget=1024,
        token_budget=1000,
    )
    pack = tmp_path / "pack.json"
    key = tmp_path / "key.json"
    template = tmp_path / "review.csv"
    assert export_agent_review(
        corpus,
        [result],
        pack_path=pack,
        key_path=key,
        template_path=template,
        seed=7,
    ) == 1
    assert "strategy" not in pack.read_text(encoding="utf-8")
    review_paths = (tmp_path / "review-a.csv", tmp_path / "review-b.csv")
    blind_id = next(iter(json.loads(key.read_text(encoding="utf-8"))["items"]))
    for reviewer, path in zip(("reviewer-a", "reviewer-b"), review_paths, strict=True):
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["reviewer_id", "blind_id", "grade", "notes"],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "reviewer_id": reviewer,
                    "blind_id": blind_id,
                    "grade": "correct",
                    "notes": "",
                }
            )

    scored = score_agent_reviews([result], key, review_paths)
    compared = compare_agent_scores([scored])

    assert scored["cohen_kappa"] == 1.0
    assert compared["tiers"]["cyan_core"]["retrieval_skill"]["root_cause_accuracy"] == 1.0
