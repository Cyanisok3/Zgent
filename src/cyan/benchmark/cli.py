from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import cast

from cyan.benchmark.agent_track import create_benchmark_provider, run_agent_strategy
from cyan.benchmark.corpus import (
    Corpus,
    fetch_and_import_logdx,
    import_historical_bundles,
    import_logdx,
    prepare_ci_core,
    prepare_controlled_core,
    prepare_stress,
)
from cyan.benchmark.historical import prepare_defects4ml_history
from cyan.benchmark.models import BenchmarkSplit, BenchmarkTier, StrategyName
from cyan.benchmark.retrieval import DEFAULT_BUDGET, retrievers
from cyan.benchmark.review import (
    compare_agent_scores,
    export_agent_review,
    export_gold_review,
    import_agent_review,
    import_gold_reviews,
    read_agent_results,
    score_agent_reviews,
)
from cyan.benchmark.runner import (
    export_features,
    read_run,
    read_scores,
    run_retrieval,
    score_retrieval_run,
    write_run,
    write_scores,
)
from cyan.benchmark.scoring import compare_scores


# 解析 name=path 形式的 LogHub 来源参数
def _source_map(values: list[str]) -> dict[str, Path]:
    sources: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid source {value!r}; expected name=path")
        name, raw_path = value.split("=", 1)
        sources[name] = Path(raw_path).expanduser().resolve()
    return sources


# 生成或导入指定层级语料并输出审计结果
def _prepare(args: argparse.Namespace) -> None:
    root = args.corpus.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.profile == "ci":
        prepare_ci_core(root)
    if args.profile in {"core", "all"}:
        prepare_controlled_core(root)
        if args.historical_source is not None:
            import_historical_bundles(args.historical_source, root)
    if args.profile in {"external", "all"}:
        if args.logdx_source is not None:
            import_logdx(args.logdx_source, root)
        else:
            fetch_and_import_logdx(root, args.cache_dir)
    if args.profile in {"stress", "all"}:
        sources = _source_map(args.loghub_source)
        prepare_stress(root, sources, sizes=tuple(args.stress_sizes))
    require_complete = args.profile in {"core", "all"} and not args.allow_incomplete_core
    audit = Corpus(root).validate(require_complete_core=require_complete)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["valid"]:
        raise SystemExit(2)


# 校验已有语料并打印 fingerprint 与分层计数
def _validate(args: argparse.Namespace) -> None:
    audit = Corpus(args.corpus).validate(require_complete_core=args.require_complete_core)
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["valid"]:
        raise SystemExit(2)


# 在固定容器中重放 Defects4ML 并生成可审核历史 bundle
def _prepare_history(args: argparse.Namespace) -> None:
    audit = prepare_defects4ml_history(
        args.archive,
        args.output,
        args.work_dir,
        required=args.required,
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2, sort_keys=True))
    if not audit["published"]:
        raise SystemExit(2)


# 执行一个确定性离线检索器
def _run(args: argparse.Namespace) -> None:
    tiers = cast(set[BenchmarkTier] | None, set(args.tier) if args.tier else None)
    splits = cast(set[BenchmarkSplit] | None, set(args.split) if args.split else None)
    run = run_retrieval(
        Corpus(args.corpus),
        args.method,
        byte_budget=args.byte_budget,
        seed=args.seed,
        tiers=tiers,
        splits=splits,
        isolated=not args.no_isolation,
    )
    write_run(args.output, run)
    print(json.dumps({"method": args.method, "cases": len(run.bundles)}, sort_keys=True))


# 对一个已持久化 Run 离线评分
def _score(args: argparse.Namespace) -> None:
    corpus = Corpus(args.corpus)
    scores = score_retrieval_run(corpus, read_run(args.run))
    write_scores(args.output, scores)
    print(json.dumps({"scores": len(scores)}, sort_keys=True))


# 汇总多个评分文件且保持 tier 完全分离
def _compare(args: argparse.Namespace) -> None:
    report = compare_scores(read_scores(args.scores), seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tiers = cast(dict[str, object], report["tiers"])
    print(json.dumps({"tiers": sorted(tiers)}, sort_keys=True))


# 导出后续 GP/LTR 可使用的候选特征 JSONL
def _features(args: argparse.Namespace) -> None:
    rows = export_features(Corpus(args.corpus), args.output, include_test=args.include_test)
    print(json.dumps({"rows": rows}, sort_keys=True))


# 依次运行可选的真实 LLM Agent 策略评测
async def _agent_run_async(args: argparse.Namespace) -> None:
    corpus = Corpus(args.corpus)
    if not args.allow_incomplete_core:
        audit = corpus.validate(require_complete_core=True)
        if not audit["valid"]:
            raise SystemExit(
                "Agent Track requires complete Core and approved test Gold; "
                "use --allow-incomplete-core only for local debugging"
            )
    selected = [
        case
        for case in corpus.cases()
        if case.split in {"test", "external"}
        and (not args.case_id or case.case_id in args.case_id)
    ]
    if not args.allow_incomplete_core and len(selected) != 47:
        raise SystemExit(f"Agent Track requires 47 cases; found {len(selected)}")
    provider = create_benchmark_provider(args.model, args.temperature)
    results = []
    for case in selected:
        result = await run_agent_strategy(
            corpus,
            case,
            cast(StrategyName, args.strategy),
            provider,
            model=args.model,
            runs_dir=args.runs_dir,
            byte_budget=args.byte_budget,
            token_budget=args.token_budget,
        )
        if args.input_price_per_million is not None and args.output_price_per_million is not None:
            result = result.model_copy(
                update={
                    "estimated_cost_usd": (
                        result.input_tokens * args.input_price_per_million
                        + result.output_tokens * args.output_price_per_million
                    )
                    / 1_000_000
                }
            )
        results.append(result.model_dump(mode="json"))
    payload = {
        "schema_version": 1,
        "strategy": args.strategy,
        "model": args.model,
        "dataset_fingerprint": corpus.fingerprint(),
        "input_price_per_million": args.input_price_per_million,
        "output_price_per_million": args.output_price_per_million,
        "byte_budget": args.byte_budget,
        "token_budget": args.token_budget,
        "temperature": args.temperature,
        "max_steps": 12,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"strategy": args.strategy, "cases": len(results)}, sort_keys=True))


# 启动异步 Agent Track 命令
def _agent_run(args: argparse.Namespace) -> None:
    asyncio.run(_agent_run_async(args))


# 导出不暴露策略与 Case 标识的人工盲审包
def _review_export(args: argparse.Namespace) -> None:
    count = export_agent_review(
        Corpus(args.corpus),
        read_agent_results(args.agent_results),
        pack_path=args.pack,
        key_path=args.key,
        template_path=args.template,
        seed=args.seed,
    )
    print(json.dumps({"review_items": count}, sort_keys=True))


# 导出不含检索和 Agent 结果的 Core test Gold 复核模板
def _gold_review_export(args: argparse.Namespace) -> None:
    count = export_gold_review(Corpus(args.corpus), args.output)
    print(json.dumps({"gold_review_cases": count}, sort_keys=True))


# 合并两份独立 Gold 复核并仅在完全一致时批准 manifest
def _gold_review_import(args: argparse.Namespace) -> None:
    result = import_gold_reviews(Corpus(args.corpus), args.review[0], args.review[1])
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    if result["disagreements"]:
        raise SystemExit(2)


# 校验并规范化一份人工盲审 CSV
def _review_import(args: argparse.Namespace) -> None:
    key = json.loads(args.key.read_text(encoding="utf-8"))["items"]
    reviewer, grades = import_agent_review(args.review, set(key))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {"schema_version": 1, "reviewer_id": reviewer, "grades": grades},
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"reviewer_id": reviewer, "grades": len(grades)}, sort_keys=True))


# 将两份盲审评分与 Agent 原始结果合并为逐 Case 评分
def _agent_score(args: argparse.Namespace) -> None:
    report = score_agent_reviews(
        read_agent_results(args.agent_results),
        args.key,
        (args.review[0], args.review[1]),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "results": len(report["results"]),
                "disagreements": len(report["disagreements"]),
            },
            sort_keys=True,
        )
    )


# 汇总一个或多个 Agent 评分文件并保持策略独立
def _agent_compare(args: argparse.Namespace) -> None:
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in args.scores]
    report = compare_agent_scores(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"tiers": sorted(report["tiers"])}, sort_keys=True))


# 构造统一 prepare/run/score/compare CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Cyan Evidence Retrieval Benchmark v1")
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="generate or import benchmark corpus")
    prepare.add_argument("--corpus", type=Path, required=True)
    prepare.add_argument(
        "--profile",
        choices=["ci", "core", "external", "stress", "all"],
        default="ci",
    )
    prepare.add_argument("--historical-source", type=Path)
    prepare.add_argument("--logdx-source", type=Path)
    prepare.add_argument("--cache-dir", type=Path)
    prepare.add_argument("--loghub-source", action="append", default=[])
    prepare.add_argument(
        "--stress-sizes",
        nargs=3,
        type=int,
        default=[5 * 1024**2, 50 * 1024**2, 500 * 1024**2],
    )
    prepare.add_argument("--allow-incomplete-core", action="store_true")
    prepare.set_defaults(handler=_prepare)

    validate = commands.add_parser("validate", help="validate corpus integrity")
    validate.add_argument("--corpus", type=Path, required=True)
    validate.add_argument("--require-complete-core", action="store_true")
    validate.set_defaults(handler=_validate)

    history = commands.add_parser(
        "history-prepare", help="replay Defects4ML candidates in fixed containers"
    )
    history.add_argument("--archive", type=Path, required=True)
    history.add_argument("--output", type=Path, required=True)
    history.add_argument("--work-dir", type=Path, required=True)
    history.add_argument("--required", type=int, default=12)
    history.set_defaults(handler=_prepare_history)

    run = commands.add_parser("run", help="run one offline retriever")
    run.add_argument("--corpus", type=Path, required=True)
    run.add_argument("--method", choices=sorted(retrievers()), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--byte-budget", type=int, default=DEFAULT_BUDGET)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--no-isolation", action="store_true")
    run.add_argument(
        "--tier",
        action="append",
        choices=["cyan_core", "external_generalization", "scale_stress"],
    )
    run.add_argument(
        "--split",
        action="append",
        choices=["train", "dev", "test", "external", "stress"],
    )
    run.set_defaults(handler=_run)

    score = commands.add_parser("score", help="score one persisted run")
    score.add_argument("--corpus", type=Path, required=True)
    score.add_argument("--run", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    score.set_defaults(handler=_score)

    compare = commands.add_parser("compare", help="compare scored runs by tier")
    compare.add_argument("--scores", type=Path, nargs="+", required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--seed", type=int, default=0)
    compare.set_defaults(handler=_compare)

    features = commands.add_parser("export-features", help="export GP/LTR candidates")
    features.add_argument("--corpus", type=Path, required=True)
    features.add_argument("--output", type=Path, required=True)
    features.add_argument("--include-test", action="store_true")
    features.set_defaults(handler=_features)

    agent = commands.add_parser("agent-run", help="run optional LLM strategy track")
    agent.add_argument("--corpus", type=Path, required=True)
    agent.add_argument(
        "--strategy",
        choices=["current_agent", "retrieval_skill", "readonly_subagents"],
        required=True,
    )
    agent.add_argument("--model", required=True)
    agent.add_argument("--output", type=Path, required=True)
    agent.add_argument("--runs-dir", type=Path, required=True)
    agent.add_argument("--case-id", action="append")
    agent.add_argument("--allow-incomplete-core", action="store_true")
    agent.add_argument("--byte-budget", type=int, default=DEFAULT_BUDGET)
    agent.add_argument("--token-budget", type=int, default=80_000)
    agent.add_argument("--temperature", type=float, default=0.0)
    agent.add_argument("--input-price-per-million", type=float)
    agent.add_argument("--output-price-per-million", type=float)
    agent.set_defaults(handler=_agent_run)

    review_export = commands.add_parser("review-export", help="export blinded Agent review pack")
    review_export.add_argument("--corpus", type=Path, required=True)
    review_export.add_argument("--agent-results", type=Path, nargs="+", required=True)
    review_export.add_argument("--pack", type=Path, required=True)
    review_export.add_argument("--key", type=Path, required=True)
    review_export.add_argument("--template", type=Path, required=True)
    review_export.add_argument("--seed", type=int, default=0)
    review_export.set_defaults(handler=_review_export)

    gold_export = commands.add_parser("gold-review-export", help="export Core test gold review")
    gold_export.add_argument("--corpus", type=Path, required=True)
    gold_export.add_argument("--output", type=Path, required=True)
    gold_export.set_defaults(handler=_gold_review_export)

    gold_import = commands.add_parser("gold-review-import", help="merge two Core gold reviews")
    gold_import.add_argument("--corpus", type=Path, required=True)
    gold_import.add_argument("--review", type=Path, nargs=2, required=True)
    gold_import.set_defaults(handler=_gold_review_import)

    review_import = commands.add_parser("review-import", help="validate one blinded review CSV")
    review_import.add_argument("--key", type=Path, required=True)
    review_import.add_argument("--review", type=Path, required=True)
    review_import.add_argument("--output", type=Path, required=True)
    review_import.set_defaults(handler=_review_import)

    agent_score = commands.add_parser("agent-score", help="join two reviews with Agent results")
    agent_score.add_argument("--agent-results", type=Path, nargs="+", required=True)
    agent_score.add_argument("--key", type=Path, required=True)
    agent_score.add_argument("--review", type=Path, nargs=2, required=True)
    agent_score.add_argument("--output", type=Path, required=True)
    agent_score.set_defaults(handler=_agent_score)

    agent_compare = commands.add_parser("agent-compare", help="compare reviewed Agent strategies")
    agent_compare.add_argument("--scores", type=Path, nargs="+", required=True)
    agent_compare.add_argument("--output", type=Path, required=True)
    agent_compare.set_defaults(handler=_agent_compare)
    return parser


# 解析命令行并分派到对应 Benchmark 操作
def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)
