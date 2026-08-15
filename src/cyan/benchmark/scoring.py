from __future__ import annotations

import math
import random
from collections import defaultdict
from collections.abc import Iterable
from statistics import mean, median

from cyan.benchmark.models import (
    BenchmarkRun,
    CaseManifest,
    EvidenceBundle,
    EvidenceItem,
    GoldFact,
    RetrievalMetrics,
    ScoredCase,
)

_BUDGETS = (64 * 1024, 128 * 1024, 256 * 1024)


# 返回不超过指定额外字节预算的证据前缀
def _items_at_budget(bundle: EvidenceBundle, budget: int) -> list[EvidenceItem]:
    items = list(bundle.initial_items)
    used = 0
    for item in bundle.items:
        if used + item.cost_bytes > budget:
            break
        items.append(item)
        used += item.cost_bytes
    return items


# 合并同一 stream 中重叠或相邻的 byte 区间
def _merged_ranges(items: Iterable[EvidenceItem]) -> dict[str, list[tuple[int, int]]]:
    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for item in items:
        grouped[item.stream].append((item.byte_start, item.byte_end))
    merged: dict[str, list[tuple[int, int]]] = {}
    for stream, ranges in grouped.items():
        output: list[tuple[int, int]] = []
        for start, end in sorted(ranges):
            if output and start <= output[-1][1]:
                output[-1] = (output[-1][0], max(output[-1][1], end))
            else:
                output.append((start, end))
        merged[stream] = output
    return merged


# 判断某个 gold fact 是否被返回区间完整覆盖
def _fact_covered(fact: GoldFact, items: Iterable[EvidenceItem]) -> bool:
    return any(
        start <= fact.byte_start and end >= fact.byte_end
        for start, end in _merged_ranges(items).get(fact.stream, [])
    )


# 计算指定重要性的事实召回率
def _recall(case: CaseManifest, items: list[EvidenceItem], importance: str) -> float:
    facts = [fact for fact in case.gold_facts if fact.importance == importance]
    if not facts:
        return 1.0
    return sum(_fact_covered(fact, items) for fact in facts) / len(facts)


# 计算证据列表的事实级 nDCG，重复命中不重复获益
def _ndcg(case: CaseManifest, bundle: EvidenceBundle) -> float:
    if not case.gold_facts:
        return 1.0 if bundle.abstained else 0.0
    seen: set[str] = set()
    gains: list[float] = []
    for item in [*bundle.initial_items, *bundle.items]:
        gain = 0.0
        for fact in case.gold_facts:
            if fact.fact_id in seen or not _fact_covered(fact, [item]):
                continue
            seen.add(fact.fact_id)
            gain += 3.0 if fact.importance == "essential" else 1.0
        gains.append(gain)
    dcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(gains))
    ideal = sorted(
        [3.0 if fact.importance == "essential" else 1.0 for fact in case.gold_facts],
        reverse=True,
    )
    idcg = sum(gain / math.log2(index + 2) for index, gain in enumerate(ideal))
    return min(1.0, dcg / idcg) if idcg else 1.0


# 返回首次命中任一 gold fact 时累计消耗的额外字节
def _first_relevant_bytes(case: CaseManifest, bundle: EvidenceBundle) -> int | None:
    if any(_fact_covered(fact, bundle.initial_items) for fact in case.gold_facts):
        return 0
    used = 0
    for item in bundle.items:
        used += item.cost_bytes
        if any(_fact_covered(fact, [item]) for fact in case.gold_facts):
            return used
    return None


# 计算返回区间与所有 gold 区间的交集字节数
def _relevant_bytes(case: CaseManifest, items: list[EvidenceItem]) -> int:
    relevant = 0
    for stream, ranges in _merged_ranges(items).items():
        gold = [
            (fact.byte_start, fact.byte_end)
            for fact in case.gold_facts
            if fact.stream == stream
        ]
        for start, end in ranges:
            intersections = [
                (max(start, gold_start), min(end, gold_end))
                for gold_start, gold_end in gold
                if max(start, gold_start) < min(end, gold_end)
            ]
            relevant += sum(right - left for left, right in _merge_tuples(intersections))
    return relevant


# 合并普通 byte 区间元组
def _merge_tuples(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for start, end in sorted(ranges):
        if output and start <= output[-1][1]:
            output[-1] = (output[-1][0], max(output[-1][1], end))
        else:
            output.append((start, end))
    return output


# 计算重复返回字节占总返回成本的比例
def _redundancy_rate(items: list[EvidenceItem]) -> float:
    total = sum(item.cost_bytes for item in items)
    if total == 0:
        return 0.0
    unique = sum(
        end - start
        for ranges in _merged_ranges(items).values()
        for start, end in ranges
    )
    return max(0.0, min(1.0, (total - unique) / total))


# 为一个 EvidenceBundle 计算全部离线检索指标
def score_bundle(case: CaseManifest, bundle: EvidenceBundle) -> ScoredCase:
    if bundle.case_id != case.case_id:
        raise ValueError("bundle case_id does not match manifest")
    at_64, at_128, at_256 = (
        _items_at_budget(bundle, budget) for budget in _BUDGETS
    )
    returned = bundle.returned_bytes
    density = _relevant_bytes(case, bundle.items) / returned if returned else 0.0
    no_evidence_expected = not case.gold_facts
    metrics = RetrievalMetrics(
        essential_recall_at_64k=_recall(case, at_64, "essential"),
        essential_recall_at_128k=_recall(case, at_128, "essential"),
        essential_recall_at_256k=_recall(case, at_256, "essential"),
        supporting_recall_at_256k=_recall(case, at_256, "supporting"),
        ndcg=_ndcg(case, bundle),
        first_relevant_evidence_bytes=_first_relevant_bytes(case, bundle),
        first_relevant_missed=(
            bool(case.gold_facts) and _first_relevant_bytes(case, bundle) is None
        ),
        evidence_density=max(0.0, min(1.0, density)),
        redundancy_rate=_redundancy_rate(bundle.items),
        abstention_correct=bundle.abstained if no_evidence_expected else None,
        elapsed_ms=bundle.elapsed_ms,
        returned_bytes=bundle.returned_bytes,
        scanned_bytes=bundle.scanned_bytes,
        peak_rss_bytes=bundle.peak_rss_bytes,
        rss_delta_bytes=bundle.rss_delta_bytes,
    )
    evidence_position = case.stress_position
    if evidence_position is None and case.tier == "scale_stress":
        suffix = case.case_id.rsplit("-", 1)[-1]
        if suffix in {"front", "middle", "tail"}:
            evidence_position = suffix  # type: ignore[assignment]
    return ScoredCase(
        case_id=case.case_id,
        tier=case.tier,
        split=case.split,
        method=bundle.method,
        metrics=metrics,
        workload=case.workload,
        log_bytes=sum(artifact.size for artifact in case.logs.values()),
        evidence_position=evidence_position,
    )


# 将一次 BenchmarkRun 与语料逐 Case 对齐并评分
def score_run(cases: list[CaseManifest], run: BenchmarkRun) -> list[ScoredCase]:
    by_id = {case.case_id: case for case in cases}
    scores: list[ScoredCase] = []
    for bundle in run.bundles:
        if bundle.case_id not in by_id:
            raise ValueError(f"run references unknown case: {bundle.case_id}")
        scores.append(score_bundle(by_id[bundle.case_id], bundle))
    return scores


# 对一组数值进行确定性 bootstrap 均值置信区间
def _bootstrap_interval(
    values: list[float],
    *,
    seed: int,
    samples: int = 1000,
) -> tuple[float, float]:
    if not values:
        return 0.0, 0.0
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _ in values)
        for _sample in range(samples)
    )
    return estimates[int(samples * 0.025)], estimates[min(samples - 1, int(samples * 0.975))]


# 返回已排序数列的最近秩百分位值
def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1)]


# 汇总一个逐 Case 数值指标并给出确定性 bootstrap 区间
def _metric_summary(values: list[float], *, seed: int) -> dict[str, float | list[float]]:
    low, high = _bootstrap_interval(values, seed=seed)
    return {
        "mean": mean(values) if values else 0.0,
        "median": median(values) if values else 0.0,
        "p95": _percentile(values, 0.95),
        "ci95": [low, high],
    }


# 对同一 Case 的方法结果与 Tail 基线执行配对 bootstrap
def _paired_summary(
    rows: list[ScoredCase],
    baseline: dict[str, ScoredCase],
    field: str,
    *,
    seed: int,
) -> dict[str, float | int | list[float]] | None:
    deltas = [
        float(getattr(row.metrics, field)) - float(getattr(baseline[row.case_id].metrics, field))
        for row in rows
        if row.case_id in baseline
    ]
    if not deltas:
        return None
    low, high = _bootstrap_interval(deltas, seed=seed)
    return {
        "paired_cases": len(deltas),
        "mean_delta": mean(deltas),
        "ci95": [low, high],
    }


# 输出压力集逐规模和证据位置结果，避免总体均值掩盖增长曲线
def _scale_breakdown(scores: list[ScoredCase]) -> dict[str, list[dict[str, object]]]:
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in sorted(
        (item for item in scores if item.tier == "scale_stress"),
        key=lambda item: (item.method, item.log_bytes, item.evidence_position or ""),
    ):
        grouped[row.method].append(
            {
                "workload": row.workload,
                "log_bytes": row.log_bytes,
                "evidence_position": row.evidence_position,
                "essential_recall_at_256k": row.metrics.essential_recall_at_256k,
                "first_relevant_evidence_bytes": row.metrics.first_relevant_evidence_bytes,
                "elapsed_ms": row.metrics.elapsed_ms,
                "scanned_bytes": row.metrics.scanned_bytes,
                "peak_rss_bytes": row.metrics.peak_rss_bytes,
                "rss_delta_bytes": row.metrics.rss_delta_bytes,
            }
        )
    return dict(grouped)


# 按 split 输出样本数和主指标，辅助检查 Core 切分差异
def _split_breakdown(scores: list[ScoredCase]) -> dict[str, dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[ScoredCase]] = defaultdict(list)
    for row in scores:
        grouped[(row.tier, row.split, row.method)].append(row)
    return {
        f"{tier}/{split}/{method}": {
            "cases": len(rows),
            "essential_recall_at_256k": mean(
                row.metrics.essential_recall_at_256k for row in rows
            ),
            "ndcg": mean(row.metrics.ndcg for row in rows),
        }
        for (tier, split, method), rows in sorted(grouped.items())
    }


# 按 tier 和 method 分层汇总，禁止生成跨 tier 总分
def compare_scores(scores: list[ScoredCase], *, seed: int = 0) -> dict[str, object]:
    grouped: dict[tuple[str, str], list[ScoredCase]] = defaultdict(list)
    for score in scores:
        grouped[(score.tier, score.method)].append(score)
    tiers: dict[str, dict[str, object]] = defaultdict(dict)
    baselines: dict[str, dict[str, ScoredCase]] = defaultdict(dict)
    for (tier, method), rows in grouped.items():
        if method == "capsule_tail":
            baselines[tier] = {row.case_id: row for row in rows}
    metric_fields = (
        "essential_recall_at_64k",
        "essential_recall_at_128k",
        "essential_recall_at_256k",
        "supporting_recall_at_256k",
        "ndcg",
        "evidence_density",
        "redundancy_rate",
        "elapsed_ms",
        "returned_bytes",
        "scanned_bytes",
        "peak_rss_bytes",
        "rss_delta_bytes",
    )
    for (tier, method), rows in sorted(grouped.items()):
        method_seed = seed + int.from_bytes(hashlib_sha(f"{tier}:{method}")[:4], "big")
        metrics: dict[str, object] = {
            field: _metric_summary(
                [float(getattr(row.metrics, field)) for row in rows],
                seed=method_seed + index,
            )
            for index, field in enumerate(metric_fields)
        }
        first_relevant = [
            float(row.metrics.first_relevant_evidence_bytes)
            for row in rows
            if row.metrics.first_relevant_evidence_bytes is not None
        ]
        metrics["first_relevant_evidence_bytes"] = _metric_summary(
            first_relevant,
            seed=method_seed + len(metric_fields),
        )
        miss_values = [float(row.metrics.first_relevant_missed) for row in rows]
        metrics["first_relevant_miss_rate"] = _metric_summary(
            miss_values,
            seed=method_seed + len(metric_fields) + 1,
        )
        abstentions = [
            float(row.metrics.abstention_correct)
            for row in rows
            if row.metrics.abstention_correct is not None
        ]
        metrics["abstention_accuracy"] = (
            _metric_summary(
                abstentions,
                seed=method_seed + len(metric_fields) + 2,
            )
            if abstentions
            else None
        )
        paired = {
            field: summary
            for index, field in enumerate(
                (
                    "essential_recall_at_64k",
                    "essential_recall_at_128k",
                    "essential_recall_at_256k",
                    "ndcg",
                    "evidence_density",
                    "elapsed_ms",
                    "scanned_bytes",
                    "rss_delta_bytes",
                )
            )
            if (
                summary := _paired_summary(
                    rows,
                    baselines[tier],
                    field,
                    seed=method_seed + 100 + index,
                )
            )
            is not None
        }
        tiers[tier][method] = {
            "cases": len(rows),
            "metrics": metrics,
            "paired_vs_capsule_tail": paired,
            "abstention_cases": len(abstentions),
        }
    return {
        "schema_version": 1,
        "tiers": dict(tiers),
        "breakdowns": {
            "splits": _split_breakdown(scores),
            "scale_stress": _scale_breakdown(scores),
        },
        "combined_score": None,
    }


# 返回字符串的稳定摘要字节，避免 Python hash 随进程变化
def hashlib_sha(value: str) -> bytes:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).digest()
