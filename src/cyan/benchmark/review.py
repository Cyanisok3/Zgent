from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Iterable
from pathlib import Path
from statistics import mean, median
from typing import Any, cast

from cyan.benchmark.corpus import Corpus
from cyan.benchmark.models import AgentStrategyResult, CaseManifest, DiagnosisGrade, GoldFact

_GRADES: tuple[DiagnosisGrade, ...] = ("correct", "partial", "incorrect", "abstain")


# 导出不含算法结果的 Core test Gold 人工复核模板
def export_gold_review(corpus: Corpus, output: Path) -> int:
    cases: list[dict[str, Any]] = []
    for case in corpus.cases():
        if case.tier != "cyan_core" or case.split != "test":
            continue
        facts: list[dict[str, Any]] = []
        for fact in case.gold_facts:
            raw = corpus.log_path(case, fact.stream).read_bytes()
            facts.append(
                {
                    **fact.model_dump(mode="json"),
                    "evidence_text": raw[fact.byte_start : fact.byte_end].decode(
                        "utf-8", errors="replace"
                    ),
                }
            )
        cases.append(
            {
                "case_id": case.case_id,
                "facts": facts,
                "root_cause_rubric": case.root_cause_rubric,
                "expected_recovery_kind": case.expected_recovery_kind,
                "approved": False,
                "notes": "",
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {"schema_version": 1, "reviewer_id": "", "cases": cases},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return len(cases)


# 规范化一份 Gold 复核内容以比较两名审核者的结论
def _normalize_gold_review(payload: dict[str, Any]) -> tuple[str, dict[str, dict[str, Any]]]:
    reviewer = str(payload.get("reviewer_id", "")).strip()
    if not reviewer:
        raise ValueError("gold review requires reviewer_id")
    normalized: dict[str, dict[str, Any]] = {}
    for item in payload.get("cases", []):
        case_id = str(item["case_id"])
        if case_id in normalized:
            raise ValueError(f"duplicate case in gold review: {case_id}")
        if item.get("approved") is not True:
            raise ValueError(f"gold review has unapproved case: {case_id}")
        facts = []
        for raw_fact in item.get("facts", []):
            fact = dict(raw_fact)
            fact.pop("evidence_text", None)
            fact.pop("review_passes", None)
            fact.pop("provenance", None)
            facts.append(fact)
        normalized[case_id] = {
            "facts": facts,
            "root_cause_rubric": list(item.get("root_cause_rubric", [])),
            "expected_recovery_kind": item["expected_recovery_kind"],
        }
    return reviewer, normalized


# 合并完全一致的两份 Gold 复核并把 test manifest 标记为已批准
def import_gold_reviews(corpus: Corpus, first_path: Path, second_path: Path) -> dict[str, Any]:
    first_payload = json.loads(first_path.read_text(encoding="utf-8"))
    second_payload = json.loads(second_path.read_text(encoding="utf-8"))
    first_reviewer, first = _normalize_gold_review(first_payload)
    second_reviewer, second = _normalize_gold_review(second_payload)
    if first_reviewer == second_reviewer:
        raise ValueError("gold reviewers must be distinct")
    expected = {
        case.case_id
        for case in corpus.cases()
        if case.tier == "cyan_core" and case.split == "test"
    }
    if set(first) != expected or set(second) != expected:
        raise ValueError("gold reviews must cover every Core test case")
    disagreements = [case_id for case_id in sorted(expected) if first[case_id] != second[case_id]]
    if disagreements:
        return {
            "approved_cases": 0,
            "reviewers": [first_reviewer, second_reviewer],
            "disagreements": disagreements,
        }
    by_id = {case.case_id: case for case in corpus.cases()}
    for case_id in sorted(expected):
        current = by_id[case_id]
        reviewed = first[case_id]
        facts = [
            GoldFact.model_validate(
                {
                    **fact,
                    "provenance": "human_confirmed",
                    "review_passes": 2,
                }
            )
            for fact in reviewed["facts"]
        ]
        approved = CaseManifest.model_validate(
            {
                **current.model_dump(mode="json"),
                "gold_facts": [fact.model_dump(mode="json") for fact in facts],
                "root_cause_rubric": reviewed["root_cause_rubric"],
                "expected_recovery_kind": reviewed["expected_recovery_kind"],
                "gold_review_status": "approved",
            }
        )
        manifest_path = corpus.root / "cases" / case_id / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                approved.model_dump(mode="json"),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "approved_cases": len(expected),
        "reviewers": [first_reviewer, second_reviewer],
        "disagreements": [],
    }


# 读取一个或多个 Agent 运行产物并校验逐 Case 结果
def read_agent_results(paths: Iterable[Path]) -> list[AgentStrategyResult]:
    results: list[AgentStrategyResult] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.extend(AgentStrategyResult.model_validate(item) for item in payload["results"])
    identities = [(result.case_id, result.strategy, result.model) for result in results]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate Agent result identity")
    return results


# 为盲审条目生成不暴露 Case 与策略的稳定标识
def _blind_id(result: AgentStrategyResult, salt: str) -> str:
    raw = f"{salt}:{result.case_id}:{result.strategy}:{result.model}".encode()
    return hashlib.sha256(raw).hexdigest()[:16]


# 导出随机排序的 Agent 诊断盲审包、映射键和空白评分表
def export_agent_review(
    corpus: Corpus,
    results: list[AgentStrategyResult],
    *,
    pack_path: Path,
    key_path: Path,
    template_path: Path,
    seed: int,
) -> int:
    cases = {case.case_id: case for case in corpus.cases()}
    salt = hashlib.sha256(f"cyan-agent-review:{seed}".encode()).hexdigest()
    items: list[dict[str, object]] = []
    key: dict[str, dict[str, str]] = {}
    for result in results:
        if result.case_id not in cases:
            raise ValueError(f"Agent result references unknown case: {result.case_id}")
        case = cases[result.case_id]
        blind_id = _blind_id(result, salt)
        key[blind_id] = {
            "case_id": result.case_id,
            "strategy": result.strategy,
            "model": result.model,
        }
        items.append(
            {
                "blind_id": blind_id,
                "failure_kind": case.failure_kind,
                "phase": case.phase,
                "root_cause_rubric": case.root_cause_rubric or case.expected_diagnosis_terms,
                "expected_recovery_kind": case.expected_recovery_kind,
                "diagnosis": result.root_cause,
            }
        )
    random.Random(seed).shuffle(items)
    pack_path.parent.mkdir(parents=True, exist_ok=True)
    pack_path.write_text(
        json.dumps({"schema_version": 1, "items": items}, ensure_ascii=False, indent=2)
        + "\n",
        encoding="utf-8",
    )
    key_path.parent.mkdir(parents=True, exist_ok=True)
    key_path.write_text(
        json.dumps({"schema_version": 1, "salt": salt, "items": key}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    template_path.parent.mkdir(parents=True, exist_ok=True)
    with template_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["reviewer_id", "blind_id", "grade", "notes"])
        writer.writeheader()
        for item in items:
            writer.writerow(
                {
                    "reviewer_id": "",
                    "blind_id": item["blind_id"],
                    "grade": "",
                    "notes": "",
                }
            )
    return len(items)


# 导入一份完整人工评分表并验证盲审标识与评分值
def import_agent_review(
    path: Path,
    expected_ids: set[str],
) -> tuple[str, dict[str, DiagnosisGrade]]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reviewers = {row["reviewer_id"].strip() for row in rows if row["reviewer_id"].strip()}
    if len(reviewers) != 1:
        raise ValueError("one review file must contain exactly one reviewer_id")
    grades: dict[str, DiagnosisGrade] = {}
    for row in rows:
        blind_id = row["blind_id"].strip()
        grade = row["grade"].strip()
        if blind_id in grades:
            raise ValueError(f"duplicate blind_id in review: {blind_id}")
        if blind_id not in expected_ids:
            raise ValueError(f"unknown blind_id in review: {blind_id}")
        if grade not in _GRADES:
            raise ValueError(f"invalid grade for {blind_id}: {grade!r}")
        grades[blind_id] = cast(DiagnosisGrade, grade)
    if set(grades) != expected_ids:
        raise ValueError("review is incomplete")
    return reviewers.pop(), grades


# 计算两名复核者四分类评分的 Cohen kappa
def _cohen_kappa(first: dict[str, DiagnosisGrade], second: dict[str, DiagnosisGrade]) -> float:
    identifiers = sorted(first)
    observed = mean(float(first[item] == second[item]) for item in identifiers)
    first_counts = Counter(first.values())
    second_counts = Counter(second.values())
    expected = sum(
        first_counts[grade] / len(identifiers) * second_counts[grade] / len(identifiers)
        for grade in _GRADES
    )
    return (observed - expected) / (1.0 - expected) if expected < 1.0 else 1.0


# 合并双人盲审并列出必须人工裁决的分歧项
def score_agent_reviews(
    results: list[AgentStrategyResult],
    key_path: Path,
    review_paths: tuple[Path, Path],
) -> dict[str, Any]:
    key_payload = json.loads(key_path.read_text(encoding="utf-8"))
    key = key_payload["items"]
    expected_ids = set(key)
    first_name, first = import_agent_review(review_paths[0], expected_ids)
    second_name, second = import_agent_review(review_paths[1], expected_ids)
    if first_name == second_name:
        raise ValueError("reviewer_id values must be distinct")
    by_identity = {(result.case_id, result.strategy, result.model): result for result in results}
    scored: list[dict[str, Any]] = []
    disagreements: list[str] = []
    for blind_id, identity in sorted(key.items()):
        lookup = (identity["case_id"], identity["strategy"], identity["model"])
        if lookup not in by_identity:
            raise ValueError(f"review key references missing Agent result: {lookup}")
        agreed = first[blind_id] == second[blind_id]
        if not agreed:
            disagreements.append(blind_id)
        result = by_identity[lookup]
        scored.append(
            {
                "blind_id": blind_id,
                **identity,
                "tier": result.tier,
                "split": result.split,
                "reviewer_grades": [first[blind_id], second[blind_id]],
                "consensus_grade": first[blind_id] if agreed else None,
                "essential_evidence_recall": result.essential_evidence_recall,
                "retrieved_essential_evidence_recall": (
                    result.retrieved_essential_evidence_recall
                ),
                "recovery_kind_correct": result.recovery_kind_correct,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "tool_calls": result.tool_calls,
                "estimated_cost_usd": result.estimated_cost_usd,
                "elapsed_ms": result.elapsed_ms,
            }
        )
    return {
        "schema_version": 1,
        "reviewers": [first_name, second_name],
        "cohen_kappa": _cohen_kappa(first, second),
        "disagreements": disagreements,
        "results": scored,
    }


# 对 Agent 逐 Case 数值执行确定性 bootstrap 均值区间
def _agent_ci(values: list[float], *, seed: int, samples: int = 1000) -> list[float]:
    if not values:
        return [0.0, 0.0]
    generator = random.Random(seed)
    estimates = sorted(
        mean(generator.choice(values) for _item in values) for _sample in range(samples)
    )
    return [estimates[25], estimates[974]]


# 按 tier 和策略汇总双人共识评分与资源指标
def compare_agent_scores(payloads: Iterable[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    disagreements: list[str] = []
    kappas: list[float] = []
    for payload in payloads:
        kappas.append(float(payload["cohen_kappa"]))
        disagreements.extend(str(item) for item in payload["disagreements"])
        for row in payload["results"]:
            grouped[(str(row.get("tier") or "unknown"), str(row["strategy"]))].append(row)
    tiers: dict[str, dict[str, object]] = defaultdict(dict)
    for (tier, strategy), rows in sorted(grouped.items()):
        agreed = [row for row in rows if row["consensus_grade"] is not None]
        costs = [
            float(row["estimated_cost_usd"])
            for row in rows
            if row["estimated_cost_usd"] is not None
        ]
        correct = [float(row["consensus_grade"] == "correct") for row in agreed]
        partial = [float(row["consensus_grade"] == "partial") for row in agreed]
        evidence = [float(row["essential_evidence_recall"]) for row in rows]
        retrieved = [float(row["retrieved_essential_evidence_recall"]) for row in rows]
        recovery = [float(row["recovery_kind_correct"]) for row in rows]
        identity_seed = int.from_bytes(hashlib.sha256(f"{tier}:{strategy}".encode()).digest()[:4])
        tiers[tier][strategy] = {
            "cases": len(rows),
            "adjudicated_cases": len(agreed),
            "root_cause_accuracy": mean(correct) if correct else None,
            "root_cause_accuracy_ci95": _agent_ci(correct, seed=identity_seed) if correct else None,
            "partial_rate": mean(partial) if partial else None,
            "essential_evidence_recall": mean(evidence),
            "essential_evidence_recall_ci95": _agent_ci(evidence, seed=identity_seed + 1),
            "retrieved_essential_evidence_recall": mean(retrieved),
            "recovery_kind_accuracy": mean(recovery),
            "recovery_kind_accuracy_ci95": _agent_ci(recovery, seed=identity_seed + 2),
            "input_tokens": sum(int(row["input_tokens"]) for row in rows),
            "output_tokens": sum(int(row["output_tokens"]) for row in rows),
            "tool_calls": sum(int(row["tool_calls"]) for row in rows),
            "estimated_cost_usd": sum(costs) if costs else None,
            "latency_p50_ms": median(float(row["elapsed_ms"]) for row in rows),
            "latency_p95_ms": sorted(float(row["elapsed_ms"]) for row in rows)[
                min(len(rows) - 1, int(0.95 * len(rows)))
            ],
        }
    return {
        "schema_version": 1,
        "tiers": dict(tiers),
        "mean_cohen_kappa": mean(kappas) if kappas else None,
        "unresolved_disagreements": sorted(set(disagreements)),
        "combined_score": None,
    }
