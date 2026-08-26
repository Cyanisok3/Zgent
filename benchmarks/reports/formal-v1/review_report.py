from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from functools import partial
from pathlib import Path
from statistics import fmean
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
RUN_SET = "formal-v1"
PACKET_PATH = BASE_DIR / "review-packet-final.csv"
REVIEWER_C_PATH = BASE_DIR / "review-packet-c.csv"
REVIEWER_D_PATH = BASE_DIR / "review-packet-d.csv"
RETEST_C_PATH = BASE_DIR / "review-packet-adjudicated-c.csv"
RETEST_D_PATH = BASE_DIR / "review-packet-adjudicated-d.csv"
KEY_PATH = BASE_DIR / "review-key.json"
FORMAL_REPORT_PATH = BASE_DIR / "reports" / RUN_SET / "report.json"
OUTPUT_JSON = BASE_DIR / "reports" / RUN_SET / "human-review.json"
OUTPUT_MARKDOWN = BASE_DIR / "reports" / RUN_SET / "human-review.md"
BASELINES = ("full_native", "tail_32", "bm25_32", "cyan_selector_32")
BINARY_FIELDS = ("verdict_correct", "patch_intent_correct")
ORDINAL_FIELDS = (
    "category_score",
    "culprit_score",
    "mechanism_score",
    "evidence_support_score",
)
BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 20260825
REVIEW_PROTOCOL = (
    "two-reviewer, baseline-blind, rubric-calibrated, third-reviewer-adjudicated"
)


# 读取一个 UTF-8 JSON 文件
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 计算文件的 SHA-256
def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# 将一个评分字段解析为整数
def _score(row: dict[str, str], field: str) -> int:
    return int(row[field].strip())


# 计算两份匿名评分表的原始字段一致率
def _reviewer_agreement(left_path: Path, right_path: Path) -> dict[str, Any]:
    with left_path.open(newline="", encoding="utf-8") as left_handle:
        left_rows = list(csv.DictReader(left_handle))
    with right_path.open(newline="", encoding="utf-8") as right_handle:
        right_rows = list(csv.DictReader(right_handle))
    left = {row["item_id"]: row for row in left_rows}
    right = {row["item_id"]: row for row in right_rows}
    if len(left) != len(left_rows) or len(right) != len(right_rows):
        raise ValueError("reviewer packet contains duplicate item_id")
    if set(left) != set(right):
        raise ValueError("reviewer packets contain different items")

    fields = BINARY_FIELDS + ORDINAL_FIELDS
    item_count = len(left)
    exact_item_count = sum(
        all(left[item_id][field] == right[item_id][field] for field in fields)
        for item_id in left
    )
    field_agreement = {}
    disagreement_field_count = 0
    for field in fields:
        agreed = sum(left[item_id][field] == right[item_id][field] for item_id in left)
        disagreement_field_count += item_count - agreed
        field_agreement[field] = {
            "agreed": agreed,
            "total": item_count,
            "rate": round(agreed / item_count, 6),
        }
    return {
        "item_count": item_count,
        "exact_item_agreement": {
            "agreed": exact_item_count,
            "total": item_count,
            "rate": round(exact_item_count / item_count, 6),
        },
        "disagreement_item_count": item_count - exact_item_count,
        "disagreement_field_count": disagreement_field_count,
        "field_agreement": field_agreement,
    }


# 读取一个已解析记录的数值评分
def _field_value(record: dict[str, Any], name: str) -> float:
    return float(record[name])


# 判断一个已解析记录的评分是否等于目标值
def _field_equals(record: dict[str, Any], name: str, target: int) -> float:
    return float(record[name] == target)


# 判断一个已解析记录的评分是否达到目标值
def _field_at_least(record: dict[str, Any], name: str, target: int) -> float:
    return float(record[name] >= target)


# 对 case 级均值执行确定性 bootstrap
def _bootstrap_ci(values: list[float], label: str) -> list[float]:
    if not values:
        return [0.0, 0.0]
    if len(values) == 1:
        value = round(values[0], 6)
        return [value, value]
    seed_material = f"{BOOTSTRAP_SEED}:{label}".encode()
    seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big")
    rng = random.Random(seed)
    estimates = sorted(
        fmean(rng.choice(values) for _ in values)
        for _ in range(BOOTSTRAP_ITERATIONS)
    )
    lower = estimates[int(0.025 * (BOOTSTRAP_ITERATIONS - 1))]
    upper = estimates[int(0.975 * (BOOTSTRAP_ITERATIONS - 1))]
    return [round(lower, 6), round(upper, 6)]


# 按案例先聚合重复轮次再计算一个指标
def _case_values(
    records: list[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], float],
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        grouped[str(record["case_id"])].append(evaluator(record))
    return {
        case_id: fmean(grouped[case_id])
        for case_id in sorted(grouped)
    }


# 按案例先聚合重复轮次再计算一个指标
def _case_metric(
    records: list[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], float],
    label: str,
) -> dict[str, Any]:
    case_values = list(_case_values(records, evaluator).values())
    return {
        "macro": round(fmean(case_values), 6),
        "case_bootstrap_95_ci": _bootstrap_ci(case_values, label),
    }


# 计算两个 Baseline 的配对案例差值
def _paired_metric(
    candidate: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    evaluator: Callable[[dict[str, Any]], float],
    label: str,
) -> dict[str, Any]:
    candidate_values = _case_values(candidate, evaluator)
    comparator_values = _case_values(comparator, evaluator)
    if set(candidate_values) != set(comparator_values):
        raise ValueError(f"paired baselines have different cases: {label}")
    differences = [
        candidate_values[case_id] - comparator_values[case_id]
        for case_id in sorted(candidate_values)
    ]
    return {
        "macro_difference": round(fmean(differences), 6),
        "paired_case_bootstrap_95_ci": _bootstrap_ci(differences, label),
    }


# 汇总 Cyan Selector 相对另一个 Baseline 的配对差值
def _paired_baseline_metrics(
    candidate: list[dict[str, Any]],
    comparator: list[dict[str, Any]],
    label: str,
) -> dict[str, Any]:
    result: dict[str, Any] = {"binary": {}, "ordinal": {}}
    for field in BINARY_FIELDS:
        result["binary"][field] = _paired_metric(
            candidate,
            comparator,
            partial(_field_value, name=field),
            f"{label}:{field}",
        )
    for field in ORDINAL_FIELDS:
        result["ordinal"][field] = {
            "mean_score": _paired_metric(
                candidate,
                comparator,
                partial(_field_value, name=field),
                f"{label}:{field}:mean",
            ),
            "strict": _paired_metric(
                candidate,
                comparator,
                partial(_field_equals, name=field, target=2),
                f"{label}:{field}:strict",
            ),
            "acceptable": _paired_metric(
                candidate,
                comparator,
                partial(_field_at_least, name=field, target=1),
                f"{label}:{field}:acceptable",
            ),
        }
    return result


# 汇总故障记录的全部人工评分字段
def _fault_metrics(records: list[dict[str, Any]], label: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "case_count": len({str(record["case_id"]) for record in records}),
        "item_count": len(records),
        "binary": {},
        "ordinal": {},
    }
    for field in BINARY_FIELDS:
        item_correct = sum(int(record[field]) for record in records)
        result["binary"][field] = {
            "correct": item_correct,
            "total": len(records),
            **_case_metric(
                records,
                partial(_field_value, name=field),
                f"{label}:{field}",
            ),
        }
    for field in ORDINAL_FIELDS:
        result["ordinal"][field] = {
            "score_sum": sum(int(record[field]) for record in records),
            "score_max": len(records) * 2,
            "mean_score": _case_metric(
                records,
                partial(_field_value, name=field),
                f"{label}:{field}:mean",
            ),
            "strict": {
                "correct": sum(int(record[field]) == 2 for record in records),
                "total": len(records),
                **_case_metric(
                    records,
                    partial(_field_equals, name=field, target=2),
                    f"{label}:{field}:strict",
                ),
            },
            "acceptable": {
                "correct": sum(int(record[field]) >= 1 for record in records),
                "total": len(records),
                **_case_metric(
                    records,
                    partial(_field_at_least, name=field, target=1),
                    f"{label}:{field}:acceptable",
                ),
            },
        }
    return result


# 汇总正常 Control 的假阳性与弃权结果
def _control_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    full_abstention = sum(
        int(record["verdict_correct"]) == 1
        and int(record["patch_intent_correct"]) == 1
        and all(int(record[field]) == 2 for field in ORDINAL_FIELDS)
        for record in records
    )
    false_alarms = sum(int(record["verdict_correct"]) == 0 for record in records)
    unnecessary_patch = sum(
        int(record["patch_intent_correct"]) == 0 for record in records
    )
    total = len(records)
    return {
        "item_count": total,
        "correct_abstention": {
            "count": full_abstention,
            "total": total,
            "rate": round(full_abstention / total, 6) if total else 0.0,
        },
        "false_alarm": {
            "count": false_alarms,
            "total": total,
            "rate": round(false_alarms / total, 6) if total else 0.0,
        },
        "unnecessary_patch_intent": {
            "count": unnecessary_patch,
            "total": total,
            "rate": round(unnecessary_patch / total, 6) if total else 0.0,
        },
    }


# 汇总一个明确 split 范围内的 Control
def _control_scope(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "case_count": len({str(record["case_id"]) for record in records}),
        "item_count": len(records),
        "overall": _control_metrics(records),
        "by_baseline": {
            baseline: _control_metrics(
                [record for record in records if record["baseline"] == baseline]
            )
            for baseline in BASELINES
        },
    }


# 验证评分表并恢复私有案例信息
def _load_reviewed_records() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = list(csv.DictReader(PACKET_PATH.open(encoding="utf-8")))
    key_root = _read_json(KEY_PATH)
    key_map = key_root["items"]
    formal_report = _read_json(FORMAL_REPORT_PATH)
    report_lookup = {
        (
            str(record["case_id"]),
            str(record["variant"]),
            str(record["baseline"]),
            int(record["repeat"]),
        ): record
        for record in formal_report["diagnosis_records"]
    }
    if len(rows) != 144 or len(key_map) != 144:
        raise ValueError(
            f"expected 144 review items, got rows={len(rows)}, keys={len(key_map)}"
        )
    if {row["item_id"] for row in rows} != set(key_map):
        raise ValueError("review packet item IDs do not match private key")

    reviewed: list[dict[str, Any]] = []
    missing_notes: list[str] = []
    adjudication_count = 0
    for row in rows:
        item_id = row["item_id"]
        key = key_map[item_id]
        record_key = (
            str(key["case_id"]),
            str(key["variant"]),
            str(key["baseline"]),
            int(key["repeat"]),
        )
        try:
            source = report_lookup[record_key]
        except KeyError as error:
            raise ValueError(f"review item has no formal record: {record_key}") from error
        for field in BINARY_FIELDS:
            if row[field].strip() not in {"0", "1"}:
                raise ValueError(f"invalid {field} for {item_id}: {row[field]!r}")
        for field in ORDINAL_FIELDS:
            if row[field].strip() not in {"0", "1", "2"}:
                raise ValueError(f"invalid {field} for {item_id}: {row[field]!r}")
        adjudication = row["needs_adjudication"].strip().lower()
        if adjudication not in {"yes", "no"}:
            raise ValueError(
                f"invalid needs_adjudication for {item_id}: {adjudication!r}"
            )
        imperfect = any(_score(row, field) == 0 for field in BINARY_FIELDS) or any(
            _score(row, field) < 2 for field in ORDINAL_FIELDS
        )
        if (imperfect or adjudication == "yes") and not row["review_note"].strip():
            missing_notes.append(item_id)
        adjudication_count += adjudication == "yes"
        reviewed.append(
            {
                "item_id": item_id,
                "case_id": record_key[0],
                "variant": record_key[1],
                "baseline": record_key[2],
                "repeat": record_key[3],
                "split": source["split"],
                "framework": source["framework"],
                "fault_family": source["fault_family"],
                "failure_stage": source["failure_stage"],
                **{
                    field: _score(row, field)
                    for field in BINARY_FIELDS + ORDINAL_FIELDS
                },
                "needs_adjudication": adjudication,
                "review_note": row["review_note"].strip(),
            }
        )
    if missing_notes:
        raise ValueError(f"imperfect items missing review_note: {missing_notes}")
    validation = {
        "row_count": len(rows),
        "unique_item_count": len({row["item_id"] for row in rows}),
        "invalid_score_count": 0,
        "missing_required_note_count": 0,
        "needs_adjudication_count": adjudication_count,
        "review_note_count": sum(bool(row["review_note"].strip()) for row in rows),
    }
    return reviewed, validation


# 将比例渲染为一位小数百分比
def _percent(value: float) -> str:
    return f"{value * 100:.1f}%"


# 将严格与可接受比例渲染到同一表格单元
def _strict_acceptable(metrics: dict[str, Any], field: str) -> str:
    ordinal = metrics["ordinal"][field]
    return (
        f"{_percent(ordinal['strict']['macro'])} / "
        f"{_percent(ordinal['acceptable']['macro'])}"
    )


# 将百分点差值及区间渲染到一个表格单元
def _difference(metric: dict[str, Any]) -> str:
    lower, upper = metric["paired_case_bootstrap_95_ci"]
    point = metric["macro_difference"]
    return f"{point * 100:+.1f} [{lower * 100:+.1f}, {upper * 100:+.1f}] pp"


# 生成供人工阅读的 Markdown 报告
def _markdown_report(report: dict[str, Any]) -> str:
    initial_agreement = report["inter_rater_agreement"]["initial_full_packet"]
    retest_agreement = report["inter_rater_agreement"]["retest_disagreement_subset"]
    lines = [
        "# Cyan Benchmark Formal v1 — Human Review",
        "",
        f"> Review protocol: `{report['review_protocol']}`.",
        "",
        "## Review integrity",
        "",
        f"- Packet SHA-256: `{report['review_packet_sha256']}`",
        f"- Reviewed items: {report['validation']['row_count']} / 144",
        "- Composition: 108 frozen-test fault + 12 test Control + 24 dev Control",
        f"- Review notes: {report['validation']['review_note_count']}",
        f"- Unresolved adjudication items: {report['validation']['needs_adjudication_count']}",
        f"- Parser/schema failures retained as failures: {report['schema_error_count']}",
        "- Initial exact-item agreement: "
        f"{initial_agreement['exact_item_agreement']['agreed']}/"
        f"{initial_agreement['exact_item_agreement']['total']}",
        "- Retest exact-item agreement on the selected disagreement subset: "
        f"{retest_agreement['exact_item_agreement']['agreed']}/"
        f"{retest_agreement['exact_item_agreement']['total']}",
        "- Third-reviewer adjudication: "
        f"{retest_agreement['disagreement_field_count']} fields across "
        f"{retest_agreement['disagreement_item_count']} items",
        "",
        "### Inter-rater raw agreement",
        "",
        "The retest columns cover only the original disagreement subset and are not "
        "directly comparable with the full-packet rates.",
        "",
        "| Field | Initial full packet | Retest disagreement subset |",
        "|---|---:|---:|",
    ]
    for field in BINARY_FIELDS + ORDINAL_FIELDS:
        initial_field = initial_agreement["field_agreement"][field]
        retest_field = retest_agreement["field_agreement"][field]
        lines.append(
            f"| {field} | {initial_field['agreed']}/{initial_field['total']} "
            f"({_percent(initial_field['rate'])}) | "
            f"{retest_field['agreed']}/{retest_field['total']} "
            f"({_percent(retest_field['rate'])}) |"
        )
    lines.extend(
        [
        "",
        "## Frozen-test fault results",
        "",
        "Ordinal fields show `strict / acceptable`, where strict means score 2 "
        "and acceptable means score at least 1.",
        "Binary and ordinal rates are case-first macro averages over nine frozen test cases.",
        "",
        "| Baseline | Verdict | Category | Culprit | Mechanism | Evidence | Patch intent |",
        "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for baseline in BASELINES:
        metrics = report["fault_by_baseline"][baseline]
        lines.append(
            "| "
            + " | ".join(
                (
                    baseline,
                    _percent(metrics["binary"]["verdict_correct"]["macro"]),
                    _strict_acceptable(metrics, "category_score"),
                    _strict_acceptable(metrics, "culprit_score"),
                    _strict_acceptable(metrics, "mechanism_score"),
                    _strict_acceptable(metrics, "evidence_support_score"),
                    _percent(
                        metrics["binary"]["patch_intent_correct"]["macro"]
                    ),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Paired Cyan Selector differences",
            "",
            "Positive values favor `cyan_selector_32`; brackets are paired "
            "case-bootstrap 95% intervals.",
            "",
            "| Comparator | Verdict | Category strict | Culprit strict | "
            "Mechanism strict | Evidence strict | Patch intent |",
            "|---|---:|---:|---:|---:|---:|---:|",
        )
    )
    for comparator in ("full_native", "tail_32", "bm25_32"):
        metrics = report["paired_cyan_selector_vs"][comparator]
        lines.append(
            "| "
            + " | ".join(
                (
                    comparator,
                    _difference(metrics["binary"]["verdict_correct"]),
                    _difference(metrics["ordinal"]["category_score"]["strict"]),
                    _difference(metrics["ordinal"]["culprit_score"]["strict"]),
                    _difference(metrics["ordinal"]["mechanism_score"]["strict"]),
                    _difference(
                        metrics["ordinal"]["evidence_support_score"]["strict"]
                    ),
                    _difference(metrics["binary"]["patch_intent_correct"]),
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Results by failure stage",
            "",
            "| Stage | Baseline | Cases | Verdict | Mechanism strict | "
            "Evidence strict | Patch intent |",
            "|---|---|---:|---:|---:|---:|---:|",
        )
    )
    for stage in ("startup", "mid_run", "finalization"):
        for baseline in BASELINES:
            metrics = report["fault_by_stage"][stage][baseline]
            lines.append(
                "| "
                + " | ".join(
                    (
                        stage,
                        baseline,
                        str(metrics["case_count"]),
                        _percent(
                            metrics["binary"]["verdict_correct"]["macro"]
                        ),
                        _percent(
                            metrics["ordinal"]["mechanism_score"]["strict"][
                                "macro"
                            ]
                        ),
                        _percent(
                            metrics["ordinal"]["evidence_support_score"][
                                "strict"
                            ]["macro"]
                        ),
                        _percent(
                            metrics["binary"]["patch_intent_correct"]["macro"]
                        ),
                    )
                )
                + " |"
            )
    lines.extend(
        (
            "",
            "## Normal controls",
            "",
            "Controls are descriptive only and do not enter the 108-item frozen-test fault result.",
            "The test split contains one Control case; two additional Control cases come from dev.",
            "",
            "### Test-split Control",
            "",
            "| Baseline | N | Correct abstention | False alarm | Unnecessary patch intent |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for baseline in BASELINES:
        metrics = report["controls"]["test"]["by_baseline"][baseline]
        lines.append(
            "| "
            + " | ".join(
                (
                    baseline,
                    str(metrics["item_count"]),
                    f"{metrics['correct_abstention']['count']}/{metrics['correct_abstention']['total']}",
                    f"{metrics['false_alarm']['count']}/{metrics['false_alarm']['total']}",
                    f"{metrics['unnecessary_patch_intent']['count']}/{metrics['unnecessary_patch_intent']['total']}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "### All cross-split Controls",
            "",
            "| Baseline | N | Correct abstention | False alarm | Unnecessary patch intent |",
            "|---|---:|---:|---:|---:|",
        )
    )
    for baseline in BASELINES:
        metrics = report["controls"]["all_splits"]["by_baseline"][baseline]
        lines.append(
            "| "
            + " | ".join(
                (
                    baseline,
                    str(metrics["item_count"]),
                    f"{metrics['correct_abstention']['count']}/{metrics['correct_abstention']['total']}",
                    f"{metrics['false_alarm']['count']}/{metrics['false_alarm']['total']}",
                    f"{metrics['unnecessary_patch_intent']['count']}/{metrics['unnecessary_patch_intent']['total']}",
                )
            )
            + " |"
        )
    lines.extend(
        (
            "",
            "## Interpretation boundaries",
            "",
            "- No weighted overall score is constructed.",
            "- Deterministic case-level bootstrap 95% intervals for every metric "
            "are stored in `human-review.json`.",
            "- Inter-rater reliability is reported as raw agreement only; no "
            "chance-corrected or weighted coefficient is claimed.",
            "- Retest agreement is conditional on the selected initial-disagreement "
            "subset and must not be compared directly with full-packet agreement.",
            "- Public upstream issues may be present in model training data, so the "
            "benchmark is not contamination-free.",
            "- Results cover stable CPU/MPS non-zero exits, not CUDA/NCCL, OOM, "
            "hangs or silent convergence failures.",
            "",
        )
    )
    return "\n".join(lines)


# 解盲并生成正式人工评审报告
def main() -> None:
    reviewed, validation = _load_reviewed_records()
    fault_records = [record for record in reviewed if record["variant"] == "buggy"]
    control_records = [
        record for record in reviewed if record["variant"] == "control"
    ]
    if len(fault_records) != 108 or len(control_records) != 36:
        raise ValueError(
            "expected 108 fault and 36 control records, "
            f"got {len(fault_records)} and {len(control_records)}"
        )
    fault_by_baseline = {
        baseline: _fault_metrics(
            [record for record in fault_records if record["baseline"] == baseline],
            f"baseline:{baseline}",
        )
        for baseline in BASELINES
    }
    fault_by_stage = {
        stage: {
            baseline: _fault_metrics(
                [
                    record
                    for record in fault_records
                    if record["failure_stage"] == stage
                    and record["baseline"] == baseline
                ],
                f"stage:{stage}:{baseline}",
            )
            for baseline in BASELINES
        }
        for stage in ("startup", "mid_run", "finalization")
    }
    fault_by_case = {
        case_id: {
            baseline: _fault_metrics(
                [
                    record
                    for record in fault_records
                    if record["case_id"] == case_id
                    and record["baseline"] == baseline
                ],
                f"case:{case_id}:{baseline}",
            )
            for baseline in BASELINES
        }
        for case_id in sorted({str(record["case_id"]) for record in fault_records})
    }
    test_control_records = [
        record for record in control_records if record["split"] == "test"
    ]
    dev_control_records = [
        record for record in control_records if record["split"] == "dev"
    ]
    if len(test_control_records) != 12 or len(dev_control_records) != 24:
        raise ValueError(
            "expected 12 test and 24 dev Control records, "
            f"got {len(test_control_records)} and {len(dev_control_records)}"
        )
    cyan_records = [
        record
        for record in fault_records
        if record["baseline"] == "cyan_selector_32"
    ]
    paired_cyan_selector_vs = {
        comparator: _paired_baseline_metrics(
            cyan_records,
            [
                record
                for record in fault_records
                if record["baseline"] == comparator
            ],
            f"cyan_selector_32-vs-{comparator}",
        )
        for comparator in ("full_native", "tail_32", "bm25_32")
    }
    diagnosis_statuses = Counter(
        str(record.get("status"))
        for record in _read_json(FORMAL_REPORT_PATH)["diagnosis_records"]
        if (
            (record["variant"] == "buggy" and record["split"] == "test")
            or record["variant"] == "control"
        )
        and record["baseline"] in BASELINES
    )
    report = {
        "schema_version": 1,
        "run_set": RUN_SET,
        "review_protocol": REVIEW_PROTOCOL,
        "publication_status": "internal-adjudicated-review",
        "inter_rater_agreement": {
            "metric": "raw exact agreement",
            "initial_full_packet": _reviewer_agreement(
                REVIEWER_C_PATH,
                REVIEWER_D_PATH,
            ),
            "retest_disagreement_subset": _reviewer_agreement(
                RETEST_C_PATH,
                RETEST_D_PATH,
            ),
        },
        "review_packet_sha256": _sha256(PACKET_PATH),
        "review_key_packet_version": _read_json(KEY_PATH)["packet_version"],
        "bootstrap": {
            "unit": "case",
            "iterations": BOOTSTRAP_ITERATIONS,
            "seed": BOOTSTRAP_SEED,
            "confidence": 0.95,
        },
        "validation": validation,
        "counts": {
            "review_items": len(reviewed),
            "fault_items": len(fault_records),
            "fault_test_items": len(fault_records),
            "control_items": len(control_records),
            "control_test_items": len(test_control_records),
            "control_dev_items": len(dev_control_records),
            "frozen_test_cases": len(
                {record["case_id"] for record in fault_records}
            ),
            "control_test_cases": len(
                {record["case_id"] for record in test_control_records}
            ),
            "control_dev_cases": len(
                {record["case_id"] for record in dev_control_records}
            ),
        },
        "schema_error_count": diagnosis_statuses.get("schema_error", 0),
        "fault_by_baseline": fault_by_baseline,
        "fault_by_stage": fault_by_stage,
        "fault_by_case": fault_by_case,
        "paired_cyan_selector_vs": paired_cyan_selector_vs,
        "controls": {
            "test": _control_scope(test_control_records),
            "dev": _control_scope(dev_control_records),
            "all_splits": _control_scope(control_records),
        },
        "limitations": [
            "inter-rater reliability uses raw agreement, not chance-corrected kappa",
            "retest agreement is conditional on the initial-disagreement subset",
            "public upstream issue contamination cannot be excluded",
            "stdout/stderr interleaving is unavailable",
            "single-machine CPU/MPS scope",
            "no CUDA/NCCL, OOM, hang, silent NaN, or convergence-regression cases",
        ],
    }
    OUTPUT_JSON.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    OUTPUT_MARKDOWN.write_text(_markdown_report(report), encoding="utf-8")
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MARKDOWN}")


if __name__ == "__main__":
    main()
