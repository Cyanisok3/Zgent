from __future__ import annotations

import json
import random
import statistics
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cyan_bench.cases import LoadedCase, discover_cases
from cyan_bench.diagnosis import (
    DIAGNOSIS_MAX_OUTPUT_TOKENS,
    DIAGNOSIS_REASONING_EFFORT,
    DIAGNOSIS_TEMPERATURE,
)
from cyan_bench.models import (
    DiagnosisRunArtifact,
    IncidentBenchmarkArtifact,
    ProcessCapture,
    SelectionArtifact,
)
from cyan_bench.paths import BenchmarkPaths
from cyan_bench.scoring import score_selection


# 返回当前 UTC 时间字符串
def _now() -> str:
    return datetime.now(UTC).isoformat()


# 计算 case-level bootstrap 均值的固定 95% 区间
def _bootstrap_ci(case_values: dict[str, float], samples: int = 2000) -> list[float] | None:
    if not case_values:
        return None
    values = list(case_values.values())
    generator = random.Random(20260824)
    means = [
        statistics.fmean(generator.choice(values) for _ in values)
        for _ in range(samples)
    ]
    means.sort()
    return [round(means[int(samples * 0.025)], 6), round(means[int(samples * 0.975)], 6)]


# 对一组逐轮记录先按 case 聚合再计算 macro average
def _macro(records: list[dict[str, Any]], key: str) -> dict[str, object]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            grouped[str(record["case_id"])].append(float(value))
        elif isinstance(value, bool):
            grouped[str(record["case_id"])].append(float(value))
    case_values = {case: statistics.fmean(values) for case, values in grouped.items()}
    return {
        "cases": len(case_values),
        "macro_mean": round(statistics.fmean(case_values.values()), 6) if case_values else None,
        "bootstrap_95_ci": _bootstrap_ci(case_values),
    }


# 按 baseline 汇总检索或诊断指标
def _by_baseline(
    records: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    summary: dict[str, dict[str, object]] = {}
    for baseline in sorted({str(item["baseline"]) for item in records}):
        items = [item for item in records if item["baseline"] == baseline]
        summary[baseline] = {metric: _macro(items, metric) for metric in metrics}
    return summary


# 按人工标注的故障阶段生成描述性分组结果
def _by_failure_stage(
    records: list[dict[str, Any]],
    metrics: tuple[str, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    for stage in ("startup", "mid_run", "finalization"):
        items = [item for item in records if item.get("failure_stage") == stage]
        summary[stage] = _by_baseline(items, metrics)
    return summary


# 按任意已冻结维度生成 baseline 分组结果
def _by_dimension(
    records: list[dict[str, Any]],
    dimension: str,
    metrics: tuple[str, ...],
) -> dict[str, dict[str, dict[str, object]]]:
    summary: dict[str, dict[str, dict[str, object]]] = {}
    values = sorted({str(item[dimension]) for item in records if item.get(dimension) is not None})
    for value in values:
        items = [item for item in records if item.get(dimension) == value]
        summary[value] = _by_baseline(items, metrics)
    return summary


# 将原始日志字节数归入固定的描述性区间
def _log_length_bucket(total_bytes: int) -> str:
    if total_bytes >= 40 * 1024:
        return "long_ge_40kib"
    if total_bytes >= 10 * 1024:
        return "medium_10_to_40kib"
    return "short_lt_10kib"


# 汇总一次正式运行的可计费 token、工具调用与总耗时
def _usage_summary(records: list[dict[str, Any]]) -> dict[str, int | float]:
    return {
        "observations": len(records),
        "input_tokens": sum(int(item.get("input_tokens") or 0) for item in records),
        "cache_read_input_tokens": sum(
            int(item.get("cache_read_input_tokens") or 0) for item in records
        ),
        "output_tokens": sum(int(item.get("output_tokens") or 0) for item in records),
        "tool_calls": sum(int(item.get("tool_calls") or 0) for item in records),
        "duration_seconds": round(
            sum(float(item.get("duration_seconds") or 0) for item in records), 6
        ),
    }


# 用预标注关键词对结构化诊断槽位做可复现的最低限度评分
def _score_diagnosis(case: LoadedCase, artifact: DiagnosisRunArtifact) -> dict[str, object]:
    answer = artifact.answer
    if artifact.status != "success" or answer is None:
        return {
            "category_correct": False,
            "culprit_hit": False,
            "causal_mechanism_hit": False,
            "correct_abstention": False,
            "false_alarm": False,
            "unnecessary_patch_intent": False,
        }
    if artifact.is_control:
        return {
            "category_correct": False,
            "culprit_hit": False,
            "causal_mechanism_hit": False,
            "correct_abstention": answer.verdict == "no_fault",
            "false_alarm": answer.verdict == "fault",
            "unnecessary_patch_intent": answer.patch_recommended,
        }
    diagnosis = answer.diagnosis or {}
    serialized = json.dumps(diagnosis, ensure_ascii=False).lower()
    category = str(diagnosis.get("category", "")).lower()
    return {
        "category_correct": category == case.expected.diagnosis.category.lower(),
        "culprit_hit": any(item.lower() in serialized for item in case.expected.diagnosis.culprit),
        "causal_mechanism_hit": any(
            item.lower() in serialized for item in case.expected.diagnosis.causal_mechanism
        ),
        "correct_abstention": False,
        "false_alarm": False,
        "unnecessary_patch_intent": False,
    }


# 汇总一个 run-set 的检索、诊断、Control 与阶段指标
def build_report(run_set: str, paths: BenchmarkPaths) -> dict[str, object]:
    run_root = paths.artifacts / "run-sets" / run_set
    cases = {case.manifest.id: case for case in discover_cases(paths.cases)}
    selection_records: list[dict[str, Any]] = []
    diagnosis_records: list[dict[str, Any]] = []
    for selection_path in sorted(run_root.glob("diagnosis/*/*/*/*/selection.json")):
        selection = SelectionArtifact.model_validate_json(
            selection_path.read_text(encoding="utf-8")
        )
        case = cases[selection.case_id]
        variant = selection_path.parts[-4]
        capture_dir = (
            paths.artifacts
            / "captures"
            / selection.case_id
            / variant
            / str(selection.repeat)
        )
        record = (
            score_selection(selection, capture_dir / "gold-ranges.json")
            if variant == "buggy"
            else {
                "case_id": selection.case_id,
                "baseline": selection.baseline,
                "repeat": selection.repeat,
            }
        )
        record.update(
            {
                "split": case.manifest.split,
                "failure_stage": case.manifest.failure_stage if variant == "buggy" else None,
                "framework": case.manifest.framework,
                "fault_family": case.manifest.fault_family,
                "log_bytes": selection.scanned_bytes,
                "log_length_bucket": _log_length_bucket(selection.scanned_bytes),
                "variant": variant,
            }
        )
        selection_records.append(record)
        diagnosis_path = selection_path.parent / "diagnosis.json"
        if diagnosis_path.is_file():
            diagnosis = DiagnosisRunArtifact.model_validate_json(
                diagnosis_path.read_text(encoding="utf-8")
            )
            diagnosis_record = {
                "case_id": diagnosis.case_id,
                "baseline": diagnosis.baseline,
                "repeat": diagnosis.repeat,
                "status": diagnosis.status,
                "split": case.manifest.split,
                "failure_stage": case.manifest.failure_stage if variant == "buggy" else None,
                "framework": case.manifest.framework,
                "fault_family": case.manifest.fault_family,
                "log_bytes": selection.scanned_bytes,
                "log_length_bucket": _log_length_bucket(selection.scanned_bytes),
                "variant": variant,
                "input_tokens": diagnosis.input_tokens,
                "cache_read_input_tokens": diagnosis.cache_read_input_tokens,
                "output_tokens": diagnosis.output_tokens,
                "duration_seconds": diagnosis.duration_seconds,
                **_score_diagnosis(case, diagnosis),
            }
            diagnosis_records.append(diagnosis_record)
    frozen = [
        item
        for item in diagnosis_records
        if item["split"] == "test" and item["variant"] == "buggy"
    ]
    frozen_selections = [
        item
        for item in selection_records
        if item["split"] == "test" and item["variant"] == "buggy"
    ]
    controls = [item for item in diagnosis_records if item["variant"] == "control"]
    selection_metrics = (
        "required_group_recall",
        "gold_evidence_hit",
        "selection_ratio",
        "selector_latency_seconds",
    )
    diagnosis_metrics = ("category_correct", "culprit_hit", "causal_mechanism_hit")
    selection_by_baseline = _by_baseline(frozen_selections, selection_metrics)
    by_baseline = _by_baseline(frozen, diagnosis_metrics)
    control_summary: dict[str, dict[str, object]] = {}
    for baseline in sorted({str(item["baseline"]) for item in controls}):
        items = [item for item in controls if item["baseline"] == baseline]
        control_summary[baseline] = {
            "observations": len(items),
            "correct_abstentions": sum(bool(item["correct_abstention"]) for item in items),
            "false_alarms": sum(bool(item["false_alarm"]) for item in items),
            "unnecessary_patch_intents": sum(
                bool(item["unnecessary_patch_intent"]) for item in items
            ),
            "correct_abstention_rate": _macro(items, "correct_abstention"),
            "false_alarm_rate": _macro(items, "false_alarm"),
            "unnecessary_patch_intent_rate": _macro(
                items, "unnecessary_patch_intent"
            ),
        }
    incident_records: list[dict[str, object]] = []
    for incident_path in sorted(run_root.glob("incident/*/*/*/incident-benchmark.json")):
        artifact = IncidentBenchmarkArtifact.model_validate_json(
            incident_path.read_text(encoding="utf-8")
        )
        case = cases[artifact.case_id]
        variant = "control" if artifact.is_control else "buggy"
        capture_path = (
            paths.artifacts
            / "captures"
            / artifact.case_id
            / variant
            / str(artifact.repeat)
            / "process.json"
        )
        log_bytes = 0
        if capture_path.is_file():
            capture = ProcessCapture.model_validate_json(
                capture_path.read_text(encoding="utf-8")
            )
            log_bytes = capture.stdout_bytes + capture.stderr_bytes
        incident_records.append(
            {
                **artifact.model_dump(mode="json"),
                "baseline": "cyan_incident",
                "split": case.manifest.split,
                "failure_stage": None if artifact.is_control else case.manifest.failure_stage,
                "framework": case.manifest.framework,
                "fault_family": case.manifest.fault_family,
                "log_bytes": log_bytes,
                "log_length_bucket": _log_length_bucket(log_bytes),
                "patchable": case.manifest.patchable,
            }
        )
    valid_incidents = [item for item in incident_records if item["error"] is None]
    incident_test = [
        item
        for item in valid_incidents
        if item["split"] == "test" and not item["is_control"]
    ]
    incident_controls = [item for item in valid_incidents if item["is_control"]]
    fault_incidents = [item for item in incident_records if not item["is_control"]]
    expected_fault_incidents = len(cases) * 3
    expected_test_incidents = len(discover_cases(paths.cases, "test")) * 3
    return {
        "schema_version": 1,
        "run_set": run_set,
        "generated_at": _now(),
        "main_score_scope": "frozen test cases only; no weighted composite",
        "diagnosis_protocol": {
            "models_requested": sorted(
                {
                    str(item.model_requested)
                    for item in (
                        DiagnosisRunArtifact.model_validate_json(path.read_text())
                        for path in run_root.glob("diagnosis/*/*/*/*/diagnosis.json")
                    )
                }
            ),
            "models_resolved": sorted(
                {
                    str(item.model_resolved)
                    for item in (
                        DiagnosisRunArtifact.model_validate_json(path.read_text())
                        for path in run_root.glob("diagnosis/*/*/*/*/diagnosis.json")
                    )
                    if item.model_resolved is not None
                }
            ),
            "temperature": DIAGNOSIS_TEMPERATURE,
            "reasoning_effort": DIAGNOSIS_REASONING_EFFORT,
            "max_output_tokens": DIAGNOSIS_MAX_OUTPUT_TOKENS,
        },
        "selection_records": selection_records,
        "selection_test_macro_by_baseline": selection_by_baseline,
        "selection_test_by_failure_stage": _by_failure_stage(
            frozen_selections, selection_metrics
        ),
        "selection_test_by_framework": _by_dimension(
            frozen_selections, "framework", selection_metrics
        ),
        "selection_test_by_fault_family": _by_dimension(
            frozen_selections, "fault_family", selection_metrics
        ),
        "selection_test_by_log_length": _by_dimension(
            frozen_selections, "log_length_bucket", selection_metrics
        ),
        "diagnosis_records": diagnosis_records,
        "test_macro_by_baseline": by_baseline,
        "diagnosis_test_by_failure_stage": _by_failure_stage(frozen, diagnosis_metrics),
        "diagnosis_test_by_framework": _by_dimension(
            frozen, "framework", diagnosis_metrics
        ),
        "diagnosis_test_by_fault_family": _by_dimension(
            frozen, "fault_family", diagnosis_metrics
        ),
        "diagnosis_test_by_log_length": _by_dimension(
            frozen, "log_length_bucket", diagnosis_metrics
        ),
        "controls": control_summary,
        "incident_records": incident_records,
        "incident_completeness": {
            "fault_expected": expected_fault_incidents,
            "fault_artifacts": len(fault_incidents),
            "fault_valid": sum(item["error"] is None for item in fault_incidents),
            "fault_infrastructure_errors": sum(
                item["error"] is not None for item in fault_incidents
            ),
            "test_expected": expected_test_incidents,
            "test_valid": len(incident_test),
        },
        "incident_test": {
            "resolved_rate": _macro(incident_test, "resolved"),
            "proposal_valid_rate": _macro(incident_test, "proposal_valid"),
            "unsafe_proposal_rate": _macro(incident_test, "unsafe_proposal"),
        },
        "incident_test_by_failure_stage": _by_dimension(
            incident_test,
            "failure_stage",
            ("resolved", "proposal_valid", "unsafe_proposal"),
        ),
        "incident_test_by_framework": _by_dimension(
            incident_test,
            "framework",
            ("resolved", "proposal_valid", "unsafe_proposal"),
        ),
        "incident_test_by_fault_family": _by_dimension(
            incident_test,
            "fault_family",
            ("resolved", "proposal_valid", "unsafe_proposal"),
        ),
        "incident_test_by_log_length": _by_dimension(
            incident_test,
            "log_length_bucket",
            ("resolved", "proposal_valid", "unsafe_proposal"),
        ),
        "usage": {
            "diagnosis_all": _usage_summary(diagnosis_records),
            "incident_fault_all": _usage_summary(
                [item for item in valid_incidents if not item["is_control"]]
            ),
            "incident_test": _usage_summary(incident_test),
        },
        "product_controls": {
            "observations": len(incident_controls),
            "spurious_incidents": sum(
                bool(item["spurious_incident"]) for item in incident_controls
            ),
            "spurious_incident_rate": _macro(
                incident_controls, "spurious_incident"
            ),
        },
        "limitations": [
            "Public upstream issues may be present in model training data.",
            "stdout and stderr are persisted separately; full-native uses stdout then stderr.",
            "v1 covers one local CPU/MPS machine and stable non-zero exits only.",
            "Category and causal-mechanism keyword scores are strict lower bounds "
            "and require separate blinded human review.",
        ],
    }


# 将 JSON 报告渲染为可提交的精简 Markdown 摘要
def write_report(run_set: str, paths: BenchmarkPaths) -> tuple[Path, Path]:
    report = build_report(run_set, paths)
    output_dir = paths.root / "reports" / run_set
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "report.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# Cyan Incident Benchmark — {run_set}",
        "",
        "主表仅包含冻结 test；没有加权总分。",
        "",
        "## Retrieval macro (frozen test)",
        "",
        "| Baseline | Required-group recall | Gold hit | Selection ratio | Latency (s) |",
        "|---|---:|---:|---:|---:|",
    ]
    selection_macro = report["selection_test_macro_by_baseline"]
    assert isinstance(selection_macro, dict)
    for baseline, values in selection_macro.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {baseline} | {values['required_group_recall']['macro_mean']} | "
            f"{values['gold_evidence_hit']['macro_mean']} | "
            f"{values['selection_ratio']['macro_mean']} | "
            f"{values['selector_latency_seconds']['macro_mean']} |"
        )
    lines.extend(
        [
            "",
            "## Diagnosis macro (frozen test)",
            "",
            "| Baseline | Category | Culprit | Mechanism |",
            "|---|---:|---:|---:|",
        ]
    )
    macro = report["test_macro_by_baseline"]
    assert isinstance(macro, dict)
    for baseline, values in macro.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {baseline} | {values['category_correct']['macro_mean']} | "
            f"{values['culprit_hit']['macro_mean']} | "
            f"{values['causal_mechanism_hit']['macro_mean']} |"
        )
    incident = report["incident_test"]
    assert isinstance(incident, dict)
    lines.extend(
        [
            "",
            "## Cyan Incident loop (frozen test)",
            "",
            "| Metric | Macro mean | Case-level bootstrap 95% CI |",
            "|---|---:|---:|",
        ]
    )
    for metric in ("resolved_rate", "proposal_valid_rate", "unsafe_proposal_rate"):
        values = incident[metric]
        assert isinstance(values, dict)
        lines.append(
            f"| {metric} | {values['macro_mean']} | {values['bootstrap_95_ci']} |"
        )
    lines.extend(
        [
            "",
            "### Incident results by failure stage",
            "",
            "| Stage | Cases | Resolved | Proposal valid | Unsafe proposal |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    incident_stages = report["incident_test_by_failure_stage"]
    assert isinstance(incident_stages, dict)
    for stage in ("startup", "mid_run", "finalization"):
        baseline = incident_stages[stage]["cyan_incident"]
        lines.append(
            f"| {stage} | {baseline['resolved']['cases']} | "
            f"{baseline['resolved']['macro_mean']} | "
            f"{baseline['proposal_valid']['macro_mean']} | "
            f"{baseline['unsafe_proposal']['macro_mean']} |"
        )
    lines.extend(
        [
            "",
            "Automatic diagnosis slot scores are strict keyword lower bounds; "
            "causal mechanisms require blinded review.",
            "",
            "## Forced-diagnosis controls",
            "",
            "| Baseline | Observations | Correct abstentions | False alarms | Patch intents |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    controls = report["controls"]
    assert isinstance(controls, dict)
    for baseline, values in controls.items():
        assert isinstance(values, dict)
        lines.append(
            f"| {baseline} | {values['observations']} | "
            f"{values['correct_abstentions']} | {values['false_alarms']} | "
            f"{values['unnecessary_patch_intents']} |"
        )
    lines.extend(
        [
            "",
            "## Retrieval by failure stage",
            "",
            "| Stage | Baseline | Cases | Required-group recall | Gold hit |",
            "|---|---|---:|---:|---:|",
        ]
    )
    stage_selection = report["selection_test_by_failure_stage"]
    assert isinstance(stage_selection, dict)
    for stage, baselines in stage_selection.items():
        assert isinstance(baselines, dict)
        for baseline, values in baselines.items():
            assert isinstance(values, dict)
            recall = values["required_group_recall"]
            hit = values["gold_evidence_hit"]
            lines.append(
                f"| {stage} | {baseline} | {recall['cases']} | "
                f"{recall['macro_mean']} | {hit['macro_mean']} |"
            )
    lines.extend(
        [
            "",
            "## Frozen test cases",
            "",
            "| Case | Stage | Framework | Resolved | Proposal valid | Unsafe proposal |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    cases = {case.manifest.id: case for case in discover_cases(paths.cases, "test")}
    incident_records = report["incident_records"]
    assert isinstance(incident_records, list)
    for case_id, case in sorted(cases.items()):
        records = [
            item
            for item in incident_records
            if item["case_id"] == case_id
            and item["error"] is None
            and not item["is_control"]
        ]
        resolved = statistics.fmean(bool(item["resolved"]) for item in records)
        proposal = statistics.fmean(bool(item["proposal_valid"]) for item in records)
        unsafe = statistics.fmean(bool(item["unsafe_proposal"]) for item in records)
        lines.append(
            f"| {case_id} | {case.manifest.failure_stage} | {case.manifest.framework} | "
            f"{resolved:.6f} | {proposal:.6f} | {unsafe:.6f} |"
        )
    product_controls = report["product_controls"]
    assert isinstance(product_controls, dict)
    completeness = report["incident_completeness"]
    assert isinstance(completeness, dict)
    lines.extend(
        [
            "",
            "## Incident completeness",
            "",
            f"Valid frozen-test observations: {completeness['test_valid']}/"
            f"{completeness['test_expected']}; valid fault observations: "
            f"{completeness['fault_valid']}/{completeness['fault_expected']}; "
            f"infrastructure errors: {completeness['fault_infrastructure_errors']}.",
            "",
            "## Product controls",
            "",
            f"Successful observations: {product_controls['observations']}; "
            f"spurious incidents: {product_controls['spurious_incidents']}.",
            "",
            "## Formal resource usage",
            "",
            "| Track | Observations | Input tokens | Output tokens | Tool calls | Duration (s) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    usage = report["usage"]
    assert isinstance(usage, dict)
    for track in ("diagnosis_all", "incident_fault_all", "incident_test"):
        values = usage[track]
        lines.append(
            f"| {track} | {values['observations']} | {values['input_tokens']} | "
            f"{values['output_tokens']} | {values['tool_calls']} | "
            f"{values['duration_seconds']} |"
        )
    lines.extend(["", "## Limitations", ""])
    limitations = report["limitations"]
    assert isinstance(limitations, list)
    lines.extend(f"- {item}" for item in limitations)
    markdown_path = output_dir / "report.md"
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, markdown_path
