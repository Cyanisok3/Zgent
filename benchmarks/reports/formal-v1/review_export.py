from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tomllib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

RUN_SET = "formal-v1"
PACKET_VERSION = 1
SHUFFLE_SEED = 20260825
CALIBRATION_SEED = 20260826
BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "cases"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
DIAGNOSIS_DIR = ARTIFACTS_DIR / "run-sets" / RUN_SET / "diagnosis"
REPORT_PATH = BASE_DIR / "reports" / RUN_SET / "report.json"
OUTPUT_PACKET = BASE_DIR / "review-packet.csv"
OUTPUT_KEY = BASE_DIR / "review-key.json"
CALIBRATION_PACKET = BASE_DIR / "review-calibration.csv"
CALIBRATION_KEY = BASE_DIR / "review-calibration-key.json"
EXCLUDED_BASELINES = {"oracle_32"}
BASELINE_ORDER = ("bm25_32", "cyan_selector_32", "full_native", "tail_32")


# 读取一个 UTF-8 JSON 文件
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 选择冻结测试故障与全部三个正常 Control
def _formal_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    records = [
        record
        for record in report["diagnosis_records"]
        if record["baseline"] not in EXCLUDED_BASELINES
        and (
            (record["variant"] == "buggy" and record["split"] == "test")
            or record["variant"] == "control"
        )
    ]
    return sorted(
        records,
        key=lambda item: (
            item["case_id"],
            item["variant"],
            item["baseline"],
            int(item["repeat"]),
        ),
    )


# 从六个开发案例中各取两项并平衡 Baseline 与重复轮次
def _calibration_records(report: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = {
        (
            str(record["case_id"]),
            str(record["baseline"]),
            int(record["repeat"]),
        ): record
        for record in report["diagnosis_records"]
        if record["variant"] == "buggy"
        and record["split"] == "dev"
        and record["baseline"] in BASELINE_ORDER
    }
    case_ids = sorted({key[0] for key in candidates})
    if len(case_ids) != 6:
        raise ValueError(f"expected 6 dev cases, got {len(case_ids)}")
    random.Random(CALIBRATION_SEED).shuffle(case_ids)
    baseline_slots = BASELINE_ORDER * 3
    repeat_slots = (1, 2, 3) * 4
    records = []
    for index, case_id in enumerate(case_ids):
        for slot in (index * 2, index * 2 + 1):
            key = (case_id, baseline_slots[slot], repeat_slots[slot])
            try:
                records.append(candidates[key])
            except KeyError as error:
                raise ValueError(f"missing calibration record: {key}") from error
    dimensions = {
        "case": Counter(str(record["case_id"]) for record in records),
        "stage": Counter(str(record["failure_stage"]) for record in records),
        "baseline": Counter(str(record["baseline"]) for record in records),
        "repeat": Counter(int(record["repeat"]) for record in records),
    }
    expected = {
        "case": Counter({case_id: 2 for case_id in case_ids}),
        "stage": Counter({"startup": 4, "mid_run": 4, "finalization": 4}),
        "baseline": Counter({baseline: 3 for baseline in BASELINE_ORDER}),
        "repeat": Counter({1: 4, 2: 4, 3: 4}),
    }
    if dimensions != expected:
        raise ValueError(f"unbalanced calibration sample: {dimensions}")
    return records


# 读取所有案例清单和 Gold 标注
def _load_cases() -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    manifests: dict[str, dict[str, Any]] = {}
    expected: dict[str, dict[str, Any]] = {}
    for case_dir in sorted(CASES_DIR.iterdir()):
        manifest_path = case_dir / "case.toml"
        expected_path = case_dir / "expected.json"
        if not manifest_path.is_file() or not expected_path.is_file():
            continue
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        case_id = str(manifest["id"])
        manifests[case_id] = manifest
        expected[case_id] = _read_json(expected_path)
    return manifests, expected


# 将一组可接受 Gold 表达渲染为明确的任一命中语义
def _acceptable(values: list[str]) -> str:
    return "ACCEPTABLE ALTERNATIVES — 任一等价表达即可：\n" + "\n".join(
        f"- {value}" for value in values
    )


# 从不可变日志中提取一个证据引用所在的完整行
def _evidence_line(payload: bytes, start: int, end: int) -> str:
    left = payload.rfind(b"\n", 0, start) + 1
    right = payload.find(b"\n", end)
    if right < 0:
        right = len(payload)
    line = payload[left:right].decode("utf-8", errors="replace")
    if len(line) <= 1200:
        return line
    relative_start = max(0, start - left)
    window_start = max(0, relative_start - 400)
    window_end = min(len(line), window_start + 1200)
    prefix = "…" if window_start else ""
    suffix = "…" if window_end < len(line) else ""
    return f"{prefix}{line[window_start:window_end]}{suffix}"


# 按本轮已解析 byte range 提取真实 Gold 日志片段
def _gold_evidence(case_id: str, repeat: int) -> str:
    capture_dir = ARTIFACTS_DIR / "captures" / case_id / "buggy" / str(repeat)
    ranges = _read_json(capture_dir / "gold-ranges.json")
    payloads = {
        source: (capture_dir / f"{source}.log").read_bytes()
        for source in ("stdout", "stderr")
    }
    grouped: dict[int, list[str]] = defaultdict(list)
    for reference in ranges:
        source = str(reference["source"])
        start = int(reference["start"])
        end = int(reference["end"])
        payload = payloads[source]
        if not 0 <= start < end <= len(payload):
            raise ValueError(f"invalid gold range: {case_id}/{repeat}/{source}:{start}-{end}")
        snippet = _evidence_line(payload, start, end)
        grouped[int(reference["group"])].append(
            f"[{source} bytes {start}-{end}] {snippet}"
        )
    return "\n".join(
        f"REQUIRED GROUP {group + 1}:\n" + "\nOR\n".join(grouped[group])
        for group in sorted(grouped)
    )


# 构造当前样本的 Gold 诊断字段
def _gold_fields(
    case_id: str,
    variant: str,
    repeat: int,
    manifest: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    if variant == "control":
        return {
            "expected_verdict": "no_fault",
            "expected_category": "no_fault",
            "expected_culprit": "",
            "expected_mechanism": "normal run, no fault injected",
            "gold_evidence": "Control run exited 0; no failure evidence is expected.",
            "expected_patch": False,
        }
    diagnosis = expected["diagnosis"]
    return {
        "expected_verdict": "fault",
        "expected_category": diagnosis["category"],
        "expected_culprit": _acceptable(diagnosis["culprit"]),
        "expected_mechanism": _acceptable(diagnosis["causal_mechanism"]),
        "gold_evidence": _gold_evidence(case_id, repeat),
        "expected_patch": bool(manifest["patchable"]),
    }


# 保留原模型证据引用并声明 formal-v1 的单位歧义
def _candidate_evidence(diagnosis: dict[str, Any]) -> str:
    references = diagnosis.get("evidence", [])
    rendered = "\n".join(
        f"[{item.get('source', '?')}] start={item.get('start', '?')} "
        f"end={item.get('end', '?')}"
        for item in references
    )
    note = "FORMAL-V1 NOTE: offset unit was unspecified; score semantic support only."
    return f"{note}\n{rendered}" if rendered else f"{note}\n(no reference submitted)"


# 将 Gold 与 Candidate 合并为同一评审单元格
def _compare(gold: object, candidate: object) -> str:
    return f"▶ GOLD:\n{gold}\n\n▶ CANDIDATE:\n{candidate}"


# 为同一源结果生成稳定匿名条目 ID
def _item_id(record: dict[str, Any], namespace: str) -> str:
    identity = "\0".join(
        (
            namespace,
            str(record["case_id"]),
            str(record["variant"]),
            str(record["baseline"]),
            str(record["repeat"]),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


# 为真实案例生成稳定且不含业务名称的别名
def _case_aliases(records: list[dict[str, Any]], seed: int) -> dict[str, str]:
    case_ids = sorted({str(record["case_id"]) for record in records})
    random.Random(seed + 1).shuffle(case_ids)
    return {case_id: f"case_{index:03d}" for index, case_id in enumerate(case_ids, 1)}


# 生成一行匿名评审内容和对应私有映射
def _packet_row(
    record: dict[str, Any],
    aliases: dict[str, str],
    manifests: dict[str, dict[str, Any]],
    expected_map: dict[str, dict[str, Any]],
    namespace: str,
) -> tuple[dict[str, object], dict[str, object]]:
    case_id = str(record["case_id"])
    variant = str(record["variant"])
    baseline = str(record["baseline"])
    repeat = int(record["repeat"])
    diagnosis_path = DIAGNOSIS_DIR / case_id / variant / str(repeat) / baseline / "diagnosis.json"
    artifact = _read_json(diagnosis_path)
    answer = artifact.get("answer") or {}
    diagnosis = answer.get("diagnosis") or {}
    manifest = manifests[case_id]
    gold = _gold_fields(case_id, variant, repeat, manifest, expected_map[case_id])
    if variant == "control":
        context = " | ".join(
            ("control", str(manifest["framework"]), str(manifest["control_role"]))
        )
    else:
        context = " | ".join(
            (
                str(record["failure_stage"]),
                str(record["framework"]),
                str(record["fault_family"]),
            )
        )
    expected_patch = bool(gold["expected_patch"])
    candidate_patch = bool(answer.get("patch_recommended", False))
    parse_note = ""
    if artifact.get("status") != "success":
        parse_note = f"[parser note: status={artifact.get('status')}]"
    item_id = _item_id(record, namespace)
    row = {
        "item_id": item_id,
        "case_alias": aliases[case_id],
        "case_context": context,
        "verdict": _compare(gold["expected_verdict"], answer.get("verdict", "")),
        "category": _compare(gold["expected_category"], diagnosis.get("category", "")),
        "culprit": _compare(gold["expected_culprit"], diagnosis.get("culprit", "")),
        "mechanism": _compare(
            gold["expected_mechanism"], diagnosis.get("causal_mechanism", "")
        ),
        "gold_evidence": gold["gold_evidence"],
        "candidate_evidence": _candidate_evidence(diagnosis),
        "patch_intent": _compare(
            "should patch" if expected_patch else "should abstain",
            "patch recommended" if candidate_patch else "no patch",
        ),
        "verdict_correct": "",
        "category_score": "",
        "culprit_score": "",
        "mechanism_score": "",
        "evidence_support_score": "",
        "patch_intent_correct": "",
        "needs_adjudication": "",
        "review_note": parse_note,
    }
    key = {
        "case_id": case_id,
        "baseline": baseline,
        "repeat": repeat,
        "variant": variant,
    }
    return row, key


# 写出一个可复现的匿名评审包和私有映射
def _write_packet(
    records: list[dict[str, Any]],
    output_packet: Path,
    output_key: Path,
    seed: int,
    namespace: str,
    expected_count: int,
) -> None:
    manifests, expected = _load_cases()
    aliases = _case_aliases(records, seed)
    rows_and_keys = [
        _packet_row(record, aliases, manifests, expected, namespace)
        for record in records
    ]
    random.Random(seed).shuffle(rows_and_keys)
    packet_rows = [item[0] for item in rows_and_keys]
    key_map = {str(row["item_id"]): key for row, key in rows_and_keys}
    if len(packet_rows) != expected_count or len(key_map) != expected_count:
        raise ValueError(
            f"expected {expected_count} unique review items, "
            f"got rows={len(packet_rows)}, keys={len(key_map)}"
        )
    with output_packet.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(packet_rows[0]))
        writer.writeheader()
        writer.writerows(packet_rows)
    output_key.write_text(
        json.dumps(
            {
                "packet_version": PACKET_VERSION,
                "run_set": RUN_SET,
                "shuffle_seed": seed,
                "items": key_map,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    buggy = sum(key["variant"] == "buggy" for key in key_map.values())
    controls = sum(key["variant"] == "control" for key in key_map.values())
    print(
        f"Exported {output_packet.name}: {len(packet_rows)} items, "
        f"buggy={buggy}, control={controls}"
    )


# 导出正式评审包或开发集校准包
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--packet", choices=("formal", "calibration"), default="formal")
    args = parser.parse_args()
    report = _read_json(REPORT_PATH)
    if args.packet == "calibration":
        _write_packet(
            _calibration_records(report),
            CALIBRATION_PACKET,
            CALIBRATION_KEY,
            CALIBRATION_SEED,
            f"{RUN_SET}:calibration",
            12,
        )
        return
    _write_packet(
        _formal_records(report),
        OUTPUT_PACKET,
        OUTPUT_KEY,
        SHUFFLE_SEED,
        RUN_SET,
        144,
    )


if __name__ == "__main__":
    main()
