from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import tomllib
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
CASES_DIR = BASE_DIR / "cases"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
SHUFFLE_SEED = 20260826


# 读取一个 UTF-8 JSON 文件
def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# 读取 run-set 固定数据集版本；旧 run-set 无配置文件时按 formal-v1 处理
def _dataset_version(run_set: str) -> str:
    run_set_json = ARTIFACTS_DIR / "run-sets" / run_set / "run-set.json"
    if run_set_json.is_file():
        return str(_read_json(run_set_json).get("dataset_version", "formal-v1"))
    return "formal-v1"


# 读取案例 manifest 和 Gold 诊断责任点，并按 run-set 数据集版本过滤
def _load_cases(run_set: str) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    dataset_version = _dataset_version(run_set)
    manifests: dict[str, dict[str, Any]] = {}
    expected: dict[str, dict[str, Any]] = {}
    for case_dir in sorted(CASES_DIR.iterdir()):
        manifest_path = case_dir / "case.toml"
        expected_path = case_dir / "expected.json"
        if not manifest_path.is_file() or not expected_path.is_file():
            continue
        manifest = tomllib.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_version", "formal-v1") != dataset_version:
            continue
        case_id = str(manifest["id"])
        manifests[case_id] = manifest
        expected[case_id] = _read_json(expected_path)
    return manifests, expected


# 将 Gold 责任点渲染为任一等价表达
def _gold_culprit(expected: dict[str, Any]) -> str:
    return "ACCEPTABLE ALTERNATIVES — 任一等价表达即可：\n" + "\n".join(
        f"- {value}" for value in expected["diagnosis"]["culprit"]
    )


# 从原始日志范围提取有限完整行，给盲审提供可读 Gold 证据
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


# 读取当前运行的动态 Gold 证据锚点
def _gold_evidence(case_id: str, repeat: int) -> str:
    capture_dir = ARTIFACTS_DIR / "captures" / case_id / "buggy" / str(repeat)
    ranges = _read_json(capture_dir / "gold-ranges.json")
    payloads = {
        source: (capture_dir / f"{source}.log").read_bytes()
        for source in ("stdout", "stderr")
    }
    return "\n".join(
        f"REQUIRED GROUP {int(item['group']) + 1}: "
        f"[{item['source']} bytes {item['start']}-{item['end']}] "
        f"{_evidence_line(payloads[str(item['source'])], int(item['start']), int(item['end']))}"
        for item in ranges
    )


# 为 Incident 结果生成稳定匿名条目 ID
def _item_id(run_set: str, artifact: dict[str, Any]) -> str:
    identity = "\0".join(
        (
            run_set,
            str(artifact["case_id"]),
            str(artifact["repeat"]),
            str(artifact.get("incident_id")),
        )
    )
    return hashlib.sha256(identity.encode()).hexdigest()[:12]


# 导出仅使用 Incident root_cause 的最小人工评审包
def export_packet(run_set: str, split: str, output: Path, key_path: Path) -> None:
    manifests, expected = _load_cases(run_set)
    root = ARTIFACTS_DIR / "run-sets" / run_set / "incident"
    artifacts = []
    for path in sorted(root.glob("*/*/*/incident-benchmark.json")):
        artifact = _read_json(path)
        case_id = str(artifact["case_id"])
        manifest = manifests[case_id]
        if str(manifest["split"]) != split or bool(artifact.get("is_control")):
            continue
        if artifact.get("error") is not None:
            continue
        artifacts.append(artifact)
    if not artifacts:
        raise ValueError(f"no valid Incident artifacts for run_set={run_set!r}, split={split!r}")
    aliases = {
        case_id: f"case_{index:03d}"
        for index, case_id in enumerate(
            sorted({str(item["case_id"]) for item in artifacts}),
            1,
        )
    }
    rows: list[dict[str, object]] = []
    key: dict[str, dict[str, object]] = {}
    for artifact in artifacts:
        case_id = str(artifact["case_id"])
        item_id = _item_id(run_set, artifact)
        diagnosis_refs = artifact.get("diagnosis_evidence_refs") or []
        row = {
            "item_id": item_id,
            "case_alias": aliases[case_id],
            "case_context": " | ".join(
                (
                    str(manifests[case_id]["failure_stage"]),
                    str(manifests[case_id]["framework"]),
                    str(manifests[case_id]["fault_family"]),
                )
            ),
            "gold_culprit": _gold_culprit(expected[case_id]),
            "incident_root_cause": artifact.get("diagnosis_root_cause") or "",
            "causal_support": artifact.get("diagnosis_causal_support") or "",
            "gold_evidence": _gold_evidence(case_id, int(artifact["repeat"])),
            "incident_evidence_refs": json.dumps(
                diagnosis_refs,
                ensure_ascii=False,
                sort_keys=True,
            ),
            "expected_patch": bool(manifests[case_id]["patchable"]),
            "patch_recommended": artifact.get("diagnosis_patch_recommended"),
            "proposal_present": bool(artifact.get("proposal_present")),
            "proposal_valid": bool(artifact.get("proposal_valid")),
            "unsafe_proposal": bool(artifact.get("unsafe_proposal")),
            "culprit_score": "",
            "evidence_support_score": "",
            "patch_intent_correct": "",
            "needs_adjudication": "",
            "review_note": "",
        }
        rows.append(row)
        key[item_id] = {
            "case_id": case_id,
            "repeat": int(artifact["repeat"]),
            "run_set": run_set,
        }
    random.Random(SHUFFLE_SEED).shuffle(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    key_path.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    key_path.write_text(
        json.dumps(
            {"run_set": run_set, "split": split, "items": key},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Exported {output}: {len(rows)} Incident review items")


# 解析命令行参数并导出 Incident review packet
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-set", required=True)
    parser.add_argument("--split", choices=("dev", "test"), default="dev")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key", type=Path, required=True)
    args = parser.parse_args()
    export_packet(args.run_set, args.split, args.output, args.key)


if __name__ == "__main__":
    main()
