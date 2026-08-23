from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

TERMINAL = {"resolved", "rejected", "stale", "unresolved", "rollback_blocked"}
TOKEN_KEYS = (
    "input_tokens",
    "output_tokens",
    "cache_read_input_tokens",
    "cache_creation_input_tokens",
)


# 读取 JSON 对象，失败时将路径和原因加入 skipped
def _read_json(path: Path, skipped: list[dict[str, str]]) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("expected JSON object")
        return value
    except (OSError, ValueError) as exc:
        skipped.append({"path": str(path), "reason": str(exc)})
        return None


# 解析 ISO 8601 时间；无效值作为损坏 artifact 报告
def _time(value: object, path: Path, skipped: list[dict[str, str]]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    except ValueError as exc:
        skipped.append({"path": str(path), "reason": f"invalid timestamp: {exc}"})
        return None


# 计算两个时间点之间非负秒数
def _seconds(start: datetime | None, end: datetime | None) -> float | None:
    if start is None or end is None:
        return None
    return max(0.0, round((end - start).total_seconds(), 6))


# 汇总一个 Incident 下全部 run 的 token、工具调用和 selector 指标
def _run_metrics(
    incident_dir: Path,
    skipped: list[dict[str, str]],
) -> tuple[dict[str, int] | None, int | None, dict[str, int] | None]:
    run_files = sorted(incident_dir.glob("runs/*/events.jsonl"))
    run_records = sorted(incident_dir.glob("runs/*/run.json"))
    tokens = dict.fromkeys(TOKEN_KEYS, 0)
    tool_calls = 0
    selected_bytes = 0
    scanned_bytes = 0
    readable = False
    for path in run_files:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError as exc:
            skipped.append({"path": str(path), "reason": str(exc)})
            continue
        readable = True
        for line_number, line in enumerate(lines, 1):
            try:
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError("expected JSON object")
                if event.get("type") == "llm.usage":
                    for key in TOKEN_KEYS:
                        tokens[key] += int(event.get(key, 0))
                elif event.get("type") == "tool.call_started":
                    tool_calls += 1
            except (TypeError, ValueError) as exc:
                skipped.append({"path": f"{path}:{line_number}", "reason": str(exc)})
    for path in run_records:
        record = _read_json(path, skipped)
        if record is None:
            continue
        selected_bytes += int(record.get("selected_bytes", 0) or 0)
        scanned_bytes += int(record.get("scanned_bytes", 0) or 0)
    metrics = {
        "selected_bytes": selected_bytes,
        "scanned_bytes": scanned_bytes,
    }
    return (tokens if readable else None, tool_calls if readable else None, metrics)


# 从一个 Incident 快照提取可复核指标
def _incident_metrics(
    incident_path: Path,
    jobs_root: Path,
    skipped: list[dict[str, str]],
) -> dict[str, Any] | None:
    data = _read_json(incident_path, skipped)
    if data is None:
        return None
    required = ("id", "job_id", "attempt_id", "status", "updated_at")
    if any(not data.get(key) for key in required):
        skipped.append({"path": str(incident_path), "reason": "missing required fields"})
        return None
    directory = incident_path.parent
    failure_path = (
        jobs_root
        / str(data["job_id"])
        / "attempts"
        / str(data["attempt_id"])
        / "failure.json"
    )
    failure = _read_json(failure_path, skipped)
    failure_at = _time(failure.get("occurred_at"), failure_path, skipped) if failure else None
    diagnosis = data.get("diagnosis")
    diagnosis_path = directory / "incident.json"
    diagnosis_at = (
        _time(diagnosis.get("created_at"), diagnosis_path, skipped)
        if isinstance(diagnosis, dict)
        else None
    )
    terminal_at = (
        _time(data["updated_at"], incident_path, skipped)
        if data["status"] in TERMINAL
        else None
    )
    tokens, tool_calls, selector = _run_metrics(directory, skipped)
    return {
        "incident_id": str(data["id"]),
        "job_id": str(data["job_id"]),
        "attempt_id": str(data["attempt_id"]),
        "status": str(data["status"]),
        "resolved": data["status"] == "resolved",
        "failure_to_diagnosis_seconds": _seconds(failure_at, diagnosis_at),
        "failure_to_terminal_seconds": _seconds(failure_at, terminal_at),
        "selector": selector,
        "llm_tokens": tokens,
        "tool_calls": tool_calls,
    }


# 对一组可空数值输出有覆盖率的统计，避免把缺失值当作零
def _numeric_summary(values: list[int | float | None]) -> dict[str, int | float | None]:
    measured = [value for value in values if value is not None]
    return {
        "measured": len(measured),
        "min": min(measured) if measured else None,
        "mean": round(sum(measured) / len(measured), 6) if measured else None,
        "max": max(measured) if measured else None,
    }


# 扫描真实 cyan artifacts 并返回稳定、机器可读的评测对象
def evaluate(jobs_root: Path) -> dict[str, Any]:
    jobs_root = jobs_root.expanduser().resolve()
    skipped: list[dict[str, str]] = []
    incidents = [
        result
        for path in sorted(jobs_root.glob("*/incidents/*/incident.json"))
        if (result := _incident_metrics(path, jobs_root, skipped)) is not None
    ]
    token_totals = {key: 0 for key in TOKEN_KEYS}
    token_measured = 0
    for incident in incidents:
        if incident["llm_tokens"] is not None:
            token_measured += 1
            for key in TOKEN_KEYS:
                token_totals[key] += incident["llm_tokens"][key]
    return {
        "schema_version": 2,
        "jobs_root": str(jobs_root),
        "incidents": incidents,
        "aggregate": {
            "incidents": len(incidents),
            "resolved": sum(item["resolved"] for item in incidents),
            "status_counts": dict(sorted(Counter(item["status"] for item in incidents).items())),
            "failure_to_diagnosis_seconds": _numeric_summary(
                [item["failure_to_diagnosis_seconds"] for item in incidents]
            ),
            "failure_to_terminal_seconds": _numeric_summary(
                [item["failure_to_terminal_seconds"] for item in incidents]
            ),
            "selected_evidence_bytes": sum(
                item["selector"]["selected_bytes"] for item in incidents
            ),
            "scanned_log_bytes": sum(item["selector"]["scanned_bytes"] for item in incidents),
            "llm_tokens": {"measured": token_measured, **token_totals},
            "tool_calls": {
                "measured": sum(item["tool_calls"] is not None for item in incidents),
                "total": sum(item["tool_calls"] or 0 for item in incidents),
            },
        },
        "skipped": skipped,
    }


# 解析路径参数并把评测对象写到标准输出
def main() -> None:
    parser = argparse.ArgumentParser(description="Export real cyan Incident metrics as JSON")
    parser.add_argument("--jobs-root", type=Path, default=Path("~/.cyan/jobs"))
    parser.add_argument("--pretty", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            evaluate(args.jobs_root),
            ensure_ascii=False,
            indent=2 if args.pretty else None,
        )
    )


if __name__ == "__main__":
    main()
