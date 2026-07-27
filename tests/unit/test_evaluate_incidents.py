from __future__ import annotations

import json
from pathlib import Path

from scripts.evaluate_incidents import evaluate


# 写入测试所需的 JSON artifact
def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


# 功能：验证评测脚本从真实 artifact 精确汇总延迟、证据、token 和工具调用
# 设计：构造一个完整 resolved Incident 及两轮真实事件文件，断言逐项和聚合值
def test_evaluate_complete_incident(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    sessions = tmp_path / "sessions"
    incident_dir = jobs / "job-1" / "incidents" / "inc-1"
    _write(
        jobs / "job-1" / "attempts" / "attempt-1" / "failure.json",
        {"occurred_at": "2026-01-01T00:00:00+00:00"},
    )
    _write(
        incident_dir / "incident.json",
        {
            "id": "inc-1",
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "status": "resolved",
            "session_id": "session-1",
            "updated_at": "2026-01-01T00:00:50+00:00",
        },
    )
    _write(
        incident_dir / "diagnosis.json",
        {"created_at": "2026-01-01T00:00:10+00:00"},
    )
    _write(incident_dir / "evidence_usage.json", {"bytes_read": 1234})
    events = sessions / "session-1" / "runs" / "run-1" / "events.jsonl"
    events.parent.mkdir(parents=True)
    events.write_text(
        "\n".join(
            [
                json.dumps({"type": "tool.call_started"}),
                json.dumps({"type": "tool.call_started"}),
                json.dumps(
                    {
                        "type": "llm.usage",
                        "input_tokens": 100,
                        "output_tokens": 25,
                        "cache_read_input_tokens": 40,
                        "cache_creation_input_tokens": 5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    report = evaluate(jobs, sessions)

    incident = report["incidents"][0]
    assert incident["failure_to_diagnosis_seconds"] == 10
    assert incident["failure_to_terminal_seconds"] == 50
    assert incident["evidence_bytes"] == 1234
    assert incident["llm_tokens"]["input_tokens"] == 100
    assert incident["tool_calls"] == 2
    assert report["aggregate"]["resolved"] == 1
    assert report["aggregate"]["llm_tokens"]["cache_read_input_tokens"] == 40
    assert report["skipped"] == []


# 功能：验证损坏和未完成 artifact 不会中断整个评测，并明确进入 skipped
# 设计：混合一个损坏 Incident 与一个未完成 Incident，断言保留后者并输出空指标
def test_evaluate_skips_broken_and_incomplete_artifacts(tmp_path: Path) -> None:
    jobs = tmp_path / "jobs"
    sessions = tmp_path / "sessions"
    broken = jobs / "job-bad" / "incidents" / "inc-bad" / "incident.json"
    broken.parent.mkdir(parents=True)
    broken.write_text("{broken", encoding="utf-8")
    incident_dir = jobs / "job-2" / "incidents" / "inc-2"
    _write(
        incident_dir / "incident.json",
        {
            "id": "inc-2",
            "job_id": "job-2",
            "attempt_id": "attempt-2",
            "status": "diagnosing",
            "session_id": "session-2",
            "updated_at": "2026-01-01T00:00:01+00:00",
        },
    )
    _write(
        jobs / "job-2" / "attempts" / "attempt-2" / "failure.json",
        {"occurred_at": "2026-01-01T00:00:00+00:00"},
    )

    report = evaluate(jobs, sessions)

    assert report["aggregate"]["incidents"] == 1
    incident = report["incidents"][0]
    assert incident["status"] == "diagnosing"
    assert incident["failure_to_diagnosis_seconds"] is None
    assert incident["failure_to_terminal_seconds"] is None
    assert incident["evidence_bytes"] is None
    assert incident["llm_tokens"] is None
    assert incident["tool_calls"] is None
    assert len(report["skipped"]) == 4
