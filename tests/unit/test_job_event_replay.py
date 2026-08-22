from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from cyan.errors import HandlerError
from cyan.service.app import CoreApp
from cyan.service.protocol.events import JobFinishedEvent, JobStartedEvent
from cyan.training.jobs import AttemptRecord, JobRecord, JobSpec, JobStore, JobSupervisor


# 功能：验证磁盘终态回放使用 canonical job.finished 并服从 after_seq
# 设计：构造真实 JobStore 事件流后捕获 wire envelope，同时检查起始参数、终态退出码和 topic 匹配
async def test_job_replay_uses_canonical_schema_and_cursor(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    job = JobRecord(
        id="job-replay",
        status="failed",
        created_at="start",
        updated_at="end",
        current_attempt_id="attempt-0001",
        attempt_ids=["attempt-0001"],
    )
    store.create_job(
        job,
        JobSpec(
            argv=[sys.executable, "train.py"],
            workspace_root=tmp_path,
        ),
    )
    store.write_attempt(
        AttemptRecord(
            id="attempt-0001",
            job_id=job.id,
            status="failed",
            started_at="start",
            finished_at="end",
            returncode=3,
        )
    )
    store.append_event(
        event_type="job.started",
        job_id=job.id,
        attempt_id="attempt-0001",
        status="running",
        occurred_at="start",
    )
    store.append_event(
        event_type="job.failed",
        job_id=job.id,
        attempt_id="attempt-0001",
        status="failed",
        occurred_at="end",
    )
    app = CoreApp()
    app._job_store = store
    writer = MagicMock(spec=asyncio.StreamWriter)
    writer.drain = AsyncMock()

    count = await app._replay_job_events(
        job.id,
        cast(asyncio.StreamWriter, writer),
        ["job.*"],
        0,
    )
    payloads = [
        json.loads(call.args[0].rstrip(b"\n"))["event"]
        for call in writer.write.call_args_list
    ]

    assert count == 2
    assert [(item["type"], item["seq"]) for item in payloads] == [
        ("job.started", 1),
        ("job.finished", 2),
    ]
    assert payloads[0]["argv"] == [sys.executable, "train.py"]
    assert payloads[0]["workspace_root"] == str(tmp_path)
    assert payloads[1]["status"] == "failed"
    assert payloads[1]["exit_code"] == 3

    writer.reset_mock()
    writer.drain = AsyncMock()
    count = await app._replay_job_events(
        job.id,
        cast(asyncio.StreamWriter, writer),
        ["job.finished"],
        1,
    )
    terminal = json.loads(writer.write.call_args.args[0].rstrip(b"\n"))["event"]
    assert count == 1
    assert terminal["type"] == "job.finished"
    assert terminal["seq"] == 2


# 功能：验证首次运行的实时 Job 事件复用磁盘中已经分配的单调序号
# 设计：启动真实短进程并同时收集 EventBus 与 JobStore，逐项比较 canonical 类型和 seq
async def test_live_job_events_reuse_persisted_sequences(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    app = CoreApp()
    app._job_store = store
    app._job_supervisor = JobSupervisor(store)
    observed: list[BaseModel] = []

    # 收集 CoreApp 发布的实时事件
    async def collect(event: BaseModel) -> None:
        observed.append(event)

    app._bus.subscribe(collect)
    result = await app._job_start_handler(
        {
            "argv": [sys.executable, "-c", "print('done')"],
            "workspace_root": str(tmp_path),
        }
    )
    for _ in range(100):
        if any(isinstance(event, JobFinishedEvent) for event in observed):
            break
        await asyncio.sleep(0.01)

    live = [
        event
        for event in observed
        if isinstance(event, (JobStartedEvent, JobFinishedEvent))
    ]
    persisted = store.read_events(result.job_id)
    assert [(event.type, event.seq) for event in live] == [
        ("job.started", 1),
        ("job.finished", 2),
    ]
    assert [event.seq for event in live] == [event.seq for event in persisted]


# 功能：验证 shutdown 门禁置位后已建立连接也不能再启动新的训练进程
# 设计：直接调用 Job handler 并放入不可用 supervisor，确认在任何 spawn 前返回结构化拒绝
async def test_job_start_is_rejected_while_core_shuts_down(tmp_path: Path) -> None:
    app = CoreApp()
    app._job_supervisor = cast(JobSupervisor, object())
    app._shutting_down = True

    with pytest.raises(HandlerError, match="shutting down"):
        await app._job_start_handler(
            {
                "argv": [sys.executable, "-c", "print('must not run')"],
                "workspace_root": str(tmp_path),
            }
        )
