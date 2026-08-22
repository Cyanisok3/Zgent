from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import cast

import pytest

from cyan.training.jobs.models import (
    AttemptRecord,
    FailureRecord,
    JobRecord,
    JobSpec,
    LogStream,
)
from cyan.training.jobs.store import MAX_LOG_READ_BYTES, JobStore


# 功能：验证 JobStore 将公开状态和私有启动信息写入约定目录
# 设计：直接检查真实文件、权限和反序列化结果，覆盖原子 JSON 与 launch 隔离契约
def test_create_job_persists_records_and_private_launch(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    spec = JobSpec(
        argv=["python", "train.py"],
        workspace_root=tmp_path,
        env={"TOKEN": "secret"},
    )
    record = JobRecord(
        id="job-one",
        status="starting",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:00+00:00",
    )

    store.create_job(record, spec)

    job_dir = tmp_path / "jobs" / "job-one"
    assert store.read_job("job-one") == record
    assert store.read_spec("job-one") == spec
    assert stat.S_IMODE((job_dir / "launch.json").stat().st_mode) == 0o600
    assert "TOKEN" not in (job_dir / "job.json").read_text(encoding="utf-8")
    assert not list(job_dir.glob(".job.json.*"))
    assert not list(job_dir.glob(".launch.json.*"))


# 功能：验证 Attempt、Failure 和递增事件序号能够完整持久化
# 设计：连续追加两个终态事件后重新构造 Store 读取，覆盖磁盘恢复序号而非仅依赖内存缓存
def test_attempt_failure_and_events_roundtrip(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    record = JobRecord(
        id="job-two",
        status="starting",
        created_at="start",
        updated_at="start",
    )
    store.create_job(
        record,
        JobSpec(argv=["python"], workspace_root=tmp_path),
    )
    attempt = AttemptRecord(
        id="attempt-0001",
        job_id=record.id,
        status="failed",
        started_at="start",
        finished_at="end",
        returncode=2,
    )
    failure = FailureRecord(
        job_id=record.id,
        attempt_id=attempt.id,
        occurred_at="end",
        kind="process_exit",
        returncode=2,
        message="process exited with status 2",
    )
    store.write_attempt(attempt)
    store.write_failure(failure)
    first = store.append_event(
        event_type="job.started",
        job_id=record.id,
        attempt_id=attempt.id,
        status="running",
        occurred_at="start",
    )
    second_store = JobStore(tmp_path / "jobs")
    second = second_store.append_event(
        event_type="job.failed",
        job_id=record.id,
        attempt_id=attempt.id,
        status="failed",
        occurred_at="end",
    )

    assert store.read_attempt(record.id, attempt.id) == attempt
    assert store.read_failure(record.id, attempt.id) == failure
    assert (first.seq, second.seq) == (1, 2)
    assert [event.type for event in second_store.read_events(record.id)] == [
        "job.started",
        "job.failed",
    ]
    rows = [
        json.loads(line)
        for line in (store.job_dir(record.id) / "events.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert all("text" not in row for row in rows)


# 功能：验证崩溃遗留的 events.jsonl 截断尾行会在下一次追加前被安全修复
# 设计：先写有效 seq=1 再追加半条 JSON，以新 Store 追加事件并断言磁盘只剩连续的 seq=1,2
def test_append_event_repairs_truncated_jsonl_tail(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    record = JobRecord(
        id="job-repair",
        status="running",
        created_at="start",
        updated_at="start",
    )
    store.create_job(record, JobSpec(argv=["python"], workspace_root=tmp_path))
    store.append_event(
        event_type="job.started",
        job_id=record.id,
        attempt_id=None,
        status="running",
        occurred_at="start",
    )
    event_path = store.job_dir(record.id) / "events.jsonl"
    with event_path.open("ab") as file:
        file.write(b'{"seq":')

    restarted_store = JobStore(tmp_path / "jobs")
    appended = restarted_store.append_event(
        event_type="job.succeeded",
        job_id=record.id,
        attempt_id=None,
        status="succeeded",
        occurred_at="end",
    )

    rows = [
        json.loads(line)
        for line in event_path.read_text(encoding="utf-8").splitlines()
    ]
    assert appended.seq == 2
    assert [row["seq"] for row in rows] == [1, 2]
    assert [event.seq for event in restarted_store.read_events(record.id)] == [1, 2]


# 功能：验证 read_log 严格限制单次读取字节并返回可续读偏移量
# 设计：写入超过上限的二进制日志后分段读取，断言边界、EOF 和拼接结果
def test_read_log_is_bounded_and_cursor_based(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    record = JobRecord(
        id="job-log",
        status="running",
        created_at="start",
        updated_at="start",
        current_attempt_id="attempt-0001",
        attempt_ids=["attempt-0001"],
    )
    store.create_job(record, JobSpec(argv=["python"], workspace_root=tmp_path))
    attempt = AttemptRecord(
        id="attempt-0001",
        job_id=record.id,
        status="running",
        started_at="start",
    )
    store.write_attempt(attempt)
    payload = b"a" * (MAX_LOG_READ_BYTES + 19)
    store.log_path(record.id, attempt.id, "stdout").write_bytes(payload)

    first = store.read_log(record.id, attempt.id, "stdout")
    second = store.read_log(
        record.id,
        attempt.id,
        "stdout",
        offset=first.end_offset,
        limit=32,
    )

    assert len(first.text.encode()) == MAX_LOG_READ_BYTES
    assert first.eof is False
    assert second.start_offset == MAX_LOG_READ_BYTES
    assert second.end_offset == len(payload)
    assert second.eof is True
    assert (first.text + second.text).encode() == payload


# 功能：验证 read_log 拒绝负偏移和超过协议上限的读取请求
# 设计：直接覆盖两个输入边界，避免调用方绕过固定的上下文读取预算
@pytest.mark.parametrize(
    ("offset", "limit"),
    [(-1, 1), (0, MAX_LOG_READ_BYTES + 1)],
)
def test_read_log_rejects_invalid_bounds(
    tmp_path: Path,
    offset: int,
    limit: int,
) -> None:
    store = JobStore(tmp_path / "jobs")
    record = JobRecord(
        id="job-invalid",
        status="starting",
        created_at="start",
        updated_at="start",
    )
    store.create_job(record, JobSpec(argv=["python"], workspace_root=tmp_path))

    with pytest.raises(ValueError):
        store.read_log(record.id, "attempt-0001", "stderr", offset=offset, limit=limit)


# 功能：验证路径组件校验阻止 Job ID 逃逸存储根目录
# 设计：使用父目录路径作为恶意 ID，断言在任何文件访问前即被拒绝
def test_job_id_cannot_escape_store_root(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")

    with pytest.raises(ValueError):
        store.read_job("../outside")


# 功能：验证运行时日志流名称校验阻止路径注入
# 设计：绕过静态 Literal 类型传入恶意值，确认存储边界仍独立执行安全校验
def test_log_stream_cannot_escape_attempt_dir(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")

    with pytest.raises(ValueError):
        store.log_path("job-safe", "attempt-safe", cast(LogStream, "../../outside"))


# 功能：验证 list_jobs 按 updated_at 从新到旧返回磁盘记录
# 设计：以反向创建顺序写入两个 Job，排除目录遍历顺序恰好满足断言的偶然性
def test_list_jobs_orders_by_latest_update(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    older = JobRecord(
        id="job-older",
        status="succeeded",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:01+00:00",
    )
    newer = JobRecord(
        id="job-newer",
        status="failed",
        created_at="2026-01-01T00:00:00+00:00",
        updated_at="2026-01-01T00:00:02+00:00",
    )
    store.create_job(newer, JobSpec(argv=["python"], workspace_root=tmp_path))
    store.create_job(older, JobSpec(argv=["python"], workspace_root=tmp_path))

    assert [job.id for job in store.list_jobs()] == ["job-newer", "job-older"]


# 功能：验证 recover_interrupted 只封口活跃 Job 与 Attempt 且不生成 failure
# 设计：同时放置 running 和已成功 Job，断言状态、结束时间、事件及非故障语义
def test_recover_interrupted_closes_only_active_records(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    running = JobRecord(
        id="job-running",
        status="running",
        created_at="start",
        updated_at="start",
        current_attempt_id="attempt-0001",
        attempt_ids=["attempt-0001"],
    )
    succeeded = JobRecord(
        id="job-succeeded",
        status="succeeded",
        created_at="start",
        updated_at="end",
    )
    spec = JobSpec(argv=["python"], workspace_root=tmp_path)
    store.create_job(running, spec)
    store.write_attempt(
        AttemptRecord(
            id="attempt-0001",
            job_id=running.id,
            status="running",
            started_at="start",
        )
    )
    store.create_job(succeeded, spec)

    recovered = store.recover_interrupted()

    recovered_job = store.read_job(running.id)
    recovered_attempt = store.read_attempt(running.id, "attempt-0001")
    assert [job.id for job in recovered] == [running.id]
    assert recovered_job.status == "interrupted"
    assert recovered_attempt.status == "interrupted"
    assert recovered_attempt.finished_at == recovered_job.updated_at
    assert store.read_job(succeeded.id).status == "succeeded"
    assert [event.type for event in store.read_events(running.id)] == ["job.interrupted"]
    assert not (store.attempt_dir(running.id, "attempt-0001") / "failure.json").exists()


# 功能：验证无 current attempt 的 starting Job 也能在启动恢复时安全封口
# 设计：模拟 create_job 后 daemon 立即退出的窄窗口，确保恢复无需伪造 Attempt
def test_recover_interrupted_handles_job_before_attempt_creation(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    starting = JobRecord(
        id="job-starting",
        status="starting",
        created_at="start",
        updated_at="start",
    )
    store.create_job(starting, JobSpec(argv=["python"], workspace_root=tmp_path))

    recovered = store.recover_interrupted()

    assert recovered[0].status == "interrupted"
    event = store.read_events(starting.id)[0]
    assert event.type == "job.interrupted"
    assert event.attempt_id is None
