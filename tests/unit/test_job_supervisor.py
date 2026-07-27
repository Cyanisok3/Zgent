from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from cyan.core.jobs.models import FailureRecord, JobSpec
from cyan.core.jobs.store import JobStore
from cyan.core.jobs.supervisor import JobSupervisor


# 功能：验证成功进程的 stdout/stderr 被分别无损保存并产生成功状态
# 设计：启动真实 Python 子进程同时写两个流，覆盖异步 drain 和正常退出的完整链路
async def test_supervisor_captures_streams_and_success(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store)
    spec = JobSpec(
        argv=[
            sys.executable,
            "-c",
            "import sys; print('out'); print('err', file=sys.stderr)",
        ],
        workspace_root=tmp_path,
    )

    started = await supervisor.start(spec)
    finished = await supervisor.wait(started.id)
    attempt_id = finished.current_attempt_id

    assert finished.status == "succeeded"
    assert attempt_id is not None
    assert store.read_attempt(finished.id, attempt_id).returncode == 0
    assert store.read_log(finished.id, attempt_id, "stdout").text == "out\n"
    assert store.read_log(finished.id, attempt_id, "stderr").text == "err\n"
    assert [event.type for event in store.read_events(finished.id)] == [
        "job.started",
        "job.succeeded",
    ]


# 功能：验证非零退出会在持久化失败快照后调用 failure callback
# 设计：回调中立即读取磁盘记录，证明自动唤醒入口只会看到已完整落盘的证据
async def test_supervisor_persists_failure_before_callback(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    observed: list[FailureRecord] = []

    # 在回调时验证失败文件已经可读取
    async def on_failure(job: object, attempt: object, failure: FailureRecord) -> None:
        observed.append(store.read_failure(failure.job_id, failure.attempt_id))

    supervisor = JobSupervisor(store, failure_callback=on_failure)
    started = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "raise SystemExit(7)"],
            workspace_root=tmp_path,
        )
    )
    finished = await supervisor.wait(started.id)

    assert finished.status == "failed"
    assert len(observed) == 1
    assert observed[0].returncode == 7
    assert observed[0].kind == "process_exit"


# 功能：验证 cancel 终止受监督进程并且不会触发失败回调
# 设计：取消真实长时子进程并等待 monitor 收口，区分用户取消与意外崩溃
async def test_supervisor_cancel_is_not_failure(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    failures: list[FailureRecord] = []

    # 收集不应出现的失败通知
    def on_failure(job: object, attempt: object, failure: FailureRecord) -> None:
        failures.append(failure)

    supervisor = JobSupervisor(store, failure_callback=on_failure)
    started = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "import time; time.sleep(30)"],
            workspace_root=tmp_path,
        )
    )

    finished = await supervisor.cancel(started.id, timeout=1)

    assert finished.status == "cancelled"
    assert failures == []
    attempt_id = finished.current_attempt_id
    assert attempt_id is not None
    assert store.read_attempt(finished.id, attempt_id).status == "cancelled"
    assert not (store.attempt_dir(finished.id, attempt_id) / "failure.json").exists()


# 功能：验证 retry 使用原始私有启动信息创建新的 Attempt
# 设计：首轮进程写标记后失败、次轮据此成功，证明同一 Job 的精确重跑和历史保留
async def test_supervisor_retry_reuses_launch_spec(tmp_path: Path, monkeypatch) -> None:
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store)
    monkeypatch.setenv("CYAN_DAEMON_ONLY_SECRET", "must-not-leak")
    code = (
        "import os; from pathlib import Path; "
        "assert os.environ['CYAN_JOB_VALUE'] == 'preserved'; "
        "assert 'CYAN_DAEMON_ONLY_SECRET' not in os.environ; "
        "p=Path('retry-marker'); "
        "exists=p.exists(); "
        "p.write_text('ready'); "
        "raise SystemExit(0 if exists else 3)"
    )
    started = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", code],
            workspace_root=tmp_path,
            env={"CYAN_JOB_VALUE": "preserved"},
        )
    )
    first = await supervisor.wait(started.id)

    retried = await supervisor.retry(first.id)
    second = await supervisor.wait(retried.id)

    assert first.status == "failed"
    assert second.status == "succeeded"
    assert second.attempt_ids == ["attempt-0001", "attempt-0002"]
    assert store.read_attempt(second.id, "attempt-0001").returncode == 3
    assert store.read_attempt(second.id, "attempt-0002").returncode == 0
    assert store.read_spec(second.id).env["CYAN_JOB_VALUE"] == "preserved"
    assert "CYAN_DAEMON_ONLY_SECRET" not in store.read_spec(second.id).env


# 功能：验证 daemon 的 stdin 失效后重试仍能以非交互输入启动真实 Python
# 设计：首轮正常失败后临时关闭父进程 fd 0，再执行真实 retry 并要求子进程从 stdin 读到 EOF
async def test_supervisor_retry_isolates_revoked_stdin(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store)
    code = (
        "import os; from pathlib import Path; "
        "p=Path('stdin-retry-marker'); "
        "retry=p.exists(); "
        "p.write_text('ready'); "
        "assert not retry or os.read(0, 1) == b''; "
        "raise SystemExit(0 if retry else 3)"
    )
    started = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", code],
            workspace_root=tmp_path,
        )
    )
    first = await supervisor.wait(started.id)
    saved_stdin = os.dup(0)
    try:
        os.close(0)
        retried = await supervisor.retry(first.id)
    finally:
        os.dup2(saved_stdin, 0)
        os.close(saved_stdin)
    second = await supervisor.wait(retried.id)

    assert first.status == "failed"
    assert second.status == "succeeded"
    assert store.read_attempt(second.id, "attempt-0002").returncode == 0


# 功能：验证无法启动的命令只记录 launch_error，不把产品自身启动错误唤醒为训练 Incident
# 设计：使用确定不存在的可执行文件覆盖 spawn 前失败，并检查终态、快照和零失败回调
async def test_supervisor_records_launch_error(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    observed: list[FailureRecord] = []

    # 收集启动失败通知
    def on_failure(job: object, attempt: object, failure: FailureRecord) -> None:
        observed.append(failure)

    supervisor = JobSupervisor(store, failure_callback=on_failure)
    finished = await supervisor.start(
        JobSpec(argv=["definitely-not-a-real-cyan-command"], workspace_root=tmp_path)
    )

    assert finished.status == "failed"
    assert observed == []
    assert store.read_failure(finished.id, "attempt-0001").kind == "launch_error"


# 功能：验证 shutdown 关闭启动闸门后 Incident 重跑不能创建新 Attempt
# 设计：先完成真实失败任务，再关闭唯一进程入口并断言 retry 被拒绝且历史不变
async def test_supervisor_stop_starting_rejects_retry(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs")
    supervisor = JobSupervisor(store)
    started = await supervisor.start(
        JobSpec(
            argv=[sys.executable, "-c", "raise SystemExit(2)"],
            workspace_root=tmp_path,
        )
    )
    failed = await supervisor.wait(started.id)

    supervisor.stop_starting()
    with pytest.raises(RuntimeError, match="shutting down"):
        await supervisor.retry(failed.id)

    assert store.read_job(failed.id).attempt_ids == ["attempt-0001"]
