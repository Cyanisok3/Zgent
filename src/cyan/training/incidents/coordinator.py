from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from typing import Any, Literal, cast

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.base import LLMProvider
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig
from cyan.training.events import (
    IncidentOpenedEvent,
    IncidentStatusChangedEvent,
    JobFinishedEvent,
    JobStartedEvent,
    PatchProposedEvent,
    SmokeFinishedEvent,
)
from cyan.training.incidents.evidence import build_failure_capsule
from cyan.training.incidents.fsm import Event, accepts, recover_action, transition
from cyan.training.incidents.models import FailureCapsule, Incident, IncidentStatus
from cyan.training.incidents.patch import PatchError, PatchService
from cyan.training.incidents.runtime import IncidentRuntime
from cyan.training.incidents.smoke import (
    SmokeExecution,
    SmokeResult,
    SmokeVerifierConfig,
    SubprocessSmokeExecutor,
    load_smoke_verifier,
    smoke_verifier_fingerprint,
)
from cyan.training.incidents.store import IncidentStore, RunTrigger
from cyan.training.jobs import AttemptRecord, FailureRecord, JobRecord, JobStore, JobSupervisor
from cyan.training.jobs.models import JobEventType
from cyan.training.processes import terminate_owned_process_group

logger = logging.getLogger(__name__)


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


class IncidentCoordinator:
    # 组装 Failure Capsule、Incident Runtime、审批、Smoke 和重跑状态流
    def __init__(
        self,
        jobs: JobStore,
        supervisor: JobSupervisor,
        bus: EventBus,
        config: CyanConfig,
        *,
        provider: LLMProvider | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._jobs = jobs
        self._supervisor = supervisor
        self._bus = bus
        self._config = config
        self._provider = provider
        self._trace = trace
        self._stores: dict[str, IncidentStore] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._decision_locks: dict[str, asyncio.Lock] = {}
        self._smoke = SubprocessSmokeExecutor()

    # 为后台协程建立统一的生命周期追踪
    def _spawn(self, coroutine: Coroutine[Any, Any, Any], name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # 返回某个 Job 的 Incident store
    def _store_for_job(self, job_id: str) -> IncidentStore:
        store = self._stores.get(job_id)
        if store is None:
            store = IncidentStore(self._jobs.job_dir(job_id) / "incidents")
            self._stores[job_id] = store
        return store

    # 按 ID 查找 Incident 及其 store
    def _find_incident(self, incident_id: str) -> tuple[IncidentStore, Incident]:
        for job in self._jobs.list_jobs():
            store = self._store_for_job(job.id)
            if store.incident_path(incident_id).exists():
                return store, store.read_incident(incident_id)
        raise KeyError(f"incident not found: {incident_id}")

    # 返回 Job 最近更新的 Incident
    def _latest_incident(self, job_id: str) -> tuple[IncidentStore, Incident] | None:
        store = self._store_for_job(job_id)
        incidents = store.list_incidents()
        return (store, incidents[0]) if incidents else None

    # 将失败胶囊写回 Attempt 并返回确定性快照
    async def _persist_capsule(self, failure: FailureRecord) -> FailureCapsule:
        capsule = await build_failure_capsule(self._jobs, failure)
        self._jobs.write_failure(
            failure.model_copy(update={"capsule": capsule.model_dump(mode="json")})
        )
        return capsule

    # 更新状态并广播可重建的摘要事件
    async def _set_status(
        self, store: IncidentStore, incident: Incident, status: IncidentStatus
    ) -> None:
        incident.status = status
        incident.updated_at = _now()
        store.write_incident(incident)
        await self._bus.publish(
            IncidentStatusChangedEvent(
                job_id=incident.job_id,
                incident_id=incident.id,
                status=status,
                ts=incident.updated_at.isoformat(),
            )
        )

    # 通过 FSM 校验事件、持久化目标状态并广播
    async def _dispatch(
        self, store: IncidentStore, incident: Incident, event: Event
    ) -> IncidentStatus:
        next_status = transition(incident.status, event)
        await self._set_status(store, incident, next_status)
        return next_status

    # 在后台任务启动前原子写入本轮 run.json
    def _start_run(
        self,
        store: IncidentStore,
        incident: Incident,
        *,
        trigger: RunTrigger,
        instruction: str,
        attempt_id: str | None = None,
        previous_outcome_summary: dict[str, Any] | None = None,
    ) -> str:
        run_id = f"run-{uuid.uuid4().hex[:12]}"
        run = store.create_run(
            incident.id,
            run_id,
            trigger,
            instruction,
            attempt_id=attempt_id or incident.attempt_id,
            previous_outcome_summary=(
                self._previous_outcome(incident)
                if previous_outcome_summary is None
                else previous_outcome_summary
            ),
        )
        incident.active_run_id = run.run_id
        incident.updated_at = _now()
        store.write_incident(incident)
        self._spawn(
            self._run_diagnosis(store, incident.id, run.run_id),
            f"incident-run-{incident.id}-{run.run_id}",
        )
        return run.run_id

    # 返回上一轮有界结构化结果
    def _previous_outcome(self, incident: Incident) -> dict[str, Any] | None:
        result: dict[str, Any] = {}
        if incident.diagnosis is not None:
            result["diagnosis"] = incident.diagnosis.model_dump(mode="json")
        if incident.proposal is not None:
            result["proposal"] = incident.proposal.model_dump(mode="json")
        if incident.smoke_result is not None:
            result["smoke_result"] = incident.smoke_result.model_dump(mode="json")
        return result or None

    # 处理真实进程失败，创建或复用同一 Incident 并唤醒只读 Runtime
    async def handle_failure(
        self, job: JobRecord, attempt: AttemptRecord, failure: FailureRecord
    ) -> None:
        if failure.kind != "process_exit":
            return
        capsule = await self._persist_capsule(failure)
        latest = self._latest_incident(job.id)
        if latest is not None and latest[1].attempt_id == attempt.id:
            return
        if latest is not None and latest[1].status == "retry_running":
            store, incident = latest
            async with self._decision_locks.setdefault(incident.id, asyncio.Lock()):
                current = store.read_incident(incident.id)
                current.attempt_id = attempt.id
                current.failure_path = str(
                    self._jobs.attempt_dir(job.id, attempt.id) / "failure.json"
                )
                previous = self._previous_outcome(current)
                store.write_incident(current)
                store.clear_agent_artifacts(current.id)
                current = store.read_incident(current.id)
                await self._dispatch(store, current, Event.RETRY_FAILED)
                self._start_run(
                    store,
                    current,
                    trigger="retry_failed",
                    instruction=(
                        "The patched full training command failed. "
                        "Reinvestigate this same incident."
                    ),
                    attempt_id=attempt.id,
                    previous_outcome_summary=previous,
                )
            return
        store = self._store_for_job(job.id)
        incident = Incident(
            id=f"inc-{uuid.uuid4().hex[:12]}",
            job_id=job.id,
            attempt_id=attempt.id,
            workspace_root=capsule.cwd,
            failure_path=str(self._jobs.attempt_dir(job.id, attempt.id) / "failure.json"),
            created_at=_now(),
            updated_at=_now(),
        )
        store.write_incident(incident)
        await self._bus.publish(
            IncidentOpenedEvent(
                job_id=job.id,
                incident_id=incident.id,
                attempt_id=attempt.id,
                ts=incident.created_at.isoformat(),
            )
        )
        self._start_run(
            store,
            incident,
            trigger="initial",
            instruction=(
                "Investigate this real failed training process and submit an "
                "evidence-based diagnosis."
            ),
        )

    # 执行 Runtime 并根据结构化结果收敛 FSM
    async def _run_diagnosis(self, store: IncidentStore, incident_id: str, run_id: str) -> None:
        try:
            outcome = await IncidentRuntime(
                self._jobs,
                store,
                self._config,
                bus=self._bus,
                provider=self._provider,
                trace=self._trace,
            ).run(incident_id, run_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception(
                "incident runtime failed incident_id=%s run_id=%s", incident_id, run_id
            )
            outcome = None
        current = store.read_incident(incident_id)
        if outcome is None or current.diagnosis is None:
            if current.status == "diagnosing":
                await self._dispatch(store, current, Event.INVESTIGATION_FAILED)
            return
        if current.proposal is None:
            if current.status == "diagnosing":
                await self._dispatch(store, current, Event.INVESTIGATION_FAILED)
            return
        current.active_proposal_id = current.proposal.id
        await self._dispatch(store, current, Event.INVESTIGATION_DONE)
        await self._bus.publish(
            PatchProposedEvent(
                job_id=current.job_id,
                incident_id=current.id,
                proposal_id=current.proposal.id,
                summary=current.diagnosis.summary,
                ts=_now().isoformat(),
            )
        )

    # 生成不含私有环境的 Job/Incident 视图
    async def job_view(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.read_job(job_id)
        spec = self._jobs.read_spec(job_id)
        attempt = (
            self._jobs.read_attempt(job.id, job.current_attempt_id)
            if job.current_attempt_id
            else None
        )
        view: dict[str, Any] = {
            "job": job.model_dump(mode="json"),
            "argv": spec.argv,
            "workspace_root": str(spec.workspace_root),
            "attempt": attempt.model_dump(mode="json") if attempt else None,
            "incident": None,
            "diagnosis": None,
            "proposal": None,
            "patch": None,
            "smoke_config": None,
            "smoke_config_fingerprint": None,
            "smoke_result": None,
            "smoke": None,
        }
        latest = self._latest_incident(job_id)
        if latest is None:
            return view
        store, incident = latest
        try:
            smoke_config = load_smoke_verifier(spec.workspace_root)
        except (OSError, ValueError) as exc:
            view["smoke_config_error"] = str(exc)
            smoke_config = None
        proposal = incident.proposal
        view.update(
            {
                "incident": incident.model_dump(mode="json"),
                "diagnosis": incident.diagnosis.model_dump(mode="json")
                if incident.diagnosis
                else None,
                "proposal": proposal.model_dump(mode="json") if proposal else None,
                "patch": store.read_patch(proposal) if proposal else None,
                "smoke_config": smoke_config.model_dump(mode="json") if smoke_config else None,
                "smoke_config_fingerprint": smoke_verifier_fingerprint(smoke_config)
                if smoke_config
                else None,
                "smoke_result": incident.smoke_result.model_dump(mode="json")
                if incident.smoke_result
                else None,
                "smoke": (
                    {
                        **incident.smoke_result.model_dump(mode="json"),
                        "argv": smoke_config.argv if smoke_config else [],
                    }
                    if incident.smoke_result
                    else None
                ),
                "can_apply": bool(
                    incident.diagnosis and incident.proposal and incident.workspace_root
                ),
            }
        )
        return view

    # 返回按更新时间排序的 Job 视图
    async def list_jobs(self) -> list[dict[str, Any]]:
        return [await self.job_view(job.id) for job in self._jobs.list_jobs()]

    # 返回当前待审批 proposal 的只读前后文本
    def review_proposal(
        self, job_id: str, incident_id: str, proposal_id: str
    ) -> tuple[str, str, str]:
        spec = self._jobs.read_spec(job_id)
        store = self._store_for_job(job_id)
        incident = store.read_incident(incident_id)
        if incident.job_id != job_id or incident.status != "awaiting_approval":
            raise ValueError("incident is not awaiting approval")
        if incident.active_proposal_id != proposal_id or incident.proposal is None:
            raise ValueError("proposal is not active")
        if incident.proposal.id != proposal_id:
            raise ValueError("proposal id mismatch")
        before, after = PatchService(spec.workspace_root).review(
            incident.proposal, store.patch_path(incident.proposal)
        )
        return incident.proposal.files[0].path, before, after

    # 执行审批、Patch、可选 Smoke 和原命令重跑
    async def decide(
        self,
        incident_id: str,
        proposal_id: str,
        decision: Literal["approve", "reject"],
        *,
        run_smoke: bool,
        smoke_config_fingerprint: str | None = None,
    ) -> str:
        store, incident = self._find_incident(incident_id)
        async with self._decision_locks.setdefault(incident_id, asyncio.Lock()):
            incident = store.read_incident(incident_id)
            if incident.status != "awaiting_approval":
                return incident.status
            if incident.active_proposal_id != proposal_id or incident.proposal is None:
                raise ValueError("proposal is not active")
            if decision == "reject":
                await self._dispatch(store, incident, Event.REJECT)
                return incident.status
            spec = self._jobs.read_spec(incident.job_id)
            smoke_config: SmokeVerifierConfig | None = None
            if run_smoke:
                smoke_config = load_smoke_verifier(spec.workspace_root)
                observed = smoke_verifier_fingerprint(smoke_config) if smoke_config else None
                if observed != smoke_config_fingerprint:
                    await self._dispatch(store, incident, Event.APPROVE_INVALIDATED)
                    return incident.status
            await self._dispatch(store, incident, Event.APPROVE)
            patch_service = PatchService(spec.workspace_root)
            try:
                receipt = await patch_service.apply(
                    incident.proposal, store.patch_path(incident.proposal)
                )
            except (OSError, PatchError, ValueError):
                await self._dispatch(store, incident, Event.APPLY_FAILED)
                return incident.status
            store.write_receipt(incident.id, receipt)
            if run_smoke and smoke_config is not None:
                if not await self._run_smoke(
                    store, incident, proposal_id, smoke_config, patch_service
                ):
                    return store.read_incident(incident.id).status
            else:
                await self._dispatch(store, incident, Event.APPLY_OK_NO_SMOKE)
            await self._dispatch(store, incident, Event.RETRY_STARTED)
            try:
                retry_job = await self._supervisor.retry(incident.job_id)
            except RuntimeError:
                await self._dispatch(store, incident, Event.RETRY_ABORTED)
                return incident.status
            if retry_job.status == "running" and retry_job.current_attempt_id:
                persisted = self._jobs.find_attempt_event(
                    retry_job.id, retry_job.current_attempt_id, "job.started"
                )
                if persisted:
                    await self._bus.publish(
                        JobStartedEvent(
                            seq=persisted.seq,
                            job_id=retry_job.id,
                            attempt_id=retry_job.current_attempt_id,
                            argv=spec.argv,
                            workspace_root=str(spec.workspace_root),
                            ts=persisted.occurred_at,
                        )
                    )
            self._spawn(self._observe_retry(store, incident.id), f"incident-verify-{incident.id}")
            return incident.status

    # 执行 Smoke 并在失败且哈希安全时回滚
    async def _run_smoke(
        self,
        store: IncidentStore,
        incident: Incident,
        proposal_id: str,
        config: SmokeVerifierConfig,
        patch_service: PatchService,
    ) -> bool:
        spec = self._jobs.read_spec(incident.job_id)
        directory = store.incident_dir(incident.id)

        async def smoke_started(pid: int, process_identity: str) -> None:
            store.write_smoke_execution(
                incident.id,
                SmokeExecution(
                    status="running", pid=pid, process_identity=process_identity, started_at=_now()
                ),
            )
            await self._dispatch(store, store.read_incident(incident.id), Event.APPLY_OK_SMOKE)

        try:
            result = await self._smoke.run(
                config,
                cwd=spec.workspace_root,
                env=spec.env,
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
                on_started=smoke_started,
            )
        except (OSError, RuntimeError, ValueError):
            result = SmokeResult(
                status="failed",
                returncode=None,
                elapsed_ms=0,
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
            )
        current = store.read_incident(incident.id)
        if current.smoke_execution is not None:
            store.write_smoke_execution(
                incident.id,
                current.smoke_execution.model_copy(
                    update={"status": result.status, "finished_at": _now()}
                ),
            )
        store.write_smoke_result(incident.id, result)
        await self._bus.publish(
            SmokeFinishedEvent(
                job_id=incident.job_id,
                incident_id=incident.id,
                status=result.status,
                exit_code=result.returncode,
                ts=_now().isoformat(),
            )
        )
        if result.status == "passed":
            await self._dispatch(store, store.read_incident(incident.id), Event.SMOKE_PASSED)
            return True
        current = store.read_incident(incident.id)
        if current.proposal is None or current.apply_receipt is None:
            await self._dispatch(store, current, Event.SMOKE_FAILED_ROLLBACK_BLOCKED)
            return False
        try:
            await patch_service.reverse(
                current.proposal, store.patch_path(current.proposal), current.apply_receipt
            )
        except (OSError, PatchError, ValueError):
            await self._dispatch(store, current, Event.SMOKE_FAILED_ROLLBACK_BLOCKED)
            return False
        previous = self._previous_outcome(current)
        store.clear_agent_artifacts(incident.id)
        current = store.read_incident(incident.id)
        await self._dispatch(store, current, Event.SMOKE_FAILED_ROLLED_BACK)
        self._start_run(
            store,
            current,
            trigger="smoke_failed",
            instruction=(
                f"Smoke verifier failed after proposal {proposal_id}; "
                "inspect its result and revise."
            ),
            previous_outcome_summary=previous,
        )
        return False

    # 回收 daemon 崩溃后遗留的 Smoke 进程并收敛为 unresolved
    async def _recover_smoke(self, store: IncidentStore, incident: Incident) -> None:
        execution = incident.smoke_execution
        if execution is None:
            return
        if not await terminate_owned_process_group(execution.pid, execution.process_identity):
            logger.error(
                "could not confirm recovered smoke process stopped incident_id=%s", incident.id
            )
            return
        finished = _now()
        store.write_smoke_execution(
            incident.id,
            execution.model_copy(update={"status": "interrupted", "finished_at": finished}),
        )
        directory = store.incident_dir(incident.id)
        store.write_smoke_result(
            incident.id,
            SmokeResult(
                status="interrupted",
                returncode=None,
                elapsed_ms=max(0, int((finished - execution.started_at).total_seconds() * 1000)),
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
            ),
        )
        await self._set_status(store, store.read_incident(incident.id), "unresolved")

    # 等待重跑结束，以真实退出码决定 resolved 或由失败回调继续调查
    async def _observe_retry(self, store: IncidentStore, incident_id: str) -> None:
        incident = store.read_incident(incident_id)
        job = await self._supervisor.wait(incident.job_id)
        attempt = (
            self._jobs.read_attempt(job.id, job.current_attempt_id)
            if job.current_attempt_id
            else None
        )
        if attempt is not None:
            persisted = self._jobs.find_attempt_event(
                job.id, attempt.id, cast(JobEventType, f"job.{job.status}")
            )
            if persisted:
                await self._bus.publish(
                    JobFinishedEvent(
                        seq=persisted.seq,
                        job_id=job.id,
                        attempt_id=attempt.id,
                        status=job.status,
                        exit_code=attempt.returncode,
                        signal=attempt.signal,
                        ts=persisted.occurred_at,
                    )
                )
        current = store.read_incident(incident_id)
        if job.status == "succeeded" and current.status == "retry_running":
            await self._dispatch(store, current, Event.RETRY_SUCCEEDED)
        elif job.status in ("cancelled", "interrupted") and current.status == "retry_running":
            await self._dispatch(store, current, Event.RETRY_ABORTED)
        elif (
            job.status == "failed"
            and (attempt is None or attempt.returncode is None)
            and current.status == "retry_running"
        ):
            await self._dispatch(store, current, Event.RETRY_ABORTED)

    # 接收明确的 Incident 追问，并创建新的独立 run
    async def follow_up(self, incident_id: str, content: str) -> str:
        store, incident = self._find_incident(incident_id)
        async with self._decision_locks.setdefault(incident.id, asyncio.Lock()):
            incident = store.read_incident(incident.id)
            if not accepts(incident.status, Event.FOLLOW_UP):
                raise ValueError(f"incident does not accept follow-up in state {incident.status}")
            previous = self._previous_outcome(incident)
            store.clear_agent_artifacts(incident.id)
            incident = store.read_incident(incident.id)
            await self._dispatch(store, incident, Event.FOLLOW_UP)
            return self._start_run(
                store,
                store.read_incident(incident.id),
                trigger="follow_up",
                instruction=content,
                previous_outcome_summary=previous,
            )

    # 恢复 prepared 或 running 的 active run，保留原始追问指令
    def _resume_active_run(self, store: IncidentStore, incident: Incident) -> bool:
        if incident.active_run_id is None:
            return False
        try:
            run = store.read_run(incident.id, incident.active_run_id)
        except (FileNotFoundError, OSError, ValueError):
            return False
        if run.status not in {"prepared", "running"}:
            return False
        if run.status == "running":
            store.update_run(
                incident.id,
                run.run_id,
                lambda current: setattr(current, "status", "prepared"),
            )
        self._spawn(
            self._run_diagnosis(store, incident.id, run.run_id),
            f"incident-recover-{incident.id}-{run.run_id}",
        )
        return True

    # 恢复 daemon 重启前的 Incident run、审批和重跑状态
    async def recover(self) -> None:
        for job in self._jobs.list_jobs():
            store = self._store_for_job(job.id)
            incidents = store.list_incidents()
            if (
                job.status == "failed"
                and job.current_attempt_id
                and not any(i.attempt_id == job.current_attempt_id for i in incidents)
            ):
                try:
                    attempt = self._jobs.read_attempt(job.id, job.current_attempt_id)
                    failure = self._jobs.read_failure(job.id, job.current_attempt_id)
                except (FileNotFoundError, OSError, ValueError):
                    pass
                else:
                    if failure.kind == "process_exit":
                        await self.handle_failure(job, attempt, failure)
                        incidents = store.list_incidents()
            for incident in incidents:
                if (
                    incident.smoke_execution is not None
                    and incident.smoke_execution.status == "running"
                ):
                    await self._recover_smoke(store, incident)
                    continue
                action = recover_action(incident.status)
                if action == "quarantine":
                    incident.active_proposal_id = None
                    await self._set_status(store, incident, "unresolved")
                elif (
                    incident.status == "awaiting_approval"
                    and incident.proposal is not None
                    and incident.diagnosis is not None
                ):
                    continue
                elif incident.status == "diagnosing":
                    if not self._resume_active_run(store, incident):
                        self._start_run(
                            store,
                            incident,
                            trigger="recovery",
                            instruction="Resume the interrupted incident investigation.",
                        )
                elif incident.status == "retry_running" and job.status == "succeeded":
                    await self._set_status(store, incident, "resolved")
                elif incident.status == "retry_running" and job.status not in (
                    "starting",
                    "running",
                ):
                    await self._set_status(store, incident, "unresolved")

    # 取消仍在后台的 Incident Runtime 和重跑观察任务
    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
