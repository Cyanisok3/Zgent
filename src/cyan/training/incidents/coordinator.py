from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from cyan.agent.events.bus import EventBus
from cyan.agent.runner import RunProfile
from cyan.agent.runs import new_run_id
from cyan.agent.session import Session, SessionManager
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
from cyan.training.incidents.profile import (
    build_incident_profile,
    unavailable_incident_profile,
)
from cyan.training.incidents.smoke import (
    SmokeExecution,
    SmokeResult,
    SmokeVerifierConfig,
    SubprocessSmokeExecutor,
    load_smoke_verifier,
    smoke_verifier_fingerprint,
)
from cyan.training.incidents.store import IncidentStore
from cyan.training.jobs import (
    AttemptRecord,
    FailureRecord,
    JobRecord,
    JobStore,
    JobSupervisor,
)
from cyan.training.jobs.models import JobEventType
from cyan.training.processes import terminate_owned_process_group

logger = logging.getLogger(__name__)


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


class IncidentCoordinator:
    # 组装失败快照、只读 Agent、审批、smoke 和重跑状态流
    def __init__(
        self,
        jobs: JobStore,
        sessions: SessionManager,
        supervisor: JobSupervisor,
        bus: EventBus,
    ) -> None:
        self._jobs = jobs
        self._sessions = sessions
        self._supervisor = supervisor
        self._bus = bus
        self._stores: dict[str, IncidentStore] = {}
        self._tasks: set[asyncio.Task[Any]] = set()
        self._decision_locks: dict[str, asyncio.Lock] = {}
        self._smoke = SubprocessSmokeExecutor()

    # 为后台协程建任务并统一追踪生命周期
    def _spawn(self, coroutine: Coroutine[Any, Any, Any], name: str) -> None:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    # 返回某个 Job 的 Incident artifact store
    def _store_for_job(self, job_id: str) -> IncidentStore:
        store = self._stores.get(job_id)
        if store is None:
            store = IncidentStore(self._jobs.job_dir(job_id) / "incidents")
            self._stores[job_id] = store
        return store

    # 按 ID 查找 Incident 及其所属 store
    def _find_incident(self, incident_id: str) -> tuple[IncidentStore, Incident]:
        for job in self._jobs.list_jobs():
            store = self._store_for_job(job.id)
            path = store.incident_dir(incident_id) / "incident.json"
            if path.exists():
                return store, store.read_incident(incident_id)
        raise KeyError(f"incident not found: {incident_id}")

    # 返回 Job 最近创建的 Incident
    def _latest_incident(self, job_id: str) -> tuple[IncidentStore, Incident] | None:
        store = self._store_for_job(job_id)
        incidents = store.list_incidents()
        return (store, incidents[0]) if incidents else None

    # 按 session_id 查找 daemon 管理的 Incident session
    def _incident_for_session(self, session_id: str) -> tuple[IncidentStore, Incident] | None:
        for job in self._jobs.list_jobs():
            store = self._store_for_job(job.id)
            for incident in store.list_incidents():
                if incident.session_id == session_id:
                    return store, incident
        return None

    # 从已封口的日志、Git 和启动信息构造确定性失败胶囊
    async def _build_capsule(self, failure: FailureRecord) -> FailureCapsule:
        return await build_failure_capsule(self._jobs, failure)

    # 将失败胶囊写回 attempt/failure.json 并返回胶囊
    async def _persist_capsule(self, failure: FailureRecord) -> FailureCapsule:
        capsule = await self._build_capsule(failure)
        enriched = failure.model_copy(update={"capsule": capsule.model_dump(mode="json")})
        self._jobs.write_failure(enriched)
        return capsule

    # 更新 Incident 状态并广播可重建状态事件
    async def _set_status(
        self,
        store: IncidentStore,
        incident: Incident,
        status: IncidentStatus,
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

    # 校验事件在当前状态合法后写入并广播；非法转移抛 IllegalTransitionError
    async def _dispatch(
        self,
        store: IncidentStore,
        incident: Incident,
        event: Event,
    ) -> IncidentStatus:
        next_status = transition(incident.status, event)
        await self._set_status(store, incident, next_status)
        return next_status

    # 在锁内把 retry_running 收敛为单次复诊；已离开该状态时幂等跳过
    async def _reinvestigate(
        self,
        store: IncidentStore,
        incident: Incident,
        message: str,
        *,
        attempt_id: str | None = None,
        failure_path: str | None = None,
    ) -> None:
        lock = self._decision_locks.setdefault(incident.id, asyncio.Lock())
        async with lock:
            current = store.read_incident(incident.id)
            if current.status != "retry_running":
                return
            if attempt_id is not None:
                current.attempt_id = attempt_id
            if failure_path is not None:
                current.failure_path = failure_path
            current.active_proposal_id = None
            store.clear_agent_artifacts(current.id)
            await self._dispatch(store, current, Event.RETRY_FAILED)
            self._spawn(
                self._run_diagnosis(store, current, message),
                f"incident-reinvestigate-{current.id}",
            )

    # 处理真实进程失败：新建或恢复同一个 Incident，并自动唤醒只读 Agent
    async def handle_failure(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        failure: FailureRecord,
    ) -> None:
        if failure.kind != "process_exit":
            return
        capsule = await self._persist_capsule(failure)
        latest = self._latest_incident(job.id)
        if latest is not None and latest[1].attempt_id == attempt.id:
            return
        if latest is not None and latest[1].status == "retry_running":
            store, incident = latest
            await self._reinvestigate(
                store,
                incident,
                "The patched full training command failed. Reinvestigate this same incident.",
                attempt_id=attempt.id,
                failure_path=str(
                    self._jobs.attempt_dir(job.id, attempt.id) / "failure.json"
                ),
            )
            return

        store = self._store_for_job(job.id)
        incident_id = f"inc-{uuid.uuid4().hex[:12]}"
        session = await self._sessions.create(
            mode="incident",
            title=f"Incident {job.id}",
            workspace_root=capsule.cwd,
            incident_id=incident_id,
        )
        now = _now()
        incident = Incident(
            id=incident_id,
            job_id=job.id,
            attempt_id=attempt.id,
            workspace_root=capsule.cwd,
            failure_path=str(self._jobs.attempt_dir(job.id, attempt.id) / "failure.json"),
            session_id=session.id,
            created_at=now,
            updated_at=now,
        )
        store.write_incident(incident)
        await self._bus.publish(
            IncidentOpenedEvent(
                job_id=job.id,
                incident_id=incident.id,
                attempt_id=attempt.id,
                session_id=session.id,
                ts=now.isoformat(),
            )
        )
        self._spawn(
            self._run_diagnosis(
                store,
                incident,
                "Investigate this real failed training process and submit an "
                "evidence-based diagnosis.",
            ),
            f"incident-diagnose-{incident.id}",
        )

    # 在同一只读 Incident session 中执行一轮调查
    async def _run_diagnosis(
        self,
        store: IncidentStore,
        incident: Incident,
        message: str,
        *,
        run_id: str | None = None,
    ) -> None:
        if incident.session_id is None:
            await self._dispatch(store, incident, Event.INVESTIGATION_FAILED)
            return
        run_id = run_id or new_run_id()
        incident.active_run_id = run_id
        incident.updated_at = _now()
        store.write_incident(incident)
        try:
            await self._sessions.send_message(incident.session_id, message, run_id=run_id)
        except Exception:
            logger.exception("incident agent failed incident_id=%s", incident.id)
        try:
            diagnosis = store.read_diagnosis(incident.id)
        except (FileNotFoundError, ValueError):
            await self._dispatch(store, incident, Event.INVESTIGATION_FAILED)
            return
        try:
            proposal = store.read_proposal(incident.id)
        except (FileNotFoundError, ValueError):
            incident.active_proposal_id = None
            await self._dispatch(store, incident, Event.INVESTIGATION_FAILED)
            return
        if proposal.diagnosis_id != diagnosis.id:
            incident.active_proposal_id = None
            await self._dispatch(store, incident, Event.INVESTIGATION_FAILED)
            return
        incident.active_proposal_id = proposal.id
        await self._dispatch(store, incident, Event.INVESTIGATION_DONE)
        await self._bus.publish(
            PatchProposedEvent(
                job_id=incident.job_id,
                incident_id=incident.id,
                proposal_id=proposal.id,
                summary=diagnosis.summary,
                ts=_now().isoformat(),
            )
        )

    # 为 Incident session 保留稳定入口并委托 profile 组装模块
    def profile_for_session(self, session: Session) -> RunProfile | None:
        if session.mode != "incident":
            return None
        root = Path(session.workspace_root).resolve()
        try:
            store, incident = self._find_incident(session.incident_id)
            return build_incident_profile(session, self._jobs, store, incident)
        except (KeyError, OSError, ValueError):
            return unavailable_incident_profile(root)

    # 读取可选 artifact，文件不存在或损坏时返回 None
    def _read_optional(self, reader: Any) -> Any:
        try:
            return reader()
        except (FileNotFoundError, OSError, ValueError):
            return None

    # 生成不含私有环境变量的 Job/TUI 视图
    async def job_view(self, job_id: str) -> dict[str, Any]:
        job = self._jobs.read_job(job_id)
        spec = self._jobs.read_spec(job_id)
        attempt = (
            self._jobs.read_attempt(job.id, job.current_attempt_id)
            if job.current_attempt_id is not None
            else None
        )
        view: dict[str, Any] = {
            "job": job.model_dump(mode="json"),
            "argv": spec.argv,
            "workspace_root": str(spec.workspace_root),
            "attempt": attempt.model_dump(mode="json") if attempt is not None else None,
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
        diagnosis = self._read_optional(lambda: store.read_diagnosis(incident.id))
        proposal = self._read_optional(lambda: store.read_proposal(incident.id))
        smoke_result = self._read_optional(lambda: store.read_smoke_result(incident.id))
        try:
            smoke_config = load_smoke_verifier(spec.workspace_root)
        except (OSError, ValueError) as exc:
            view["smoke_config_error"] = str(exc)
            smoke_config = None
        failure = self._read_optional(
            lambda: self._jobs.read_failure(incident.job_id, incident.attempt_id)
        )
        capsule = (
            FailureCapsule.model_validate(failure.capsule)
            if failure is not None and failure.capsule is not None
            else None
        )
        view.update(
            {
                "incident": incident.model_dump(mode="json"),
                "diagnosis": (
                    diagnosis.model_dump(mode="json") if diagnosis is not None else None
                ),
                "proposal": (
                    proposal.model_dump(mode="json") if proposal is not None else None
                ),
                "patch": store.read_patch(proposal) if proposal is not None else None,
                "smoke_config": (
                    smoke_config.model_dump(mode="json")
                    if smoke_config is not None
                    else None
                ),
                "smoke_config_fingerprint": (
                    smoke_verifier_fingerprint(smoke_config)
                    if smoke_config is not None
                    else None
                ),
                "smoke_result": (
                    smoke_result.model_dump(mode="json")
                    if smoke_result is not None
                    else None
                ),
                "smoke": (
                    {
                        **smoke_result.model_dump(mode="json"),
                        "argv": smoke_config.argv if smoke_config is not None else [],
                    }
                    if smoke_result is not None
                    else None
                ),
                "can_apply": capsule is not None and capsule.git_head is not None,
            }
        )
        return view

    # 返回按更新时间排序的精简 Job 列表
    async def list_jobs(self) -> list[dict[str, Any]]:
        return [await self.job_view(job.id) for job in self._jobs.list_jobs()]

    # 返回当前待审批单文件 proposal 的不可变审阅文本
    def review_proposal(
        self,
        job_id: str,
        incident_id: str,
        proposal_id: str,
    ) -> tuple[str, str, str]:
        spec = self._jobs.read_spec(job_id)
        store = self._store_for_job(job_id)
        incident = store.read_incident(incident_id)
        if incident.job_id != job_id:
            raise ValueError("incident belongs to another job")
        if incident.status != "awaiting_approval":
            raise ValueError("incident is not awaiting approval")
        if incident.active_proposal_id != proposal_id:
            raise ValueError("proposal is not active")
        proposal = store.read_proposal(incident_id)
        if proposal.id != proposal_id:
            raise ValueError("proposal id mismatch")
        before, after = PatchService(spec.workspace_root).review(
            proposal,
            store.patch_path(proposal),
        )
        return proposal.files[0].path, before, after

    # 校验 proposal 后应用补丁，按用户选择执行 smoke，再启动完整原命令
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
        lock = self._decision_locks.setdefault(incident_id, asyncio.Lock())
        async with lock:
            incident = store.read_incident(incident_id)
            if incident.status != "awaiting_approval":
                return incident.status
            if incident.active_proposal_id != proposal_id:
                raise ValueError("proposal is not active")
            if decision == "reject":
                await self._dispatch(store, incident, Event.REJECT)
                return incident.status

            proposal = store.read_proposal(incident.id)
            if proposal.id != proposal_id:
                raise ValueError("proposal id mismatch")
            spec = self._jobs.read_spec(incident.job_id)
            smoke_config: SmokeVerifierConfig | None = None
            if run_smoke:
                try:
                    smoke_config = load_smoke_verifier(spec.workspace_root)
                except (OSError, ValueError):
                    await self._dispatch(store, incident, Event.APPROVE_INVALIDATED)
                    return incident.status
                observed_fingerprint = (
                    smoke_verifier_fingerprint(smoke_config)
                    if smoke_config is not None
                    else None
                )
                if observed_fingerprint != smoke_config_fingerprint:
                    await self._dispatch(store, incident, Event.APPROVE_INVALIDATED)
                    return incident.status

            await self._dispatch(store, incident, Event.APPROVE)
            patch_service = PatchService(spec.workspace_root)
            try:
                receipt = await patch_service.apply(proposal, store.patch_path(proposal))
            except (OSError, PatchError, ValueError):
                await self._dispatch(store, incident, Event.APPLY_FAILED)
                return incident.status
            store.write_receipt(incident.id, receipt)

            if run_smoke and smoke_config is not None:
                smoke_ok = await self._run_smoke(
                    store,
                    incident,
                    proposal_id,
                    smoke_config,
                    patch_service,
                )
                if not smoke_ok:
                    return store.read_incident(incident.id).status
            else:
                await self._dispatch(store, incident, Event.APPLY_OK_NO_SMOKE)

            await self._dispatch(store, incident, Event.RETRY_STARTED)
            try:
                retry_job = await self._supervisor.retry(incident.job_id)
            except RuntimeError:
                await self._dispatch(store, incident, Event.RETRY_ABORTED)
                return incident.status
            if (
                retry_job.status == "running"
                and retry_job.current_attempt_id is not None
            ):
                persisted = self._jobs.find_attempt_event(
                    retry_job.id,
                    retry_job.current_attempt_id,
                    "job.started",
                )
                if persisted is not None:
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
                else:
                    logger.error(
                        "missing persisted retry start event job_id=%s attempt_id=%s",
                        retry_job.id,
                        retry_job.current_attempt_id,
                    )
            self._spawn(
                self._observe_retry(store, incident),
                f"incident-verify-{incident.id}",
            )
            return incident.status

    # 执行 smoke；失败时仅在补丁后哈希未变化时安全回滚并恢复同一 Agent session
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

        # 在 verifier 开始执行后立即持久化可校验身份，再把 Incident 暴露为 smoke_running
        async def smoke_started(pid: int, process_identity: str) -> None:
            store.write_smoke_execution(
                incident.id,
                SmokeExecution(
                    status="running",
                    pid=pid,
                    process_identity=process_identity,
                    started_at=_now(),
                ),
            )
            await self._dispatch(store, incident, Event.APPLY_OK_SMOKE)

        try:
            result = await self._smoke.run(
                config,
                cwd=spec.workspace_root,
                env=spec.env,
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
                on_started=smoke_started,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            logger.warning("smoke launch failed incident_id=%s error=%s", incident.id, exc)
            result = SmokeResult(
                status="failed",
                returncode=None,
                elapsed_ms=0,
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
            )
        execution = self._read_optional(lambda: store.read_smoke_execution(incident.id))
        if execution is not None and execution.status == "running":
            store.write_smoke_execution(
                incident.id,
                execution.model_copy(
                    update={
                        "status": result.status,
                        "finished_at": _now(),
                    }
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
            await self._dispatch(store, incident, Event.SMOKE_PASSED)
            return True

        proposal = store.read_proposal(incident.id)
        receipt = store.read_receipt(incident.id)
        try:
            await patch_service.reverse(proposal, store.patch_path(proposal), receipt)
        except (OSError, PatchError, ValueError):
            await self._dispatch(store, incident, Event.SMOKE_FAILED_ROLLBACK_BLOCKED)
            return False
        incident.active_proposal_id = None
        store.clear_agent_artifacts(incident.id)
        await self._dispatch(store, incident, Event.SMOKE_FAILED_ROLLED_BACK)
        self._spawn(
            self._run_diagnosis(
                store,
                incident,
                f"Smoke verifier failed after proposal {proposal_id}; "
                "inspect its result and revise.",
            ),
            f"incident-smoke-rediagnose-{incident.id}",
        )
        return False

    # 终止 daemon 崩溃后遗留的 smoke 进程并把执行观察封口为 interrupted
    async def _recover_smoke(
        self,
        store: IncidentStore,
        incident: Incident,
        execution: SmokeExecution,
    ) -> None:
        stopped = await terminate_owned_process_group(
            execution.pid,
            execution.process_identity,
        )
        if not stopped:
            logger.error(
                "could not confirm recovered smoke process stopped incident_id=%s pid=%s",
                incident.id,
                execution.pid,
            )
            return
        finished_at = _now()
        store.write_smoke_execution(
            incident.id,
            execution.model_copy(
                update={
                    "status": "interrupted",
                    "finished_at": finished_at,
                }
            ),
        )
        directory = store.incident_dir(incident.id)
        store.write_smoke_result(
            incident.id,
            SmokeResult(
                status="interrupted",
                returncode=None,
                elapsed_ms=max(
                    0,
                    int((finished_at - execution.started_at).total_seconds() * 1000),
                ),
                stdout_path=directory / "smoke.stdout.log",
                stderr_path=directory / "smoke.stderr.log",
            ),
        )
        await self._set_status(store, incident, "unresolved")

    # 等待原命令验证结束，以真实退出码决定 resolved 或交由失败回调继续调查
    async def _observe_retry(self, store: IncidentStore, incident: Incident) -> None:
        job = await self._supervisor.wait(incident.job_id)
        attempt = (
            self._jobs.read_attempt(job.id, job.current_attempt_id)
            if job.current_attempt_id is not None
            else None
        )
        if attempt is not None:
            persisted = self._jobs.find_attempt_event(
                job.id,
                attempt.id,
                cast(JobEventType, f"job.{job.status}"),
            )
            if persisted is None:
                logger.error(
                    "missing persisted retry event job_id=%s attempt_id=%s status=%s",
                    job.id,
                    attempt.id,
                    job.status,
                )
            else:
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
        if job.status == "succeeded":
            current = store.read_incident(incident.id)
            await self._dispatch(store, current, Event.RETRY_SUCCEEDED)
        elif job.status in ("cancelled", "interrupted"):
            current = store.read_incident(incident.id)
            await self._dispatch(store, current, Event.RETRY_ABORTED)
        elif job.status == "failed":
            if attempt is not None and attempt.returncode is not None:
                # 进程非零退出由 handle_failure 统一复诊，这里避免双写者竞态
                return
            current = store.read_incident(incident.id)
            if current.status == "retry_running":
                await self._dispatch(store, current, Event.RETRY_ABORTED)

    # 判断一个 session 是否属于受控 Incident
    def is_incident_session(self, session_id: str) -> bool:
        return self._incident_for_session(session_id) is not None

    # 校验状态后启动同一只读 Incident session 的用户追问
    async def follow_up(self, session_id: str, content: str, run_id: str) -> None:
        found = self._incident_for_session(session_id)
        if found is None:
            raise KeyError(f"incident session not found: {session_id}")
        store, incident = found
        lock = self._decision_locks.setdefault(incident.id, asyncio.Lock())
        async with lock:
            incident = store.read_incident(incident.id)
            if not accepts(incident.status, Event.FOLLOW_UP):
                raise ValueError(
                    f"incident does not accept follow-up in state {incident.status}"
                )
            incident.active_proposal_id = None
            store.clear_agent_artifacts(incident.id)
            await self._dispatch(store, incident, Event.FOLLOW_UP)
            self._spawn(
                self._run_diagnosis(
                    store,
                    incident,
                    content,
                    run_id=run_id,
                ),
                f"incident-follow-up-{incident.id}",
            )

    # 恢复 daemon 重启前未完成的 Incident，不把 interrupted Job 当作新故障
    async def recover(self) -> None:
        for job in self._jobs.list_jobs():
            store = self._store_for_job(job.id)
            incidents = store.list_incidents()
            if job.status == "failed" and job.current_attempt_id is not None:
                has_incident = any(
                    incident.attempt_id == job.current_attempt_id
                    for incident in incidents
                )
                if not has_incident:
                    try:
                        attempt = self._jobs.read_attempt(
                            job.id,
                            job.current_attempt_id,
                        )
                        failure = self._jobs.read_failure(
                            job.id,
                            job.current_attempt_id,
                        )
                    except (FileNotFoundError, OSError, ValueError):
                        pass
                    else:
                        if failure.kind == "process_exit":
                            await self.handle_failure(job, attempt, failure)
                            continue
            for incident in incidents:
                smoke_execution = self._read_optional(
                    lambda: store.read_smoke_execution(incident.id)
                )
                if (
                    smoke_execution is not None
                    and smoke_execution.status == "running"
                ):
                    await self._recover_smoke(store, incident, smoke_execution)
                    continue
                action = recover_action(incident.status)
                if action == "quarantine":
                    incident.active_proposal_id = None
                    await self._set_status(store, incident, "unresolved")
                    continue
                if action == "keep":
                    continue
                if incident.status == "awaiting_approval":
                    diagnosis = self._read_optional(
                        lambda: store.read_diagnosis(incident.id)
                    )
                    proposal = self._read_optional(
                        lambda: store.read_proposal(incident.id)
                    )
                    if (
                        diagnosis is None
                        or proposal is None
                        or proposal.diagnosis_id != diagnosis.id
                        or incident.active_proposal_id != proposal.id
                    ):
                        store.clear_agent_artifacts(incident.id)
                        incident.active_proposal_id = None
                        await self._set_status(store, incident, "diagnosing")
                        self._spawn(
                            self._run_diagnosis(
                                store,
                                incident,
                                "Resume the interrupted incident investigation.",
                            ),
                            f"incident-resume-{incident.id}",
                        )
                    continue
                if incident.status == "diagnosing":
                    diagnosis = self._read_optional(
                        lambda: store.read_diagnosis(incident.id)
                    )
                    proposal = self._read_optional(
                        lambda: store.read_proposal(incident.id)
                    )
                    if (
                        diagnosis is not None
                        and proposal is not None
                        and proposal.diagnosis_id == diagnosis.id
                    ):
                        incident.active_proposal_id = proposal.id
                        await self._set_status(store, incident, "awaiting_approval")
                        continue
                    self._spawn(
                        self._clear_and_resume_diagnosis(store, incident),
                        f"incident-resume-{incident.id}",
                    )
                elif incident.status == "retry_running":
                    if job.status == "succeeded":
                        await self._set_status(store, incident, "resolved")
                    elif job.status not in ("starting", "running"):
                        await self._set_status(store, incident, "unresolved")

    # 清除不完整调查制品后恢复同一 Incident session
    async def _clear_and_resume_diagnosis(
        self,
        store: IncidentStore,
        incident: Incident,
    ) -> None:
        store.clear_agent_artifacts(incident.id)
        await self._run_diagnosis(
            store,
            incident,
            "Resume the interrupted incident investigation.",
        )

    # 取消尚未完成的 Agent/验证观察任务
    async def close(self) -> None:
        for task in list(self._tasks):
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
