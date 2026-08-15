from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import sys
import uuid
from collections.abc import Coroutine
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from cyan.core.bus.events import (
    IncidentOpenedEvent,
    IncidentStatusChangedEvent,
    JobFinishedEvent,
    JobStartedEvent,
    PatchProposedEvent,
    SmokeFinishedEvent,
)
from cyan.core.events.bus import EventBus
from cyan.core.incidents.log_tool import (
    FileJobLogReader,
    JobLogReader,
    LogStream,
    ReadJobLogTool,
)
from cyan.core.incidents.models import (
    Diagnosis,
    EvidenceRef,
    FailureCapsule,
    Incident,
    IncidentStatus,
    LogSnapshot,
    Recovery,
    RecoveryAction,
)
from cyan.core.incidents.patch import PatchError, PatchService
from cyan.core.incidents.smoke import (
    SmokeExecution,
    SmokeResult,
    SmokeVerifierConfig,
    SubprocessSmokeExecutor,
    load_smoke_verifier,
    smoke_verifier_fingerprint,
)
from cyan.core.incidents.store import IncidentStore
from cyan.core.incidents.tools import ProposePatchTool, SubmitDiagnosisTool
from cyan.core.jobs import (
    AttemptRecord,
    FailureRecord,
    JobRecord,
    JobStore,
    JobSupervisor,
    snapshot_artifact,
    workflow_contract_fingerprint,
)
from cyan.core.jobs.models import JobEventType
from cyan.core.processes import terminate_owned_process_group
from cyan.core.runner import RunProfile
from cyan.core.runs import new_run_id
from cyan.core.session import Session, SessionManager
from cyan.core.tools.builtin import ReadFileTool, SearchTextTool

logger = logging.getLogger(__name__)

_CAPSULE_LOG_BYTES = 32 * 1024
_EVIDENCE_BUDGET_BYTES = 256 * 1024
_SAFE_ENV_NAMES = frozenset(
    {
        "CONDA_DEFAULT_ENV",
        "CUDA_VISIBLE_DEVICES",
        "LOCAL_RANK",
        "MKL_NUM_THREADS",
        "NCCL_DEBUG",
        "OMP_NUM_THREADS",
        "PYTHONHASHSEED",
        "PYTHONPATH",
        "RANK",
        "VIRTUAL_ENV",
        "WORLD_SIZE",
    }
)
_SAFE_ENV_PREFIXES = ("CUDA_", "NCCL_", "TORCH_")
_SECRET_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "AUTH", "COOKIE", "KEY")
_FOLLOW_UP_STATUSES: frozenset[IncidentStatus] = frozenset(
    {
        "awaiting_approval",
        "stale",
        "unresolved",
        "action_required",
        "rollback_blocked",
    }
)
_RECOVERY_UNCERTAIN_STATUSES: frozenset[IncidentStatus] = frozenset(
    {
        "applying",
        "smoke_running",
        "smoke_passed",
        "smoke_skipped",
    }
)
_WORKSPACE_REFERENCE = re.compile(
    r"^(?P<identity>.+@sha256:[0-9a-f]{64})#L"
    r"(?P<start>[1-9][0-9]*)(?:-L(?P<end>[1-9][0-9]*))?$"
)
_LOG_REFERENCE = re.compile(
    r"^(?P<identity>(?:stdout|stderr):[^/]+/[^@]+@bytes:)"
    r"(?P<start>[0-9]+)-(?P<end>[0-9]+)$"
)


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


# 将支持的 evidence reference 拆成稳定身份和闭区间
def _reference_range(reference: str) -> tuple[str, int, int] | None:
    for pattern in (_WORKSPACE_REFERENCE, _LOG_REFERENCE):
        match = pattern.fullmatch(reference)
        if match is None:
            continue
        start = int(match.group("start"))
        end_group = match.group("end")
        end = int(end_group) if end_group is not None else start
        if end < start:
            return None
        return match.group("identity"), start, end
    return None


# 接受已观察引用本身或其同源子范围，拒绝扩大和身份替换
def _reference_was_observed(reference: str, observed: set[str]) -> bool:
    if reference in observed:
        return True
    requested = _reference_range(reference)
    if requested is None:
        return False
    identity, start, end = requested
    for item in observed:
        registered = _reference_range(item)
        if registered is None:
            continue
        observed_identity, observed_start, observed_end = registered
        if (
            identity == observed_identity
            and observed_start <= start
            and end <= observed_end
        ):
            return True
    return False


# 计算文件 SHA-256；不存在时按空内容计算
def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    if path.exists():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(64 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


# 从文件尾部读取固定字节并构造可引用快照
def _snapshot_log(path: Path, limit: int) -> LogSnapshot:
    size = path.stat().st_size if path.exists() else 0
    start = max(0, size - max(0, limit))
    raw = b""
    if path.exists() and limit > 0:
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(limit)
    return LogSnapshot(
        size=size,
        sha256=_sha256_file(path),
        included_start=start,
        included_end=start + len(raw),
        tail=raw.decode("utf-8", errors="replace"),
    )


# 在线程中为两个已封口日志计算哈希和 stderr 优先的固定尾部
def _snapshot_failure_logs(
    stdout_path: Path,
    stderr_path: Path,
) -> tuple[LogSnapshot, LogSnapshot]:
    stderr_size = stderr_path.stat().st_size if stderr_path.exists() else 0
    stderr_limit = min(stderr_size, _CAPSULE_LOG_BYTES)
    stdout_limit = _CAPSULE_LOG_BYTES - stderr_limit
    return (
        _snapshot_log(stdout_path, stdout_limit),
        _snapshot_log(stderr_path, stderr_limit),
    )


# 从任意大小的日志尾部读取有界 UTF-8 文本，不把完整 smoke 输出载入内存
def _tail_text(path: Path, limit: int) -> str:
    if limit <= 0 or not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        handle.seek(max(0, size - limit))
        raw = handle.read(limit)
    return raw.decode("utf-8", errors="replace")


# 从完整启动环境提取有限且不含凭据的诊断摘要
def _safe_environment(env: dict[str, str]) -> dict[str, str]:
    summary = {
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
    }
    for key in sorted(env):
        upper = key.upper()
        if any(marker in upper for marker in _SECRET_MARKERS):
            continue
        if key not in _SAFE_ENV_NAMES and not key.startswith(_SAFE_ENV_PREFIXES):
            continue
        summary[key] = env[key][:512]
    return summary


class _BudgetedJobLogReader(JobLogReader):
    # 将日志 reader 绑定到单个失败 Attempt，并持久化累计读取字节数
    def __init__(
        self,
        jobs: JobStore,
        store: IncidentStore,
        incident: Incident,
        byte_limit: int = _EVIDENCE_BUDGET_BYTES,
    ) -> None:
        self._jobs = jobs
        self._store = store
        self._incident = incident
        self._byte_limit = byte_limit
        self._bytes_read = 0
        self._store.write_evidence_usage(
            incident.id,
            bytes_read=0,
            byte_limit=byte_limit,
        )

    # 校验 Agent 只能访问当前 Incident 对应日志
    def _validate(self, job_id: str, attempt_id: str) -> None:
        if job_id != self._incident.job_id or attempt_id != self._incident.attempt_id:
            raise ValueError("log access is limited to the current incident attempt")

    # 返回当前 Incident 日志流大小
    async def size(self, job_id: str, attempt_id: str, stream: LogStream) -> int:
        self._validate(job_id, attempt_id)
        path = self._jobs.log_path(job_id, attempt_id, stream)
        return path.stat().st_size if path.exists() else 0

    # 在总证据预算内读取日志并记录实际字节数
    async def read(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        limit: int,
    ) -> bytes:
        self._validate(job_id, attempt_id)
        remaining = self._byte_limit - self._bytes_read
        if remaining <= 0:
            raise ValueError("incident log evidence budget exhausted")
        path = self._jobs.log_path(job_id, attempt_id, stream)
        if not path.exists():
            return b""
        with path.open("rb") as handle:
            handle.seek(offset)
            content = handle.read(min(limit, remaining))
        self._bytes_read += len(content)
        self._store.write_evidence_usage(
            self._incident.id,
            bytes_read=self._bytes_read,
            byte_limit=self._byte_limit,
        )
        return content

    # 扫描当前 Attempt 的完整日志，扫描字节不计入返回证据预算
    async def search(
        self,
        job_id: str,
        attempt_id: str,
        stream: LogStream,
        offset: int,
        query: bytes,
    ) -> int | None:
        self._validate(job_id, attempt_id)
        reader = FileJobLogReader(self._jobs.log_path)
        return await reader.search(job_id, attempt_id, stream, offset, query)


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

    # 执行短 Git 查询；非 Git 或命令失败时返回 None
    async def _git_output(self, root: Path, *args: str) -> str | None:
        try:
            process = await asyncio.create_subprocess_exec(
                "git",
                "-C",
                str(root),
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.DEVNULL,
            )
            stdout, _ = await asyncio.wait_for(process.communicate(), timeout=5)
        except (OSError, TimeoutError):
            return None
        if process.returncode != 0:
            return None
        return stdout.decode("utf-8", errors="replace").strip()

    # 从已封口的日志、Git 和启动信息构造确定性失败胶囊
    async def _build_capsule(self, failure: FailureRecord) -> FailureCapsule:
        spec = self._jobs.read_spec(failure.job_id)
        attempt = self._jobs.read_attempt(failure.job_id, failure.attempt_id)
        root = spec.workspace_root
        stderr_path = self._jobs.log_path(failure.job_id, failure.attempt_id, "stderr")
        stdout_path = self._jobs.log_path(failure.job_id, failure.attempt_id, "stdout")
        git_head_result, dirty_output, snapshots = await asyncio.gather(
            self._git_output(root, "rev-parse", "HEAD"),
            self._git_output(
                root,
                "status",
                "--porcelain=v1",
                "--untracked-files=all",
            ),
            asyncio.to_thread(
                _snapshot_failure_logs,
                stdout_path,
                stderr_path,
            ),
        )
        stdout_snapshot, stderr_snapshot = snapshots
        dirty_paths = (
            [line[3:] for line in dirty_output.splitlines() if len(line) >= 4][:200]
            if dirty_output is not None
            else []
        )
        artifact_before = next(
            (item for item in attempt.artifact_baseline if item.path == failure.artifact_path),
            None,
        )
        artifact_after = None
        if spec.workflow_contract is not None and failure.artifact_path is not None:
            artifact = next(
                (
                    item
                    for item in spec.workflow_contract.artifacts
                    if item.path == failure.artifact_path
                ),
                None,
            )
            if artifact is not None:
                try:
                    artifact_after = snapshot_artifact(root, artifact)
                except (OSError, ValueError):
                    pass
        return FailureCapsule(
            job_id=failure.job_id,
            attempt_id=failure.attempt_id,
            argv=spec.argv,
            cwd=str(root),
            occurred_at=datetime.fromisoformat(failure.occurred_at),
            failure_kind=failure.kind,
            returncode=failure.returncode,
            signal=failure.signal,
            git_head=git_head_result,
            dirty_paths=dirty_paths,
            environment=_safe_environment(spec.env),
            phase=failure.phase,
            check_id=failure.check_id,
            artifact_path=failure.artifact_path,
            contract_fingerprint=failure.contract_fingerprint,
            violation_rule=failure.violation_rule,
            artifact_before=artifact_before,
            artifact_after=artifact_after,
            stdout=stdout_snapshot,
            stderr=stderr_snapshot,
        )

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

    # 判断 preflight artifact 违约是否可直接路由为人工动作
    def _is_deterministic_operator_failure(self, failure: FailureRecord) -> bool:
        if (
            failure.kind != "contract_violation"
            or failure.phase != "preflight"
            or failure.artifact_path is None
            or failure.check_id is not None
        ):
            return False
        spec = self._jobs.read_spec(failure.job_id)
        if spec.workflow_contract is None:
            return False
        return any(
            artifact.path == failure.artifact_path and artifact.role in {"input", "config"}
            for artifact in spec.workflow_contract.artifacts
        )

    # 为确定性 Contract 违约直接写入零 LLM 的 operator_action Diagnosis
    async def _write_operator_diagnosis(
        self,
        store: IncidentStore,
        incident: Incident,
        failure: FailureRecord,
    ) -> None:
        assert failure.artifact_path is not None
        fingerprint = failure.contract_fingerprint or "unknown"
        reference = f"contract:{fingerprint}#artifact:{failure.artifact_path}"
        spec = self._jobs.read_spec(failure.job_id)
        artifact = next(
            (
                item
                for item in (spec.workflow_contract.artifacts if spec.workflow_contract else [])
                if item.path == failure.artifact_path
            ),
            None,
        )
        diagnosis = Diagnosis(
            id=uuid.uuid4().hex,
            incident_id=incident.id,
            category="config" if artifact is not None and artifact.role == "config" else "data",
            summary=failure.message,
            root_cause=(
                "The frozen workflow contract was violated before the workflow command started."
            ),
            evidence=[
                EvidenceRef(
                    source="contract",
                    reference=reference,
                    description=(
                        f"Frozen contract rule {failure.violation_rule!r} failed for "
                        f"{failure.artifact_path}."
                    ),
                )
            ],
            confidence=1.0,
            recovery=Recovery(
                kind="operator_action",
                summary="Provide or repair the required workflow artifact.",
                actions=[
                    RecoveryAction(
                        target=failure.artifact_path,
                        instruction=(
                            f"Create or repair {failure.artifact_path} so it satisfies "
                            f"the frozen contract rule {failure.violation_rule}."
                        ),
                        verification=(
                            "Choose Recheck workflow; Cyan must pass preflight before "
                            "starting the frozen command."
                        ),
                    )
                ],
            ),
            created_at=_now(),
        )
        store.clear_proposal(incident.id)
        store.write_diagnosis(diagnosis)
        incident.active_run_id = None
        incident.active_proposal_id = None
        await self._set_status(store, incident, "action_required")

    # 将一次失败路由到确定性人工动作或现有只读 Incident Agent
    async def _route_incident_failure(
        self,
        store: IncidentStore,
        incident: Incident,
        failure: FailureRecord,
        message: str,
    ) -> None:
        if self._is_deterministic_operator_failure(failure):
            await self._write_operator_diagnosis(store, incident, failure)
            return
        self._spawn(
            self._run_diagnosis(store, incident, message),
            f"incident-diagnose-{incident.id}",
        )

    # 处理可诊断 workflow 失败：新建或恢复同一个 Incident 并进行确定性路由
    async def handle_failure(
        self,
        job: JobRecord,
        attempt: AttemptRecord,
        failure: FailureRecord,
    ) -> None:
        if failure.kind not in {"process_exit", "contract_violation"}:
            return
        capsule = await self._persist_capsule(failure)
        latest = self._latest_incident(job.id)
        if latest is not None and latest[1].attempt_id == attempt.id:
            return
        if latest is not None and latest[1].status == "retry_running":
            store, incident = latest
            incident.attempt_id = attempt.id
            incident.failure_path = str(
                self._jobs.attempt_dir(job.id, attempt.id) / "failure.json"
            )
            incident.active_proposal_id = None
            store.clear_agent_artifacts(incident.id)
            await self._set_status(store, incident, "diagnosing")
            await self._route_incident_failure(
                store,
                incident,
                failure,
                "The replayed full workflow failed. Reinvestigate this same incident.",
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
        await self._route_incident_failure(
            store,
            incident,
            failure,
            "Investigate this real failed Data/ML workflow and submit an evidence-based diagnosis.",
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
            await self._set_status(store, incident, "unresolved")
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
            await self._set_status(store, incident, "unresolved")
            return
        if diagnosis.recovery is not None:
            if diagnosis.recovery.kind == "operator_action":
                store.clear_proposal(incident.id)
                incident.active_proposal_id = None
                await self._set_status(store, incident, "action_required")
                return
            if diagnosis.recovery.kind == "none":
                store.clear_proposal(incident.id)
                incident.active_proposal_id = None
                await self._set_status(store, incident, "unresolved")
                return
        try:
            proposal = store.read_proposal(incident.id)
        except (FileNotFoundError, ValueError):
            incident.active_proposal_id = None
            await self._set_status(store, incident, "unresolved")
            return
        if proposal.diagnosis_id != diagnosis.id:
            incident.active_proposal_id = None
            await self._set_status(store, incident, "unresolved")
            return
        incident.active_proposal_id = proposal.id
        await self._set_status(store, incident, "awaiting_approval")
        await self._bus.publish(
            PatchProposedEvent(
                job_id=incident.job_id,
                incident_id=incident.id,
                proposal_id=proposal.id,
                summary=diagnosis.summary,
                ts=_now().isoformat(),
            )
        )

    # 为 Incident session 构造不可被对话压缩移除的安全 profile
    def profile_for_session(self, session: Session) -> RunProfile | None:
        if session.mode != "incident":
            return None
        root = Path(session.workspace_root).resolve()
        try:
            store, incident = self._find_incident(session.incident_id)
            failure = self._jobs.read_failure(incident.job_id, incident.attempt_id)
            capsule = FailureCapsule.model_validate(failure.capsule)
        except (KeyError, OSError, ValueError):
            return RunProfile(
                workspace_root=root,
                system_prompt_override=(
                    "This incident metadata is unavailable. Do not inspect or modify files. "
                    "Explain that the incident cannot be resumed."
                ),
                tool_whitelist=[],
                max_steps=1,
                compact_threshold=0.0,
                summary_only_events=True,
                include_context=False,
            )

        smoke_context = ""
        try:
            smoke = store.read_smoke_result(incident.id)
            smoke_stdout = _tail_text(smoke.stdout_path, 16 * 1024)
            smoke_stderr = _tail_text(smoke.stderr_path, 16 * 1024)
            smoke_context = (
                "\nLatest smoke verifier result:\n"
                f"{smoke.model_dump_json()}\n"
                f"smoke stdout tail:\n{smoke_stdout}\n"
                f"smoke stderr tail:\n{smoke_stderr}\n"
            )
        except (FileNotFoundError, OSError, ValueError):
            pass

        reader = _BudgetedJobLogReader(self._jobs, store, incident)
        evidence_refs: set[str] = set()
        stderr_reference = (
            f"stderr:{incident.job_id}/{incident.attempt_id}"
            f"@bytes:{capsule.stderr.included_start}-{capsule.stderr.included_end}"
        )
        stdout_reference = (
            f"stdout:{incident.job_id}/{incident.attempt_id}"
            f"@bytes:{capsule.stdout.included_start}-{capsule.stdout.included_end}"
        )
        if capsule.stderr.included_end > capsule.stderr.included_start:
            evidence_refs.add(stderr_reference)
        if capsule.stdout.included_end > capsule.stdout.included_start:
            evidence_refs.add(stdout_reference)
        contract_reference = None
        if capsule.contract_fingerprint is not None:
            subject = (
                f"artifact:{capsule.artifact_path}"
                if capsule.artifact_path is not None
                else f"check:{capsule.check_id or capsule.phase or 'workflow'}"
            )
            contract_reference = f"contract:{capsule.contract_fingerprint}#{subject}"
            evidence_refs.add(contract_reference)

        # 只接受本轮工具实际返回或 failure capsule 明确给出的引用
        def validate_evidence(evidence: EvidenceRef) -> str | None:
            if not _reference_was_observed(evidence.reference, evidence_refs):
                return f"evidence reference was not observed: {evidence.reference}"
            if evidence.source in ("stdout", "stderr"):
                if not evidence.reference.startswith(f"{evidence.source}:"):
                    return "log evidence source does not match its reference"
            elif evidence.source == "contract":
                if not evidence.reference.startswith("contract:"):
                    return "contract evidence source does not match its reference"
            elif "@sha256:" not in evidence.reference:
                return "workspace evidence must include a path and SHA-256"
            return None

        system_prompt = (
            "You are cyan's read-only Incident Agent for a failed local Data/ML workflow. "
            "This is incident response, not metric optimization or experiment research.\n"
            "You may inspect only the supplied workspace and this incident's immutable logs. "
            "Never claim a command ran, never modify the workspace, and never optimize loss or "
            "accuracy. Start from the failure capsule and traceback, then use targeted log search "
            "and small source reads only as needed. As soon as the evidence is sufficient, call "
            "submit_diagnosis exactly once before considering a patch. Every diagnosis must "
            "select recovery.kind=patch, operator_action, or none and provide a concise "
            "recovery summary plus structured operator actions when applicable. Log evidence "
            "must cite "
            "references returned by read_job_log; workspace evidence must cite path@sha256#line "
            "references returned by read/search tools. If the crash is fully caused by an invalid "
            "launch argument, missing external path or data, or the host environment, stop after "
            "diagnosis with recovery.kind=operator_action; do not add validation or fallback code. "
            "Otherwise propose a patch only when one minimal source or config edit directly fixes "
            "the observed crash. Modify only the causal call site; do not harden similar sites, "
            "refactor, or make unrelated improvements. Then call propose_patch with one relative "
            "file path and one exact SEARCH/REPLACE pair. Copy the smallest "
            "unique contiguous SEARCH text verbatim from read_file or search_text output; do not "
            "write diff headers, hunks, or line numbers. If exact matching fails, use the returned "
            "real source feedback to correct the proposal once. If the corrected proposal still "
            "fails, keep the diagnosis and stop proposing a patch. "
            "A diagnosis without a patch is valid.\n"
            f"Failure capsule:\n{capsule.model_dump_json(indent=2)}\n"
            f"Default stderr reference: {stderr_reference}\n"
            f"Default stdout reference: {stdout_reference}\n"
            f"Frozen contract reference: {contract_reference}\n"
            f"{smoke_context}"
        )
        return RunProfile(
            workspace_root=root,
            system_prompt_override=system_prompt,
            tool_whitelist=[
                "read_file",
                "list_dir",
                "search_text",
                "read_job_log",
                "submit_diagnosis",
                "propose_patch",
            ],
            extra_tools=[
                ReadFileTool(
                    root,
                    evidence_refs=evidence_refs,
                    max_bytes=64 * 1024,
                    text_only=True,
                ),
                SearchTextTool(root, evidence_refs=evidence_refs),
                ReadJobLogTool(reader, evidence_refs=evidence_refs),
                SubmitDiagnosisTool(
                    store,
                    incident.id,
                    evidence_validator=validate_evidence,
                ),
                ProposePatchTool(
                    store,
                    incident.id,
                    root,
                    evidence_validator=validate_evidence,
                    patch_service=(
                        PatchService(root) if capsule.git_head is not None else None
                    ),
                ),
            ],
            max_steps=12,
            compact_threshold=0.0,
            summary_only_events=True,
            include_context=False,
            evidence_refs=evidence_refs,
        )

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
            "workflow": (
                {
                    "version": spec.workflow_contract.version,
                    "artifact_count": len(spec.workflow_contract.artifacts),
                    "check_count": len(spec.workflow_contract.checks),
                    "fingerprint": workflow_contract_fingerprint(spec.workflow_contract),
                }
                if spec.workflow_contract is not None
                else None
            ),
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
                "can_apply": (
                    capsule is not None
                    and capsule.git_head is not None
                    and (
                        diagnosis is None
                        or diagnosis.recovery is None
                        or diagnosis.recovery.kind == "patch"
                    )
                ),
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
                await self._set_status(store, incident, "rejected")
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
                    await self._set_status(store, incident, "stale")
                    return incident.status
                observed_fingerprint = (
                    smoke_verifier_fingerprint(smoke_config)
                    if smoke_config is not None
                    else None
                )
                if observed_fingerprint != smoke_config_fingerprint:
                    await self._set_status(store, incident, "stale")
                    return incident.status

            await self._set_status(store, incident, "applying")
            patch_service = PatchService(spec.workspace_root)
            try:
                receipt = await patch_service.apply(proposal, store.patch_path(proposal))
            except (OSError, PatchError, ValueError):
                await self._set_status(store, incident, "stale")
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
                await self._set_status(store, incident, "smoke_skipped")

            await self._set_status(store, incident, "retry_running")
            try:
                retry_job = await self._supervisor.retry(incident.job_id)
            except RuntimeError:
                await self._set_status(store, incident, "unresolved")
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

    # 从 action_required 使用 frozen JobSpec 启动完整 workflow 重检
    async def retry(self, incident_id: str) -> str:
        store, incident = self._find_incident(incident_id)
        lock = self._decision_locks.setdefault(incident_id, asyncio.Lock())
        async with lock:
            incident = store.read_incident(incident_id)
            if incident.status != "action_required":
                raise ValueError("incident is not awaiting an operator action")
            spec = self._jobs.read_spec(incident.job_id)
            await self._set_status(store, incident, "retry_running")
            try:
                retry_job = await self._supervisor.retry(incident.job_id)
            except RuntimeError:
                await self._set_status(store, incident, "unresolved")
                return incident.status
            if retry_job.status == "running" and retry_job.current_attempt_id is not None:
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
            self._spawn(
                self._observe_retry(store, incident),
                f"incident-recheck-{incident.id}",
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
            await self._set_status(store, incident, "smoke_running")

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
            await self._set_status(store, incident, "smoke_passed")
            return True

        proposal = store.read_proposal(incident.id)
        receipt = store.read_receipt(incident.id)
        try:
            await patch_service.reverse(proposal, store.patch_path(proposal), receipt)
        except (OSError, PatchError, ValueError):
            await self._set_status(store, incident, "rollback_blocked")
            return False
        incident.active_proposal_id = None
        store.clear_agent_artifacts(incident.id)
        await self._set_status(store, incident, "diagnosing")
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
            await self._set_status(store, current, "resolved")
        elif job.status in ("cancelled", "interrupted"):
            current = store.read_incident(incident.id)
            await self._set_status(store, current, "unresolved")
        elif job.status == "failed":
            current = store.read_incident(incident.id)
            if current.status == "retry_running":
                await self._set_status(store, current, "unresolved")

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
            if incident.status not in _FOLLOW_UP_STATUSES:
                raise ValueError(
                    f"incident does not accept follow-up in state {incident.status}"
                )
            incident.active_proposal_id = None
            store.clear_agent_artifacts(incident.id)
            await self._set_status(store, incident, "diagnosing")
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
                        if failure.kind in {"process_exit", "contract_violation"}:
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
                if incident.status in _RECOVERY_UNCERTAIN_STATUSES:
                    incident.active_proposal_id = None
                    await self._set_status(store, incident, "unresolved")
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
