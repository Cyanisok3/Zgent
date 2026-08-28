from __future__ import annotations

import asyncio

from cyan.agent.events.bus import EventBus
from cyan.agent.llm.base import LLMProvider
from cyan.agent.runner import AgentRunner, RunOutcome
from cyan.agent.trace.writer import TraceWriter
from cyan.config import CyanConfig
from cyan.training.incidents.context import MAX_INITIAL_EVIDENCE_BYTES
from cyan.training.incidents.evidence import SmokeEvidenceSelection, select_smoke_evidence
from cyan.training.incidents.models import FailureCapsule
from cyan.training.incidents.profile import build_incident_profile
from cyan.training.incidents.selector import select_evidence
from cyan.training.incidents.store import IncidentRun, IncidentStore
from cyan.training.jobs.store import JobStore


class IncidentRuntime:
    # 初始化一个具体 Incident Agent 运行器，不暴露通用编排协议
    def __init__(
        self,
        jobs: JobStore,
        store: IncidentStore,
        config: CyanConfig,
        *,
        bus: EventBus | None = None,
        provider: LLMProvider | None = None,
        trace: TraceWriter | None = None,
    ) -> None:
        self._jobs = jobs
        self._store = store
        self._config = config
        self._bus = bus
        self._provider = provider
        self._trace = trace

    # 执行一轮 selector → bounded context → six-tool Agent
    async def run(self, incident_id: str, run_id: str) -> RunOutcome:
        incident = self._store.read_incident(incident_id)
        run = self._store.read_run(incident_id, run_id)
        try:
            failure = self._jobs.read_failure(incident.job_id, incident.attempt_id)
            capsule = FailureCapsule.model_validate(failure.capsule)
        except (FileNotFoundError, OSError, ValueError):
            def mark_capsule_unavailable(current: IncidentRun) -> None:
                current.status = "failed"
                current.reason = "failure_capsule_unavailable"

            self._store.update_run(incident_id, run_id, mark_capsule_unavailable)
            return RunOutcome("failed", "", "failure_capsule_unavailable")
        stdout_path = self._jobs.log_path(incident.job_id, incident.attempt_id, "stdout")
        stderr_path = self._jobs.log_path(incident.job_id, incident.attempt_id, "stderr")
        selection = await asyncio.to_thread(
            select_evidence,
            capsule,
            stdout_path,
            stderr_path,
            MAX_INITIAL_EVIDENCE_BYTES,
        )
        if (
            selection.stdout_sha256 != capsule.stdout.sha256
            or selection.stderr_sha256 != capsule.stderr.sha256
        ):
            def mark_log_changed(current: IncidentRun) -> None:
                current.status = "failed"
                current.reason = "log_changed"

            self._store.update_run(incident_id, run_id, mark_log_changed)
            return RunOutcome("failed", "", "log_changed")
        smoke_evidence: SmokeEvidenceSelection | None = None
        if run.trigger == "smoke_failed":
            if run.smoke_proposal_id is None:
                def mark_smoke_unavailable(current: IncidentRun) -> None:
                    current.status = "failed"
                    current.reason = "smoke_evidence_unavailable"

                self._store.update_run(incident_id, run_id, mark_smoke_unavailable)
                self._store.update_incident(
                    incident_id,
                    lambda current: setattr(
                        current,
                        "last_outcome",
                        "smoke_evidence_unavailable",
                    ),
                )
                return RunOutcome("failed", "", "smoke_evidence_unavailable")
            smoke_stdout, smoke_stderr = self._store.smoke_log_paths(
                incident_id,
                run.smoke_proposal_id,
            )
            try:
                smoke_evidence = await asyncio.to_thread(
                    select_smoke_evidence,
                    smoke_stdout,
                    smoke_stderr,
                    incident_id,
                    run.smoke_proposal_id,
                )
            except (OSError, ValueError):
                smoke_evidence = None
            if (
                smoke_evidence is None
                or not smoke_evidence.blocks
                or smoke_evidence.stdout_sha256 != run.smoke_stdout_sha256
                or smoke_evidence.stderr_sha256 != run.smoke_stderr_sha256
            ):
                def mark_smoke_unavailable(current: IncidentRun) -> None:
                    current.status = "failed"
                    current.reason = "smoke_evidence_unavailable"

                self._store.update_run(incident_id, run_id, mark_smoke_unavailable)
                self._store.update_incident(
                    incident_id,
                    lambda current: setattr(
                        current,
                        "last_outcome",
                        "smoke_evidence_unavailable",
                    ),
                )
                return RunOutcome("failed", "", "smoke_evidence_unavailable")
        refs = [
            {
                "source": item.source,
                "reference": item.as_reference(capsule),
                "kind": item.kind,
                "selection_reason": item.selection_reason,
                "sha256": item.sha256,
            }
            for item in selection.references
        ]
        if smoke_evidence is not None:
            refs.extend(
                {
                    "source": block.source,
                    "reference": block.reference,
                    "kind": "smoke_tail",
                    "selection_reason": block.description,
                    "sha256": block.sha256,
                }
                for block in smoke_evidence.blocks
            )

        def mark_running(current: IncidentRun) -> None:
            current.status = "running"
            current.selected_evidence = refs
            current.stdout_sha256 = selection.stdout_sha256
            current.stderr_sha256 = selection.stderr_sha256
            current.scanned_bytes = selection.scanned_bytes
            current.selected_bytes = selection.selected_bytes + (
                smoke_evidence.selected_bytes if smoke_evidence else 0
            )
            current.duplicates_removed = selection.duplicates_removed

        self._store.update_run(incident_id, run_id, mark_running)
        profile = build_incident_profile(
            self._jobs,
            self._store,
            incident,
            capsule,
            selection,
            run.instruction,
            run.previous_outcome_summary,
            smoke_evidence,
        )
        runner = AgentRunner(
            self._config,
            bus=self._bus,
            provider=self._provider,
            trace=self._trace,
        )
        outcome = await runner.run_and_capture(
            run.instruction,
            run_id=run_id,
            profile=profile,
            run_path=self._store.run_dir(incident_id, run_id),
        )
        current_incident = self._store.read_incident(incident_id)
        diagnosis = current_incident.diagnosis
        proposal = current_incident.proposal
        if outcome.status == "success" and diagnosis is None:
            outcome = RunOutcome(
                status="failed",
                result=outcome.result,
                reason="diagnosis_missing",
                initial_input_bytes=outcome.initial_input_bytes,
                peak_input_bytes=outcome.peak_input_bytes,
                budget_exhausted=outcome.budget_exhausted,
            )
        def mark_finished(current: IncidentRun) -> None:
            current.status = outcome.status
            current.reason = outcome.reason
            current.diagnosis = diagnosis.model_dump(mode="json") if diagnosis else None
            current.proposal = proposal.model_dump(mode="json") if proposal else None
            current.initial_input_bytes = outcome.initial_input_bytes
            current.peak_input_bytes = outcome.peak_input_bytes
            current.budget_exhausted = outcome.budget_exhausted

        self._store.update_run(incident_id, run_id, mark_finished)
        return outcome
