from __future__ import annotations

import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cyan.training.incidents.models import Diagnosis, Incident, PatchReceipt, Proposal
from cyan.training.incidents.smoke import SmokeExecution, SmokeResult

_SAFE_ID_CHARS = frozenset("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-")
RunTrigger = Literal["initial", "follow_up", "smoke_failed", "retry_failed", "recovery"]


class IncidentRun(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    attempt_id: str
    trigger: RunTrigger
    instruction: str
    status: str = "prepared"
    selected_evidence: list[dict[str, Any]] = Field(default_factory=list)
    stdout_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    previous_outcome_summary: dict[str, Any] | None = None
    initial_input_bytes: int = 0
    peak_input_bytes: int = 0
    budget_exhausted: bool = False
    scanned_bytes: int = 0
    selected_bytes: int = 0
    duplicates_removed: int = 0
    diagnosis: dict[str, Any] | None = None
    proposal: dict[str, Any] | None = None
    reason: str | None = None
    created_at: datetime
    updated_at: datetime


# 校验用于磁盘目录和文件名的标识符
def _validate_id(value: str) -> str:
    if (
        not value
        or len(value) > 128
        or value[0] in ".-"
        or any(char not in _SAFE_ID_CHARS for char in value)
    ):
        raise ValueError(f"unsafe incident id: {value!r}")
    return value


# 原子替换文本文件，避免 daemon 中断留下半个 JSON
def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# 将 Pydantic 模型稳定写入 JSON
def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(path, model.model_dump_json(indent=2) + "\n")


class IncidentStore:
    # 初始化单 Job 的 Incident artifact 根目录
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    # 返回指定 Incident 的安全目录
    def incident_dir(self, incident_id: str) -> Path:
        return self._root / _validate_id(incident_id)

    # 返回 Incident 快照路径
    def incident_path(self, incident_id: str) -> Path:
        return self.incident_dir(incident_id) / "incident.json"

    # 原子保存 Incident 当前快照
    def write_incident(self, incident: Incident) -> None:
        _write_model(self.incident_path(incident.id), incident)

    # 读取并校验 Incident 当前快照
    def read_incident(self, incident_id: str) -> Incident:
        return Incident.model_validate_json(
            self.incident_path(incident_id).read_text(encoding="utf-8")
        )

    # 在单次原子写入中更新 Incident 嵌套结果
    def update_incident(self, incident_id: str, mutator: Callable[[Incident], None]) -> Incident:
        incident = self.read_incident(incident_id)
        mutator(incident)
        incident.updated_at = datetime.now(UTC)
        self.write_incident(incident)
        return incident

    # 在后台运行启动前写入 prepared run 记录
    def create_run(
        self,
        incident_id: str,
        run_id: str,
        trigger: RunTrigger,
        instruction: str,
        *,
        attempt_id: str | None = None,
        previous_outcome_summary: dict[str, Any] | None = None,
    ) -> IncidentRun:
        now = datetime.now(UTC)
        run = IncidentRun(
            run_id=run_id,
            attempt_id=attempt_id or self.read_incident(incident_id).attempt_id,
            trigger=trigger,
            instruction=instruction,
            previous_outcome_summary=previous_outcome_summary,
            created_at=now,
            updated_at=now,
        )
        _write_model(self.run_dir(incident_id, run_id) / "run.json", run)
        return run

    # 返回单轮运行记录目录
    def run_dir(self, incident_id: str, run_id: str) -> Path:
        _validate_id(run_id)
        return self.incident_dir(incident_id) / "runs" / run_id

    # 读取单轮有限运行记录
    def read_run(self, incident_id: str, run_id: str) -> IncidentRun:
        path = self.run_dir(incident_id, run_id) / "run.json"
        return IncidentRun.model_validate_json(path.read_text(encoding="utf-8"))

    # 原子更新单轮状态和指标
    def update_run(
        self, incident_id: str, run_id: str, mutator: Callable[[IncidentRun], None]
    ) -> IncidentRun:
        run = self.read_run(incident_id, run_id)
        mutator(run)
        run.updated_at = datetime.now(UTC)
        _write_model(self.run_dir(incident_id, run_id) / "run.json", run)
        return run

    # 将结构化诊断嵌入 Incident 快照
    def write_diagnosis(self, diagnosis: Diagnosis) -> None:
        self.update_incident(
            diagnosis.incident_id, lambda incident: setattr(incident, "diagnosis", diagnosis)
        )

    # 读取 Incident 中的结构化诊断
    def read_diagnosis(self, incident_id: str) -> Diagnosis:
        diagnosis = self.read_incident(incident_id).diagnosis
        if diagnosis is None:
            raise FileNotFoundError(f"diagnosis not found: {incident_id}")
        return diagnosis

    # 先写完整 diff，再原子更新 Incident proposal 元数据
    def write_proposal(self, proposal: Proposal, patch: str) -> None:
        directory = self.incident_dir(proposal.incident_id)
        _atomic_write(directory / "proposal.diff", patch)
        self.update_incident(
            proposal.incident_id, lambda incident: setattr(incident, "proposal", proposal)
        )

    # 读取 Incident 中当前 proposal
    def read_proposal(self, incident_id: str) -> Proposal:
        proposal = self.read_incident(incident_id).proposal
        if proposal is None:
            raise FileNotFoundError(f"proposal not found: {incident_id}")
        return proposal

    # 读取 proposal 对应的完整 diff
    def read_patch(self, proposal: Proposal) -> str:
        return (self.incident_dir(proposal.incident_id) / "proposal.diff").read_text(
            encoding="utf-8"
        )

    # 返回 proposal diff 的绝对路径供 PatchService 使用
    def patch_path(self, proposal: Proposal) -> Path:
        return (self.incident_dir(proposal.incident_id) / "proposal.diff").resolve()

    # 先清空元数据，再删除 diff，避免快照继续指向已删除内容
    def clear_proposal(self, incident_id: str) -> None:
        self.update_incident(incident_id, lambda incident: setattr(incident, "proposal", None))
        (self.incident_dir(incident_id) / "proposal.diff").unlink(missing_ok=True)

    # 将 apply 凭据嵌入 Incident 快照
    def write_receipt(self, incident_id: str, receipt: PatchReceipt) -> None:
        self.update_incident(
            incident_id, lambda incident: setattr(incident, "apply_receipt", receipt)
        )

    # 读取 apply 凭据
    def read_receipt(self, incident_id: str) -> PatchReceipt:
        receipt = self.read_incident(incident_id).apply_receipt
        if receipt is None:
            raise FileNotFoundError(f"receipt not found: {incident_id}")
        return receipt

    # 将 smoke 结果嵌入 Incident 快照
    def write_smoke_result(self, incident_id: str, result: SmokeResult) -> None:
        self.update_incident(
            incident_id, lambda incident: setattr(incident, "smoke_result", result)
        )

    # 读取 smoke 结果
    def read_smoke_result(self, incident_id: str) -> SmokeResult:
        result = self.read_incident(incident_id).smoke_result
        if result is None:
            raise FileNotFoundError(f"smoke result not found: {incident_id}")
        return result

    # 在 smoke_running 对外可见前持久化进程身份
    def write_smoke_execution(self, incident_id: str, execution: SmokeExecution) -> None:
        self.update_incident(
            incident_id, lambda incident: setattr(incident, "smoke_execution", execution)
        )

    # 读取 smoke 进程身份
    def read_smoke_execution(self, incident_id: str) -> SmokeExecution:
        execution = self.read_incident(incident_id).smoke_execution
        if execution is None:
            raise FileNotFoundError(f"smoke execution not found: {incident_id}")
        return execution

    # 枚举当前 Job 可恢复的 Incident，旧 schema 直接忽略
    def list_incidents(self) -> list[Incident]:
        incidents: list[Incident] = []
        for path in sorted(self._root.glob("*/incident.json")):
            try:
                incidents.append(Incident.model_validate_json(path.read_text(encoding="utf-8")))
            except (OSError, ValueError):
                continue
        return sorted(incidents, key=lambda item: item.updated_at, reverse=True)

    # 清理本轮诊断和 proposal，保留 Incident 身份与可恢复 smoke 状态
    def clear_agent_artifacts(self, incident_id: str) -> None:
        def clear(incident: Incident) -> None:
            incident.diagnosis = None
            incident.proposal = None
            incident.active_proposal_id = None

        self.update_incident(incident_id, clear)
        (self.incident_dir(incident_id) / "proposal.diff").unlink(missing_ok=True)
