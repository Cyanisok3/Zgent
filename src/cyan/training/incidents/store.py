from __future__ import annotations

import os
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from cyan.training.incidents.models import (
    Diagnosis,
    Incident,
    PatchReceipt,
    Proposal,
)
from cyan.training.incidents.smoke import SmokeExecution, SmokeResult

_SAFE_ID_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)


# 校验用于目录名的标识符，阻止目录遍历和路径分隔符
def _validate_id(value: str) -> str:
    if not value or len(value) > 128 or value[0] in ".-" or any(
        char not in _SAFE_ID_CHARS for char in value
    ):
        raise ValueError(f"unsafe incident id: {value!r}")
    return value


# 在目标目录内原子替换文本文件，避免 daemon 中断留下半个 JSON
def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(content, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


# 将 Pydantic 模型以稳定格式原子写入 JSON
def _write_model(path: Path, model: BaseModel) -> None:
    _atomic_write(path, model.model_dump_json(indent=2) + "\n")


class IncidentStore:
    # 初始化 incident artifact 根目录
    def __init__(self, root: Path) -> None:
        self._root = root.expanduser()
        self._root.mkdir(parents=True, exist_ok=True)

    # 返回指定 incident 的安全目录路径
    def incident_dir(self, incident_id: str) -> Path:
        return self._root / _validate_id(incident_id)

    # 原子保存 incident 元数据
    def write_incident(self, incident: Incident) -> None:
        _write_model(self.incident_dir(incident.id) / "incident.json", incident)

    # 读取并校验 incident 元数据
    def read_incident(self, incident_id: str) -> Incident:
        path = self.incident_dir(incident_id) / "incident.json"
        return Incident.model_validate_json(path.read_text(encoding="utf-8"))

    # 原子保存结构化诊断
    def write_diagnosis(self, diagnosis: Diagnosis) -> None:
        _write_model(
            self.incident_dir(diagnosis.incident_id) / "diagnosis.json",
            diagnosis,
        )

    # 读取并校验结构化诊断
    def read_diagnosis(self, incident_id: str) -> Diagnosis:
        path = self.incident_dir(incident_id) / "diagnosis.json"
        return Diagnosis.model_validate_json(path.read_text(encoding="utf-8"))

    # 先保存 diff 再保存元数据，使 proposal.json 永远指向完整 patch
    def write_proposal(self, proposal: Proposal, patch: str) -> None:
        directory = self.incident_dir(proposal.incident_id)
        _atomic_write(directory / proposal.patch_path, patch)
        _write_model(directory / "proposal.json", proposal)

    # 读取并校验 proposal 元数据
    def read_proposal(self, incident_id: str) -> Proposal:
        path = self.incident_dir(incident_id) / "proposal.json"
        return Proposal.model_validate_json(path.read_text(encoding="utf-8"))

    # 读取 proposal 对应的 unified diff
    def read_patch(self, proposal: Proposal) -> str:
        path = self.incident_dir(proposal.incident_id) / proposal.patch_path
        return path.read_text(encoding="utf-8")

    # 返回 proposal diff 的绝对路径供 git apply 使用
    def patch_path(self, proposal: Proposal) -> Path:
        return (self.incident_dir(proposal.incident_id) / proposal.patch_path).resolve()

    # 仅清除未通过审批前校验的 proposal，不影响已保存的 diagnosis
    def clear_proposal(self, incident_id: str) -> None:
        directory = self.incident_dir(incident_id)
        for name in ("proposal.json", "proposal.diff"):
            (directory / name).unlink(missing_ok=True)

    # 保存 apply 后的文件哈希，用于 smoke 失败时安全反向应用
    def write_receipt(self, incident_id: str, receipt: PatchReceipt) -> None:
        _write_model(self.incident_dir(incident_id) / "apply.json", receipt)

    # 读取 apply 后的文件哈希凭据
    def read_receipt(self, incident_id: str) -> PatchReceipt:
        path = self.incident_dir(incident_id) / "apply.json"
        return PatchReceipt.model_validate_json(path.read_text(encoding="utf-8"))

    # 保存 smoke verifier 的真实执行结果
    def write_smoke_result(self, incident_id: str, result: SmokeResult) -> None:
        _write_model(self.incident_dir(incident_id) / "smoke.json", result)

    # 读取 smoke verifier 的真实执行结果
    def read_smoke_result(self, incident_id: str) -> SmokeResult:
        path = self.incident_dir(incident_id) / "smoke.json"
        return SmokeResult.model_validate_json(path.read_text(encoding="utf-8"))

    # 原子保存可恢复的 smoke 子进程身份与执行状态
    def write_smoke_execution(
        self,
        incident_id: str,
        execution: SmokeExecution,
    ) -> None:
        _write_model(
            self.incident_dir(incident_id) / "smoke-execution.json",
            execution,
        )

    # 读取可恢复的 smoke 子进程身份与执行状态
    def read_smoke_execution(self, incident_id: str) -> SmokeExecution:
        path = self.incident_dir(incident_id) / "smoke-execution.json"
        return SmokeExecution.model_validate_json(path.read_text(encoding="utf-8"))

    # 保存本轮 Agent 的日志证据读取预算与实际用量
    def write_evidence_usage(
        self,
        incident_id: str,
        *,
        bytes_read: int,
        byte_limit: int,
    ) -> None:
        _atomic_write(
            self.incident_dir(incident_id) / "evidence_usage.json",
            f'{{"bytes_read":{bytes_read},"byte_limit":{byte_limit}}}\n',
        )

    # 枚举当前 Job 目录中可恢复的 Incident，损坏项跳过
    def list_incidents(self) -> list[Incident]:
        incidents: list[Incident] = []
        for path in sorted(self._root.glob("*/incident.json")):
            try:
                incidents.append(
                    Incident.model_validate_json(path.read_text(encoding="utf-8"))
                )
            except (OSError, ValueError):
                continue
        return sorted(incidents, key=lambda item: item.updated_at, reverse=True)

    # 清除上一轮 Agent 的活动诊断和 proposal，避免误把旧制品当作新结果
    def clear_agent_artifacts(self, incident_id: str) -> None:
        directory = self.incident_dir(incident_id)
        (directory / "diagnosis.json").unlink(missing_ok=True)
        self.clear_proposal(incident_id)
