from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cyan.core.incidents.models import (
    Diagnosis,
    DiagnosisCategory,
    EvidenceRef,
    Proposal,
)
from cyan.core.incidents.patch import (
    PatchError,
    PatchService,
    build_proposal_files,
    parse_unified_diff,
    sha256_bytes,
)
from cyan.core.incidents.store import IncidentStore
from cyan.core.tools.base import BaseTool, ToolResult

_MAX_PATCH_BYTES = 1024 * 1024
EvidenceValidator = Callable[[EvidenceRef], str | None]


class ProposePatchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    patch: str = Field(min_length=1)
    diagnosis_id: str | None = Field(default=None, min_length=1, max_length=128)
    evidence: list[EvidenceRef] = Field(default_factory=list, max_length=32)


class SubmitDiagnosisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: DiagnosisCategory
    summary: str = Field(min_length=1, max_length=4000)
    root_cause: str = Field(min_length=1, max_length=8000)
    evidence: list[EvidenceRef] = Field(min_length=1, max_length=32)
    confidence: float = Field(ge=0.0, le=1.0)


# 将业务校验错误统一转换为不可重试的工具结果
def _validation_error(message: str) -> ToolResult:
    return ToolResult(
        content=message,
        is_error=True,
        error_type="schema_error",
    )


# 使用 harness 注入的登记表校验所有 evidence 引用
def _evidence_error(
    evidence: list[EvidenceRef],
    validator: EvidenceValidator | None,
) -> str | None:
    if validator is None:
        return None
    for item in evidence:
        error = validator(item)
        if error is not None:
            return error
    return None


class ProposePatchTool(BaseTool):
    params_model = ProposePatchParams
    name = "propose_patch"
    description = (
        "Validate and save a unified diff as an incident artifact. "
        "This tool never modifies the workspace."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "patch": {"type": "string"},
            "diagnosis_id": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": ["stdout", "stderr", "workspace"],
                        },
                        "reference": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "reference", "description"],
                },
            },
        },
        "required": ["patch"],
    }

    # 将工具绑定到单个 incident 和不可变 workspace root
    def __init__(
        self,
        store: IncidentStore,
        incident_id: str,
        workspace_root: Path,
        *,
        evidence_validator: EvidenceValidator | None = None,
        patch_service: PatchService | None = None,
    ) -> None:
        self._store = store
        self._incident_id = incident_id
        self._workspace_root = workspace_root.resolve(strict=True)
        self._evidence_validator = evidence_validator
        self._patch_service = patch_service

    # 校验 diff 并只向 incident artifact 目录写 proposal
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed_params = ProposePatchParams.model_validate(params)
        evidence_error = _evidence_error(
            parsed_params.evidence,
            self._evidence_validator,
        )
        if evidence_error is not None:
            return _validation_error(evidence_error)
        patch = parsed_params.patch
        if not patch.endswith("\n"):
            patch += "\n"
        patch_bytes = patch.encode("utf-8")
        if len(patch_bytes) > _MAX_PATCH_BYTES:
            return _validation_error(
                f"patch is too large: {len(patch_bytes)} bytes "
                f"(limit {_MAX_PATCH_BYTES})"
            )
        try:
            parsed_files = parse_unified_diff(patch)
            files = build_proposal_files(self._workspace_root, parsed_files)
            if parsed_params.diagnosis_id is not None:
                diagnosis = self._store.read_diagnosis(self._incident_id)
                if diagnosis.id != parsed_params.diagnosis_id:
                    raise PatchError("diagnosis_id does not match diagnosis artifact")
        except (OSError, PatchError, ValueError) as exc:
            return _validation_error(str(exc))

        proposal = Proposal(
            id=uuid4().hex,
            incident_id=self._incident_id,
            diagnosis_id=parsed_params.diagnosis_id,
            patch_sha256=sha256_bytes(patch_bytes),
            files=files,
            evidence=parsed_params.evidence,
            created_at=datetime.now(UTC),
        )
        self._store.write_proposal(proposal, patch)
        if self._patch_service is not None:
            try:
                await self._patch_service.check(
                    proposal,
                    self._store.patch_path(proposal),
                )
            except (OSError, PatchError, ValueError) as exc:
                self._store.clear_proposal(self._incident_id)
                return _validation_error(str(exc))
        return ToolResult(content=proposal.model_dump_json())


class SubmitDiagnosisTool(BaseTool):
    params_model = SubmitDiagnosisParams
    name = "submit_diagnosis"
    description = (
        "Save one structured ML training failure diagnosis with evidence references."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "category": {
                "type": "string",
                "enum": [
                    "data",
                    "config",
                    "shape",
                    "dtype",
                    "device",
                    "memory",
                    "environment",
                    "checkpoint",
                    "io",
                    "runtime",
                    "unknown",
                ],
            },
            "summary": {"type": "string"},
            "root_cause": {"type": "string"},
            "evidence": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {
                            "type": "string",
                            "enum": ["stdout", "stderr", "workspace"],
                        },
                        "reference": {"type": "string"},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "reference", "description"],
                },
            },
            "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        },
        "required": [
            "category",
            "summary",
            "root_cause",
            "evidence",
            "confidence",
        ],
    }

    # 将诊断工具绑定到单个 incident artifact store
    def __init__(
        self,
        store: IncidentStore,
        incident_id: str,
        *,
        evidence_validator: EvidenceValidator | None = None,
    ) -> None:
        self._store = store
        self._incident_id = incident_id
        self._evidence_validator = evidence_validator

    # 校验 taxonomy 和证据后保存结构化诊断
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed = SubmitDiagnosisParams.model_validate(params)
        evidence_error = _evidence_error(parsed.evidence, self._evidence_validator)
        if evidence_error is not None:
            return _validation_error(evidence_error)
        diagnosis = Diagnosis(
            id=uuid4().hex,
            incident_id=self._incident_id,
            category=parsed.category,
            summary=parsed.summary,
            root_cause=parsed.root_cause,
            evidence=parsed.evidence,
            confidence=parsed.confidence,
            created_at=datetime.now(UTC),
        )
        self._store.write_diagnosis(diagnosis)
        return ToolResult(content=diagnosis.model_dump_json())
