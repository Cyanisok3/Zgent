from __future__ import annotations

import re
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from cyan.agent.tools.base import BaseTool, ToolResult
from cyan.training.incidents.models import (
    Diagnosis,
    DiagnosisCategory,
    EvidenceRef,
    Proposal,
)
from cyan.training.incidents.patch import (
    PatchError,
    PatchService,
    build_proposal_files,
    build_replacement_diff,
    parse_unified_diff,
    resolve_workspace_path,
    sha256_bytes,
)
from cyan.training.incidents.store import IncidentStore

_MAX_FILE_BYTES = 1024 * 1024
_MAX_FEEDBACK_BYTES = 8 * 1024
_MAX_CANDIDATE_LINES = 20
_WORKSPACE_REFERENCE = re.compile(
    r"^(?P<path>.+)@sha256:(?P<sha256>[0-9a-f]{64})#L"
    r"(?P<start>[1-9][0-9]*)(?:-L(?P<end>[1-9][0-9]*))?$"
)
EvidenceValidator = Callable[[EvidenceRef], str | None]


class ProposePatchParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(min_length=1, max_length=1024)
    search: str = Field(min_length=1, max_length=_MAX_FILE_BYTES)
    replace: str = Field(max_length=_MAX_FILE_BYTES)
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


# 解析与目标文件匹配的 workspace evidence 身份和行范围
def _workspace_evidence_range(
    evidence: list[EvidenceRef],
    path: str,
    digest: str,
) -> tuple[int, int] | None:
    for item in evidence:
        if item.source != "workspace":
            continue
        match = _WORKSPACE_REFERENCE.fullmatch(item.reference)
        if match is None:
            continue
        if match.group("path") != path or match.group("sha256") != digest:
            continue
        start = int(match.group("start"))
        end = int(match.group("end") or start)
        if end >= start:
            return start, end
    return None


# 从 evidence 行范围截取有界真实源码作为唯一一次校正反馈
def _evidence_snippet(text: str, start: int, end: int) -> str:
    lines = text.splitlines(keepends=True)
    snippet = "".join(lines[start - 1 : end])
    raw = snippet.encode("utf-8")
    if len(raw) <= _MAX_FEEDBACK_BYTES:
        return snippet
    bounded = raw[:_MAX_FEEDBACK_BYTES].decode("utf-8", errors="ignore")
    return f"{bounded}\n[truncated]"


# 返回所有精确匹配的 1-based 起始行并限制反馈大小
def _match_start_lines(text: str, search: str) -> list[int]:
    lines: list[int] = []
    offset = 0
    while len(lines) < _MAX_CANDIDATE_LINES:
        match = text.find(search, offset)
        if match < 0:
            break
        lines.append(text.count("\n", 0, match) + 1)
        offset = match + max(1, len(search))
    return lines


class ProposePatchTool(BaseTool):
    params_model = ProposePatchParams
    name = "propose_patch"
    description = (
        "Replace one exact, unique source block in one existing text file and save "
        "a generated unified diff as an incident artifact. This tool never modifies "
        "the workspace."
    )
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "path": {"type": "string"},
            "search": {
                "type": "string",
                "description": "Exact contiguous text copied from the current file.",
            },
            "replace": {
                "type": "string",
                "description": "Replacement text; may be empty.",
            },
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
        "required": ["path", "search", "replace"],
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
        self._match_feedback_sent = False

    # 校验精确替换并只向 incident artifact 目录写生成后的 proposal
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        parsed_params = ProposePatchParams.model_validate(params)
        evidence_error = _evidence_error(
            parsed_params.evidence,
            self._evidence_validator,
        )
        if evidence_error is not None:
            return _validation_error(evidence_error)
        try:
            try:
                diagnosis = self._store.read_diagnosis(self._incident_id)
            except FileNotFoundError as exc:
                raise PatchError(
                    "submit_diagnosis must succeed before propose_patch"
                ) from exc
            target = resolve_workspace_path(
                self._workspace_root,
                parsed_params.path,
            )
            raw = target.read_bytes()
            if len(raw) > _MAX_FILE_BYTES:
                raise PatchError(
                    f"target file is too large: {len(raw)} bytes "
                    f"(limit {_MAX_FILE_BYTES})"
                )
            if b"\x00" in raw:
                raise PatchError("binary files are not patchable")
            try:
                original = raw.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise PatchError("target file is not valid UTF-8") from exc
            digest = sha256_bytes(raw)
            evidence_range = _workspace_evidence_range(
                parsed_params.evidence,
                parsed_params.path,
                digest,
            )
            if evidence_range is None:
                raise PatchError(
                    "proposal requires workspace evidence for the target path "
                    "and current SHA-256"
                )
            if parsed_params.search == parsed_params.replace:
                raise PatchError("replacement does not change the file")
            match_count = original.count(parsed_params.search)
            if match_count != 1:
                if self._match_feedback_sent:
                    raise PatchError(
                        "exact SEARCH still does not match uniquely; keep the diagnosis "
                        "and stop proposing a patch"
                    )
                self._match_feedback_sent = True
                if match_count == 0:
                    start, end = evidence_range
                    snippet = _evidence_snippet(original, start, end)
                    raise PatchError(
                        "SEARCH has no exact match. Copy exact text from this observed "
                        f"source range and retry once:\n{snippet}"
                    )
                candidate_lines = _match_start_lines(
                    original,
                    parsed_params.search,
                )
                suffix = " (truncated)" if match_count > len(candidate_lines) else ""
                raise PatchError(
                    f"SEARCH matches {match_count} locations at starting lines "
                    f"{candidate_lines}{suffix}; expand SEARCH to make it unique and "
                    "retry once"
                )
            updated = original.replace(
                parsed_params.search,
                parsed_params.replace,
                1,
            )
            patch = build_replacement_diff(
                parsed_params.path,
                original,
                updated,
            )
            patch_bytes = patch.encode("utf-8")
            if len(patch_bytes) > _MAX_FILE_BYTES:
                raise PatchError(
                    f"generated patch is too large: {len(patch_bytes)} bytes "
                    f"(limit {_MAX_FILE_BYTES})"
                )
            parsed_files = parse_unified_diff(patch)
            files = build_proposal_files(self._workspace_root, parsed_files)
        except (OSError, PatchError, ValueError) as exc:
            return _validation_error(str(exc))

        proposal = Proposal(
            id=uuid4().hex,
            incident_id=self._incident_id,
            diagnosis_id=diagnosis.id,
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
