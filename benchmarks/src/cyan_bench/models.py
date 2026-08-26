from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Split = Literal["dev", "test"]
FailureStage = Literal["startup", "mid_run", "finalization"]
OriginKind = Literal["reproduced_upstream", "provenance_preserving_port"]
Hardware = Literal["cpu", "mps"]
Variant = Literal["buggy", "fixed", "control"]
Baseline = Literal["full_native", "tail_32", "bm25_32", "cyan_selector_32", "oracle_32"]
ControlRole = Literal["short_quiet", "long_clean", "warning_heavy"]
TrainingDomain = Literal["llm", "general_ml"]
DatasetVersion = Literal["formal-v1", "formal-v2"]


class CaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    title: str = Field(min_length=1, max_length=200)
    dataset_version: DatasetVersion = "formal-v1"
    split: Split
    framework: str = Field(min_length=1, max_length=80)
    training_domain: TrainingDomain
    fault_family: str = Field(min_length=1, max_length=80)
    mechanism_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    failure_stage: FailureStage
    origin_kind: OriginKind
    patchable: bool
    repo_url: str = Field(pattern=r"^https://github\.com/")
    repo_sha: str = Field(pattern=r"^[0-9a-f]{40}$")
    issue_url: str = Field(pattern=r"^https://")
    fix_url: str = Field(pattern=r"^https://")
    license: str = Field(min_length=1, max_length=80)
    env_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,79}$")
    argv: list[str] = Field(min_length=1)
    cwd: str = "."
    timeout_s: int = Field(ge=1, le=7200)
    hardware: Hardware = "cpu"
    milestone_anchor: str | None = Field(default=None, max_length=1000)
    control_role: ControlRole | None = None
    environment: dict[str, str] = Field(default_factory=dict)
    data_sha256: dict[str, str] = Field(default_factory=dict)
    weight_sha256: dict[str, str] = Field(default_factory=dict)

    # 保证执行目录与命令均不能逃出临时工作区
    @model_validator(mode="after")
    def _validate_paths_and_command(self) -> CaseManifest:
        cwd = Path(self.cwd)
        if cwd.is_absolute() or ".." in cwd.parts:
            raise ValueError("cwd must stay inside the prepared workspace")
        if any(not item or "\x00" in item for item in self.argv):
            raise ValueError("argv entries must be non-empty and contain no NUL")
        if self.failure_stage != "startup" and not self.milestone_anchor:
            raise ValueError("mid_run and finalization cases require milestone_anchor")
        return self


class EvidenceAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["stdout", "stderr"]
    literal: str | None = Field(default=None, min_length=1)
    regex: str | None = Field(default=None, min_length=1)

    # 每个证据锚点必须且只能选择 literal 或 regex
    @model_validator(mode="after")
    def _validate_matcher(self) -> EvidenceAnchor:
        if (self.literal is None) == (self.regex is None):
            raise ValueError("anchor requires exactly one of literal or regex")
        return self


class DiagnosisOracle(BaseModel):
    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=80)
    culprit: list[str] = Field(min_length=1)
    causal_mechanism: list[str] = Field(min_length=1)


class ExpectedOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1]
    diagnosis: DiagnosisOracle
    required_groups: list[list[EvidenceAnchor]] = Field(min_length=1)
    # formal-v2 必须显式提供；formal-v1 允许缺失
    causal_support: Literal["direct", "inferred"] | None = None
    patch_recommended: bool | None = None

    # 禁止空的 any-of 证据组，否则 recall 没有确定含义
    @model_validator(mode="after")
    def _validate_groups(self) -> ExpectedOutcome:
        if any(not group for group in self.required_groups):
            raise ValueError("required_groups cannot contain an empty group")
        return self


class ResolvedAnchor(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group: int = Field(ge=0)
    source: Literal["stdout", "stderr"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    matcher: str


class ProcessCapture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    case_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    environment_lock_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    variant: Variant
    repeat: int = Field(ge=1)
    argv: list[str]
    cwd: str
    returncode: int | None
    timed_out: bool
    duration_seconds: float = Field(ge=0)
    stdout_path: str
    stderr_path: str
    stdout_bytes: int = Field(ge=0)
    stderr_bytes: int = Field(ge=0)
    stdout_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    stderr_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime


class AdmissionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    admitted: bool
    reasons: list[str]
    captures: list[str]
    created_at: datetime


class SelectionReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["stdout", "stderr"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    kind: str
    selection_reason: str


class SelectionArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    baseline: Baseline
    repeat: int = Field(ge=1)
    content_path: str
    selected_bytes: int = Field(ge=0)
    unique_selected_bytes: int = Field(ge=0)
    scanned_bytes: int = Field(ge=0)
    latency_seconds: float = Field(ge=0)
    references: list[SelectionReference]
    context_overflow: bool = False


class DiagnosisAnswer(BaseModel):
    """旧 formal-v1 宽松模型：diagnosis 为自由字典。"""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["fault", "no_fault"]
    diagnosis: dict[str, object] | None
    patch_recommended: bool

    # no_fault 必须对应空诊断与不建议修复
    @model_validator(mode="after")
    def _validate_abstention(self) -> DiagnosisAnswer:
        if self.verdict == "no_fault" and (self.diagnosis is not None or self.patch_recommended):
            raise ValueError("no_fault requires null diagnosis and patch_recommended=false")
        if self.verdict == "fault" and self.diagnosis is None:
            raise ValueError("fault requires diagnosis")
        return self


class DiagnosisEvidenceRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: Literal["stdout", "stderr"]
    start: int = Field(ge=0)
    end: int = Field(gt=0)

    # 拒绝空区间或反向区间
    @model_validator(mode="after")
    def _validate_range(self) -> DiagnosisEvidenceRef:
        if self.end <= self.start:
            raise ValueError("evidence end must be greater than start")
        return self


class StructuredDiagnosis(BaseModel):
    """formal-v2 严格诊断：强制因果支撑、证据引用与补丁意图。"""

    model_config = ConfigDict(extra="forbid")

    category: str = Field(min_length=1, max_length=80)
    culprit: str = Field(min_length=1, max_length=1000)
    causal_mechanism: str = Field(min_length=1, max_length=4000)
    causal_support: Literal["direct", "inferred"]
    evidence: list[DiagnosisEvidenceRef] = Field(min_length=1, max_length=32)


class DiagnosisAnswerV2(BaseModel):
    """formal-v2 严格响应模型：诊断必须结构化且缺失字段时直接失败。"""

    model_config = ConfigDict(extra="forbid")

    verdict: Literal["fault", "no_fault"]
    diagnosis: StructuredDiagnosis | None
    patch_recommended: bool

    # no_fault 必须对应空诊断与不建议修复
    @model_validator(mode="after")
    def _validate_abstention(self) -> DiagnosisAnswerV2:
        if self.verdict == "no_fault" and (self.diagnosis is not None or self.patch_recommended):
            raise ValueError("no_fault requires null diagnosis and patch_recommended=false")
        if self.verdict == "fault" and self.diagnosis is None:
            raise ValueError("fault requires diagnosis")
        return self


class DiagnosisRunArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    baseline: Baseline
    repeat: int = Field(ge=1)
    is_control: bool
    prompt_version: str = "legacy"
    status: Literal["success", "schema_error", "context_overflow", "infrastructure_error"]
    model_requested: str
    model_resolved: str | None = None
    response_id: str | None = None
    # 新运行解析为严格 V2；旧 formal-v1 artifact 回落为宽松 V1 模型
    answer: DiagnosisAnswerV2 | DiagnosisAnswer | None = None
    raw_response_path: str | None = None
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    cache_read_input_tokens: int | None = Field(default=None, ge=0)
    cache_creation_input_tokens: int | None = Field(default=None, ge=0)
    duration_seconds: float = Field(ge=0)
    transport_attempts: int = Field(ge=1, le=3)
    error: str | None = None
    created_at: datetime


class IncidentBenchmarkArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    case_id: str
    repeat: int = Field(ge=1)
    is_control: bool
    prompt_version: str = "legacy"
    job_id: str
    incident_id: str | None = None
    final_job_status: str
    final_incident_status: str | None = None
    spurious_incident: bool = False
    diagnosis_present: bool = False
    diagnosis_category: str | None = None
    diagnosis_root_cause: str | None = None
    diagnosis_causal_support: Literal["direct", "inferred"] | None = None
    diagnosis_patch_recommended: bool | None = None
    diagnosis_evidence_refs: list[dict[str, object]] = Field(
        default_factory=list,
        max_length=32,
    )
    proposal_present: bool = False
    proposal_valid: bool = False
    unsafe_proposal: bool = False
    correct_patch_abstention: bool = False
    missed_patch_opportunity: bool = False
    abstention_gate_violated: bool = False
    resolved: bool = False
    capsule_tail_bytes: int = Field(ge=0)
    selector_selected_bytes: int = Field(ge=0)
    unique_evidence_bytes: int = Field(ge=0)
    peak_input_bytes: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    error: str | None = None
    created_at: datetime
