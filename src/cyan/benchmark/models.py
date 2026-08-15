from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from cyan.core.incidents.models import FailureCapsule, RecoveryKind

BenchmarkTier = Literal["cyan_core", "external_generalization", "scale_stress"]
BenchmarkSplit = Literal["train", "dev", "test", "external", "stress"]
FactImportance = Literal["essential", "supporting"]
FactProvenance = Literal[
    "injected",
    "automatic_candidate",
    "human_confirmed",
    "llm_suggested",
]
EvidenceStream = Literal["stdout", "stderr"]
StrategyName = Literal["current_agent", "retrieval_skill", "readonly_subagents"]
DiagnosisGrade = Literal["correct", "partial", "incorrect", "abstain"]

_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class LogArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    size: int = Field(ge=0)

    # 拒绝绝对路径和目录逃逸，使语料可安全移动
    @field_validator("path")
    @classmethod
    def _path_must_be_relative(cls, value: str) -> str:
        from pathlib import PurePosixPath

        path = PurePosixPath(value)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("log path must be corpus-relative")
        return value


class SourceProvenance(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    repository: str | None = None
    issue_url: str | None = None
    failing_commit: str | None = None
    fixing_commit: str | None = None
    failing_revision: str | None = None
    fixing_revision: str | None = None
    license: str | None = None
    upstream_case_id: str | None = None
    artifact_sha256: str | None = Field(default=None, pattern=_SHA256_PATTERN)
    runtime_image: str | None = None
    historical: bool = False


class ReplayReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    failing_argv: list[str] = Field(min_length=1)
    fixed_argv: list[str] = Field(min_length=1)
    failing_returncode: int
    fixed_returncode: int
    failing_log_sha256: str = Field(pattern=_SHA256_PATTERN)
    fixed_log_sha256: str = Field(pattern=_SHA256_PATTERN)

    # 要求凭证真正证明一次失败和一次成功重放
    @model_validator(mode="after")
    def _validate_replay(self) -> ReplayReceipt:
        if self.failing_returncode == 0:
            raise ValueError("failing replay must have a non-zero return code")
        if self.fixed_returncode != 0:
            raise ValueError("fixed replay must have return code zero")
        return self


class GoldFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    fact_id: str = Field(min_length=1, max_length=128)
    importance: FactImportance
    stream: EvidenceStream
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    description: str = Field(min_length=1, max_length=2000)
    provenance: FactProvenance
    review_passes: int = Field(default=0, ge=0, le=8)

    # 保证证据区间为非空半开区间
    @model_validator(mode="after")
    def _validate_range(self) -> GoldFact:
        if self.byte_end <= self.byte_start:
            raise ValueError("gold fact byte_end must be greater than byte_start")
        return self


class CaseManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    case_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    tier: BenchmarkTier
    workload: str = Field(min_length=1, max_length=128)
    split: BenchmarkSplit
    template_id: str = Field(min_length=1, max_length=128)
    failure_kind: Literal[
        "launch_error",
        "process_exit",
        "supervisor_error",
        "contract_violation",
    ]
    phase: Literal["preflight", "main", "postflight"] | None = None
    expected_recovery_kind: RecoveryKind
    capsule: FailureCapsule
    logs: dict[EvidenceStream, LogArtifact]
    fixed_logs: dict[EvidenceStream, LogArtifact] = Field(default_factory=dict)
    source: SourceProvenance = Field(default_factory=SourceProvenance)
    replay: ReplayReceipt | None = None
    gold_facts: list[GoldFact] = Field(default_factory=list)
    expected_diagnosis_terms: list[str] = Field(default_factory=list, max_length=64)
    root_cause_rubric: list[str] = Field(default_factory=list, max_length=64)
    gold_review_status: Literal["draft", "approved"] = "draft"
    stress_position: Literal["front", "middle", "tail"] | None = None
    annotation_notes: str | None = Field(default=None, max_length=4000)

    # 校验 tier/split、唯一事实和日志范围等跨字段约束
    @model_validator(mode="after")
    def _validate_case(self) -> CaseManifest:
        allowed_splits = {
            "cyan_core": {"train", "dev", "test"},
            "external_generalization": {"external"},
            "scale_stress": {"stress"},
        }
        if self.split not in allowed_splits[self.tier]:
            raise ValueError(f"split {self.split!r} is invalid for tier {self.tier!r}")
        ids = [fact.fact_id for fact in self.gold_facts]
        if len(ids) != len(set(ids)):
            raise ValueError("gold fact ids must be unique within a case")
        for fact in self.gold_facts:
            artifact = self.logs.get(fact.stream)
            if artifact is None:
                raise ValueError(f"missing {fact.stream} log for fact {fact.fact_id}")
            if fact.byte_end > artifact.size:
                raise ValueError(f"fact {fact.fact_id} exceeds {fact.stream} size")
            if (
                self.split == "test"
                and self.gold_review_status == "approved"
                and fact.review_passes < 2
            ):
                raise ValueError("approved test gold requires two review passes")
            if (
                self.split == "test"
                and self.gold_review_status == "approved"
                and fact.provenance != "human_confirmed"
            ):
                raise ValueError("approved test gold must be human-confirmed")
        if self.tier == "cyan_core":
            if self.replay is None:
                raise ValueError("cyan_core cases require a replay receipt")
            if set(self.fixed_logs) != {"stdout", "stderr"}:
                raise ValueError("cyan_core cases require fixed stdout/stderr artifacts")
        return self


class EvidenceItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    stream: EvidenceStream
    byte_start: int = Field(ge=0)
    byte_end: int = Field(gt=0)
    score: float = 0.0
    reason: str = Field(min_length=1, max_length=512)
    cost_bytes: int = Field(ge=0)

    # 保证返回证据区间非空且成本不超过区间大小
    @model_validator(mode="after")
    def _validate_item(self) -> EvidenceItem:
        length = self.byte_end - self.byte_start
        if length <= 0:
            raise ValueError("evidence item must be a non-empty half-open range")
        if self.cost_bytes > length:
            raise ValueError("evidence cost may not exceed its byte range")
        return self


class EvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    case_id: str
    method: str
    initial_items: list[EvidenceItem] = Field(default_factory=list)
    items: list[EvidenceItem] = Field(default_factory=list)
    byte_budget: int = Field(ge=0)
    returned_bytes: int = Field(ge=0)
    scanned_bytes: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
    peak_rss_bytes: int = Field(default=0, ge=0)
    rss_delta_bytes: int = Field(default=0, ge=0)
    abstained: bool = False
    metadata: dict[str, str | int | float | bool] = Field(default_factory=dict)

    # 复核 bundle 的累计成本和预算，防止评测器接受越界结果
    @model_validator(mode="after")
    def _validate_budget(self) -> EvidenceBundle:
        actual = sum(item.cost_bytes for item in self.items)
        if actual != self.returned_bytes:
            raise ValueError("returned_bytes does not match evidence item costs")
        if actual > self.byte_budget:
            raise ValueError("evidence bundle exceeds byte budget")
        if any(item.cost_bytes > 32 * 1024 for item in self.items):
            raise ValueError("one evidence item exceeds the 32 KiB limit")
        if self.abstained and self.items:
            raise ValueError("an abstained bundle may not contain retrieved items")
        return self


class RetrievalMetrics(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    essential_recall_at_64k: float = Field(ge=0.0, le=1.0)
    essential_recall_at_128k: float = Field(ge=0.0, le=1.0)
    essential_recall_at_256k: float = Field(ge=0.0, le=1.0)
    supporting_recall_at_256k: float = Field(ge=0.0, le=1.0)
    ndcg: float = Field(ge=0.0, le=1.0)
    first_relevant_evidence_bytes: int | None = Field(default=None, ge=0)
    first_relevant_missed: bool = False
    evidence_density: float = Field(ge=0.0, le=1.0)
    redundancy_rate: float = Field(ge=0.0, le=1.0)
    abstention_correct: bool | None = None
    elapsed_ms: float = Field(ge=0.0)
    returned_bytes: int = Field(default=0, ge=0)
    scanned_bytes: int = Field(ge=0)
    peak_rss_bytes: int = Field(default=0, ge=0)
    rss_delta_bytes: int = Field(default=0, ge=0)


class ScoredCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str
    tier: BenchmarkTier
    split: BenchmarkSplit
    method: str
    metrics: RetrievalMetrics
    workload: str | None = None
    log_bytes: int = Field(default=0, ge=0)
    evidence_position: Literal["front", "middle", "tail"] | None = None


class BenchmarkRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    created_at: datetime
    code_revision: str | None = None
    dataset_fingerprint: str = Field(pattern=_SHA256_PATTERN)
    method: str
    seed: int
    byte_budget: int
    max_item_bytes: int
    bundles: list[EvidenceBundle]


class AgentStrategyResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    case_id: str
    tier: BenchmarkTier | None = None
    split: BenchmarkSplit | None = None
    strategy: StrategyName
    model: str
    diagnosis_submitted: bool
    root_cause: str | None = None
    recovery_kind: RecoveryKind | None = None
    evidence_references: list[str] = Field(default_factory=list)
    evidence_bundle: EvidenceBundle
    diagnosis_term_recall: float = Field(ge=0.0, le=1.0)
    essential_evidence_recall: float = Field(ge=0.0, le=1.0)
    retrieved_essential_evidence_recall: float = Field(default=0.0, ge=0.0, le=1.0)
    recovery_kind_correct: bool
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float | None = Field(default=None, ge=0.0)
    tool_calls: int = Field(ge=0)
    elapsed_ms: float = Field(ge=0.0)
    status: str


class AgentReviewEntry(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    blind_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    grade: DiagnosisGrade
    notes: str | None = Field(default=None, max_length=2000)


class AgentReviewSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal[1] = 1
    reviewer_id: str = Field(min_length=1, max_length=128)
    entries: list[AgentReviewEntry]
