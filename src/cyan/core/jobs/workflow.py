from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

WorkflowPhase = Literal["preflight", "main", "postflight"]
ArtifactRole = Literal["input", "config", "output"]


class WorkflowArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str = Field(min_length=1, max_length=1024)
    role: ArtifactRole
    required: bool = True
    min_bytes: int = Field(default=0, ge=0)
    fresh: bool = False

    # 将 artifact path 规范化为安全的 POSIX workspace 相对路径
    @field_validator("path")
    @classmethod
    def _normalize_path(cls, value: str) -> str:
        pure = PurePosixPath(value)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError(f"unsafe workflow artifact path: {value!r}")
        normalized = pure.as_posix()
        if normalized in {"", "."}:
            raise ValueError(f"unsafe workflow artifact path: {value!r}")
        return normalized

    # 校验 freshness 只约束 required output
    @model_validator(mode="after")
    def _validate_contract(self) -> WorkflowArtifact:
        if self.fresh and (self.role != "output" or not self.required):
            raise ValueError("fresh=true requires a required output artifact")
        return self


class WorkflowCheck(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    phase: Literal["preflight", "postflight"]
    argv: list[str] = Field(min_length=1)
    timeout_s: float = Field(default=60.0, gt=0)

    # 拒绝空 argv 元素，确保 harness 可以直接使用 exec 启动
    @field_validator("argv")
    @classmethod
    def _argv_must_be_nonempty(cls, value: list[str]) -> list[str]:
        if any(not item for item in value):
            raise ValueError("workflow check argv entries must not be empty")
        return value


class WorkflowContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    version: Literal[1] = 1
    artifacts: list[WorkflowArtifact] = Field(default_factory=list, max_length=128)
    checks: list[WorkflowCheck] = Field(default_factory=list, max_length=64)

    # 保证 artifact path 与 check ID 在一份 contract 内唯一
    @model_validator(mode="after")
    def _identities_must_be_unique(self) -> WorkflowContract:
        paths = [artifact.path for artifact in self.artifacts]
        if len(paths) != len(set(paths)):
            raise ValueError("workflow artifact paths must be unique")
        check_ids = [check.id for check in self.checks]
        if len(check_ids) != len(set(check_ids)):
            raise ValueError("workflow check ids must be unique")
        return self


class ArtifactMetadata(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    exists: bool
    size: int | None = None
    mtime_ns: int | None = None
    inode: int | None = None


# 将 artifact path 解析到 workspace 内，允许不逃逸的内部 symlink
def resolve_artifact_path(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve(strict=True)
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise ValueError(f"unsafe workflow artifact path: {relative_path!r}")
    candidate = root.joinpath(*pure.parts)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"workflow artifact escapes workspace: {relative_path}") from exc
    return resolved


# 读取轻量文件 metadata，不读取或哈希 artifact 内容
def snapshot_artifact(workspace_root: Path, artifact: WorkflowArtifact) -> ArtifactMetadata:
    path = resolve_artifact_path(workspace_root, artifact.path)
    try:
        stat = path.stat()
    except FileNotFoundError:
        return ArtifactMetadata(path=artifact.path, exists=False)
    return ArtifactMetadata(
        path=artifact.path,
        exists=True,
        size=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        inode=stat.st_ino,
    )


# 判断 output 是否在本次 Attempt 中新建或更新
def artifact_is_fresh(before: ArtifactMetadata, after: ArtifactMetadata) -> bool:
    if not after.exists:
        return False
    if not before.exists:
        return True
    return (before.size, before.mtime_ns, before.inode) != (
        after.size,
        after.mtime_ns,
        after.inode,
    )


# 返回 contract 的规范化 SHA-256 指纹
def workflow_contract_fingerprint(contract: WorkflowContract | None) -> str | None:
    if contract is None:
        return None
    payload = json.dumps(
        contract.model_dump(mode="json"),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# 从 workspace 读取不可为 symlink 的可选 Workflow Contract
def load_workflow_contract(workspace_root: Path) -> WorkflowContract | None:
    root = workspace_root.expanduser().resolve(strict=True)
    path = root / ".cyan" / "workflow.toml"
    if path.is_symlink():
        raise ValueError("workflow contract may not be a symlink")
    if not path.exists():
        return None
    resolved_path = path.resolve(strict=True)
    try:
        resolved_path.relative_to(root)
    except ValueError as exc:
        raise ValueError("workflow contract escapes workspace") from exc
    try:
        raw = tomllib.loads(resolved_path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"workflow contract parse error: {exc}") from exc
    contract = WorkflowContract.model_validate(raw)
    for artifact in contract.artifacts:
        resolve_artifact_path(root, artifact.path)
    return contract
