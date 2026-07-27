from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from cyan.core.incidents.models import (
    AppliedFile,
    ChangeType,
    PatchReceipt,
    Proposal,
    ProposalFile,
)

_PROTECTED_CONFIGS = frozenset({".cyan/config.toml"})
_BINARY_MARKERS = ("GIT binary patch", "Binary files ")
_UNSAFE_GIT_MODES = ("120000", "160000")


class PatchError(ValueError):
    pass


class ParsedPatchFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str
    change_type: ChangeType


class CommandResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    returncode: int
    stdout: str = ""
    stderr: str = ""


class CommandRunner(Protocol):
    # 直接执行 argv 并返回退出码和文本输出
    async def run(self, argv: Sequence[str], cwd: Path) -> CommandResult: ...


class SubprocessCommandRunner:
    # 初始化安全命令执行器并固定超时
    def __init__(self, timeout_s: float = 30.0) -> None:
        self._timeout_s = timeout_s

    # 不经 shell 执行短命令，超时后终止进程
    async def run(self, argv: Sequence[str], cwd: Path) -> CommandResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            cwd=cwd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=self._timeout_s,
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            raise PatchError(f"command timed out after {self._timeout_s}s") from None
        return CommandResult(
            returncode=process.returncode or 0,
            stdout=stdout.decode("utf-8", errors="replace"),
            stderr=stderr.decode("utf-8", errors="replace"),
        )


# 计算字节内容的 SHA-256
def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


# 计算普通文件的 SHA-256，拒绝 symlink 和非文件节点
def sha256_file(path: Path) -> str:
    if path.is_symlink():
        raise PatchError(f"symlink target is not patchable: {path}")
    if not path.is_file():
        raise PatchError(f"not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


# 规范化 unified diff 文件头路径并拒绝危险路径
def _normalize_header_path(raw: str) -> str | None:
    value = raw.split("\t", 1)[0]
    if value == "/dev/null":
        return None
    if not value or value.startswith('"') or "\\" in value:
        raise PatchError(f"unsupported patch path: {value!r}")
    if value.startswith(("a/", "b/")):
        value = value[2:]
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise PatchError(f"unsafe patch path: {value!r}")
    normalized = path.as_posix()
    if normalized in _PROTECTED_CONFIGS:
        raise PatchError(f"verifier config is protected: {normalized}")
    if path.parts[0] == ".git":
        raise PatchError("patch may not modify .git")
    return normalized


# 解析 unified diff 的文件对并限制为普通新增、修改或删除
def parse_unified_diff(patch: str) -> list[ParsedPatchFile]:
    if "\x00" in patch or any(marker in patch for marker in _BINARY_MARKERS):
        raise PatchError("binary patches are not supported")
    lines = patch.splitlines()
    if any(
        line.startswith(("rename from ", "rename to ", "copy from ", "copy to "))
        for line in lines
    ):
        raise PatchError("rename and copy patches are not supported")
    if any(
        mode in _UNSAFE_GIT_MODES
        for line in lines
        for mode in (
            line.removeprefix("new file mode "),
            line.removeprefix("deleted file mode "),
            line.removeprefix("old mode "),
            line.removeprefix("new mode "),
            line.rsplit(" ", 1)[-1] if line.startswith("index ") else "",
        )
    ):
        raise PatchError("symlink and submodule patches are not supported")

    parsed: list[ParsedPatchFile] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if not line.startswith("--- "):
            index += 1
            continue
        if index + 1 >= len(lines) or not lines[index + 1].startswith("+++ "):
            raise PatchError("each --- header must be followed by +++")
        old_path = _normalize_header_path(line[4:])
        new_path = _normalize_header_path(lines[index + 1][4:])
        if old_path is None and new_path is None:
            raise PatchError("both patch paths cannot be /dev/null")
        if old_path is None:
            assert new_path is not None
            path = new_path
            change_type: ChangeType = "create"
        elif new_path is None:
            path = old_path
            change_type = "delete"
        else:
            if old_path != new_path:
                raise PatchError("renames are not supported")
            path = old_path
            change_type = "modify"
        parsed.append(ParsedPatchFile(path=path, change_type=change_type))
        index += 2

    if not parsed:
        raise PatchError("patch contains no unified diff file headers")
    paths = [item.path for item in parsed]
    if len(paths) != len(set(paths)):
        raise PatchError("patch contains duplicate file sections")
    if not any(line.startswith("@@ ") for line in lines):
        raise PatchError("patch contains no hunks")
    return parsed


# 将相对路径约束在 workspace 内并拒绝路径链上的 symlink
def resolve_workspace_path(workspace_root: Path, relative_path: str) -> Path:
    root = workspace_root.resolve(strict=True)
    pure = PurePosixPath(relative_path)
    if pure.is_absolute() or any(part in ("", ".", "..") for part in pure.parts):
        raise PatchError(f"unsafe workspace path: {relative_path!r}")
    current = root
    for part in pure.parts:
        current = current / part
        if current.is_symlink():
            raise PatchError(f"symlink path is not patchable: {relative_path}")
    resolved = current.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise PatchError(f"path escapes workspace: {relative_path}") from exc
    return current


# 从已解析 patch 记录每个目标文件的原始哈希
def build_proposal_files(
    workspace_root: Path,
    parsed_files: list[ParsedPatchFile],
) -> list[ProposalFile]:
    files: list[ProposalFile] = []
    for item in parsed_files:
        path = resolve_workspace_path(workspace_root, item.path)
        if item.change_type == "create":
            if path.exists():
                raise PatchError(f"create target already exists: {item.path}")
            base_sha256 = None
        else:
            base_sha256 = sha256_file(path)
        files.append(
            ProposalFile(
                path=item.path,
                change_type=item.change_type,
                base_sha256=base_sha256,
            )
        )
    return files


class PatchService:
    # 用固定 Git workspace 和可注入命令 runner 初始化 patch 服务
    def __init__(
        self,
        workspace_root: Path,
        runner: CommandRunner | None = None,
    ) -> None:
        self._workspace_root = workspace_root.resolve(strict=True)
        self._runner = runner or SubprocessCommandRunner()

    # 执行 git 子命令并把非零退出转换为 PatchError
    async def _git(self, *args: str) -> CommandResult:
        result = await self._runner.run(
            ["git", "-C", str(self._workspace_root), *args],
            self._workspace_root,
        )
        if result.returncode != 0:
            detail = result.stderr.strip() or result.stdout.strip()
            raise PatchError(detail or f"git {' '.join(args)} failed")
        return result

    # 确认 workspace_root 本身就是 Git 工作树根目录
    async def _ensure_git_root(self) -> None:
        result = await self._git("rev-parse", "--show-toplevel")
        reported = Path(result.stdout.strip()).resolve(strict=True)
        if reported != self._workspace_root:
            raise PatchError("workspace_root must be the Git worktree root")

    # 拒绝直接修改 Git submodule 或在其路径下创建文件
    async def _ensure_no_submodule_targets(self, proposal: Proposal) -> None:
        result = await self._git("ls-files", "--stage")
        gitlinks = {
            line.split("\t", 1)[1]
            for line in result.stdout.splitlines()
            if line.startswith("160000 ") and "\t" in line
        }
        for item in proposal.files:
            if any(item.path == link or item.path.startswith(f"{link}/") for link in gitlinks):
                raise PatchError(f"submodule target is not patchable: {item.path}")

    # 校验 proposal 元数据、patch 内容和原始文件哈希仍完全一致
    def _verify_base_state(self, proposal: Proposal, patch_path: Path) -> None:
        patch_bytes = patch_path.read_bytes()
        if sha256_bytes(patch_bytes) != proposal.patch_sha256:
            raise PatchError("proposal patch hash changed")
        parsed = parse_unified_diff(patch_bytes.decode("utf-8"))
        expected = [(item.path, item.change_type) for item in proposal.files]
        actual = [(item.path, item.change_type) for item in parsed]
        if actual != expected:
            raise PatchError("proposal metadata does not match patch")
        for item in proposal.files:
            path = resolve_workspace_path(self._workspace_root, item.path)
            if item.change_type == "create":
                if path.exists():
                    raise PatchError(f"stale create target: {item.path}")
            elif not path.exists() or sha256_file(path) != item.base_sha256:
                raise PatchError(f"stale target: {item.path}")

    # 校验当前文件与 apply receipt 一致，防止覆盖 smoke 期间的用户修改
    def _verify_applied_state(self, receipt: PatchReceipt) -> None:
        for item in receipt.files:
            path = resolve_workspace_path(self._workspace_root, item.path)
            if item.applied_sha256 is None:
                if path.exists():
                    raise PatchError(f"applied state changed: {item.path}")
            elif not path.exists() or sha256_file(path) != item.applied_sha256:
                raise PatchError(f"applied state changed: {item.path}")

    # 记录 apply 完成后的文件哈希作为安全回滚凭据
    def _build_receipt(self, proposal: Proposal) -> PatchReceipt:
        files: list[AppliedFile] = []
        for item in proposal.files:
            path = resolve_workspace_path(self._workspace_root, item.path)
            applied_sha256 = sha256_file(path) if path.exists() else None
            files.append(AppliedFile(path=item.path, applied_sha256=applied_sha256))
        return PatchReceipt(
            proposal_id=proposal.id,
            files=files,
            applied_at=datetime.now(UTC),
        )

    # 复验哈希和 git apply --check 后原子应用 proposal
    async def apply(self, proposal: Proposal, patch_path: Path) -> PatchReceipt:
        await self._ensure_git_root()
        await self._ensure_no_submodule_targets(proposal)
        self._verify_base_state(proposal, patch_path)
        await self._git("apply", "--check", str(patch_path))
        self._verify_base_state(proposal, patch_path)
        await self._git("apply", str(patch_path))
        return self._build_receipt(proposal)

    # 仅在文件仍处于 apply 后状态时反向应用并恢复原始哈希
    async def reverse(
        self,
        proposal: Proposal,
        patch_path: Path,
        receipt: PatchReceipt,
    ) -> None:
        if receipt.proposal_id != proposal.id:
            raise PatchError("receipt belongs to another proposal")
        await self._ensure_git_root()
        await self._ensure_no_submodule_targets(proposal)
        self._verify_applied_state(receipt)
        await self._git("apply", "--reverse", "--check", str(patch_path))
        self._verify_applied_state(receipt)
        await self._git("apply", "--reverse", str(patch_path))
        self._verify_base_state(proposal, patch_path)
