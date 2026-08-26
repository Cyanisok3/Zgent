from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

from cyan_bench.models import CaseManifest, ExpectedOutcome, ResolvedAnchor


@dataclass(frozen=True)
class LoadedCase:
    root: Path
    manifest: CaseManifest
    expected: ExpectedOutcome


# 计算所有案例定义文件的稳定内容指纹
def case_fingerprint(case: LoadedCase) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in case.root.rglob("*") if item.is_file()):
        relative = path.relative_to(case.root).as_posix().encode()
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return digest.hexdigest()


# 读取一个案例的 TOML、oracle 与必要 patch 文件
def load_case(case_dir: Path) -> LoadedCase:
    manifest = CaseManifest.model_validate(
        tomllib.loads((case_dir / "case.toml").read_text(encoding="utf-8"))
    )
    expected = ExpectedOutcome.model_validate_json(
        (case_dir / "expected.json").read_text(encoding="utf-8")
    )
    for name in ("fault.patch", "fix.patch"):
        path = case_dir / name
        if not path.is_file() or not path.read_bytes():
            raise ValueError(f"{manifest.id}: missing non-empty {name}")
    return LoadedCase(case_dir.resolve(), manifest, expected)


# 按稳定 ID 顺序读取指定 split 与数据集版本的案例
def discover_cases(
    cases_root: Path,
    split: str | None = None,
    dataset_version: str | None = None,
) -> list[LoadedCase]:
    loaded = [
        load_case(path)
        for path in sorted(cases_root.iterdir())
        if path.is_dir() and not (path / "REJECTED.md").exists()
    ]
    if split is not None:
        loaded = [case for case in loaded if case.manifest.split == split]
    if dataset_version is not None:
        loaded = [
            case for case in loaded if case.manifest.dataset_version == dataset_version
        ]
    ids = [case.manifest.id for case in loaded]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate case id")
    return loaded


# 将一次运行的动态日志锚点解析为实际字节范围
def resolve_anchors(
    expected: ExpectedOutcome,
    stdout_path: Path,
    stderr_path: Path,
) -> list[ResolvedAnchor]:
    payloads = {
        "stdout": stdout_path.read_bytes(),
        "stderr": stderr_path.read_bytes(),
    }
    resolved: list[ResolvedAnchor] = []
    for group_index, group in enumerate(expected.required_groups):
        group_matches: list[ResolvedAnchor] = []
        for anchor in group:
            data = payloads[anchor.source]
            if anchor.literal is not None:
                needle = anchor.literal.encode("utf-8")
                starts = [match.start() for match in re.finditer(re.escape(needle), data)]
                matcher = f"literal:{anchor.literal}"
                length = len(needle)
            else:
                assert anchor.regex is not None
                pattern = re.compile(anchor.regex.encode("utf-8"))
                matches = list(pattern.finditer(data))
                starts = [match.start() for match in matches]
                matcher = f"regex:{anchor.regex}"
                length = matches[-1].end() - matches[-1].start() if matches else 0
            if starts:
                start = starts[-1]
                group_matches.append(
                    ResolvedAnchor(
                        group=group_index,
                        source=anchor.source,
                        start=start,
                        end=start + length,
                        matcher=matcher,
                    )
                )
        if not group_matches:
            raise ValueError(f"required evidence group {group_index} did not match")
        resolved.extend(group_matches)
    return resolved


# 将解析后的动态证据范围写入本轮 artifact
def write_resolved_anchors(path: Path, anchors: list[ResolvedAnchor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([item.model_dump(mode="json") for item in anchors], indent=2) + "\n",
        encoding="utf-8",
    )
