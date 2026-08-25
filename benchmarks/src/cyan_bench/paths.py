from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkPaths:
    root: Path
    cases: Path
    environments: Path
    cache: Path
    artifacts: Path


# 从 benchmark 包位置与可选环境变量解析所有本地路径
def benchmark_paths() -> BenchmarkPaths:
    root = Path(__file__).resolve().parents[2]
    cache = Path(os.environ.get("CYAN_BENCH_CACHE", root / ".cache")).expanduser().resolve()
    artifacts = Path(
        os.environ.get("CYAN_BENCH_ARTIFACTS", root / "artifacts")
    ).expanduser().resolve()
    return BenchmarkPaths(
        root=root,
        cases=root / "cases",
        environments=root / "envs",
        cache=cache,
        artifacts=artifacts,
    )
