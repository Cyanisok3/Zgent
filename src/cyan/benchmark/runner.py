from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from cyan.benchmark.corpus import Corpus
from cyan.benchmark.models import (
    BenchmarkRun,
    BenchmarkSplit,
    BenchmarkTier,
    CaseManifest,
    EvidenceBundle,
    ScoredCase,
)
from cyan.benchmark.retrieval import (
    DEFAULT_BUDGET,
    MAX_ITEM_BYTES,
    candidate_feature_rows,
    retrievers,
)
from cyan.benchmark.scoring import score_run


# 返回当前 Git revision；非 Git 环境返回 None
def code_revision(root: Path | None = None) -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


# 按 tier 和 split 过滤 Case，空过滤表示全部
def _select_cases(
    corpus: Corpus,
    tiers: set[BenchmarkTier] | None,
    splits: set[BenchmarkSplit] | None,
) -> list[CaseManifest]:
    return [
        case
        for case in corpus.cases()
        if (tiers is None or case.tier in tiers)
        and (splits is None or case.split in splits)
    ]


# 执行一个离线检索器并返回可重放 Run artifact
def _retrieve_isolated(
    corpus: Corpus,
    case: CaseManifest,
    method: str,
    *,
    byte_budget: int,
    seed: int,
) -> EvidenceBundle:
    environment = dict(os.environ)
    source_root = str(Path(__file__).resolve().parents[2])
    existing = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not existing else f"{source_root}{os.pathsep}{existing}"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "cyan.benchmark.worker",
            "--corpus",
            str(corpus.root),
            "--case-id",
            case.case_id,
            "--method",
            method,
            "--byte-budget",
            str(byte_budget),
            "--seed",
            str(seed),
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"isolated retriever failed for {case.case_id}: {result.stderr.strip()}"
        )
    return EvidenceBundle.model_validate_json(result.stdout)


# 执行一个离线检索器并返回可重放 Run artifact
def run_retrieval(
    corpus: Corpus,
    method: str,
    *,
    byte_budget: int = DEFAULT_BUDGET,
    seed: int = 0,
    tiers: set[BenchmarkTier] | None = None,
    splits: set[BenchmarkSplit] | None = None,
    isolated: bool = True,
) -> BenchmarkRun:
    registry = retrievers()
    if method not in registry:
        raise ValueError(f"unknown retriever {method!r}; choose from {sorted(registry)}")
    cases = _select_cases(corpus, tiers, splits)
    bundles = []
    for case in cases:
        if isolated:
            bundles.append(
                _retrieve_isolated(
                    corpus,
                    case,
                    method,
                    byte_budget=byte_budget,
                    seed=seed,
                )
            )
        else:
            bundles.append(
                registry[method].retrieve(
                    corpus,
                    case,
                    byte_budget=byte_budget,
                    seed=seed,
                )
            )
    return BenchmarkRun(
        created_at=datetime.now(UTC),
        code_revision=code_revision(),
        dataset_fingerprint=corpus.fingerprint(),
        method=method,
        seed=seed,
        byte_budget=byte_budget,
        max_item_bytes=MAX_ITEM_BYTES,
        bundles=bundles,
    )


# 将 BenchmarkRun 写为稳定 JSON artifact
def write_run(path: Path, run: BenchmarkRun) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(run.model_dump(mode="json"), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


# 从 JSON artifact 读取并校验 BenchmarkRun
def read_run(path: Path) -> BenchmarkRun:
    return BenchmarkRun.model_validate_json(path.read_text(encoding="utf-8"))


# 校验 fingerprint 后对 Run 评分
def score_retrieval_run(corpus: Corpus, run: BenchmarkRun) -> list[ScoredCase]:
    current = corpus.fingerprint()
    if run.dataset_fingerprint != current:
        raise ValueError(
            f"dataset fingerprint mismatch: run={run.dataset_fingerprint} current={current}"
        )
    return score_run(corpus.cases(), run)


# 写出逐 Case 评分列表
def write_scores(path: Path, scores: Iterable[ScoredCase]) -> None:
    payload = {
        "schema_version": 1,
        "scores": [score.model_dump(mode="json") for score in scores],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


# 读取一个或多个评分 artifact
def read_scores(paths: Iterable[Path]) -> list[ScoredCase]:
    scores: list[ScoredCase] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        scores.extend(ScoredCase.model_validate(item) for item in payload["scores"])
    return scores


# 导出 train/dev 候选特征，默认拒绝 test gold 泄漏
def export_features(
    corpus: Corpus,
    output: Path,
    *,
    include_test: bool = False,
) -> int:
    rows = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for case in corpus.cases():
            if case.tier != "cyan_core":
                continue
            if case.split == "test" and not include_test:
                continue
            for row in candidate_feature_rows(corpus, case):
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
                rows += 1
    return rows
