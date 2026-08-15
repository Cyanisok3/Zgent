from __future__ import annotations

import argparse
import json
import resource
import sys
from pathlib import Path

from cyan.benchmark.corpus import Corpus
from cyan.benchmark.retrieval import retrievers


# 将平台相关的 ru_maxrss 统一为字节
def _rss_bytes(value: int) -> int:
    return value if sys.platform == "darwin" else value * 1024


# 在独立进程中执行一个 Case 的检索并输出 JSON
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--byte-budget", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    corpus = Corpus(args.corpus)
    cases = {case.case_id: case for case in corpus.cases()}
    if args.case_id not in cases:
        raise SystemExit(f"unknown case: {args.case_id}")
    registry = retrievers()
    if args.method not in registry:
        raise SystemExit(f"unknown retriever: {args.method}")
    baseline = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    bundle = registry[args.method].retrieve(
        corpus,
        cases[args.case_id],
        byte_budget=args.byte_budget,
        seed=args.seed,
    )
    peak = _rss_bytes(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    measured = bundle.model_copy(
        update={
            "peak_rss_bytes": peak,
            "rss_delta_bytes": max(0, peak - baseline),
        }
    )
    print(json.dumps(measured.model_dump(mode="json"), sort_keys=True))


if __name__ == "__main__":
    main()
