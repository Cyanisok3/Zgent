from __future__ import annotations

import hashlib
import heapq
import math
import random
import re
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from cyan.benchmark.corpus import Corpus
from cyan.benchmark.models import CaseManifest, EvidenceBundle, EvidenceItem, EvidenceStream
from cyan.core.incidents.models import LogSnapshot

DEFAULT_BUDGET = 256 * 1024
MAX_ITEM_BYTES = 32 * 1024
CHUNK_BYTES = 8 * 1024

_TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_.:/-]{2,}")
_SEVERITY = re.compile(
    rb"(?i)(traceback|exception|fatal|runtimeerror|valueerror|assertionerror|"
    rb"out of memory|permission denied|failed|root[-_ ]cause|cyan_(?:evidence|stress_evidence))"
)


@dataclass(frozen=True)
class LogChunk:
    stream: EvidenceStream
    start: int
    end: int
    raw: bytes
    total_size: int


@dataclass(frozen=True)
class RankedChunk:
    chunk: LogChunk
    score: float
    reason: str


class Retriever(Protocol):
    name: str

    # 在固定预算内为一个 Case 返回证据包
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle: ...


# 将文本规范化为有限词法 token
def tokenize(text: str) -> list[str]:
    return [token.lower() for token in _TOKEN.findall(text)]


# 以稳定非重叠 byte 块流式读取单个日志
def iter_chunks(path: Path, stream: EvidenceStream) -> Iterator[LogChunk]:
    total = path.stat().st_size
    offset = 0
    with path.open("rb") as handle:
        while raw := handle.read(CHUNK_BYTES):
            yield LogChunk(
                stream=stream,
                start=offset,
                end=offset + len(raw),
                raw=raw,
                total_size=total,
            )
            offset += len(raw)


# 依次遍历一个 Case 的 stderr 和 stdout 块
def iter_case_chunks(corpus: Corpus, case: CaseManifest) -> Iterator[LogChunk]:
    for stream in ("stderr", "stdout"):
        yield from iter_chunks(corpus.log_path(case, stream), stream)


# 把 Capsule 已携带的免费日志区间转为初始证据
def capsule_items(case: CaseManifest) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    snapshots: tuple[tuple[EvidenceStream, LogSnapshot], ...] = (
        ("stderr", case.capsule.stderr),
        ("stdout", case.capsule.stdout),
    )
    for stream, snapshot in snapshots:
        if snapshot.included_end <= snapshot.included_start:
            continue
        items.append(
            EvidenceItem(
                stream=stream,
                byte_start=snapshot.included_start,
                byte_end=snapshot.included_end,
                score=0.0,
                reason="failure capsule tail",
                cost_bytes=0,
            )
        )
    return items


# 从 Failure Capsule 提取不含金标的检索查询
def query_terms(case: CaseManifest) -> list[str]:
    text = " ".join(
        [
            case.failure_kind,
            case.phase or "",
            case.capsule.check_id or "",
            case.capsule.artifact_path or "",
            case.capsule.violation_rule or "",
            case.capsule.stderr.tail,
            case.capsule.stdout.tail,
        ]
    )
    counts = Counter(tokenize(text))
    ignored = {
        "python",
        "file",
        "line",
        "main",
        "workflow",
        "attempt",
        "process_exit",
        "contract_violation",
    }
    return [token for token, _count in counts.most_common(64) if token not in ignored]


# 按 score 排序并在预算内去重选择证据块
def _select(
    ranked: Iterable[RankedChunk],
    *,
    byte_budget: int,
) -> list[EvidenceItem]:
    selected: list[EvidenceItem] = []
    used = 0
    seen: set[tuple[str, int, int]] = set()
    for candidate in sorted(ranked, key=lambda item: (-item.score, item.chunk.start)):
        chunk = candidate.chunk
        identity = (chunk.stream, chunk.start, chunk.end)
        if identity in seen:
            continue
        cost = chunk.end - chunk.start
        if cost > MAX_ITEM_BYTES or used + cost > byte_budget:
            continue
        selected.append(
            EvidenceItem(
                stream=chunk.stream,
                byte_start=chunk.start,
                byte_end=chunk.end,
                score=candidate.score,
                reason=candidate.reason,
                cost_bytes=cost,
            )
        )
        seen.add(identity)
        used += cost
    return selected


# 构造已校验 EvidenceBundle 并记录耗时与扫描量
def _bundle(
    case: CaseManifest,
    method: str,
    items: list[EvidenceItem],
    budget: int,
    scanned: int,
    started: float,
    *,
    abstained: bool = False,
    metadata: dict[str, str | int | float | bool] | None = None,
) -> EvidenceBundle:
    return EvidenceBundle(
        case_id=case.case_id,
        method=method,
        initial_items=capsule_items(case),
        items=items,
        byte_budget=budget,
        returned_bytes=sum(item.cost_bytes for item in items),
        scanned_bytes=scanned,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        abstained=abstained,
        metadata=metadata or {},
    )


class RandomRetriever:
    name = "random"

    # 使用 Case 稳定种子随机排列全部日志块
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        started = time.perf_counter()
        chunks = list(iter_case_chunks(corpus, case))
        scanned = sum(len(chunk.raw) for chunk in chunks)
        digest = hashlib.sha256(f"{seed}:{case.case_id}".encode()).digest()
        generator = random.Random(int.from_bytes(digest[:8], "big"))
        generator.shuffle(chunks)
        ranked = [
            RankedChunk(chunk=chunk, score=float(len(chunks) - index), reason="random baseline")
            for index, chunk in enumerate(chunks)
        ]
        items = _select(ranked, byte_budget=byte_budget)
        return _bundle(case, self.name, items, byte_budget, scanned, started)


class CapsuleTailRetriever:
    name = "capsule_tail"

    # 只使用免费 Capsule 尾部，不主动读取额外日志
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        del corpus, seed
        started = time.perf_counter()
        return _bundle(
            case,
            self.name,
            [],
            byte_budget,
            0,
            started,
            abstained=True,
        )


class LiteralSearchRetriever:
    name = "literal_search_policy"

    # 按 Capsule token 搜索首次匹配块并补充严重错误块
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        del seed
        started = time.perf_counter()
        terms = [term.encode("utf-8") for term in query_terms(case)[:24]]
        ranked: list[RankedChunk] = []
        scanned = 0
        for chunk in iter_case_chunks(corpus, case):
            scanned += len(chunk.raw)
            lowered = chunk.raw.lower()
            hits = sum(term in lowered for term in terms)
            severity = len(_SEVERITY.findall(chunk.raw))
            if hits or severity:
                ranked.append(
                    RankedChunk(
                        chunk=chunk,
                        score=float(hits * 2 + severity),
                        reason=f"literal_hits={hits};severity={severity}",
                    )
                )
        items = _select(ranked, byte_budget=byte_budget)
        return _bundle(
            case,
            self.name,
            items,
            byte_budget,
            scanned,
            started,
            abstained=not items,
            metadata={"query_terms": len(terms)},
        )


# 扫描 query term 文档频率和平均块长度
def _bm25_statistics(
    corpus: Corpus,
    case: CaseManifest,
    terms: set[str],
) -> tuple[int, float, Counter[str], int]:
    documents = 0
    total_length = 0
    document_frequency: Counter[str] = Counter()
    scanned = 0
    for chunk in iter_case_chunks(corpus, case):
        scanned += len(chunk.raw)
        tokens = tokenize(chunk.raw.decode("utf-8", errors="replace"))
        documents += 1
        total_length += len(tokens)
        present = set(tokens) & terms
        document_frequency.update(present)
    return documents, total_length / max(1, documents), document_frequency, scanned


# 计算一个日志块的 BM25 分数
def _bm25_score(
    tokens: list[str],
    terms: set[str],
    documents: int,
    avg_length: float,
    document_frequency: Counter[str],
) -> float:
    frequencies = Counter(tokens)
    score = 0.0
    k1 = 1.5
    b = 0.75
    for term in terms:
        frequency = frequencies[term]
        if not frequency:
            continue
        df = document_frequency[term]
        inverse = math.log(1.0 + (documents - df + 0.5) / (df + 0.5))
        denominator = frequency + k1 * (1.0 - b + b * len(tokens) / max(1.0, avg_length))
        score += inverse * frequency * (k1 + 1.0) / denominator
    return score


# 为 BM25 和 Hybrid 生成有界 top-k 候选并返回扫描字节数
def _rank_lexical(
    corpus: Corpus,
    case: CaseManifest,
    *,
    hybrid: bool,
    byte_budget: int,
) -> tuple[list[RankedChunk], int]:
    terms = set(query_terms(case))
    documents, avg_length, df, scanned = _bm25_statistics(corpus, case, terms)
    capacity = max(32, byte_budget // CHUNK_BYTES * 4)
    heap: list[tuple[float, int, RankedChunk]] = []
    serial = 0
    for chunk in iter_case_chunks(corpus, case):
        scanned += len(chunk.raw)
        text = chunk.raw.decode("utf-8", errors="replace")
        tokens = tokenize(text)
        lexical = _bm25_score(tokens, terms, documents, avg_length, df)
        severity = len(_SEVERITY.findall(chunk.raw))
        phase_hits = int(bool(case.phase and case.phase in text.lower()))
        stream_weight = 0.4 if chunk.stream == "stderr" else 0.0
        position = chunk.end / max(1, chunk.total_size)
        score = lexical
        reason = f"bm25={lexical:.6f}"
        if hybrid:
            score += severity * 3.0 + phase_hits * 0.8 + stream_weight + position * 0.15
            reason += (
                f";severity={severity};phase={phase_hits};stderr={stream_weight:.1f};"
                f"position={position:.3f}"
            )
        if score <= 0:
            continue
        ranked = RankedChunk(chunk=chunk, score=score, reason=reason)
        item = (score, serial, ranked)
        serial += 1
        if len(heap) < capacity:
            heapq.heappush(heap, item)
        elif score > heap[0][0]:
            heapq.heapreplace(heap, item)
    return [item[2] for item in heap], scanned


class BM25Retriever:
    name = "bm25"

    # 用 Capsule 查询在全部固定块上执行 BM25 排序
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        del seed
        started = time.perf_counter()
        ranked, scanned = _rank_lexical(corpus, case, hybrid=False, byte_budget=byte_budget)
        items = _select(ranked, byte_budget=byte_budget)
        return _bundle(
            case,
            self.name,
            items,
            byte_budget,
            scanned,
            started,
            abstained=not items,
        )


class HeuristicHybridRetriever:
    name = "heuristic_hybrid"

    # 融合 BM25、错误模式、stream、phase 和位置特征
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        del seed
        started = time.perf_counter()
        ranked, scanned = _rank_lexical(corpus, case, hybrid=True, byte_budget=byte_budget)
        items = _select(ranked, byte_budget=byte_budget)
        return _bundle(
            case,
            self.name,
            items,
            byte_budget,
            scanned,
            started,
            abstained=not items,
        )


class OracleRetriever:
    name = "oracle"

    # 直接返回 gold 区间，仅用于验证评分器理论上限
    def retrieve(
        self,
        corpus: Corpus,
        case: CaseManifest,
        *,
        byte_budget: int,
        seed: int,
    ) -> EvidenceBundle:
        del corpus, seed
        started = time.perf_counter()
        ranked: list[RankedChunk] = []
        for fact in case.gold_facts:
            raw = b"x" * (fact.byte_end - fact.byte_start)
            ranked.append(
                RankedChunk(
                    chunk=LogChunk(
                        stream=fact.stream,
                        start=fact.byte_start,
                        end=fact.byte_end,
                        raw=raw,
                        total_size=case.logs[fact.stream].size,
                    ),
                    score=2.0 if fact.importance == "essential" else 1.0,
                    reason="oracle gold fact",
                )
            )
        items = _select(ranked, byte_budget=byte_budget)
        return _bundle(
            case,
            self.name,
            items,
            byte_budget,
            0,
            started,
            abstained=not case.gold_facts,
        )


# 返回首版检索器注册表
def retrievers() -> dict[str, Retriever]:
    values: list[Retriever] = [
        RandomRetriever(),
        CapsuleTailRetriever(),
        LiteralSearchRetriever(),
        BM25Retriever(),
        HeuristicHybridRetriever(),
        OracleRetriever(),
    ]
    return {value.name: value for value in values}


# 导出后续 GP 可直接消费的逐候选确定性特征
def candidate_feature_rows(corpus: Corpus, case: CaseManifest) -> Iterator[dict[str, object]]:
    terms = set(query_terms(case))
    for chunk in iter_case_chunks(corpus, case):
        text = chunk.raw.decode("utf-8", errors="replace")
        tokens = tokenize(text)
        covered_facts = [
            fact.fact_id
            for fact in case.gold_facts
            if fact.stream == chunk.stream
            and chunk.start <= fact.byte_start
            and chunk.end >= fact.byte_end
        ]
        yield {
            "case_id": case.case_id,
            "stream": chunk.stream,
            "byte_start": chunk.start,
            "byte_end": chunk.end,
            "token_count": len(tokens),
            "query_term_hits": sum(token in terms for token in tokens),
            "unique_query_term_hits": len(set(tokens) & terms),
            "severity_hits": len(_SEVERITY.findall(chunk.raw)),
            "is_stderr": chunk.stream == "stderr",
            "relative_position": chunk.end / max(1, chunk.total_size),
            "gold_fact_ids": covered_facts,
            "relevance": max(
                [
                    2 if fact.importance == "essential" else 1
                    for fact in case.gold_facts
                    if fact.fact_id in covered_facts
                ],
                default=0,
            ),
        }
