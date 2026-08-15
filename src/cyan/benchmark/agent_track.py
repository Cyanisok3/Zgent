from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Literal

import anthropic
from pydantic import BaseModel, ConfigDict, Field

from cyan.benchmark.corpus import Corpus
from cyan.benchmark.models import (
    AgentStrategyResult,
    CaseManifest,
    EvidenceBundle,
    EvidenceItem,
    EvidenceStream,
    StrategyName,
)
from cyan.benchmark.retrieval import DEFAULT_BUDGET, MAX_ITEM_BYTES, capsule_items
from cyan.benchmark.scoring import score_bundle
from cyan.core.config import CyanConfig
from cyan.core.events.bus import EventBus
from cyan.core.llm.base import LLMProvider
from cyan.core.llm.provider import AnthropicProvider
from cyan.core.llm.types import LlmResponse
from cyan.core.runner import AgentRunner, RunOutcome
from cyan.core.tools.base import BaseTool, ToolResult


class ReadBenchmarkLogParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stream: EvidenceStream
    mode: Literal["tail", "range", "search"] = "tail"
    offset: int = Field(default=0, ge=0)
    limit: int = Field(default=MAX_ITEM_BYTES, ge=1, le=MAX_ITEM_BYTES)
    query: str | None = Field(default=None, min_length=1, max_length=1024)


class SubmitBenchmarkDiagnosisParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root_cause: str = Field(min_length=1, max_length=8000)
    recovery_kind: Literal["patch", "operator_action", "none"]
    evidence_references: list[str] = Field(min_length=1, max_length=32)


_REFERENCE = re.compile(r"^(stdout|stderr)@bytes:(\d+)-(\d+)$")


class _TemperatureMessages:
    # 包装 Anthropic messages 端点并固定 Benchmark temperature
    def __init__(self, delegate: Any, temperature: float) -> None:
        self._delegate = delegate
        self._temperature = temperature

    # 给每次流式请求注入同一个 temperature
    def stream(self, **kwargs: object) -> Any:
        kwargs["temperature"] = self._temperature
        return self._delegate.stream(**kwargs)


class _TemperatureClient:
    # 暴露 AnthropicProvider 所需的 messages 接口
    def __init__(self, client: Any, temperature: float) -> None:
        self.messages = _TemperatureMessages(client.messages, temperature)


# 创建只用于 Benchmark 的固定 temperature Anthropic Provider
def create_benchmark_provider(model: str, temperature: float) -> AnthropicProvider:
    if not 0.0 <= temperature <= 1.0:
        raise ValueError("benchmark temperature must be between 0 and 1")
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise SystemExit("ANTHROPIC_API_KEY not set")
    client = anthropic.AsyncAnthropic(api_key=api_key)
    return AnthropicProvider(model, client=_TemperatureClient(client, temperature))


# 分块搜索日志并返回首次匹配位置与实际扫描字节数
def _search_log(path: Path, offset: int, query: bytes) -> tuple[int | None, int]:
    overlap = b""
    position = min(offset, path.stat().st_size)
    scanned = 0
    with path.open("rb") as handle:
        handle.seek(position)
        while raw := handle.read(64 * 1024):
            scanned += len(raw)
            window = overlap + raw
            found = window.find(query)
            if found >= 0:
                return position - len(overlap) + found, scanned
            overlap_size = max(0, len(query) - 1)
            overlap = window[-overlap_size:] if overlap_size else b""
            position += len(raw)
    return None, scanned


class TokenBudgetProvider:
    # 包装 Provider 并跨单 Agent 或多个只读角色累计 token
    def __init__(self, provider: LLMProvider, token_budget: int) -> None:
        self._provider = provider
        self.token_budget = token_budget
        self.input_tokens = 0
        self.output_tokens = 0

    # 在共享 token 预算耗尽后拒绝新的 LLM 调用
    async def chat(
        self,
        messages: list[dict[str, object]],
        tool_schemas: list[dict[str, object]],
        bus: EventBus,
        run_id: str,
        *,
        step: int = 0,
        system: str | None = None,
    ) -> LlmResponse:
        if self.input_tokens + self.output_tokens >= self.token_budget:
            raise RuntimeError("benchmark token budget exhausted")
        response = await self._provider.chat(
            messages,
            tool_schemas,
            bus,
            run_id,
            step=step,
            system=system,
        )
        if response.usage is not None:
            self.input_tokens += response.usage.input_tokens
            self.output_tokens += response.usage.output_tokens
        if self.input_tokens + self.output_tokens > self.token_budget:
            raise RuntimeError("benchmark token budget exceeded by final response")
        return response


class BenchmarkEvidenceReader:
    # 绑定一个 Case 的封存日志并实施共享返回字节预算
    def __init__(self, corpus: Corpus, case: CaseManifest, byte_budget: int) -> None:
        self._corpus = corpus
        self._case = case
        self._byte_budget = byte_budget
        self.items: list[EvidenceItem] = []
        self.scanned_bytes = 0
        self.read_calls = 0
        self.observed = {
            f"{item.stream}@bytes:{item.byte_start}-{item.byte_end}"
            for item in capsule_items(case)
        }

    # 返回尚可供 Agent 读取的证据字节数
    def remaining(self) -> int:
        return self._byte_budget - sum(item.cost_bytes for item in self.items)

    # 按 tail/range/search 读取一个有界日志区间并登记引用
    def read(self, params: ReadBenchmarkLogParams) -> dict[str, object]:
        self.read_calls += 1
        path = self._corpus.log_path(self._case, params.stream)
        size = path.stat().st_size
        limit = min(params.limit, self.remaining(), MAX_ITEM_BYTES)
        if limit <= 0:
            raise ValueError("evidence byte budget exhausted")
        match_offset: int | None = None
        if params.mode == "tail":
            start = max(0, size - limit)
        elif params.mode == "range":
            start = min(params.offset, size)
        else:
            if params.query is None:
                raise ValueError("query is required for search mode")
            query = params.query.encode("utf-8")
            match_offset, scanned = _search_log(path, params.offset, query)
            self.scanned_bytes += scanned
            if match_offset is None:
                return {
                    "stream": params.stream,
                    "size": size,
                    "match_offset": None,
                    "slice": None,
                }
            start = match_offset
        with path.open("rb") as handle:
            handle.seek(start)
            raw = handle.read(limit)
        self.scanned_bytes += len(raw)
        end = start + len(raw)
        reference = f"{params.stream}@bytes:{start}-{end}"
        item = EvidenceItem(
            stream=params.stream,
            byte_start=start,
            byte_end=end,
            score=0.0,
            reason=f"agent {params.mode}",
            cost_bytes=len(raw),
        )
        self.items.append(item)
        self.observed.add(reference)
        return {
            "stream": params.stream,
            "size": size,
            "match_offset": match_offset,
            "slice": {
                "start": start,
                "end": end,
                "content": raw.decode("utf-8", errors="replace"),
            },
            "reference": reference,
            "remaining_budget": self.remaining(),
        }


class ReadBenchmarkLogTool(BaseTool):
    name = "read_benchmark_log"
    description = (
        "Read one immutable benchmark stdout/stderr slice by tail, byte range, or literal search."
    )
    params_model = ReadBenchmarkLogParams
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "stream": {"type": "string", "enum": ["stdout", "stderr"]},
            "mode": {"type": "string", "enum": ["tail", "range", "search"]},
            "offset": {"type": "integer", "minimum": 0},
            "limit": {"type": "integer", "minimum": 1, "maximum": MAX_ITEM_BYTES},
            "query": {"type": "string"},
        },
        "required": ["stream"],
    }

    # 注入共享证据 reader
    def __init__(self, reader: BenchmarkEvidenceReader) -> None:
        self._reader = reader

    # 执行一次只读日志检索并返回稳定引用
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        try:
            result = self._reader.read(ReadBenchmarkLogParams.model_validate(params))
            return ToolResult(content=json.dumps(result, ensure_ascii=False))
        except ValueError as exc:
            return ToolResult(content=str(exc), is_error=True, error_type="schema_error")


class SubmitBenchmarkDiagnosisTool(BaseTool):
    name = "submit_benchmark_diagnosis"
    description = "Submit one root-cause diagnosis with observed evidence references."
    params_model = SubmitBenchmarkDiagnosisParams
    input_schema: dict[str, object] = {
        "type": "object",
        "properties": {
            "root_cause": {"type": "string"},
            "recovery_kind": {
                "type": "string",
                "enum": ["patch", "operator_action", "none"],
            },
            "evidence_references": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 32,
            },
        },
        "required": ["root_cause", "recovery_kind", "evidence_references"],
    }

    # 注入允许引用集合并初始化空提交槽
    def __init__(self, observed: set[str]) -> None:
        self._observed = observed
        self.submission: SubmitBenchmarkDiagnosisParams | None = None
        self.call_count = 0

    # 仅接受本轮真实观察到的日志引用
    async def invoke(self, params: dict[str, object]) -> ToolResult:
        self.call_count += 1
        parsed = SubmitBenchmarkDiagnosisParams.model_validate(params)
        unknown = [ref for ref in parsed.evidence_references if ref not in self._observed]
        if unknown:
            return ToolResult(
                content=f"unobserved evidence references: {unknown}",
                is_error=True,
                error_type="schema_error",
            )
        self.submission = parsed
        return ToolResult(content="benchmark diagnosis accepted")


# 返回当前 Agent 和 Skill 两种固定策略提示
def strategy_prompt(case: CaseManifest, strategy: StrategyName) -> str:
    base = (
        "You are a read-only evidence retrieval agent. Inspect only the sealed benchmark logs. "
        "Use read_benchmark_log sparingly, then call submit_benchmark_diagnosis exactly once. "
        "Do not propose code, run commands, or invent references.\n"
        f"Failure capsule:\n{case.capsule.model_dump_json(indent=2)}"
    )
    if strategy == "retrieval_skill":
        return (
            base
            + "\nRetrieval skill procedure:\n"
            "1. Extract exception, file, check, artifact and phase terms from the capsule.\n"
            "2. Search for the first causal occurrence, not only the final secondary error.\n"
            "3. Compare the matching region with nearby traceback or phase boundaries.\n"
            "4. Stop once essential evidence is sufficient and avoid duplicate slices."
        )
    return base


# 用同一个只读工具集运行一个 Agent 角色
async def _run_role(
    case: CaseManifest,
    strategy: StrategyName,
    provider: TokenBudgetProvider,
    reader: BenchmarkEvidenceReader,
    runs_dir: Path,
    *,
    max_steps: int,
    prompt_suffix: str = "",
    allow_log_read: bool = True,
) -> tuple[RunOutcome, SubmitBenchmarkDiagnosisParams | None, int]:
    config = CyanConfig()
    config.agent.max_steps = max_steps
    submit = SubmitBenchmarkDiagnosisTool(reader.observed)
    tools: list[BaseTool] = [submit]
    whitelist = [submit.name]
    if allow_log_read:
        log_tool = ReadBenchmarkLogTool(reader)
        tools.insert(0, log_tool)
        whitelist.insert(0, log_tool.name)
    runner = AgentRunner(
        config,
        provider=provider,
        runs_dir=runs_dir,
    )
    reads_before = reader.read_calls
    prompt = strategy_prompt(case, strategy) + prompt_suffix
    outcome = await runner.run_and_capture(
        "Retrieve evidence and submit the benchmark diagnosis.",
        system_prompt_override=prompt,
        tool_whitelist=whitelist,
        workspace_root=runs_dir,
        extra_tools=tools,
        max_steps=max_steps,
        compact_threshold=0.0,
        summary_only_events=True,
        include_context=False,
    )
    return outcome, submit.submission, reader.read_calls - reads_before + submit.call_count


# 计算诊断文本对 Case 必须词的覆盖率
def _diagnosis_term_recall(case: CaseManifest, root_cause: str | None) -> float:
    terms = [term.lower() for term in case.expected_diagnosis_terms]
    if not terms:
        return 1.0
    lowered = (root_cause or "").lower()
    return sum(term in lowered for term in terms) / len(terms)


# 把 Agent 最终引用解析为零成本 EvidenceItem 供引用正确率评分
def _cited_items(references: list[str]) -> list[EvidenceItem]:
    items: list[EvidenceItem] = []
    for reference in references:
        match = _REFERENCE.fullmatch(reference)
        if match is None:
            continue
        start, end = int(match.group(2)), int(match.group(3))
        items.append(
            EvidenceItem(
                stream=match.group(1),  # type: ignore[arg-type]
                byte_start=start,
                byte_end=end,
                score=0.0,
                reason="agent diagnosis citation",
                cost_bytes=0,
            )
        )
    return items


# 运行 current、Skill 或基准专用只读 Subagent 策略
async def run_agent_strategy(
    corpus: Corpus,
    case: CaseManifest,
    strategy: StrategyName,
    provider: LLMProvider,
    *,
    model: str,
    runs_dir: Path,
    byte_budget: int = DEFAULT_BUDGET,
    token_budget: int = 80_000,
) -> AgentStrategyResult:
    if case.split not in {"test", "external"}:
        raise ValueError("agent strategy track accepts only cyan_core test or external cases")
    started = time.perf_counter()
    budgeted_provider = TokenBudgetProvider(provider, token_budget)
    reader = BenchmarkEvidenceReader(corpus, case, byte_budget)
    outcome: RunOutcome
    submission: SubmitBenchmarkDiagnosisParams | None
    if strategy != "readonly_subagents":
        outcome, submission, tool_calls = await _run_role(
            case,
            strategy,
            budgeted_provider,
            reader,
            runs_dir,
            max_steps=12,
        )
    else:
        findings: list[str] = []
        roles = (
            "Focus only on traceback and exception causality.",
            "Focus only on temporal precursors and first abnormal occurrence.",
            "Focus only on workflow phase, check, artifact, and recovery boundary.",
        )
        tool_calls = 0
        for role in roles:
            role_outcome, role_submission, role_calls = await _run_role(
                case,
                "current_agent",
                budgeted_provider,
                reader,
                runs_dir,
                max_steps=3,
                prompt_suffix=f"\nRole assignment: {role}",
            )
            tool_calls += role_calls
            if role_submission is not None:
                findings.append(role_submission.model_dump_json())
            elif role_outcome.result:
                findings.append(role_outcome.result)
        outcome, submission, aggregate_calls = await _run_role(
            case,
            "current_agent",
            budgeted_provider,
            reader,
            runs_dir,
            max_steps=3,
            prompt_suffix="\nRead-only specialist findings:\n" + "\n".join(findings),
            allow_log_read=False,
        )
        tool_calls += aggregate_calls
    bundle = EvidenceBundle(
        case_id=case.case_id,
        method=strategy,
        initial_items=capsule_items(case),
        items=reader.items,
        byte_budget=byte_budget,
        returned_bytes=sum(item.cost_bytes for item in reader.items),
        scanned_bytes=reader.scanned_bytes,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        abstained=not reader.items,
        metadata={
            "token_budget": token_budget,
            "token_budget_exceeded": (
                budgeted_provider.input_tokens + budgeted_provider.output_tokens > token_budget
            ),
        },
    )
    retrieved_score = score_bundle(case, bundle)
    root_cause = submission.root_cause if submission is not None else None
    recovery_kind = submission.recovery_kind if submission is not None else None
    references = submission.evidence_references if submission is not None else []
    cited_items = _cited_items(references)
    citation_bundle = EvidenceBundle(
        case_id=case.case_id,
        method=f"{strategy}-citations",
        items=cited_items,
        byte_budget=byte_budget,
        returned_bytes=0,
        scanned_bytes=0,
        elapsed_ms=0.0,
        abstained=not cited_items,
    )
    citation_score = score_bundle(case, citation_bundle)
    return AgentStrategyResult(
        case_id=case.case_id,
        tier=case.tier,
        split=case.split,
        strategy=strategy,
        model=model,
        diagnosis_submitted=submission is not None,
        root_cause=root_cause,
        recovery_kind=recovery_kind,
        evidence_references=references,
        evidence_bundle=bundle,
        diagnosis_term_recall=_diagnosis_term_recall(case, root_cause),
        essential_evidence_recall=citation_score.metrics.essential_recall_at_256k,
        retrieved_essential_evidence_recall=(
            retrieved_score.metrics.essential_recall_at_256k
        ),
        recovery_kind_correct=recovery_kind == case.expected_recovery_kind,
        input_tokens=budgeted_provider.input_tokens,
        output_tokens=budgeted_provider.output_tokens,
        tool_calls=tool_calls,
        elapsed_ms=(time.perf_counter() - started) * 1000,
        status=outcome.status,
    )
