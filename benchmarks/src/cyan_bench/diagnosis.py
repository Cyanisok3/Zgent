from __future__ import annotations

import asyncio
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

import anthropic
import httpx
from cyan.config import get_config

from cyan_bench.baselines import capsule_metadata
from cyan_bench.cases import LoadedCase
from cyan_bench.models import (
    DiagnosisAnswerV2,
    DiagnosisRunArtifact,
    ProcessCapture,
    SelectionArtifact,
)

DIAGNOSIS_PROMPT_VERSION = "causal-support-abstention-v4"

_SYSTEM = """You diagnose whether a local machine-learning training process failed.
Use only the supplied process metadata and log evidence. There are no tools.
Return exactly one JSON object and no Markdown:
{
  "verdict": "fault" | "no_fault",
  "diagnosis": null | {
    "category": "short category, at most 80 characters",
    "culprit": "specific file, component, setting, or contract",
    "causal_mechanism": "why the observed process outcome occurred",
    "causal_support": "direct" | "inferred",
    "evidence": [{"source": "stdout|stderr", "start": 0, "end": 1}]
  },
  "patch_recommended": true | false
}
For a successful run without causal failure evidence, use no_fault, null, and false.
When causal_support is direct, locate the earliest evidence-backed upstream configuration,
data contract, component, or producer rather than only the traceback leaf. When it is inferred,
culprit must begin with "Inference — not directly established by observed evidence:" and
must not invent an unobserved file, setting, variable, or dependency version.
Set patch_recommended=true only when the observed cause is a local workspace change in one
existing file with one exact replacement and the original command can verify it. For external
data, environment or framework limitations, dependency changes, multi-file fixes, or insufficient
evidence, set patch_recommended=false. Do not infer a patch merely because warnings exist.
The evidence array must contain at least one {"source", "start", "end"} reference to the log
bytes that support your diagnosis."""

DIAGNOSIS_MAX_OUTPUT_TOKENS = 8192
DIAGNOSIS_TEMPERATURE = 0
DIAGNOSIS_REASONING_EFFORT = "low"


# 返回当前 UTC 时间
def _now() -> datetime:
    return datetime.now(UTC)


# 从 Anthropic 兼容响应中拼接完整文本块
def _response_text(response: Any) -> str:
    return "".join(
        block.text for block in response.content if getattr(block, "type", None) == "text"
    )


# 判断 provider 错误是否明确表示上下文超限
def _is_context_overflow(exc: Exception) -> bool:
    message = str(exc).lower()
    return "context" in message and any(word in message for word in ("long", "limit", "token"))


# 将无工具诊断请求发送给固定模型并保存原始输出与 usage
async def run_diagnosis(
    case: LoadedCase,
    capture: ProcessCapture,
    selection: SelectionArtifact,
    selection_dir: Path,
    output_dir: Path,
    *,
    is_control: bool,
) -> DiagnosisRunArtifact:
    config = get_config()
    model = config.llm.default_model
    evidence = (selection_dir / selection.content_path).read_text(
        encoding="utf-8", errors="replace"
    )
    metadata = capsule_metadata(case, capture)
    user = (
        f"Process metadata:\n{json.dumps(metadata, ensure_ascii=False)}\n\n"
        f"Log evidence:\n{evidence}"
    )
    client = anthropic.AsyncAnthropic()
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    response: Any = None
    error: str | None = None
    status: Literal[
        "success", "schema_error", "context_overflow", "infrastructure_error"
    ] = "infrastructure_error"
    attempts = 0
    for attempts in range(1, 4):
        try:
            response = await client.messages.create(
                model=model,
                max_tokens=DIAGNOSIS_MAX_OUTPUT_TOKENS,
                temperature=DIAGNOSIS_TEMPERATURE,
                output_config=cast(
                    Any, {"effort": DIAGNOSIS_REASONING_EFFORT}
                ),
                system=_SYSTEM,
                messages=[{"role": "user", "content": user}],
            )
            break
        except anthropic.BadRequestError as exc:
            error = str(exc)
            status = "context_overflow" if _is_context_overflow(exc) else "infrastructure_error"
            break
        except (httpx.ConnectError, httpx.ReadError, httpx.RemoteProtocolError) as exc:
            error = str(exc)
            if attempts < 3:
                await asyncio.sleep(float(attempts))
    answer: DiagnosisAnswerV2 | None = None
    raw_path: str | None = None
    if response is not None:
        raw = _response_text(response)
        raw_file = output_dir / "response.txt"
        raw_file.write_text(raw, encoding="utf-8")
        raw_path = raw_file.name
        try:
            # 新运行强制严格结构化诊断；缺 causal_support/evidence/patch_recommended 即 schema_error
            answer = DiagnosisAnswerV2.model_validate_json(raw)
            status = "success"
        except ValueError as exc:
            status = "schema_error"
            error = str(exc)
    usage = getattr(response, "usage", None)
    artifact = DiagnosisRunArtifact(
        case_id=case.manifest.id,
        baseline=selection.baseline,
        repeat=capture.repeat,
        is_control=is_control,
        prompt_version=DIAGNOSIS_PROMPT_VERSION,
        status=status,
        model_requested=model,
        model_resolved=getattr(response, "model", None),
        response_id=getattr(response, "id", None),
        answer=answer,
        raw_response_path=raw_path,
        input_tokens=getattr(usage, "input_tokens", None),
        output_tokens=getattr(usage, "output_tokens", None),
        cache_read_input_tokens=getattr(usage, "cache_read_input_tokens", None),
        cache_creation_input_tokens=getattr(usage, "cache_creation_input_tokens", None),
        duration_seconds=round(time.monotonic() - started, 6),
        transport_attempts=attempts,
        error=error,
        created_at=_now(),
    )
    (output_dir / "diagnosis.json").write_text(
        artifact.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return artifact
