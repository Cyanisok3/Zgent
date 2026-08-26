from __future__ import annotations

import argparse
import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

from cyan.config import get_config
from cyan.training.incidents.context import INCIDENT_PROMPT_VERSION

from cyan_bench.admission import admit_case
from cyan_bench.audit import audit_dataset
from cyan_bench.baselines import select_baseline
from cyan_bench.cases import LoadedCase, discover_cases, load_case
from cyan_bench.diagnosis import (
    DIAGNOSIS_MAX_OUTPUT_TOKENS,
    DIAGNOSIS_PROMPT_VERSION,
    DIAGNOSIS_REASONING_EFFORT,
    DIAGNOSIS_TEMPERATURE,
    run_diagnosis,
)
from cyan_bench.execution import (
    discard_workspace,
    prepare_environment,
    prepare_repository,
    prepare_workspace,
)
from cyan_bench.incident_track import run_incident_track
from cyan_bench.models import (
    Baseline,
    DiagnosisRunArtifact,
    IncidentBenchmarkArtifact,
    ProcessCapture,
    Variant,
)
from cyan_bench.paths import BenchmarkPaths, benchmark_paths
from cyan_bench.reporting import write_report
from cyan_bench.scoring import score_selection

_RANKED_BASELINES: tuple[Baseline, ...] = (
    "full_native",
    "tail_32",
    "bm25_32",
    "cyan_selector_32",
)
_ALL_BASELINES: tuple[Baseline, ...] = (*_RANKED_BASELINES, "oracle_32")


# 按 ID 加载单个案例并给出稳定的缺失错误
def _case(case_id: str, paths: BenchmarkPaths) -> LoadedCase:
    directory = paths.cases / case_id
    if not directory.is_dir():
        raise SystemExit(f"unknown case: {case_id}")
    return load_case(directory)


# 准备上游源码与锁定训练环境
def _prepare(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    case = _case(args.case, paths)
    repository = prepare_repository(case, paths)
    environment = prepare_environment(case, paths)
    print(
        json.dumps(
            {
                "case_id": case.manifest.id,
                "repository": str(repository),
                "environment": str(environment),
            }
        )
    )


# 执行三变体三重复准入并以退出码表达结果
def _admit(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    case = _case(args.case, paths)
    artifact = admit_case(case, paths, repeats=args.repeats)
    print(artifact.model_dump_json(indent=2))
    if not artifact.admitted:
        raise SystemExit(1)


# 读取一次已准入进程 artifact
def _capture(
    case_id: str,
    variant: str,
    repeat: int,
    paths: BenchmarkPaths,
) -> tuple[ProcessCapture, Path]:
    run_dir = paths.artifacts / "captures" / case_id / variant / str(repeat)
    capture = ProcessCapture.model_validate_json(
        (run_dir / "process.json").read_text(encoding="utf-8")
    )
    return capture, run_dir


# 按 split、数据集版本与可选 case ID 过滤本轮运行范围
def _selected_cases(args: argparse.Namespace, paths: BenchmarkPaths) -> list[LoadedCase]:
    cases = discover_cases(
        paths.cases,
        args.split,
        dataset_version=str(args.dataset),
    )
    requested = set(args.case or [])
    if requested:
        available = {case.manifest.id for case in cases}
        missing = sorted(requested - available)
        if missing:
            raise SystemExit(f"cases not in {args.split} split: {', '.join(missing)}")
        cases = [case for case in cases if case.manifest.id in requested]
    if args.controls:
        cases = [case for case in cases if case.manifest.control_role is not None]
    if not cases:
        raise SystemExit("no cases matched the requested run scope")
    return cases


# 选择本轮显式请求的诊断 baseline
def _selected_baselines(args: argparse.Namespace, case: LoadedCase) -> tuple[Baseline, ...]:
    requested = set(args.baseline or [])
    baselines: tuple[Baseline, ...] = tuple(
        item for item in _RANKED_BASELINES if not requested or item in requested
    )
    include_oracle = (
        case.manifest.split == "dev"
        and not args.controls
        and (not requested or "oracle_32" in requested)
    )
    return (*baselines, "oracle_32") if include_oracle else baselines


# 选择本轮显式请求的重复编号
def _selected_repeats(args: argparse.Namespace) -> tuple[int, ...]:
    requested = args.repeat or []
    return tuple(sorted(set(requested))) if requested else (1, 2, 3)


# 判断已有诊断 artifact 是否是可安全跳过的模型结果
def _diagnosis_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    artifact = DiagnosisRunArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    return artifact.status != "infrastructure_error"


# 判断已有 Incident artifact 是否已经完成且没有 harness 错误
def _incident_complete(path: Path) -> bool:
    if not path.is_file():
        return False
    artifact = IncidentBenchmarkArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if artifact.error is not None:
        return False
    for run_path in path.parent.glob("cyan-jobs/*/incidents/*/runs/*/run.json"):
        run = json.loads(run_path.read_text(encoding="utf-8"))
        if run.get("reason") == "llm_error":
            return False
    return True


# 清理单次未完成或基础设施失败的 Incident 输出
def _reset_incident_output(output_dir: Path, paths: BenchmarkPaths) -> None:
    if not output_dir.exists():
        return
    output_dir.relative_to(paths.artifacts / "run-sets")
    shutil.rmtree(output_dir)


# 为一个 split 运行固定基线和可选真实模型诊断
async def _run_diagnosis_track(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    cases = _selected_cases(args, paths)
    variant: Variant = "control" if args.controls else "buggy"
    for case in cases:
        admission = paths.artifacts / "admissions" / case.manifest.id / "admission.json"
        if not admission.is_file():
            raise SystemExit(f"case is not admitted: {case.manifest.id}")
        for repeat in _selected_repeats(args):
            capture, capture_dir = _capture(case.manifest.id, variant, repeat, paths)
            baselines = _selected_baselines(args, case)
            if "oracle_32" in baselines and repeat != 1:
                baselines = tuple(item for item in baselines if item != "oracle_32")
            for baseline in baselines:
                output_dir = (
                    paths.artifacts
                    / "run-sets"
                    / args.run_set
                    / "diagnosis"
                    / case.manifest.id
                    / variant
                    / str(repeat)
                    / baseline
                )
                selection = select_baseline(
                    case,
                    capture,
                    capture_dir,
                    baseline,
                    output_dir,
                )
                if variant == "buggy":
                    score = score_selection(selection, capture_dir / "gold-ranges.json")
                    (output_dir / "selection-score.json").write_text(
                        json.dumps(score, ensure_ascii=False, indent=2) + "\n",
                        encoding="utf-8",
                    )
                if not args.no_llm:
                    diagnosis_path = output_dir / "diagnosis.json"
                    if args.resume and _diagnosis_complete(diagnosis_path):
                        continue
                    await run_diagnosis(
                        case,
                        capture,
                        selection,
                        output_dir,
                        output_dir,
                        is_control=args.controls,
                    )


# 在全新临时 Git 中运行当前 Cyan Incident 闭环
async def _run_incident_track(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    cases = _selected_cases(args, paths)
    variant: Variant = "control" if args.controls else "buggy"
    for case in cases:
        admission = paths.artifacts / "admissions" / case.manifest.id / "admission.json"
        if not admission.is_file():
            raise SystemExit(f"case is not admitted: {case.manifest.id}")
        for repeat in _selected_repeats(args):
            output_dir = (
                paths.artifacts
                / "run-sets"
                / args.run_set
                / "incident"
                / case.manifest.id
                / variant
                / str(repeat)
            )
            incident_path = output_dir / "incident-benchmark.json"
            if args.resume and _incident_complete(incident_path):
                continue
            _reset_incident_output(output_dir, paths)
            workspace = prepare_workspace(case, paths, variant)
            try:
                await run_incident_track(
                    case,
                    paths,
                    workspace,
                    repeat,
                    output_dir,
                    is_control=args.controls,
                )
            finally:
                discard_workspace(workspace, paths)


# 冻结 run-set 固定配置；配置不一致或向旧 run-set 写入 v2 时拒绝
def _freeze_run_set(
    run_root: Path,
    dataset_version: str,
    split: str,
    selected_case_ids: tuple[str, ...],
    requested_model: str,
) -> None:
    run_root.mkdir(parents=True, exist_ok=True)
    path = run_root / "run-set.json"
    expected: dict[str, object] = {
        "dataset_version": dataset_version,
        "split": split,
        "selected_case_ids": list(selected_case_ids),
        "requested_model": requested_model,
        "diagnosis_prompt_version": DIAGNOSIS_PROMPT_VERSION,
        "incident_prompt_version": INCIDENT_PROMPT_VERSION,
        "temperature": DIAGNOSIS_TEMPERATURE,
        "reasoning_effort": DIAGNOSIS_REASONING_EFFORT,
        "max_output_tokens": DIAGNOSIS_MAX_OUTPUT_TOKENS,
    }
    if path.is_file():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise SystemExit(
                    f"run-set {run_root.name} config mismatch for {key}: "
                    f"frozen={existing.get(key)!r} requested={value!r}; refusing to continue"
                )
        return
    has_artifacts = any(run_root.rglob("*.json")) if run_root.exists() else False
    if has_artifacts:
        raise SystemExit(
            f"legacy run-set {run_root.name} has artifacts but no run-set.json; "
            "it can only be read as formal-v1 history, use a new run-set name"
        )
    expected["created_at"] = _now_iso()
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(expected, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


# 返回当前本地时间的 ISO 字符串
def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


# 分派无工具 diagnosis 或当前完整 Incident track
def _run(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    if args.track == "incident" and args.baseline:
        raise SystemExit("--baseline is only valid for the diagnosis track")
    cases = _selected_cases(args, paths)
    requested_model = str(get_config().llm.default_model)
    _freeze_run_set(
        paths.artifacts / "run-sets" / args.run_set,
        str(args.dataset),
        str(args.split),
        tuple(sorted(case.manifest.id for case in cases)),
        requested_model,
    )
    if args.track == "incident":
        asyncio.run(_run_incident_track(args, paths))
    else:
        asyncio.run(_run_diagnosis_track(args, paths))


# 生成 JSON 与 Markdown 聚合报告
def _report(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    run_root = paths.artifacts / "run-sets" / args.run_set
    if not (run_root / "run-set.json").is_file():
        raise SystemExit(
            f"legacy run-set {args.run_set} is read-only; use its committed formal-v1 report"
        )
    json_path, markdown_path = write_report(args.run_set, paths)
    print(json.dumps({"json": str(json_path), "markdown": str(markdown_path)}))


# 输出数据集配额与真实性门禁结果
def _audit(args: argparse.Namespace, paths: BenchmarkPaths) -> None:
    result = audit_dataset(
        paths,
        dataset_version=str(args.dataset),
        scope=str(args.scope),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["ready"]:
        raise SystemExit(1)


# 构建最小 benchmark 命令行接口
def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cyan-bench")
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("case")
    prepare.set_defaults(handler=_prepare)
    admit = subparsers.add_parser("admit")
    admit.add_argument("case")
    admit.add_argument("--repeats", type=int, choices=(1, 3), default=3)
    admit.set_defaults(handler=_admit)
    run = subparsers.add_parser("run")
    run.add_argument("--split", choices=("dev", "test"), required=True)
    run.add_argument("--track", choices=("diagnosis", "incident"), required=True)
    run.add_argument("--run-set", required=True)
    run.add_argument("--dataset", choices=("formal-v1", "formal-v2"), default="formal-v1")
    run.add_argument("--case", action="append")
    run.add_argument("--baseline", action="append", choices=_ALL_BASELINES)
    run.add_argument("--repeat", type=int, action="append", choices=(1, 2, 3))
    run.add_argument("--controls", action="store_true")
    run.add_argument("--no-llm", action="store_true")
    run.add_argument("--resume", action="store_true")
    run.set_defaults(handler=_run)
    report = subparsers.add_parser("report")
    report.add_argument("run_set")
    report.set_defaults(handler=_report)
    audit = subparsers.add_parser("audit")
    audit.add_argument("--dataset", choices=("formal-v1", "formal-v2"), default="formal-v1")
    audit.add_argument("--scope", choices=("dev", "release"), default="release")
    audit.set_defaults(handler=_audit)
    return parser


# 解析参数并执行对应 benchmark 子命令
def main() -> None:
    args = _parser().parse_args()
    paths = benchmark_paths()
    Path(paths.artifacts).mkdir(parents=True, exist_ok=True)
    args.handler(args, paths)
