# Cyan Incident Benchmark v1

This project evaluates Cyan on reproducible, provenance-linked ML training failures without adding
training frameworks to Cyan's runtime environment.

The frozen dataset contains 6 development failures, 9 test failures, and 3 successful controls
reused from those cases. Each admitted failure has three successful control runs, three stable
non-zero buggy runs, and three successful fixed runs. Case manifests record the upstream repository,
commit, issue/fix source, dependency lock, failure stage, hardware, and evidence anchors.

## Local workflow

Run benchmark commands from the repository root:

```bash
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench prepare <case-id>
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench admit <case-id>
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench audit
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench run \
  --split test --track diagnosis --run-set offline-v1 --no-llm
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench report offline-v1
```

`--no-llm` runs FullNative, Tail32, BM25-32, and CyanSelector-32 locally and scores their byte-range
evidence. Omitting it sends selected logs to the configured external model. Do that only with explicit
authorization for log transmission and API cost. Full Incident runs use the current Cyan product:

```bash
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench run \
  --split test --track incident --run-set <name>
```

`run` and `audit` accept `--dataset formal-v1|formal-v2` (default `formal-v1`). Each new run-set
writes a frozen `run-set.json` on its first run containing `dataset_version`, `requested_model`,
`diagnosis_prompt_version`, `incident_prompt_version`, `temperature`, `reasoning_effort`,
`max_output_tokens`, and `created_at`; a subsequent run or `--resume` with a different configuration
is rejected. Legacy run-sets without `run-set.json` can only be read as `formal-v1` history and
cannot be written to.

For an authorized, bounded pilot, repeat `--case`, select one `--baseline`, and use `--repeat 1`
to send one request per case. `--resume` skips completed model or Incident artifacts so an interrupted
run does not repeat paid calls:

```bash
PYTHONPATH=benchmarks/src .venv/bin/python -m cyan_bench run \
  --split test --track diagnosis --run-set pilot-token-check \
  --case hf-11102-late-token \
  --case hf-17875-short-final-block-heldout \
  --case hf-28293-attention-save-heldout \
  --baseline full_native --repeat 1 --resume
```

## Reproducibility boundary

- Third-party repositories, environments, raw logs, temporary workspaces, weights, and credentials
  stay outside Git.
- Measurement runs are offline after preparation; stdout and stderr are preserved separately.
- `fault.patch` carries a traced upstream mechanism; `workload.patch` only lengthens genuine training,
  evaluation, saving, or preprocessing work for every variant.
- Public issues may already appear in model training data. The report must disclose that limitation.
- v1 covers deterministic non-zero exits on one local CPU/MPS machine, not CUDA/NCCL, OOM, hangs,
  silent NaNs, or convergence degradation.

## Validation

```bash
cd benchmarks
PYTHONPATH=src ../.venv/bin/python -m ruff check src tests
PYTHONPATH=src ../.venv/bin/python -m mypy src
PYTHONPATH=src ../.venv/bin/python -m pytest tests -q
```

The full paid/model-backed report additionally requires three repetitions per `(case, baseline)` and
manual blinded review of causal-mechanism answers. Offline retrieval results are available under
`reports/offline-v1/` and are not presented as end-to-end diagnosis or repair results.

## Formal v1 result

The committed machine-readable and Markdown summaries are in
[`reports/formal-v1/`](reports/formal-v1/). The frozen protocol used `deepseek-v4-flash`, temperature
0, low reasoning effort, and an 8192-token output cap. It contains 222 no-tool diagnosis
observations, 45 valid fault Incident observations, and 9 normal product controls; no infrastructure
error is included in capability metrics.

The automatic category and causal-mechanism fields are strict keyword lower bounds. Final semantic
scores come from two independent baseline-blind reviewers, a rubric-calibrated disagreement retest,
and third-reviewer adjudication. Raw agreement and its selected-subset limitation are reported in
[`reports/formal-v1/human-review.md`](reports/formal-v1/human-review.md).

新的诊断运行使用 `causal-support-abstention-v2` prompt 版本，并按严格结构化响应模型解析：
新诊断必须提供 `causal_support`、至少一条证据引用和 `patch_recommended`，缺失字段按
schema failure 计，不静默降级；旧的 formal-v1 artifact 仍按旧模型读取。Incident Track B 的
artifact 保存审批前诊断/Proposal 快照与最终验证状态：`proposal_valid` 以审批前
`can_apply=true` 为准，`resolved`/最终状态来自验证后视图。

Incident 指标使用各自适用分母：`unsafe_proposal_rate` 与 `correct_patch_abstention_rate`
只统计不可修复案例；`missed_patch_opportunity_rate` 与 `patchable_resolved_rate` 只统计可修复
案例；`abstention_gate_violated_rate` 统计全部有效 Incident；整体 `resolved_rate` 保留但不能
代替可修复成功率。formal-v2 案例的 `expected.json` 必须提供 `causal_support` 与
`patch_recommended` Gold 字段，v1 允许缺失。新结果写入新的 run-set，不覆盖 `formal-v1`。
可用 `incident_review_export.py` 生成只评审 Incident `root_cause` 的匿名 packet。

```bash
python benchmarks/incident_review_export.py --run-set <name> --split dev \
  --output /tmp/incident-review.csv --key /tmp/incident-review-key.json
```
