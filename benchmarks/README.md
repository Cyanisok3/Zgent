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

The automatic category and causal-mechanism fields are strict keyword lower bounds. They remain
pending independent blinded human review and must not be presented as final semantic accuracy.
