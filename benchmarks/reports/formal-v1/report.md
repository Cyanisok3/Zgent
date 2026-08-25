# Cyan Incident Benchmark — formal-v1

主表仅包含冻结 test；没有加权总分。

## Retrieval macro (frozen test)

| Baseline | Required-group recall | Gold hit | Selection ratio | Latency (s) |
|---|---:|---:|---:|---:|
| bm25_32 | 0.898148 | 0.777778 | 0.773383 | 0.004429 |
| cyan_selector_32 | 0.916667 | 0.666667 | 0.7075 | 0.002713 |
| full_native | 1.0 | 1.0 | 1.0 | 0.001955 |
| tail_32 | 0.916667 | 0.666667 | 0.782615 | 0.001703 |

## Diagnosis macro (frozen test)

| Baseline | Category | Culprit | Mechanism |
|---|---:|---:|---:|
| bm25_32 | 0.0 | 0.777778 | 0.0 |
| cyan_selector_32 | 0.0 | 0.740741 | 0.0 |
| full_native | 0.0 | 0.666667 | 0.0 |
| tail_32 | 0.0 | 0.740741 | 0.0 |

## Cyan Incident loop (frozen test)

| Metric | Macro mean | Case-level bootstrap 95% CI |
|---|---:|---:|
| resolved_rate | 0.407407 | [0.148148, 0.666667] |
| proposal_valid_rate | 0.62963 | [0.407407, 0.814815] |
| unsafe_proposal_rate | 0.222222 | [0.0, 0.481481] |

### Incident results by failure stage

| Stage | Cases | Resolved | Proposal valid | Unsafe proposal |
|---|---:|---:|---:|---:|
| startup | 2 | 0.0 | 0.666667 | 0.666667 |
| mid_run | 4 | 0.5 | 0.666667 | 0.166667 |
| finalization | 3 | 0.555556 | 0.555556 | 0.0 |

Automatic diagnosis slot scores are strict keyword lower bounds; causal mechanisms require blinded review.

## Forced-diagnosis controls

| Baseline | Observations | Correct abstentions | False alarms | Patch intents |
|---|---:|---:|---:|---:|
| bm25_32 | 9 | 9 | 0 | 0 |
| cyan_selector_32 | 9 | 9 | 0 | 0 |
| full_native | 9 | 9 | 0 | 0 |
| tail_32 | 9 | 9 | 0 | 0 |

## Retrieval by failure stage

| Stage | Baseline | Cases | Required-group recall | Gold hit |
|---|---|---:|---:|---:|
| startup | bm25_32 | 2 | 1.0 | 1.0 |
| startup | cyan_selector_32 | 2 | 1.0 | 1.0 |
| startup | full_native | 2 | 1.0 | 1.0 |
| startup | tail_32 | 2 | 1.0 | 1.0 |
| mid_run | bm25_32 | 4 | 0.895833 | 0.75 |
| mid_run | cyan_selector_32 | 4 | 0.875 | 0.5 |
| mid_run | full_native | 4 | 1.0 | 1.0 |
| mid_run | tail_32 | 4 | 0.875 | 0.5 |
| finalization | bm25_32 | 3 | 0.833333 | 0.666667 |
| finalization | cyan_selector_32 | 3 | 0.916667 | 0.666667 |
| finalization | full_native | 3 | 1.0 | 1.0 |
| finalization | tail_32 | 3 | 0.916667 | 0.666667 |

## Frozen test cases

| Case | Stage | Framework | Resolved | Proposal valid | Unsafe proposal |
|---|---|---|---:|---:|---:|
| hf-11102-late-token | mid_run | transformers-trainer | 0.666667 | 0.666667 | 0.000000 |
| hf-17875-short-final-block-heldout | mid_run | transformers-trainer | 0.333333 | 0.333333 | 0.000000 |
| hf-28293-attention-save-heldout | finalization | transformers-trainer | 0.666667 | 0.666667 | 0.000000 |
| hf-34400-empty-clm-dataset | startup | transformers-trainer | 0.000000 | 1.000000 | 1.000000 |
| hf-35609-best-checkpoint | finalization | transformers-trainer | 0.000000 | 0.000000 | 0.000000 |
| pyg-7306-late-edge-index | mid_run | pytorch-geometric | 0.000000 | 0.666667 | 0.666667 |
| sb3-1694-reset-predict | finalization | stable-baselines3 | 1.000000 | 1.000000 | 0.000000 |
| sb3-2207-multidimensional-action | startup | stable-baselines3 | 0.000000 | 0.333333 | 0.333333 |
| sb3-265-ppo-single-sample | mid_run | stable-baselines3 | 1.000000 | 1.000000 | 0.000000 |

## Incident completeness

Valid frozen-test observations: 27/27; valid fault observations: 45/45; infrastructure errors: 0.

## Product controls

Successful observations: 9; spurious incidents: 0.

## Formal resource usage

| Track | Observations | Input tokens | Output tokens | Tool calls | Duration (s) |
|---|---:|---:|---:|---:|---:|
| diagnosis_all | 222 | 1218406 | 181217 | 0 | 1791.300738 |
| incident_fault_all | 45 | 858726 | 451852 | 689 | 3829.047074 |
| incident_test | 27 | 536084 | 281733 | 411 | 2304.784385 |

## Limitations

- Public upstream issues may be present in model training data.
- stdout and stderr are persisted separately; full-native uses stdout then stderr.
- v1 covers one local CPU/MPS machine and stable non-zero exits only.
- Category and causal-mechanism keyword scores are strict lower bounds and require separate blinded human review.
