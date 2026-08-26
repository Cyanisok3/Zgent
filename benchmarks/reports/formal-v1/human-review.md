# Cyan Benchmark Formal v1 — Human Review

> Review protocol: `two-reviewer, baseline-blind, rubric-calibrated, third-reviewer-adjudicated`.

## Review integrity

- Packet SHA-256: `5c77d5e28ad74e4eae41c483c28156dbbbb8c097bfcf588222583a93bdfd61ae`
- Reviewed items: 144 / 144
- Composition: 108 frozen-test fault + 12 test Control + 24 dev Control
- Review notes: 144
- Unresolved adjudication items: 0
- Parser/schema failures retained as failures: 1
- Initial exact-item agreement: 61/144
- Retest exact-item agreement on the selected disagreement subset: 39/83
- Third-reviewer adjudication: 58 fields across 44 items

### Inter-rater raw agreement

The retest columns cover only the original disagreement subset and are not directly comparable with the full-packet rates.

| Field | Initial full packet | Retest disagreement subset |
|---|---:|---:|
| verdict_correct | 144/144 (100.0%) | 83/83 (100.0%) |
| patch_intent_correct | 144/144 (100.0%) | 83/83 (100.0%) |
| category_score | 124/144 (86.1%) | 68/83 (81.9%) |
| culprit_score | 121/144 (84.0%) | 59/83 (71.1%) |
| mechanism_score | 142/144 (98.6%) | 79/83 (95.2%) |
| evidence_support_score | 79/144 (54.9%) | 68/83 (81.9%) |

## Frozen-test fault results

Ordinal fields show `strict / acceptable`, where strict means score 2 and acceptable means score at least 1.
Binary and ordinal rates are case-first macro averages over nine frozen test cases.

| Baseline | Verdict | Category | Culprit | Mechanism | Evidence | Patch intent |
|---|---:|---:|---:|---:|---:|---:|
| full_native | 100.0% | 77.8% / 92.6% | 59.3% / 96.3% | 85.2% / 100.0% | 55.6% / 100.0% | 66.7% |
| tail_32 | 96.3% | 63.0% / 88.9% | 66.7% / 92.6% | 88.9% / 96.3% | 48.1% / 92.6% | 63.0% |
| bm25_32 | 100.0% | 74.1% / 96.3% | 55.6% / 100.0% | 92.6% / 96.3% | 59.3% / 100.0% | 66.7% |
| cyan_selector_32 | 100.0% | 85.2% / 96.3% | 48.1% / 88.9% | 92.6% / 96.3% | 55.6% / 100.0% | 66.7% |

## Paired Cyan Selector differences

Positive values favor `cyan_selector_32`; brackets are paired case-bootstrap 95% intervals.

| Comparator | Verdict | Category strict | Culprit strict | Mechanism strict | Evidence strict | Patch intent |
|---|---:|---:|---:|---:|---:|---:|
| full_native | +0.0 [+0.0, +0.0] pp | +7.4 [+0.0, +18.5] pp | -11.1 [-33.3, +14.8] pp | +7.4 [-7.4, +25.9] pp | +0.0 [+0.0, +0.0] pp | +0.0 [+0.0, +0.0] pp |
| tail_32 | +3.7 [+0.0, +11.1] pp | +22.2 [+7.4, +40.7] pp | -18.5 [-33.3, -3.7] pp | +3.7 [-11.1, +22.2] pp | +7.4 [+0.0, +18.5] pp | +3.7 [+0.0, +11.1] pp |
| bm25_32 | +0.0 [+0.0, +0.0] pp | +11.1 [+0.0, +22.2] pp | -7.4 [-29.6, +11.1] pp | -0.0 [-11.1, +11.1] pp | -3.7 [-11.1, +0.0] pp | +0.0 [+0.0, +0.0] pp |

## Results by failure stage

| Stage | Baseline | Cases | Verdict | Mechanism strict | Evidence strict | Patch intent |
|---|---|---:|---:|---:|---:|---:|
| startup | full_native | 2 | 100.0% | 100.0% | 50.0% | 0.0% |
| startup | tail_32 | 2 | 100.0% | 100.0% | 50.0% | 0.0% |
| startup | bm25_32 | 2 | 100.0% | 100.0% | 66.7% | 0.0% |
| startup | cyan_selector_32 | 2 | 100.0% | 83.3% | 50.0% | 0.0% |
| mid_run | full_native | 4 | 100.0% | 83.3% | 50.0% | 75.0% |
| mid_run | tail_32 | 4 | 100.0% | 91.7% | 41.7% | 75.0% |
| mid_run | bm25_32 | 4 | 100.0% | 83.3% | 50.0% | 75.0% |
| mid_run | cyan_selector_32 | 4 | 100.0% | 91.7% | 50.0% | 75.0% |
| finalization | full_native | 3 | 100.0% | 77.8% | 66.7% | 100.0% |
| finalization | tail_32 | 3 | 88.9% | 77.8% | 55.6% | 88.9% |
| finalization | bm25_32 | 3 | 100.0% | 100.0% | 66.7% | 100.0% |
| finalization | cyan_selector_32 | 3 | 100.0% | 100.0% | 66.7% | 100.0% |

## Normal controls

Controls are descriptive only and do not enter the 108-item frozen-test fault result.
The test split contains one Control case; two additional Control cases come from dev.

### Test-split Control

| Baseline | N | Correct abstention | False alarm | Unnecessary patch intent |
|---|---:|---:|---:|---:|
| full_native | 3 | 3/3 | 0/3 | 0/3 |
| tail_32 | 3 | 3/3 | 0/3 | 0/3 |
| bm25_32 | 3 | 3/3 | 0/3 | 0/3 |
| cyan_selector_32 | 3 | 3/3 | 0/3 | 0/3 |

### All cross-split Controls

| Baseline | N | Correct abstention | False alarm | Unnecessary patch intent |
|---|---:|---:|---:|---:|
| full_native | 9 | 9/9 | 0/9 | 0/9 |
| tail_32 | 9 | 9/9 | 0/9 | 0/9 |
| bm25_32 | 9 | 9/9 | 0/9 | 0/9 |
| cyan_selector_32 | 9 | 9/9 | 0/9 | 0/9 |

## Interpretation boundaries

- No weighted overall score is constructed.
- Deterministic case-level bootstrap 95% intervals for every metric are stored in `human-review.json`.
- Inter-rater reliability is reported as raw agreement only; no chance-corrected or weighted coefficient is claimed.
- Retest agreement is conditional on the selected initial-disagreement subset and must not be compared directly with full-packet agreement.
- Public upstream issues may be present in model training data, so the benchmark is not contamination-free.
- Results cover stable CPU/MPS non-zero exits, not CUDA/NCCL, OOM, hangs or silent convergence failures.
