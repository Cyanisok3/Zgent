# Rejected candidate: HF #39285

This candidate is excluded from dev and test scoring.

- The target upstream failure depends on Hugging Face Hub streaming-split semantics.
- Local `text` data under pinned Datasets 3.6 accepts `train[:20%]`, so the reversed
  code completes training instead of reproducing the reported `Bad split` failure.
- The clean streaming path performs real Trainer steps, but a local reproduction would
  require a fabricated exception or a mock Hub service. Both violate the benchmark's
  provenance and minimum-design rules.
- Observed attempts are retained only in ignored local artifacts. This rejection must not
  be reported as an admitted benchmark case.
