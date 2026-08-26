# Formal-v2 candidate audit

This short record prevents re-admitting candidates already checked for the abstention benchmark.

| Candidate | Decision | Reason |
|---|---|---|
| SB3 #1968 | reject | Windows/spawn-specific; the macOS runtime does not reproduce the failure mechanism. |
| PyTorch examples #1105 | reject | `share_memory()` is already fixed or succeeds silently in the pinned modern Torch path. |
| Hugging Face #31897 | reject | The referenced MPS issue is fixed by the modern upstream revision under consideration. |
| Defects4ML / EMSE24 pool | hold out | Historical TensorFlow/Python environments are not reproducible in the current local v1 setup. |
| LitGPT #2048 | admitted locally | Official one-file `TextFiles`/`TokensLoader` fix; three control/buggy/fixed repeats passed on CPU. Buggy fails in `generate_roi` with `NoneType // int`; control and fixed exit 0. |

formal-v2 is not released until two additional real held-out abstention cases are found and pass
the same admission gate. This record does not treat a simple parameter or missing-file error as a
replacement.
