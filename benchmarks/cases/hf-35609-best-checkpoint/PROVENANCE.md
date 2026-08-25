# Provenance note

Issue #35609 documents the exact state sequence: evaluation updates `best_model_checkpoint` at a
step that is not saved, then checkpoint rotation calls `samefile` on the nonexistent path. This
local Trainer run uses deterministic metrics to create that sequence. The safe user-level fix aligns
evaluation and save cadence; the upstream library fix moved best-checkpoint assignment into saving.
