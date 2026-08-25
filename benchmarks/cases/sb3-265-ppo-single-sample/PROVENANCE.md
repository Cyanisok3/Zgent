# Provenance note

Issue #265 provides the original one-line PPO reproducer and explains the failure chain: one rollout
sample has undefined standard deviation, normalized advantages become NaN, the update corrupts model
parameters, and the next rollout exits non-zero. The fault patch only reverts the current constructor
guard that the maintainers added for this unsupported effective batch size.
