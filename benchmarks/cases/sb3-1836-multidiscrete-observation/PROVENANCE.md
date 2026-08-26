# Provenance note

Issue #1836 shows SB3 PPO failing to construct a policy from a multidimensional
`MultiDiscrete` observation space: `get_flattened_obs_dim` returns an ndarray and the
`FlattenExtractor` assert fails with "The truth value of an array with more than one
element is ambiguous". Upstream PR #2003 only adds an env-checker warning; SB3 still does
not support multidimensional MultiDiscrete observations. The recommended resolution is a
user-side flattening wrapper, so the case is deliberately not safe for automatic patching.
The fixed variant demonstrates one valid one-dimensional nvec mapping.
