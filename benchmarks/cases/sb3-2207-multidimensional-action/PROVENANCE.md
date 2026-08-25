# Provenance note

Issue #2207 provides this multidimensional `MultiDiscrete` structure and the model-construction
failure. The upstream PR warns that such action spaces are unsupported and recommends a wrapper;
it does not define a semantics-preserving flattening for arbitrary environments. The fixed variant
demonstrates one valid toy mapping, but the case is deliberately not safe for automatic patching.
