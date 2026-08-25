# Provenance note

The upstream report is a custom mT5 fine-tuning failure and does not provide a complete reproducer.
Maintainers confirm that non-contiguous parameters cannot use safe serialization and recommend
`safe_serialization=False`. This port preserves that exact save-time tensor-layout contract with a
small local BERT training run. The one-line contiguous conversion is a widely verified workaround,
not an upstream Transformers product fix.
