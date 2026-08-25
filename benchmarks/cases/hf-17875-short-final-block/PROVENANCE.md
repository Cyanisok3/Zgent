# Provenance note

Issue #17875 reports that the distributed `run_clm.py` grouping function retained a final sequence
shorter than `block_size`; a much later default-collator batch then failed on unequal lengths. The
current script unconditionally rounds down. This local port preserves the same grouping function,
default collator, causal LM training path, and delayed failure without requiring a remote dataset.
The locked environment supplies the package; inherited host Python path variables are removed.
