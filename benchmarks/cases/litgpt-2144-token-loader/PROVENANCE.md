# Provenance note

The upstream TinyStories path downloaded and processed 100,000 stories before failing. This port
uses local deterministic token arrays but preserves LitData 0.2.58, `optimize`, `TokensLoader`, the
generated chunk metadata, and `StreamingDataset`. The official PR adds the same missing
`item_loader=TokensLoader()` argument in LitGPT's TinyStories module.
