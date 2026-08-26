# Provenance note

This case uses the official LitGPT #2048 fixed commit as the repository baseline. `fault.patch`
removes the exact `TokensLoader` import and `item_loader` arguments changed by that commit;
`fix.patch` restores the same one-file upstream change. The overlay runs LitGPT's real
`TextFiles.prepare_data` and LitData `optimize` path with local deterministic text, without
exposing the issue or fix URL to the Agent workspace.
