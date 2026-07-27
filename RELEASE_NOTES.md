# Release notes

## cyan 0.0.1

This release is a hard rename. No command aliases, import aliases, or runtime migration branch are
provided.

One-time manual migration for an existing local checkout:

```bash
mv ~/.kama ~/.cyan
mv .kama .cyan
```

Then update environment variable prefixes from `KAMA_` to `CYAN_` and reinstall the package so the
`cyan` and `cyan-core` entry points replace the previous commands.
