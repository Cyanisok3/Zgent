# Domain Docs

How engineering skills consume this repository's domain documentation.

## Before exploring

Read `CONTEXT.md` at the repository root and relevant ADRs under `docs/adr/`.

If these files do not exist, proceed silently. Domain-modeling creates them lazily when
terminology or decisions are actually resolved.

## File structure

```text
/
├── CONTEXT.md
├── docs/adr/
└── src/
```

## Vocabulary

Use domain concepts as defined in `CONTEXT.md`. Do not drift to synonyms that the glossary
explicitly avoids.

If a required concept is missing, reconsider whether the term belongs to the project or record
the gap for domain-modeling.

## ADR conflicts

If proposed work contradicts an existing ADR, surface the conflict explicitly rather than
silently overriding it.
