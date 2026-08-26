# Issue tracker: Local Markdown

Issues and specs for this repo live as Markdown files in `.scratch/`.

## Conventions

- One feature per directory: `.scratch/<feature-slug>/`
- The spec is `.scratch/<feature-slug>/spec.md`
- Implementation issues are stored one per file at
  `.scratch/<feature-slug>/issues/<NN>-<slug>.md`, numbered from `01`
- Comments and conversation history are appended under a `## Comments` heading

## Publishing and fetching

When a skill says “publish to the issue tracker”, create the corresponding file under
`.scratch/<feature-slug>/`.

When a skill says “fetch the relevant ticket”, read the referenced local Markdown file.

## Wayfinding operations

- Map: `.scratch/<effort>/map.md`
- Child ticket: `.scratch/<effort>/issues/<NN>-<slug>.md`
- Each ticket records `Type:` and `Status:` near the top
- Blocking is recorded as `Blocked by: NN, NN`
- The frontier consists of open, unblocked and unclaimed tickets; the lowest number wins
- Claim a ticket by setting `Status: claimed` before starting work
- Resolve it by adding an `## Answer`, setting `Status: resolved`, and adding a context
  pointer to the map’s `Decisions so far`
