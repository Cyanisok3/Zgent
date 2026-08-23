# AGENTS.md

## Design principles

Think before coding: state material assumptions and trade-offs. Prefer the smallest change that
solves the requested problem. Do not add speculative frameworks, databases, dashboards, compatibility
shells, or generic Agent capabilities. Preserve unrelated user changes.

Incident integration tests use real temporary Git repositories and real subprocesses; never fake a
frontend success result. Root documentation describes the current product only.

## Product boundary

cyan is a local ML-training Incident Agent, not an AutoResearch system.

- The daemon supervises the real training process and persists stdout/stderr.
- A non-zero exit creates a Failure Capsule and wakes a fresh read-only Incident Agent.
- The Agent can diagnose and propose one exact single-file SEARCH/REPLACE, but never writes the
  workspace, runs shell, or generates executable diff syntax.
- One explicit approval lets the harness validate/apply the proposal. An optional user smoke verifier
  runs before the original command, whose exit code is always the final verifier.
- v1 does not perform paper search, metric optimization, hyperparameter search, experiment loops,
  ordinary Chat, Skills, MCP, Task, subagents, or Python Sandbox execution.

The default user surface is the TUI:

```bash
cyan
# use /monitor, paste one training command, review the preview, then use /start
```

The only local commands are `/monitor`, `/start`, `/jobs`, `/incident <text>`, and `/help`. Ordinary
text in the default state is rejected; `/monitor` interprets text as a training command and
`/incident` is the explicit follow-up route. Patch approval, rejection, smoke choice, and running-job
cancellation are contextual controls, not global slash commands.

Public process/CLI surface:

```text
cyan                         # TUI
cyan --version
cyan core start [--json]     # daemon discovery/start
cyan core stop               # RPC shutdown
cyan-core                    # daemon process
```

`watch`, `chat`, `run`, `ping`, `trace`, `core status`, and the separate `cyan-tui` console script do
not exist.

`vscode/` is a thin optional local client. It may display native logs, Markdown and diffs, but Python
daemon snapshots remain authoritative; it must not reimplement launch parsing, Incident state,
patching, retries, or Agent logic.

## Architecture

```text
cyan / cyan VS Code
    │ TCP loopback, NDJSON framing, JSON-RPC 2.0
cyan-core
    ├── service/    daemon, protocol, transport
    ├── agent/      AgentRunner, AgentLoop, EventBus, ToolRegistry
    └── training/   JobSupervisor, JobStore, IncidentCoordinator/Runtime
```

The transport is TCP on `127.0.0.1:7437`, not a Unix socket. Internal modules are grouped by
ownership: `agent` is generic and depends on neither `training` nor `service`; `training.jobs` does
not depend on Agent; `training.incidents` adapts the Agent; `service` composes both.

### Protocol

Pydantic v2 command/event models are discriminated by `type`. The wire protocol version is `2`.
The only Agent-facing operation is the Incident-owned `incident.follow_up`; generic Agent, Session,
Permission, Skill, Subagent and Compaction RPCs/events are removed. `event.subscribe` retains
`replay_from_run` because a TUI can receive a run ID before subscribing.

After changing protocol models, regenerate and check [WIRE_PROTOCOL.md](WIRE_PROTOCOL.md):

```bash
uv run python scripts/gen_protocol_doc.py
uv run python scripts/gen_protocol_doc.py --check
```

### Job runtime

`JobSupervisor` is the only owner of long-running process groups. It uses
`asyncio.create_subprocess_exec`, separate stdout/stderr drains, and exact persisted argv/cwd/env for
retries. It never uses a generic shell tool. Daemon-owned training and smoke processes use
`DEVNULL` stdin.

```text
~/.cyan/jobs/<job_id>/
├── job.json
├── events.jsonl
├── launch.json                 # private, 0600
├── attempts/<attempt_id>/{attempt.json,stdout.log,stderr.log,failure.json}
└── incidents/<incident_id>/
```

### Incident runtime and tools

Every run rebuilds its fixed Incident profile. It has a 12-step cap, a 128 KiB serialized input
budget, a 32 KiB initial selector budget, and exactly six tools:

```text
read_file, list_dir, search_text, read_job_log, submit_diagnosis, propose_patch
```

The first four are read-only; the last two only write Incident artifacts. No shell, write, MCP,
Skill, Task, subagent, permission, session or compaction module is registered.

The deterministic v1 selector scans immutable logs in chunks, returns byte-range references and does
not read source, call an LLM, cache results, or change tool permissions. Context contains only the
current Failure Capsule, selected evidence, current instruction, prior bounded outcome, smoke/retry
result, and the fixed system prompt.

`propose_patch` performs one exact unique replacement in memory and writes only a harness-generated
diff. v1 edits one existing UTF-8 text file of at most 1 MiB; `PatchService` owns hash checks,
workspace/path checks, `git apply --check`, application, and safe reverse application.

### Persistence

```text
~/.cyan/jobs/<job_id>/incidents/<incident_id>/
├── incident.json
├── proposal.diff              # only while a proposal exists
├── smoke.stdout.log           # only if smoke ran
├── smoke.stderr.log           # only if smoke ran
└── runs/<run_id>/{run.json,events.jsonl}
```

`incident.json` is the current Incident snapshot and contains diagnosis, proposal metadata, apply
receipt, smoke execution/result and FSM state. `run.json` contains only bounded run metadata,
selection references, byte metrics and structured outcomes. Logs, full tool output and token text are
not duplicated in Incident artifacts. `run.json` is atomically written before its background task
starts. Old `~/.cyan/sessions` is neither migrated nor read.

Raw process output remains only in Attempt logs. IPC queues are bounded; daemon trace stores summaries,
not log, patch, tool-I/O or user text payloads.

### Configuration and recovery

Configuration priority is defaults, `~/.cyan/config.toml`, project `.cyan/config.toml`, `.env`, then
`CYAN_*` environment variables. The daemon resolves its startup configuration once. The optional
smoke declaration is `[incident.smoke]` in the Job workspace and cannot be modified by the Agent.
Smoke process identity is stored in `incident.json.smoke_execution`; recovery terminates it only when
PID, identity, session leader and process-group leader match.

## Verification

```bash
uv sync
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
cd vscode
npm run lint
npm run test:unit
npm run test:extension
npm run package
```

## Code style

Every function has one concise Chinese comment immediately above its `def` line. Every test function
has two Chinese comment lines immediately above its `def` line. Do not replace these comments with
multi-line function docstrings. TypeScript functions and test cases follow the same concise Chinese
comment rule.
