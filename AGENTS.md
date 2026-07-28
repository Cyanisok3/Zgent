# AGENTS.md

## Design principles

### Think before coding

Do not assume or hide ambiguity. State material assumptions and trade-offs.

### Simplicity first

Implement the minimum code that solves the requested problem. Do not add speculative abstractions,
middleware, databases, dashboards, or compatibility layers.

### Surgical changes

Touch only what the task requires. Preserve unrelated user changes and match the existing style.

### Goal-driven execution

For every behavior change, define the observable outcome and verify it with the narrowest useful
test. Incident integration tests must use real temporary Git repositories and real subprocesses;
never fake a frontend result.

## Product boundary

cyan is a local ML-training Incident Agent, not an AutoResearch system.

- During training, the daemon only supervises the real process and persists stdout/stderr.
- A non-zero exit or unexpected signal creates a failure capsule and wakes a fresh read-only Agent.
- The Agent may diagnose a crash and propose a diff, but it cannot write the workspace or run shell.
- One explicit approval lets the harness validate and apply that proposal.
- A user-declared smoke verifier is optional; the original command is always the final verifier.
- v1 does not perform paper search, metric optimization, hyperparameter search, or experiment loops.

The product surface is the TUI. Normal users need only:

```bash
cyan
# Then use /monitor, paste one training command, review the preview, and use /start.
```

`cyan watch -- <argv>` remains a compatibility entrypoint. The remaining CLI commands are
development diagnostics and must not complicate the primary help. `/monitor` is client-local:
the harness parses and previews the command without an Agent or shell, then reuses `job.start`.
Typing `/` must show keyboard-selectable local commands and ordinary chat skills. Ordinary chat
permission requests use the existing `permission.*` events and `permission.respond`; this must not
broaden the read-only Incident profile.

## Architecture

cyan is a dual-process local system:

```text
cyan / cyan-tui
    │ TCP loopback, NDJSON framing, JSON-RPC 2.0 commands
cyan-core
    ├── JobSupervisor
    ├── file-backed JobStore
    ├── IncidentCoordinator
    ├── AgentRunner / EventBus / ToolRegistry
    └── SessionManager
```

The transport is TCP on `127.0.0.1:7437`, not a Unix socket.

### Protocol (`src/cyan/core/bus/`)

Commands and events are Pydantic v2 models discriminated by `type`. When changing these models,
update their unions and regenerate [WIRE_PROTOCOL.md](WIRE_PROTOCOL.md):

```bash
uv run python scripts/gen_protocol_doc.py
uv run python scripts/gen_protocol_doc.py --check
```

### Job runtime (`src/cyan/core/jobs/`)

`JobSupervisor` is the only owner of long-running process groups. It uses
`asyncio.create_subprocess_exec`, separate stdout/stderr drains, and exact persisted argv/cwd/env
for retries. It must never use the generic 120-second `BashTool`.
Daemon-owned training and smoke processes use `DEVNULL` stdin; watched jobs are non-interactive.

`JobStore` persists:

```text
~/.cyan/jobs/<job_id>/
├── job.json
├── events.jsonl
├── launch.json
├── attempts/<attempt_id>/{attempt.json,stdout.log,stderr.log,failure.json}
└── incidents/<incident_id>/...
```

`launch.json` is private (`0600`) and must not be returned by RPC or exposed to the Agent.

### Incident harness (`src/cyan/core/incidents/`)

Incident profiles are rebuilt from persisted metadata on every run. They have a fixed workspace,
12-step cap, disabled compaction, 256 KiB total log evidence budget, and only these tools:

```text
read_file, list_dir, search_text, read_job_log, submit_diagnosis, propose_patch
```

Do not register write, shell, MCP, Skill, TaskManager, or subagent tools for Incident sessions.
Do not load global/project context or session notes, and do not let slash commands or manual
compaction override this profile. v1 must not start configured MCP servers.

`propose_patch` writes an artifact only. `PatchService` owns hash checks, path checks,
`git apply --check`, application, and safe reverse application.

### Events and logs

Raw process output is written only to attempt log files. The TUI reads it by byte cursor; never send
each log line through EventBus. Incident `events.jsonl` omits per-token events and full tool output.
IPC subscriptions use bounded per-client queues so slow clients cannot block process output.
Daemon trace must summarize, rather than copy, logs, patches, tool I/O, token text, and user text.

### Sessions

Session metadata is restored from disk at daemon startup. Reconnecting a TUI must attach to the same
Job and Incident session. An Incident follow-up must reuse its read-only profile.

### Configuration

Priority is defaults, `~/.cyan/config.toml`, project `.cyan/config.toml`, `.env`, then environment
variables. Relevant prefixes are `CYAN_*`. The optional smoke declaration is:

```toml
[incident.smoke]
argv = ["python", "smoke_train.py", "--steps", "2"]
timeout_s = 300
```

The Agent may not modify this file.

An active smoke verifier persists `smoke-execution.json` with PID and process identity. On daemon
recovery, terminate it only after PID, identity, session leader, and process-group leader all match;
do not reopen follow-up while the recorded process cannot be confirmed stopped.

## Commands

```bash
uv sync
uv run ruff check src tests scripts
uv run mypy src
uv run pytest tests/unit -v
uv run pytest tests/integration -v
uv run pytest tests/ -v
uv run python scripts/gen_protocol_doc.py --check
```

## Code style

Every function must have one concise Chinese comment immediately above its `def` line:

```python
# 发送 JSON-RPC 响应并刷新写缓冲区
async def _send(...) -> None:
    ...
```

Every test function must have two Chinese comment lines immediately above its `def`:

```python
# 功能：验证 publish 后订阅者能收到事件
# 设计：用内联 handler 收集引用，避免引入网络层
async def test_publish_reaches_subscriber() -> None:
    ...
```

Do not add multi-line function docstrings in place of these comments.
