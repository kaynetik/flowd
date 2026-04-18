# flowd integrations

Wires the `flowd` daemon into agent environments.

## Prerequisites

- `flowd` binary on `$PATH` (`cargo install --path crates/flowd-cli` or symlink the release build)
- Qdrant reachable on `http://localhost:6334` (default) for vector search
- For hooks: `bash 4+`, `jq`, `uuidgen`

## Claude Code

Two pieces:

1. `mcpServers.flowd` registers `flowd start` as a stdio MCP server. Claude Code starts the daemon on session open and speaks JSON-RPC over stdio.
2. `hooks.*` invoke shell scripts on session-start, post-tool-use, and session-end. Each hook calls `flowd observe` to persist a structured row keyed to the Claude session.

### Install

1. Replace `/ABSOLUTE/PATH` in `claude-code/settings.json` with the absolute path to this repo.
2. Merge that JSON into `~/.claude/settings.json` (preserve your existing keys; deep-merge `mcpServers` and `hooks`).
3. Restart Claude Code.

### What gets recorded

| Event         | Content                               | Metadata keys                                   |
| ------------- | ------------------------------------- | ----------------------------------------------- |
| `SessionStart` | `session_start project=... claude_session=...` | `kind`, `claude_session`                        |
| `PostToolUse`  | `tool=...`, truncated input/response  | `kind`, `tool`, `claude_session`, `input`, `response` |
| `SessionEnd`   | `session_end project=... claude_session=...`   | `kind`, `claude_session`                        |

The Claude-to-flowd session mapping lives at `$FLOWD_HOME/hook-sessions/<claude_session_id>`. `SessionEnd` removes it.

## Cursor

Copy `cursor/mcp.json` to either:

- `~/.cursor/mcp.json` — global, all projects
- `<repo>/.cursor/mcp.json` — per-project

Restart Cursor. The agent's tools panel should list: `memory_store`, `memory_search`, `memory_context`, `plan_create`, `plan_confirm`, `plan_status`, `rules_check`, `rules_list`.

Cursor does not expose session hooks, so observation recording happens exclusively through `memory_store` calls the agent makes during a task.

## Verifying

After setup:

```bash
flowd status               # daemon state + row counts
flowd history              # sessions, newest first
flowd search "keyword"     # FTS5 over hot/warm tiers
flowd export -o /tmp/dump  # browsable markdown dump
```

A healthy first-use path looks like: `flowd status` shows zero rows, you open a Claude/Cursor session, after one tool invocation `flowd history` shows exactly one session and `flowd search` finds the event content.

## Failure modes

- **Hook silently no-ops.** The hook scripts swallow errors by design (hooks must not block the parent). Run a script directly with a sample payload to surface errors: `echo '{"session_id":"s1","cwd":"/tmp/foo"}' | integrations/claude-code/hooks/session-start.sh`.
- **Daemon fails to start.** Run `flowd start` in a plain terminal; stderr has tracing output. Common cause: Qdrant unreachable — pass `--qdrant-url` or start Qdrant first.
- **Stale PID.** `flowd stop` reports `no PID file`; a crashed daemon left a stale entry. Remove `$FLOWD_HOME/flowd.pid` and retry.
