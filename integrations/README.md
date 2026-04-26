# flowd integrations

Wires the `flowd` daemon into agent environments.

## What changed

Hooks are no longer shell scripts. `flowd hook session-start`, `flowd hook post-tool-use`, and `flowd hook session-end` are first-class subcommands of the `flowd` binary -- they read the same JSON payload Claude Code already writes on stdin and call the same observation-write path that `flowd observe` uses. The previous scripts under `claude-code/hooks/` (`session-start.sh`, `post-tool-use.sh`, `session-end.sh`, `_lib.sh`) are gone; Claude Code's `settings.json` now invokes the subcommands directly.

Cursor install is now automated via `flowd init cursor --global` (or `--project <path>`), which deep-merges the canonical MCP stanza into the target `mcp.json`, pins `command` to the running `flowd` binary, and writes atomically. Re-runs against an already-correct file are no-ops.

`cursor/mcp.json` and `claude-code/settings.json` in this directory are kept as reference snapshots -- use them for manual-merge fallback or to review what `flowd init` produces.

## Prerequisites

- `flowd` binary on `$PATH` (`cargo install --path crates/flowd-cli` or symlink the release build)
- Qdrant reachable on `http://localhost:6334` (default) for vector search

No `bash`, `jq`, or `uuidgen` requirement -- the hook subcommands do their own JSON parsing, UUID generation, and mapping persistence.

## Claude Code

Two pieces:

1. `mcpServers.flowd` registers `flowd start` as a stdio MCP server. Claude Code starts the daemon on session open and speaks JSON-RPC over stdio.
2. `hooks.*` invoke `flowd hook <event>` on session-start, post-tool-use, and session-end. Each subcommand records a structured row keyed to the Claude session.

### Install

1. Merge `claude-code/settings.json` into `~/.claude/settings.json` (preserve your existing keys; deep-merge `mcpServers` and `hooks`). No path substitution required.
2. Restart Claude Code.

### What gets recorded

| Event         | Content                               | Metadata keys                                   |
| ------------- | ------------------------------------- | ----------------------------------------------- |
| `SessionStart` | `session_start project=... claude_session=...` | `kind`, `claude_session`                        |
| `PostToolUse`  | `tool=...`, truncated input/response  | `kind`, `tool`, `claude_session`, `input`, `response` |
| `SessionEnd`   | `session_end project=... claude_session=...`   | `kind`, `claude_session`                        |

The Claude-to-flowd session mapping lives at `$FLOWD_HOME/hook-sessions/<claude_session_id>`. `SessionEnd` removes it.

## Cursor

Run the init command:

```bash
flowd init cursor --global                 # ~/.cursor/mcp.json
flowd init cursor --project /path/to/repo  # <repo>/.cursor/mcp.json
```

Or, if you prefer to hand-edit, copy the `cursor/mcp.json` reference snapshot to either `~/.cursor/mcp.json` or `<repo>/.cursor/mcp.json` (ensure `flowd` is on `$PATH`).

Restart Cursor. The agent's tools panel should list: `memory_store`, `memory_search`, `memory_context`, `plan_create`, `plan_confirm`, `plan_status`, `rules_check`, `rules_list`.

Cursor does not expose session hooks, so observation recording happens exclusively through `memory_store` calls the agent makes during a task.

## Workspace inference

Both integrations rely on the same mechanism to tell the daemon *which* repository a `plan_create` call is targeting:

- The IDE spawns `flowd mcp` (the stdio-to-socket proxy) inside the workspace it has open. The proxy probes its own cwd, and if it sits inside a git checkout, splices that path into outbound `plan_create` requests as `project_root`. The daemon -- whose cwd is wherever `flowd start` was launched -- then canonicalises and verifies it.
- If the proxy can't infer a workspace (cwd is not in a git repo, or `flowd mcp` is launched outside the IDE), it forwards the request unchanged and the daemon falls back to `FLOWD_WORKSPACE_ROOT` and finally its own cwd.
- Set `FLOWD_WORKSPACE_ROOT` on either the proxy process *or* the daemon when neither cwd is the right answer (e.g. running `flowd start` under systemd, or wiring a non-IDE MCP client).

`project` (the namespace label used by rules and `flowd history`) is *not* the same as `project_root` (the absolute filesystem path agents run against). The README's [Workspace and project scoping](../README.md#workspace-and-project-scoping) section is the canonical reference for resolution order, the wrong-repo guard, and the recovery rule for plans rooted in the wrong repository.

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

- **Hook silently no-ops.** `flowd hook <event>` swallows every error by design (hooks must not block the parent -- see `.flowd/rules/hook-error-swallowing.yaml`). Exercise it directly with a sample payload to surface errors in your own terminal: `echo '{"session_id":"s1","cwd":"/tmp/foo"}' | flowd hook session-start` and watch `RUST_LOG=flowd=warn` output on stderr.
- **Daemon fails to start.** Run `flowd start` in a plain terminal; stderr has tracing output. Common cause: Qdrant unreachable -- pass `--qdrant-url` or start Qdrant first.
- **Stale PID.** `flowd stop` reports `no PID file`; a crashed daemon left a stale entry. Remove `$FLOWD_HOME/flowd.pid` and retry.
