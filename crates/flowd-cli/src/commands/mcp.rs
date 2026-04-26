//! `flowd mcp` -- per-client stdio bridge to the central daemon.
//!
//! Cursor and Claude expect an MCP server on stdio. The real flowd daemon now
//! owns state behind a local Unix socket; this command was originally a tiny
//! byte-pump but now does one structural transform on the way to the daemon:
//! stamps the **invoking workspace** onto outbound `plan_create` requests.
//!
//! Why here, and not in the daemon: the daemon's `current_dir()` reflects
//! wherever `flowd start` was launched -- typically nothing to do with the
//! Cursor/Claude window the operator just clicked "new plan" in. The proxy,
//! by contrast, is spawned *by the MCP client*, in the workspace root the
//! client owns. By the time the request hits the daemon we've already lost
//! that signal, so the proxy injects it now.
//!
//! The injection is opt-out, not mandatory: we only set `project_root` when
//! the workspace looks like a git checkout (env override or cwd) and the
//! caller didn't already supply one. The daemon's resolver does the final
//! canonicalisation and validation.

use anyhow::{Context, Result};
use serde_json::Value;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader, copy};
use tokio::net::UnixStream;

use flowd_core::orchestration::{FLOWD_WORKSPACE_ROOT_ENV, resolve_workspace_root};

use crate::paths::FlowdPaths;

pub async fn run(paths: &FlowdPaths) -> Result<()> {
    let socket = paths.socket_file();
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connect to flowd daemon socket at {}", socket.display()))?;
    let (mut daemon_read, mut daemon_write) = stream.into_split();
    let mut stdout = tokio::io::stdout();

    // Best-effort workspace probe at startup. Failure is fine: the daemon
    // will run its own resolver against `FLOWD_WORKSPACE_ROOT` and its cwd,
    // and surface a structured error if no source resolves.
    let workspace_hint = detect_workspace_root();
    if let Some(root) = workspace_hint.as_deref() {
        tracing::debug!(workspace = %root, "flowd mcp proxy injecting workspace_root");
    } else {
        tracing::debug!(
            "flowd mcp proxy could not infer workspace_root; daemon will fall back to its own resolver"
        );
    }

    let to_daemon = async {
        let stdin = tokio::io::stdin();
        let mut lines = BufReader::new(stdin).lines();
        while let Some(line) = lines.next_line().await? {
            let payload = if line.trim().is_empty() {
                line
            } else {
                inject_workspace_root_in_line(&line, workspace_hint.as_deref())
            };
            daemon_write.write_all(payload.as_bytes()).await?;
            daemon_write.write_all(b"\n").await?;
        }
        daemon_write.shutdown().await
    };
    let to_client = async {
        copy(&mut daemon_read, &mut stdout).await?;
        stdout.shutdown().await
    };

    tokio::try_join!(to_daemon, to_client)?;
    Ok(())
}

/// Inspect a single JSON-RPC line and, if it's a `tools/call` for
/// `plan_create` without an explicit `project_root`, splice the proxy's
/// inferred workspace into the arguments. Lines we don't recognise (parse
/// failures, other tools, already-set `project_root`) pass through verbatim
/// so a malformed message reaches the daemon's parse error path unchanged
/// rather than being silently dropped here.
fn inject_workspace_root_in_line(line: &str, workspace_hint: Option<&str>) -> String {
    let Some(hint) = workspace_hint else {
        return line.to_owned();
    };
    let Ok(mut value) = serde_json::from_str::<Value>(line) else {
        return line.to_owned();
    };
    if !try_inject_workspace_root(&mut value, hint) {
        return line.to_owned();
    }
    serde_json::to_string(&value).unwrap_or_else(|_| line.to_owned())
}

/// Returns true when the request was rewritten. Pulled out so unit tests
/// can exercise every branch without parsing JSON twice.
fn try_inject_workspace_root(value: &mut Value, workspace: &str) -> bool {
    let Some(method) = value.get("method").and_then(Value::as_str) else {
        return false;
    };
    if method != "tools/call" {
        return false;
    }
    let Some(params) = value.get_mut("params").and_then(Value::as_object_mut) else {
        return false;
    };
    let name = params.get("name").and_then(Value::as_str).unwrap_or("");
    if name != "plan_create" {
        return false;
    }
    // `arguments` defaults to `{}` if the client omitted it; tools/call is
    // valid that way per the spec, and we still want to inject so the
    // daemon sees the workspace.
    let entry = params
        .entry("arguments".to_owned())
        .or_insert_with(|| Value::Object(serde_json::Map::new()));
    let Some(args) = entry.as_object_mut() else {
        return false;
    };
    if let Some(existing) = args.get("project_root") {
        if existing.as_str().is_some_and(|s| !s.trim().is_empty()) {
            return false;
        }
    }
    args.insert(
        "project_root".to_owned(),
        Value::String(workspace.to_owned()),
    );
    true
}

/// Resolve the workspace root the proxy will forward to the daemon.
///
/// Asks [`resolve_workspace_root`] with no client hint so the resolver
/// walks the proxy process's `FLOWD_WORKSPACE_ROOT` env first and then
/// `current_dir()` -- which, because the proxy is spawned by the MCP
/// client, points at the workspace root the client owns. The git-repo
/// gate inside the resolver makes this an opt-in signal: a developer
/// running the proxy outside any repository keeps the previous "leave
/// it to the daemon" behaviour.
fn detect_workspace_root() -> Option<String> {
    match resolve_workspace_root(None) {
        Ok((path, _src)) => Some(path),
        Err(err) => {
            tracing::debug!(error = %err, env = FLOWD_WORKSPACE_ROOT_ENV,
                "flowd mcp proxy: no workspace_root resolved");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn injects_project_root_into_plan_create_arguments() {
        let line = json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "plan_create",
                "arguments": { "project": "rnd", "prose": "do a thing" }
            }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["params"]["arguments"]["project_root"],
            "/repos/widget"
        );
        // Existing fields survive untouched.
        assert_eq!(parsed["params"]["arguments"]["project"], "rnd");
        assert_eq!(parsed["params"]["arguments"]["prose"], "do a thing");
    }

    #[test]
    fn preserves_caller_supplied_project_root() {
        // If the MCP client (or the operator hand-crafting the call)
        // already filled in `project_root`, we MUST NOT overwrite it --
        // the explicit value is more trustworthy than the proxy's CWD.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "plan_create",
                "arguments": {
                    "project": "rnd",
                    "prose": "do",
                    "project_root": "/repos/explicit"
                }
            }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["params"]["arguments"]["project_root"],
            "/repos/explicit"
        );
    }

    #[test]
    fn replaces_blank_project_root() {
        // Empty / whitespace strings are equivalent to "unset" -- the
        // daemon resolver would also reject them, so we treat the
        // proxy hint as more useful.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "plan_create",
                "arguments": { "project": "rnd", "prose": "do", "project_root": "   " }
            }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["params"]["arguments"]["project_root"],
            "/repos/widget"
        );
    }

    #[test]
    fn ignores_other_tools() {
        // memory_store / rules_check / ... are unaffected. The proxy
        // only targets plan_create because that's the one tool that
        // persists project_root.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "memory_store",
                "arguments": { "project": "rnd", "session_id": "...", "content": "hi" }
            }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert!(parsed["params"]["arguments"].get("project_root").is_none());
    }

    #[test]
    fn ignores_non_tools_call_methods() {
        // initialize / tools/list shouldn't gain a workspace root --
        // they don't consume it and we want the request to round-trip
        // exactly so initialize stays compatible with stricter clients.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 5,
            "method": "initialize",
            "params": { "protocolVersion": "2024-11-05" }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        // Round-trip: parse and re-serialise gives the same shape.
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(parsed["method"], "initialize");
        assert!(parsed["params"].get("project_root").is_none());
    }

    #[test]
    fn passes_malformed_json_through_unchanged() {
        // Malformed lines are forwarded verbatim so the daemon sees
        // exactly what the client sent and can reply with the canonical
        // -32700 parse error.
        let line = "{not really json}";
        let out = inject_workspace_root_in_line(line, Some("/repos/widget"));
        assert_eq!(out, line);
    }

    #[test]
    fn no_workspace_hint_is_a_noop() {
        // Without a hint the proxy must not invent one; the daemon will
        // run its own resolver and surface a structured error if every
        // signal misses.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "plan_create",
                "arguments": { "project": "rnd", "prose": "do" }
            }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, None);
        assert_eq!(out, line);
    }

    #[test]
    fn handles_missing_arguments_field() {
        // tools/call with `name` but no `arguments` is a legal MCP
        // shape (the daemon defaults arguments to `{}`); the injection
        // should still happen so plan_create receives the workspace.
        let line = json!({
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": { "name": "plan_create" }
        })
        .to_string();
        let out = inject_workspace_root_in_line(&line, Some("/repos/widget"));
        let parsed: Value = serde_json::from_str(&out).unwrap();
        assert_eq!(
            parsed["params"]["arguments"]["project_root"],
            "/repos/widget"
        );
    }
}
