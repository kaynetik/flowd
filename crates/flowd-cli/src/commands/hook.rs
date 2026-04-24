//! `flowd hook <event>` -- in-process Claude Code hook receiver.
//!
//! Replaces the shell scripts under `integrations/claude-code/hooks/` so
//! Claude Code's `settings.json` can call `flowd hook session-start`
//! (etc.) directly. No more `/ABSOLUTE/PATH` placeholders pointing at
//! `bash`, no more `jq`/`uuidgen` requirements.
//!
//! # Stdin payload shape
//!
//! Claude Code writes a single JSON object on stdin. We consume the same
//! fields the shell hooks consumed:
//!
//! | field           | type   | used for                                         |
//! | --------------- | ------ | ------------------------------------------------ |
//! | `session_id`    | string | Claude session id -> flowd session UUID mapping  |
//! | `cwd`           | string | Project slug (basename of `cwd`; falls back to $PWD) |
//! | `tool_name`     | string | `PostToolUse` content/metadata                   |
//! | `tool_input`    | any    | `PostToolUse` metadata (full JSON)               |
//! | `tool_response` | any    | `PostToolUse` metadata (full JSON)               |
//!
//! All fields are optional; missing fields are tolerated. Empty stdin is
//! tolerated too (we still emit a best-effort row).
//!
//! # Environment variables
//!
//! The handler consults no environment variables beyond what
//! [`crate::paths::FlowdPaths::from_env`] already reads (`FLOWD_HOME`,
//! `HOME`). No new vars are introduced.
//!
//! # Error handling
//!
//! Per `.flowd/rules/hook-error-swallowing.yaml`, every error path in
//! this module MUST be logged via `tracing::warn!` / `tracing::error!`
//! and returned as `Ok(())` so the parent Claude Code process is never
//! blocked. Persistence guarantees flow through MCP (`memory_store`),
//! not through hooks -- a failed hook is a telemetry gap, not a
//! correctness bug. Do not "fix" the missing error propagation.

use std::fs;
use std::io::{self, Read};
use std::path::PathBuf;

use anyhow::Result;
use serde::Deserialize;
use serde_json::{Value as JsonValue, json};
use tracing::{error, warn};
use uuid::Uuid;

use crate::cli::HookEvent;
use crate::commands::observe::{ObservationPayload, write_observation};
use crate::paths::FlowdPaths;

/// Truncation budget for the human-readable `content` column. Mirrors
/// the `MAX_CONTENT=2000` the shell hooks used. Full JSON survives in
/// `metadata`, so this only bounds the FTS-indexed column.
const MAX_CONTENT: usize = 2000;

/// Stdin payload shape emitted by Claude Code. All fields optional so
/// a malformed or partial payload still parses into a sensible default.
#[derive(Debug, Default, Deserialize)]
struct HookPayload {
    #[serde(default)]
    session_id: Option<String>,
    #[serde(default)]
    cwd: Option<String>,
    #[serde(default)]
    tool_name: Option<String>,
    #[serde(default)]
    tool_input: Option<JsonValue>,
    #[serde(default)]
    tool_response: Option<JsonValue>,
}

/// Entry point for `flowd hook <event>`. Always returns `Ok(())` --
/// every failure path is logged and swallowed (see module docs and
/// `.flowd/rules/hook-error-swallowing.yaml`).
pub async fn run(event: HookEvent) -> Result<()> {
    if let Err(err) = dispatch(event).await {
        error!(error = ?err, "flowd hook failed (swallowed)");
    }
    Ok(())
}

async fn dispatch(event: HookEvent) -> Result<()> {
    let paths = FlowdPaths::from_env()?;
    let payload = read_payload();
    let project = derive_project(&payload);
    let claude_session = payload.session_id.clone().unwrap_or_default();
    let flowd_session = resolve_flowd_session(&paths, &claude_session);

    match event {
        HookEvent::SessionStart => {
            let content =
                format!("session_start project={project} claude_session={claude_session}");
            let metadata = json!({
                "kind": "session_start",
                "claude_session": claude_session,
            });
            write(&paths, project, flowd_session, content, metadata).await;
        }
        HookEvent::PostToolUse => {
            let tool = payload
                .tool_name
                .clone()
                .filter(|s| !s.is_empty())
                .unwrap_or_else(|| "unknown".into());
            let input = payload.tool_input.clone().unwrap_or_else(|| json!({}));
            let response = payload.tool_response.clone().unwrap_or_else(|| json!({}));
            let input_summary = truncate_to_bytes(
                &serde_json::to_string(&input).unwrap_or_else(|_| "{}".into()),
                MAX_CONTENT,
            );
            let response_summary = truncate_to_bytes(
                &serde_json::to_string(&response).unwrap_or_else(|_| "{}".into()),
                MAX_CONTENT,
            );
            let content =
                format!("tool={tool}\ninput={input_summary}\nresponse={response_summary}\n");
            let metadata = json!({
                "kind": "post_tool_use",
                "tool": tool,
                "claude_session": claude_session,
                "input": input,
                "response": response,
            });
            write(&paths, project, flowd_session, content, metadata).await;
        }
        HookEvent::SessionEnd => {
            let content = format!("session_end project={project} claude_session={claude_session}");
            let metadata = json!({
                "kind": "session_end",
                "claude_session": claude_session,
            });
            write(&paths, project, flowd_session, content, metadata).await;
            forget_flowd_session(&paths, &claude_session);
        }
    }

    Ok(())
}

async fn write(
    paths: &FlowdPaths,
    project: String,
    session_id: Uuid,
    content: String,
    metadata: JsonValue,
) {
    let payload = ObservationPayload {
        paths: paths.clone(),
        project,
        session_id,
        content,
        metadata,
    };
    if let Err(err) = write_observation(payload).await {
        warn!(error = ?err, "flowd hook observation write failed (swallowed)");
    }
}

/// Reads stdin best-effort. Returns a default payload on any failure so
/// the hook still emits *something* -- a gap here would lose a session
/// boundary, and the shell version tolerated it too.
fn read_payload() -> HookPayload {
    let mut buf = String::new();
    if let Err(err) = io::stdin().read_to_string(&mut buf) {
        warn!(error = ?err, "flowd hook failed to read stdin (swallowed)");
        return HookPayload::default();
    }
    let trimmed = buf.trim();
    if trimmed.is_empty() {
        return HookPayload::default();
    }
    match serde_json::from_str::<HookPayload>(trimmed) {
        Ok(p) => p,
        Err(err) => {
            warn!(error = ?err, "flowd hook failed to parse stdin JSON (swallowed)");
            HookPayload::default()
        }
    }
}

/// Mirror the shell `flowd_project`: basename of `.cwd`, falling back
/// to the process's own cwd, falling back to `"unknown"`.
fn derive_project(payload: &HookPayload) -> String {
    let cwd = payload
        .cwd
        .clone()
        .filter(|s| !s.is_empty())
        .or_else(|| {
            std::env::current_dir()
                .ok()
                .map(|p| p.to_string_lossy().into_owned())
        })
        .unwrap_or_default();
    PathBuf::from(&cwd)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".into())
}

/// Return the flowd session UUID bound to this Claude session id,
/// creating (and persisting) a fresh one on first call. When the Claude
/// session id is missing, a one-shot anonymous UUID is returned so the
/// observation still has somewhere to land.
fn resolve_flowd_session(paths: &FlowdPaths, claude_session: &str) -> Uuid {
    if claude_session.is_empty() {
        return Uuid::new_v4();
    }
    let dir = paths.hook_sessions_dir();
    let file = dir.join(claude_session);

    match fs::read_to_string(&file) {
        Ok(contents) => match Uuid::parse_str(contents.trim()) {
            Ok(uuid) => return uuid,
            Err(err) => {
                warn!(
                    error = ?err,
                    path = %file.display(),
                    "stale hook-session mapping; regenerating"
                );
            }
        },
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            warn!(
                error = ?err,
                path = %file.display(),
                "failed to read hook-session mapping; using fresh UUID (swallowed)"
            );
        }
    }

    if let Err(err) = fs::create_dir_all(&dir) {
        warn!(
            error = ?err,
            path = %dir.display(),
            "failed to create hook-sessions dir; using ephemeral UUID (swallowed)"
        );
        return Uuid::new_v4();
    }

    let uuid = Uuid::new_v4();
    if let Err(err) = fs::write(&file, uuid.to_string()) {
        warn!(
            error = ?err,
            path = %file.display(),
            "failed to persist hook-session mapping; using ephemeral UUID (swallowed)"
        );
        return Uuid::new_v4();
    }
    uuid
}

fn forget_flowd_session(paths: &FlowdPaths, claude_session: &str) {
    if claude_session.is_empty() {
        return;
    }
    let file = paths.hook_sessions_dir().join(claude_session);
    match fs::remove_file(&file) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::NotFound => {}
        Err(err) => {
            warn!(
                error = ?err,
                path = %file.display(),
                "failed to remove hook-session mapping (swallowed)"
            );
        }
    }
}

/// Byte-budget truncation that respects UTF-8 char boundaries. Matches
/// the shell `head -c` budget in spirit but never splits a codepoint.
fn truncate_to_bytes(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    s[..end].to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_respects_char_boundary() {
        let s = "aaa\u{1f600}bbb";
        let t = truncate_to_bytes(s, 4);
        assert!(t.len() <= 4);
        assert!(s.starts_with(&t));
    }

    #[test]
    fn truncate_noop_when_under_budget() {
        assert_eq!(truncate_to_bytes("abc", 10), "abc");
    }

    #[test]
    fn derive_project_uses_basename() {
        let p = HookPayload {
            cwd: Some("/home/user/rnd/flowd".into()),
            ..Default::default()
        };
        assert_eq!(derive_project(&p), "flowd");
    }

    #[test]
    fn derive_project_falls_back_to_unknown_on_root() {
        let p = HookPayload {
            cwd: Some("/".into()),
            ..Default::default()
        };
        assert_eq!(derive_project(&p), "unknown");
    }

    // Env mutation is racy across tests; serialize every test that touches
    // FLOWD_HOME so parallel runs don't observe each other's state. Held
    // across `.await` on purpose -- each `#[tokio::test]` has its own
    // runtime, so no deadlock is possible.
    static ENV_LOCK: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

    #[allow(unsafe_code)]
    fn set_flowd_home(path: &std::path::Path) {
        // SAFETY: all callers hold ENV_LOCK for the duration of the mutation.
        unsafe {
            std::env::set_var("FLOWD_HOME", path);
        }
    }

    #[allow(unsafe_code)]
    fn clear_flowd_home() {
        // SAFETY: all callers hold ENV_LOCK for the duration of the mutation.
        unsafe {
            std::env::remove_var("FLOWD_HOME");
        }
    }

    #[test]
    fn session_start_payload_parses() {
        let raw = r#"{"session_id":"abc","cwd":"/home/u/proj"}"#;
        let parsed: HookPayload = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.session_id.as_deref(), Some("abc"));
        assert_eq!(parsed.cwd.as_deref(), Some("/home/u/proj"));
    }

    #[test]
    fn post_tool_use_payload_parses() {
        let raw = r#"{"session_id":"abc","tool_name":"Read","tool_input":{"file_path":"/x"},"tool_response":{"ok":true}}"#;
        let parsed: HookPayload = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.tool_name.as_deref(), Some("Read"));
        assert!(parsed.tool_input.is_some());
        assert!(parsed.tool_response.is_some());
    }

    #[test]
    fn session_end_payload_parses() {
        let raw = r#"{"session_id":"abc"}"#;
        let parsed: HookPayload = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.session_id.as_deref(), Some("abc"));
    }

    #[test]
    fn invalid_json_payload_fails_to_parse() {
        assert!(serde_json::from_str::<HookPayload>("not valid json").is_err());
    }

    #[tokio::test]
    async fn run_session_start_is_ok() {
        let _g = ENV_LOCK.lock().await;
        let tmp = tempfile::TempDir::new().unwrap();
        set_flowd_home(tmp.path());
        let result = run(HookEvent::SessionStart).await;
        clear_flowd_home();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn run_post_tool_use_is_ok() {
        let _g = ENV_LOCK.lock().await;
        let tmp = tempfile::TempDir::new().unwrap();
        set_flowd_home(tmp.path());
        let result = run(HookEvent::PostToolUse).await;
        clear_flowd_home();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn run_session_end_is_ok() {
        let _g = ENV_LOCK.lock().await;
        let tmp = tempfile::TempDir::new().unwrap();
        set_flowd_home(tmp.path());
        let result = run(HookEvent::SessionEnd).await;
        clear_flowd_home();
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn run_swallows_storage_unavailable() {
        let _g = ENV_LOCK.lock().await;
        // /dev/null is a char device; create_dir_all under it cannot succeed.
        set_flowd_home(std::path::Path::new("/dev/null/nope"));
        let result = run(HookEvent::SessionStart).await;
        clear_flowd_home();
        assert!(result.is_ok(), "hook must swallow storage errors");
    }
}
