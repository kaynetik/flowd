//! `flowd observe` -- write a single observation straight to `SQLite`.
//!
//! Used by Claude Code hook scripts that fire outside the MCP session
//! (session-start, post-tool-use, session-end). The observation lands at
//! `MemoryTier::Hot` and is immediately keyword-searchable via FTS5.
//!
//! Vector embedding is intentionally skipped: hooks must be fast and must
//! not require the ONNX runtime. The daemon (`flowd start`) can re-embed
//! hot-tier rows on demand; that backfill is out of scope for this crate.
//!
//! Content is read from stdin to avoid shell-quoting hell and metadata
//! overflow. A `--session` value of `-` or absence means "start a fresh
//! session for this observation".

use std::io::{self, Read};

use anyhow::{Context, Result};
use chrono::Utc;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use flowd_core::memory::MemoryBackend;
use flowd_core::types::{MemoryTier, Observation, Session};
use flowd_storage::sqlite::SqliteBackend;

use crate::output::Style;
use crate::paths::FlowdPaths;

pub async fn run(
    paths: &FlowdPaths,
    _style: Style,
    project: String,
    session: Option<String>,
    metadata: Option<String>,
) -> Result<()> {
    paths.ensure_home()?;

    let mut content = String::new();
    io::stdin()
        .read_to_string(&mut content)
        .context("read observation content from stdin")?;
    let content = content.trim_end_matches(['\n', '\r']).to_owned();
    if content.is_empty() {
        anyhow::bail!("observation content is empty (read nothing from stdin)");
    }

    let metadata: JsonValue = match metadata {
        Some(s) if !s.is_empty() => serde_json::from_str(&s).context("parse --metadata as JSON")?,
        _ => JsonValue::Object(serde_json::Map::new()),
    };

    let db = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    let session_id = resolve_session(&db, &project, session.as_deref()).await?;

    let now = Utc::now();
    let observation = Observation {
        id: Uuid::new_v4(),
        session_id,
        project,
        content,
        tier: MemoryTier::Hot,
        metadata,
        created_at: now,
        updated_at: now,
    };

    db.store(&observation).await.context("store observation")?;

    println!("{}", observation.id);
    eprintln!("session={session_id}");
    Ok(())
}

/// Look up an existing session (by UUID) or create a fresh one.
///
/// Returns the session UUID to attach to the new observation.
async fn resolve_session(db: &SqliteBackend, project: &str, session: Option<&str>) -> Result<Uuid> {
    if let Some(raw) = session {
        if raw != "-" && !raw.is_empty() {
            let id =
                Uuid::parse_str(raw).with_context(|| format!("parse --session {raw:?} as UUID"))?;
            let existing = db
                .list_sessions(Some(project))
                .await
                .context("list sessions")?;
            if !existing.iter().any(|s| s.id == id) {
                let now = Utc::now();
                db.upsert_session(&Session {
                    id,
                    project: project.to_owned(),
                    summary: None,
                    started_at: now,
                    ended_at: None,
                })
                .await
                .context("upsert session for hook observation")?;
            }
            return Ok(id);
        }
    }

    let now = Utc::now();
    let id = Uuid::new_v4();
    db.upsert_session(&Session {
        id,
        project: project.to_owned(),
        summary: None,
        started_at: now,
        ended_at: None,
    })
    .await
    .context("create anonymous session for observation")?;
    Ok(id)
}
