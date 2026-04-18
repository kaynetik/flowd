//! `flowd history` -- list sessions, optionally filtered by project /
//! since timestamp.

use anyhow::{Context, Result};
use chrono::{DateTime, Utc};

use flowd_core::memory::MemoryBackend;
use flowd_storage::sqlite::SqliteBackend;

use crate::output::{Style, banner, format_sessions};
use crate::paths::FlowdPaths;

pub async fn run(
    paths: &FlowdPaths,
    style: Style,
    project: Option<String>,
    since: Option<String>,
    limit: usize,
) -> Result<()> {
    paths.ensure_home()?;
    let db = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    let since_ts = since
        .as_deref()
        .map(parse_since)
        .transpose()
        .context("parse --since timestamp (expected RFC3339, e.g. 2026-04-01T00:00:00Z)")?;

    let mut sessions = db
        .list_sessions(project.as_deref())
        .await
        .context("list sessions from sqlite")?;

    if let Some(cutoff) = since_ts {
        sessions.retain(|s| s.started_at >= cutoff);
    }
    sessions.sort_by(|a, b| b.started_at.cmp(&a.started_at));
    sessions.truncate(limit);

    let title = match project.as_deref() {
        Some(p) => format!(
            "history for `{p}` ({} session{})",
            sessions.len(),
            if sessions.len() == 1 { "" } else { "s" }
        ),
        None => format!(
            "history ({} session{})",
            sessions.len(),
            if sessions.len() == 1 { "" } else { "s" }
        ),
    };
    print!("{}", banner(&title, style));
    print!("{}", format_sessions(&sessions, style));
    Ok(())
}

fn parse_since(s: &str) -> Result<DateTime<Utc>> {
    Ok(DateTime::parse_from_rfc3339(s)?.with_timezone(&Utc))
}
