//! `flowd export` -- dump all observations as browsable markdown files.
//!
//! Output layout:
//!
//! ```text
//! <output>/README.md                  -- index, grouped by project
//! <output>/<project>/<session>.md    -- all observations for a session
//! ```
//!
//! Designed for human `grep` / editor search: frontmatter is plain-text,
//! observations are delimited with a `## <timestamp>` heading, and
//! metadata lands in a fenced JSON block.

use std::collections::BTreeMap;
use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::Utc;

use flowd_core::memory::MemoryBackend;
use flowd_core::types::{MemoryTier, Observation, Session};
use flowd_storage::sqlite::SqliteBackend;

use crate::output::{Style, fmt_ts};
use crate::paths::FlowdPaths;

pub async fn run(
    paths: &FlowdPaths,
    style: Style,
    output: PathBuf,
    project: Option<String>,
) -> Result<()> {
    paths.ensure_home()?;
    let db = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    fs::create_dir_all(&output)
        .with_context(|| format!("create output dir: {}", output.display()))?;

    let now = Utc::now();
    let mut observations = Vec::new();
    for tier in [MemoryTier::Hot, MemoryTier::Warm, MemoryTier::Cold] {
        let mut batch = db
            .list_by_tier_and_age(tier, now)
            .await
            .with_context(|| format!("load observations from tier {tier:?}"))?;
        observations.append(&mut batch);
    }
    if let Some(p) = &project {
        observations.retain(|o| &o.project == p);
    }

    let sessions = db.list_sessions(project.as_deref()).await?;
    let session_by_id: BTreeMap<_, _> = sessions.iter().map(|s| (s.id, s)).collect();

    // Group observations by (project, session).
    let mut grouped: BTreeMap<(String, uuid::Uuid), Vec<Observation>> = BTreeMap::new();
    for obs in observations {
        grouped
            .entry((obs.project.clone(), obs.session_id))
            .or_default()
            .push(obs);
    }

    let mut index_entries: BTreeMap<String, Vec<(uuid::Uuid, PathBuf)>> = BTreeMap::new();
    let mut total_sessions = 0usize;
    let mut total_obs = 0usize;

    for ((proj, session_id), mut obs_list) in grouped {
        obs_list.sort_by_key(|o| o.created_at);
        let session = session_by_id.get(&session_id).copied();
        let file_rel = PathBuf::from(sanitize(&proj)).join(format!("{session_id}.md"));
        let file_abs = output.join(&file_rel);
        write_session_file(&file_abs, &proj, session_id, session, &obs_list)?;
        index_entries
            .entry(proj.clone())
            .or_default()
            .push((session_id, file_rel));
        total_sessions += 1;
        total_obs += obs_list.len();
    }

    write_index(&output.join("README.md"), &index_entries)?;

    eprintln!(
        "{} wrote {} session file(s), {} observation(s) -> {}",
        style.green("exported"),
        total_sessions,
        total_obs,
        output.display(),
    );
    Ok(())
}

fn write_session_file(
    path: &Path,
    project: &str,
    session_id: uuid::Uuid,
    session: Option<&Session>,
    observations: &[Observation],
) -> Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).with_context(|| format!("create {}", parent.display()))?;
    }
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;

    writeln!(f, "# {project} -- session {session_id}")?;
    if let Some(s) = session {
        writeln!(
            f,
            "*started {} | ended {}*",
            fmt_ts(s.started_at),
            s.ended_at.map_or_else(|| "(open)".to_owned(), fmt_ts)
        )?;
        if let Some(summary) = &s.summary {
            writeln!(f, "\n> {summary}")?;
        }
    }
    writeln!(f)?;

    for obs in observations {
        writeln!(f, "## {}", fmt_ts(obs.created_at))?;
        writeln!(f, "- **id:** `{}`", obs.id)?;
        writeln!(f, "- **tier:** {:?}", obs.tier)?;
        writeln!(f)?;
        writeln!(f, "{}", obs.content.trim())?;
        if !obs.metadata.is_null() {
            writeln!(f, "\n```json")?;
            writeln!(f, "{}", serde_json::to_string_pretty(&obs.metadata)?)?;
            writeln!(f, "```")?;
        }
        writeln!(f)?;
    }
    Ok(())
}

fn write_index(path: &Path, entries: &BTreeMap<String, Vec<(uuid::Uuid, PathBuf)>>) -> Result<()> {
    let mut f = fs::File::create(path).with_context(|| format!("create {}", path.display()))?;

    writeln!(f, "# flowd memory export")?;
    writeln!(f, "\n*generated {}*", fmt_ts(Utc::now()))?;
    writeln!(f)?;

    if entries.is_empty() {
        writeln!(f, "_no sessions exported._")?;
        return Ok(());
    }

    for (project, sessions) in entries {
        writeln!(f, "## {project}")?;
        writeln!(f)?;
        for (session_id, rel_path) in sessions {
            writeln!(f, "- [{session_id}]({})", rel_path.display())?;
        }
        writeln!(f)?;
    }
    Ok(())
}

/// Sanitise a project name for use as a directory name. Keeps letters,
/// digits, underscores, dots, dashes, and slashes (the last so nested
/// project naming like `team/repo` is preserved).
fn sanitize(raw: &str) -> String {
    raw.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '_' | '-' | '.' | '/' => c,
            _ => '-',
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_replaces_unsafe_chars() {
        assert_eq!(sanitize("team/repo"), "team/repo");
        assert_eq!(sanitize("my project"), "my-project");
        assert_eq!(sanitize("weird*name"), "weird-name");
    }
}
