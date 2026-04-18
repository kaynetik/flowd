//! `flowd search` -- keyword search over stored memory (FTS5 only;
//! vector search requires the daemon).
//!
//! We open the `SQLite` file via the same path the daemon uses;
//! `SQLite`'s WAL mode permits concurrent readers, so this is safe even
//! while `flowd start` holds the file.

use anyhow::{Context, Result};

use flowd_core::memory::MemoryBackend;
use flowd_core::types::{SearchQuery, SearchResult, SearchSource};
use flowd_storage::sqlite::SqliteBackend;

use crate::output::{Style, banner, format_search_results};
use crate::paths::FlowdPaths;

pub async fn run(
    paths: &FlowdPaths,
    style: Style,
    query: String,
    project: Option<String>,
    limit: usize,
) -> Result<()> {
    paths.ensure_home()?;
    let db = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    let sq = SearchQuery {
        text: query.clone(),
        project: project.clone(),
        since: None,
        limit,
    };

    let hits = db
        .keyword_search(&sq)
        .await
        .context("fts5 keyword search")?;

    let results: Vec<SearchResult> = hits
        .into_iter()
        .map(|(observation, score)| SearchResult {
            observation,
            rrf_score: score,
            source: SearchSource::Keyword,
        })
        .collect();

    let title = match project.as_deref() {
        Some(p) => format!(
            "search `{query}` in project `{p}` ({} hit{})",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        ),
        None => format!(
            "search `{query}` ({} hit{})",
            results.len(),
            if results.len() == 1 { "" } else { "s" }
        ),
    };
    print!("{}", banner(&title, style));
    print!("{}", format_search_results(&results, style));
    Ok(())
}
