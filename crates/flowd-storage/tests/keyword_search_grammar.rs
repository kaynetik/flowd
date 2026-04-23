//! Regression tests for the grammar contract between the auto-context
//! synthesizer (`flowd_core::memory::context::AutoContextQuery`) and the
//! FTS5 `MATCH` query expected by `SqliteBackend::keyword_search`.
//!
//! Bug: `AutoContextQuery::to_search_query()` previously joined caller
//! fragments (hint, `file_path`, project) with spaces and fed the result to
//! FTS5 verbatim. Any fragment containing characters outside the FTS5
//! query grammar -- the most common case being a path separator `/` in
//! `file_path` -- caused `SQLite` to return
//! `fts5: syntax error near "<char>"`, which surfaced to MCP callers as
//! `storage error: fts5: syntax error near "/"`.
//!
//! The contract these tests pin: **whatever string `AutoContextQuery`
//! produces, the backend must accept as a valid FTS5 MATCH expression.**
//! The tests assert at the boundary that broke (storage), not at the
//! synthesizer's internal shape, so they survive any reasonable
//! implementation choice (phrase escaping, tokenize-and-rebuild, etc).

use chrono::Utc;
use flowd_core::memory::MemoryBackend;
use flowd_core::memory::context::AutoContextQuery;
use flowd_core::types::{MemoryTier, Observation, Session};
use flowd_storage::sqlite::SqliteBackend;
use serde_json::json;
use tempfile::TempDir;
use uuid::Uuid;

fn open_backend() -> (TempDir, SqliteBackend) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let backend = SqliteBackend::open(&tmp.path().join("flowd.db")).expect("open sqlite");
    (tmp, backend)
}

async fn seed(backend: &SqliteBackend, project: &str, content: &str) -> Uuid {
    let session = Session {
        id: Uuid::new_v4(),
        project: project.into(),
        summary: None,
        started_at: Utc::now(),
        ended_at: None,
    };
    backend
        .upsert_session(&session)
        .await
        .expect("upsert session");
    let obs = Observation {
        id: Uuid::new_v4(),
        session_id: session.id,
        project: project.into(),
        content: content.into(),
        tier: MemoryTier::Hot,
        metadata: json!({}),
        created_at: Utc::now(),
        updated_at: Utc::now(),
    };
    backend.store(&obs).await.expect("store observation");
    obs.id
}

/// Reproduces the exact failing call shape from the dog-fooding session:
/// `memory_context(project="flowd", file_path="rnd_docs/notes.md", ...)`.
///
/// The synthesized FTS5 MATCH expression must (a) be accepted by `SQLite`
/// (the bug under repair), and (b) preserve the historical
/// implicit-AND token-matching semantics of `keyword_search`, so that an
/// observation containing the same tokens as the synthesized query is
/// recovered. Planted content is crafted to contain every token the
/// synthesizer emits for these inputs.
#[tokio::test]
async fn auto_context_with_path_separator_in_file_path_is_searchable() {
    let (_tmp, backend) = open_backend();
    let planted = seed(
        &backend,
        "flowd",
        "rnd_docs/notes.md belongs to project flowd",
    )
    .await;

    let query = AutoContextQuery::new("flowd")
        .with_file("rnd_docs/notes.md")
        .to_search_query();

    let hits = backend
        .keyword_search(&query)
        .await
        .expect("synthesized auto-context query must be a valid FTS5 MATCH expression");

    let ids: Vec<Uuid> = hits.iter().map(|(o, _)| o.id).collect();
    assert!(
        ids.contains(&planted),
        "planted observation should be recoverable via the synthesized \
         auto-context query, got ids: {ids:?}"
    );
}

/// Broader hostile-input matrix. Any character outside the FTS5 bareword
/// set (`A-Za-z0-9_` plus the operator alphabet `AND OR NOT NEAR + - : ( ) " * ^`)
/// trips the query parser when used unquoted. The synthesizer must defang
/// all of them.
#[tokio::test]
async fn auto_context_with_punctuation_heavy_inputs_does_not_break_fts5() {
    let (_tmp, backend) = open_backend();
    let _ = seed(
        &backend,
        "flowd",
        "filler observation so the index is non-empty",
    )
    .await;

    let cases: &[(&str, Option<&str>)] = &[
        ("rnd_docs/notes.md", None),
        ("src/main.rs:42", Some("found bug at line 42")),
        ("crates/flowd-mcp/src/handlers.rs", Some("(see also)")),
        ("README.md", Some("why did we pick \"JWT\"?")),
        ("a&b||c", Some("symbols: ! ? @ # $ % ^ & * ( )")),
        ("path with spaces/file.rs", Some("2 + 2 = 4")),
    ];

    for (file_path, hint) in cases {
        let mut q = AutoContextQuery::new("flowd").with_file(*file_path);
        if let Some(h) = hint {
            q = q.with_hint(*h);
        }
        let search = q.to_search_query();
        backend.keyword_search(&search).await.unwrap_or_else(|e| {
            panic!(
                "synthesized FTS5 MATCH must accept hostile inputs; \
                 file_path={file_path:?} hint={hint:?} -> error: {e}"
            )
        });
    }
}
