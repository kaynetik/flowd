//! Regression tests for the compactor's hot-to-warm summarize step
//! against the real `SqliteBackend`.
//!
//! Bug observed in production logs (every compaction tick, identical
//! across many sessions, looping forever):
//!
//! ```text
//! WARN flowd_core::memory::compactor: failed to summarize session;
//!      leaving observations in hot tier
//!      session_id=... error=storage error: SQL logic error
//! ```
//!
//! Two stacked defects produce that surface:
//!
//! 1. **Broken FTS5 maintenance trigger (primary).** The
//!    `observations_ad` AFTER-DELETE trigger from MIGRATION_001 used
//!    the FTS5 `'delete'` command form, which is only valid for
//!    contentless / external-content tables. `observations_fts` is a
//!    regular FTS5 table, so SQLite returns `SQLITE_ERROR` ("SQL
//!    logic error") for every DELETE on `observations`. The compactor
//!    deletes originals after writing each summary, so this fired on
//!    every session every tick. Fixed by MIGRATION_008 (regular
//!    `DELETE FROM observations_fts WHERE rowid = old.rowid`).
//!
//! 2. **Latent FK violation (secondary).** When `compact_once` cannot
//!    find a `sessions` row for an observation it falls back to a
//!    synthesized `placeholder_session`. The summary insert then
//!    violates `observations.session_id REFERENCES sessions(id)`.
//!    Fixed by upserting the session in `summarize_and_replace`
//!    before storing the summary (`ON CONFLICT(id) DO UPDATE`, so
//!    near-no-op when the row already exists).
//!
//! The two tests below pin both fixes from the storage side: a
//! correctly-seeded session must compact, and an orphaned observation
//! must also compact instead of looping forever.

use std::sync::Arc;

use chrono::{Duration as ChronoDuration, Utc};
use flowd_core::error::Result;
use flowd_core::memory::compactor::{ActivityMonitor, Compactor, CompactorConfig};
use flowd_core::memory::tier::TieringPolicy;
use flowd_core::memory::{EmbeddingProvider, MemoryBackend, Summarizer, VectorIndex};
use flowd_core::types::{Embedding, MemoryTier, Observation, Session};
use flowd_storage::sqlite::SqliteBackend;
use serde_json::json;
use tempfile::TempDir;
use uuid::Uuid;

const PROJECT: &str = "regression";

// ---- Test stubs ------------------------------------------------------------

struct ConstSummarizer;

impl Summarizer for ConstSummarizer {
    async fn summarize(&self, _session: &Session, observations: &[Observation]) -> Result<String> {
        Ok(format!("summary of {} observations", observations.len()))
    }
}

#[derive(Default)]
struct NoopVector;

impl VectorIndex for NoopVector {
    async fn upsert(&self, _embedding: &Embedding) -> Result<()> {
        Ok(())
    }

    async fn search(
        &self,
        _query_vector: &[f32],
        _limit: usize,
        _project_filter: Option<&str>,
    ) -> Result<Vec<(Uuid, f64)>> {
        Ok(Vec::new())
    }

    async fn delete(&self, _observation_id: Uuid) -> Result<()> {
        Ok(())
    }
}

#[derive(Default)]
struct ZeroEmbedder;

impl EmbeddingProvider for ZeroEmbedder {
    fn embed(&self, _text: &str) -> Result<Vec<f32>> {
        Ok(vec![0.0; 4])
    }

    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
        Ok(texts.iter().map(|_| vec![0.0; 4]).collect())
    }

    fn dimensions(&self) -> usize {
        4
    }
}

// ---- Helpers ---------------------------------------------------------------

fn open_backend() -> (TempDir, Arc<SqliteBackend>) {
    let tmp = tempfile::tempdir().expect("tempdir");
    let backend = SqliteBackend::open(&tmp.path().join("flowd.db")).expect("open sqlite");
    (tmp, Arc::new(backend))
}

/// Plant a hot-tier observation directly on the backend, bypassing
/// `MemoryService` so we don't accidentally upsert the session for the
/// caller. Used to model legacy data and to check that the session
/// upsert lives in the compactor, not the test fixture.
async fn plant_hot_observation(
    backend: &SqliteBackend,
    session: &Session,
    age: ChronoDuration,
) -> Uuid {
    backend
        .upsert_session(session)
        .await
        .expect("upsert session");

    let now = Utc::now();
    let obs = Observation {
        id: Uuid::new_v4(),
        session_id: session.id,
        project: session.project.clone(),
        content: "an old hot observation".into(),
        tier: MemoryTier::Hot,
        metadata: json!({}),
        created_at: now - age,
        updated_at: now - age,
    };
    backend.store(&obs).await.expect("store observation");
    obs.id
}

fn make_compactor(
    backend: Arc<SqliteBackend>,
) -> Compactor<SqliteBackend, NoopVector, ZeroEmbedder, ConstSummarizer> {
    let policy =
        TieringPolicy::new(ChronoDuration::hours(1), ChronoDuration::days(7)).expect("policy");
    let config = CompactorConfig {
        check_interval: std::time::Duration::from_secs(60),
        idle_threshold: std::time::Duration::from_secs(0),
        batch_limit: 256,
    };
    Compactor::new(
        backend,
        Arc::new(NoopVector),
        Arc::new(ZeroEmbedder),
        Arc::new(ConstSummarizer),
        policy,
        ActivityMonitor::new(),
        config,
    )
}

// ---- Tests ----------------------------------------------------------------

/// Happy-path control: a hot observation whose session row exists must
/// compact cleanly. If this fails, the test fixture is broken, not the
/// compactor.
#[tokio::test]
async fn compaction_succeeds_when_session_row_present() {
    let (_tmp, backend) = open_backend();

    let session = Session {
        id: Uuid::new_v4(),
        project: PROJECT.into(),
        summary: None,
        started_at: Utc::now() - ChronoDuration::hours(2),
        ended_at: None,
    };
    let obs_id = plant_hot_observation(&backend, &session, ChronoDuration::hours(2)).await;

    let compactor = make_compactor(Arc::clone(&backend));
    let stats = compactor.compact_once().await.expect("compact_once ok");

    assert_eq!(stats.sessions_summarized, 1);
    assert_eq!(stats.hot_replaced, 1);

    let gone = backend.get(obs_id).await.expect("get original");
    assert!(gone.is_none(), "original should be deleted");
}

/// Regression test for the production-log bug. The hot observation
/// belongs to a session whose row is missing from `sessions` (modelled
/// by inserting via raw SQL with FKs temporarily off, simulating
/// orphaned data from older binaries / manual imports / a deleted
/// session row). The compactor must not bail with a generic SQL error
/// and must not loop forever leaving the same observations in hot tier.
#[tokio::test]
async fn compaction_recovers_when_session_row_missing() {
    let (tmp, backend) = open_backend();

    // Orphan the row by writing it through a side-channel connection
    // with foreign_keys=OFF. This faithfully reproduces the production
    // condition without depending on any specific historical bug path.
    let orphan_session_id = Uuid::new_v4();
    let obs_id = Uuid::new_v4();
    let now = Utc::now();
    let stale = now - ChronoDuration::hours(2);
    {
        let raw = rusqlite::Connection::open(tmp.path().join("flowd.db")).expect("raw open");
        raw.execute_batch("PRAGMA foreign_keys=OFF;")
            .expect("fk off");
        raw.execute(
            "INSERT INTO observations (id, session_id, project, content, tier, metadata, \
             created_at, updated_at) VALUES (?1, ?2, ?3, ?4, 'hot', '{}', ?5, ?5)",
            rusqlite::params![
                obs_id.to_string(),
                orphan_session_id.to_string(),
                PROJECT,
                "orphaned hot observation",
                stale.to_rfc3339(),
            ],
        )
        .expect("orphan insert");
    }

    // Sanity-check: no session row for this id.
    let sessions = backend
        .list_sessions(Some(PROJECT))
        .await
        .expect("list sessions");
    assert!(
        sessions.iter().all(|s| s.id != orphan_session_id),
        "fixture must leave session orphaned"
    );

    let compactor = make_compactor(Arc::clone(&backend));
    let stats = compactor
        .compact_once()
        .await
        .expect("compact_once must not fail wholesale");

    // The fix is observable as: the orphaned observation was rolled
    // into a summary AND a sessions row exists for it now.
    assert_eq!(
        stats.sessions_summarized, 1,
        "expected exactly one session to be summarized"
    );
    assert_eq!(
        stats.hot_replaced, 1,
        "the hot observation must be replaced"
    );

    let gone = backend.get(obs_id).await.expect("get original");
    assert!(
        gone.is_none(),
        "original hot observation must have been deleted by compaction"
    );

    let sessions_after = backend
        .list_sessions(Some(PROJECT))
        .await
        .expect("list sessions after");
    assert!(
        sessions_after.iter().any(|s| s.id == orphan_session_id),
        "compactor must upsert the missing session row before writing the summary"
    );
}
