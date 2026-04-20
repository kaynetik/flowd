//! `SQLite` `MemoryBackend` implementation.
//!
//! All database I/O runs on a blocking thread via `tokio::task::spawn_blocking`
//! so the async runtime is never stalled by `SQLite` file locks.

use chrono::{DateTime, Utc};
use flowd_core::error::{FlowdError, Result};
use flowd_core::memory::MemoryBackend;
use flowd_core::types::{MemoryTier, Observation, SearchQuery, Session};
use rusqlite::{Connection, OptionalExtension};
use std::path::Path;
use std::sync::{Arc, Mutex};
use uuid::Uuid;

use crate::plan_event_store::SqlitePlanEventStore;
use crate::plan_store::SqlitePlanStore;

#[derive(Clone)]
pub struct SqliteBackend {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteBackend {
    /// Open (or create) a `SQLite` database at `path` and run migrations.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` if the database cannot be opened or migrations fail.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(|e| FlowdError::Storage(e.to_string()))?;

        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )
        .map_err(|e| FlowdError::Storage(e.to_string()))?;

        crate::migrations::run_migrations(&conn).map_err(|e| FlowdError::Storage(e.to_string()))?;

        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Persistent plan store sharing this database connection (same mutex as queries).
    #[must_use]
    pub fn plan_store(&self) -> SqlitePlanStore {
        SqlitePlanStore::from_connection(Arc::clone(&self.conn))
    }

    /// Plan-lifecycle event log sharing this database connection. Used by
    /// the daemon's plan observer (HL-39) and `flowd plan events`.
    #[must_use]
    pub fn plan_event_store(&self) -> SqlitePlanEventStore {
        SqlitePlanEventStore::from_connection(Arc::clone(&self.conn))
    }
}

fn storage_err(e: impl std::fmt::Display) -> FlowdError {
    FlowdError::Storage(e.to_string())
}

/// Acquire the connection lock, mapping a poisoned mutex to `FlowdError`.
fn lock(conn: &Mutex<Connection>) -> Result<std::sync::MutexGuard<'_, Connection>> {
    conn.lock().map_err(storage_err)
}

impl MemoryBackend for SqliteBackend {
    async fn store(&self, observation: &Observation) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let obs = observation.clone();

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;

            conn.execute(
                "INSERT INTO observations (id, session_id, project, content, tier, metadata, created_at, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    obs.id.to_string(),
                    obs.session_id.to_string(),
                    obs.project,
                    obs.content,
                    tier_as_str(obs.tier),
                    obs.metadata.to_string(),
                    obs.created_at.to_rfc3339(),
                    obs.updated_at.to_rfc3339(),
                ],
            )
            .map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn get(&self, id: Uuid) -> Result<Option<Observation>> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;
            let mut stmt = conn
                .prepare(
                    "SELECT id, session_id, project, content, tier, metadata, created_at, updated_at
                     FROM observations WHERE id = ?1",
                )
                .map_err(storage_err)?;

            let result = stmt
                .query_row([id.to_string()], |row| Ok(row_to_observation(row)))
                .optional()
                .map_err(storage_err)?;

            match result {
                Some(obs) => Ok(Some(obs.map_err(storage_err)?)),
                None => Ok(None),
            }
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    #[allow(clippy::cast_possible_wrap)]
    async fn keyword_search(&self, query: &SearchQuery) -> Result<Vec<(Observation, f64)>> {
        let conn = Arc::clone(&self.conn);
        let query = query.clone();

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;
            let mut results = Vec::new();

            let mapper = |row: &rusqlite::Row<'_>| -> rusqlite::Result<(Observation, f64)> {
                let obs = row_to_observation(row)?;
                let rank: f64 = row.get(8)?;
                Ok((obs, rank))
            };

            if let Some(ref project) = query.project {
                let mut stmt = conn
                    .prepare(
                        "SELECT o.id, o.session_id, o.project, o.content, o.tier, o.metadata,
                                o.created_at, o.updated_at, rank
                         FROM observations_fts fts
                         JOIN observations o ON o.rowid = fts.rowid
                         WHERE observations_fts MATCH ?1 AND o.project = ?2
                         ORDER BY rank
                         LIMIT ?3",
                    )
                    .map_err(storage_err)?;

                let rows = stmt
                    .query_map(
                        rusqlite::params![query.text, project, query.limit as i64],
                        mapper,
                    )
                    .map_err(storage_err)?;

                for row in rows {
                    let (obs, rank) = row.map_err(storage_err)?;
                    results.push((obs, rank.abs()));
                }
            } else {
                let mut stmt = conn
                    .prepare(
                        "SELECT o.id, o.session_id, o.project, o.content, o.tier, o.metadata,
                                o.created_at, o.updated_at, rank
                         FROM observations_fts fts
                         JOIN observations o ON o.rowid = fts.rowid
                         WHERE observations_fts MATCH ?1
                         ORDER BY rank
                         LIMIT ?2",
                    )
                    .map_err(storage_err)?;

                let rows = stmt
                    .query_map(rusqlite::params![query.text, query.limit as i64], mapper)
                    .map_err(storage_err)?;

                for row in rows {
                    let (obs, rank) = row.map_err(storage_err)?;
                    results.push((obs, rank.abs()));
                }
            }

            Ok(results)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn list_sessions(&self, project: Option<&str>) -> Result<Vec<Session>> {
        let conn = Arc::clone(&self.conn);
        let project = project.map(ToOwned::to_owned);

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;

            let (sql, params): (&str, Vec<Box<dyn rusqlite::types::ToSql>>) = match project {
                Some(ref p) => (
                    "SELECT id, project, summary, started_at, ended_at \
                     FROM sessions WHERE project = ?1 ORDER BY started_at DESC",
                    vec![Box::new(p.clone())],
                ),
                None => (
                    "SELECT id, project, summary, started_at, ended_at \
                     FROM sessions ORDER BY started_at DESC",
                    vec![],
                ),
            };

            let mut stmt = conn.prepare(sql).map_err(storage_err)?;
            let rows = stmt
                .query_map(rusqlite::params_from_iter(params.iter()), |row| {
                    row_to_session(row)
                })
                .map_err(storage_err)?;

            rows.collect::<std::result::Result<Vec<_>, _>>()
                .map_err(storage_err)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn upsert_session(&self, session: &Session) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let session = session.clone();

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;
            conn.execute(
                "INSERT INTO sessions (id, project, summary, started_at, ended_at)
                 VALUES (?1, ?2, ?3, ?4, ?5)
                 ON CONFLICT(id) DO UPDATE SET
                    summary = excluded.summary,
                    ended_at = excluded.ended_at",
                rusqlite::params![
                    session.id.to_string(),
                    session.project,
                    session.summary,
                    session.started_at.to_rfc3339(),
                    session.ended_at.map(|dt| dt.to_rfc3339()),
                ],
            )
            .map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn compact(&self, tier: MemoryTier) -> Result<u64> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;
            let tier_str = tier_as_str(tier);

            let deleted: usize = conn
                .execute(
                    "DELETE FROM observations WHERE tier = ?1",
                    rusqlite::params![tier_str],
                )
                .map_err(storage_err)?;

            Ok(u64::try_from(deleted).unwrap_or(u64::MAX))
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn list_by_tier_and_age(
        &self,
        tier: MemoryTier,
        cutoff: DateTime<Utc>,
    ) -> Result<Vec<Observation>> {
        let conn = Arc::clone(&self.conn);

        tokio::task::spawn_blocking(move || {
            let conn = lock(&conn)?;
            let tier_str = tier_as_str(tier);

            let mut stmt = conn
                .prepare(
                    "SELECT id, session_id, project, content, tier, metadata, created_at, updated_at
                     FROM observations
                     WHERE tier = ?1 AND created_at < ?2
                     ORDER BY session_id, created_at",
                )
                .map_err(storage_err)?;

            let rows = stmt
                .query_map(rusqlite::params![tier_str, cutoff.to_rfc3339()], |row| {
                    Ok(row_to_observation(row))
                })
                .map_err(storage_err)?;

            let mut out = Vec::new();
            for row in rows {
                out.push(row.map_err(storage_err)?.map_err(storage_err)?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn update_tier(&self, ids: &[Uuid], new_tier: MemoryTier) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }

        let conn = Arc::clone(&self.conn);
        let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();

        tokio::task::spawn_blocking(move || {
            let mut conn = lock(&conn)?;
            let tier_str = tier_as_str(new_tier);
            let now = chrono::Utc::now().to_rfc3339();
            let tx = conn.transaction().map_err(storage_err)?;

            let mut updated: usize = 0;
            {
                let mut stmt = tx
                    .prepare("UPDATE observations SET tier = ?1, updated_at = ?2 WHERE id = ?3")
                    .map_err(storage_err)?;
                for id in &ids {
                    updated += stmt
                        .execute(rusqlite::params![tier_str, now, id])
                        .map_err(storage_err)?;
                }
            }

            tx.commit().map_err(storage_err)?;
            Ok(u64::try_from(updated).unwrap_or(u64::MAX))
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    async fn delete_observations(&self, ids: &[Uuid]) -> Result<u64> {
        if ids.is_empty() {
            return Ok(0);
        }

        let conn = Arc::clone(&self.conn);
        let ids: Vec<String> = ids.iter().map(Uuid::to_string).collect();

        tokio::task::spawn_blocking(move || {
            let mut conn = lock(&conn)?;
            let tx = conn.transaction().map_err(storage_err)?;

            let mut deleted: usize = 0;
            {
                let mut stmt = tx
                    .prepare("DELETE FROM observations WHERE id = ?1")
                    .map_err(storage_err)?;
                for id in &ids {
                    deleted += stmt.execute(rusqlite::params![id]).map_err(storage_err)?;
                }
            }

            tx.commit().map_err(storage_err)?;
            Ok(u64::try_from(deleted).unwrap_or(u64::MAX))
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }
}

fn tier_as_str(tier: MemoryTier) -> &'static str {
    match tier {
        MemoryTier::Hot => "hot",
        MemoryTier::Warm => "warm",
        MemoryTier::Cold => "cold",
    }
}

fn tier_from_str(s: &str) -> MemoryTier {
    match s {
        "warm" => MemoryTier::Warm,
        "cold" => MemoryTier::Cold,
        _ => MemoryTier::Hot,
    }
}

fn row_to_observation(row: &rusqlite::Row<'_>) -> rusqlite::Result<Observation> {
    let id_str: String = row.get(0)?;
    let session_str: String = row.get(1)?;
    let tier_str: String = row.get(4)?;
    let meta_str: String = row.get(5)?;
    let created_str: String = row.get(6)?;
    let updated_str: String = row.get(7)?;

    Ok(Observation {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        session_id: Uuid::parse_str(&session_str).unwrap_or_default(),
        project: row.get(2)?,
        content: row.get(3)?,
        tier: tier_from_str(&tier_str),
        metadata: serde_json::from_str(&meta_str).unwrap_or_default(),
        created_at: chrono::DateTime::parse_from_rfc3339(&created_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_default(),
        updated_at: chrono::DateTime::parse_from_rfc3339(&updated_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_default(),
    })
}

fn row_to_session(row: &rusqlite::Row<'_>) -> rusqlite::Result<Session> {
    let id_str: String = row.get(0)?;
    let started_str: String = row.get(3)?;
    let ended_str: Option<String> = row.get(4)?;

    Ok(Session {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        project: row.get(1)?,
        summary: row.get(2)?,
        started_at: chrono::DateTime::parse_from_rfc3339(&started_str)
            .map(|dt| dt.with_timezone(&chrono::Utc))
            .unwrap_or_default(),
        ended_at: ended_str.and_then(|s| {
            chrono::DateTime::parse_from_rfc3339(&s)
                .map(|dt| dt.with_timezone(&chrono::Utc))
                .ok()
        }),
    })
}
