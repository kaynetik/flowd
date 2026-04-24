//! `SQLite` persistence for plan-lifecycle events (`plan_events`).
//!
//! Shares the same connection pool as
//! [`crate::plan_store::SqlitePlanStore`] / [`crate::sqlite::SqliteBackend`]
//! so the executor's plan persistence and its event log live in the same
//! `SQLite` file (and therefore the same WAL).
//!
//! Reads are CLI-friendly (`flowd plan events <id>`) and use the
//! `idx_plan_events_plan` covering index defined in
//! [`crate::migrations`].

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::observer::PlanEvent;
use flowd_core::orchestration::plan_events::{
    PlanEventQuery, PlanEventStore, StoredPlanEvent, event_agent_type, event_kind, event_payload,
    event_step_id,
};
use rusqlite::Connection;
use uuid::Uuid;

#[derive(Clone)]
pub struct SqlitePlanEventStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqlitePlanEventStore {
    /// Open a standalone connection at `path` and run migrations. Useful
    /// for CLI read commands (`flowd plan events`) that do not boot the
    /// full daemon.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` if the database cannot be opened or
    /// migrations fail.
    pub fn open(path: &Path) -> Result<Self> {
        let conn = Connection::open(path).map_err(storage_err)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )
        .map_err(storage_err)?;
        crate::migrations::run_migrations(&conn).map_err(storage_err)?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Share an existing connection (e.g. from
    /// [`crate::sqlite::SqliteBackend`]). The daemon uses this so plan
    /// events sit in the same WAL as plans and observations.
    #[must_use]
    pub fn from_connection(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }
}

fn storage_err(e: impl std::fmt::Display) -> FlowdError {
    FlowdError::Storage(e.to_string())
}

impl PlanEventStore for SqlitePlanEventStore {
    fn record(&self, event: &PlanEvent) -> impl std::future::Future<Output = Result<()>> + Send {
        let conn = Arc::clone(&self.conn);
        let plan_id = event.plan_id().to_string();
        let project = event.project().to_owned();
        let kind = event_kind(event).to_owned();
        let step_id = event_step_id(event).map(ToOwned::to_owned);
        let agent_type = event_agent_type(event).map(ToOwned::to_owned);
        let payload = event_payload(event).to_string();

        async move {
            tokio::task::spawn_blocking(move || {
                let conn = conn.lock().map_err(storage_err)?;
                conn.execute(
                    "INSERT INTO plan_events (plan_id, project, kind, step_id, agent_type, payload)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                    rusqlite::params![&plan_id, &project, &kind, &step_id, &agent_type, &payload],
                )
                .map_err(storage_err)?;
                Ok(())
            })
            .await
            .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
        }
    }

    fn list_for_plan(
        &self,
        plan_id: Uuid,
        query: PlanEventQuery,
    ) -> impl std::future::Future<Output = Result<Vec<StoredPlanEvent>>> + Send {
        let conn = Arc::clone(&self.conn);
        let plan_id_str = plan_id.to_string();

        async move {
            tokio::task::spawn_blocking(move || {
                let conn = conn.lock().map_err(storage_err)?;
                // Cap the limit to a sane i64 for SQLite. `usize::MAX` would
                // otherwise overflow on conversion.
                let limit = i64::try_from(query.limit).unwrap_or(i64::MAX);
                let mut rows = Vec::new();

                if query.kinds.is_empty() {
                    let mut stmt = conn
                        .prepare(
                            "SELECT id, plan_id, project, kind, step_id, agent_type, payload, created_at
                             FROM plan_events
                             WHERE plan_id = ?1
                             ORDER BY created_at ASC, id ASC
                             LIMIT ?2",
                        )
                        .map_err(storage_err)?;
                    let iter = stmt
                        .query_map(rusqlite::params![&plan_id_str, limit], row_to_event)
                        .map_err(storage_err)?;
                    for r in iter {
                        rows.push(r.map_err(storage_err)?);
                    }
                } else {
                    // SQLite does not support binding `IN (?)` arrays, so
                    // build a `?,?,?` placeholder list of the right arity.
                    let placeholders = std::iter::repeat_n("?", query.kinds.len())
                        .collect::<Vec<_>>()
                        .join(",");
                    let sql = format!(
                        "SELECT id, plan_id, project, kind, step_id, agent_type, payload, created_at
                         FROM plan_events
                         WHERE plan_id = ?1 AND kind IN ({placeholders})
                         ORDER BY created_at ASC, id ASC
                         LIMIT ?{n}",
                        n = query.kinds.len() + 2,
                    );
                    let mut stmt = conn.prepare(&sql).map_err(storage_err)?;

                    let mut params: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();
                    params.push(Box::new(plan_id_str.clone()));
                    for k in &query.kinds {
                        params.push(Box::new(k.clone()));
                    }
                    params.push(Box::new(limit));

                    let iter = stmt
                        .query_map(rusqlite::params_from_iter(params.iter()), row_to_event)
                        .map_err(storage_err)?;
                    for r in iter {
                        rows.push(r.map_err(storage_err)?);
                    }
                }
                Ok(rows)
            })
            .await
            .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
        }
    }
}

fn row_to_event(row: &rusqlite::Row<'_>) -> rusqlite::Result<StoredPlanEvent> {
    let plan_id_str: String = row.get(1)?;
    let payload_str: String = row.get(6)?;
    let created_str: String = row.get(7)?;

    Ok(StoredPlanEvent {
        id: row.get(0)?,
        plan_id: Uuid::parse_str(&plan_id_str).unwrap_or_default(),
        project: row.get(2)?,
        kind: row.get(3)?,
        step_id: row.get(4)?,
        agent_type: row.get(5)?,
        payload: serde_json::from_str(&payload_str).unwrap_or(serde_json::Value::Null),
        created_at: parse_sqlite_timestamp(&created_str),
    })
}

/// Parse a `created_at` string written by `SQLite`'s `datetime('now')`
/// (e.g. `2026-04-20 16:13:55`) or by an explicit RFC-3339 caller. The
/// stock `datetime('now')` output has no `T` separator and no offset, so
/// `DateTime::parse_from_rfc3339` rejects it; we try both forms.
fn parse_sqlite_timestamp(s: &str) -> DateTime<Utc> {
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return dt.with_timezone(&Utc);
    }
    if let Ok(naive) = chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%d %H:%M:%S") {
        return DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc);
    }
    Utc::now()
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::PlanStatus;
    use flowd_core::orchestration::plan_events::kind;

    fn store() -> SqlitePlanEventStore {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        crate::migrations::run_migrations(&conn).expect("migrations");
        SqlitePlanEventStore::from_connection(Arc::new(Mutex::new(conn)))
    }

    #[tokio::test]
    async fn round_trip_records_and_lists_in_order() {
        let store = store();
        let plan_id = Uuid::new_v4();
        let project = "demo".to_owned();

        let events = [
            PlanEvent::Submitted {
                plan_id,
                name: "p".into(),
                project: project.clone(),
            },
            PlanEvent::Started {
                plan_id,
                project: project.clone(),
            },
            PlanEvent::StepCompleted {
                plan_id,
                project: project.clone(),
                step_id: "build".into(),
                agent_type: "echo".into(),
                output: "ok".into(),
                metrics: None,
            },
            PlanEvent::Finished {
                plan_id,
                project,
                status: PlanStatus::Completed,
                total_metrics: None,
                step_count: flowd_core::orchestration::observer::PlanStepCounts::default(),
            },
        ];
        for e in &events {
            store.record(e).await.expect("record");
        }

        let rows = store
            .list_for_plan(plan_id, PlanEventQuery::new(100))
            .await
            .expect("list");

        assert_eq!(rows.len(), 4);
        assert_eq!(rows[0].kind, kind::SUBMITTED);
        assert_eq!(rows[1].kind, kind::STARTED);
        assert_eq!(rows[2].kind, kind::STEP_COMPLETED);
        assert_eq!(rows[2].step_id.as_deref(), Some("build"));
        assert_eq!(rows[2].agent_type.as_deref(), Some("echo"));
        assert_eq!(rows[2].payload["output"], "ok");
        assert_eq!(rows[3].kind, kind::FINISHED);
        assert_eq!(rows[3].payload["status"], "completed");
    }

    #[tokio::test]
    async fn list_filters_by_kind() {
        let store = store();
        let plan_id = Uuid::new_v4();
        let project = "demo".to_owned();

        store
            .record(&PlanEvent::Submitted {
                plan_id,
                name: "p".into(),
                project: project.clone(),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepFailed {
                plan_id,
                project: project.clone(),
                step_id: "a".into(),
                agent_type: "echo".into(),
                error: "boom".into(),
                metrics: None,
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project,
                step_id: "b".into(),
                agent_type: "echo".into(),
                output: "ok".into(),
                metrics: None,
            })
            .await
            .unwrap();

        let only_failed = store
            .list_for_plan(
                plan_id,
                PlanEventQuery::new(100).with_kinds([kind::STEP_FAILED]),
            )
            .await
            .unwrap();
        assert_eq!(only_failed.len(), 1);
        assert_eq!(only_failed[0].kind, kind::STEP_FAILED);

        let two_kinds = store
            .list_for_plan(
                plan_id,
                PlanEventQuery::new(100).with_kinds([kind::STEP_FAILED, kind::STEP_COMPLETED]),
            )
            .await
            .unwrap();
        assert_eq!(two_kinds.len(), 2);
    }

    #[tokio::test]
    async fn list_respects_limit() {
        let store = store();
        let plan_id = Uuid::new_v4();
        for _ in 0..5 {
            store
                .record(&PlanEvent::Started {
                    plan_id,
                    project: "demo".into(),
                })
                .await
                .unwrap();
        }
        let rows = store
            .list_for_plan(plan_id, PlanEventQuery::new(2))
            .await
            .unwrap();
        assert_eq!(rows.len(), 2);
    }

    #[tokio::test]
    async fn list_scopes_to_plan_id() {
        let store = store();
        let plan_a = Uuid::new_v4();
        let plan_b = Uuid::new_v4();
        store
            .record(&PlanEvent::Started {
                plan_id: plan_a,
                project: "demo".into(),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::Started {
                plan_id: plan_b,
                project: "demo".into(),
            })
            .await
            .unwrap();

        let a_only = store
            .list_for_plan(plan_a, PlanEventQuery::new(100))
            .await
            .unwrap();
        assert_eq!(a_only.len(), 1);
        assert_eq!(a_only[0].plan_id, plan_a);
    }

    #[test]
    fn migration_004_is_idempotent() {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        crate::migrations::run_migrations(&conn).expect("first run");
        crate::migrations::run_migrations(&conn).expect("second run");
        // Second run is a no-op because of the `migrations` table guard;
        // raw `IF NOT EXISTS` on the schema statements is defence-in-depth.
        let table_exists: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='plan_events'",
                [],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(table_exists, 1);
    }
}
