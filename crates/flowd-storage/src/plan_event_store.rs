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

use std::collections::BTreeMap;
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

/// Cross-plan token + cost rollup, computed straight from the JSON
/// `payload` column on `plan_events` rows. We sum only `step_completed`
/// and `step_failed` (failed steps are billable spend) and deliberately
/// skip `finished` rows -- their `total_metrics` is itself a sum of those
/// step rows, so including them would double-count.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct UsageTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub total_cost_usd: f64,
    pub step_events: u64,
}

impl UsageTotals {
    /// Cache-hit ratio over the input-side cache counters:
    /// `cache_read / (cache_read + cache_creation)`.
    ///
    /// Returns `None` when both counters are zero. Callers should
    /// suppress the metric in that case rather than render `NaN` or a
    /// misleading `0.00` -- there was simply no cache activity to
    /// report a ratio over.
    #[must_use]
    #[allow(clippy::cast_precision_loss)]
    pub fn cache_hit_rate(&self) -> Option<f64> {
        let denom = self.cache_read_tokens + self.cache_creation_tokens;
        if denom == 0 {
            return None;
        }
        Some(self.cache_read_tokens as f64 / denom as f64)
    }
}

/// Per-model rollup of token + cost usage, derived from the
/// `metrics.model_usage` map embedded in step-event payloads. Field
/// names mirror the storage keys (`cache_read_input_tokens` etc. on
/// `ModelUsage`) but expose the shorter, ratio-friendly aliases used
/// elsewhere in this module.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ModelTotals {
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    pub cost_usd: f64,
}

/// Shared SELECT clause for [`UsageTotals`] aggregation. Callers
/// append a `WHERE` clause and any extra parameters; the column order
/// must stay aligned with [`row_to_usage_totals`].
const USAGE_SELECT: &str = "SELECT
    COALESCE(SUM(json_extract(payload, '$.metrics.input_tokens')), 0),
    COALESCE(SUM(json_extract(payload, '$.metrics.output_tokens')), 0),
    COALESCE(SUM(json_extract(payload, '$.metrics.cache_read_input_tokens')), 0),
    COALESCE(SUM(json_extract(payload, '$.metrics.cache_creation_input_tokens')), 0),
    COALESCE(SUM(json_extract(payload, '$.metrics.total_cost_usd')), 0.0),
    COUNT(*)
 FROM plan_events";

/// Predicate every usage aggregate shares: only step-outcome rows
/// (failed steps still cost money, so they stay in), and only rows
/// that actually carry a `metrics` block. The `kind IN (...)` clause
/// is what keeps `finished` rows out, so their `total_metrics` --
/// itself a sum of these step rows -- never gets double-counted.
const USAGE_BASE_WHERE: &str = "kind IN ('step_completed', 'step_failed')
       AND json_extract(payload, '$.metrics') IS NOT NULL";

fn row_to_usage_totals(row: &rusqlite::Row<'_>) -> rusqlite::Result<UsageTotals> {
    let to_u64 = |v: i64| u64::try_from(v).unwrap_or(0);
    Ok(UsageTotals {
        input_tokens: to_u64(row.get::<_, i64>(0)?),
        output_tokens: to_u64(row.get::<_, i64>(1)?),
        cache_read_tokens: to_u64(row.get::<_, i64>(2)?),
        cache_creation_tokens: to_u64(row.get::<_, i64>(3)?),
        total_cost_usd: row.get::<_, f64>(4)?,
        step_events: to_u64(row.get::<_, i64>(5)?),
    })
}

impl SqlitePlanEventStore {
    /// Aggregate token + cost totals across all persisted step events.
    ///
    /// Returns `UsageTotals::default()` (all zeros) when no events carry
    /// a `metrics` payload. Pre-metrics rows are ignored by the
    /// `WHERE json_extract(...) IS NOT NULL` predicate.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` if the underlying `SQLite` query
    /// fails or the connection mutex is poisoned.
    pub async fn usage_totals(&self) -> Result<UsageTotals> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let sql = format!("{USAGE_SELECT}\n WHERE {USAGE_BASE_WHERE}");
            conn.query_row(&sql, [], row_to_usage_totals)
                .map_err(storage_err)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    /// Aggregate token + cost totals for a single plan.
    ///
    /// Same shape and exclusions as [`Self::usage_totals`]; returns
    /// `UsageTotals::default()` when the plan has no metrics-bearing
    /// step events (or doesn't exist).
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on `SQLite` or mutex failures.
    pub async fn usage_totals_for_plan(&self, plan_id: Uuid) -> Result<UsageTotals> {
        let conn = Arc::clone(&self.conn);
        let plan_id_str = plan_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let sql = format!("{USAGE_SELECT}\n WHERE {USAGE_BASE_WHERE}\n   AND plan_id = ?1");
            conn.query_row(&sql, rusqlite::params![&plan_id_str], row_to_usage_totals)
                .map_err(storage_err)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    /// Aggregate token + cost totals for events recorded in the
    /// half-open interval `[start, end)`, suitable for daily / weekly /
    /// monthly status reports.
    ///
    /// The query ranges over the persisted `created_at` column rather
    /// than any payload-internal timestamp, so it sees exactly what the
    /// audit log saw. Inputs are normalised through `datetime()` so the
    /// comparison works regardless of whether a row was written by
    /// `datetime('now')` (`YYYY-MM-DD HH:MM:SS`) or by a caller that
    /// stamped an explicit RFC-3339 string.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on `SQLite` or mutex failures.
    pub async fn usage_totals_for_period(
        &self,
        start: DateTime<Utc>,
        end: DateTime<Utc>,
    ) -> Result<UsageTotals> {
        let conn = Arc::clone(&self.conn);
        let start_s = start.to_rfc3339();
        let end_s = end.to_rfc3339();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let sql = format!(
                "{USAGE_SELECT}\n WHERE {USAGE_BASE_WHERE}\n   \
                 AND datetime(created_at) >= datetime(?1)\n   \
                 AND datetime(created_at) <  datetime(?2)"
            );
            conn.query_row(
                &sql,
                rusqlite::params![&start_s, &end_s],
                row_to_usage_totals,
            )
            .map_err(storage_err)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    /// Per-model rollup across all step-outcome events that reported
    /// a `metrics.model_usage` map. Events with no `model_usage`
    /// (or an empty map) drop out of the aggregate naturally because
    /// `json_each` produces zero rows for them; the resulting
    /// `BTreeMap` is empty when nothing in the database has a
    /// per-model breakdown.
    ///
    /// We deliberately do not synthesise model splits from top-level
    /// `metrics.input_tokens` etc. -- the spawner is the only place
    /// that knows which model handled what, and inferring otherwise
    /// would require provider-specific rate-card logic this module
    /// avoids.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on `SQLite` or mutex failures.
    pub async fn model_usage_totals(&self) -> Result<BTreeMap<String, ModelTotals>> {
        let conn = Arc::clone(&self.conn);
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let mut stmt = conn
                .prepare(
                    "SELECT
                        je.key,
                        COALESCE(SUM(json_extract(je.value, '$.input_tokens')), 0),
                        COALESCE(SUM(json_extract(je.value, '$.output_tokens')), 0),
                        COALESCE(SUM(json_extract(je.value, '$.cache_read_input_tokens')), 0),
                        COALESCE(SUM(json_extract(je.value, '$.cache_creation_input_tokens')), 0),
                        COALESCE(SUM(json_extract(je.value, '$.cost_usd')), 0.0)
                     FROM plan_events,
                          json_each(json_extract(payload, '$.metrics.model_usage')) AS je
                     WHERE kind IN ('step_completed', 'step_failed')
                       AND json_extract(payload, '$.metrics.model_usage') IS NOT NULL
                     GROUP BY je.key",
                )
                .map_err(storage_err)?;
            let rows = stmt
                .query_map([], |row| {
                    let to_u64 = |v: i64| u64::try_from(v).unwrap_or(0);
                    let model: String = row.get(0)?;
                    let totals = ModelTotals {
                        input_tokens: to_u64(row.get::<_, i64>(1)?),
                        output_tokens: to_u64(row.get::<_, i64>(2)?),
                        cache_read_tokens: to_u64(row.get::<_, i64>(3)?),
                        cache_creation_tokens: to_u64(row.get::<_, i64>(4)?),
                        cost_usd: row.get::<_, f64>(5)?,
                    };
                    Ok((model, totals))
                })
                .map_err(storage_err)?;
            let mut out = BTreeMap::new();
            for r in rows {
                let (k, v) = r.map_err(storage_err)?;
                out.insert(k, v);
            }
            Ok(out)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }
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
            PlanEvent::StepStarted {
                plan_id,
                project: project.clone(),
                step_id: "build".into(),
                agent_type: "echo".into(),
                started_at: Utc::now(),
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
                elapsed_ms: None,
            },
        ];
        for e in &events {
            store.record(e).await.expect("record");
        }

        let rows = store
            .list_for_plan(plan_id, PlanEventQuery::new(100))
            .await
            .expect("list");

        assert_eq!(rows.len(), 5);
        assert_eq!(rows[0].kind, kind::SUBMITTED);
        assert_eq!(rows[1].kind, kind::STARTED);
        assert_eq!(rows[2].kind, kind::STEP_STARTED);
        assert_eq!(rows[2].step_id.as_deref(), Some("build"));
        assert_eq!(rows[2].agent_type.as_deref(), Some("echo"));
        // Per-step identity / agent live on dedicated columns; the JSON
        // payload carries only the executor-stamped `started_at` so a
        // post-restart consumer can recover the real start moment even
        // if the row's `created_at` drifted past it.
        let payload_obj = rows[2]
            .payload
            .as_object()
            .expect("payload is an object");
        assert_eq!(
            payload_obj.len(),
            1,
            "unexpected step_started payload shape: {}",
            rows[2].payload
        );
        assert!(
            payload_obj
                .get("started_at")
                .and_then(|v| v.as_str())
                .is_some(),
            "step_started payload must carry started_at; got {}",
            rows[2].payload
        );
        assert_eq!(rows[3].kind, kind::STEP_COMPLETED);
        assert_eq!(rows[3].step_id.as_deref(), Some("build"));
        assert_eq!(rows[3].agent_type.as_deref(), Some("echo"));
        assert_eq!(rows[3].payload["output"], "ok");
        assert_eq!(rows[4].kind, kind::FINISHED);
        assert_eq!(rows[4].payload["status"], "completed");
    }

    #[tokio::test]
    async fn list_filters_by_step_started_kind() {
        // The kind-filter path is a separate code branch from the
        // unfiltered list; exercise it for the new variant so a future
        // tweak to the placeholder builder cannot silently drop
        // step_started rows.
        let store = store();
        let plan_id = Uuid::new_v4();
        let project = "demo".to_owned();

        store
            .record(&PlanEvent::StepStarted {
                plan_id,
                project: project.clone(),
                step_id: "build".into(),
                agent_type: "claude".into(),
                started_at: Utc::now(),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project,
                step_id: "build".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: None,
            })
            .await
            .unwrap();

        let only_started = store
            .list_for_plan(
                plan_id,
                PlanEventQuery::new(100).with_kinds([kind::STEP_STARTED]),
            )
            .await
            .unwrap();
        assert_eq!(only_started.len(), 1);
        assert_eq!(only_started[0].kind, kind::STEP_STARTED);
        assert_eq!(only_started[0].step_id.as_deref(), Some("build"));
        assert_eq!(only_started[0].agent_type.as_deref(), Some("claude"));
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

    #[tokio::test]
    async fn usage_totals_sums_step_rows_and_skips_finished() {
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_id = Uuid::new_v4();
        let project = "demo".to_owned();

        let m1 = AgentMetrics {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 10,
            cache_creation_input_tokens: 5,
            total_cost_usd: 0.0123,
            duration_ms: 1000,
            duration_api_ms: 800,
            ..AgentMetrics::default()
        };
        let m2 = AgentMetrics {
            input_tokens: 200,
            output_tokens: 75,
            total_cost_usd: 0.0456,
            ..AgentMetrics::default()
        };
        // Failed steps still cost money -- include them in the rollup.
        let m_failed = AgentMetrics {
            input_tokens: 30,
            output_tokens: 0,
            total_cost_usd: 0.0010,
            ..AgentMetrics::default()
        };
        // `total_metrics` on Finished is itself a sum of the step rows;
        // including it in the aggregate would double-count.
        let mut total = AgentMetrics::default();
        total.merge(&m1);
        total.merge(&m2);
        total.merge(&m_failed);

        let events = [
            PlanEvent::StepCompleted {
                plan_id,
                project: project.clone(),
                step_id: "a".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(m1),
            },
            PlanEvent::StepCompleted {
                plan_id,
                project: project.clone(),
                step_id: "b".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(m2),
            },
            PlanEvent::StepFailed {
                plan_id,
                project: project.clone(),
                step_id: "c".into(),
                agent_type: "claude".into(),
                error: "rate limited".into(),
                metrics: Some(m_failed),
            },
            // Pre-metrics row -- must be ignored without poisoning the sum.
            PlanEvent::StepCompleted {
                plan_id,
                project: project.clone(),
                step_id: "legacy".into(),
                agent_type: "echo".into(),
                output: "ok".into(),
                metrics: None,
            },
            PlanEvent::Finished {
                plan_id,
                project,
                status: PlanStatus::Completed,
                total_metrics: Some(total),
                step_count: flowd_core::orchestration::observer::PlanStepCounts::default(),
                elapsed_ms: None,
            },
        ];
        for e in &events {
            store.record(e).await.expect("record");
        }

        let totals = store.usage_totals().await.expect("usage_totals");
        assert_eq!(totals.input_tokens, 100 + 200 + 30);
        assert_eq!(totals.output_tokens, 50 + 75);
        assert_eq!(totals.cache_read_tokens, 10);
        assert_eq!(totals.cache_creation_tokens, 5);
        assert!((totals.total_cost_usd - (0.0123 + 0.0456 + 0.0010)).abs() < 1e-9);
        // Three step rows had metrics; the `metrics: None` row is filtered
        // out by the `json_extract IS NOT NULL` predicate, the Finished
        // row by the `kind IN (...)` clause.
        assert_eq!(totals.step_events, 3);
    }

    #[tokio::test]
    async fn usage_totals_returns_zero_on_empty_db() {
        let store = store();
        let totals = store.usage_totals().await.expect("usage_totals");
        assert_eq!(totals, UsageTotals::default());
    }

    #[tokio::test]
    async fn usage_totals_excludes_finished_rows_with_no_step_events() {
        // A `finished` row carrying its own `total_metrics` is the
        // shape that would double-count if our `kind IN (...)` filter
        // ever regressed: there are no step rows here, so any non-zero
        // total can only come from the finished row leaking in.
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_id = Uuid::new_v4();
        let total = AgentMetrics {
            input_tokens: 999,
            output_tokens: 999,
            total_cost_usd: 9.99,
            ..AgentMetrics::default()
        };
        store
            .record(&PlanEvent::Finished {
                plan_id,
                project: "demo".into(),
                status: PlanStatus::Completed,
                total_metrics: Some(total),
                step_count: flowd_core::orchestration::observer::PlanStepCounts::default(),
                elapsed_ms: None,
            })
            .await
            .unwrap();

        let totals = store.usage_totals().await.expect("usage_totals");
        assert_eq!(totals, UsageTotals::default());
    }

    #[tokio::test]
    async fn usage_totals_for_plan_scopes_to_one_plan() {
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_a = Uuid::new_v4();
        let plan_b = Uuid::new_v4();
        let metrics_a = AgentMetrics {
            input_tokens: 10,
            output_tokens: 5,
            total_cost_usd: 0.01,
            ..AgentMetrics::default()
        };
        let metrics_b = AgentMetrics {
            input_tokens: 1_000,
            output_tokens: 500,
            total_cost_usd: 1.00,
            ..AgentMetrics::default()
        };
        store
            .record(&PlanEvent::StepCompleted {
                plan_id: plan_a,
                project: "demo".into(),
                step_id: "a".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(metrics_a),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepCompleted {
                plan_id: plan_b,
                project: "demo".into(),
                step_id: "b".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(metrics_b),
            })
            .await
            .unwrap();

        let only_a = store
            .usage_totals_for_plan(plan_a)
            .await
            .expect("usage_totals_for_plan");
        assert_eq!(only_a.input_tokens, 10);
        assert_eq!(only_a.output_tokens, 5);
        assert_eq!(only_a.step_events, 1);
        assert!((only_a.total_cost_usd - 0.01).abs() < 1e-9);

        let unknown = store
            .usage_totals_for_plan(Uuid::new_v4())
            .await
            .expect("usage_totals_for_plan");
        assert_eq!(unknown, UsageTotals::default());
    }

    #[tokio::test]
    async fn usage_totals_for_plan_includes_failed_step_cost() {
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_id = Uuid::new_v4();
        let succeeded = AgentMetrics {
            input_tokens: 50,
            total_cost_usd: 0.05,
            ..AgentMetrics::default()
        };
        let failed = AgentMetrics {
            input_tokens: 20,
            total_cost_usd: 0.02,
            ..AgentMetrics::default()
        };
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "ok".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(succeeded),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepFailed {
                plan_id,
                project: "demo".into(),
                step_id: "boom".into(),
                agent_type: "claude".into(),
                error: "rate limited".into(),
                metrics: Some(failed),
            })
            .await
            .unwrap();

        let totals = store
            .usage_totals_for_plan(plan_id)
            .await
            .expect("usage_totals_for_plan");
        assert_eq!(totals.input_tokens, 70);
        assert_eq!(totals.step_events, 2);
        assert!((totals.total_cost_usd - 0.07).abs() < 1e-9);
    }

    #[tokio::test]
    async fn usage_totals_for_period_filters_by_created_at() {
        use chrono::Duration;
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_id = Uuid::new_v4();
        let metrics = AgentMetrics {
            input_tokens: 42,
            total_cost_usd: 0.42,
            ..AgentMetrics::default()
        };
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "a".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(metrics),
            })
            .await
            .unwrap();

        let now = Utc::now();
        let in_window = store
            .usage_totals_for_period(now - Duration::minutes(5), now + Duration::minutes(5))
            .await
            .expect("usage_totals_for_period");
        assert_eq!(in_window.input_tokens, 42);
        assert_eq!(in_window.step_events, 1);

        let future = store
            .usage_totals_for_period(now + Duration::hours(1), now + Duration::hours(2))
            .await
            .expect("usage_totals_for_period");
        assert_eq!(future, UsageTotals::default());
    }

    #[tokio::test]
    async fn model_usage_totals_sums_per_model_and_excludes_finished() {
        use flowd_core::orchestration::executor::{AgentMetrics, ModelUsage};

        let store = store();
        let plan_id = Uuid::new_v4();

        let mut step_one = AgentMetrics {
            input_tokens: 10,
            output_tokens: 5,
            total_cost_usd: 0.01,
            ..AgentMetrics::default()
        };
        step_one.model_usage.insert(
            "sonnet".into(),
            ModelUsage {
                input_tokens: 10,
                output_tokens: 5,
                cost_usd: 0.01,
                ..Default::default()
            },
        );

        let mut step_two = AgentMetrics {
            input_tokens: 7,
            output_tokens: 3,
            total_cost_usd: 0.007,
            ..AgentMetrics::default()
        };
        step_two.model_usage.insert(
            "sonnet".into(),
            ModelUsage {
                input_tokens: 4,
                output_tokens: 1,
                cache_read_input_tokens: 2,
                cost_usd: 0.004,
                ..Default::default()
            },
        );
        step_two.model_usage.insert(
            "haiku".into(),
            ModelUsage {
                input_tokens: 3,
                output_tokens: 2,
                cost_usd: 0.003,
                ..Default::default()
            },
        );

        let mut total = AgentMetrics::default();
        total.merge(&step_one);
        total.merge(&step_two);

        let events = [
            PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "a".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(step_one),
            },
            PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "b".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(step_two),
            },
            // Finished carries the same models in `total_metrics`. If the
            // aggregate ever forgot to filter `kind`, sonnet/haiku would
            // double-count off this row.
            PlanEvent::Finished {
                plan_id,
                project: "demo".into(),
                status: PlanStatus::Completed,
                total_metrics: Some(total),
                step_count: flowd_core::orchestration::observer::PlanStepCounts::default(),
                elapsed_ms: None,
            },
        ];
        for e in &events {
            store.record(e).await.expect("record");
        }

        let by_model = store
            .model_usage_totals()
            .await
            .expect("model_usage_totals");
        assert_eq!(by_model.len(), 2);

        let sonnet = by_model.get("sonnet").expect("sonnet bucket");
        assert_eq!(sonnet.input_tokens, 14);
        assert_eq!(sonnet.output_tokens, 6);
        assert_eq!(sonnet.cache_read_tokens, 2);
        assert!((sonnet.cost_usd - 0.014).abs() < 1e-9);

        let haiku = by_model.get("haiku").expect("haiku bucket");
        assert_eq!(haiku.input_tokens, 3);
        assert_eq!(haiku.output_tokens, 2);
        assert!((haiku.cost_usd - 0.003).abs() < 1e-9);
    }

    #[tokio::test]
    async fn model_usage_totals_returns_empty_when_no_model_usage_present() {
        use flowd_core::orchestration::executor::AgentMetrics;

        let store = store();
        let plan_id = Uuid::new_v4();
        // Metrics present, but no per-model breakdown -- common for
        // spawners that don't differentiate.
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "a".into(),
                agent_type: "echo".into(),
                output: "ok".into(),
                metrics: Some(AgentMetrics {
                    input_tokens: 1,
                    total_cost_usd: 0.001,
                    ..AgentMetrics::default()
                }),
            })
            .await
            .unwrap();

        let by_model = store
            .model_usage_totals()
            .await
            .expect("model_usage_totals");
        assert!(by_model.is_empty());
    }

    #[test]
    fn cache_hit_rate_returns_none_when_denominator_zero() {
        // No cache activity at all -- the rate would be 0/0. Suppress
        // it instead of rendering NaN or a misleading 0.0.
        let totals = UsageTotals {
            input_tokens: 100,
            output_tokens: 50,
            total_cost_usd: 0.5,
            step_events: 1,
            ..UsageTotals::default()
        };
        assert_eq!(totals.cache_hit_rate(), None);
    }

    #[test]
    fn cache_hit_rate_computes_when_cache_creation_present() {
        // 0 hits out of 100 created tokens = 0% hit rate. The
        // denominator is non-zero, so we report the ratio rather than
        // suppressing it.
        let totals = UsageTotals {
            cache_creation_tokens: 100,
            ..UsageTotals::default()
        };
        let rate = totals.cache_hit_rate().expect("denominator non-zero");
        assert!(rate.abs() < 1e-9);

        let totals = UsageTotals {
            cache_read_tokens: 75,
            cache_creation_tokens: 25,
            ..UsageTotals::default()
        };
        let rate = totals.cache_hit_rate().expect("denominator non-zero");
        assert!((rate - 0.75).abs() < 1e-9);
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
