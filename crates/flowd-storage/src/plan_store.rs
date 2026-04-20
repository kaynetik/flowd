//! `SQLite` persistence for orchestration plans (`plans` + `plan_steps`).

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{Plan, PlanStatus, PlanStore, PlanSummary, StepStatus};
use rusqlite::OptionalExtension;
use uuid::Uuid;

#[derive(Clone)]
pub struct SqlitePlanStore {
    conn: Arc<Mutex<rusqlite::Connection>>,
}

impl SqlitePlanStore {
    /// Open the same database file as [`crate::sqlite::SqliteBackend`] and run migrations.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` if the database cannot be opened or migrations fail.
    pub fn open(path: &Path) -> Result<Self> {
        let conn =
            rusqlite::Connection::open(path).map_err(|e| FlowdError::Storage(e.to_string()))?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL; PRAGMA foreign_keys=ON; PRAGMA busy_timeout=5000;",
        )
        .map_err(|e| FlowdError::Storage(e.to_string()))?;
        crate::migrations::run_migrations(&conn).map_err(|e| FlowdError::Storage(e.to_string()))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }

    /// Share an existing `SQLite` connection (e.g. from [`crate::sqlite::SqliteBackend`]).
    #[must_use]
    pub fn from_connection(conn: Arc<Mutex<rusqlite::Connection>>) -> Self {
        Self { conn }
    }
}

fn storage_err(e: impl std::fmt::Display) -> FlowdError {
    FlowdError::Storage(e.to_string())
}

impl PlanStore for SqlitePlanStore {
    fn save_plan(&self, plan: &Plan) -> impl std::future::Future<Output = Result<()>> + Send {
        let conn = Arc::clone(&self.conn);
        let plan = plan.clone();

        async move {
            tokio::task::spawn_blocking(move || {
                // `mut` is required: `Connection::transaction(&mut self)`.
                let mut conn = conn
                    .lock()
                    .map_err(|e| FlowdError::Storage(e.to_string()))?;
                let definition = serde_json::to_string(&plan).map_err(|e| {
                    FlowdError::Serialization(format!("persist plan {}: {e}", plan.id))
                })?;

                let tx = conn.transaction().map_err(storage_err)?;
                let id = plan.id.to_string();
                tx.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?1",
                    rusqlite::params![&id],
                )
                .map_err(storage_err)?;
                tx.execute(
                    "INSERT OR REPLACE INTO plans (id, name, status, definition, created_at, started_at, completed_at, project)
                     VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                    rusqlite::params![
                        &id,
                        &plan.name,
                        plan_status_str(plan.status),
                        &definition,
                        plan.created_at.to_rfc3339(),
                        plan.started_at.map(|t| t.to_rfc3339()),
                        plan.completed_at.map(|t| t.to_rfc3339()),
                        plan.project.as_deref(),
                    ],
                )
                .map_err(storage_err)?;

                for step in &plan.steps {
                    tx.execute(
                        "INSERT INTO plan_steps (id, plan_id, agent_type, prompt, depends_on, timeout_secs, retry_count, status, output, error, started_at, completed_at)
                         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12)",
                        rusqlite::params![
                            &step.id,
                            &id,
                            &step.agent_type,
                            &step.prompt,
                            serde_json::to_string(&step.depends_on).map_err(storage_err)?,
                            step.timeout_secs
                                .map(|n| i64::try_from(n).unwrap_or(i64::MAX)),
                            i64::from(step.retry_count),
                            step_status_str(step.status),
                            step.output.as_deref(),
                            step.error.as_deref(),
                            step.started_at.map(|t| t.to_rfc3339()),
                            step.completed_at.map(|t| t.to_rfc3339()),
                        ],
                    )
                    .map_err(storage_err)?;
                }
                tx.commit().map_err(storage_err)?;
                Ok(())
            })
            .await
            .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
        }
    }

    fn load_plan(
        &self,
        id: Uuid,
    ) -> impl std::future::Future<Output = Result<Option<Plan>>> + Send {
        let conn = Arc::clone(&self.conn);
        let id_str = id.to_string();

        async move {
            tokio::task::spawn_blocking(move || {
                let conn = conn
                    .lock()
                    .map_err(|e| FlowdError::Storage(e.to_string()))?;
                let mut stmt = conn
                    .prepare("SELECT definition FROM plans WHERE id = ?1")
                    .map_err(storage_err)?;
                let row: Option<String> = stmt
                    .query_row([&id_str], |row| row.get(0))
                    .optional()
                    .map_err(storage_err)?;
                let Some(json) = row else {
                    return Ok(None);
                };
                let plan: Plan = serde_json::from_str(&json)
                    .map_err(|e| FlowdError::Serialization(e.to_string()))?;
                Ok(Some(plan))
            })
            .await
            .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
        }
    }

    fn list_plans(
        &self,
        project: Option<&str>,
    ) -> impl std::future::Future<Output = Result<Vec<PlanSummary>>> + Send {
        let conn = Arc::clone(&self.conn);
        let project = project.map(ToOwned::to_owned);

        async move {
            tokio::task::spawn_blocking(move || {
                let conn = conn
                    .lock()
                    .map_err(|e| FlowdError::Storage(e.to_string()))?;

                let mut rows: Vec<PlanSummary> = Vec::new();
                if let Some(ref p) = project {
                    let mut stmt = conn
                        .prepare(
                            "SELECT id, name, status, created_at, project
                             FROM plans WHERE project = ?1
                             ORDER BY created_at DESC",
                        )
                        .map_err(storage_err)?;
                    let iter = stmt
                        .query_map(rusqlite::params![p], row_to_summary)
                        .map_err(storage_err)?;
                    for r in iter {
                        rows.push(r.map_err(storage_err)?);
                    }
                } else {
                    let mut stmt = conn
                        .prepare(
                            "SELECT id, name, status, created_at, project
                             FROM plans ORDER BY created_at DESC",
                        )
                        .map_err(storage_err)?;
                    let iter = stmt.query_map([], row_to_summary).map_err(storage_err)?;
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

    fn delete_plan(&self, id: Uuid) -> impl std::future::Future<Output = Result<()>> + Send {
        let conn = Arc::clone(&self.conn);
        let id = id.to_string();

        async move {
            tokio::task::spawn_blocking(move || {
                let conn = conn
                    .lock()
                    .map_err(|e| FlowdError::Storage(e.to_string()))?;
                conn.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?1",
                    rusqlite::params![&id],
                )
                .map_err(storage_err)?;
                conn.execute("DELETE FROM plans WHERE id = ?1", rusqlite::params![&id])
                    .map_err(storage_err)?;
                Ok(())
            })
            .await
            .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
        }
    }
}

fn row_to_summary(row: &rusqlite::Row<'_>) -> rusqlite::Result<PlanSummary> {
    let id_str: String = row.get(0)?;
    let status_str: String = row.get(2)?;
    let created_str: String = row.get(3)?;
    let project: Option<String> = row.get(4)?;

    Ok(PlanSummary {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        name: row.get(1)?,
        status: match status_str.as_str() {
            "confirmed" => PlanStatus::Confirmed,
            "running" => PlanStatus::Running,
            "completed" => PlanStatus::Completed,
            "failed" => PlanStatus::Failed,
            "cancelled" => PlanStatus::Cancelled,
            _ => PlanStatus::Draft,
        },
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map_or_else(|_| Utc::now(), |dt| dt.with_timezone(&Utc)),
        project,
    })
}

fn plan_status_str(s: PlanStatus) -> &'static str {
    match s {
        PlanStatus::Draft => "draft",
        PlanStatus::Confirmed => "confirmed",
        PlanStatus::Running => "running",
        PlanStatus::Completed => "completed",
        PlanStatus::Failed => "failed",
        PlanStatus::Cancelled => "cancelled",
    }
}

fn step_status_str(s: StepStatus) -> &'static str {
    match s {
        StepStatus::Pending => "pending",
        StepStatus::Ready => "ready",
        StepStatus::Running => "running",
        StepStatus::Completed => "completed",
        StepStatus::Failed => "failed",
        StepStatus::Skipped => "skipped",
        StepStatus::Cancelled => "cancelled",
    }
}
