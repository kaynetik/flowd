//! `SQLite` persistence for orchestration plans (`plans` + `plan_steps`).

use std::path::Path;
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{
    DecisionRecord, OpenQuestion, Plan, PlanStatus, PlanStore, PlanSummary, StepStatus,
};
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

/// Positional row tuple returned by [`SqlitePlanStore::load_plan`]'s
/// `SELECT`: (`definition` JSON, `source_doc`, `open_questions` JSON,
/// `decisions` JSON, `definition_dirty` flag, `project_root`). Aliased
/// to keep clippy's `type_complexity` lint happy now that the row
/// carries six distinct values across two nullable columns.
type LoadPlanRow = (String, Option<String>, String, String, i64, Option<String>);

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
                // The `definition` blob remains the catch-all for any
                // future Plan field we haven't promoted yet; the four
                // clarification fields below now also live in dedicated
                // columns so they're queryable / indexable. On read we
                // trust the columns over the blob (see [`load_plan`]).
                let definition = serde_json::to_string(&plan).map_err(|e| {
                    FlowdError::Serialization(format!("persist plan {}: {e}", plan.id))
                })?;
                let open_questions_json =
                    serde_json::to_string(&plan.open_questions).map_err(|e| {
                        FlowdError::Serialization(format!(
                            "persist plan {} open_questions: {e}",
                            plan.id
                        ))
                    })?;
                let decisions_json = serde_json::to_string(&plan.decisions).map_err(|e| {
                    FlowdError::Serialization(format!("persist plan {} decisions: {e}", plan.id))
                })?;

                let tx = conn.transaction().map_err(storage_err)?;
                let id = plan.id.to_string();
                tx.execute(
                    "DELETE FROM plan_steps WHERE plan_id = ?1",
                    rusqlite::params![&id],
                )
                .map_err(storage_err)?;
                tx.execute(
                    "INSERT OR REPLACE INTO plans (
                        id, name, status, definition, created_at, started_at, completed_at, project,
                        source_doc, open_questions_json, decisions_json, definition_dirty,
                        project_root
                     ) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13)",
                    rusqlite::params![
                        &id,
                        &plan.name,
                        plan_status_str(plan.status),
                        &definition,
                        plan.created_at.to_rfc3339(),
                        plan.started_at.map(|t| t.to_rfc3339()),
                        plan.completed_at.map(|t| t.to_rfc3339()),
                        &plan.project,
                        plan.source_doc.as_deref(),
                        &open_questions_json,
                        &decisions_json,
                        i64::from(plan.definition_dirty),
                        plan.project_root.as_deref(),
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
                    .prepare(
                        "SELECT definition, source_doc, open_questions_json, decisions_json, definition_dirty, project_root
                         FROM plans WHERE id = ?1",
                    )
                    .map_err(storage_err)?;
                // Tuple per column (positional, no alias to keep the
                // statement-vs-item ordering quiet for clippy). Legacy
                // DBs (post-migration but pre-Phase-4 binary writes)
                // still load because the columns have SQL defaults so
                // every read is well-typed. `project_root` is nullable
                // (added in MIGRATION_006); rows written before that
                // surface here as `None` and the caller treats it as
                // the legacy "fall back to per-call discovery" case.
                let row: Option<LoadPlanRow> =
                    stmt.query_row([&id_str], |row| {
                        Ok((
                            row.get(0)?,
                            row.get(1)?,
                            row.get(2)?,
                            row.get(3)?,
                            row.get(4)?,
                            row.get(5)?,
                        ))
                    })
                    .optional()
                    .map_err(storage_err)?;
                let Some((
                    json,
                    source_doc,
                    open_questions_json,
                    decisions_json,
                    dirty,
                    project_root,
                )) = row
                else {
                    return Ok(None);
                };

                // Base from the JSON blob (gives us steps, status,
                // timestamps, project, etc.). The clarification fields
                // and `project_root` are then overlaid from their
                // dedicated columns so the columns are the canonical
                // source of truth even when the blob is stale (e.g.
                // written by an older binary).
                let mut plan: Plan = serde_json::from_str(&json)
                    .map_err(|e| FlowdError::Serialization(e.to_string()))?;
                plan.source_doc = source_doc;
                plan.open_questions = parse_open_questions(&open_questions_json, plan.id)?;
                plan.decisions = parse_decisions(&decisions_json, plan.id)?;
                plan.definition_dirty = dirty != 0;
                plan.project_root = project_root;

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
                            "SELECT id, name, status, created_at, project, project_root
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
                            "SELECT id, name, status, created_at, project, project_root
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
    // After migration 003 the column is NOT NULL; legacy rows backfilled
    // to '__legacy__'. Read as String directly.
    let project: String = row.get(4)?;
    // `project_root` is nullable (introduced in MIGRATION_006); rows
    // persisted before this column existed surface as `None`.
    let project_root: Option<String> = row.get(5)?;

    Ok(PlanSummary {
        id: Uuid::parse_str(&id_str).unwrap_or_default(),
        name: row.get(1)?,
        status: match status_str.as_str() {
            "confirmed" => PlanStatus::Confirmed,
            "running" => PlanStatus::Running,
            "interrupted" => PlanStatus::Interrupted,
            "completed" => PlanStatus::Completed,
            "failed" => PlanStatus::Failed,
            "cancelled" => PlanStatus::Cancelled,
            _ => PlanStatus::Draft,
        },
        created_at: DateTime::parse_from_rfc3339(&created_str)
            .map_or_else(|_| Utc::now(), |dt| dt.with_timezone(&Utc)),
        project,
        project_root,
    })
}

fn plan_status_str(s: PlanStatus) -> &'static str {
    match s {
        PlanStatus::Draft => "draft",
        PlanStatus::Confirmed => "confirmed",
        PlanStatus::Running => "running",
        PlanStatus::Interrupted => "interrupted",
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

fn parse_open_questions(json: &str, plan_id: Uuid) -> Result<Vec<OpenQuestion>> {
    serde_json::from_str(json)
        .map_err(|e| FlowdError::Serialization(format!("plan {plan_id} open_questions_json: {e}")))
}

fn parse_decisions(json: &str, plan_id: Uuid) -> Result<Vec<DecisionRecord>> {
    serde_json::from_str(json)
        .map_err(|e| FlowdError::Serialization(format!("plan {plan_id} decisions_json: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::{Plan, PlanStep, QuestionOption, StepStatus};
    use rusqlite::Connection;

    fn store() -> SqlitePlanStore {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        crate::migrations::run_migrations(&conn).expect("migrations");
        SqlitePlanStore::from_connection(Arc::new(Mutex::new(conn)))
    }

    fn step(id: &str) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: "hi".into(),
            depends_on: vec![],
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn open_question(id: &str) -> OpenQuestion {
        OpenQuestion {
            id: id.into(),
            prompt: "pick".into(),
            rationale: "test".into(),
            options: vec![QuestionOption {
                id: "x".into(),
                label: "X".into(),
                rationale: "any".into(),
            }],
            allow_explain_more: false,
            allow_none: false,
            depends_on_decisions: vec![],
        }
    }

    #[tokio::test]
    async fn save_and_load_round_trip_preserves_clarification_fields() {
        let store = store();
        let mut plan = Plan::new("p", "proj", vec![step("a")]);
        plan.source_doc = Some("# prose".into());
        plan.open_questions.push(open_question("q1"));
        plan.decisions
            .push(DecisionRecord::new_user("q0", "yes", vec![]));
        plan.definition_dirty = true;

        let id = plan.id;
        store.save_plan(&plan).await.unwrap();
        let loaded = store.load_plan(id).await.unwrap().expect("load");

        assert_eq!(loaded.source_doc.as_deref(), Some("# prose"));
        assert_eq!(loaded.open_questions.len(), 1);
        assert_eq!(loaded.open_questions[0].id, "q1");
        assert_eq!(loaded.decisions.len(), 1);
        assert_eq!(loaded.decisions[0].chosen_option_id, "yes");
        assert!(loaded.definition_dirty);
        assert_eq!(loaded.steps.len(), 1);
        assert_eq!(loaded.project, "proj");
    }

    /// Empty defaults must round-trip cleanly through both columns and
    /// JSON blob. Catches the "we forgot to write `[]` and end up with a
    /// NOT NULL violation" regression.
    #[tokio::test]
    async fn save_and_load_round_trip_with_empty_clarification_fields() {
        let store = store();
        let plan = Plan::new("p", "proj", vec![step("a")]);
        let id = plan.id;
        store.save_plan(&plan).await.unwrap();
        let loaded = store.load_plan(id).await.unwrap().expect("load");

        assert!(loaded.source_doc.is_none());
        assert!(loaded.open_questions.is_empty());
        assert!(loaded.decisions.is_empty());
        assert!(!loaded.definition_dirty);
    }

    /// Simulate a row written by a pre-Phase-4 binary: the `definition`
    /// blob has no clarification fields, but the schema columns are
    /// present (with their `MIGRATION_005` defaults). Loading must succeed
    /// and surface the column defaults rather than failing.
    #[tokio::test]
    async fn load_falls_back_to_column_defaults_for_legacy_blob() {
        let store = store();
        let plan = Plan::new("p", "proj", vec![step("a")]);
        let id = plan.id;

        // Hand-craft a "legacy" definition blob without the clarification
        // fields, then write directly through SQL so we don't rely on the
        // current Plan serializer.
        let mut json = serde_json::to_value(&plan).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("source_doc");
        obj.remove("open_questions");
        obj.remove("decisions");
        obj.remove("definition_dirty");
        let legacy_blob = serde_json::to_string(&json).unwrap();

        // Insert with column defaults so the row mirrors a real
        // post-migration / pre-write upgrade.
        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO plans (id, name, status, definition, created_at, project)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    &id.to_string(),
                    &plan.name,
                    plan_status_str(plan.status),
                    &legacy_blob,
                    plan.created_at.to_rfc3339(),
                    &plan.project,
                ],
            )
            .unwrap();

        let loaded = store.load_plan(id).await.unwrap().expect("load");
        assert!(loaded.source_doc.is_none());
        assert!(loaded.open_questions.is_empty());
        assert!(loaded.decisions.is_empty());
        assert!(!loaded.definition_dirty);
    }

    /// When the dedicated columns and the JSON blob disagree (because an
    /// older binary wrote both, then a newer binary updated only the
    /// columns), the columns win.
    #[tokio::test]
    async fn columns_take_precedence_over_blob_when_they_disagree() {
        let store = store();
        let mut plan = Plan::new("p", "proj", vec![step("a")]);
        plan.source_doc = Some("# from blob".into());
        plan.open_questions.push(open_question("blob-q"));
        let id = plan.id;
        store.save_plan(&plan).await.unwrap();

        // Mutate columns without touching the blob.
        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "UPDATE plans
                 SET source_doc = ?1,
                     open_questions_json = ?2,
                     decisions_json = ?3,
                     definition_dirty = 1
                 WHERE id = ?4",
                rusqlite::params![
                    "# from columns",
                    "[]",
                    serde_json::to_string(&[DecisionRecord::new_user("col-q", "x", vec![])])
                        .unwrap(),
                    &id.to_string(),
                ],
            )
            .unwrap();

        let loaded = store.load_plan(id).await.unwrap().expect("load");
        assert_eq!(loaded.source_doc.as_deref(), Some("# from columns"));
        assert!(loaded.open_questions.is_empty());
        assert_eq!(loaded.decisions.len(), 1);
        assert_eq!(loaded.decisions[0].question_id, "col-q");
        assert!(loaded.definition_dirty);
    }

    /// New plans persist their execution root and round-trip through
    /// the dedicated column so resume / list / show paths see the same
    /// path the daemon captured at submission.
    #[tokio::test]
    async fn save_and_load_round_trip_preserves_project_root() {
        let store = store();
        let plan = Plan::new("p", "proj", vec![step("a")]).with_project_root("/repos/flowd");
        let id = plan.id;

        store.save_plan(&plan).await.unwrap();
        let loaded = store.load_plan(id).await.unwrap().expect("load");

        assert_eq!(loaded.project_root.as_deref(), Some("/repos/flowd"));
        // The namespace label and execution root are independent, so a
        // round-trip must not collapse them into one field.
        assert_eq!(loaded.project, "proj");
    }

    /// A row written before `project_root` existed (or by an older
    /// binary that never wrote the column) must still load cleanly,
    /// surfacing `None` as the documented legacy fallback.
    #[tokio::test]
    async fn load_falls_back_to_none_for_legacy_row_without_project_root() {
        let store = store();
        let plan = Plan::new("p", "proj", vec![step("a")]);
        let id = plan.id;

        // Hand-craft a "legacy" definition blob without the field, then
        // INSERT directly so the `project_root` column stays NULL --
        // mirroring a row that was written before MIGRATION_006 ran.
        let mut json = serde_json::to_value(&plan).unwrap();
        json.as_object_mut().unwrap().remove("project_root");
        let legacy_blob = serde_json::to_string(&json).unwrap();

        store
            .conn
            .lock()
            .unwrap()
            .execute(
                "INSERT INTO plans (id, name, status, definition, created_at, project)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    &id.to_string(),
                    &plan.name,
                    plan_status_str(plan.status),
                    &legacy_blob,
                    plan.created_at.to_rfc3339(),
                    &plan.project,
                ],
            )
            .unwrap();

        let loaded = store.load_plan(id).await.unwrap().expect("load");
        assert!(
            loaded.project_root.is_none(),
            "legacy row must surface project_root = None"
        );
        assert_eq!(loaded.project, "proj");
    }

    /// `list_plans` must surface `project_root` per row -- consumers
    /// (CLI list, MCP `plan_list`) will eventually want to render it
    /// alongside the project label, and the column has to be on the
    /// summary path before that's safe to do.
    #[tokio::test]
    async fn list_plans_includes_project_root_in_summary() {
        let store = store();
        let mut alpha = Plan::new("a", "proj", vec![step("a")]).with_project_root("/repos/alpha");
        let beta = Plan::new("b", "proj", vec![step("a")]).with_project_root("/repos/beta");
        // Make ordering deterministic: alpha was created earlier.
        alpha.created_at = beta.created_at - chrono::Duration::seconds(1);

        store.save_plan(&alpha).await.unwrap();
        store.save_plan(&beta).await.unwrap();

        let summaries = store.list_plans(Some("proj")).await.unwrap();
        assert_eq!(summaries.len(), 2);
        // ORDER BY created_at DESC -> beta first.
        assert_eq!(summaries[0].project_root.as_deref(), Some("/repos/beta"));
        assert_eq!(summaries[1].project_root.as_deref(), Some("/repos/alpha"));
    }
}
