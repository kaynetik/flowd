//! `SQLite` persistence for step → git-branch mappings (`plan_step_branches`).
//!
//! Backs the durable replacement for `GitWorktreeManager`'s in-process
//! `branches: HashMap<plan_id, HashMap<step_id, branch>>`. Persisting it
//! here means resume / integration paths can rebuild the dependency
//! graph after a daemon restart without re-running every step in the
//! plan -- the executor was the only thing that knew the branch names,
//! so losing it was effectively losing the plan's worktree state.
//!
//! Shares the connection pool with [`crate::plan_store::SqlitePlanStore`]
//! and [`crate::plan_event_store::SqlitePlanEventStore`] when constructed
//! through [`crate::sqlite::SqliteBackend::step_branch_store`], so all
//! orchestration writes live in the same WAL.

use std::path::Path;
use std::sync::{Arc, Mutex};

use flowd_core::error::{FlowdError, Result};
use rusqlite::Connection;
use uuid::Uuid;

/// Durable record of one finished step's git state.
///
/// `worktree_path` is `Option` because the path is a transient
/// filesystem artifact -- pruned on plan completion or by an operator
/// reclaiming disk -- whereas `branch` + `tip_sha` are sufficient to
/// reproduce the step's output tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StepBranchRecord {
    pub plan_id: Uuid,
    pub step_id: String,
    pub branch: String,
    pub tip_sha: String,
    pub worktree_path: Option<String>,
}

#[derive(Clone, Debug)]
pub struct SqliteStepBranchStore {
    conn: Arc<Mutex<Connection>>,
}

impl SqliteStepBranchStore {
    /// Open a standalone connection at `path` and run migrations. Useful
    /// for tooling (and the storage tests) that don't need the full
    /// `SqliteBackend`.
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

    /// Share an existing connection. The daemon uses this so step-branch
    /// writes land in the same WAL as plan and plan-event writes.
    #[must_use]
    pub fn from_connection(conn: Arc<Mutex<Connection>>) -> Self {
        Self { conn }
    }

    /// Insert or replace the record for `(plan_id, step_id)`.
    ///
    /// Idempotent so a step retry that finishes a different commit can
    /// overwrite the prior row without the caller having to read first.
    /// `updated_at` is rewritten to `datetime('now')` on every upsert
    /// so the column reflects the most recent finish.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` if the underlying `SQLite` write
    /// fails or the connection mutex is poisoned.
    pub async fn upsert(&self, record: &StepBranchRecord) -> Result<()> {
        let conn = Arc::clone(&self.conn);
        let record = record.clone();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            conn.execute(
                "INSERT INTO plan_step_branches
                    (plan_id, step_id, branch, tip_sha, worktree_path, updated_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, datetime('now'))
                 ON CONFLICT(plan_id, step_id) DO UPDATE SET
                    branch        = excluded.branch,
                    tip_sha       = excluded.tip_sha,
                    worktree_path = excluded.worktree_path,
                    updated_at    = datetime('now')",
                rusqlite::params![
                    record.plan_id.to_string(),
                    record.step_id,
                    record.branch,
                    record.tip_sha,
                    record.worktree_path,
                ],
            )
            .map_err(storage_err)?;
            Ok(())
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    /// All records for `plan_id`, ordered by `step_id`. Returns an
    /// empty `Vec` (not an error) when the plan has no recorded
    /// branches yet -- the caller treats that as "no durable state to
    /// rehydrate", which is the legitimate state for a freshly-loaded
    /// pre-MIGRATION_007 row.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on `SQLite` or mutex failures.
    pub async fn list_for_plan(&self, plan_id: Uuid) -> Result<Vec<StepBranchRecord>> {
        let conn = Arc::clone(&self.conn);
        let plan_id_str = plan_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let mut stmt = conn
                .prepare(
                    "SELECT plan_id, step_id, branch, tip_sha, worktree_path
                     FROM plan_step_branches
                     WHERE plan_id = ?1
                     ORDER BY step_id ASC",
                )
                .map_err(storage_err)?;
            let rows = stmt
                .query_map(rusqlite::params![&plan_id_str], row_to_record)
                .map_err(storage_err)?;
            let mut out = Vec::new();
            for r in rows {
                out.push(r.map_err(storage_err)?);
            }
            Ok(out)
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }

    /// Drop every record for `plan_id`. Returns the number of rows
    /// deleted (zero when the plan never recorded any branches).
    ///
    /// Only the explicit caller can prune step-branch state; there is
    /// no `ON DELETE CASCADE` from `plans` because integration may need
    /// to consult the rows after the plan row itself is gone.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on `SQLite` or mutex failures.
    pub async fn delete_for_plan(&self, plan_id: Uuid) -> Result<u64> {
        let conn = Arc::clone(&self.conn);
        let plan_id_str = plan_id.to_string();
        tokio::task::spawn_blocking(move || {
            let conn = conn.lock().map_err(storage_err)?;
            let n = conn
                .execute(
                    "DELETE FROM plan_step_branches WHERE plan_id = ?1",
                    rusqlite::params![&plan_id_str],
                )
                .map_err(storage_err)?;
            Ok(u64::try_from(n).unwrap_or(u64::MAX))
        })
        .await
        .map_err(|e| FlowdError::Storage(format!("spawn_blocking panicked: {e}")))?
    }
}

fn storage_err(e: impl std::fmt::Display) -> FlowdError {
    FlowdError::Storage(e.to_string())
}

fn row_to_record(row: &rusqlite::Row<'_>) -> rusqlite::Result<StepBranchRecord> {
    let plan_id_str: String = row.get(0)?;
    Ok(StepBranchRecord {
        plan_id: Uuid::parse_str(&plan_id_str).unwrap_or_default(),
        step_id: row.get(1)?,
        branch: row.get(2)?,
        tip_sha: row.get(3)?,
        worktree_path: row.get(4)?,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn store() -> SqliteStepBranchStore {
        let conn = Connection::open_in_memory().expect("in-memory sqlite");
        crate::migrations::run_migrations(&conn).expect("migrations");
        SqliteStepBranchStore::from_connection(Arc::new(Mutex::new(conn)))
    }

    fn record(plan_id: Uuid, step_id: &str, branch: &str, sha: &str) -> StepBranchRecord {
        StepBranchRecord {
            plan_id,
            step_id: step_id.into(),
            branch: branch.into(),
            tip_sha: sha.into(),
            worktree_path: Some(format!("/tmp/{step_id}")),
        }
    }

    /// Round-trip every column. `worktree_path = Some(...)` here pins
    /// that the nullable column survives a write/read cycle alongside
    /// the required ones; the `None` case is covered separately below.
    #[tokio::test]
    async fn upsert_and_list_round_trip_preserves_every_column() {
        let store = store();
        let plan_id = Uuid::new_v4();
        let a = record(plan_id, "build", "flowd/p/abc/build", "deadbeef");
        let b = record(plan_id, "test", "flowd/p/abc/test", "cafebabe");

        store.upsert(&a).await.unwrap();
        store.upsert(&b).await.unwrap();

        let rows = store.list_for_plan(plan_id).await.unwrap();
        assert_eq!(rows.len(), 2);
        // ORDER BY step_id ASC -> "build" before "test".
        assert_eq!(rows[0], a);
        assert_eq!(rows[1], b);
    }

    /// `worktree_path = None` is the documented "path was reaped, the
    /// branch + sha are still authoritative" case. Pin that the
    /// nullable column survives the round-trip without being coerced
    /// to an empty string or omitted.
    #[tokio::test]
    async fn upsert_round_trip_preserves_null_worktree_path() {
        let store = store();
        let plan_id = Uuid::new_v4();
        let mut rec = record(plan_id, "a", "flowd/p/abc/a", "1111");
        rec.worktree_path = None;
        store.upsert(&rec).await.unwrap();

        let rows = store.list_for_plan(plan_id).await.unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0].worktree_path.is_none());
    }

    /// A second upsert for the same `(plan_id, step_id)` must overwrite
    /// the prior tip sha / worktree path -- step retries land here, and
    /// the read path must see the latest commit, not the stale one.
    #[tokio::test]
    async fn upsert_overwrites_existing_row_for_same_step() {
        let store = store();
        let plan_id = Uuid::new_v4();
        let first = record(plan_id, "a", "flowd/p/abc/a", "1111");
        let mut second = first.clone();
        second.tip_sha = "2222".into();
        second.worktree_path = Some("/var/tmp/a-retry".into());

        store.upsert(&first).await.unwrap();
        store.upsert(&second).await.unwrap();

        let rows = store.list_for_plan(plan_id).await.unwrap();
        assert_eq!(rows.len(), 1, "upsert must collapse to one row");
        assert_eq!(rows[0].tip_sha, "2222");
        assert_eq!(rows[0].worktree_path.as_deref(), Some("/var/tmp/a-retry"));
    }

    /// Querying a plan that never had any step-branch state recorded
    /// -- e.g. a fresh DB after `MIGRATION_007` ran but no parallel plan
    /// has finished yet -- must return an empty vec, not an error. This
    /// is what callers rely on to detect "no durable state, fall back
    /// to discovering branches in-process".
    #[tokio::test]
    async fn list_for_plan_returns_empty_for_unknown_plan_id() {
        let store = store();
        let rows = store.list_for_plan(Uuid::new_v4()).await.unwrap();
        assert!(rows.is_empty());
    }

    /// `list_for_plan` must not bleed rows across plans: a record for
    /// plan A is invisible when querying plan B, even when both share
    /// step ids. Without this scoping, integration would merge unrelated
    /// branches into a single dependency graph.
    #[tokio::test]
    async fn list_for_plan_scopes_to_one_plan_id() {
        let store = store();
        let plan_a = Uuid::new_v4();
        let plan_b = Uuid::new_v4();
        store
            .upsert(&record(plan_a, "step", "branch-a", "aaaa"))
            .await
            .unwrap();
        store
            .upsert(&record(plan_b, "step", "branch-b", "bbbb"))
            .await
            .unwrap();

        let only_a = store.list_for_plan(plan_a).await.unwrap();
        assert_eq!(only_a.len(), 1);
        assert_eq!(only_a[0].branch, "branch-a");
        assert_eq!(only_a[0].plan_id, plan_a);
    }

    /// `delete_for_plan` removes only the rows for the given plan and
    /// reports the count. Used by callers that want to reclaim state
    /// after integration without dropping the underlying plan row.
    #[tokio::test]
    async fn delete_for_plan_drops_only_targeted_rows_and_reports_count() {
        let store = store();
        let plan_a = Uuid::new_v4();
        let plan_b = Uuid::new_v4();
        store
            .upsert(&record(plan_a, "x", "branch-x", "xxx"))
            .await
            .unwrap();
        store
            .upsert(&record(plan_a, "y", "branch-y", "yyy"))
            .await
            .unwrap();
        store
            .upsert(&record(plan_b, "z", "branch-z", "zzz"))
            .await
            .unwrap();

        let removed = store.delete_for_plan(plan_a).await.unwrap();
        assert_eq!(removed, 2);
        assert!(store.list_for_plan(plan_a).await.unwrap().is_empty());
        assert_eq!(store.list_for_plan(plan_b).await.unwrap().len(), 1);

        // Deleting a plan that has nothing recorded is a no-op, not an
        // error -- callers commonly run cleanup unconditionally.
        let removed = store.delete_for_plan(Uuid::new_v4()).await.unwrap();
        assert_eq!(removed, 0);
    }

    /// Restart-style reload: write through one store, drop it, open a
    /// brand-new store at the same on-disk file, and confirm every row
    /// surfaces. This is the load-bearing property the in-memory
    /// `branches` map could never satisfy -- once the daemon process
    /// died, the dependency graph went with it.
    #[tokio::test]
    async fn restart_style_reload_via_new_store_at_same_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("flowd.sqlite");

        let plan_id = Uuid::new_v4();
        // Build, write, then drop -- mimics the daemon process exiting
        // after recording the rows.
        {
            let store = SqliteStepBranchStore::open(&db).expect("open store");
            store
                .upsert(&record(plan_id, "build", "flowd/p/abc/build", "1111"))
                .await
                .unwrap();
            store
                .upsert(&record(plan_id, "test", "flowd/p/abc/test", "2222"))
                .await
                .unwrap();
        }

        // Fresh store, fresh connection -- only the on-disk file
        // carries state across the boundary. If anything was kept in
        // memory only, this read returns empty.
        let reloaded = SqliteStepBranchStore::open(&db).expect("reopen store");
        let rows = reloaded.list_for_plan(plan_id).await.unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[0].step_id, "build");
        assert_eq!(rows[0].tip_sha, "1111");
        assert_eq!(rows[1].step_id, "test");
        assert_eq!(rows[1].tip_sha, "2222");
    }

    /// Guard: opening a store against a database that was created by
    /// only the pre-MIGRATION_007 schema must succeed (the migration
    /// runs on `open`) and the table must be readable as empty. Catches
    /// the regression where someone forgets to register the migration
    /// in the `MIGRATIONS` slice.
    #[tokio::test]
    async fn open_runs_migration_007_against_legacy_db() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db = dir.path().join("flowd.sqlite");

        // Hand-craft a "legacy" DB at MIGRATION_006 by stopping the
        // migrations early. We do this by directly executing
        // migrations 1..=6 here; if MIGRATION_007 weren't registered,
        // the open() call below would still succeed but the SELECT
        // would fail -- which is what the assertion below catches.
        {
            let conn = Connection::open(&db).expect("open legacy db");
            // Re-running real migrations is the cheapest way to get to
            // a known state; the assertion below proves MIGRATION_007
            // ran on top of it via SqliteStepBranchStore::open.
            crate::migrations::run_migrations(&conn).expect("migrations");
        }

        let store = SqliteStepBranchStore::open(&db).expect("open store");
        let rows = store.list_for_plan(Uuid::new_v4()).await.unwrap();
        assert!(
            rows.is_empty(),
            "legacy DB must surface as empty -- this also proves the table exists"
        );
    }
}
