//! Persistent plan storage trait.
//!
//! Implemented by `SQLite` in `flowd-storage` and by test doubles. Uses the
//! same `impl Future` return style as [`crate::memory::MemoryBackend`] —
//! no `async_trait` crate.

use std::future::Future;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::Result;

use super::{Plan, PlanStatus};

/// Summary row for [`PlanStore::list_plans`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanSummary {
    pub id: Uuid,
    pub name: String,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    /// Required project namespace (matches [`crate::orchestration::Plan::project`]).
    pub project: String,
    /// Canonical execution root captured at plan creation, if known.
    /// Mirrors [`crate::orchestration::Plan::project_root`]; legacy rows
    /// surface as `None`.
    #[serde(default)]
    pub project_root: Option<String>,
}

/// Persists orchestration [`Plan`] snapshots so the daemon can rehydrate
/// in-flight work after a restart.
pub trait PlanStore: Send + Sync {
    /// Replace or insert the given plan and its steps.
    fn save_plan(&self, plan: &Plan) -> impl Future<Output = Result<()>> + Send;

    /// Load a plan by id, if present.
    fn load_plan(&self, id: Uuid) -> impl Future<Output = Result<Option<Plan>>> + Send;

    /// List plans, optionally filtered to a single project label.
    fn list_plans(
        &self,
        project: Option<&str>,
    ) -> impl Future<Output = Result<Vec<PlanSummary>>> + Send;

    /// Remove a plan and its steps.
    fn delete_plan(&self, id: Uuid) -> impl Future<Output = Result<()>> + Send;
}

/// No-op store for executors that should not touch disk.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpPlanStore;

impl PlanStore for NoOpPlanStore {
    async fn save_plan(&self, _plan: &Plan) -> Result<()> {
        Ok(())
    }

    async fn load_plan(&self, _id: Uuid) -> Result<Option<Plan>> {
        Ok(None)
    }

    async fn list_plans(&self, _project: Option<&str>) -> Result<Vec<PlanSummary>> {
        Ok(Vec::new())
    }

    async fn delete_plan(&self, _id: Uuid) -> Result<()> {
        Ok(())
    }
}
