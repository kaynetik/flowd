//! Plan-lifecycle observer hook.
//!
//! Today, a plan run leaves no trail in flowd's memory unless the agent
//! itself chooses to call `memory_store`. This module gives the executor a
//! thin observation surface so an external adapter (e.g.
//! `MemoryPlanObserver` in `flowd-mcp`) can durably record what the
//! orchestrator actually did, on every transition.
//!
//! ## Design choices
//!
//! - **Synchronous trait method.** Keeping `on_event` sync lets the
//!   executor stay free of `async_trait` and keeps the trait object-safe
//!   (`Arc<dyn PlanObserver>`). Adapters that want to do async work --
//!   write to a database, ship to telemetry -- are expected to
//!   `tokio::spawn` from inside their `on_event` impl. Fire-and-forget is
//!   the right default: the executor must not stall on a slow observer.
//! - **Owned event payloads.** Every variant owns its strings so the
//!   adapter can move them across `tokio::spawn` boundaries without
//!   borrow gymnastics in the executor.
//! - **One trait, optional install.** The executor stores
//!   `Option<Arc<dyn PlanObserver>>`; with no observer (the default), there
//!   is zero overhead per step.

use std::sync::Arc;

use uuid::Uuid;

use super::PlanStatus;

/// Lifecycle event emitted by the executor at well-defined transitions.
#[derive(Debug, Clone)]
pub enum PlanEvent {
    /// A plan has been validated and registered (post-`submit`).
    Submitted {
        plan_id: Uuid,
        name: String,
        project: Option<String>,
    },
    /// `execute` has begun draining DAG layers.
    Started {
        plan_id: Uuid,
        project: Option<String>,
    },
    /// A step finished successfully and its output was captured.
    StepCompleted {
        plan_id: Uuid,
        project: Option<String>,
        step_id: String,
        agent_type: String,
        /// May be truncated by the adapter for storage.
        output: String,
    },
    /// A step failed after exhausting retries (or never ran due to a panic).
    StepFailed {
        plan_id: Uuid,
        project: Option<String>,
        step_id: String,
        agent_type: String,
        error: String,
    },
    /// A step was refused by the executor's gate (currently the rule gate).
    StepRefused {
        plan_id: Uuid,
        project: Option<String>,
        step_id: String,
        agent_type: String,
        reason: String,
    },
    /// A step was aborted by user-initiated `cancel`.
    StepCancelled {
        plan_id: Uuid,
        project: Option<String>,
        step_id: String,
        agent_type: String,
    },
    /// `execute` has finished -- success, failure, or cancellation.
    Finished {
        plan_id: Uuid,
        project: Option<String>,
        status: PlanStatus,
    },
}

impl PlanEvent {
    /// Plan id every variant carries; convenience for adapters.
    #[must_use]
    pub fn plan_id(&self) -> Uuid {
        match self {
            Self::Submitted { plan_id, .. }
            | Self::Started { plan_id, .. }
            | Self::StepCompleted { plan_id, .. }
            | Self::StepFailed { plan_id, .. }
            | Self::StepRefused { plan_id, .. }
            | Self::StepCancelled { plan_id, .. }
            | Self::Finished { plan_id, .. } => *plan_id,
        }
    }

    /// Project every variant carries; `None` for plans without a project.
    #[must_use]
    pub fn project(&self) -> Option<&str> {
        match self {
            Self::Submitted { project, .. }
            | Self::Started { project, .. }
            | Self::StepCompleted { project, .. }
            | Self::StepFailed { project, .. }
            | Self::StepRefused { project, .. }
            | Self::StepCancelled { project, .. }
            | Self::Finished { project, .. } => project.as_deref(),
        }
    }
}

/// Observer of plan-lifecycle events.
///
/// Implementations should treat `on_event` as best-effort, idempotent, and
/// fast. Heavy work (I/O, embedding, network) belongs in a `tokio::spawn`'d
/// task started from inside the impl; the executor must not be blocked.
pub trait PlanObserver: Send + Sync {
    fn on_event(&self, event: PlanEvent);
}

/// Convenience alias for the boxed observer the executor stores.
pub type SharedPlanObserver = Arc<dyn PlanObserver>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn project_helper_extracts_payload() {
        let id = Uuid::nil();
        let evt = PlanEvent::Started {
            plan_id: id,
            project: Some("flowd".into()),
        };
        assert_eq!(evt.plan_id(), id);
        assert_eq!(evt.project(), Some("flowd"));
    }

    #[test]
    fn project_helper_handles_missing_project() {
        let evt = PlanEvent::Finished {
            plan_id: Uuid::nil(),
            project: None,
            status: PlanStatus::Completed,
        };
        assert!(evt.project().is_none());
    }
}
