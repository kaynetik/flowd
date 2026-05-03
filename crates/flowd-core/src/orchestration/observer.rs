//! Plan-lifecycle observer hook.
//!
//! Plan-lifecycle events are first-class telemetry: the executor calls
//! [`PlanObserver::on_event`] at every well-defined transition, and an
//! external adapter (e.g. `flowd_mcp::PlanEventObserver`) durably persists
//! them into the dedicated `plan_events` table (HL-39). Memory is for
//! semantic recall; this surface is for operational audit trails.
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

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::PlanStatus;
use super::executor::AgentMetrics;
use super::integration::{IntegrationMode, IntegrationStatus};

/// Per-terminal-outcome counts rolled up on the `Finished` event.
///
/// Kept as a dedicated struct (rather than a loose `(u32, u32)`) so the
/// renderer and any future consumer stay type-safe and self-documenting.
/// `completed` counts every step that ended with a `StepCompleted`
/// event; `failed` counts every step that ended with a `StepFailed`
/// event (retries exhausted, panic) or was refused by the rule gate
/// (which settles into the plan's `Failed` terminal status). Cancelled
/// steps are excluded: the plan's terminal status already reflects
/// cancellation.
#[derive(Debug, Default, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub struct PlanStepCounts {
    pub completed: u32,
    pub failed: u32,
}

/// Lifecycle event emitted by the executor at well-defined transitions.
///
/// Every variant carries a non-optional `project` string -- mirroring the
/// `Plan.project` invariant. Adapters can rely on it being a usable scope
/// key without nil-checks.
#[derive(Debug, Clone)]
pub enum PlanEvent {
    /// A plan has been validated and registered (post-`submit`).
    Submitted {
        plan_id: Uuid,
        name: String,
        project: String,
    },
    /// `execute` has begun draining DAG layers.
    Started { plan_id: Uuid, project: String },
    /// A step has been accepted by every executor-side gate and is about
    /// to spawn. Emitted *after* the step's `PlanStatus::Running` snapshot
    /// has been mirrored into the in-memory runtime and persisted, so any
    /// observer that re-queries `status()` on receipt sees the same
    /// timestamp the event carries. Paired with the eventual
    /// `StepCompleted` / `StepFailed` / `StepCancelled` outcome; consumers
    /// can use the gap between `StepStarted` and its terminal outcome to
    /// surface in-flight steps without polling the plan snapshot.
    StepStarted {
        plan_id: Uuid,
        project: String,
        step_id: String,
        agent_type: String,
        /// Wall-clock moment the executor flipped this step to Running.
        /// Matches [`super::PlanStep::started_at`] verbatim and is the
        /// `t0` for any per-step duration the eventual `StepCompleted`
        /// / `StepFailed` consumer wants to compute.
        started_at: DateTime<Utc>,
    },
    /// A step finished successfully and its output was captured.
    StepCompleted {
        plan_id: Uuid,
        project: String,
        step_id: String,
        agent_type: String,
        /// May be truncated by the adapter for storage.
        output: String,
        /// Token / cost usage captured alongside the successful run, if
        /// the spawner reported any. Adapters may persist this verbatim
        /// in the event payload; `None` means the spawner did not
        /// produce metrics (e.g. non-JSON passthrough).
        metrics: Option<AgentMetrics>,
    },
    /// A step failed after exhausting retries (or never ran due to a panic).
    StepFailed {
        plan_id: Uuid,
        project: String,
        step_id: String,
        agent_type: String,
        error: String,
        /// Token / cost usage from the last failing attempt, when the
        /// spawner attached metrics to [`FlowdError::PlanExecution`].
        /// Failed steps still cost money, so the audit log keeps the
        /// spend visible.
        metrics: Option<AgentMetrics>,
    },
    /// A step was refused by the executor's gate (currently the rule gate).
    StepRefused {
        plan_id: Uuid,
        project: String,
        step_id: String,
        agent_type: String,
        reason: String,
    },
    /// A step was aborted by user-initiated `cancel`.
    StepCancelled {
        plan_id: Uuid,
        project: String,
        step_id: String,
        agent_type: String,
    },
    /// `execute` has finished -- success, failure, or cancellation.
    ///
    /// `total_metrics` is the `AgentMetrics::merge`-accumulated sum of
    /// every per-step metrics payload the spawner reported during this
    /// run. `None` means the plan did not run any step that attached
    /// metrics (e.g. a cancel from `Draft`). `step_count` summarises the
    /// per-outcome counts so the renderer does not have to re-walk the
    /// step list.
    ///
    /// `elapsed_ms` is the wall-clock span between `Plan.started_at` and
    /// the moment this event is emitted. It is `None` for transitions
    /// that never actually executed (cancel from `Draft`/`Confirmed`)
    /// or that span a daemon restart (rehydrate-as-`Interrupted`), where
    /// the wall-clock figure would be either zero or misleading.
    /// Distinct from the summed `total_metrics.duration_ms`: with parallel
    /// steps the sum of per-step runtimes can exceed wall-clock elapsed.
    Finished {
        plan_id: Uuid,
        project: String,
        status: PlanStatus,
        total_metrics: Option<AgentMetrics>,
        step_count: PlanStepCounts,
        elapsed_ms: Option<u64>,
    },
    /// The compiler surfaced one or more new clarification questions on a
    /// `Draft` plan. Emitted from the prose-first plan-creation MCP path
    /// every time a `compile_prose` / `apply_answers` / `refine` call
    /// returns a non-empty `open_questions` set.
    ///
    /// The `question_ids` list is deterministic and matches the order the
    /// compiler emitted, so audit consumers can reconstruct the
    /// clarification timeline without diffing snapshots.
    ClarificationOpened {
        plan_id: Uuid,
        project: String,
        question_ids: Vec<String>,
    },
    /// One or more clarification questions crystallised into
    /// [`super::DecisionRecord`]s. Emitted whenever `apply_answers` or
    /// `compile_prose` returns a non-empty `new_decisions` set.
    ClarificationResolved {
        plan_id: Uuid,
        project: String,
        decision_ids: Vec<String>,
    },
    /// The user submitted a freeform refinement and the compiler accepted
    /// it. The truncated `feedback_summary` lets operators correlate plan
    /// shape changes with the prompt that triggered them without bloating
    /// the event log.
    RefinementApplied {
        plan_id: Uuid,
        project: String,
        feedback_summary: String,
    },
    /// Integration of a [`PlanStatus::Completed`] plan has been
    /// initiated. Emitted exactly once per `plan_integrate` invocation
    /// that gets past the eligibility check, before any git work runs.
    /// Distinct from [`Self::Started`], which is the *plan*'s execution
    /// transition.
    IntegrationStarted {
        plan_id: Uuid,
        project: String,
        integration_branch: String,
        base_branch: String,
        mode: IntegrationMode,
    },
    /// An integration run finished without a failure. `status` carries
    /// the post-run [`IntegrationStatus`]: `Staged` for a confirm-mode
    /// run that produced an integration branch awaiting promotion;
    /// `Promoted` when the fast-forward to base completed.
    /// `promoted_tip` is `Some` only for `Promoted`.
    IntegrationSucceeded {
        plan_id: Uuid,
        project: String,
        integration_branch: String,
        base_branch: String,
        status: IntegrationStatus,
        promoted_tip: Option<String>,
    },
    /// An integration run failed. `reason` is a one-line, human-readable
    /// summary suitable for log lines and terse UIs; the structured
    /// failure / refusal lives on the plan's
    /// [`super::integration::IntegrationMetadata`] so consumers that
    /// need a stable error code read it from there rather than
    /// re-parsing the reason.
    IntegrationFailed {
        plan_id: Uuid,
        project: String,
        integration_branch: String,
        base_branch: String,
        reason: String,
    },
}

impl PlanEvent {
    /// Plan id every variant carries; convenience for adapters.
    #[must_use]
    pub fn plan_id(&self) -> Uuid {
        match self {
            Self::Submitted { plan_id, .. }
            | Self::Started { plan_id, .. }
            | Self::StepStarted { plan_id, .. }
            | Self::StepCompleted { plan_id, .. }
            | Self::StepFailed { plan_id, .. }
            | Self::StepRefused { plan_id, .. }
            | Self::StepCancelled { plan_id, .. }
            | Self::Finished { plan_id, .. }
            | Self::ClarificationOpened { plan_id, .. }
            | Self::ClarificationResolved { plan_id, .. }
            | Self::RefinementApplied { plan_id, .. }
            | Self::IntegrationStarted { plan_id, .. }
            | Self::IntegrationSucceeded { plan_id, .. }
            | Self::IntegrationFailed { plan_id, .. } => *plan_id,
        }
    }

    /// Project every variant carries.
    #[must_use]
    pub fn project(&self) -> &str {
        match self {
            Self::Submitted { project, .. }
            | Self::Started { project, .. }
            | Self::StepStarted { project, .. }
            | Self::StepCompleted { project, .. }
            | Self::StepFailed { project, .. }
            | Self::StepRefused { project, .. }
            | Self::StepCancelled { project, .. }
            | Self::Finished { project, .. }
            | Self::ClarificationOpened { project, .. }
            | Self::ClarificationResolved { project, .. }
            | Self::RefinementApplied { project, .. }
            | Self::IntegrationStarted { project, .. }
            | Self::IntegrationSucceeded { project, .. }
            | Self::IntegrationFailed { project, .. } => project.as_str(),
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
            project: "flowd".into(),
        };
        assert_eq!(evt.plan_id(), id);
        assert_eq!(evt.project(), "flowd");
    }

    #[test]
    fn project_helper_round_trips() {
        let evt = PlanEvent::Finished {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            status: PlanStatus::Completed,
            total_metrics: None,
            step_count: PlanStepCounts::default(),
            elapsed_ms: None,
        };
        assert_eq!(evt.project(), "demo");
    }
}
