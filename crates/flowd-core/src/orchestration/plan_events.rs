//! Persistent plan-lifecycle event log.
//!
//! Plan events ([`PlanEvent`]) are operational telemetry: the executor
//! emits one at every well-defined transition (submitted, started, per-step
//! outcomes, finished). They are deliberately *not* memory observations:
//!
//! * **Different purpose.** Memory is for semantic recall (FTS5 + ANN
//!   hybrid). Plan events are an audit log; conflating them pollutes hybrid
//!   search results and wastes the embedding budget on structured logging.
//! * **Different retention.** Events deserve a longer, simpler TTL than
//!   tier-aged memory.
//! * **First-class read path.** `flowd plan events <plan_id>` is more
//!   useful than `flowd search "plan started"`.
//!
//! This module owns the persistence-side trait surface only. The concrete
//! [`PlanEventStore`] implementation backed by `SQLite` lives in
//! `flowd-storage`; the executor-facing
//! [`super::observer::PlanObserver`] adapter that fans events into a store
//! lives in `flowd-mcp`.
//!
//! ## Style: `impl Future` over `async_trait`
//!
//! Matches the rest of the orchestration trait surface
//! ([`super::store::PlanStore`], [`crate::memory::MemoryBackend`]). Callers
//! that need a boxed observer keep the store generic and re-box at the
//! sync [`super::observer::PlanObserver`] seam (see
//! `flowd_mcp::observer::PlanEventObserver`).

use std::future::Future;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::error::Result;

use super::observer::PlanEvent;

/// Stable string identifiers for [`PlanEvent`] variants. Used as the
/// `kind` column in the persisted event row and as the `--kind` filter
/// vocabulary in the CLI.
pub mod kind {
    pub const SUBMITTED: &str = "submitted";
    pub const STARTED: &str = "started";
    pub const STEP_COMPLETED: &str = "step_completed";
    pub const STEP_FAILED: &str = "step_failed";
    pub const STEP_REFUSED: &str = "step_refused";
    pub const STEP_CANCELLED: &str = "step_cancelled";
    pub const FINISHED: &str = "finished";
    pub const CLARIFICATION_OPENED: &str = "clarification_opened";
    pub const CLARIFICATION_RESOLVED: &str = "clarification_resolved";
    pub const REFINEMENT_APPLIED: &str = "refinement_applied";
}

/// Map a [`PlanEvent`] variant to its persisted `kind` string.
#[must_use]
pub fn event_kind(event: &PlanEvent) -> &'static str {
    match event {
        PlanEvent::Submitted { .. } => kind::SUBMITTED,
        PlanEvent::Started { .. } => kind::STARTED,
        PlanEvent::StepCompleted { .. } => kind::STEP_COMPLETED,
        PlanEvent::StepFailed { .. } => kind::STEP_FAILED,
        PlanEvent::StepRefused { .. } => kind::STEP_REFUSED,
        PlanEvent::StepCancelled { .. } => kind::STEP_CANCELLED,
        PlanEvent::Finished { .. } => kind::FINISHED,
        PlanEvent::ClarificationOpened { .. } => kind::CLARIFICATION_OPENED,
        PlanEvent::ClarificationResolved { .. } => kind::CLARIFICATION_RESOLVED,
        PlanEvent::RefinementApplied { .. } => kind::REFINEMENT_APPLIED,
    }
}

/// Optional `step_id` carried by per-step variants; `None` for plan-scoped
/// events. Centralised so the storage adapter and any future consumer
/// agree on which variants populate the column.
#[must_use]
pub fn event_step_id(event: &PlanEvent) -> Option<&str> {
    match event {
        PlanEvent::StepCompleted { step_id, .. }
        | PlanEvent::StepFailed { step_id, .. }
        | PlanEvent::StepRefused { step_id, .. }
        | PlanEvent::StepCancelled { step_id, .. } => Some(step_id.as_str()),
        PlanEvent::Submitted { .. }
        | PlanEvent::Started { .. }
        | PlanEvent::Finished { .. }
        | PlanEvent::ClarificationOpened { .. }
        | PlanEvent::ClarificationResolved { .. }
        | PlanEvent::RefinementApplied { .. } => None,
    }
}

/// Optional `agent_type` carried by per-step variants.
#[must_use]
pub fn event_agent_type(event: &PlanEvent) -> Option<&str> {
    match event {
        PlanEvent::StepCompleted { agent_type, .. }
        | PlanEvent::StepFailed { agent_type, .. }
        | PlanEvent::StepRefused { agent_type, .. }
        | PlanEvent::StepCancelled { agent_type, .. } => Some(agent_type.as_str()),
        PlanEvent::Submitted { .. }
        | PlanEvent::Started { .. }
        | PlanEvent::Finished { .. }
        | PlanEvent::ClarificationOpened { .. }
        | PlanEvent::ClarificationResolved { .. }
        | PlanEvent::RefinementApplied { .. } => None,
    }
}

/// JSON payload persisted alongside each event. Captures variant-specific
/// fields not promoted to dedicated columns (output, error, reason,
/// final status, plan name).
#[must_use]
pub fn event_payload(event: &PlanEvent) -> JsonValue {
    use serde_json::json;

    match event {
        PlanEvent::Submitted { name, .. } => json!({ "name": name }),
        PlanEvent::Started { .. } | PlanEvent::StepCancelled { .. } => json!({}),
        PlanEvent::StepCompleted { output, .. } => json!({ "output": output }),
        PlanEvent::StepFailed { error, .. } => json!({ "error": error }),
        PlanEvent::StepRefused { reason, .. } => json!({ "reason": reason }),
        PlanEvent::Finished { status, .. } => {
            json!({ "status": format!("{status:?}").to_lowercase() })
        }
        PlanEvent::ClarificationOpened { question_ids, .. } => {
            json!({ "question_ids": question_ids })
        }
        PlanEvent::ClarificationResolved { decision_ids, .. } => {
            json!({ "decision_ids": decision_ids })
        }
        PlanEvent::RefinementApplied {
            feedback_summary, ..
        } => json!({ "feedback_summary": feedback_summary }),
    }
}

/// A row read back from a [`PlanEventStore`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredPlanEvent {
    pub id: i64,
    pub plan_id: Uuid,
    pub project: String,
    pub kind: String,
    pub step_id: Option<String>,
    pub agent_type: Option<String>,
    pub payload: JsonValue,
    pub created_at: DateTime<Utc>,
}

/// Filter for [`PlanEventStore::list_for_plan`].
#[derive(Debug, Clone, Default)]
pub struct PlanEventQuery {
    /// Restrict to these event kinds. Empty means "all kinds".
    pub kinds: Vec<String>,
    /// Hard cap on rows returned.
    pub limit: usize,
}

impl PlanEventQuery {
    #[must_use]
    pub fn new(limit: usize) -> Self {
        Self {
            kinds: Vec::new(),
            limit,
        }
    }

    #[must_use]
    pub fn with_kinds<I, S>(mut self, kinds: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.kinds = kinds.into_iter().map(Into::into).collect();
        self
    }
}

/// Durable sink for plan-lifecycle events.
///
/// Implementations should be cheap to clone (or wrap in [`std::sync::Arc`])
/// and tolerate concurrent `record` calls from many spawned observer tasks.
pub trait PlanEventStore: Send + Sync {
    /// Append a new event row.
    fn record(&self, event: &PlanEvent) -> impl Future<Output = Result<()>> + Send;

    /// Fetch persisted events for a plan in `created_at, id` ascending order.
    fn list_for_plan(
        &self,
        plan_id: Uuid,
        query: PlanEventQuery,
    ) -> impl Future<Output = Result<Vec<StoredPlanEvent>>> + Send;
}

#[cfg(test)]
mod tests {
    use super::super::PlanStatus;
    use super::*;

    #[test]
    fn kinds_cover_every_variant() {
        let plan_id = Uuid::nil();
        let project = "demo".to_owned();
        let cases = [
            (
                PlanEvent::Submitted {
                    plan_id,
                    name: "p".into(),
                    project: project.clone(),
                },
                kind::SUBMITTED,
            ),
            (
                PlanEvent::Started {
                    plan_id,
                    project: project.clone(),
                },
                kind::STARTED,
            ),
            (
                PlanEvent::StepCompleted {
                    plan_id,
                    project: project.clone(),
                    step_id: "a".into(),
                    agent_type: "echo".into(),
                    output: "ok".into(),
                },
                kind::STEP_COMPLETED,
            ),
            (
                PlanEvent::StepFailed {
                    plan_id,
                    project: project.clone(),
                    step_id: "a".into(),
                    agent_type: "echo".into(),
                    error: "boom".into(),
                },
                kind::STEP_FAILED,
            ),
            (
                PlanEvent::StepRefused {
                    plan_id,
                    project: project.clone(),
                    step_id: "a".into(),
                    agent_type: "echo".into(),
                    reason: "deny".into(),
                },
                kind::STEP_REFUSED,
            ),
            (
                PlanEvent::StepCancelled {
                    plan_id,
                    project: project.clone(),
                    step_id: "a".into(),
                    agent_type: "echo".into(),
                },
                kind::STEP_CANCELLED,
            ),
            (
                PlanEvent::Finished {
                    plan_id,
                    project,
                    status: PlanStatus::Completed,
                },
                kind::FINISHED,
            ),
        ];

        for (event, expected) in cases {
            assert_eq!(event_kind(&event), expected, "kind mismatch for {event:?}");
        }
    }

    #[test]
    fn step_columns_only_populated_for_step_variants() {
        let evt = PlanEvent::Started {
            plan_id: Uuid::nil(),
            project: "demo".into(),
        };
        assert!(event_step_id(&evt).is_none());
        assert!(event_agent_type(&evt).is_none());

        let step_evt = PlanEvent::StepFailed {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "build".into(),
            agent_type: "codex".into(),
            error: "x".into(),
        };
        assert_eq!(event_step_id(&step_evt), Some("build"));
        assert_eq!(event_agent_type(&step_evt), Some("codex"));
    }

    #[test]
    fn payload_finished_lowercases_status() {
        let evt = PlanEvent::Finished {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            status: PlanStatus::Failed,
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["status"], "failed");
    }

    #[test]
    fn payload_step_completed_carries_output() {
        let evt = PlanEvent::StepCompleted {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "a".into(),
            agent_type: "echo".into(),
            output: "hello".into(),
        };
        assert_eq!(event_payload(&evt)["output"], "hello");
    }

    #[test]
    fn clarification_and_refinement_events_have_kinds_and_payloads() {
        let opened = PlanEvent::ClarificationOpened {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            question_ids: vec!["q1".into(), "q2".into()],
        };
        assert_eq!(event_kind(&opened), kind::CLARIFICATION_OPENED);
        assert!(event_step_id(&opened).is_none());
        assert!(event_agent_type(&opened).is_none());
        let payload = event_payload(&opened);
        assert_eq!(payload["question_ids"][0], "q1");
        assert_eq!(payload["question_ids"][1], "q2");

        let resolved = PlanEvent::ClarificationResolved {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            decision_ids: vec!["d1".into()],
        };
        assert_eq!(event_kind(&resolved), kind::CLARIFICATION_RESOLVED);
        assert_eq!(event_payload(&resolved)["decision_ids"][0], "d1");

        let refined = PlanEvent::RefinementApplied {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            feedback_summary: "less yak shaving".into(),
        };
        assert_eq!(event_kind(&refined), kind::REFINEMENT_APPLIED);
        assert_eq!(
            event_payload(&refined)["feedback_summary"],
            "less yak shaving"
        );
    }
}
