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

use super::executor::AgentMetrics;
use super::observer::PlanEvent;

/// Insert a `"metrics"` field into `payload` when `metrics` is
/// `Some`. A `None` stays absent from the JSON rather than rendering
/// as `"metrics": null` -- consumers distinguish "no metrics reported"
/// from "metrics explicitly null".
fn attach_metrics(payload: &mut JsonValue, metrics: Option<&AgentMetrics>) {
    let Some(m) = metrics else {
        return;
    };
    let Ok(value) = serde_json::to_value(m) else {
        return;
    };
    if let Some(obj) = payload.as_object_mut() {
        obj.insert("metrics".into(), value);
    }
}

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
        PlanEvent::StepCompleted {
            output, metrics, ..
        } => {
            let mut payload = json!({ "output": output });
            attach_metrics(&mut payload, metrics.as_ref());
            payload
        }
        PlanEvent::StepFailed { error, metrics, .. } => {
            let mut payload = json!({ "error": error });
            attach_metrics(&mut payload, metrics.as_ref());
            payload
        }
        PlanEvent::StepRefused { reason, .. } => json!({ "reason": reason }),
        PlanEvent::Finished {
            status,
            total_metrics,
            step_count,
            elapsed_ms,
            ..
        } => {
            let mut payload = json!({
                "status": format!("{status:?}").to_lowercase(),
                "step_count": {
                    "completed": step_count.completed,
                    "failed": step_count.failed,
                },
            });
            attach_metrics(&mut payload, total_metrics.as_ref());
            // Wall-clock span; absent rather than `null` so older event
            // rows and "never executed" transitions render the rollup
            // without the elapsed segment.
            if let Some(ms) = elapsed_ms {
                if let Some(obj) = payload.as_object_mut() {
                    obj.insert("elapsed_ms".into(), json!(ms));
                }
            }
            payload
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
                    metrics: None,
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
                    metrics: None,
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
                    total_metrics: None,
                    step_count: super::super::observer::PlanStepCounts::default(),
                    elapsed_ms: None,
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
            metrics: None,
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
            total_metrics: None,
            step_count: super::super::observer::PlanStepCounts::default(),
            elapsed_ms: None,
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["status"], "failed");
        assert_eq!(payload["step_count"]["completed"], 0);
        assert_eq!(payload["step_count"]["failed"], 0);
        let obj = payload.as_object().expect("payload is an object");
        assert!(
            !obj.contains_key("metrics"),
            "metrics key must be absent when total_metrics is None; got {payload}"
        );
        assert!(
            !obj.contains_key("elapsed_ms"),
            "elapsed_ms key must be absent when None; got {payload}"
        );
    }

    #[test]
    fn payload_finished_serializes_rollup_metrics_and_counts() {
        let metrics = AgentMetrics {
            input_tokens: 100,
            output_tokens: 50,
            total_cost_usd: 0.42,
            duration_ms: 1_500,
            duration_api_ms: 1_200,
            ..AgentMetrics::default()
        };
        let evt = PlanEvent::Finished {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            status: PlanStatus::Completed,
            total_metrics: Some(metrics),
            step_count: super::super::observer::PlanStepCounts {
                completed: 3,
                failed: 1,
            },
            elapsed_ms: Some(900),
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["status"], "completed");
        assert_eq!(payload["step_count"]["completed"], 3);
        assert_eq!(payload["step_count"]["failed"], 1);
        assert_eq!(payload["metrics"]["input_tokens"], 100);
        assert_eq!(payload["metrics"]["output_tokens"], 50);
        assert_eq!(payload["metrics"]["total_cost_usd"], 0.42);
        // Elapsed gets serialised when present; the per-step duration sum
        // (1500ms here) lives inside `metrics.duration_ms` and is independent.
        assert_eq!(payload["elapsed_ms"], 900);
    }

    #[test]
    fn payload_step_completed_carries_output() {
        let evt = PlanEvent::StepCompleted {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "a".into(),
            agent_type: "echo".into(),
            output: "hello".into(),
            metrics: None,
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["output"], "hello");
        // `None` metrics must be *absent* rather than serialized as
        // `null`: consumers distinguish "no metrics reported" from
        // "metrics explicitly null".
        assert!(
            !payload
                .as_object()
                .expect("payload is an object")
                .contains_key("metrics"),
            "metrics key must be absent when None; got {payload}"
        );
    }

    #[test]
    fn payload_step_completed_serializes_metrics_when_present() {
        let mut metrics = AgentMetrics {
            input_tokens: 10,
            output_tokens: 5,
            total_cost_usd: 0.25,
            duration_ms: 500,
            ..AgentMetrics::default()
        };
        metrics.model_usage.insert(
            "sonnet".into(),
            crate::orchestration::executor::ModelUsage {
                input_tokens: 10,
                output_tokens: 5,
                cost_usd: 0.25,
                ..Default::default()
            },
        );
        let evt = PlanEvent::StepCompleted {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "a".into(),
            agent_type: "echo".into(),
            output: "hello".into(),
            metrics: Some(metrics),
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["output"], "hello");
        assert_eq!(payload["metrics"]["input_tokens"], 10);
        assert_eq!(payload["metrics"]["output_tokens"], 5);
        assert_eq!(payload["metrics"]["total_cost_usd"], 0.25);
        assert_eq!(payload["metrics"]["duration_ms"], 500);
        assert_eq!(
            payload["metrics"]["model_usage"]["sonnet"]["input_tokens"],
            10
        );
    }

    #[test]
    fn payload_step_failed_carries_error() {
        let evt = PlanEvent::StepFailed {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "a".into(),
            agent_type: "echo".into(),
            error: "boom".into(),
            metrics: None,
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["error"], "boom");
        assert!(
            !payload
                .as_object()
                .expect("payload is an object")
                .contains_key("metrics"),
            "metrics key must be absent when None; got {payload}"
        );
    }

    #[test]
    fn payload_step_failed_serializes_metrics_when_present() {
        let metrics = AgentMetrics {
            input_tokens: 7,
            output_tokens: 3,
            total_cost_usd: 0.004,
            duration_ms: 250,
            duration_api_ms: 240,
            ..AgentMetrics::default()
        };
        let evt = PlanEvent::StepFailed {
            plan_id: Uuid::nil(),
            project: "demo".into(),
            step_id: "a".into(),
            agent_type: "echo".into(),
            error: "boom".into(),
            metrics: Some(metrics),
        };
        let payload = event_payload(&evt);
        assert_eq!(payload["error"], "boom");
        assert_eq!(payload["metrics"]["input_tokens"], 7);
        assert_eq!(payload["metrics"]["output_tokens"], 3);
        assert_eq!(payload["metrics"]["total_cost_usd"], 0.004);
        assert_eq!(payload["metrics"]["duration_ms"], 250);
        assert_eq!(payload["metrics"]["duration_api_ms"], 240);
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
