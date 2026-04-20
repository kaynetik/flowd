//! `MemoryService`-backed implementation of [`PlanObserver`].
//!
//! This adapter is what closes the gap of "plan runs leave no trail in
//! flowd's memory unless the agent itself calls `memory_store`". Once the
//! daemon installs a `MemoryPlanObserver` on the executor, every lifecycle
//! transition (plan submitted, step completed/failed/refused/cancelled,
//! plan finished) is durably recorded as an observation, which makes it
//! searchable through the same hybrid (FTS5 + ANN) machinery as everything
//! else in flowd's memory.
//!
//! ## Session synthesis
//!
//! `MemoryService::record` requires both a `project` and a `session_id`.
//! Plans carry an optional project but no session, so this adapter:
//!
//! - **Skips events with no project.** Without a project we cannot satisfy
//!   the storage FK; logging via `tracing::debug!` makes the skip visible.
//! - **Synthesises a deterministic session per plan** via
//!   `Uuid::new_v5(NAMESPACE_OID, plan_id.as_bytes())`. Every event from the
//!   same plan thus lands under the same session row, so listing the
//!   session in `flowd history` shows the full execution trail.
//!
//! ## Async work via `tokio::spawn`
//!
//! `PlanObserver::on_event` is synchronous so the executor's hot path stays
//! `async_trait`-free and dyn-friendly. Storage and embedding I/O are real
//! work, so each event is handed off to a `tokio::spawn`'d task. The
//! adapter is fire-and-forget by design: a slow or broken memory backend
//! must never stall the executor. Failures are logged via `tracing` and
//! dropped.

use std::sync::Arc;

use flowd_core::error::FlowdError;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, MemoryBackend, VectorIndex};
use flowd_core::orchestration::observer::{PlanEvent, PlanObserver};
use serde_json::{Value as JsonValue, json};
use uuid::Uuid;

/// Truncate a step's captured output before storing it as an observation.
/// The full output stays on the `PlanStep` itself; the memory observation
/// is meant to be a searchable summary, not a verbatim log dump.
const MAX_OUTPUT_BYTES: usize = 4 * 1024;

/// Adapter turning [`PlanEvent`]s into [`MemoryService::record`] calls.
pub struct MemoryPlanObserver<M, V, E>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
{
    service: Arc<MemoryService<M, V, E>>,
}

impl<M, V, E> MemoryPlanObserver<M, V, E>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
{
    #[must_use]
    pub fn new(service: Arc<MemoryService<M, V, E>>) -> Self {
        Self { service }
    }
}

/// Synthesise a deterministic session UUID for `plan_id`. Stable across
/// daemon restarts so resumed plans append to the original session.
#[must_use]
pub fn session_for_plan(plan_id: Uuid) -> Uuid {
    Uuid::new_v5(&Uuid::NAMESPACE_OID, plan_id.as_bytes())
}

impl<M, V, E> PlanObserver for MemoryPlanObserver<M, V, E>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
{
    fn on_event(&self, event: PlanEvent) {
        let plan_id = event.plan_id();
        let Some(project) = event.project().map(str::to_owned) else {
            tracing::debug!(
                plan_id = %plan_id,
                event = ?event_kind(&event),
                "skipping plan-event memory write: plan has no project",
            );
            return;
        };
        let session_id = session_for_plan(plan_id);

        let (content, metadata) = render(&event);
        let service = Arc::clone(&self.service);

        tokio::spawn(async move {
            if let Err(e) = service.ensure_session(&project, session_id).await {
                log_record_failure(&e, plan_id, "ensure_session");
                return;
            }
            if let Err(e) = service
                .record(&project, session_id, content, Some(metadata))
                .await
            {
                log_record_failure(&e, plan_id, "record");
            }
        });
    }
}

/// Short identifier used in tracing skip-logs.
fn event_kind(event: &PlanEvent) -> &'static str {
    match event {
        PlanEvent::Submitted { .. } => "submitted",
        PlanEvent::Started { .. } => "started",
        PlanEvent::StepCompleted { .. } => "step_completed",
        PlanEvent::StepFailed { .. } => "step_failed",
        PlanEvent::StepRefused { .. } => "step_refused",
        PlanEvent::StepCancelled { .. } => "step_cancelled",
        PlanEvent::Finished { .. } => "finished",
    }
}

fn log_record_failure(e: &FlowdError, plan_id: Uuid, op: &str) {
    tracing::warn!(
        plan_id = %plan_id,
        op,
        error = %e,
        "memory plan observer dropped event due to storage failure",
    );
}

/// Format an event into the human-searchable observation body and its
/// structured metadata sidecar.
fn render(event: &PlanEvent) -> (String, JsonValue) {
    match event {
        PlanEvent::Submitted {
            plan_id,
            name,
            project: _,
        } => (
            format!("plan submitted: {name} (id={plan_id})"),
            json!({
                "kind": "plan_event",
                "event": "submitted",
                "plan_id": plan_id.to_string(),
                "plan_name": name,
            }),
        ),
        PlanEvent::Started { plan_id, .. } => (
            format!("plan started: {plan_id}"),
            json!({
                "kind": "plan_event",
                "event": "started",
                "plan_id": plan_id.to_string(),
            }),
        ),
        PlanEvent::StepCompleted {
            plan_id,
            step_id,
            agent_type,
            output,
            ..
        } => (
            format!(
                "step `{step_id}` completed via `{agent_type}`:\n{}",
                truncate_chars(output, MAX_OUTPUT_BYTES)
            ),
            json!({
                "kind": "plan_event",
                "event": "step_completed",
                "plan_id": plan_id.to_string(),
                "step_id": step_id,
                "agent_type": agent_type,
            }),
        ),
        PlanEvent::StepFailed {
            plan_id,
            step_id,
            agent_type,
            error,
            ..
        } => (
            format!("step `{step_id}` failed via `{agent_type}`: {error}"),
            json!({
                "kind": "plan_event",
                "event": "step_failed",
                "plan_id": plan_id.to_string(),
                "step_id": step_id,
                "agent_type": agent_type,
            }),
        ),
        PlanEvent::StepRefused {
            plan_id,
            step_id,
            agent_type,
            reason,
            ..
        } => (
            format!("step `{step_id}` refused (`{agent_type}`): {reason}"),
            json!({
                "kind": "plan_event",
                "event": "step_refused",
                "plan_id": plan_id.to_string(),
                "step_id": step_id,
                "agent_type": agent_type,
            }),
        ),
        PlanEvent::StepCancelled {
            plan_id,
            step_id,
            agent_type,
            ..
        } => (
            format!("step `{step_id}` cancelled (`{agent_type}`)"),
            json!({
                "kind": "plan_event",
                "event": "step_cancelled",
                "plan_id": plan_id.to_string(),
                "step_id": step_id,
                "agent_type": agent_type,
            }),
        ),
        PlanEvent::Finished {
            plan_id, status, ..
        } => (
            format!("plan finished: {plan_id} -> {status:?}"),
            json!({
                "kind": "plan_event",
                "event": "finished",
                "plan_id": plan_id.to_string(),
                "status": format!("{status:?}").to_lowercase(),
            }),
        ),
    }
}

fn truncate_chars(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let target = max_bytes.saturating_sub(16);
    let mut cut = 0usize;
    for (i, _) in s.char_indices() {
        if i > target {
            break;
        }
        cut = i;
    }
    format!("{}…[+{} bytes]", &s[..cut], s.len() - cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::PlanStatus;

    #[test]
    fn session_is_deterministic_per_plan() {
        let id = Uuid::new_v4();
        assert_eq!(session_for_plan(id), session_for_plan(id));
    }

    #[test]
    fn session_differs_across_plans() {
        assert_ne!(
            session_for_plan(Uuid::new_v4()),
            session_for_plan(Uuid::new_v4())
        );
    }

    #[test]
    fn render_step_completed_includes_agent_and_truncates_output() {
        let big = "x".repeat(MAX_OUTPUT_BYTES * 2);
        let (body, meta) = render(&PlanEvent::StepCompleted {
            plan_id: Uuid::nil(),
            project: Some("flowd".into()),
            step_id: "build".into(),
            agent_type: "codex".into(),
            output: big,
        });
        assert!(body.contains("step `build` completed via `codex`"));
        assert!(body.contains("…[+"));
        assert_eq!(meta["event"], "step_completed");
        assert_eq!(meta["agent_type"], "codex");
    }

    #[test]
    fn render_finished_lowercases_status_for_metadata() {
        let (_body, meta) = render(&PlanEvent::Finished {
            plan_id: Uuid::nil(),
            project: Some("flowd".into()),
            status: PlanStatus::Failed,
        });
        assert_eq!(meta["status"], "failed");
    }
}
