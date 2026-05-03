//! Per-layer execution runner extracted from [`super::executor`] (HL-41).
//!
//! A plan's execution loop is a sequence of *layer-runs*, where each layer is
//! a maximal set of steps whose dependencies are all settled. The main
//! executor used to inline the per-layer logic into one ~230-line `execute`
//! function carrying `#[allow(clippy::too_many_lines)]`. This module isolates
//! that logic so:
//!
//! * The executor's top-level `execute` shrinks to a readable lifecycle
//!   sketch (preflight -> loop layers -> finalise) and the lint allowance
//!   can go away.
//! * Each phase has a single, named home; future cross-cutting concerns
//!   (per-layer tracing, metrics, distributed tracing) have one obvious
//!   place to plug into.
//! * Per-layer behaviour can be reasoned about (and unit-tested) without
//!   pulling in plan persistence and the lifecycle state machine.
//!
//! `LayerRunner` is intentionally `pub(super)`-only; outside callers go
//! through [`super::executor::InMemoryPlanExecutor`].
//!
//! ## Behaviour-preserving extraction
//!
//! The semantics match pre-HL-41 `execute` exactly:
//!
//! * Steps in a layer are spawned in parallel.
//! * A configured rule gate is consulted *before* spawn; a `deny` settles
//!   the step inline as `Skipped`, emits [`PlanEvent::StepRefused`], and
//!   marks the layer failed. `warn` violations log and the step proceeds.
//! * After spawning, abort handles are published to the [`PlanRuntime`]
//!   so [`super::PlanExecutor::cancel`] can terminate the in-flight tasks.
//! * Outcomes are awaited in spawn order and mutated into `plan.steps`.
//! * `Cancelled` outcomes flip the cancellation flag (so the *next* layer
//!   does not start) but do **not** mark the layer failed -- the executor
//!   distinguishes `PlanStatus::Cancelled` from `PlanStatus::Failed`.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

use chrono::{DateTime, Utc};
use tokio::task::JoinHandle;
use uuid::Uuid;

use crate::error::{FlowdError, Result};

use super::Plan;
use super::executor::{
    AgentMetrics, AgentSpawnContext, AgentSpawner, PlanRuntime, PlanSessionHandle, StepOutcome,
    apply_outcome, mark_step_running, run_step,
};
use super::gate::SharedRuleGate;
use super::observer::{PlanEvent, PlanStepCounts, SharedPlanObserver};
use super::store::PlanStore;
use super::template;

/// Per-layer fan-out coordinator. Holds borrowed references to the
/// executor's collaborators; constructed fresh inside `execute` for every
/// plan run, so its lifetime is bounded by the run.
pub(super) struct LayerRunner<'a, S: AgentSpawner + 'static, PS: PlanStore> {
    pub(super) spawner: &'a Arc<S>,
    pub(super) rule_gate: Option<&'a SharedRuleGate>,
    pub(super) observer: Option<&'a SharedPlanObserver>,
    pub(super) plans: &'a Arc<Mutex<HashMap<Uuid, PlanRuntime>>>,
    /// Plan snapshot store. Used to publish the per-step Running
    /// transition mid-layer so any external `plan_status` query (or
    /// post-restart rehydrate) sees the in-flight state instead of
    /// the pre-layer Pending row.
    pub(super) store: &'a PS,
    /// Daemon-wide fallback timeout threaded in from
    /// [`super::executor::InMemoryPlanExecutor::default_step_timeout_secs`].
    /// Consulted per spawned step via [`run_step`]; per-step
    /// `timeout_secs` wins when set.
    pub(super) default_step_timeout_secs: Option<u64>,
}

/// Per-step record captured at the moment the layer runner flipped a
/// step to Running. Carries everything the post-spawn `StepStarted`
/// event needs without re-walking `plan.steps`.
struct StartedStep {
    step_id: String,
    agent_type: String,
    started_at: DateTime<Utc>,
}

/// Outcome of a single layer-run.
#[derive(Debug, Default)]
pub(super) struct LayerOutcome {
    /// `true` iff at least one step in the layer was refused or failed
    /// (i.e. the layer should turn the overall plan into `Failed`).
    /// Cancellation is *not* counted here: the executor inspects the
    /// cancellation flag separately to settle the plan as `Cancelled`.
    pub(super) failed: bool,
    /// Accumulated metrics from every step in this layer that reported
    /// a non-`None` metrics payload (both successes and failures).
    /// Kept as `AgentMetrics` with zero-valued defaults so the caller
    /// can `merge` it into a running plan-wide total unconditionally.
    pub(super) metrics: AgentMetrics,
    /// Per-terminal-outcome counts for this layer; the executor sums
    /// them across layers for the `Finished` event's rollup.
    pub(super) step_count: PlanStepCounts,
}

impl<S: AgentSpawner + 'static, PS: PlanStore> LayerRunner<'_, S, PS> {
    /// Run every step in `layer` against `plan`, mutating `plan.steps` in
    /// place with the outcomes and emitting per-step lifecycle events.
    ///
    /// Returns once every spawned task has been awaited. Persistence of
    /// the post-layer snapshot is the executor's responsibility (it is
    /// also the executor that drives the layer iteration), keeping
    /// `LayerRunner` purely concerned with one layer's fan-out.
    ///
    /// `last_session` carries the predecessor's conversation handle
    /// across layers (see [`PlanSessionHandle`]). It is updated
    /// in-place: cleared at the start of every parallel layer (a
    /// single conversation cannot fan out across N concurrent CLIs),
    /// updated to the just-completed step's `session_id` after a
    /// serial-layer success, and cleared when a serial-layer step
    /// fails / cancels / returns no `session_id`. The executor owns
    /// the slot and threads it from one `run` call to the next.
    pub(super) async fn run(
        &self,
        plan_id: Uuid,
        plan: &mut Plan,
        layer: &[String],
        cancel: &Arc<AtomicBool>,
        last_session: &mut Option<PlanSessionHandle>,
    ) -> Result<LayerOutcome> {
        // Snapshot owned (id, output) pairs for completed steps so the
        // prompt substituter can run while we still hold a `&` to `plan`,
        // before the spawn loop needs `&mut`. Validation guarantees that
        // every `{{steps.<id>.output}}` reference points to a transitive
        // dependency, so the value is either present (Completed) or a
        // sentinel (Failed/Cancelled, which would normally have aborted
        // the plan but we resolve to the placeholder rather than crash).
        let outputs_owned = collect_completed_outputs(plan);
        let outputs: HashMap<&str, &str> = outputs_owned
            .iter()
            .map(|(k, v)| (k.as_str(), v.as_str()))
            .collect();

        let mut handles: Vec<(String, JoinHandle<StepOutcome>)> = Vec::with_capacity(layer.len());
        let mut started: Vec<StartedStep> = Vec::with_capacity(layer.len());
        let mut layer_failed = false;
        let mut rollup = AgentMetrics::default();
        let mut step_count = PlanStepCounts::default();

        let plan_parallel = plan.execution_layers()?.iter().any(|l| l.len() > 1);
        let layer_serial = layer.len() == 1;

        // Parallel layers cannot resume a session: the conversation
        // would fork into N concurrent CLI invocations and there is no
        // sound merge. Drop the slot now so the post-layer
        // bookkeeping below cannot accidentally re-use it.
        if !layer_serial {
            *last_session = None;
        }

        for step_id in layer {
            let Some(mut step) = plan.steps.iter().find(|s| &s.id == step_id).cloned() else {
                continue;
            };
            step.prompt =
                template::substitute(&step.prompt, &outputs, template::DEFAULT_PER_REF_BYTES);

            if self.refuse_via_gate(plan_id, plan, &step) {
                layer_failed = true;
                step_count.failed = step_count.failed.saturating_add(1);
                continue;
            }

            // Stamp the real start moment *before* spawn so the snapshot
            // we publish below carries the timestamp the eventual
            // StepCompleted/Failed/Cancelled outcome will preserve.
            let started_at = Utc::now();
            mark_step_running(plan, &step.id, started_at);
            step.status = super::StepStatus::Running;
            step.started_at = Some(started_at);

            let prior_session_id =
                resume_id_for_step(layer_serial, last_session.as_ref(), &step.agent_type);
            let ctx = AgentSpawnContext {
                plan_id,
                project: plan.project.clone(),
                plan_parallel,
                layer_width: layer.len(),
                project_root: plan.project_root.as_deref().map(std::path::PathBuf::from),
                prior_session_id,
            };
            started.push(StartedStep {
                step_id: step.id.clone(),
                agent_type: step.agent_type.clone(),
                started_at,
            });
            let spawner = Arc::clone(self.spawner);
            let cancel_for_task = Arc::clone(cancel);
            let default_timeout = self.default_step_timeout_secs;
            let handle = tokio::spawn(async move {
                run_step(&*spawner, ctx, step, cancel_for_task, default_timeout).await
            });
            handles.push((step_id.clone(), handle));
        }

        self.publish_in_flight(plan_id, &handles)?;
        // Mirror the Running state into both the in-memory runtime
        // (so `status()` callers see the live transition) and the
        // store (so a post-restart rehydrate can settle in-flight
        // steps as Failed/Interrupted instead of leaving the
        // pre-layer Pending row in place).
        if !started.is_empty() {
            self.publish_running_snapshot(plan_id, plan).await?;
            for s in &started {
                self.emit(PlanEvent::StepStarted {
                    plan_id,
                    project: plan.project.clone(),
                    step_id: s.step_id.clone(),
                    agent_type: s.agent_type.clone(),
                    started_at: s.started_at,
                });
            }
        }

        for (step_id, handle) in handles {
            let outcome = await_outcome(handle, &step_id).await;
            apply_outcome(plan, &step_id, &outcome);

            let agent_type = plan
                .steps
                .iter()
                .find(|s| s.id == step_id)
                .map(|s| s.agent_type.clone())
                .unwrap_or_default();

            self.handle_outcome(
                plan_id,
                plan,
                step_id,
                agent_type,
                outcome,
                layer_serial,
                cancel,
                last_session,
                &mut layer_failed,
                &mut step_count,
                &mut rollup,
            );
        }

        Ok(LayerOutcome {
            failed: layer_failed,
            metrics: rollup,
            step_count,
        })
    }

    /// Settle a single completed task: update the resume slot, the
    /// per-layer counters, the cancellation latch, and emit the
    /// matching lifecycle event. Pulled out of [`Self::run`] purely
    /// for clippy's `too_many_lines` budget; all behaviour is
    /// identical to the inline match this replaced.
    #[allow(clippy::too_many_arguments)] // intermediate helper, scoped to one caller.
    fn handle_outcome(
        &self,
        plan_id: Uuid,
        plan: &Plan,
        step_id: String,
        agent_type: String,
        outcome: StepOutcome,
        layer_serial: bool,
        cancel: &Arc<AtomicBool>,
        last_session: &mut Option<PlanSessionHandle>,
        layer_failed: &mut bool,
        step_count: &mut PlanStepCounts,
        rollup: &mut AgentMetrics,
    ) {
        match outcome {
            StepOutcome::Failed { error, metrics } => {
                *layer_failed = true;
                step_count.failed = step_count.failed.saturating_add(1);
                if let Some(m) = &metrics {
                    rollup.merge(m);
                }
                // A failed step in a serial layer breaks the resume
                // chain: the next step starts fresh.
                if layer_serial {
                    *last_session = None;
                }
                self.emit(PlanEvent::StepFailed {
                    plan_id,
                    project: plan.project.clone(),
                    step_id,
                    agent_type,
                    error,
                    metrics,
                });
            }
            StepOutcome::Cancelled => {
                // Flip the cancellation latch so the next layer never
                // starts, but do *not* mark the layer failed: the
                // executor settles the plan as `Cancelled` (not
                // `Failed`) when this latch is set at finalisation.
                cancel.store(true, Ordering::SeqCst);
                if layer_serial {
                    *last_session = None;
                }
                self.emit(PlanEvent::StepCancelled {
                    plan_id,
                    project: plan.project.clone(),
                    step_id,
                    agent_type,
                });
            }
            StepOutcome::Completed {
                output,
                metrics,
                session_id,
            } => {
                step_count.completed = step_count.completed.saturating_add(1);
                if let Some(m) = &metrics {
                    rollup.merge(m);
                }
                // Only refresh the resume slot for serial layers.
                // Parallel layers never write here; the slot was
                // already cleared at layer entry above.
                if layer_serial {
                    *last_session = session_id.map(|sid| PlanSessionHandle {
                        agent_type: agent_type.clone(),
                        session_id: sid,
                    });
                }
                self.emit(PlanEvent::StepCompleted {
                    plan_id,
                    project: plan.project.clone(),
                    step_id,
                    agent_type,
                    output,
                    metrics,
                });
            }
            // Refused outcomes never reach this loop: a denied step is
            // settled inline by `refuse_via_gate` and never pushed
            // into `handles`.
            StepOutcome::Refused { .. } => {
                debug_assert!(false, "Refused outcome should never reach the join loop");
            }
        }
    }

    /// Apply the rule gate to a step before spawn. Returns `true` when the
    /// gate refused (and we settled the step inline as Skipped); the
    /// caller then skips pushing the step into the join loop.
    fn refuse_via_gate(&self, plan_id: Uuid, plan: &mut Plan, step: &super::PlanStep) -> bool {
        let Some(gate) = self.rule_gate else {
            return false;
        };
        let result = gate.gate(step, plan.project.as_str());
        for w in result.warnings() {
            tracing::warn!(
                plan_id = %plan_id,
                step_id = %step.id,
                agent = %step.agent_type,
                rule_id = %w.rule_id,
                description = %w.description,
                "rule gate warning at step boundary"
            );
        }
        if result.allowed {
            return false;
        }
        let denial = result.denials().next().map_or_else(
            || "denied by rule gate".to_owned(),
            |d| format!("denied by rule `{}`: {}", d.rule_id, d.description),
        );
        tracing::warn!(
            plan_id = %plan_id,
            step_id = %step.id,
            agent = %step.agent_type,
            denial,
            "rule gate refused to spawn step"
        );
        apply_outcome(
            plan,
            &step.id,
            &StepOutcome::Refused {
                reason: denial.clone(),
            },
        );
        self.emit(PlanEvent::StepRefused {
            plan_id,
            project: plan.project.clone(),
            step_id: step.id.clone(),
            agent_type: step.agent_type.clone(),
            reason: denial,
        });
        true
    }

    /// Publish abort handles for the current layer's tasks so
    /// [`super::PlanExecutor::cancel`] can terminate them mid-layer. Holds
    /// the runtime mutex only across a tight non-async block.
    fn publish_in_flight(
        &self,
        plan_id: Uuid,
        handles: &[(String, JoinHandle<StepOutcome>)],
    ) -> Result<()> {
        let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
            message: "plan store poisoned".into(),
            metrics: None,
        })?;
        if let Some(runtime) = guard.get_mut(&plan_id) {
            runtime.in_flight = handles.iter().map(|(_, h)| h.abort_handle()).collect();
        }
        Ok(())
    }

    /// Make the just-marked Running steps observable: mirror them into
    /// the in-memory runtime (so any concurrent `status()` returns the
    /// transition immediately) and persist via the configured store.
    /// Persistence runs outside the runtime mutex so a slow store does
    /// not block readers.
    async fn publish_running_snapshot(&self, plan_id: Uuid, plan: &Plan) -> Result<()> {
        {
            let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
                message: "plan store poisoned".into(),
                metrics: None,
            })?;
            if let Some(runtime) = guard.get_mut(&plan_id) {
                runtime.plan.steps.clone_from(&plan.steps);
            }
        }
        self.store.save_plan(plan).await
    }

    fn emit(&self, event: PlanEvent) {
        if let Some(obs) = self.observer {
            obs.on_event(event);
        }
    }
}

/// Snapshot owned (id, output) pairs for steps that already completed.
/// Returned as `Vec` so the caller can build a `&str -> &str` `HashMap`
/// against borrows that outlive the spawn loop.
fn collect_completed_outputs(plan: &Plan) -> Vec<(String, String)> {
    plan.steps
        .iter()
        .filter_map(|s| s.output.as_ref().map(|o| (s.id.clone(), o.clone())))
        .collect()
}

/// Decide what `prior_session_id` to thread into the next step's
/// [`AgentSpawnContext`].
///
/// Returns `Some(id)` only when all three guards hold:
///
/// 1. `layer_serial` -- a parallel layer cannot share one
///    conversation across N concurrent CLI invocations.
/// 2. The slot is populated -- nothing to resume otherwise.
/// 3. The predecessor targeted the same `agent_type` -- a Claude
///    session id handed to Codex / Aider would either be a no-op or
///    a hard error from the foreign CLI.
fn resume_id_for_step(
    layer_serial: bool,
    last_session: Option<&PlanSessionHandle>,
    next_agent_type: &str,
) -> Option<String> {
    if !layer_serial {
        return None;
    }
    last_session
        .filter(|h| h.agent_type == next_agent_type)
        .map(|h| h.session_id.clone())
}

/// Translate a `JoinHandle` result into a [`StepOutcome`]. Centralised so
/// panics, cooperative cancellation, and other join errors map to the
/// same `Failed`/`Cancelled` semantics used by the executor pre-HL-41.
async fn await_outcome(handle: JoinHandle<StepOutcome>, step_id: &str) -> StepOutcome {
    match handle.await {
        Ok(o) => o,
        Err(join_err) if join_err.is_cancelled() => StepOutcome::Cancelled,
        Err(join_err) if join_err.is_panic() => StepOutcome::Failed {
            error: format!("step `{step_id}` panicked: {join_err}"),
            metrics: None,
        },
        Err(join_err) => StepOutcome::Failed {
            error: format!("step `{step_id}` join error: {join_err}"),
            metrics: None,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::{PlanStep, StepStatus};

    fn step_with_output(id: &str, output: Option<&str>) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: format!("do {id}"),
            depends_on: Vec::new(),
            timeout_secs: None,
            retry_count: 0,
            status: if output.is_some() {
                StepStatus::Completed
            } else {
                StepStatus::Pending
            },
            output: output.map(str::to_owned),
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    #[test]
    fn collect_completed_outputs_includes_only_settled_steps() {
        let plan = Plan::new(
            "p",
            "proj",
            vec![
                step_with_output("a", Some("hello")),
                step_with_output("b", None),
                step_with_output("c", Some("world")),
            ],
        );
        let mut outs = collect_completed_outputs(&plan);
        outs.sort();
        assert_eq!(
            outs,
            vec![
                ("a".to_owned(), "hello".to_owned()),
                ("c".to_owned(), "world".to_owned()),
            ],
        );
    }

    #[test]
    fn collect_completed_outputs_empty_for_fresh_plan() {
        let plan = Plan::new("p", "proj", vec![step_with_output("a", None)]);
        assert!(collect_completed_outputs(&plan).is_empty());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn await_outcome_translates_panic_into_failed() {
        let handle: JoinHandle<StepOutcome> = tokio::spawn(async {
            panic!("boom");
        });
        let outcome = await_outcome(handle, "explode").await;
        match outcome {
            StepOutcome::Failed { error, .. } => {
                assert!(error.contains("explode"), "error should mention step id");
                assert!(error.contains("panicked"), "error should mention panic");
            }
            other => panic!("expected Failed for panicked task, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn await_outcome_translates_abort_into_cancelled() {
        let handle: JoinHandle<StepOutcome> = tokio::spawn(async {
            std::future::pending::<()>().await;
            unreachable!()
        });
        handle.abort();
        let outcome = await_outcome(handle, "blocked").await;
        assert!(
            matches!(outcome, StepOutcome::Cancelled),
            "aborted task should map to Cancelled"
        );
    }

    #[tokio::test]
    async fn await_outcome_passes_through_completed() {
        let handle: JoinHandle<StepOutcome> = tokio::spawn(async {
            StepOutcome::Completed {
                output: "ok".into(),
                metrics: None,
                session_id: None,
            }
        });
        let outcome = await_outcome(handle, "fast").await;
        match outcome {
            StepOutcome::Completed { output, .. } => assert_eq!(output, "ok"),
            other => panic!("expected Completed, got {other:?}"),
        }
    }

    /// `StepOutcome::Refused` is constructed only by the rule-gate path
    /// inside `LayerRunner`; the join loop must never see it. This test
    /// pins the contract used by the `debug_assert!` guard.
    #[test]
    fn refused_variant_constructible_with_reason() {
        let r = StepOutcome::Refused {
            reason: "no".into(),
        };
        assert!(matches!(r, StepOutcome::Refused { .. }));
    }
}
