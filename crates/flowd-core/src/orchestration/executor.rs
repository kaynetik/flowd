//! In-process orchestration executor.
//!
//! Drives a registered plan through its execution layers, supervising each
//! step via an injected [`AgentSpawner`]. The executor itself is transport
//! agnostic; it never spawns OS processes directly. Concrete spawners live in
//! `flowd-cli` / `flowd-mcp` (and a no-op spawner is provided for tests).
//!
//! ## Panic isolation
//!
//! Every step runs inside `tokio::spawn`, whose `JoinHandle` reports panics
//! as `JoinError::is_panic()`. The executor converts a panicked step into a
//! `Failed` `StepStatus` with a recorded error message instead of unwinding
//! the executor task itself -- a hostile or buggy `AgentSpawner` cannot crash
//! the orchestrator.
//!
//! ## Cancellation
//!
//! `cancel(plan_id)` flips an `Arc<AtomicBool>` and aborts any currently
//! running per-step tasks. Layer transitions also re-check the flag, so a
//! cancellation issued mid-plan stops further layers from starting.
//!
//! ## Concurrency model
//!
//! Plan state lives in a `std::sync::Mutex<HashMap>`. Critical sections only
//! hold the lock while reading or mutating the store -- never across an
//! `.await` boundary -- so the executor remains responsive under load. When a
//! [`crate::orchestration::PlanStore`] is configured, every transition is
//! mirrored to durable storage so the daemon can `rehydrate` after restart.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::gate::SharedRuleGate;
use super::layer_runner::LayerRunner;
use super::observer::{PlanEvent, SharedPlanObserver};

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::task::AbortHandle;
use uuid::Uuid;

use crate::error::{FlowdError, Result};

use super::{
    NoOpPlanStore, Plan, PlanExecutor, PlanPreview, PlanStatus, PlanStep, PlanStore, StepStatus,
    build_preview, compiler::CompileOutput, validate_plan, validate_plan_pending,
};

/// Output captured from an agent invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentOutput {
    pub stdout: String,
    pub exit_code: Option<i32>,
}

impl AgentOutput {
    #[must_use]
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit_code: Some(0),
        }
    }
}

/// Strategy for invoking a single plan step.
///
/// Implementations are free to spawn an OS process, call an HTTP endpoint,
/// or simply return canned data in tests. Returning `Err(_)` marks the step
/// as failed and triggers any configured retries.
pub trait AgentSpawner: Send + Sync {
    /// Run a step and return its captured output.
    ///
    /// # Errors
    /// Implementations should return `FlowdError::PlanExecution` for
    /// transport / process failures.
    fn spawn(&self, step: &PlanStep) -> impl Future<Output = Result<AgentOutput>> + Send;
}

/// Internal record kept per registered plan.
///
/// `pub(super)` so the sibling [`super::layer_runner`] module can publish
/// abort handles into `in_flight` between spawn and await; nothing outside
/// the orchestration module sees this type.
pub(super) struct PlanRuntime {
    pub(super) plan: Plan,
    pub(super) cancel: Arc<AtomicBool>,
    /// Abort handles for the currently in-flight step tasks. Populated by
    /// [`super::layer_runner::LayerRunner`] at every layer boundary;
    /// consumed by `cancel`. We store `AbortHandle` (cheap to clone, can
    /// be shared across owners) rather than the `JoinHandle` itself,
    /// which the runner needs to keep so it can `.await` each task.
    pub(super) in_flight: Vec<AbortHandle>,
}

/// Default executor. Generic over the spawner and an optional [`PlanStore`].
///
/// An optional [`SharedRuleGate`] can be installed via [`Self::with_rule_gate`];
/// when present it is consulted before each step is spawned (see `execute`).
pub struct InMemoryPlanExecutor<S: AgentSpawner + 'static, PS: PlanStore = NoOpPlanStore> {
    spawner: Arc<S>,
    plans: Arc<Mutex<HashMap<Uuid, PlanRuntime>>>,
    store: PS,
    rule_gate: Option<SharedRuleGate>,
    observer: Option<SharedPlanObserver>,
}

impl<S: AgentSpawner + 'static> InMemoryPlanExecutor<S, NoOpPlanStore> {
    pub fn new(spawner: S) -> Self {
        Self::with_plan_store(spawner, NoOpPlanStore)
    }

    pub fn from_shared(spawner: Arc<S>) -> Self {
        Self {
            spawner,
            plans: Arc::new(Mutex::new(HashMap::new())),
            store: NoOpPlanStore,
            rule_gate: None,
            observer: None,
        }
    }
}

impl<S: AgentSpawner + 'static, PS: PlanStore> InMemoryPlanExecutor<S, PS> {
    pub fn with_plan_store(spawner: S, store: PS) -> Self {
        Self {
            spawner: Arc::new(spawner),
            plans: Arc::new(Mutex::new(HashMap::new())),
            store,
            rule_gate: None,
            observer: None,
        }
    }

    pub fn from_shared_with_store(spawner: Arc<S>, store: PS) -> Self {
        Self {
            spawner,
            plans: Arc::new(Mutex::new(HashMap::new())),
            store,
            rule_gate: None,
            observer: None,
        }
    }

    /// Install a rule gate that the executor will consult before spawning
    /// each step. With no gate (the default), the executor runs every step
    /// the DAG allows -- preserving pre-existing behaviour.
    #[must_use]
    pub fn with_rule_gate(mut self, gate: SharedRuleGate) -> Self {
        self.rule_gate = Some(gate);
        self
    }

    /// Install a plan observer that will receive lifecycle events.
    ///
    /// With no observer (the default), no events are emitted -- preserving
    /// the executor's previous zero-side-effect behaviour. Observer impls
    /// are expected to be fast and to defer heavy work to spawned tasks.
    #[must_use]
    pub fn with_observer(mut self, observer: SharedPlanObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Fan out an event to the installed observer (if any). Hot-path call:
    /// inlined to a single `Option::if_let` when no observer is present.
    fn emit(&self, event: PlanEvent) {
        if let Some(obs) = &self.observer {
            obs.on_event(event);
        }
    }

    /// Reload non-terminal plans from [`PlanStore`] into the in-memory map.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn rehydrate(&self) -> Result<()> {
        let summaries = self.store.list_plans(None).await?;
        for s in summaries {
            if s.status.is_terminal() {
                continue;
            }
            let Some(plan) = self.store.load_plan(s.id).await? else {
                continue;
            };
            let runtime = PlanRuntime {
                plan,
                cancel: Arc::new(AtomicBool::new(false)),
                in_flight: Vec::new(),
            };
            let mut guard = self
                .plans
                .lock()
                .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
            guard.insert(s.id, runtime);
        }
        Ok(())
    }

    /// Look up a plan by id without mutating; used by external callers that
    /// want a snapshot without going through the trait.
    #[must_use]
    pub fn snapshot(&self, plan_id: Uuid) -> Option<Plan> {
        self.plans
            .lock()
            .ok()?
            .get(&plan_id)
            .map(|r| r.plan.clone())
    }

    async fn persist_snapshot(&self, plan_id: Uuid) -> Result<()> {
        let plan = self
            .snapshot(plan_id)
            .ok_or(FlowdError::PlanNotFound(plan_id))?;
        self.store.save_plan(&plan).await
    }

    fn with_plan_mut<R>(&self, plan_id: Uuid, f: impl FnOnce(&mut PlanRuntime) -> R) -> Result<R> {
        let mut guard = self
            .plans
            .lock()
            .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
        let entry = guard
            .get_mut(&plan_id)
            .ok_or(FlowdError::PlanNotFound(plan_id))?;
        Ok(f(entry))
    }

    /// Pre-flight phase of `execute`: take an owned plan snapshot, flip
    /// the runtime to `Running`, persist, and emit
    /// [`PlanEvent::Started`]. Returns the plan snapshot and the shared
    /// cancellation latch the per-layer runner mutates.
    ///
    /// Errors propagate the same `PlanNotFound` / `PlanExecution` codes
    /// the inlined version returned, so the public `execute` contract
    /// is unchanged.
    async fn start_execution(&self, plan_id: Uuid) -> Result<(Plan, Arc<AtomicBool>)> {
        // Scope the lock so its destructor runs *before* the `.await`.
        // Required for the future's auto-`Send` analysis even though
        // `MutexGuard: Send` would otherwise be fine.
        let (plan, cancel) = {
            let mut guard = self
                .plans
                .lock()
                .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
            let runtime = guard
                .get_mut(&plan_id)
                .ok_or(FlowdError::PlanNotFound(plan_id))?;
            if runtime.plan.status != PlanStatus::Confirmed {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` must be Confirmed before execute (currently {:?})",
                    runtime.plan.status
                )));
            }
            runtime.plan.status = PlanStatus::Running;
            runtime.plan.started_at = Some(Utc::now());
            (runtime.plan.clone(), Arc::clone(&runtime.cancel))
        };

        self.persist_snapshot(plan_id).await?;
        self.emit(PlanEvent::Started {
            plan_id,
            project: plan.project.clone(),
        });
        Ok((plan, cancel))
    }

    /// Finalise phase of `execute`: derive the terminal `PlanStatus`,
    /// commit the final step snapshot to the runtime, persist, and emit
    /// [`PlanEvent::Finished`].
    ///
    /// Cancellation always wins over failure: a plan whose first layer
    /// failed *and* was then cancelled mid-finalise still settles as
    /// `Cancelled`, matching the pre-HL-41 ordering.
    async fn finalize_execution(
        &self,
        plan_id: Uuid,
        plan: &Plan,
        cancel: &AtomicBool,
        overall_failed: bool,
    ) -> Result<()> {
        let final_status = if cancel.load(Ordering::SeqCst) {
            PlanStatus::Cancelled
        } else if overall_failed {
            PlanStatus::Failed
        } else {
            PlanStatus::Completed
        };

        self.with_plan_mut(plan_id, |runtime| {
            runtime.plan.status = final_status;
            runtime.plan.completed_at = Some(Utc::now());
            runtime.plan.steps.clone_from(&plan.steps);
            runtime.in_flight.clear();
        })?;
        self.persist_snapshot(plan_id).await?;

        self.emit(PlanEvent::Finished {
            plan_id,
            project: plan.project.clone(),
            status: final_status,
        });
        Ok(())
    }
}

impl<S: AgentSpawner + 'static, PS: PlanStore> PlanExecutor for InMemoryPlanExecutor<S, PS> {
    fn validate(&self, plan: &Plan) -> Result<()> {
        validate_plan(plan)
    }

    fn preview(&self, plan: &Plan) -> Result<PlanPreview> {
        build_preview(plan)
    }

    async fn submit(&self, plan: Plan) -> Result<Uuid> {
        // Use the pending-friendly variant: prose-first plans legitimately
        // have no compiled steps until clarification finishes. Strict
        // validation fires later, at confirm time.
        validate_plan_pending(&plan)?;
        let id = plan.id;
        let name = plan.name.clone();
        let project = plan.project.clone();
        let runtime = PlanRuntime {
            plan,
            cancel: Arc::new(AtomicBool::new(false)),
            in_flight: Vec::new(),
        };
        // Scope the guard so its destructor runs *before* the `.await`.
        // Explicit `drop(guard)` is not enough for the future's auto-Send
        // analysis — a syntactic scope block is.
        {
            let mut guard = self
                .plans
                .lock()
                .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
            guard.insert(id, runtime);
        }
        self.persist_snapshot(id).await?;
        self.emit(PlanEvent::Submitted {
            plan_id: id,
            name,
            project,
        });
        Ok(id)
    }

    async fn confirm(&self, plan_id: Uuid) -> Result<PlanPreview> {
        let preview = self.with_plan_mut(plan_id, |runtime| {
            if runtime.plan.status != PlanStatus::Draft {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` is not in Draft state (currently {:?})",
                    runtime.plan.status
                )));
            }
            let preview = build_preview(&runtime.plan)?;
            runtime.plan.status = PlanStatus::Confirmed;
            Ok(preview)
        })??;
        self.persist_snapshot(plan_id).await?;
        Ok(preview)
    }

    // The plan lifecycle reads top-to-bottom: pre-flight, fan-out one
    // layer at a time via [`LayerRunner`], finalise. Per-layer logic
    // lives in [`super::layer_runner`] (HL-41) so this method stays a
    // readable sketch of the state machine.
    async fn execute(&self, plan_id: Uuid) -> Result<()> {
        let (mut plan, cancel) = self.start_execution(plan_id).await?;

        let runner = LayerRunner {
            spawner: &self.spawner,
            rule_gate: self.rule_gate.as_ref(),
            observer: self.observer.as_ref(),
            plans: &self.plans,
        };

        let mut overall_failed = false;
        for layer in plan.execution_layers()? {
            // Cancellation can be set by an external `cancel(plan_id)`
            // call between layers, or by a `Cancelled` step outcome
            // within the previous layer.
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            let outcome = runner.run(plan_id, &mut plan, &layer, &cancel).await?;

            // Persist incremental state after every layer so `status()`
            // calls observe progress in real time.
            self.with_plan_mut(plan_id, |runtime| {
                runtime.plan.steps.clone_from(&plan.steps);
            })?;
            self.persist_snapshot(plan_id).await?;
            // Allow other tasks (and tests) to observe state between layers.
            tokio::task::yield_now().await;

            if outcome.failed {
                overall_failed = true;
                break;
            }
        }

        self.finalize_execution(plan_id, &plan, &cancel, overall_failed)
            .await
    }

    async fn status(&self, plan_id: Uuid) -> Result<Plan> {
        let guard = self
            .plans
            .lock()
            .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
        guard
            .get(&plan_id)
            .map(|r| r.plan.clone())
            .ok_or(FlowdError::PlanNotFound(plan_id))
    }

    async fn cancel(&self, plan_id: Uuid) -> Result<()> {
        // Three execution paths depending on where the plan is in its
        // lifecycle:
        //   * Terminal -> idempotent no-op.
        //   * Draft / Confirmed (no execute() task in flight) -> directly
        //     transition to Cancelled, persist, emit Finished.
        //   * Running -> set the latch + abort handles; the executing
        //     task's finalize_execution will see the latch and settle
        //     the plan as Cancelled (preserving the historic ordering
        //     guarantees).
        enum Action {
            Noop,
            DirectTransition { project: String },
            LatchAndAbort { handles: Vec<AbortHandle> },
        }

        let action = self.with_plan_mut(plan_id, |runtime| match runtime.plan.status {
            PlanStatus::Completed | PlanStatus::Failed | PlanStatus::Cancelled => Action::Noop,
            PlanStatus::Draft | PlanStatus::Confirmed => {
                runtime.cancel.store(true, Ordering::SeqCst);
                runtime.plan.status = PlanStatus::Cancelled;
                runtime.plan.completed_at = Some(Utc::now());
                runtime.in_flight.clear();
                Action::DirectTransition {
                    project: runtime.plan.project.clone(),
                }
            }
            PlanStatus::Running => {
                runtime.cancel.store(true, Ordering::SeqCst);
                Action::LatchAndAbort {
                    handles: std::mem::take(&mut runtime.in_flight),
                }
            }
        })?;

        match action {
            Action::Noop => Ok(()),
            Action::DirectTransition { project } => {
                self.persist_snapshot(plan_id).await?;
                self.emit(PlanEvent::Finished {
                    plan_id,
                    project,
                    status: PlanStatus::Cancelled,
                });
                Ok(())
            }
            Action::LatchAndAbort { handles } => {
                for h in handles {
                    h.abort();
                }
                Ok(())
            }
        }
    }

    async fn resume_plan(&self, plan_id: Uuid) -> Result<()> {
        self.with_plan_mut(plan_id, |runtime| {
            // Only `Failed` is resumable. `Completed` has nothing to do;
            // `Cancelled` is an explicit user stop (re-create the plan if
            // you want to retry); `Running` is racing with another caller.
            if runtime.plan.status != PlanStatus::Failed {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` is in {:?} state; only Failed plans can be resumed",
                    runtime.plan.status
                )));
            }
            for step in &mut runtime.plan.steps {
                if step.status == StepStatus::Failed {
                    step.status = StepStatus::Pending;
                    step.error = None;
                    step.output = None;
                    step.started_at = None;
                    step.completed_at = None;
                }
            }
            runtime.plan.status = PlanStatus::Confirmed;
            runtime.cancel = Arc::new(AtomicBool::new(false));
            runtime.in_flight.clear();
            Ok(())
        })??;
        self.persist_snapshot(plan_id).await
    }

    async fn apply_compile_output(&self, plan_id: Uuid, output: CompileOutput) -> Result<()> {
        self.with_plan_mut(plan_id, |runtime| {
            if runtime.plan.status != PlanStatus::Draft {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` is not in Draft state (currently {:?}); compile outputs only apply during clarification",
                    runtime.plan.status
                )));
            }
            runtime.plan.apply_compile_output(output);
            Ok(())
        })??;
        self.persist_snapshot(plan_id).await
    }

    async fn invalidate_decision(&self, plan_id: Uuid, question_id: String) -> Result<Vec<String>> {
        let removed = self.with_plan_mut(plan_id, |runtime| {
            if runtime.plan.status != PlanStatus::Draft {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` is not in Draft state (currently {:?}); cannot invalidate decisions",
                    runtime.plan.status
                )));
            }
            Ok(runtime.plan.invalidate_decision(&question_id))
        })??;
        self.persist_snapshot(plan_id).await?;
        Ok(removed)
    }
}

/// Outcome of a single step's execution attempt.
///
/// `pub(super)` so the sibling [`super::layer_runner`] module that drives
/// per-layer fan-out can construct and pattern-match on it.
#[derive(Debug)]
pub(super) enum StepOutcome {
    Completed {
        output: String,
    },
    Failed {
        error: String,
    },
    Cancelled,
    /// Step was never spawned because an executor-side gate (currently the
    /// rule gate) refused to run it. Settles into `StepStatus::Skipped`.
    Refused {
        reason: String,
    },
}

/// Execute one step, honouring its retry / timeout configuration. Runs inside
/// a `tokio::spawn`'d task, so panics surface as `JoinError::is_panic()` to
/// the caller.
pub(super) async fn run_step<S: AgentSpawner + ?Sized>(
    spawner: &S,
    step: PlanStep,
    cancel: Arc<AtomicBool>,
) -> StepOutcome {
    let attempts = step.retry_count.saturating_add(1);
    let mut last_error: Option<String> = None;

    for _ in 0..attempts {
        if cancel.load(Ordering::SeqCst) {
            return StepOutcome::Cancelled;
        }

        let call = spawner.spawn(&step);
        let result = match step.timeout_secs {
            Some(secs) => match tokio::time::timeout(Duration::from_secs(secs), call).await {
                Ok(r) => r,
                Err(_) => Err(FlowdError::PlanExecution(format!(
                    "step `{}` timed out after {secs}s",
                    step.id
                ))),
            },
            None => call.await,
        };

        match result {
            Ok(out) => return StepOutcome::Completed { output: out.stdout },
            Err(e) => last_error = Some(e.to_string()),
        }
    }

    StepOutcome::Failed {
        error: last_error.unwrap_or_else(|| "unknown step failure".into()),
    }
}

pub(super) fn apply_outcome(plan: &mut Plan, step_id: &str, outcome: &StepOutcome) {
    let Some(step) = plan.steps.iter_mut().find(|s| s.id == step_id) else {
        return;
    };
    let now = Some(Utc::now());
    if step.started_at.is_none() {
        step.started_at = now;
    }
    step.completed_at = now;
    match outcome {
        StepOutcome::Completed { output } => {
            step.status = StepStatus::Completed;
            step.output = Some(output.clone());
        }
        StepOutcome::Failed { error } => {
            step.status = StepStatus::Failed;
            step.error = Some(error.clone());
        }
        StepOutcome::Cancelled => {
            step.status = StepStatus::Cancelled;
            step.error = Some("cancelled".into());
        }
        StepOutcome::Refused { reason } => {
            step.status = StepStatus::Skipped;
            step.error = Some(reason.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::PlanStep;
    use std::sync::atomic::AtomicUsize;

    fn step(id: &str, deps: &[&str]) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: format!("do {id}"),
            depends_on: deps.iter().map(|s| (*s).to_owned()).collect(),
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    /// In-memory [`PlanStore`] for unit tests.
    #[derive(Clone, Default)]
    struct MapPlanStore(Arc<Mutex<HashMap<Uuid, Plan>>>);

    impl PlanStore for MapPlanStore {
        fn save_plan(&self, plan: &Plan) -> impl Future<Output = Result<()>> + Send {
            let m = Arc::clone(&self.0);
            let plan = plan.clone();
            async move {
                m.lock()
                    .map_err(|_| FlowdError::PlanExecution("map store poisoned".into()))?
                    .insert(plan.id, plan);
                Ok(())
            }
        }

        fn load_plan(&self, id: Uuid) -> impl Future<Output = Result<Option<Plan>>> + Send {
            let m = Arc::clone(&self.0);
            async move {
                Ok(m.lock()
                    .map_err(|_| FlowdError::PlanExecution("map store poisoned".into()))?
                    .get(&id)
                    .cloned())
            }
        }

        fn list_plans(
            &self,
            project: Option<&str>,
        ) -> impl Future<Output = Result<Vec<crate::orchestration::PlanSummary>>> + Send {
            let m = Arc::clone(&self.0);
            async move {
                let guard = m
                    .lock()
                    .map_err(|_| FlowdError::PlanExecution("map store poisoned".into()))?;
                let mut out: Vec<crate::orchestration::PlanSummary> = guard
                    .values()
                    .filter(|p| project.is_none_or(|pr| p.project == pr))
                    .map(|p| crate::orchestration::PlanSummary {
                        id: p.id,
                        name: p.name.clone(),
                        status: p.status,
                        created_at: p.created_at,
                        project: p.project.clone(),
                    })
                    .collect();
                out.sort_by_key(|s| std::cmp::Reverse(s.created_at));
                Ok(out)
            }
        }

        fn delete_plan(&self, id: Uuid) -> impl Future<Output = Result<()>> + Send {
            let m = Arc::clone(&self.0);
            async move {
                m.lock()
                    .map_err(|_| FlowdError::PlanExecution("map store poisoned".into()))?
                    .remove(&id);
                Ok(())
            }
        }
    }

    /// Spawner that records invocations and returns canned responses.
    struct EchoSpawner {
        invocations: Arc<AtomicUsize>,
    }

    impl AgentSpawner for EchoSpawner {
        async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(AgentOutput::success(format!("ran:{}", step.id)))
        }
    }

    /// Spawner whose output is the exact prompt it received. Lets tests
    /// assert that `{{steps.<id>.output}}` substitution happened before the
    /// spawner saw the prompt.
    struct RecordingSpawner {
        seen: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl AgentSpawner for RecordingSpawner {
        async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
            self.seen
                .lock()
                .expect("seen poisoned")
                .push((step.id.clone(), step.prompt.clone()));
            Ok(AgentOutput::success(step.prompt.clone()))
        }
    }

    /// Spawner that always fails -- exercises the retry / failure path.
    struct AlwaysFailSpawner {
        attempts: Arc<AtomicUsize>,
    }

    impl AgentSpawner for AlwaysFailSpawner {
        async fn spawn(&self, _: &PlanStep) -> Result<AgentOutput> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(FlowdError::PlanExecution("nope".into()))
        }
    }

    /// Spawner that panics on its first invocation.
    struct PanicSpawner;

    impl AgentSpawner for PanicSpawner {
        async fn spawn(&self, _: &PlanStep) -> Result<AgentOutput> {
            panic!("boom");
        }
    }

    /// First global spawn attempt fails; later attempts succeed (for resume).
    struct FailOnceThenSucceed {
        n: Arc<AtomicUsize>,
    }

    impl AgentSpawner for FailOnceThenSucceed {
        async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
            let c = self.n.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Err(FlowdError::PlanExecution("first run fails".into()))
            } else {
                Ok(AgentOutput::success(format!("ok:{}", step.id)))
            }
        }
    }

    /// Blocks indefinitely on step id `"c"` (for partial multi-layer runs).
    struct BlockStepC;

    impl AgentSpawner for BlockStepC {
        async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
            if step.id == "c" {
                std::future::poll_fn(|_| std::task::Poll::<()>::Pending).await;
            }
            Ok(AgentOutput::success(format!("ran:{}", step.id)))
        }
    }

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn full_lifecycle_completes_plan() {
        let invocations = Arc::new(AtomicUsize::new(0));
        let spawner = EchoSpawner {
            invocations: Arc::clone(&invocations),
        };
        let exec = InMemoryPlanExecutor::new(spawner);

        let plan = Plan::new(
            "p",
            "proj",
            vec![step("a", &[]), step("b", &["a"]), step("c", &["b"])],
        );

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            let preview = exec.confirm(id).await.unwrap();
            assert_eq!(preview.total_steps, 3);

            exec.execute(id).await.unwrap();
            let final_plan = exec.status(id).await.unwrap();
            assert_eq!(final_plan.status, PlanStatus::Completed);
            assert!(
                final_plan
                    .steps
                    .iter()
                    .all(|s| s.status == StepStatus::Completed)
            );
            assert_eq!(invocations.load(Ordering::SeqCst), 3);
        });
    }

    #[test]
    fn execute_requires_confirmation() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });
        rt().block_on(async {
            let id = exec
                .submit(Plan::new("p", "proj", vec![step("a", &[])]))
                .await
                .unwrap();
            let err = exec.execute(id).await.unwrap_err();
            assert!(matches!(err, FlowdError::PlanExecution(_)));
        });
    }

    #[test]
    fn retry_then_fail_marks_plan_failed() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let spawner = AlwaysFailSpawner {
            attempts: Arc::clone(&attempts),
        };
        let exec = InMemoryPlanExecutor::new(spawner);

        let mut s = step("a", &[]);
        s.retry_count = 2;
        let plan = Plan::new("p", "proj", vec![s]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
            let result = exec.status(id).await.unwrap();
            assert_eq!(result.status, PlanStatus::Failed);
            assert_eq!(attempts.load(Ordering::SeqCst), 3, "1 + 2 retries = 3");
            assert_eq!(result.steps[0].status, StepStatus::Failed);
        });
    }

    #[test]
    fn panicking_step_fails_isolated() {
        let exec = InMemoryPlanExecutor::new(PanicSpawner);
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
            let result = exec.status(id).await.unwrap();
            assert_eq!(result.status, PlanStatus::Failed);
            assert_eq!(result.steps[0].status, StepStatus::Failed);
            assert!(
                result.steps[0]
                    .error
                    .as_deref()
                    .is_some_and(|e| e.contains("panicked"))
            );
        });
    }

    #[test]
    fn failure_in_first_layer_skips_subsequent_layers() {
        let attempts = Arc::new(AtomicUsize::new(0));
        let spawner = AlwaysFailSpawner {
            attempts: Arc::clone(&attempts),
        };
        let exec = InMemoryPlanExecutor::new(spawner);

        let plan = Plan::new("p", "proj", vec![step("a", &[]), step("b", &["a"])]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
            let result = exec.status(id).await.unwrap();
            assert_eq!(result.status, PlanStatus::Failed);
            assert_eq!(
                attempts.load(Ordering::SeqCst),
                1,
                "second layer must not have run"
            );
            // step b never started
            assert_eq!(result.steps[1].status, StepStatus::Pending);
        });
    }

    #[test]
    fn unknown_plan_returns_not_found() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });
        rt().block_on(async {
            let err = exec.status(Uuid::new_v4()).await.unwrap_err();
            assert!(matches!(err, FlowdError::PlanNotFound(_)));
        });
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn survives_restart() {
        let store = MapPlanStore::default();
        let spawner = BlockStepC;
        let exec1 = InMemoryPlanExecutor::with_plan_store(spawner, store.clone());

        let plan = Plan::new(
            "p",
            "proj",
            vec![step("a", &[]), step("b", &[]), step("c", &["a", "b"])],
        );
        let id = plan.id;
        exec1.submit(plan).await.unwrap();
        exec1.confirm(id).await.unwrap();

        let exec_a = Arc::new(exec1);
        let jh = tokio::spawn({
            let e = Arc::clone(&exec_a);
            async move { e.execute(id).await }
        });

        loop {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
            if let Ok(p) = exec_a.status(id).await {
                if p.status == PlanStatus::Running
                    && p.steps.iter().find(|s| s.id == "a").unwrap().status == StepStatus::Completed
                    && p.steps.iter().find(|s| s.id == "b").unwrap().status == StepStatus::Completed
                    && p.steps.iter().find(|s| s.id == "c").unwrap().status == StepStatus::Pending
                {
                    break;
                }
            }
        }

        jh.abort();

        let exec2 = InMemoryPlanExecutor::with_plan_store(BlockStepC, store.clone());
        exec2.rehydrate().await.unwrap();
        let p = exec2.status(id).await.unwrap();
        assert_eq!(p.status, PlanStatus::Running);
        assert_eq!(p.steps[0].status, StepStatus::Completed);
        assert_eq!(p.steps[1].status, StepStatus::Completed);
        assert_eq!(p.steps[2].status, StepStatus::Pending);
    }

    #[test]
    fn observer_receives_full_plan_lifecycle() {
        use crate::orchestration::observer::{PlanEvent, PlanObserver};

        #[derive(Default)]
        struct CollectingObserver {
            events: Mutex<Vec<PlanEvent>>,
        }
        impl PlanObserver for CollectingObserver {
            fn on_event(&self, event: PlanEvent) {
                self.events.lock().unwrap().push(event);
            }
        }

        let invocations = Arc::new(AtomicUsize::new(0));
        let obs = Arc::new(CollectingObserver::default());
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::clone(&invocations),
        })
        .with_observer(Arc::clone(&obs) as Arc<dyn PlanObserver>);

        let plan = Plan::new("p", "proj", vec![step("a", &[]), step("b", &["a"])]);
        let plan_id = plan.id;

        rt().block_on(async {
            exec.submit(plan).await.unwrap();
            exec.confirm(plan_id).await.unwrap();
            exec.execute(plan_id).await.unwrap();
        });

        let events = obs.events.lock().unwrap().clone();
        let kinds: Vec<&'static str> = events
            .iter()
            .map(|e| match e {
                PlanEvent::Submitted { .. } => "submitted",
                PlanEvent::Started { .. } => "started",
                PlanEvent::StepCompleted { .. } => "step_completed",
                PlanEvent::StepFailed { .. } => "step_failed",
                PlanEvent::StepRefused { .. } => "step_refused",
                PlanEvent::StepCancelled { .. } => "step_cancelled",
                PlanEvent::Finished { .. } => "finished",
            })
            .collect();
        assert_eq!(
            kinds,
            vec![
                "submitted",
                "started",
                "step_completed",
                "step_completed",
                "finished"
            ],
            "unexpected event sequence: {kinds:?}"
        );

        match &events[4] {
            PlanEvent::Finished { status, .. } => assert_eq!(*status, PlanStatus::Completed),
            other => panic!("expected Finished event, got: {other:?}"),
        }
    }

    #[test]
    fn rule_gate_deny_skips_step_and_fails_plan() {
        use crate::error::RuleLevel;
        use crate::orchestration::gate::RuleGate;
        use crate::rules::{InMemoryRuleEvaluator, Rule, RuleEvaluator};

        let invocations = Arc::new(AtomicUsize::new(0));
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(Rule {
            id: "no-echo".into(),
            scope: "**".into(),
            level: RuleLevel::Deny,
            description: "echo agent banned".into(),
            match_pattern: "^echo$".into(),
        })
        .unwrap();
        let gate: Arc<dyn RuleGate> = Arc::new(ev);

        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::clone(&invocations),
        })
        .with_rule_gate(gate);

        // The rule's scope (`**`) is matched against the plan's project,
        // which is now required, so always-on deny rules fire reliably.
        let plan = Plan::new("p", "flowd", vec![step("a", &[])]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();

            let final_plan = exec.status(id).await.unwrap();
            assert_eq!(final_plan.status, PlanStatus::Failed);
            assert_eq!(final_plan.steps[0].status, StepStatus::Skipped);
            let err = final_plan.steps[0].error.as_deref().unwrap_or_default();
            assert!(err.contains("no-echo"), "expected rule id in error: {err}");
        });

        assert_eq!(
            invocations.load(Ordering::SeqCst),
            0,
            "denied step must not be spawned"
        );
    }

    #[test]
    fn step_prompt_substitutes_dependency_output_before_spawn() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let spawner = RecordingSpawner {
            seen: Arc::clone(&seen),
        };
        let exec = InMemoryPlanExecutor::new(spawner);

        let mut a = step("a", &[]);
        a.prompt = "produce-A".into();
        let mut b = step("b", &["a"]);
        b.prompt = "carry: {{steps.a.output}}".into();
        let plan = Plan::new("p", "proj", vec![a, b]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
        });

        let log = seen.lock().unwrap().clone();
        let b_prompt = log
            .iter()
            .find_map(|(id, p)| (id == "b").then_some(p.clone()))
            .expect("b should have been spawned");
        assert_eq!(b_prompt, "carry: produce-A");
    }

    #[test]
    fn resume_resets_failed_steps() {
        let store = MapPlanStore::default();
        let spawner = FailOnceThenSucceed {
            n: Arc::new(AtomicUsize::new(0)),
        };
        let exec = InMemoryPlanExecutor::with_plan_store(spawner, store);

        let plan = Plan::new("p", "proj", vec![step("a", &[]), step("b", &["a"])]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
            let failed = exec.status(id).await.unwrap();
            assert_eq!(failed.status, PlanStatus::Failed);
            assert_eq!(failed.steps[0].status, StepStatus::Failed);

            exec.resume_plan(id).await.unwrap();
            let ready = exec.status(id).await.unwrap();
            assert_eq!(ready.status, PlanStatus::Confirmed);
            assert_eq!(ready.steps[0].status, StepStatus::Pending);

            exec.execute(id).await.unwrap();
            let done = exec.status(id).await.unwrap();
            assert_eq!(done.status, PlanStatus::Completed);
            assert!(done.steps.iter().all(|s| s.status == StepStatus::Completed));
        });
    }

    // ---------- prose-first / clarification lifecycle (Phase 3) ----------

    use crate::orchestration::clarification::{DecisionRecord, OpenQuestion, QuestionOption};
    use crate::orchestration::compiler::CompileOutput;
    use crate::orchestration::loader::{PlanDefinition, StepDefinition};

    fn open_question(id: &str, deps: &[&str]) -> OpenQuestion {
        OpenQuestion {
            id: id.into(),
            prompt: format!("pick for {id}"),
            rationale: "test".into(),
            options: vec![QuestionOption {
                id: "x".into(),
                label: "X".into(),
                rationale: "any".into(),
            }],
            allow_explain_more: false,
            allow_none: false,
            depends_on_decisions: deps.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    fn def_with_step(id: &str) -> PlanDefinition {
        PlanDefinition {
            name: "p".into(),
            project: Some("proj".into()),
            steps: vec![StepDefinition {
                id: id.into(),
                agent_type: "echo".into(),
                prompt: "hi".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        }
    }

    /// A prose-first plan with open questions and no compiled steps must
    /// still pass `submit` (which uses the pending-friendly validator).
    /// The strict check fires only at confirm.
    #[test]
    fn submit_accepts_draft_with_open_questions_and_no_steps() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });

        let mut plan = Plan::new("p", "proj", vec![]);
        plan.open_questions.push(open_question("q1", &[]));

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            let snap = exec.status(id).await.unwrap();
            assert_eq!(snap.status, PlanStatus::Draft);
            assert_eq!(snap.open_questions.len(), 1);
            assert!(snap.steps.is_empty());

            // ...but confirming must reject with the inline-id message.
            let err = exec.confirm(id).await.unwrap_err();
            let msg = match err {
                FlowdError::PlanValidation(m) => m,
                other => panic!("unexpected: {other:?}"),
            };
            assert!(msg.contains("unresolved"), "msg: {msg}");
            assert!(msg.contains("q1"), "msg: {msg}");
        });
    }

    /// The full prose-first happy path: submit a draft with questions,
    /// drain them through `apply_compile_output`, then confirm.
    #[test]
    fn apply_compile_output_drains_questions_and_enables_confirm() {
        let store = MapPlanStore::default();
        let exec = InMemoryPlanExecutor::with_plan_store(
            EchoSpawner {
                invocations: Arc::new(AtomicUsize::new(0)),
            },
            store.clone(),
        );

        let mut plan = Plan::new("p", "proj", vec![]);
        plan.open_questions.push(open_question("q1", &[]));
        plan.source_doc = Some("# initial prose".into());

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();

            // Round 1: compiler returns the resolved DAG.
            exec.apply_compile_output(
                id,
                CompileOutput::ready("# resolved", def_with_step("step-a")),
            )
            .await
            .unwrap();

            let after = exec.status(id).await.unwrap();
            assert!(after.open_questions.is_empty());
            assert!(!after.definition_dirty);
            assert_eq!(after.steps.len(), 1);
            assert_eq!(after.source_doc.as_deref(), Some("# resolved"));

            // confirm now succeeds because there are real, valid steps.
            let preview = exec.confirm(id).await.unwrap();
            assert_eq!(preview.total_steps, 1);
        });
    }

    #[test]
    fn apply_compile_output_rejects_non_draft_plan() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            // Now plan is Confirmed, not Draft -- compile output rejected.
            let err = exec
                .apply_compile_output(id, CompileOutput::ready("# late", def_with_step("step-a")))
                .await
                .unwrap_err();
            assert!(matches!(err, FlowdError::PlanExecution(m) if m.contains("Draft")));
        });
    }

    #[test]
    fn invalidate_decision_executor_walks_dependents() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });

        let mut plan = Plan::new("p", "proj", vec![]);
        plan.open_questions.push(open_question("seed", &[])); // keep in Draft after submit
        plan.decisions.extend([
            DecisionRecord::new_user("q1", "x", vec![]),
            DecisionRecord::new_user("q2", "x", vec!["q1".into()]),
            DecisionRecord::new_user("q3", "x", vec!["q2".into()]),
        ]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            let removed = exec.invalidate_decision(id, "q1".into()).await.unwrap();
            assert_eq!(removed, vec!["q1", "q2", "q3"]);

            let snap = exec.status(id).await.unwrap();
            assert!(snap.decisions.is_empty());
            assert!(snap.definition_dirty);
        });
    }

    /// `cancel(plan_id)` on a `Draft` (never executed) plan must transition
    /// it directly to `Cancelled` and persist; previously cancel only set a
    /// latch which a never-running plan never observed.
    #[test]
    fn cancel_from_draft_transitions_to_cancelled_and_persists() {
        let store = MapPlanStore::default();
        let exec = InMemoryPlanExecutor::with_plan_store(
            EchoSpawner {
                invocations: Arc::new(AtomicUsize::new(0)),
            },
            store.clone(),
        );

        let mut plan = Plan::new("p", "proj", vec![]);
        plan.open_questions.push(open_question("q1", &[]));

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.cancel(id).await.unwrap();

            let snap = exec.status(id).await.unwrap();
            assert_eq!(snap.status, PlanStatus::Cancelled);
            assert!(snap.completed_at.is_some());

            // Cancellation is idempotent -- re-calling on a terminal plan
            // is a no-op rather than an error.
            exec.cancel(id).await.unwrap();

            let persisted = store.0.lock().unwrap().get(&id).unwrap().clone();
            assert_eq!(persisted.status, PlanStatus::Cancelled);
        });
    }

    #[test]
    fn cancel_from_confirmed_transitions_directly_when_no_executor_running() {
        let exec = InMemoryPlanExecutor::new(EchoSpawner {
            invocations: Arc::new(AtomicUsize::new(0)),
        });
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.cancel(id).await.unwrap();
            let snap = exec.status(id).await.unwrap();
            assert_eq!(snap.status, PlanStatus::Cancelled);
        });
    }
}
