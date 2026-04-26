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

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::gate::SharedRuleGate;
use super::layer_runner::LayerRunner;
use super::observer::{PlanEvent, PlanStepCounts, SharedPlanObserver};

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
    #[serde(default)]
    pub metrics: Option<AgentMetrics>,
}

impl AgentOutput {
    #[must_use]
    pub fn success(stdout: impl Into<String>) -> Self {
        Self {
            stdout: stdout.into(),
            exit_code: Some(0),
            metrics: None,
        }
    }
}

/// Per-model token / cost usage reported by an agent invocation.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct ModelUsage {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub cost_usd: f64,
}

impl ModelUsage {
    fn merge(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.cost_usd += other.cost_usd;
    }
}

/// Aggregate metrics captured alongside an agent invocation.
///
/// `model_usage` is keyed by model name so callers can report the split
/// across, e.g., a primary model and a cache-hit-heavy auxiliary.
#[derive(Debug, Default, Clone, Serialize, Deserialize, PartialEq)]
pub struct AgentMetrics {
    #[serde(default)]
    pub input_tokens: u64,
    #[serde(default)]
    pub output_tokens: u64,
    #[serde(default)]
    pub cache_creation_input_tokens: u64,
    #[serde(default)]
    pub cache_read_input_tokens: u64,
    #[serde(default)]
    pub total_cost_usd: f64,
    #[serde(default)]
    pub duration_ms: u64,
    #[serde(default)]
    pub duration_api_ms: u64,
    #[serde(default)]
    pub model_usage: BTreeMap<String, ModelUsage>,
}

impl AgentMetrics {
    pub fn merge(&mut self, other: &Self) {
        self.input_tokens += other.input_tokens;
        self.output_tokens += other.output_tokens;
        self.cache_creation_input_tokens += other.cache_creation_input_tokens;
        self.cache_read_input_tokens += other.cache_read_input_tokens;
        self.total_cost_usd += other.total_cost_usd;
        self.duration_ms += other.duration_ms;
        self.duration_api_ms += other.duration_api_ms;
        for (model, usage) in &other.model_usage {
            self.model_usage
                .entry(model.clone())
                .or_default()
                .merge(usage);
        }
    }
}

/// Plan-scoped context supplied to an [`AgentSpawner`] for one step.
#[derive(Debug, Clone)]
pub struct AgentSpawnContext {
    pub plan_id: Uuid,
    pub project: String,
    pub plan_parallel: bool,
    pub layer_width: usize,
    /// Absolute filesystem root of the workspace this plan operates on.
    /// Captured on [`super::Plan::project_root`] at submission time and
    /// threaded verbatim into every spawn so the agent inherits the
    /// plan's workspace instead of whatever directory the daemon process
    /// happens to live in. `None` only for legacy plans persisted before
    /// the field existed; spawners fall back to their own cwd then.
    pub project_root: Option<std::path::PathBuf>,
}

/// Strategy for invoking a single plan step.
///
/// Implementations are free to spawn an OS process, call an HTTP endpoint,
/// or simply return canned data in tests. Returning `Err(_)` marks the step
/// as failed and triggers any configured retries.
pub trait AgentSpawner: Send + Sync {
    /// Whether this spawner can isolate parallel plan steps in git worktrees.
    #[must_use]
    fn supports_worktree_isolation(&self) -> bool {
        false
    }

    /// Validate any plan-level prerequisites before a parallel plan is
    /// confirmed. Worktree-aware spawners use this to reject dirty base repos
    /// before the plan leaves Draft.
    fn prepare_plan(
        &self,
        _plan: &Plan,
        _preview: &PlanPreview,
    ) -> impl Future<Output = Result<()>> + Send {
        async { Ok(()) }
    }

    /// Run a step and return its captured output.
    ///
    /// # Errors
    /// Implementations should return `FlowdError::PlanExecution` for
    /// transport / process failures.
    fn spawn(
        &self,
        ctx: AgentSpawnContext,
        step: &PlanStep,
    ) -> impl Future<Output = Result<AgentOutput>> + Send;
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
    /// Daemon-wide fallback timeout for steps whose per-step
    /// `timeout_secs` is unset. `None` (the default) preserves the
    /// historic "run forever unless per-step timeout is set" behaviour,
    /// so non-CLI callers and existing tests do not regress. The CLI
    /// installs `Some(_)` from `[plan].step_timeout_secs`.
    default_step_timeout_secs: Option<u64>,
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
            default_step_timeout_secs: None,
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
            default_step_timeout_secs: None,
        }
    }

    pub fn from_shared_with_store(spawner: Arc<S>, store: PS) -> Self {
        Self {
            spawner,
            plans: Arc::new(Mutex::new(HashMap::new())),
            store,
            rule_gate: None,
            observer: None,
            default_step_timeout_secs: None,
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

    /// Set the daemon-wide fallback step timeout. A step's own
    /// `timeout_secs` (when `Some`) always wins; this value is only
    /// consulted when the step itself left the field unset -- which
    /// prose-compiled plans always do today, since neither the LLM
    /// prompt nor the structured-stub markdown grammar can emit it.
    /// Passing `None` preserves the historic "no fallback" behaviour.
    #[must_use]
    pub fn with_default_step_timeout(mut self, secs: Option<u64>) -> Self {
        self.default_step_timeout_secs = secs;
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
    /// Plans that were `Running` when the daemon stopped no longer have an
    /// executor task driving them, so we settle them as `Interrupted`
    /// (persisting the transition and emitting [`PlanEvent::Finished`]) to
    /// prevent the "stuck forever in Running" state. `resume_plan` can then
    /// reset the synthetic-Failed steps back to `Pending` so the user retries
    /// cleanly.
    ///
    /// # Errors
    /// Propagates storage failures.
    pub async fn rehydrate(&self) -> Result<()> {
        let summaries = self.store.list_plans(None).await?;
        for s in summaries {
            if s.status.is_terminal() {
                continue;
            }
            let Some(mut plan) = self.store.load_plan(s.id).await? else {
                continue;
            };
            match plan.status {
                PlanStatus::Running => {
                    let now = Utc::now();
                    for step in &mut plan.steps {
                        if !matches!(
                            step.status,
                            StepStatus::Completed
                                | StepStatus::Failed
                                | StepStatus::Skipped
                                | StepStatus::Cancelled
                        ) {
                            step.status = StepStatus::Failed;
                            step.error = Some("daemon restarted while step was in flight".into());
                            step.completed_at = Some(now);
                        }
                    }
                    plan.status = PlanStatus::Interrupted;
                    plan.completed_at = Some(now);

                    let mut step_count = PlanStepCounts::default();
                    for step in &plan.steps {
                        match step.status {
                            StepStatus::Completed => {
                                step_count.completed = step_count.completed.saturating_add(1);
                            }
                            StepStatus::Failed => {
                                step_count.failed = step_count.failed.saturating_add(1);
                            }
                            _ => {}
                        }
                    }

                    self.store.save_plan(&plan).await?;
                    self.emit(PlanEvent::Finished {
                        plan_id: s.id,
                        project: plan.project.clone(),
                        status: PlanStatus::Interrupted,
                        total_metrics: None,
                        step_count,
                        // Wall-clock would span the daemon restart -- not a
                        // meaningful runtime number for the operator.
                        elapsed_ms: None,
                    });

                    let runtime = PlanRuntime {
                        plan,
                        cancel: Arc::new(AtomicBool::new(false)),
                        in_flight: Vec::new(),
                    };
                    let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
                        message: "plan store poisoned".into(),
                        metrics: None,
                    })?;
                    guard.insert(s.id, runtime);
                }
                PlanStatus::Draft | PlanStatus::Confirmed | PlanStatus::Interrupted => {
                    let runtime = PlanRuntime {
                        plan,
                        cancel: Arc::new(AtomicBool::new(false)),
                        in_flight: Vec::new(),
                    };
                    let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
                        message: "plan store poisoned".into(),
                        metrics: None,
                    })?;
                    guard.insert(s.id, runtime);
                }
                PlanStatus::Completed | PlanStatus::Failed | PlanStatus::Cancelled => {}
            }
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

    /// Snapshot ids of every plan currently in [`PlanStatus::Running`].
    ///
    /// Intended for the daemon's shutdown path: on SIGTERM / Ctrl+C the
    /// caller issues `cancel(id)` for each returned id so the executing
    /// task commits a terminal status to `SQLite` instead of leaving a
    /// stale `Running` row for the rehydrate-as-Failed fallback.
    #[must_use]
    pub fn list_running(&self) -> Vec<Uuid> {
        let Ok(guard) = self.plans.lock() else {
            return Vec::new();
        };
        guard
            .iter()
            .filter_map(|(id, r)| (r.plan.status == PlanStatus::Running).then_some(*id))
            .collect()
    }

    async fn persist_snapshot(&self, plan_id: Uuid) -> Result<()> {
        let plan = self
            .snapshot(plan_id)
            .ok_or(FlowdError::PlanNotFound(plan_id))?;
        self.store.save_plan(&plan).await
    }

    fn with_plan_mut<R>(&self, plan_id: Uuid, f: impl FnOnce(&mut PlanRuntime) -> R) -> Result<R> {
        let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
            message: "plan store poisoned".into(),
            metrics: None,
        })?;
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
            let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
                message: "plan store poisoned".into(),
                metrics: None,
            })?;
            let runtime = guard
                .get_mut(&plan_id)
                .ok_or(FlowdError::PlanNotFound(plan_id))?;
            if runtime.plan.status != PlanStatus::Confirmed {
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{plan_id}` must be Confirmed before execute (currently {:?})",
                        runtime.plan.status
                    ),
                    metrics: None,
                });
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
        total_metrics: AgentMetrics,
        step_count: PlanStepCounts,
    ) -> Result<()> {
        let final_status = if cancel.load(Ordering::SeqCst) {
            PlanStatus::Cancelled
        } else if overall_failed {
            PlanStatus::Failed
        } else {
            PlanStatus::Completed
        };

        let completed_at = Utc::now();
        self.with_plan_mut(plan_id, |runtime| {
            runtime.plan.status = final_status;
            runtime.plan.completed_at = Some(completed_at);
            runtime.plan.steps.clone_from(&plan.steps);
            runtime.in_flight.clear();
        })?;
        self.persist_snapshot(plan_id).await?;

        // `AgentMetrics` is additive, so the zero-valued default means
        // "no step reported metrics"; collapse it back to `None` so
        // downstream consumers can render "no rollup" without inspecting
        // every field.
        let total_metrics = if total_metrics == AgentMetrics::default() {
            None
        } else {
            Some(total_metrics)
        };
        // Wall-clock from start_execution's `started_at` to the moment
        // we wrote `completed_at`. With parallel steps the sum of
        // per-step `duration_ms` can exceed this; the renderer surfaces
        // both so operators can see the speed-up.
        let elapsed_ms = plan.started_at.and_then(|s| {
            let delta = completed_at.signed_duration_since(s).num_milliseconds();
            u64::try_from(delta).ok()
        });
        self.emit(PlanEvent::Finished {
            plan_id,
            project: plan.project.clone(),
            status: final_status,
            total_metrics,
            step_count,
            elapsed_ms,
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
            let mut guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
                message: "plan store poisoned".into(),
                metrics: None,
            })?;
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
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{plan_id}` is not in Draft state (currently {:?})",
                        runtime.plan.status
                    ),
                    metrics: None,
                });
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
            default_step_timeout_secs: self.default_step_timeout_secs,
        };

        let mut overall_failed = false;
        let mut total_metrics = AgentMetrics::default();
        let mut step_count = PlanStepCounts::default();
        for layer in plan.execution_layers()? {
            // Cancellation can be set by an external `cancel(plan_id)`
            // call between layers, or by a `Cancelled` step outcome
            // within the previous layer.
            if cancel.load(Ordering::SeqCst) {
                break;
            }

            let outcome = runner.run(plan_id, &mut plan, &layer, &cancel).await?;
            total_metrics.merge(&outcome.metrics);
            step_count.completed = step_count
                .completed
                .saturating_add(outcome.step_count.completed);
            step_count.failed = step_count.failed.saturating_add(outcome.step_count.failed);

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

        self.finalize_execution(
            plan_id,
            &plan,
            &cancel,
            overall_failed,
            total_metrics,
            step_count,
        )
        .await
    }

    async fn status(&self, plan_id: Uuid) -> Result<Plan> {
        let guard = self.plans.lock().map_err(|_| FlowdError::PlanExecution {
            message: "plan store poisoned".into(),
            metrics: None,
        })?;
        guard
            .get(&plan_id)
            .map(|r| r.plan.clone())
            .ok_or(FlowdError::PlanNotFound(plan_id))
    }

    async fn list_plans(&self, project: Option<String>) -> Result<Vec<super::PlanSummary>> {
        self.store.list_plans(project.as_deref()).await
    }

    fn supports_worktree_isolation(&self) -> bool {
        self.spawner.supports_worktree_isolation()
    }

    async fn prepare_plan(&self, plan_id: Uuid) -> Result<PlanPreview> {
        let plan = self.status(plan_id).await?;
        if plan.status != PlanStatus::Draft {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "plan `{plan_id}` is not in Draft state (currently {:?})",
                    plan.status
                ),
                metrics: None,
            });
        }
        let preview = build_preview(&plan)?;
        self.spawner.prepare_plan(&plan, &preview).await?;
        Ok(preview)
    }

    async fn cancel(&self, plan_id: Uuid) -> Result<()> {
        // Three execution paths depending on where the plan is in its
        // lifecycle:
        //   * Terminal -> idempotent no-op.
        //   * Draft / Confirmed / Interrupted (no execute() task in flight)
        //     -> directly transition to Cancelled, persist, emit Finished.
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
            PlanStatus::Draft | PlanStatus::Confirmed | PlanStatus::Interrupted => {
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
                    total_metrics: None,
                    step_count: PlanStepCounts::default(),
                    // Cancelled before `start_execution` ever ran, so
                    // there is no wall-clock to report.
                    elapsed_ms: None,
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
            if !matches!(
                runtime.plan.status,
                PlanStatus::Failed | PlanStatus::Interrupted
            ) {
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{plan_id}` is in {:?} state; only Failed or Interrupted plans can be resumed",
                        runtime.plan.status
                    ),
                    metrics: None,
                });
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
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{plan_id}` is not in Draft state (currently {:?}); compile outputs only apply during clarification",
                        runtime.plan.status
                    ),
                    metrics: None,
                });
            }
            runtime.plan.apply_compile_output(output);
            Ok(())
        })??;
        self.persist_snapshot(plan_id).await
    }

    async fn invalidate_decision(&self, plan_id: Uuid, question_id: String) -> Result<Vec<String>> {
        let removed = self.with_plan_mut(plan_id, |runtime| {
            if runtime.plan.status != PlanStatus::Draft {
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{plan_id}` is not in Draft state (currently {:?}); cannot invalidate decisions",
                        runtime.plan.status
                    ),
                    metrics: None,
                });
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
        metrics: Option<AgentMetrics>,
    },
    Failed {
        error: String,
        metrics: Option<AgentMetrics>,
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
///
/// `default_timeout_secs` is the daemon-wide fallback applied when the
/// step itself left `timeout_secs` unset (prose-compiled plans always do).
/// The per-step value always wins via [`Option::or`]; passing `None`
/// preserves the historic "run forever when per-step is unset" behaviour.
pub(super) async fn run_step<S: AgentSpawner + ?Sized>(
    spawner: &S,
    ctx: AgentSpawnContext,
    step: PlanStep,
    cancel: Arc<AtomicBool>,
    default_timeout_secs: Option<u64>,
) -> StepOutcome {
    let attempts = step.retry_count.saturating_add(1);
    let mut last_error: Option<String> = None;
    let mut last_metrics: Option<AgentMetrics> = None;

    for _ in 0..attempts {
        if cancel.load(Ordering::SeqCst) {
            return StepOutcome::Cancelled;
        }

        let call = spawner.spawn(ctx.clone(), &step);
        let result = match step.timeout_secs.or(default_timeout_secs) {
            Some(secs) => match tokio::time::timeout(Duration::from_secs(secs), call).await {
                Ok(r) => r,
                Err(_) => Err(FlowdError::PlanExecution {
                    message: format!("step `{}` timed out after {secs}s", step.id),
                    metrics: None,
                }),
            },
            None => call.await,
        };

        match result {
            Ok(out) => {
                return StepOutcome::Completed {
                    output: out.stdout,
                    metrics: out.metrics,
                };
            }
            Err(e) => {
                if let FlowdError::PlanExecution { metrics, .. } = &e {
                    last_metrics.clone_from(metrics);
                }
                last_error = Some(e.to_string());
            }
        }
    }

    StepOutcome::Failed {
        error: last_error.unwrap_or_else(|| "unknown step failure".into()),
        metrics: last_metrics,
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
        StepOutcome::Completed { output, .. } => {
            step.status = StepStatus::Completed;
            step.output = Some(output.clone());
        }
        StepOutcome::Failed { error, .. } => {
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
                    .map_err(|_| FlowdError::PlanExecution {
                        message: "map store poisoned".into(),
                        metrics: None,
                    })?
                    .insert(plan.id, plan);
                Ok(())
            }
        }

        fn load_plan(&self, id: Uuid) -> impl Future<Output = Result<Option<Plan>>> + Send {
            let m = Arc::clone(&self.0);
            async move {
                Ok(m.lock()
                    .map_err(|_| FlowdError::PlanExecution {
                        message: "map store poisoned".into(),
                        metrics: None,
                    })?
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
                let guard = m.lock().map_err(|_| FlowdError::PlanExecution {
                    message: "map store poisoned".into(),
                    metrics: None,
                })?;
                let mut out: Vec<crate::orchestration::PlanSummary> = guard
                    .values()
                    .filter(|p| project.is_none_or(|pr| p.project == pr))
                    .map(|p| crate::orchestration::PlanSummary {
                        id: p.id,
                        name: p.name.clone(),
                        status: p.status,
                        created_at: p.created_at,
                        project: p.project.clone(),
                        project_root: p.project_root.clone(),
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
                    .map_err(|_| FlowdError::PlanExecution {
                        message: "map store poisoned".into(),
                        metrics: None,
                    })?
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
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
            self.invocations.fetch_add(1, Ordering::SeqCst);
            Ok(AgentOutput::success(format!("ran:{}", step.id)))
        }
    }

    /// Spawner that records the `project_root` field of every
    /// [`AgentSpawnContext`] it sees. The point is to pin that
    /// `LayerRunner` actually copies `Plan::project_root` onto the
    /// per-step context: if the wiring regresses, every spawn falls
    /// back to the daemon-process cwd and parallel runs land in the
    /// wrong workspace without anything erroring.
    struct ProjectRootCapturingSpawner {
        captured: Arc<Mutex<Vec<Option<std::path::PathBuf>>>>,
    }

    impl AgentSpawner for ProjectRootCapturingSpawner {
        async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
            self.captured
                .lock()
                .expect("captured poisoned")
                .push(ctx.project_root.clone());
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
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
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
        async fn spawn(&self, _ctx: AgentSpawnContext, _: &PlanStep) -> Result<AgentOutput> {
            self.attempts.fetch_add(1, Ordering::SeqCst);
            Err(FlowdError::PlanExecution {
                message: "nope".into(),
                metrics: None,
            })
        }
    }

    /// Spawner that panics on its first invocation.
    struct PanicSpawner;

    impl AgentSpawner for PanicSpawner {
        async fn spawn(&self, _ctx: AgentSpawnContext, _: &PlanStep) -> Result<AgentOutput> {
            panic!("boom");
        }
    }

    /// First global spawn attempt fails; later attempts succeed (for resume).
    struct FailOnceThenSucceed {
        n: Arc<AtomicUsize>,
    }

    impl AgentSpawner for FailOnceThenSucceed {
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
            let c = self.n.fetch_add(1, Ordering::SeqCst);
            if c == 0 {
                Err(FlowdError::PlanExecution {
                    message: "first run fails".into(),
                    metrics: None,
                })
            } else {
                Ok(AgentOutput::success(format!("ok:{}", step.id)))
            }
        }
    }

    /// Sleeps for a fixed duration before returning success. Used by the
    /// default-step-timeout tests to simulate a wedged agent without
    /// actually spinning.
    struct SleepSpawner {
        duration: Duration,
    }

    impl AgentSpawner for SleepSpawner {
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
            tokio::time::sleep(self.duration).await;
            Ok(AgentOutput::success(format!("ran:{}", step.id)))
        }
    }

    /// Blocks indefinitely on step id `"c"` (for partial multi-layer runs).
    struct BlockStepC;

    impl AgentSpawner for BlockStepC {
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
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

    /// `Plan::project_root` is the only signal the spawner has for the
    /// workspace the plan was authored against; if `LayerRunner` ever
    /// stops forwarding it, the spawner silently anchors every step to
    /// the daemon's process cwd. Pin the threading end-to-end:
    /// authored value -> persisted plan -> per-step
    /// `AgentSpawnContext.project_root` seen by the spawner.
    #[test]
    fn executor_threads_plan_project_root_into_spawn_context() {
        let captured: Arc<Mutex<Vec<Option<std::path::PathBuf>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let exec = InMemoryPlanExecutor::new(ProjectRootCapturingSpawner {
            captured: Arc::clone(&captured),
        });

        let plan = Plan::new("p", "proj", vec![step("a", &[]), step("b", &["a"])])
            .with_project_root("/abs/plan/workspace");

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
        });

        let seen = captured.lock().unwrap();
        assert_eq!(seen.len(), 2, "both steps must spawn");
        for (i, entry) in seen.iter().enumerate() {
            assert_eq!(
                entry.as_deref(),
                Some(std::path::Path::new("/abs/plan/workspace")),
                "step {i} saw ctx.project_root = {entry:?}; \
                 expected the plan's persisted /abs/plan/workspace",
            );
        }
    }

    /// Mirror image of the threading test: a plan WITHOUT a
    /// `project_root` (the legacy / pre-HL-PROJECT_ROOT shape) must
    /// surface as `None` on the context so the spawner can knowingly
    /// fall back. Pinning the negative case prevents a future change
    /// from defaulting `project_root` to `current_dir()` -- which
    /// would re-introduce the silent-daemon-cwd bug for legacy plans.
    #[test]
    fn executor_propagates_none_project_root_for_legacy_plans() {
        let captured: Arc<Mutex<Vec<Option<std::path::PathBuf>>>> =
            Arc::new(Mutex::new(Vec::new()));
        let exec = InMemoryPlanExecutor::new(ProjectRootCapturingSpawner {
            captured: Arc::clone(&captured),
        });

        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        assert!(plan.project_root.is_none(), "test precondition");

        rt().block_on(async {
            let id = exec.submit(plan).await.unwrap();
            exec.confirm(id).await.unwrap();
            exec.execute(id).await.unwrap();
        });

        let seen = captured.lock().unwrap();
        assert_eq!(seen.len(), 1);
        assert!(
            seen[0].is_none(),
            "ctx.project_root must be None for a plan with no project_root, got {:?}",
            seen[0]
        );
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
            assert!(matches!(err, FlowdError::PlanExecution { .. }));
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

    /// A prose-compiled plan leaves `step.timeout_secs = None`, so the
    /// daemon-wide fallback is the only thing standing between a wedged
    /// agent and an indefinitely-blocked execution layer. Pin the fire
    /// path so a regression surfaces before it lands in production.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn default_step_timeout_fires_when_per_step_is_none() {
        let spawner = SleepSpawner {
            duration: Duration::from_secs(5),
        };
        let exec = InMemoryPlanExecutor::new(spawner).with_default_step_timeout(Some(1));

        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        let id = plan.id;
        exec.submit(plan).await.unwrap();
        exec.confirm(id).await.unwrap();
        exec.execute(id).await.unwrap();

        let result = exec.status(id).await.unwrap();
        assert_eq!(result.status, PlanStatus::Failed);
        assert_eq!(result.steps[0].status, StepStatus::Failed);
        let err = result.steps[0].error.as_deref().unwrap_or_default();
        assert!(
            err.contains("timed out after 1s"),
            "expected fallback timeout error, got: {err}"
        );
    }

    /// The per-step field is the precise override: when it's set, the
    /// daemon-wide default must be ignored entirely. Pin the precedence
    /// so a future refactor doesn't silently invert it (e.g. accidentally
    /// using `min(per_step, default)` or the other way around).
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn per_step_timeout_overrides_default() {
        let spawner = SleepSpawner {
            duration: Duration::from_secs(5),
        };
        let exec = InMemoryPlanExecutor::new(spawner).with_default_step_timeout(Some(60));

        let mut s = step("a", &[]);
        s.timeout_secs = Some(2);
        let plan = Plan::new("p", "proj", vec![s]);
        let id = plan.id;
        exec.submit(plan).await.unwrap();
        exec.confirm(id).await.unwrap();
        exec.execute(id).await.unwrap();

        let result = exec.status(id).await.unwrap();
        assert_eq!(result.status, PlanStatus::Failed);
        assert_eq!(result.steps[0].status, StepStatus::Failed);
        let err = result.steps[0].error.as_deref().unwrap_or_default();
        assert!(
            err.contains("timed out after 2s"),
            "expected per-step timeout (not the 60s default), got: {err}"
        );
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
        assert_eq!(p.status, PlanStatus::Interrupted);
        assert_eq!(p.steps[0].status, StepStatus::Completed);
        assert_eq!(p.steps[1].status, StepStatus::Completed);
        assert_eq!(p.steps[2].status, StepStatus::Failed);
        assert!(
            p.steps[2]
                .error
                .as_deref()
                .is_some_and(|e| e.contains("daemon restarted"))
        );
    }

    #[tokio::test]
    async fn rehydrate_settles_running_plans_as_interrupted() {
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

        let store = MapPlanStore::default();
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.status = PlanStatus::Running;
        plan.started_at = Some(Utc::now());
        plan.steps[0].status = StepStatus::Running;
        plan.steps[0].started_at = Some(Utc::now());
        let id = plan.id;
        store.0.lock().unwrap().insert(id, plan);

        let obs = Arc::new(CollectingObserver::default());
        let exec = InMemoryPlanExecutor::with_plan_store(
            EchoSpawner {
                invocations: Arc::new(AtomicUsize::new(0)),
            },
            store.clone(),
        )
        .with_observer(Arc::clone(&obs) as Arc<dyn PlanObserver>);

        exec.rehydrate().await.unwrap();

        let snap = exec.status(id).await.unwrap();
        assert_eq!(snap.status, PlanStatus::Interrupted);
        assert!(snap.completed_at.is_some());
        assert_eq!(snap.steps[0].status, StepStatus::Failed);
        assert_eq!(
            snap.steps[0].error.as_deref(),
            Some("daemon restarted while step was in flight")
        );
        assert!(snap.steps[0].completed_at.is_some());

        let persisted = store.0.lock().unwrap().get(&id).unwrap().clone();
        assert_eq!(persisted.status, PlanStatus::Interrupted);
        assert_eq!(persisted.steps[0].status, StepStatus::Failed);
        assert_eq!(
            persisted.steps[0].error.as_deref(),
            Some("daemon restarted while step was in flight")
        );

        let events = obs.events.lock().unwrap().clone();
        let finished = events
            .iter()
            .find_map(|e| match e {
                PlanEvent::Finished {
                    plan_id,
                    status,
                    step_count,
                    ..
                } if *plan_id == id => Some((*status, *step_count)),
                _ => None,
            })
            .expect("rehydrate must emit a Finished event for settled Running plans");
        assert_eq!(finished.0, PlanStatus::Interrupted);
        assert_eq!(finished.1.failed, 1);
        assert_eq!(finished.1.completed, 0);
    }

    #[tokio::test]
    async fn rehydrate_preserves_draft_and_confirmed() {
        let store = MapPlanStore::default();

        let draft = Plan::new("draft", "proj", vec![step("a", &[])]);
        let draft_id = draft.id;
        assert_eq!(draft.status, PlanStatus::Draft);

        let mut confirmed = Plan::new("confirmed", "proj", vec![step("b", &[])]);
        confirmed.status = PlanStatus::Confirmed;
        let confirmed_id = confirmed.id;

        store.0.lock().unwrap().insert(draft_id, draft);
        store.0.lock().unwrap().insert(confirmed_id, confirmed);

        let exec = InMemoryPlanExecutor::with_plan_store(
            EchoSpawner {
                invocations: Arc::new(AtomicUsize::new(0)),
            },
            store.clone(),
        );
        exec.rehydrate().await.unwrap();

        let draft_snap = exec.status(draft_id).await.unwrap();
        assert_eq!(draft_snap.status, PlanStatus::Draft);
        let confirmed_snap = exec.status(confirmed_id).await.unwrap();
        assert_eq!(confirmed_snap.status, PlanStatus::Confirmed);

        let persisted = store.0.lock().unwrap();
        assert_eq!(persisted.get(&draft_id).unwrap().status, PlanStatus::Draft);
        assert_eq!(
            persisted.get(&confirmed_id).unwrap().status,
            PlanStatus::Confirmed,
        );
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
                PlanEvent::ClarificationOpened { .. } => "clarification_opened",
                PlanEvent::ClarificationResolved { .. } => "clarification_resolved",
                PlanEvent::RefinementApplied { .. } => "refinement_applied",
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
            PlanEvent::Finished {
                status,
                step_count,
                total_metrics,
                ..
            } => {
                assert_eq!(*status, PlanStatus::Completed);
                assert_eq!(step_count.completed, 2);
                assert_eq!(step_count.failed, 0);
                // EchoSpawner reports no metrics, so the rollup collapses
                // back to `None` at finalise time.
                assert!(total_metrics.is_none());
            }
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
            project_root: None,
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
            assert!(
                matches!(err, FlowdError::PlanExecution { message, .. } if message.contains("Draft"))
            );
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

    /// Graceful shutdown contract: calling `cancel` on a plan whose step
    /// is wedged in a long sleep must unblock and settle the plan as
    /// `Cancelled` well within the 10s budget the daemon's shutdown path
    /// allows. The spawner sleeps 30s so a regression that forgot to
    /// abort the in-flight task would miss the 2s assertion by an order
    /// of magnitude.
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn cancel_settles_long_running_step_within_budget() {
        let spawner = SleepSpawner {
            duration: Duration::from_secs(30),
        };
        let exec = Arc::new(InMemoryPlanExecutor::new(spawner));

        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        let id = plan.id;
        exec.submit(plan).await.unwrap();
        exec.confirm(id).await.unwrap();

        let exec_for_task = Arc::clone(&exec);
        let jh = tokio::spawn(async move { exec_for_task.execute(id).await });

        // Wait until the executor has flipped the plan to Running; any
        // cancellation issued before this point takes the Confirmed
        // direct-transition branch, which is not what we want to test.
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if exec.status(id).await.unwrap().status == PlanStatus::Running {
                break;
            }
            assert!(
                std::time::Instant::now() < deadline,
                "plan never reached Running",
            );
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert_eq!(exec.list_running(), vec![id]);

        tokio::time::timeout(Duration::from_secs(2), async {
            exec.cancel(id).await.unwrap();
            jh.await.unwrap().unwrap();
        })
        .await
        .expect("cancel + execute must settle within 2s");

        let snap = exec.status(id).await.unwrap();
        assert_eq!(snap.status, PlanStatus::Cancelled);
        assert!(exec.list_running().is_empty());
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

    // ---------- AgentMetrics ----------

    fn model_usage(
        input: u64,
        output: u64,
        cache_create: u64,
        cache_read: u64,
        cost: f64,
    ) -> ModelUsage {
        ModelUsage {
            input_tokens: input,
            output_tokens: output,
            cache_creation_input_tokens: cache_create,
            cache_read_input_tokens: cache_read,
            cost_usd: cost,
        }
    }

    #[test]
    fn agent_metrics_merge_sums_numeric_fields() {
        let mut a = AgentMetrics {
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens: 3,
            cache_read_input_tokens: 4,
            total_cost_usd: 0.5,
            duration_ms: 100,
            duration_api_ms: 80,
            model_usage: BTreeMap::new(),
        };
        let b = AgentMetrics {
            input_tokens: 1,
            output_tokens: 2,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 6,
            total_cost_usd: 1.25,
            duration_ms: 10,
            duration_api_ms: 7,
            model_usage: BTreeMap::new(),
        };

        a.merge(&b);

        assert_eq!(a.input_tokens, 11);
        assert_eq!(a.output_tokens, 22);
        assert_eq!(a.cache_creation_input_tokens, 8);
        assert_eq!(a.cache_read_input_tokens, 10);
        assert!((a.total_cost_usd - 1.75).abs() < f64::EPSILON);
        assert_eq!(a.duration_ms, 110);
        assert_eq!(a.duration_api_ms, 87);
    }

    #[test]
    fn agent_metrics_merge_unions_model_usage_with_no_duplicates() {
        let mut a = AgentMetrics::default();
        a.model_usage
            .insert("sonnet".into(), model_usage(10, 20, 0, 0, 0.1));
        a.model_usage
            .insert("haiku".into(), model_usage(1, 2, 0, 0, 0.01));

        let mut b = AgentMetrics::default();
        // Collides with `sonnet`: fields should sum.
        b.model_usage
            .insert("sonnet".into(), model_usage(5, 7, 2, 3, 0.4));
        // New model: should be inserted as-is.
        b.model_usage
            .insert("opus".into(), model_usage(100, 200, 0, 0, 1.0));

        a.merge(&b);

        assert_eq!(a.model_usage.len(), 3, "no duplicate keys");
        assert_eq!(
            a.model_usage.get("sonnet").unwrap(),
            &model_usage(15, 27, 2, 3, 0.5),
            "collision must sum all fields"
        );
        assert_eq!(
            a.model_usage.get("haiku").unwrap(),
            &model_usage(1, 2, 0, 0, 0.01),
            "untouched entry survives"
        );
        assert_eq!(
            a.model_usage.get("opus").unwrap(),
            &model_usage(100, 200, 0, 0, 1.0),
            "new entry inserted"
        );
    }

    /// Spawner that returns a preconfigured (output-or-error, metrics)
    /// pair per step id. Lets a single test drive a plan where one
    /// step succeeds with metrics and another fails with metrics, then
    /// verify the executor accumulated both into the `Finished` event.
    struct MetricsScriptedSpawner {
        script: HashMap<String, std::result::Result<AgentMetrics, AgentMetrics>>,
    }

    impl AgentSpawner for MetricsScriptedSpawner {
        async fn spawn(&self, _ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
            match self.script.get(&step.id) {
                Some(Ok(metrics)) => Ok(AgentOutput {
                    stdout: format!("ok:{}", step.id),
                    exit_code: Some(0),
                    metrics: Some(metrics.clone()),
                }),
                Some(Err(metrics)) => Err(FlowdError::PlanExecution {
                    message: format!("scripted failure for {}", step.id),
                    metrics: Some(metrics.clone()),
                }),
                None => panic!("no scripted outcome for step `{}`", step.id),
            }
        }
    }

    /// End-to-end rollup check: a plan with both a successful step and a
    /// failing step sees its `Finished` event carry `total_metrics` equal
    /// to the element-wise sum of the per-step metrics emitted on the
    /// intervening `StepCompleted` / `StepFailed` events. This pins the
    /// accumulation contract the CLI / MCP renderers depend on.
    #[test]
    fn finished_event_aggregates_metrics_across_success_and_failure_steps() {
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

        let ok_metrics = AgentMetrics {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 10,
            cache_read_input_tokens: 40,
            total_cost_usd: 0.25,
            duration_ms: 1_000,
            duration_api_ms: 800,
            model_usage: BTreeMap::new(),
        };
        let fail_metrics = AgentMetrics {
            input_tokens: 20,
            output_tokens: 5,
            cache_creation_input_tokens: 2,
            cache_read_input_tokens: 8,
            total_cost_usd: 0.04,
            duration_ms: 200,
            duration_api_ms: 150,
            model_usage: BTreeMap::new(),
        };

        let mut script: HashMap<String, std::result::Result<AgentMetrics, AgentMetrics>> =
            HashMap::new();
        script.insert("ok".into(), Ok(ok_metrics.clone()));
        script.insert("bad".into(), Err(fail_metrics.clone()));

        let obs = Arc::new(CollectingObserver::default());
        let exec = InMemoryPlanExecutor::new(MetricsScriptedSpawner { script })
            .with_observer(Arc::clone(&obs) as Arc<dyn PlanObserver>);

        // Two independent steps so both outcomes contribute within a
        // single layer run; any downstream step would be skipped once
        // the layer fails and muddy the rollup assertion.
        let plan = Plan::new("p", "proj", vec![step("ok", &[]), step("bad", &[])]);
        let plan_id = plan.id;

        rt().block_on(async {
            exec.submit(plan).await.unwrap();
            exec.confirm(plan_id).await.unwrap();
            exec.execute(plan_id).await.unwrap();
        });

        let events = obs.events.lock().unwrap().clone();
        // Cross-check the per-step events carried the individual metrics
        // we scripted, so if the rollup assertion fails we can tell
        // whether the bug is in the spawner wiring or the accumulator.
        let mut saw_ok_step_metrics = false;
        let mut saw_bad_step_metrics = false;
        for e in &events {
            match e {
                PlanEvent::StepCompleted {
                    step_id, metrics, ..
                } if step_id == "ok" => {
                    assert_eq!(metrics.as_ref(), Some(&ok_metrics));
                    saw_ok_step_metrics = true;
                }
                PlanEvent::StepFailed {
                    step_id, metrics, ..
                } if step_id == "bad" => {
                    assert_eq!(metrics.as_ref(), Some(&fail_metrics));
                    saw_bad_step_metrics = true;
                }
                _ => {}
            }
        }
        assert!(saw_ok_step_metrics && saw_bad_step_metrics);

        let finished = events
            .iter()
            .find_map(|e| match e {
                PlanEvent::Finished {
                    status,
                    total_metrics,
                    step_count,
                    ..
                } => Some((*status, total_metrics.clone(), *step_count)),
                _ => None,
            })
            .expect("Finished event must be present");
        assert_eq!(finished.0, PlanStatus::Failed);
        assert_eq!(finished.2.completed, 1);
        assert_eq!(finished.2.failed, 1);

        let mut expected = ok_metrics.clone();
        expected.merge(&fail_metrics);
        let total = finished
            .1
            .expect("total_metrics must be present when any step reported metrics");
        assert_eq!(total, expected);
    }

    #[test]
    fn agent_metrics_serde_roundtrip_all_zero_default() {
        let m = AgentMetrics::default();
        let json = serde_json::to_string(&m).unwrap();
        // Shape matters more than exact ordering for BTreeMap (which is stable).
        assert_eq!(
            json,
            r#"{"input_tokens":0,"output_tokens":0,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"total_cost_usd":0.0,"duration_ms":0,"duration_api_ms":0,"model_usage":{}}"#
        );
        let back: AgentMetrics = serde_json::from_str(&json).unwrap();
        assert_eq!(back, m);
    }
}
