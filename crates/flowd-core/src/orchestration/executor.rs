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
//! `.await` boundary -- so the executor remains responsive under load.

use std::collections::HashMap;
use std::future::Future;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::task::{AbortHandle, JoinHandle};
use uuid::Uuid;

use crate::error::{FlowdError, Result};

use super::{
    Plan, PlanExecutor, PlanPreview, PlanStatus, PlanStep, StepStatus, build_preview, validate_plan,
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
struct PlanRuntime {
    plan: Plan,
    cancel: Arc<AtomicBool>,
    /// Abort handles for the currently in-flight step tasks. Populated by
    /// `execute` at every layer boundary; consumed by `cancel`. We store
    /// `AbortHandle` (cheap to clone, can be shared across owners) rather
    /// than the `JoinHandle` itself, which `execute` needs to keep so it can
    /// `.await` each task.
    in_flight: Vec<AbortHandle>,
}

/// Default executor. Generic over the spawner so the type system encodes the
/// concrete agent-invocation strategy chosen at composition time -- no
/// dynamic dispatch on the hot path.
pub struct InMemoryPlanExecutor<S: AgentSpawner + 'static> {
    spawner: Arc<S>,
    plans: Arc<Mutex<HashMap<Uuid, PlanRuntime>>>,
}

impl<S: AgentSpawner + 'static> InMemoryPlanExecutor<S> {
    pub fn new(spawner: S) -> Self {
        Self {
            spawner: Arc::new(spawner),
            plans: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn from_shared(spawner: Arc<S>) -> Self {
        Self {
            spawner,
            plans: Arc::new(Mutex::new(HashMap::new())),
        }
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
}

impl<S: AgentSpawner + 'static> PlanExecutor for InMemoryPlanExecutor<S> {
    fn validate(&self, plan: &Plan) -> Result<()> {
        validate_plan(plan)
    }

    fn preview(&self, plan: &Plan) -> Result<PlanPreview> {
        build_preview(plan)
    }

    async fn submit(&self, plan: Plan) -> Result<Uuid> {
        validate_plan(&plan)?;
        let id = plan.id;
        let runtime = PlanRuntime {
            plan,
            cancel: Arc::new(AtomicBool::new(false)),
            in_flight: Vec::new(),
        };
        let mut guard = self
            .plans
            .lock()
            .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
        guard.insert(id, runtime);
        Ok(id)
    }

    async fn confirm(&self, plan_id: Uuid) -> Result<PlanPreview> {
        self.with_plan_mut(plan_id, |runtime| {
            if runtime.plan.status != PlanStatus::Draft {
                return Err(FlowdError::PlanExecution(format!(
                    "plan `{plan_id}` is not in Draft state (currently {:?})",
                    runtime.plan.status
                )));
            }
            let preview = build_preview(&runtime.plan)?;
            runtime.plan.status = PlanStatus::Confirmed;
            Ok(preview)
        })?
    }

    async fn execute(&self, plan_id: Uuid) -> Result<()> {
        // ---- Pre-flight: take an owned plan snapshot + the cancel flag ----
        let (mut plan, cancel) = {
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

        let layers = plan.execution_layers()?;
        let mut overall_failed = false;

        'outer: for layer in layers {
            if cancel.load(Ordering::SeqCst) {
                break 'outer;
            }

            // Spawn all steps in this layer in parallel.
            let mut handles: Vec<(String, JoinHandle<StepOutcome>)> =
                Vec::with_capacity(layer.len());
            for step_id in &layer {
                let Some(step) = plan.steps.iter().find(|s| &s.id == step_id).cloned() else {
                    continue;
                };
                let spawner = Arc::clone(&self.spawner);
                let cancel = Arc::clone(&cancel);
                let handle = tokio::spawn(async move { run_step(&*spawner, step, cancel).await });
                handles.push((step_id.clone(), handle));
            }

            // Publish abort handles so `cancel` can terminate this layer's
            // in-flight tasks. The `JoinHandle`s themselves stay below for
            // `.await`, since they cannot be cloned.
            {
                let mut guard = self
                    .plans
                    .lock()
                    .map_err(|_| FlowdError::PlanExecution("plan store poisoned".into()))?;
                if let Some(runtime) = guard.get_mut(&plan_id) {
                    runtime.in_flight = handles.iter().map(|(_, h)| h.abort_handle()).collect();
                }
            }

            for (step_id, handle) in handles {
                let outcome = match handle.await {
                    Ok(o) => o,
                    Err(join_err) if join_err.is_cancelled() => StepOutcome::Cancelled,
                    Err(join_err) if join_err.is_panic() => StepOutcome::Failed {
                        error: format!("step `{step_id}` panicked: {join_err}"),
                    },
                    Err(join_err) => StepOutcome::Failed {
                        error: format!("step `{step_id}` join error: {join_err}"),
                    },
                };

                apply_outcome(&mut plan, &step_id, &outcome);

                match outcome {
                    StepOutcome::Failed { .. } => overall_failed = true,
                    StepOutcome::Cancelled => {
                        cancel.store(true, Ordering::SeqCst);
                    }
                    StepOutcome::Completed { .. } => {}
                }
            }

            // Persist incremental state after every layer so `status()` calls
            // observe progress in real time.
            self.with_plan_mut(plan_id, |runtime| {
                runtime.plan.steps.clone_from(&plan.steps);
            })?;

            if overall_failed {
                break 'outer;
            }
        }

        // ---- Finalise ----
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

        Ok(())
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
        let handles = self.with_plan_mut(plan_id, |runtime| {
            runtime.cancel.store(true, Ordering::SeqCst);
            // Abort everything currently in flight; new layers won't start
            // because `cancel` is now `true`.
            std::mem::take(&mut runtime.in_flight)
        })?;
        for h in handles {
            h.abort();
        }
        Ok(())
    }
}

/// Outcome of a single step's execution attempt.
enum StepOutcome {
    Completed { output: String },
    Failed { error: String },
    Cancelled,
}

/// Execute one step, honouring its retry / timeout configuration. Runs inside
/// a `tokio::spawn`'d task, so panics surface as `JoinError::is_panic()` to
/// the caller.
async fn run_step<S: AgentSpawner + ?Sized>(
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

fn apply_outcome(plan: &mut Plan, step_id: &str, outcome: &StepOutcome) {
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
            step.status = StepStatus::Skipped;
            step.error = Some("cancelled".into());
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

    fn rt() -> tokio::runtime::Runtime {
        tokio::runtime::Builder::new_current_thread()
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
                .submit(Plan::new("p", vec![step("a", &[])]))
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
        let plan = Plan::new("p", vec![s]);

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
        let plan = Plan::new("p", vec![step("a", &[])]);
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

        let plan = Plan::new("p", vec![step("a", &[]), step("b", &["a"])]);

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
}
