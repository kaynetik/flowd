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

use chrono::Utc;
use serde::{Deserialize, Serialize};
use tokio::task::{AbortHandle, JoinHandle};
use uuid::Uuid;

use crate::error::{FlowdError, Result};

use super::{
    NoOpPlanStore, Plan, PlanExecutor, PlanPreview, PlanStatus, PlanStep, PlanStore, StepStatus,
    build_preview, validate_plan,
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

/// Default executor. Generic over the spawner and an optional [`PlanStore`].
pub struct InMemoryPlanExecutor<S: AgentSpawner + 'static, PS: PlanStore = NoOpPlanStore> {
    spawner: Arc<S>,
    plans: Arc<Mutex<HashMap<Uuid, PlanRuntime>>>,
    store: PS,
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
        }
    }
}

impl<S: AgentSpawner + 'static, PS: PlanStore> InMemoryPlanExecutor<S, PS> {
    pub fn with_plan_store(spawner: S, store: PS) -> Self {
        Self {
            spawner: Arc::new(spawner),
            plans: Arc::new(Mutex::new(HashMap::new())),
            store,
        }
    }

    pub fn from_shared_with_store(spawner: Arc<S>, store: PS) -> Self {
        Self {
            spawner,
            plans: Arc::new(Mutex::new(HashMap::new())),
            store,
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
}

impl<S: AgentSpawner + 'static, PS: PlanStore> PlanExecutor for InMemoryPlanExecutor<S, PS> {
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

        self.persist_snapshot(plan_id).await?;

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
            self.persist_snapshot(plan_id).await?;
            // Allow other tasks (and tests) to observe state between DAG layers.
            tokio::task::yield_now().await;

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
        self.persist_snapshot(plan_id).await?;

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
                    .filter(|p| project.is_none_or(|pr| p.project.as_deref() == Some(pr)))
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn survives_restart() {
        let store = MapPlanStore::default();
        let spawner = BlockStepC;
        let exec1 = InMemoryPlanExecutor::with_plan_store(spawner, store.clone());

        let plan = Plan::new(
            "p",
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
    fn resume_resets_failed_steps() {
        let store = MapPlanStore::default();
        let spawner = FailOnceThenSucceed {
            n: Arc::new(AtomicUsize::new(0)),
        };
        let exec = InMemoryPlanExecutor::with_plan_store(spawner, store);

        let plan = Plan::new("p", vec![step("a", &[]), step("b", &["a"])]);

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
}
