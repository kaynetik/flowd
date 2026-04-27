//! MCP `plan_integrate` tool contract tests.
//!
//! Wires `FlowdHandlers` to:
//!   * a real `SqliteBackend` + in-memory vector / embedder stubs,
//!   * the in-crate `EchoSpawner` for plan execution,
//!   * a `StubIntegrator` that replays the pure `assess_eligibility`
//!     contract from `flowd-core` (so eligibility refusals surface
//!     verbatim) but stops short of running git -- staging just echoes
//!     a deterministic integration tip.
//!
//! Pinning the wiring here means the daemon's real integrator
//! (`flowd_cli::integration::PlanIntegrator`) can change its git plumbing
//! without breaking the MCP-side contract: the handler must still route,
//! refuse non-`Completed` plans, and short-circuit already-promoted plans.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use tempfile::TempDir;
use uuid::Uuid;

use flowd_core::error::Result as FlowdResult;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, VectorIndex};
use flowd_core::orchestration::integration::{
    CleanupPolicy, IntegrationMetadata, IntegrationMode, IntegrationStatus, PlanIntegrateOutcome,
    PlanIntegrateRequest, assess_eligibility, integration_branch_ref,
};
use flowd_core::orchestration::{
    AgentOutput, AgentSpawnContext, AgentSpawner, InMemoryPlanExecutor, Plan, PlanExecutor,
    PlanStatus, PlanStep, StepStatus,
};
use flowd_core::rules::InMemoryRuleEvaluator;
use flowd_core::types::Embedding;
use flowd_mcp::integration::{IntegrationDriver, IntegrationFuture};
use flowd_mcp::tools::PlanIntegrateParams;
use flowd_mcp::{FlowdHandlers, McpHandlers, StubPlanCompiler};
use flowd_storage::plan_store::SqlitePlanStore;
use flowd_storage::sqlite::SqliteBackend;

// ---------- Test doubles --------------------------------------------------

struct HashEmbedder;

impl EmbeddingProvider for HashEmbedder {
    fn embed(&self, text: &str) -> FlowdResult<Vec<f32>> {
        #[allow(clippy::cast_precision_loss)]
        let sum = text.bytes().map(u32::from).sum::<u32>() as f32;
        Ok(vec![sum, sum * 0.5, sum * 0.25, sum * 0.125])
    }
    fn embed_batch(&self, texts: &[&str]) -> FlowdResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    fn dimensions(&self) -> usize {
        4
    }
}

#[derive(Default)]
struct MemVectors {
    inner: Mutex<HashMap<Uuid, String>>,
}

impl VectorIndex for MemVectors {
    async fn upsert(&self, embedding: &Embedding) -> FlowdResult<()> {
        self.inner
            .lock()
            .unwrap()
            .insert(embedding.observation_id, embedding.project.clone());
        Ok(())
    }
    async fn search(
        &self,
        _query_vector: &[f32],
        _limit: usize,
        _project_filter: Option<&str>,
    ) -> FlowdResult<Vec<(Uuid, f64)>> {
        Ok(Vec::new())
    }
    async fn delete(&self, observation_id: Uuid) -> FlowdResult<()> {
        self.inner.lock().unwrap().remove(&observation_id);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct EchoSpawner;

impl AgentSpawner for EchoSpawner {
    fn spawn(
        &self,
        _ctx: AgentSpawnContext,
        step: &PlanStep,
    ) -> impl Future<Output = FlowdResult<AgentOutput>> + Send {
        let id = step.id.clone();
        async move { Ok(AgentOutput::success(format!("ran:{id}"))) }
    }
}

/// Driver double that runs the pure eligibility contract, then synthesises
/// a deterministic git outcome instead of touching a real repository.
/// Refusals are surfaced as `IntegrationError::Refusal` exactly like the
/// production driver does, so the handler-level mapping stays under test.
struct StubIntegrator;

impl IntegrationDriver for StubIntegrator {
    fn integrate<'a>(
        &'a self,
        plan: &'a Plan,
        request: &'a PlanIntegrateRequest,
    ) -> IntegrationFuture<'a> {
        Box::pin(async move {
            let intended = assess_eligibility(plan, request)?;
            if intended.mode == IntegrationMode::DryRun {
                return Ok(PlanIntegrateOutcome::DryRun { intended });
            }
            Ok(PlanIntegrateOutcome::Staged {
                intended,
                integration_tip: "deadbeef".into(),
            })
        })
    }

    fn promote<'a>(
        &'a self,
        plan: &'a Plan,
        request: &'a PlanIntegrateRequest,
    ) -> IntegrationFuture<'a> {
        Box::pin(async move {
            let intended = assess_eligibility(plan, request)?;
            Ok(PlanIntegrateOutcome::Promoted {
                intended,
                promoted_tip: "deadbeef".into(),
            })
        })
    }
}

// ---------- Harness -------------------------------------------------------

type Handlers = FlowdHandlers<
    SqliteBackend,
    MemVectors,
    HashEmbedder,
    InMemoryPlanExecutor<EchoSpawner, SqlitePlanStore>,
    StubPlanCompiler,
    InMemoryRuleEvaluator,
>;

struct Harness {
    _tmp: TempDir,
    handlers: Arc<Handlers>,
    executor: Arc<InMemoryPlanExecutor<EchoSpawner, SqlitePlanStore>>,
}

fn build_harness_with_integrator(integrator: Option<Arc<dyn IntegrationDriver>>) -> Harness {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("flowd.db");
    let backend = SqliteBackend::open(&db_path).expect("open sqlite");
    let plan_store = backend.plan_store();
    let memory = Arc::new(MemoryService::new(
        backend,
        MemVectors::default(),
        HashEmbedder,
    ));
    let executor = Arc::new(InMemoryPlanExecutor::with_plan_store(
        EchoSpawner,
        plan_store,
    ));
    let rules = Arc::new(InMemoryRuleEvaluator::new());
    let compiler = Arc::new(StubPlanCompiler::new());
    let mut handlers = FlowdHandlers::new(memory, Arc::clone(&executor), compiler, rules);
    if let Some(driver) = integrator {
        handlers = handlers.with_integrator(driver);
    }
    Harness {
        _tmp: tmp,
        handlers: Arc::new(handlers),
        executor,
    }
}

fn build_harness() -> Harness {
    build_harness_with_integrator(Some(Arc::new(StubIntegrator)))
}

// ---------- Plan fixtures -------------------------------------------------

fn completed_step(id: &str) -> PlanStep {
    PlanStep {
        id: id.into(),
        agent_type: "echo".into(),
        prompt: format!("do {id}"),
        depends_on: vec![],
        timeout_secs: None,
        retry_count: 0,
        status: StepStatus::Completed,
        output: Some("ok".into()),
        error: None,
        started_at: None,
        completed_at: None,
    }
}

fn failed_step(id: &str) -> PlanStep {
    PlanStep {
        id: id.into(),
        agent_type: "echo".into(),
        prompt: format!("do {id}"),
        depends_on: vec![],
        timeout_secs: None,
        retry_count: 0,
        status: StepStatus::Failed,
        output: None,
        error: Some("boom".into()),
        started_at: None,
        completed_at: None,
    }
}

async fn submit_plan(h: &Harness, name: &str, status: PlanStatus, steps: Vec<PlanStep>) -> Uuid {
    let mut plan = Plan::new(name, "rnd", steps);
    plan.status = status;
    h.executor.submit(plan).await.expect("submit plan")
}

// ---------- Tests --------------------------------------------------------

#[tokio::test]
async fn plan_integrate_happy_path_returns_staged_outcome() {
    let h = build_harness();
    let plan_id = submit_plan(
        &h,
        "happy",
        PlanStatus::Completed,
        vec![completed_step("a")],
    )
    .await;

    let payload = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: None,
            promote: false,
            cleanup: None,
        })
        .await
        .expect("plan_integrate succeeds for a Completed plan");

    assert_eq!(payload["kind"], "staged", "stub integrator returns Staged");
    assert_eq!(payload["integration_tip"], "deadbeef");
    let intended = &payload["intended"];
    assert_eq!(intended["base_branch"], "main");
    assert_eq!(
        intended["integration_branch"],
        integration_branch_ref("rnd", plan_id)
    );
    assert_eq!(intended["cherry_picks"][0]["step_id"], "a");
}

#[tokio::test]
async fn plan_integrate_dry_run_returns_dry_run_outcome() {
    // dry_run is one of two `mode` values in the schema; pin it so a
    // schema/handler drift on the enum value surfaces here.
    let h = build_harness();
    let plan_id = submit_plan(&h, "dry", PlanStatus::Completed, vec![completed_step("a")]).await;

    let payload = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: Some("dry_run".into()),
            promote: false,
            cleanup: None,
        })
        .await
        .expect("dry_run plan_integrate succeeds");

    assert_eq!(payload["kind"], "dry_run");
    assert!(
        payload.get("integration_tip").is_none(),
        "dry_run reports intended only, no tip"
    );
}

#[tokio::test]
async fn plan_integrate_refuses_non_completed_plan() {
    // Pure-contract refusal: a Failed plan must not reach the
    // integrator's git plumbing. The handler propagates the
    // `IntegrationRefusal::PlanStatus` variant as a structured
    // PlanValidation error.
    let h = build_harness();
    let plan_id = submit_plan(&h, "nope", PlanStatus::Failed, vec![failed_step("a")]).await;

    let err = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: None,
            promote: false,
            cleanup: None,
        })
        .await
        .expect_err("non-Completed plan must refuse");

    let msg = format!("{err}");
    assert!(
        msg.contains("plan_integrate refusal"),
        "expected refusal prefix, got: {msg}"
    );
    assert!(
        msg.contains("Completed") || msg.contains("Failed"),
        "expected eligibility cause in: {msg}"
    );
}

#[tokio::test]
async fn plan_integrate_is_idempotent_for_already_promoted_plans() {
    // Re-invoking on a plan whose persisted integration is `Promoted`
    // must short-circuit *before* the driver is consulted. We prove
    // that by wiring an integrator that panics on call; the assertion
    // is that the handler returns a structured `already_promoted`
    // payload without ever reaching it.
    struct PanicIntegrator;
    impl IntegrationDriver for PanicIntegrator {
        fn integrate<'a>(
            &'a self,
            _plan: &'a Plan,
            _request: &'a PlanIntegrateRequest,
        ) -> IntegrationFuture<'a> {
            Box::pin(async { panic!("idempotent call must not reach the driver") })
        }
        fn promote<'a>(
            &'a self,
            _plan: &'a Plan,
            _request: &'a PlanIntegrateRequest,
        ) -> IntegrationFuture<'a> {
            Box::pin(async { panic!("idempotent call must not reach the driver") })
        }
    }
    let h = build_harness_with_integrator(Some(Arc::new(PanicIntegrator)));

    let mut plan = Plan::new("done", "rnd", vec![completed_step("a")]);
    plan.status = PlanStatus::Completed;
    plan.integration = Some(IntegrationMetadata {
        status: IntegrationStatus::Promoted,
        integration_branch: integration_branch_ref("rnd", plan.id),
        base_branch: "main".into(),
        mode: IntegrationMode::Confirm,
        cleanup: CleanupPolicy::KeepOnFailure,
        started_at: None,
        completed_at: None,
        failure: None,
        refusal: None,
    });
    let plan_id = plan.id;
    h.executor.submit(plan).await.expect("submit promoted plan");

    let payload = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: None,
            promote: false,
            cleanup: None,
        })
        .await
        .expect("idempotent call succeeds");

    assert_eq!(payload["kind"], "already_promoted");
    assert_eq!(payload["integration"]["status"], "promoted");
}

#[tokio::test]
async fn plan_integrate_without_driver_returns_structured_error() {
    // The handler must not silently accept calls when no driver is
    // bound: a failure surfaces a `PlanExecution` error with an
    // actionable message so callers know the surface is unconfigured
    // (vs. having a real integrator refuse for plan-state reasons).
    let h = build_harness_with_integrator(None);
    let plan_id = submit_plan(
        &h,
        "noint",
        PlanStatus::Completed,
        vec![completed_step("a")],
    )
    .await;

    let err = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: None,
            promote: false,
            cleanup: None,
        })
        .await
        .expect_err("missing driver must error");

    let msg = format!("{err}");
    assert!(
        msg.contains("no integration driver"),
        "expected actionable message, got: {msg}"
    );
}

#[tokio::test]
async fn plan_integrate_refuses_promote_with_dry_run() {
    let h = build_harness();
    let plan_id = submit_plan(
        &h,
        "bad-combo",
        PlanStatus::Completed,
        vec![completed_step("a")],
    )
    .await;

    let err = h
        .handlers
        .plan_integrate(PlanIntegrateParams {
            plan_id: plan_id.to_string(),
            base_branch: "main".into(),
            mode: Some("dry_run".into()),
            promote: true,
            cleanup: None,
        })
        .await
        .expect_err("promote+dry_run is a contradiction");

    assert!(format!("{err}").contains("dry_run"));
}
