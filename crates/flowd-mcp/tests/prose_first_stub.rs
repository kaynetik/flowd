//! End-to-end coverage of the prose-first surface against the daemon's
//! default [`StubPlanCompiler`].
//!
//! `tests/prose_first.rs` already pins the cross-crate contract using a
//! scripted [`MockPlanCompiler`] -- those tests prove the wiring
//! (handlers, executor, observer, persistence). This file proves the
//! *real* compiler the daemon ships with: structured-markdown prose flows
//! all the way from `plan_create(prose: ...)` through `plan_confirm` to
//! a terminal `plan_status`, with no LLM in the loop.
//!
//! Two scenarios:
//!
//! 1. Structured prose -> plan compiles in one shot, runs, finishes.
//! 2. Freeform prose   -> stub surfaces its `stub.structure_required`
//!    open question, `plan_refine` with structured feedback closes the
//!    loop, then `plan_confirm` advances and execution finishes.
//!
//! Both scenarios exercise the same wire path the running daemon uses,
//! so any regression in the `StubPlanCompiler` -> handler -> executor seam
//! shows up here.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::Value;
use tempfile::TempDir;
use uuid::Uuid;

use flowd_core::error::Result as FlowdResult;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, VectorIndex};
use flowd_core::orchestration::{AgentOutput, AgentSpawner, InMemoryPlanExecutor, PlanStep};
use flowd_core::rules::InMemoryRuleEvaluator;
use flowd_core::types::Embedding;
use flowd_mcp::tools::{
    PlanConfirmParams, PlanCreateParams, PlanRefineParams, PlanStatusParams,
};
use flowd_mcp::{FlowdHandlers, McpHandlers, StubPlanCompiler};
use flowd_storage::plan_store::SqlitePlanStore;
use flowd_storage::sqlite::SqliteBackend;

// ---------- Test doubles (mirrors tests/prose_first.rs) -------------------

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
    fn spawn(&self, step: &PlanStep) -> impl Future<Output = FlowdResult<AgentOutput>> + Send {
        let id = step.id.clone();
        async move { Ok(AgentOutput::success(format!("ran:{id}"))) }
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
}

fn build_harness() -> Harness {
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
    let handlers = Arc::new(FlowdHandlers::new(memory, executor, compiler, rules));
    Harness {
        _tmp: tmp,
        handlers,
    }
}

async fn poll_until_terminal(handlers: &Arc<Handlers>, plan_id: &str) -> Value {
    for _ in 0..40u32 {
        let payload = handlers
            .plan_status(PlanStatusParams {
                plan_id: plan_id.into(),
            })
            .await
            .expect("plan_status");
        let status = payload["status"].as_str().unwrap_or("");
        if matches!(status, "completed" | "failed" | "cancelled") {
            return payload;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("plan never reached terminal state")
}

// ---------- Scenarios -----------------------------------------------------

#[tokio::test]
async fn structured_prose_compiles_confirms_and_executes_to_completed() {
    let h = build_harness();

    let prose = "# refactor-auth

## extract-jwt [agent: echo]
Pull the JWT helpers out of `auth/mod.rs` into their own module.

## migrate-callers [agent: echo] depends_on: [extract-jwt]
Update every call site to use the new module.

## smoke-test [agent: echo] depends_on: [migrate-callers]
Run the integration tests and capture failures.
";

    // 1) plan_create with structured prose should compile in one shot:
    //    no open questions, status still Draft (per the design contract --
    //    every prose plan starts as Draft until plan_confirm), and the
    //    DAG preview is rendered.
    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some(prose.into()),
        })
        .await
        .expect("plan_create(prose) succeeds with structured input");

    let plan_id = created["plan_id"]
        .as_str()
        .expect("plan_id present")
        .to_owned();
    assert_eq!(created["status"], "draft");
    assert_eq!(
        created["open_questions"].as_array().map(Vec::len),
        Some(0),
        "no open questions when stub parses cleanly"
    );
    assert_eq!(
        created["name"], "refactor-auth",
        "plan name taken from H1 in prose"
    );
    assert!(
        created.get("preview").is_some(),
        "preview rendered once the DAG compiled (3 steps)"
    );
    assert_eq!(created["preview"]["total_steps"], 3);
    let layers = created["preview"]["execution_order"]
        .as_array()
        .expect("preview.execution_order");
    assert_eq!(
        layers.len(),
        3,
        "three sequential layers (extract-jwt -> migrate-callers -> smoke-test)"
    );

    // 2) plan_confirm advances Draft -> Running and kicks the executor.
    let confirmed = h
        .handlers
        .plan_confirm(PlanConfirmParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_confirm succeeds for a clean draft");
    assert_eq!(confirmed["status"], "running");

    // 3) plan_status polls to a terminal state. EchoSpawner always
    //    succeeds, so we expect Completed.
    let final_status = poll_until_terminal(&h.handlers, &plan_id).await;
    assert_eq!(final_status["status"], "completed");
    let steps = final_status["steps"]
        .as_array()
        .expect("steps in final status");
    assert_eq!(steps.len(), 3);
    for step in steps {
        assert_eq!(step["status"], "completed", "step {step:?} did not succeed");
    }
}

#[tokio::test]
async fn freeform_prose_loops_through_refine_then_executes() {
    let h = build_harness();

    // 1) Freeform prose: the stub cannot parse this, so it surfaces its
    //    `stub.structure_required` open question.
    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some(
                "I want to refactor the auth module so we can swap the JWT \
                 algorithm without rewriting callers."
                    .into(),
            ),
        })
        .await
        .expect("plan_create(prose) returns Draft + question even on freeform input");

    let plan_id = created["plan_id"].as_str().unwrap().to_owned();
    assert_eq!(created["status"], "draft");
    let questions = created["open_questions"]
        .as_array()
        .expect("open_questions array");
    assert_eq!(questions.len(), 1);
    assert_eq!(questions[0]["id"], "stub.structure_required");
    assert_eq!(
        questions[0]["allow_explain_more"], true,
        "stub asks for restructured prose via ExplainMore"
    );

    // 2) plan_confirm refuses while a question is open.
    let pending = h
        .handlers
        .plan_confirm(PlanConfirmParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_confirm returns structured pending payload");
    assert_eq!(pending["status"], "pending_clarification");
    assert_eq!(pending["reason"], "open_questions_remain");

    // 3) plan_refine with structured-markdown feedback closes the loop.
    //    The stub treats `feedback` as the new source_doc verbatim, so
    //    pasting a structured plan here is the documented escape hatch
    //    for environments without an LLM.
    let refined = h
        .handlers
        .plan_refine(PlanRefineParams {
            plan_id: plan_id.clone(),
            feedback: "# auth-refactor

## extract-jwt [agent: echo]
Pull the JWT helpers out of auth/mod.rs.

## migrate [agent: echo] depends_on: [extract-jwt]
Update call sites.
"
            .into(),
        })
        .await
        .expect("plan_refine succeeds with structured feedback");
    assert_eq!(refined["status"], "draft");
    assert_eq!(
        refined["open_questions"].as_array().map(Vec::len),
        Some(0),
        "stub re-parses feedback and clears the question"
    );
    assert!(refined.get("preview").is_some());

    // 4) Confirm + execute as before.
    let confirmed = h
        .handlers
        .plan_confirm(PlanConfirmParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_confirm succeeds after refine");
    assert_eq!(confirmed["status"], "running");

    let final_status = poll_until_terminal(&h.handlers, &plan_id).await;
    assert_eq!(final_status["status"], "completed");
    assert_eq!(final_status["steps"].as_array().unwrap().len(), 2);
}
