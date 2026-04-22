//! Prose-first plan-creation contract tests.
//!
//! Wires `FlowdHandlers` to:
//!   * real `SqliteBackend` (temp file, migrations run) + in-process
//!     vector / embedder stubs, just like `tests/e2e.rs`,
//!   * the in-crate `EchoSpawner` for plan execution,
//!   * a real `InMemoryRuleEvaluator` with no rules,
//!   * a *scripted* [`MockPlanCompiler`] so we can drive a deterministic
//!     clarification → answer → confirm loop without any LLM,
//!   * a recording [`PlanObserver`] so we can assert that
//!     `ClarificationOpened` / `ClarificationResolved` /
//!     `RefinementApplied` events fire from the right handlers.
//!
//! These tests pin the cross-crate contract for HL-44 phase 5/6: the
//! polymorphic `plan_create`, the `plan_answer` / `plan_refine` loop,
//! the structured `plan_confirm` rejection, and the new event variants.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tempfile::TempDir;
use uuid::Uuid;

use flowd_core::error::Result as FlowdResult;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, VectorIndex};
use flowd_core::orchestration::observer::{PlanEvent, PlanObserver, SharedPlanObserver};
use flowd_core::orchestration::{
    AgentOutput, AgentSpawner, CompileOutput, InMemoryPlanExecutor, MockPlanCompiler, OpenQuestion,
    PlanStep, QuestionOption, loader::StepDefinition,
};
use flowd_core::rules::InMemoryRuleEvaluator;
use flowd_core::types::Embedding;
use flowd_mcp::tools::{
    PlanAnswerEntry, PlanAnswerParams, PlanCancelParams, PlanConfirmParams, PlanCreateParams,
    PlanRefineParams, PlanStatusParams,
};
use flowd_mcp::{FlowdHandlers, McpHandlers};
use flowd_storage::plan_store::SqlitePlanStore;
use flowd_storage::sqlite::SqliteBackend;

// ---------- Shared test doubles (mirrors tests/e2e.rs) --------------------

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

/// Records every event in arrival order. Synchronous so we can poll it
/// straight after the handler returns -- the handlers emit on the
/// caller's task.
#[derive(Default)]
struct RecordingObserver {
    events: Mutex<Vec<PlanEvent>>,
}

impl RecordingObserver {
    fn snapshot(&self) -> Vec<PlanEvent> {
        self.events.lock().unwrap().clone()
    }
}

impl PlanObserver for RecordingObserver {
    fn on_event(&self, event: PlanEvent) {
        self.events.lock().unwrap().push(event);
    }
}

// ---------- Harness -------------------------------------------------------

type Handlers = FlowdHandlers<
    SqliteBackend,
    MemVectors,
    HashEmbedder,
    InMemoryPlanExecutor<EchoSpawner, SqlitePlanStore>,
    MockPlanCompiler,
    InMemoryRuleEvaluator,
>;

struct Harness {
    _tmp: TempDir,
    handlers: Arc<Handlers>,
    observer: Arc<RecordingObserver>,
}

fn build_harness(script: Vec<CompileOutput>) -> Harness {
    build_harness_with_budget(script, None)
}

fn build_harness_with_budget(script: Vec<CompileOutput>, budget: Option<usize>) -> Harness {
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
    let compiler = Arc::new(MockPlanCompiler::new(script));
    let observer = Arc::new(RecordingObserver::default());
    let handlers = Arc::new(
        FlowdHandlers::new(memory, executor, compiler, rules)
            .with_observer(Arc::clone(&observer) as SharedPlanObserver)
            .with_question_budget(budget),
    );
    Harness {
        _tmp: tmp,
        handlers,
        observer,
    }
}

// ---------- Fixtures ------------------------------------------------------

fn opt(id: &str, label: &str) -> QuestionOption {
    QuestionOption {
        id: id.into(),
        label: label.into(),
        rationale: format!("rationale for {id}"),
    }
}

fn question(id: &str, options: Vec<QuestionOption>) -> OpenQuestion {
    OpenQuestion {
        id: id.into(),
        prompt: format!("pick for {id}"),
        rationale: "needed to compile".into(),
        options,
        allow_explain_more: false,
        allow_none: false,
        depends_on_decisions: vec![],
    }
}

fn echo_def(name: &str) -> flowd_core::orchestration::loader::PlanDefinition {
    flowd_core::orchestration::loader::PlanDefinition {
        name: name.into(),
        project: Some("rnd".into()),
        steps: vec![StepDefinition {
            id: "a".into(),
            agent_type: "echo".into(),
            prompt: "do a".into(),
            depends_on: vec![],
            timeout_secs: None,
            retry_count: 0,
        }],
    }
}

// ---------- The tests -----------------------------------------------------

#[tokio::test]
async fn plan_create_with_definition_keeps_legacy_behaviour() {
    // No prose path exercised; mock script stays empty -- if the handler
    // ever called the compiler the test would fail loudly.
    let h = build_harness(vec![]);
    let payload = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: Some(json!({
                "name": "legacy",
                "steps": [{ "id": "a", "agent_type": "echo", "prompt": "do a" }]
            })),
            prose: None,
            compiler_override: None,
        })
        .await
        .expect("plan_create with definition succeeds");
    assert!(payload["plan_id"].is_string());
    assert_eq!(payload["preview"]["total_steps"], 1);
}

#[tokio::test]
async fn plan_create_rejects_both_definition_and_prose() {
    let h = build_harness(vec![]);
    let err = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: Some(json!({"name": "x", "steps": []})),
            prose: Some("do a thing".into()),
            compiler_override: None,
        })
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("exactly one"));
}

#[tokio::test]
async fn plan_create_rejects_compiler_override_on_definition_path() {
    // `compiler_override` only applies to the prose-first path. Pairing
    // it with `definition` is almost always a copy/paste mistake; the
    // handler refuses loudly so the caller fixes the request rather than
    // having the field silently ignored.
    let h = build_harness(vec![]);
    let err = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: Some(json!({
                "name": "legacy",
                "steps": [{ "id": "a", "agent_type": "echo", "prompt": "do a" }]
            })),
            prose: None,
            compiler_override: Some("claude-cli".into()),
        })
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("compiler_override"));
}

#[tokio::test]
async fn plan_create_with_compiler_override_on_prose_routes_through_default_compiler() {
    // The mock compiler doesn't override `compile_prose_with_override`
    // so the trait default applies: the override is ignored and
    // `compile_prose` runs normally. This test pins that behaviour --
    // single-backend compilers MUST be transparent to the override
    // field, otherwise we'd break every caller that wires in a custom
    // PlanCompiler outside the daemon's `DaemonPlanCompiler`.
    let h = build_harness(vec![CompileOutput::ready("# title", echo_def("compiled"))]);
    let payload = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("# Title".into()),
            compiler_override: Some("mlx".into()),
        })
        .await
        .expect("override is no-op for single-backend compilers");
    assert_eq!(payload["status"], "draft");
    assert!(payload.get("preview").is_some());
}

#[tokio::test]
async fn plan_create_rejects_when_neither_definition_nor_prose_provided() {
    let h = build_harness(vec![]);
    let err = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: None,
            compiler_override: None,
        })
        .await
        .unwrap_err();
    assert!(format!("{err}").contains("either"));
}

#[tokio::test]
async fn prose_loop_runs_create_answer_confirm_and_emits_events() {
    // Round 1: compile_prose surfaces one question.
    // Round 2: apply_answers resolves it and returns a compiled DAG.
    let h = build_harness(vec![
        CompileOutput::pending("# initial draft", vec![question("q1", vec![opt("a", "A")])]),
        CompileOutput {
            source_doc: "# resolved".into(),
            open_questions: vec![],
            new_decisions: vec![flowd_core::orchestration::DecisionRecord {
                question_id: "q1".into(),
                chosen_option_id: "a".into(),
                depends_on_decisions: vec![],
                auto: false,
                decided_at: chrono::Utc::now(),
            }],
            definition: Some(echo_def("compiled-from-prose")),
        },
    ]);

    // 1) plan_create with prose
    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("# Refactor auth\nMake it secure.".into()),
            compiler_override: None,
        })
        .await
        .expect("plan_create(prose)");
    let plan_id = created["plan_id"].as_str().expect("plan_id").to_owned();
    assert_eq!(created["status"], "draft");
    assert_eq!(created["open_questions"].as_array().unwrap().len(), 1);
    assert_eq!(
        created["name"], "Refactor auth",
        "name derived from first heading"
    );
    assert!(
        created.get("preview").is_none(),
        "preview only present once compiled"
    );

    // 2) plan_confirm should refuse: open_questions still present.
    let pending = h
        .handlers
        .plan_confirm(PlanConfirmParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_confirm returns ok payload even when rejecting");
    assert_eq!(pending["status"], "pending_clarification");
    assert_eq!(pending["reason"], "open_questions_remain");
    assert_eq!(pending["open_questions"].as_array().unwrap().len(), 1);

    // 3) plan_answer resolves the question.
    let answered = h
        .handlers
        .plan_answer(PlanAnswerParams {
            plan_id: plan_id.clone(),
            answers: vec![PlanAnswerEntry {
                question_id: "q1".into(),
                answer: flowd_core::orchestration::Answer::Choose {
                    option_id: "a".into(),
                },
            }],
            defer_remaining: false,
        })
        .await
        .expect("plan_answer succeeds");
    assert_eq!(answered["status"], "draft");
    assert_eq!(answered["open_questions"].as_array().unwrap().len(), 0);
    assert_eq!(answered["decisions"].as_array().unwrap().len(), 1);
    assert!(
        answered.get("preview").is_some(),
        "preview rendered once the DAG compiled"
    );

    // 4) plan_confirm now succeeds and kicks execution.
    let confirmed = h
        .handlers
        .plan_confirm(PlanConfirmParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_confirm succeeds after resolution");
    assert_eq!(confirmed["status"], "running");

    // Wait for the spawned execution task to complete.
    poll_until_terminal(&h.handlers, &plan_id).await;

    // Event audit: ClarificationOpened (round 1), ClarificationResolved
    // (round 2), then the executor's own Submitted/Started/.../Finished
    // chain. We only assert on the prose-loop variants here -- the
    // executor's events have their own tests.
    let kinds: Vec<&'static str> = h
        .observer
        .snapshot()
        .iter()
        .filter_map(|e| match e {
            PlanEvent::ClarificationOpened { .. } => Some("clarification_opened"),
            PlanEvent::ClarificationResolved { .. } => Some("clarification_resolved"),
            PlanEvent::RefinementApplied { .. } => Some("refinement_applied"),
            _ => None,
        })
        .collect();
    assert_eq!(
        kinds,
        vec!["clarification_opened", "clarification_resolved"]
    );
}

#[tokio::test]
async fn plan_refine_emits_refinement_applied_and_clarification_deltas() {
    let h = build_harness(vec![
        // create -> resolves immediately so we can refine on a stable Draft.
        CompileOutput::ready("# round-zero", echo_def("draft-v0")),
        // refine -> introduces a new question + flips draft dirty.
        CompileOutput::pending("# refined", vec![question("q2", vec![opt("x", "X")])]),
    ]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("seed".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    let plan_id = created["plan_id"].as_str().unwrap().to_owned();
    assert!(
        created["open_questions"].as_array().unwrap().is_empty(),
        "first round resolved everything"
    );

    let refined = h
        .handlers
        .plan_refine(PlanRefineParams {
            plan_id,
            feedback: "tighten the failure path".into(),
        })
        .await
        .expect("plan_refine");
    assert_eq!(refined["status"], "draft");
    assert_eq!(refined["open_questions"].as_array().unwrap().len(), 1);
    assert_eq!(refined["definition_dirty"], true);

    let ev_kinds: Vec<&'static str> = h
        .observer
        .snapshot()
        .iter()
        .filter_map(|e| match e {
            PlanEvent::RefinementApplied { .. } => Some("refinement_applied"),
            PlanEvent::ClarificationOpened { .. } => Some("clarification_opened"),
            _ => None,
        })
        .collect();
    // RefinementApplied must come before the ClarificationOpened it
    // produced so audit consumers can correlate trigger -> effect.
    assert_eq!(ev_kinds, vec!["refinement_applied", "clarification_opened"]);
}

#[tokio::test]
async fn plan_answer_overwriting_a_decision_invalidates_dependent_chain() {
    // Round 1: surface q1.
    // Round 2: q1 answered (a) -> compile fully resolves.
    // Round 3 (overwrite): user re-answers q1 with (b); compiler then
    //   surfaces q1 again (no resolution this time) so the test can
    //   observe both `invalidate_decision` (drops the prior decision) and
    //   the executor reaching the open-question state again.
    let h = build_harness(vec![
        CompileOutput::pending(
            "# r1",
            vec![question("q1", vec![opt("a", "A"), opt("b", "B")])],
        ),
        CompileOutput {
            source_doc: "# r2".into(),
            open_questions: vec![],
            new_decisions: vec![flowd_core::orchestration::DecisionRecord {
                question_id: "q1".into(),
                chosen_option_id: "a".into(),
                depends_on_decisions: vec![],
                auto: false,
                decided_at: chrono::Utc::now(),
            }],
            definition: Some(echo_def("first-pass")),
        },
        CompileOutput::pending(
            "# r3",
            vec![question("q1", vec![opt("a", "A"), opt("b", "B")])],
        ),
    ]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("# Title".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    let plan_id = created["plan_id"].as_str().unwrap().to_owned();

    // First answer crystallises q1.
    let after_first = h
        .handlers
        .plan_answer(PlanAnswerParams {
            plan_id: plan_id.clone(),
            answers: vec![PlanAnswerEntry {
                question_id: "q1".into(),
                answer: flowd_core::orchestration::Answer::Choose {
                    option_id: "a".into(),
                },
            }],
            defer_remaining: false,
        })
        .await
        .unwrap();
    assert_eq!(after_first["decisions"].as_array().unwrap().len(), 1);
    assert!(after_first.get("preview").is_some());

    // Second answer overwrites q1; the handler should call
    // invalidate_decision before recompiling, so we end up with no
    // decisions but a fresh open question.
    let after_overwrite = h
        .handlers
        .plan_answer(PlanAnswerParams {
            plan_id: plan_id.clone(),
            answers: vec![PlanAnswerEntry {
                question_id: "q1".into(),
                answer: flowd_core::orchestration::Answer::Choose {
                    option_id: "b".into(),
                },
            }],
            defer_remaining: false,
        })
        .await
        .unwrap();
    assert_eq!(after_overwrite["decisions"].as_array().unwrap().len(), 0);
    assert_eq!(
        after_overwrite["open_questions"].as_array().unwrap().len(),
        1
    );
    assert_eq!(after_overwrite["definition_dirty"], true);
}

#[tokio::test]
async fn plan_cancel_terminates_a_draft_plan_idempotently() {
    let h = build_harness(vec![CompileOutput::pending(
        "# stuck",
        vec![question("q1", vec![opt("a", "A")])],
    )]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("anything".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    let plan_id = created["plan_id"].as_str().unwrap().to_owned();

    let cancelled = h
        .handlers
        .plan_cancel(PlanCancelParams {
            plan_id: plan_id.clone(),
        })
        .await
        .unwrap();
    assert_eq!(cancelled["status"], "cancelled");

    // Idempotency: a second cancel returns ok with the same status.
    let again = h
        .handlers
        .plan_cancel(PlanCancelParams { plan_id })
        .await
        .unwrap();
    assert_eq!(again["status"], "cancelled");
}

#[tokio::test]
async fn plan_refine_sets_clarification_reopened_flag_when_introducing_questions() {
    let h = build_harness(vec![
        // Round 1: clean compile -> no questions, definition present.
        CompileOutput::ready("# v0", echo_def("draft-v0")),
        // Round 2 (refine): compiler decides the change opens a new
        // architectural question and surfaces it.
        CompileOutput::pending("# v1", vec![question("q1", vec![opt("a", "A")])]),
    ]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("seed".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    let plan_id = created["plan_id"].as_str().unwrap().to_owned();

    let refined = h
        .handlers
        .plan_refine(PlanRefineParams {
            plan_id,
            feedback: "tighten error handling".into(),
        })
        .await
        .unwrap();
    assert_eq!(refined["status"], "draft");
    assert_eq!(refined["clarification_reopened"], true);
    assert_eq!(refined["open_questions"].as_array().unwrap().len(), 1);
}

#[tokio::test]
async fn plan_refine_clarification_reopened_is_false_when_dag_compiled_cleanly() {
    let h = build_harness(vec![
        CompileOutput::ready("# v0", echo_def("draft-v0")),
        // Refine compiles cleanly again -- no questions surfaced.
        CompileOutput::ready("# v1", echo_def("draft-v1")),
    ]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("seed".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    let plan_id = created["plan_id"].as_str().unwrap().to_owned();

    let refined = h
        .handlers
        .plan_refine(PlanRefineParams {
            plan_id,
            feedback: "rename a step".into(),
        })
        .await
        .unwrap();
    assert_eq!(refined["clarification_reopened"], false);
    assert!(refined.get("preview").is_some());
}

#[tokio::test]
async fn plan_answer_emits_budget_exceeded_warning_when_load_overflows() {
    // Tight budget = 1; the first round surfaces 2 questions on its
    // own which is already over budget.
    let h = build_harness_with_budget(
        vec![
            CompileOutput::pending(
                "# r1",
                vec![
                    question("q1", vec![opt("a", "A")]),
                    question("q2", vec![opt("a", "A")]),
                ],
            ),
            // Round 2: q1 answered, q2 still open -> 1 decision + 1 question = 2 (still > 1).
            CompileOutput {
                source_doc: "# r2".into(),
                open_questions: vec![question("q2", vec![opt("a", "A")])],
                new_decisions: vec![flowd_core::orchestration::DecisionRecord {
                    question_id: "q1".into(),
                    chosen_option_id: "a".into(),
                    depends_on_decisions: vec![],
                    auto: false,
                    decided_at: chrono::Utc::now(),
                }],
                definition: None,
            },
        ],
        Some(1),
    );

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("anything".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    // First round already over budget -> warning surfaces on plan_create.
    let warnings = created["warnings"]
        .as_array()
        .expect("warnings on plan_create");
    let codes: Vec<&str> = warnings
        .iter()
        .map(|w| w["code"].as_str().unwrap())
        .collect();
    assert!(
        codes.contains(&"BudgetExceeded"),
        "got codes: {codes:?} payload: {created:#}"
    );

    let plan_id = created["plan_id"].as_str().unwrap().to_owned();

    // Second round: pre-flight sees load == 2 (>= 1), so defer is
    // coerced and a DeferRemainingCoerced warning is emitted *as well
    // as* the post-compile BudgetExceeded one.
    let answered = h
        .handlers
        .plan_answer(PlanAnswerParams {
            plan_id,
            answers: vec![PlanAnswerEntry {
                question_id: "q1".into(),
                answer: flowd_core::orchestration::Answer::Choose {
                    option_id: "a".into(),
                },
            }],
            defer_remaining: false,
        })
        .await
        .unwrap();
    let codes: Vec<&str> = answered["warnings"]
        .as_array()
        .expect("warnings on plan_answer")
        .iter()
        .map(|w| w["code"].as_str().unwrap())
        .collect();
    assert!(
        codes.contains(&"DeferRemainingCoerced"),
        "expected DeferRemainingCoerced in {codes:?}"
    );
    assert!(
        codes.contains(&"BudgetExceeded"),
        "expected BudgetExceeded in {codes:?}"
    );
}

#[tokio::test]
async fn no_warnings_emitted_when_budget_is_unset() {
    let h = build_harness(vec![CompileOutput::pending(
        "# r1",
        vec![
            question("q1", vec![opt("a", "A")]),
            question("q2", vec![opt("a", "A")]),
            question("q3", vec![opt("a", "A")]),
        ],
    )]);

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("anything".into()),
            compiler_override: None,
        })
        .await
        .unwrap();
    assert!(
        created.get("warnings").is_none(),
        "no warnings field when budget unset; got {created:#}"
    );
}

// ---------- Helpers -------------------------------------------------------

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
