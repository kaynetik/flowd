//! Trait surface for prose → DAG compilation.
//!
//! The compiler is the bridge between the prose-first front door of
//! [`super::Plan`] and the DAG-shaped [`super::loader::PlanDefinition`] the
//! executor speaks. `flowd-core` deliberately stays I/O-free, so this
//! module owns only:
//!
//! * The trait shape ([`PlanCompiler`]) every implementation must satisfy.
//! * A view-only snapshot ([`PlanDraftSnapshot`]) that hides runtime fields
//!   from the compiler so it cannot accidentally mutate executor state.
//! * The structured output ([`CompileOutput`]) the executor knows how to
//!   apply onto a [`super::Plan`].
//! * A scripted [`MockPlanCompiler`] for cross-crate tests, modelled after
//!   [`super::NoOpPlanStore`] (always available, no extra deps).
//!
//! The actual LLM-touching implementation lives in `flowd-mcp` so the heavy
//! transport / model-routing concerns stay outside the domain layer; the
//! same trait is implemented there with a real callback.
//!
//! ## Trait shape
//!
//! Three methods, mirroring the three loop entry points:
//!
//! | Caller intent          | Method            |
//! |------------------------|-------------------|
//! | `plan_create(prose)`   | [`PlanCompiler::compile_prose`] |
//! | `plan_answer(...)`     | [`PlanCompiler::apply_answers`] |
//! | `plan_refine(...)`     | [`PlanCompiler::refine`]        |
//!
//! Every method returns a [`CompileOutput`]. `definition.is_some()` is the
//! signal that the plan is ready to compile to a DAG; otherwise the
//! returned `open_questions` keep the loop going.
//!
//! ## Why no `async_trait`
//!
//! Like every other async trait in this crate (see
//! [`super::PlanExecutor`], [`super::AgentSpawner`],
//! [`crate::memory::MemoryBackend`]), we use `impl Future` returns rather
//! than the `async_trait` macro. This keeps `flowd-core` dep-light and
//! preserves zero-cost dispatch when the concrete implementation is known
//! statically (the executor side). The trade-off is that
//! `Arc<dyn PlanCompiler>` is not directly possible; downstream crates
//! that need dynamic dispatch wrap a generic implementation behind their
//! own object-safe facade.

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Mutex;

use crate::error::{FlowdError, Result};

use super::Plan;
use super::clarification::{Answer, DecisionRecord, OpenQuestion};
use super::loader::PlanDefinition;

/// Read-only view of a [`Plan`] that the compiler is allowed to inspect.
///
/// Excludes runtime fields ([`Plan::status`], [`Plan::started_at`], step
/// status / output / error / timestamps) so the compiler cannot reach into
/// executor state. Construct via [`PlanDraftSnapshot::from_plan`].
#[derive(Debug, Clone)]
pub struct PlanDraftSnapshot {
    pub plan_name: String,
    pub project: String,
    pub source_doc: Option<String>,
    pub open_questions: Vec<OpenQuestion>,
    pub decisions: Vec<DecisionRecord>,
    /// Last successfully compiled definition, if any. `None` when the plan
    /// has never compiled (typical first-round case) or has been reset by
    /// a structural refinement.
    pub previous_definition: Option<PlanDefinition>,
}

impl PlanDraftSnapshot {
    /// Project a [`Plan`] into a snapshot. The `previous_definition` is
    /// reconstructed from the plan's authored steps when present so the
    /// compiler can diff against its own prior output.
    #[must_use]
    pub fn from_plan(plan: &Plan) -> Self {
        let previous_definition = if plan.steps.is_empty() {
            None
        } else {
            Some(plan_to_definition(plan))
        };
        Self {
            plan_name: plan.name.clone(),
            project: plan.project.clone(),
            source_doc: plan.source_doc.clone(),
            open_questions: plan.open_questions.clone(),
            decisions: plan.decisions.clone(),
            previous_definition,
        }
    }
}

/// Round-trip a [`Plan`] into the authored shape the compiler operates on.
///
/// We deliberately do *not* call [`PlanDefinition`] derive paths through
/// serde here -- the runtime [`super::PlanStep`] carries fields the
/// [`super::loader::StepDefinition`] doesn't know about. Manual projection
/// keeps the compiler-facing shape aligned with what an author would write.
fn plan_to_definition(plan: &Plan) -> PlanDefinition {
    use super::loader::StepDefinition;
    PlanDefinition {
        name: plan.name.clone(),
        project: Some(plan.project.clone()),
        steps: plan
            .steps
            .iter()
            .map(|s| StepDefinition {
                id: s.id.clone(),
                agent_type: s.agent_type.clone(),
                prompt: s.prompt.clone(),
                depends_on: s.depends_on.clone(),
                timeout_secs: s.timeout_secs,
                retry_count: s.retry_count,
            })
            .collect(),
    }
}

/// Structured result of a single compiler invocation.
///
/// `definition.is_some()` iff `open_questions.is_empty()`. The trait
/// rustdoc requires implementations to enforce this; the executor
/// double-checks before transitioning the plan.
#[derive(Debug, Clone, Default)]
pub struct CompileOutput {
    /// Updated prose plan. The executor stores this verbatim into
    /// [`Plan::source_doc`].
    pub source_doc: String,
    /// Outstanding clarifications. Empty when the plan is ready to compile.
    pub open_questions: Vec<OpenQuestion>,
    /// Decisions that crystallised during this round (because the user's
    /// answers resolved them or `defer_remaining` filled them in).
    /// Appended to [`Plan::decisions`] by the executor.
    pub new_decisions: Vec<DecisionRecord>,
    /// Compiled DAG, populated only when `open_questions.is_empty()`.
    pub definition: Option<PlanDefinition>,
}

impl CompileOutput {
    /// Build an output that resolves the plan immediately (no questions,
    /// no new decisions). Convenience for tests and trivial compilers.
    #[must_use]
    pub fn ready(source_doc: impl Into<String>, definition: PlanDefinition) -> Self {
        Self {
            source_doc: source_doc.into(),
            open_questions: Vec::new(),
            new_decisions: Vec::new(),
            definition: Some(definition),
        }
    }

    /// Build an output that surfaces clarifications and leaves compilation
    /// for a later round.
    #[must_use]
    pub fn pending(source_doc: impl Into<String>, open_questions: Vec<OpenQuestion>) -> Self {
        Self {
            source_doc: source_doc.into(),
            open_questions,
            new_decisions: Vec::new(),
            definition: None,
        }
    }
}

/// Compiles prose-first plans into [`PlanDefinition`]s.
///
/// Implementations are expected to be referentially transparent for fixed
/// inputs but may legitimately call out to LLMs, so callers must treat
/// every method as potentially expensive. The trait carries no retry,
/// caching, or budget concerns -- those live one layer up in the MCP
/// handlers.
///
/// ### Contract
///
/// * `definition.is_some()` iff `open_questions.is_empty()`.
/// * Every emitted [`OpenQuestion::depends_on_decisions`] entry must
///   correspond to either a question already answered (present in
///   `snapshot.decisions`) or a question that will appear in this same
///   `CompileOutput`'s `new_decisions`. The executor's debug-build
///   invariant flags violations.
/// * Implementations should avoid renaming question ids across rounds; the
///   executor uses ids as the stable handle for invalidation.
pub trait PlanCompiler: Send + Sync {
    /// Compile a fresh prose plan into either a DAG or a list of open
    /// questions.
    ///
    /// # Errors
    /// Implementations should return `FlowdError::PlanValidation` for
    /// malformed prose and `FlowdError::Internal` for transport failures.
    fn compile_prose(
        &self,
        prose: String,
        project: String,
    ) -> impl Future<Output = Result<CompileOutput>> + Send;

    /// Apply a batch of answers (and optionally fill in remaining
    /// questions on a best-effort basis) to an in-flight draft.
    ///
    /// `defer_remaining = true` instructs the compiler to invent
    /// best-effort answers for any still-open questions and emit them as
    /// [`DecisionRecord`]s with `auto = true`.
    ///
    /// # Errors
    /// As above.
    fn apply_answers(
        &self,
        snapshot: PlanDraftSnapshot,
        answers: Vec<(String, Answer)>,
        defer_remaining: bool,
    ) -> impl Future<Output = Result<CompileOutput>> + Send;

    /// Apply a freeform refinement instruction to an in-flight draft.
    ///
    /// Allowed to legitimately re-introduce open questions if the
    /// refinement raises new architectural concerns; the executor will
    /// re-enter the clarification loop in that case.
    ///
    /// # Errors
    /// As above.
    fn refine(
        &self,
        snapshot: PlanDraftSnapshot,
        feedback: String,
    ) -> impl Future<Output = Result<CompileOutput>> + Send;
}

/// Scripted compiler that returns pre-canned [`CompileOutput`]s in order.
///
/// Useful for unit and integration tests across crate boundaries (e.g.
/// `flowd-mcp` exercising the full `plan_create` → `plan_answer` →
/// `plan_confirm` loop without depending on a real LLM).
///
/// The script is consumed strictly in FIFO order: each call to any of the
/// three trait methods pops the next entry, regardless of which method was
/// invoked. This is intentional -- tests can express a complete
/// interaction sequence as one ordered list, and accidental method
/// confusion shows up as a missing-script failure rather than silent
/// success.
#[derive(Debug, Default)]
pub struct MockPlanCompiler {
    script: Mutex<VecDeque<CompileOutput>>,
}

impl MockPlanCompiler {
    /// Build a mock from an ordered script of outputs.
    #[must_use]
    pub fn new(script: Vec<CompileOutput>) -> Self {
        Self {
            script: Mutex::new(script.into()),
        }
    }

    /// Append another scripted output; chainable for fluent test setup.
    ///
    /// # Panics
    /// Panics if the internal script mutex was poisoned by a previous
    /// panic in another thread. Tests are single-threaded against the
    /// mock so this is a programming-error escape hatch, not a real
    /// runtime concern.
    pub fn push(&self, output: CompileOutput) {
        self.script
            .lock()
            .expect("mock script mutex poisoned")
            .push_back(output);
    }

    /// True when the script has been fully drained. Tests typically
    /// `assert!(mock.is_exhausted())` at the end of a scenario to catch
    /// stale fixtures.
    ///
    /// # Panics
    /// See [`Self::push`].
    #[must_use]
    pub fn is_exhausted(&self) -> bool {
        self.script
            .lock()
            .expect("mock script mutex poisoned")
            .is_empty()
    }

    fn pop(&self, method: &'static str) -> Result<CompileOutput> {
        self.script
            .lock()
            .expect("mock script mutex poisoned")
            .pop_front()
            .ok_or_else(|| {
                FlowdError::Internal(format!(
                    "MockPlanCompiler::{method} called with empty script"
                ))
            })
    }
}

impl PlanCompiler for MockPlanCompiler {
    async fn compile_prose(&self, _prose: String, _project: String) -> Result<CompileOutput> {
        self.pop("compile_prose")
    }

    async fn apply_answers(
        &self,
        _snapshot: PlanDraftSnapshot,
        _answers: Vec<(String, Answer)>,
        _defer_remaining: bool,
    ) -> Result<CompileOutput> {
        self.pop("apply_answers")
    }

    async fn refine(
        &self,
        _snapshot: PlanDraftSnapshot,
        _feedback: String,
    ) -> Result<CompileOutput> {
        self.pop("refine")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::clarification::{OpenQuestion, QuestionOption};
    use crate::orchestration::loader::{PlanDefinition, StepDefinition};

    fn def() -> PlanDefinition {
        PlanDefinition {
            name: "p".into(),
            project: Some("proj".into()),
            steps: vec![StepDefinition {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "hi".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        }
    }

    fn question(id: &str) -> OpenQuestion {
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
            depends_on_decisions: vec![],
        }
    }

    #[tokio::test]
    async fn mock_drains_script_in_order_across_methods() {
        let mock = MockPlanCompiler::new(vec![
            CompileOutput::pending("# round 1", vec![question("q1"), question("q2")]),
            CompileOutput::pending("# round 2", vec![question("q2")]),
            CompileOutput::ready("# done", def()),
        ]);

        let snap = PlanDraftSnapshot {
            plan_name: "p".into(),
            project: "proj".into(),
            source_doc: None,
            open_questions: vec![],
            decisions: vec![],
            previous_definition: None,
        };

        let r1 = mock
            .compile_prose("anything".into(), "proj".into())
            .await
            .unwrap();
        assert_eq!(r1.open_questions.len(), 2);
        assert!(r1.definition.is_none());

        let r2 = mock
            .apply_answers(snap.clone(), vec![], false)
            .await
            .unwrap();
        assert_eq!(r2.open_questions.len(), 1);

        let r3 = mock.refine(snap, "tweak it".into()).await.unwrap();
        assert!(r3.open_questions.is_empty());
        assert!(r3.definition.is_some());

        assert!(mock.is_exhausted());
    }

    #[tokio::test]
    async fn mock_returns_internal_error_when_script_empty() {
        let mock = MockPlanCompiler::default();
        let err = mock
            .compile_prose("hi".into(), "proj".into())
            .await
            .unwrap_err();
        assert!(matches!(err, FlowdError::Internal(m) if m.contains("empty script")));
    }

    #[tokio::test]
    async fn mock_push_extends_script_after_construction() {
        let mock = MockPlanCompiler::default();
        mock.push(CompileOutput::ready("d", def()));
        let out = mock.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_some());
        assert!(mock.is_exhausted());
    }

    #[test]
    fn snapshot_from_plan_round_trips_authored_shape() {
        use crate::orchestration::{Plan, PlanStep, StepStatus};

        let plan = Plan::new(
            "p",
            "proj",
            vec![PlanStep {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "hi".into(),
                depends_on: vec![],
                timeout_secs: Some(5),
                retry_count: 2,
                status: StepStatus::Pending,
                output: None,
                error: None,
                started_at: None,
                completed_at: None,
            }],
        );

        let snap = PlanDraftSnapshot::from_plan(&plan);
        let prev = snap.previous_definition.expect("definition projected");
        assert_eq!(prev.name, "p");
        assert_eq!(prev.project.as_deref(), Some("proj"));
        assert_eq!(prev.steps.len(), 1);
        assert_eq!(prev.steps[0].timeout_secs, Some(5));
        assert_eq!(prev.steps[0].retry_count, 2);
    }

    #[test]
    fn snapshot_from_empty_plan_has_no_previous_definition() {
        use crate::orchestration::Plan;
        let plan = Plan::new("p", "proj", vec![]);
        let snap = PlanDraftSnapshot::from_plan(&plan);
        assert!(snap.previous_definition.is_none());
    }
}
