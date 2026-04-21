//! Placeholder [`PlanCompiler`] for the prose-first plan-creation surface.
//!
//! The MCP layer accepts `plan_create` calls with either a structured
//! `definition` (DAG-first) or freeform `prose` (prose-first; HL-44).
//! Prose-first plans need a [`PlanCompiler`] implementation, but the real
//! LLM-backed compiler ships in a follow-up PR. To keep the daemon
//! compileable and the legacy DAG-first path working in the meantime,
//! [`RejectingPlanCompiler`] satisfies the trait by returning a clear
//! [`FlowdError::PlanValidation`] error for every call.
//!
//! Once the real compiler is wired into `flowd-cli/src/commands/start.rs`,
//! this type can either be deleted or kept for unit tests that want a
//! deterministic "no compiler installed" surface.
//!
//! ## Why not just `Option<PlanCompiler>` on `FlowdHandlers`?
//!
//! Using `Option` would push the "compiler missing" check into every
//! prose-first handler and force callers to interpret an
//! `Internal("compiler unset")` error. Shipping a typed placeholder
//! instead keeps the handler code path uniform: the compiler is always
//! present, the error message is always actionable, and the trait
//! implementation acts as living documentation of the contract that any
//! real compiler must eventually satisfy.

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{Answer, CompileOutput, PlanCompiler, PlanDraftSnapshot};

/// [`PlanCompiler`] that refuses every call with a clear validation error.
///
/// Useful as a placeholder until a real LLM-backed compiler is wired in,
/// and as a deterministic test double when a handler test wants to assert
/// that the prose-first surface is reachable but does not want to script
/// a [`flowd_core::orchestration::MockPlanCompiler`].
#[derive(Debug, Default, Clone, Copy)]
pub struct RejectingPlanCompiler;

impl RejectingPlanCompiler {
    /// Construct a new rejecting compiler. Cheap; the type is zero-sized.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    fn unavailable(method: &'static str) -> FlowdError {
        FlowdError::PlanValidation(format!(
            "prose-first plan creation is not enabled in this build (called {method}); \
             pass `definition` to plan_create instead, or rebuild with a real PlanCompiler wired in"
        ))
    }
}

impl PlanCompiler for RejectingPlanCompiler {
    async fn compile_prose(&self, _prose: String, _project: String) -> Result<CompileOutput> {
        Err(Self::unavailable("compile_prose"))
    }

    async fn apply_answers(
        &self,
        _snapshot: PlanDraftSnapshot,
        _answers: Vec<(String, Answer)>,
        _defer_remaining: bool,
    ) -> Result<CompileOutput> {
        Err(Self::unavailable("apply_answers"))
    }

    async fn refine(
        &self,
        _snapshot: PlanDraftSnapshot,
        _feedback: String,
    ) -> Result<CompileOutput> {
        Err(Self::unavailable("refine"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn every_method_returns_unavailable_validation_error() {
        let c = RejectingPlanCompiler::new();

        let err = c
            .compile_prose("hi".into(), "proj".into())
            .await
            .unwrap_err();
        match err {
            FlowdError::PlanValidation(m) => {
                assert!(m.contains("compile_prose"));
                assert!(m.contains("not enabled"));
            }
            other => panic!("expected PlanValidation, got {other:?}"),
        }

        let snap = PlanDraftSnapshot {
            plan_name: "p".into(),
            project: "proj".into(),
            source_doc: None,
            open_questions: vec![],
            decisions: vec![],
            previous_definition: None,
        };
        assert!(matches!(
            c.apply_answers(snap.clone(), vec![], false).await,
            Err(FlowdError::PlanValidation(_))
        ));
        assert!(matches!(
            c.refine(snap, "tweak".into()).await,
            Err(FlowdError::PlanValidation(_))
        ));
    }
}
