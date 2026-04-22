//! Runtime [`PlanCompiler`] selection for the daemon.
//!
//! `flowd-mcp` ships several concrete compilers ([`StubPlanCompiler`],
//! [`RejectingPlanCompiler`], and the upcoming `LlmPlanCompiler`).
//! `FlowdHandlers` is generic over the compiler so each compiler stays
//! statically dispatched in tests, but the daemon needs to pick one at
//! startup based on `flowd.toml`. The trait uses `impl Future` returns
//! which means a plain `Arc<dyn PlanCompiler>` is impossible; we
//! enum-wrap the concrete implementations and forward each method
//! by hand instead.
//!
//! The enum lives in the CLI crate (rather than `flowd-mcp`) on
//! purpose: it owns the deployment policy ("which compiler ships in
//! the binary"), and `flowd-mcp` should keep exposing the individual
//! compilers without picking one for callers.

use flowd_core::error::Result;
use flowd_core::orchestration::{Answer, CompileOutput, PlanCompiler, PlanDraftSnapshot};
use flowd_mcp::{RejectingPlanCompiler, StubPlanCompiler};

use crate::config::CompilerSelection;

/// Sum-type wrapper over every [`PlanCompiler`] the daemon can host.
///
/// One variant per [`CompilerSelection`]; the [`PlanCompiler`] impl
/// pattern-matches and forwards. New variants land here when their
/// respective compiler comes online (the upcoming `Llm` variant will
/// wrap an `LlmPlanCompiler<DaemonLlmCallback>` once the callback's
/// transport story is settled).
#[derive(Debug)]
pub enum DaemonPlanCompiler {
    Stub(StubPlanCompiler),
    Rejecting(RejectingPlanCompiler),
}

impl DaemonPlanCompiler {
    /// Build the configured compiler. Infallible today: every variant
    /// is zero-sized. Returns `Result` anyway so the upcoming LLM
    /// wiring (which needs to talk to a credentials store and a HTTP
    /// client) can surface configuration errors here without breaking
    /// every caller's signature.
    ///
    /// # Errors
    /// Returns an error when the requested compiler cannot be
    /// instantiated. None of the current variants can fail; the
    /// `unnecessary_wraps` lint is silenced because removing the
    /// wrapper now would force a churn-y reintroduction once the LLM
    /// variant lands.
    #[allow(clippy::unnecessary_wraps)]
    pub fn from_selection(selection: CompilerSelection) -> Result<Self> {
        Ok(match selection {
            CompilerSelection::Stub => Self::Stub(StubPlanCompiler::new()),
            CompilerSelection::Rejecting => Self::Rejecting(RejectingPlanCompiler::new()),
        })
    }
}

impl PlanCompiler for DaemonPlanCompiler {
    async fn compile_prose(&self, prose: String, project: String) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.compile_prose(prose, project).await,
            Self::Rejecting(c) => c.compile_prose(prose, project).await,
        }
    }

    async fn apply_answers(
        &self,
        snapshot: PlanDraftSnapshot,
        answers: Vec<(String, Answer)>,
        defer_remaining: bool,
    ) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.apply_answers(snapshot, answers, defer_remaining).await,
            Self::Rejecting(c) => c.apply_answers(snapshot, answers, defer_remaining).await,
        }
    }

    async fn refine(&self, snapshot: PlanDraftSnapshot, feedback: String) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.refine(snapshot, feedback).await,
            Self::Rejecting(c) => c.refine(snapshot, feedback).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::error::FlowdError;

    fn snapshot() -> PlanDraftSnapshot {
        PlanDraftSnapshot {
            plan_name: "p".into(),
            project: "proj".into(),
            source_doc: None,
            open_questions: vec![],
            decisions: vec![],
            previous_definition: None,
        }
    }

    #[tokio::test]
    async fn stub_variant_forwards_to_stub_parser() {
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Stub).unwrap();
        let out = c
            .compile_prose("## a [agent: echo]\nhi\n".into(), "proj".into())
            .await
            .unwrap();
        assert!(out.definition.is_some());
    }

    #[tokio::test]
    async fn rejecting_variant_returns_disabled_error() {
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Rejecting).unwrap();
        let err = c
            .compile_prose("anything".into(), "proj".into())
            .await
            .unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("disabled")));
        let err = c.refine(snapshot(), "x".into()).await.unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(_)));
    }
}
