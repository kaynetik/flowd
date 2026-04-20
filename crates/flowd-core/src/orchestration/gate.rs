//! Rule-gating hook applied by the executor at every step boundary.
//!
//! `flowd` already exposes a [`crate::rules::RuleEvaluator`] callable as the
//! `rules_check` MCP tool — but a well-behaved agent has to choose to call
//! it. The executor never enforced rules, so a plan step could happily run
//! something a `deny` rule would have blocked.
//!
//! This module fixes that with the smallest surface possible:
//!
//! - [`RuleGate`] is a tiny, object-safe view of the evaluator: a single
//!   `gate(step, project) -> GateResult` method.
//! - A blanket impl makes every existing `RuleEvaluator + Send + Sync` a
//!   `RuleGate` automatically, so wiring is `Arc::clone(&rules)`.
//! - The executor stores `Option<Arc<dyn RuleGate>>` and only consults it
//!   when set, so existing callers keep their previous behaviour.
//!
//! ## Mapping plan steps to `ProposedAction`s
//!
//! The default `RuleEvaluator` matches against `tool` (regex) and `project`
//! / `file_path` (glob scope). For a plan step we construct:
//!
//! ```text
//! ProposedAction { tool = step.agent_type, project = plan.project, file_path = None }
//! ```
//!
//! This means rule authors can write deny rules like:
//! ```yaml
//! - id: no-codex-on-secrets
//!   scope: "secrets/**"
//!   level: deny
//!   match_pattern: "^codex$"
//!   description: "do not run the codex agent against secrets/"
//! ```
//! and the executor will refuse to spawn that step. File-path-based rules
//! (`scope: "**/*.lock"`) cannot be enforced here because the executor does
//! not see the *contents* of the agent's prompt — only its identity. That is
//! still the agent's responsibility (via `rules_check` before it writes).

use std::sync::Arc;

use crate::rules::{GateResult, ProposedAction, RuleEvaluator};

use super::PlanStep;

/// Object-safe gate consulted by the executor before spawning each step.
///
/// Implementations decide whether the step is allowed to proceed. Returning
/// a `GateResult` with `allowed = false` causes the executor to mark the
/// step `StepStatus::Skipped` and fail the plan.
pub trait RuleGate: Send + Sync {
    /// Inspect a step in the context of its plan's project and decide.
    fn gate(&self, step: &PlanStep, project: Option<&str>) -> GateResult;
}

impl<T> RuleGate for T
where
    T: RuleEvaluator + Send + Sync + ?Sized,
{
    fn gate(&self, step: &PlanStep, project: Option<&str>) -> GateResult {
        let mut action = ProposedAction::new(&step.agent_type);
        if let Some(p) = project {
            action = action.with_project(p);
        }
        <T as RuleEvaluator>::check(self, &action)
    }
}

/// Convenience alias for the boxed gate the executor stores.
pub type SharedRuleGate = Arc<dyn RuleGate>;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::RuleLevel;
    use crate::orchestration::{PlanStep, StepStatus};
    use crate::rules::{InMemoryRuleEvaluator, Rule, RuleEvaluator};

    fn step(agent: &str) -> PlanStep {
        PlanStep {
            id: "s".into(),
            agent_type: agent.into(),
            prompt: "x".into(),
            depends_on: vec![],
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn rule(id: &str, scope: &str, level: RuleLevel, regex: &str) -> Rule {
        Rule {
            id: id.into(),
            scope: scope.into(),
            level,
            description: format!("desc-{id}"),
            match_pattern: regex.into(),
        }
    }

    #[test]
    fn blanket_impl_denies_when_evaluator_denies() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("no-codex", "**", RuleLevel::Deny, "^codex$"))
            .unwrap();
        let gate: Arc<dyn RuleGate> = Arc::new(ev);
        let result = gate.gate(&step("codex"), Some("flowd"));
        assert!(!result.allowed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].rule_id, "no-codex");
    }

    #[test]
    fn blanket_impl_allows_when_no_match() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("no-codex", "**", RuleLevel::Deny, "^codex$"))
            .unwrap();
        let gate: Arc<dyn RuleGate> = Arc::new(ev);
        let result = gate.gate(&step("claude"), Some("flowd"));
        assert!(result.allowed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn warn_level_does_not_block() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("loud", "**", RuleLevel::Warn, "^codex$"))
            .unwrap();
        let gate: Arc<dyn RuleGate> = Arc::new(ev);
        let result = gate.gate(&step("codex"), Some("flowd"));
        assert!(result.allowed);
        assert!(result.has_warnings());
    }
}
