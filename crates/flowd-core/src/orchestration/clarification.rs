//! Structured clarification + decision records for prose-first plan creation.
//!
//! Prose-first plans (see `crate::orchestration` module docs) live in
//! [`super::PlanStatus::Draft`] until an external compiler has either
//! produced a valid [`super::loader::PlanDefinition`] or surfaced a list of
//! [`OpenQuestion`]s the user must answer before compilation can proceed.
//!
//! This module owns only the data; the state-machine rules (which
//! transitions are legal, when a question is considered resolved, how a
//! prior answer is invalidated) live next to the executor in
//! [`super`]/[`super::executor`].
//!
//! ## Identity model
//!
//! Question identity is the only stable handle the user, the compiler, and
//! the executor share. We deliberately conflate "decision id" with
//! "question id": once an answer crystallises into a [`DecisionRecord`],
//! the decision is keyed by the same string the original [`OpenQuestion`]
//! used. This means [`OpenQuestion::depends_on_decisions`] and
//! [`DecisionRecord::depends_on_decisions`] both list question ids, and
//! invalidation walks become trivial graph traversals.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A single choice the user can pick when answering an [`OpenQuestion`].
///
/// `rationale` is required by convention (not by the type) because the whole
/// point of surfacing options up front is to make trade-offs explicit. The
/// MCP layer currently treats an empty rationale as a soft warning rather
/// than a hard rejection so that the compiler trait surface stays cheap to
/// implement; if that turns out to be too lax we can tighten later.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QuestionOption {
    pub id: String,
    pub label: String,
    pub rationale: String,
}

/// A single open clarification the compiler has surfaced to the user.
///
/// Lifetime: created by the compiler, drained by [`super::Plan::apply_compile_output`]
/// (added in a later phase) when an answer arrives. Two escape hatches
/// (`allow_explain_more`, `allow_none`) are explicit booleans rather than
/// implicit options so the caller can render them as needed without
/// guessing.
///
/// `depends_on_decisions` lets the compiler declare which previously
/// resolved decisions had to be in place before this question made sense.
/// When a user overwrites one of those answers, the executor uses this
/// edge list to invalidate downstream questions and decisions.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OpenQuestion {
    pub id: String,
    pub prompt: String,
    /// Why this question is being asked at all. Helps the user understand
    /// the cost of each choice without re-reading the whole plan.
    pub rationale: String,
    /// Real options the user can pick from. Capped at 5 by convention to
    /// stay tractable; the compiler is expected to either fold or split.
    pub options: Vec<QuestionOption>,
    /// If true, the caller may answer with [`Answer::ExplainMore`] to ask
    /// the compiler to elaborate (does not resolve the question).
    #[serde(default)]
    pub allow_explain_more: bool,
    /// If true, the caller may answer with [`Answer::NoneOfThese`], forcing
    /// the compiler to propose alternatives on the next round (does not
    /// resolve the question).
    #[serde(default)]
    pub allow_none: bool,
    /// Question ids whose [`DecisionRecord`]s must exist for this question
    /// to be relevant. Empty for top-level questions.
    #[serde(default)]
    pub depends_on_decisions: Vec<String>,
}

/// What the user (or the agent on their behalf) submitted for a single
/// question.
///
/// Only [`Answer::Choose`] resolves a question into a [`DecisionRecord`];
/// the other two variants are signals back to the compiler that a follow-up
/// round is needed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Answer {
    /// Pick one of the question's [`QuestionOption`]s by id.
    Choose { option_id: String },
    /// Ask the compiler to elaborate. The optional `note` carries any
    /// extra context the user wants the compiler to consider.
    ExplainMore {
        #[serde(default)]
        note: String,
    },
    /// Reject all surfaced options; the compiler must propose new ones.
    NoneOfThese,
}

impl Answer {
    /// True iff this answer crystallises a decision (i.e. resolves the
    /// question).
    #[must_use]
    pub const fn resolves(&self) -> bool {
        matches!(self, Self::Choose { .. })
    }
}

/// A persisted record of a resolved [`OpenQuestion`].
///
/// `auto` distinguishes user-driven decisions from compiler best-effort
/// fills emitted in response to `__defer_remaining__`. The audit trail
/// matters: if the plan misbehaves we want to know which choices the user
/// actually made and which the compiler invented on their behalf.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecisionRecord {
    /// Same id as the [`OpenQuestion`] it resolves; conflating the two
    /// keeps invalidation walks simple.
    pub question_id: String,
    /// Which [`QuestionOption::id`] was selected.
    pub chosen_option_id: String,
    /// Question ids this decision presupposed; mirrors the corresponding
    /// [`OpenQuestion::depends_on_decisions`] at the time of resolution.
    #[serde(default)]
    pub depends_on_decisions: Vec<String>,
    /// True iff the compiler filled this in because the user passed
    /// `defer_remaining` rather than answering explicitly.
    #[serde(default)]
    pub auto: bool,
    pub decided_at: DateTime<Utc>,
}

impl DecisionRecord {
    /// Construct a fresh, user-driven decision (auto = false) stamped with
    /// the current time.
    #[must_use]
    pub fn new_user(
        question_id: impl Into<String>,
        chosen_option_id: impl Into<String>,
        depends_on_decisions: Vec<String>,
    ) -> Self {
        Self {
            question_id: question_id.into(),
            chosen_option_id: chosen_option_id.into(),
            depends_on_decisions,
            auto: false,
            decided_at: Utc::now(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn answer_resolves_only_for_choose() {
        assert!(
            Answer::Choose {
                option_id: "x".into()
            }
            .resolves()
        );
        assert!(
            !Answer::ExplainMore {
                note: String::new()
            }
            .resolves()
        );
        assert!(!Answer::NoneOfThese.resolves());
    }

    #[test]
    fn answer_serde_round_trip() {
        for a in [
            Answer::Choose {
                option_id: "opt-a".into(),
            },
            Answer::ExplainMore {
                note: "tell me more about CQRS".into(),
            },
            Answer::NoneOfThese,
        ] {
            let json = serde_json::to_string(&a).unwrap();
            let back: Answer = serde_json::from_str(&json).unwrap();
            assert_eq!(a, back);
        }
    }

    #[test]
    fn answer_none_serializes_with_snake_case_tag() {
        let json = serde_json::to_string(&Answer::NoneOfThese).unwrap();
        assert!(
            json.contains("\"none_of_these\""),
            "unexpected wire form: {json}"
        );
    }

    #[test]
    fn open_question_defaults_for_omitted_fields() {
        let json = r#"{
            "id": "q1",
            "prompt": "Pick a charting library",
            "rationale": "Affects bundle size and SSR story",
            "options": [
                { "id": "recharts", "label": "Recharts", "rationale": "tiny" },
                { "id": "visx", "label": "visx", "rationale": "low-level" }
            ]
        }"#;
        let q: OpenQuestion = serde_json::from_str(json).unwrap();
        assert!(!q.allow_explain_more);
        assert!(!q.allow_none);
        assert!(q.depends_on_decisions.is_empty());
        assert_eq!(q.options.len(), 2);
    }

    #[test]
    fn decision_record_defaults() {
        let json = r#"{
            "question_id": "q1",
            "chosen_option_id": "recharts",
            "decided_at": "2025-01-01T00:00:00Z"
        }"#;
        let d: DecisionRecord = serde_json::from_str(json).unwrap();
        assert!(!d.auto);
        assert!(d.depends_on_decisions.is_empty());
    }

    #[test]
    fn decision_record_new_user_sets_auto_false() {
        let d = DecisionRecord::new_user("q1", "recharts", vec!["q0".into()]);
        assert_eq!(d.question_id, "q1");
        assert_eq!(d.chosen_option_id, "recharts");
        assert_eq!(d.depends_on_decisions, vec!["q0".to_owned()]);
        assert!(!d.auto);
    }
}
