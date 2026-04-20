//! Step-prompt template substitution.
//!
//! Adds the smallest possible context-passing primitive between plan steps:
//! a step's prompt may reference the captured output of any of its
//! transitive dependencies via the syntax `{{steps.<id>.output}}`. References
//! are validated at plan-submission time and substituted at spawn time.
//!
//! ## Why this shape
//!
//! - **Explicit declaration through `depends_on`.** A step may only read the
//!   output of a step it explicitly depends on (directly or transitively).
//!   This keeps the dependency DAG honest: if step B uses step A's output,
//!   the plan must say so, otherwise topological execution could spawn B
//!   before A finishes.
//! - **Bounded growth.** Each substitution is truncated to a per-reference
//!   byte budget so a chatty upstream step cannot blow downstream prompt
//!   budgets. Truncation is character-boundary safe and visibly marked.
//! - **Unknown placeholders pass through unchanged.** The substituter only
//!   replaces well-formed `{{steps.<id>.output}}` tokens that resolve to a
//!   completed step's output. Anything else is left literally; this is what
//!   `validate_plan` relies on to surface authoring mistakes early.
//! - **No new dependencies.** Pure string scanning; no regex crate, no
//!   templating engine.

use std::collections::{HashMap, HashSet};
use std::hash::BuildHasher;

use crate::error::{FlowdError, Result};

use super::Plan;

/// Default per-substitution byte budget (8 KiB). Enough to carry a code
/// block or a structured summary, small enough that ten substitutions still
/// fit in a typical model context.
pub const DEFAULT_PER_REF_BYTES: usize = 8 * 1024;

/// Marker pair that brackets a template token. Two-character open/close so
/// they cannot collide with a single brace that may appear in JSON or code
/// snippets inside an authored prompt.
const OPEN: &str = "{{";
const CLOSE: &str = "}}";

/// Find every `{{steps.<id>.output}}` reference in `prompt` and return the
/// referenced step IDs (de-duplicated, in first-seen order). Used by
/// `validate_plan` so authoring mistakes surface at submit time.
#[must_use]
pub fn extract_step_refs(prompt: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    let mut seen: HashSet<String> = HashSet::new();
    for token in iter_tokens(prompt) {
        if let Some(id) = parse_ref(token) {
            if seen.insert(id.to_owned()) {
                out.push(id.to_owned());
            }
        }
    }
    out
}

/// Validate every step's prompt: each `{{steps.<id>.output}}` reference must
/// point to a known step that is a transitive dependency of the referencing
/// step. Self-references and forward references are rejected.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` describing the first violation.
pub fn validate_step_references(plan: &Plan) -> Result<()> {
    let known: HashSet<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();
    let transitive = transitive_deps(plan);
    let empty: HashSet<&str> = HashSet::new();

    for step in &plan.steps {
        for referenced in extract_step_refs(&step.prompt) {
            if referenced == step.id {
                return Err(FlowdError::PlanValidation(format!(
                    "step `{}` references its own output",
                    step.id
                )));
            }
            if !known.contains(referenced.as_str()) {
                return Err(FlowdError::PlanValidation(format!(
                    "step `{}` references unknown step `{}` in its prompt",
                    step.id, referenced
                )));
            }
            let deps = transitive.get(step.id.as_str()).unwrap_or(&empty);
            if !deps.contains(referenced.as_str()) {
                return Err(FlowdError::PlanValidation(format!(
                    "step `{}` references `{}` but does not (transitively) depend on it; \
                     add `{}` to its depends_on chain",
                    step.id, referenced, referenced
                )));
            }
        }
    }

    Ok(())
}

/// Substitute every `{{steps.<id>.output}}` in `prompt` with the captured
/// output of the matching step in `outputs`. Each substituted value is
/// truncated to `per_ref_bytes` (UTF-8 boundary safe) so an oversized
/// upstream cannot dominate the prompt; truncation appends a visible
/// `…[+N bytes]` marker. References whose target has no recorded output
/// are replaced with the literal placeholder `<unavailable>` so the
/// downstream prompt never carries an unresolved `{{...}}` token.
#[must_use]
pub fn substitute<S: BuildHasher>(
    prompt: &str,
    outputs: &HashMap<&str, &str, S>,
    per_ref_bytes: usize,
) -> String {
    let mut out = String::with_capacity(prompt.len());
    let mut i = 0;
    while i < prompt.len() {
        let rest = &prompt[i..];
        if let Some(rel_open) = rest.find(OPEN) {
            out.push_str(&rest[..rel_open]);
            let after_open = &rest[rel_open + OPEN.len()..];
            if let Some(rel_close) = after_open.find(CLOSE) {
                let token = &after_open[..rel_close];
                if let Some(id) = parse_ref(token) {
                    let value = outputs.get(id).copied().unwrap_or("<unavailable>");
                    out.push_str(&truncate_chars(value, per_ref_bytes));
                } else {
                    out.push_str(OPEN);
                    out.push_str(token);
                    out.push_str(CLOSE);
                }
                i += rel_open + OPEN.len() + rel_close + CLOSE.len();
                continue;
            }
            out.push_str(OPEN);
            i += rel_open + OPEN.len();
        } else {
            out.push_str(rest);
            break;
        }
    }
    out
}

/// Iterate over every `{{...}}` token body inside `prompt`. Yields the
/// content between markers, not including the markers themselves.
fn iter_tokens(prompt: &str) -> impl Iterator<Item = &str> {
    let mut idx = 0;
    std::iter::from_fn(move || {
        if idx >= prompt.len() {
            return None;
        }
        let rest = &prompt[idx..];
        let rel_open = rest.find(OPEN)?;
        let after_open_idx = idx + rel_open + OPEN.len();
        let after_open = &prompt[after_open_idx..];
        let rel_close = after_open.find(CLOSE)?;
        let token = &prompt[after_open_idx..after_open_idx + rel_close];
        idx = after_open_idx + rel_close + CLOSE.len();
        Some(token)
    })
}

/// Parse a `steps.<id>.output` token body and return the `<id>` slice if it
/// matches the supported shape. The id is the substring between the two
/// dots; we deliberately do not allow whitespace or further attributes so
/// that future extensions (`steps.<id>.error`, `plan.id`) can be added
/// without ambiguity.
fn parse_ref(token: &str) -> Option<&str> {
    let inner = token.strip_prefix("steps.")?;
    let id = inner.strip_suffix(".output")?;
    if id.is_empty() || id.contains('.') {
        return None;
    }
    Some(id)
}

/// Compute, for every step, the set of transitive dependency IDs.
fn transitive_deps(plan: &Plan) -> HashMap<&str, HashSet<&str>> {
    let mut direct: HashMap<&str, Vec<&str>> = HashMap::new();
    for step in &plan.steps {
        direct.insert(
            step.id.as_str(),
            step.depends_on.iter().map(String::as_str).collect(),
        );
    }

    let mut out: HashMap<&str, HashSet<&str>> = HashMap::new();
    for step in &plan.steps {
        let mut acc: HashSet<&str> = HashSet::new();
        let mut stack: Vec<&str> = direct.get(step.id.as_str()).cloned().unwrap_or_default();
        while let Some(id) = stack.pop() {
            if acc.insert(id) {
                if let Some(parents) = direct.get(id) {
                    stack.extend(parents.iter().copied());
                }
            }
        }
        out.insert(step.id.as_str(), acc);
    }
    out
}

/// Truncate `s` to at most `max_bytes`, respecting UTF-8 boundaries. When
/// truncation occurs, append a visible `…[+N bytes]` marker so a reader can
/// see the loss. Mirrors the helper in `flowd-mcp::summarizer` rather than
/// pulling that crate as a dep.
fn truncate_chars(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    let target = max_bytes.saturating_sub(16);
    let mut cut = 0usize;
    for (i, _) in s.char_indices() {
        if i > target {
            break;
        }
        cut = i;
    }
    format!("{}…[+{} bytes]", &s[..cut], s.len() - cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::orchestration::{Plan, PlanStep, StepStatus};

    fn step(id: &str, deps: &[&str], prompt: &str) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: prompt.into(),
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

    #[test]
    fn extract_finds_each_ref_once() {
        let refs = extract_step_refs(
            "use {{steps.a.output}} and {{steps.b.output}} and {{steps.a.output}} again",
        );
        assert_eq!(refs, vec!["a".to_owned(), "b".to_owned()]);
    }

    #[test]
    fn extract_ignores_malformed_tokens() {
        let refs = extract_step_refs("not a token: {{wrong}} {{steps..output}} {{steps.x.error}}");
        assert!(refs.is_empty());
    }

    #[test]
    fn validate_accepts_transitive_dep() {
        let plan = Plan::new(
            "p",
            vec![
                step("a", &[], "first"),
                step("b", &["a"], "depends on a"),
                step("c", &["b"], "carry {{steps.a.output}} forward"),
            ],
        );
        validate_step_references(&plan).unwrap();
    }

    #[test]
    fn validate_rejects_self_reference() {
        let plan = Plan::new("p", vec![step("a", &[], "loop {{steps.a.output}}")]);
        let err = validate_step_references(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("its own output")));
    }

    #[test]
    fn validate_rejects_unknown_ref() {
        let plan = Plan::new("p", vec![step("a", &[], "use {{steps.ghost.output}}")]);
        let err = validate_step_references(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("ghost")));
    }

    #[test]
    fn validate_rejects_non_dependency_ref() {
        let plan = Plan::new(
            "p",
            vec![
                step("a", &[], "first"),
                step("b", &[], "use {{steps.a.output}}"),
            ],
        );
        let err = validate_step_references(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("does not")));
    }

    #[test]
    fn substitute_replaces_known_refs() {
        let mut outputs = HashMap::new();
        outputs.insert("a", "ALPHA");
        outputs.insert("b", "BETA");
        let out = substitute(
            "x={{steps.a.output}} y={{steps.b.output}} z={{steps.c.output}}",
            &outputs,
            DEFAULT_PER_REF_BYTES,
        );
        assert_eq!(out, "x=ALPHA y=BETA z=<unavailable>");
    }

    #[test]
    fn substitute_truncates_oversized_value() {
        let big = "x".repeat(50);
        let mut outputs = HashMap::new();
        outputs.insert("a", big.as_str());
        let out = substitute("v={{steps.a.output}}", &outputs, 32);
        assert!(out.starts_with("v="));
        assert!(
            out.contains("…[+"),
            "expected truncation marker, got: {out}"
        );
        assert!(out.len() <= 32 + 8, "out unexpectedly long: {out}");
    }

    #[test]
    fn substitute_passes_through_malformed_tokens() {
        let outputs = HashMap::new();
        let out = substitute(
            "keep {{not a ref}} verbatim",
            &outputs,
            DEFAULT_PER_REF_BYTES,
        );
        assert_eq!(out, "keep {{not a ref}} verbatim");
    }

    #[test]
    fn substitute_handles_unterminated_braces() {
        let outputs = HashMap::new();
        let out = substitute(
            "trailing {{open without close",
            &outputs,
            DEFAULT_PER_REF_BYTES,
        );
        assert_eq!(out, "trailing {{open without close");
    }
}
