//! [`PlanCompiler`] implementations that back the prose-first
//! `plan_create` / `plan_answer` / `plan_refine` MCP surface.
//!
//! `flowd-mcp` ships three [`PlanCompiler`]s, each with a distinct role:
//!
//! | Type                       | When to use                                                                 |
//! |----------------------------|-----------------------------------------------------------------------------|
//! | [`StubPlanCompiler`]       | Default. No LLM, no network. Parses structured-markdown prose into a DAG.   |
//! | [`LlmPlanCompiler`]        | Future LLM-backed compiler. Stubbed in this PR; real wiring lands next.     |
//! | [`RejectingPlanCompiler`]  | Test double / "prose-first disabled" build flag. Always errors.             |
//!
//! ## Trait contract recap
//!
//! Every implementation must honour the [`PlanCompiler`] invariants:
//!
//! * `definition.is_some()` iff `open_questions.is_empty()`.
//! * Every emitted [`OpenQuestion::depends_on_decisions`] entry must
//!   reference either a question already answered or one that will appear
//!   in this same `CompileOutput`'s `new_decisions`.
//! * Question ids are stable across rounds (the executor uses them as the
//!   handle for invalidation).
//!
//! ## Stub-compiler grammar
//!
//! The [`StubPlanCompiler`] is intentionally tiny so that operators can
//! drive the prose-first loop end-to-end without an LLM, as long as their
//! prose is already structured. It recognises:
//!
//! ```text
//! # <plan name>                           (optional, sets PlanDefinition.name)
//!
//! ## <step-id> [agent: <agent_type>]      (depends_on omitted -> no deps)
//! prompt body line 1
//! prompt body line 2
//!
//! ## <step-id> [agent: <agent_type>] depends_on: [a, b]
//! prompt body
//! ```
//!
//! Anything that does not match a step heading and follows one is treated
//! as part of the preceding step's prompt. Unrecognised content before the
//! first step heading is silently dropped (the original prose is still
//! preserved verbatim in [`Plan::source_doc`]). Validation done in the
//! parser:
//!
//! * at least one step,
//! * unique step ids,
//! * each `depends_on` id refers to a defined step,
//! * no self-cycle (full cycle detection is left to `validate_plan`).
//!
//! On any parse failure the stub returns one [`OpenQuestion`] with
//! `allow_explain_more = true`, instructing the caller to either restructure
//! the prose (via `plan_refine`) or paste a structured version through
//! `Answer::ExplainMore`.

use std::collections::HashSet;
use std::future::Future;
use std::sync::Arc;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{
    Answer, CompileOutput, OpenQuestion, PlanCompiler, PlanDefinition, PlanDraftSnapshot,
    StepDefinition,
};

// ---------------------------------------------------------------------------
// RejectingPlanCompiler
// ---------------------------------------------------------------------------

/// [`PlanCompiler`] that refuses every call with a clear validation error.
///
/// Useful as a test double when a handler test wants to assert that the
/// prose-first surface is reachable but does not want to script a
/// [`flowd_core::orchestration::MockPlanCompiler`], or as a build-time
/// flag for deployments that want to disable prose-first plans entirely.
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
            "prose-first plan creation is disabled in this build (called {method}); \
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

// ---------------------------------------------------------------------------
// StubPlanCompiler
// ---------------------------------------------------------------------------

/// Question id surfaced when the stub cannot parse the prose.
///
/// Stable so callers can match on it and so subsequent rounds keep the
/// same id (the executor uses ids as the invalidation handle).
const STRUCTURE_QUESTION_ID: &str = "stub.structure_required";

/// Deterministic, no-LLM [`PlanCompiler`] that parses structured-markdown
/// prose into a [`PlanDefinition`].
///
/// See the module docs for the grammar.
#[derive(Debug, Default, Clone, Copy)]
pub struct StubPlanCompiler;

impl StubPlanCompiler {
    /// Construct a new stub compiler. Cheap; the type is zero-sized.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl PlanCompiler for StubPlanCompiler {
    async fn compile_prose(&self, prose: String, _project: String) -> Result<CompileOutput> {
        Ok(parse_or_pending(
            &prose, &prose, /* surface_error_in_prompt */ true,
        ))
    }

    async fn apply_answers(
        &self,
        snapshot: PlanDraftSnapshot,
        answers: Vec<(String, Answer)>,
        defer_remaining: bool,
    ) -> Result<CompileOutput> {
        // The stub cannot interpret `Choose` answers (it has nothing to
        // pick between). The only signal it can act on is `ExplainMore`,
        // whose `note` field is the natural channel for the user (or a
        // helper LLM running outside flowd) to paste a structured version
        // of the plan.
        let extra: String = answers
            .iter()
            .filter_map(|(_, a)| match a {
                Answer::ExplainMore { note } if !note.trim().is_empty() => Some(note.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n\n");

        let base = snapshot.source_doc.unwrap_or_default();
        let candidate = if extra.is_empty() {
            base.clone()
        } else if base.trim().is_empty() {
            extra.clone()
        } else {
            format!("{base}\n\n{extra}")
        };

        let out = parse_or_pending(&candidate, &candidate, true);

        if defer_remaining && out.definition.is_none() {
            // The stub has no model, so it cannot invent best-effort
            // answers. Surface that explicitly so callers do not loop.
            return Err(FlowdError::PlanValidation(
                "StubPlanCompiler cannot honour `defer_remaining`: it has no LLM to invent \
                 answers from. Either restructure the prose via plan_refine, or wire an \
                 LlmPlanCompiler into the daemon."
                    .to_string(),
            ));
        }
        Ok(out)
    }

    async fn refine(
        &self,
        _snapshot: PlanDraftSnapshot,
        feedback: String,
    ) -> Result<CompileOutput> {
        // Refinement contract for the stub: `feedback` is the new
        // source_doc verbatim. The user is expected to paste a corrected
        // structured-markdown plan. This is the documented escape hatch
        // for environments without an LLM.
        Ok(parse_or_pending(&feedback, &feedback, true))
    }
}

/// Parse `prose` and turn the outcome into a [`CompileOutput`].
///
/// `source_doc` is the value the executor will store into
/// [`flowd_core::orchestration::Plan::source_doc`]; in the typical case
/// it is identical to `prose`, but callers that synthesise a candidate
/// (e.g. base + extra in `apply_answers`) can pass the synthesised text.
///
/// When `surface_error_in_prompt` is true and parsing fails, the parse
/// error is included in the open question's prompt so the caller can see
/// exactly what went wrong without re-running the parser. We always set
/// it true today; the parameter exists so future callers (e.g. a
/// silent-mode CLI) can opt out.
fn parse_or_pending(prose: &str, source_doc: &str, surface_error_in_prompt: bool) -> CompileOutput {
    match parse_structured(prose) {
        Ok(definition) => CompileOutput::ready(source_doc, definition),
        Err(err) => CompileOutput::pending(
            source_doc,
            vec![structure_question(err.as_str(), surface_error_in_prompt)],
        ),
    }
}

/// Build the open question the stub surfaces when parsing fails.
fn structure_question(reason: &str, include_reason: bool) -> OpenQuestion {
    let prompt = if include_reason && !reason.is_empty() {
        format!(
            "Could not parse the prose plan deterministically: {reason}. \
             Either restructure your prose so each step looks like \
             `## <step-id> [agent: <type>] depends_on: [a, b]` followed by the \
             prompt body, or paste a structured version via Answer::ExplainMore."
        )
    } else {
        "Could not parse the prose plan deterministically; \
         please structure it or attach an LLM compiler."
            .to_string()
    };

    OpenQuestion {
        id: STRUCTURE_QUESTION_ID.to_string(),
        prompt,
        rationale: "The stub compiler is parser-only and needs each step to be expressed \
                    in the documented structured-markdown grammar."
            .to_string(),
        // No options to choose from; the only meaningful response is
        // `ExplainMore { note: <restructured prose> }` or a `plan_refine`
        // call with the structured prose as feedback.
        options: Vec::new(),
        allow_explain_more: true,
        allow_none: false,
        depends_on_decisions: Vec::new(),
    }
}

/// Hand-rolled line-based parser for the stub grammar. Returns
/// `Err(reason)` with a human-readable explanation on failure.
fn parse_structured(prose: &str) -> std::result::Result<PlanDefinition, String> {
    let mut name: Option<String> = None;
    let mut steps: Vec<StepDefinition> = Vec::new();
    let mut current_header: Option<StepHeader> = None;
    let mut current_body: Vec<&str> = Vec::new();

    let flush = |header: StepHeader,
                 body: &[&str],
                 acc: &mut Vec<StepDefinition>|
     -> std::result::Result<(), String> {
        let prompt = body.join("\n").trim().to_string();
        if prompt.is_empty() {
            return Err(format!("step '{}' has no prompt body", header.id));
        }
        acc.push(StepDefinition {
            id: header.id,
            agent_type: header.agent_type,
            prompt,
            depends_on: header.depends_on,
            timeout_secs: None,
            retry_count: 0,
        });
        Ok(())
    };

    for line in prose.lines() {
        // Optional H1 plan title -- only honoured before the first step
        // heading. `# foo` matches; `## foo` does not (different prefix).
        if name.is_none() && current_header.is_none() && steps.is_empty() {
            if let Some(rest) = line.strip_prefix("# ") {
                let trimmed = rest.trim();
                if !trimmed.is_empty() {
                    name = Some(trimmed.to_string());
                    continue;
                }
            }
        }

        if let Some(header) = parse_step_header(line) {
            if let Some(prev) = current_header.take() {
                flush(prev, &current_body, &mut steps)?;
                current_body.clear();
            }
            current_header = Some(header);
        } else if current_header.is_some() {
            current_body.push(line);
        }
        // Lines before the first step heading that are not the H1 title
        // are deliberately ignored. The original prose is preserved
        // verbatim by the executor in `Plan::source_doc`, so nothing is
        // lost in the audit trail.
    }

    if let Some(prev) = current_header.take() {
        flush(prev, &current_body, &mut steps)?;
    }

    if steps.is_empty() {
        return Err("no `## <id> [agent: <type>]` step headings found".into());
    }

    // Unique ids.
    let mut seen: HashSet<&String> = HashSet::new();
    for s in &steps {
        if !seen.insert(&s.id) {
            return Err(format!("duplicate step id: '{}'", s.id));
        }
    }

    // Dependencies refer to defined ids, no self-cycle. Full multi-step
    // cycle detection is left to `validate_plan` downstream so we don't
    // duplicate that logic here.
    let id_set: HashSet<&String> = steps.iter().map(|s| &s.id).collect();
    for s in &steps {
        for dep in &s.depends_on {
            if dep == &s.id {
                return Err(format!("step '{}' depends on itself", s.id));
            }
            if !id_set.contains(dep) {
                return Err(format!(
                    "step '{}' depends_on '{dep}' which is not defined in this plan",
                    s.id
                ));
            }
        }
    }

    Ok(PlanDefinition {
        name: name.unwrap_or_else(|| "prose-plan".to_string()),
        project: None,
        steps,
    })
}

#[derive(Debug)]
struct StepHeader {
    id: String,
    agent_type: String,
    depends_on: Vec<String>,
}

/// Parse a single `## <id> [agent: <type>] [depends_on: [...]]` line.
///
/// Returns `None` for any line that does not match the grammar; the
/// caller treats those as either body text (if a step is open) or
/// preamble (if not).
fn parse_step_header(line: &str) -> Option<StepHeader> {
    let body = line.strip_prefix("## ")?.trim();
    if body.is_empty() {
        return None;
    }

    // <id> is the first whitespace-delimited token.
    let id_end = body.find(char::is_whitespace)?;
    let id = body[..id_end].trim().to_string();
    if id.is_empty() {
        return None;
    }
    let after_id = body[id_end..].trim_start();

    // [agent: <type>]
    let agent_marker = "[agent:";
    let agent_start = after_id.find(agent_marker)?;
    let after_agent = &after_id[agent_start + agent_marker.len()..];
    let agent_close = after_agent.find(']')?;
    let agent_type = after_agent[..agent_close].trim().to_string();
    if agent_type.is_empty() {
        return None;
    }
    let after_bracket = after_agent[agent_close + 1..].trim_start();

    // Optional `depends_on: [a, b, c]`. Anything else after the agent
    // bracket is ignored by the stub (room for future fields without
    // breaking existing inputs).
    let marker = "depends_on:";
    let depends_on = match after_bracket.find(marker) {
        Some(idx) => {
            let after_marker = after_bracket[idx + marker.len()..].trim_start();
            let open = after_marker.find('[')?;
            let close = after_marker[open + 1..].find(']')?;
            let inner = &after_marker[open + 1..open + 1 + close];
            inner
                .split(',')
                .map(|s| s.trim().to_string())
                .filter(|s| !s.is_empty())
                .collect()
        }
        None => Vec::new(),
    };

    Some(StepHeader {
        id,
        agent_type,
        depends_on,
    })
}

// ---------------------------------------------------------------------------
// LlmPlanCompiler -- placeholder until the real LLM wiring lands
// ---------------------------------------------------------------------------

/// Callback shape the future [`LlmPlanCompiler`] will invoke to talk to a
/// model. Modelled after [`flowd_core::memory::Summarizer`] so that the
/// MCP layer can plug in a `sampling/createMessage` adapter (or any other
/// transport) without changing the compiler's public surface.
///
/// Today the trait is unused at runtime: every [`LlmPlanCompiler`] method
/// returns `FlowdError::Internal("not yet wired")` instead of calling
/// the callback. The trait is shipped now so downstream code (and tests)
/// can compile against the final shape.
pub trait LlmCallback: Send + Sync {
    /// Send `prompt` to the underlying model and return the completion.
    ///
    /// # Errors
    /// Implementations should return `FlowdError::Internal` for transport
    /// failures and `FlowdError::PlanValidation` for model-rejected
    /// prompts (e.g. content-policy rejections).
    fn complete(&self, prompt: String) -> impl Future<Output = Result<String>> + Send;
}

/// LLM-backed [`PlanCompiler`] -- skeleton only.
///
/// All three trait methods currently return
/// `FlowdError::Internal("LlmPlanCompiler::<method> is not yet wired ...")`.
/// The real implementation (prompt assembly, response parsing, JSON-schema
/// validation) lands in the follow-up PR. The type is generic over an
/// [`LlmCallback`] so the wiring story is settled now and the daemon's
/// composition code can be migrated in one focused diff later.
#[derive(Debug, Clone)]
pub struct LlmPlanCompiler<C: LlmCallback> {
    callback: Arc<C>,
}

impl<C: LlmCallback> LlmPlanCompiler<C> {
    /// Construct a new LLM-backed compiler around `callback`.
    #[must_use]
    pub fn new(callback: Arc<C>) -> Self {
        Self { callback }
    }

    /// Borrow the callback. Exposed so the future implementation (and any
    /// integration tests) can verify wiring; today it is the only way to
    /// observe the field given that no method invokes the callback yet.
    #[must_use]
    pub fn callback(&self) -> &Arc<C> {
        &self.callback
    }

    fn not_yet_wired(method: &'static str) -> FlowdError {
        FlowdError::Internal(format!(
            "LlmPlanCompiler::{method} is not yet wired; \
             a follow-up PR will implement the LLM-driven prose-first compiler. \
             Use StubPlanCompiler in the meantime."
        ))
    }
}

impl<C: LlmCallback + 'static> PlanCompiler for LlmPlanCompiler<C> {
    async fn compile_prose(&self, _prose: String, _project: String) -> Result<CompileOutput> {
        Err(Self::not_yet_wired("compile_prose"))
    }

    async fn apply_answers(
        &self,
        _snapshot: PlanDraftSnapshot,
        _answers: Vec<(String, Answer)>,
        _defer_remaining: bool,
    ) -> Result<CompileOutput> {
        Err(Self::not_yet_wired("apply_answers"))
    }

    async fn refine(
        &self,
        _snapshot: PlanDraftSnapshot,
        _feedback: String,
    ) -> Result<CompileOutput> {
        Err(Self::not_yet_wired("refine"))
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::Answer;

    // -----------------------------------------------------------------
    // RejectingPlanCompiler -- existing contract from PR2
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn rejecting_compiler_returns_unavailable_validation_error_for_every_method() {
        let c = RejectingPlanCompiler::new();

        let err = c
            .compile_prose("hi".into(), "proj".into())
            .await
            .unwrap_err();
        match err {
            FlowdError::PlanValidation(m) => {
                assert!(m.contains("compile_prose"));
                assert!(m.contains("disabled"));
            }
            other => panic!("expected PlanValidation, got {other:?}"),
        }

        let snap = snapshot(None);
        assert!(matches!(
            c.apply_answers(snap.clone(), vec![], false).await,
            Err(FlowdError::PlanValidation(_))
        ));
        assert!(matches!(
            c.refine(snap, "tweak".into()).await,
            Err(FlowdError::PlanValidation(_))
        ));
    }

    // -----------------------------------------------------------------
    // StubPlanCompiler -- happy paths
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn stub_compile_prose_parses_structured_markdown_into_definition() {
        let prose = r"# refactor-auth

## extract-jwt [agent: rust-engineer]
Pull the JWT signing/verification helpers out of `auth/mod.rs` into
their own module so we can swap the algorithm without touching the
session layer.

## migrate-callers [agent: rust-engineer] depends_on: [extract-jwt]
Update every call site to use the new module.

## smoke-test [agent: tester] depends_on: [migrate-callers]
Run the auth integration tests and capture failures.
";

        let out = StubPlanCompiler::new()
            .compile_prose(prose.to_string(), "proj".into())
            .await
            .expect("compile_prose returns Ok");

        assert!(out.open_questions.is_empty(), "no questions on success");
        assert!(out.new_decisions.is_empty(), "stub never auto-fills");
        assert_eq!(out.source_doc, prose, "source_doc preserved verbatim");

        let def = out.definition.expect("definition present on success");
        assert_eq!(def.name, "refactor-auth");
        assert_eq!(def.steps.len(), 3);
        assert_eq!(def.steps[0].id, "extract-jwt");
        assert_eq!(def.steps[0].agent_type, "rust-engineer");
        assert!(def.steps[0].depends_on.is_empty());
        assert!(
            def.steps[0].prompt.contains("JWT signing"),
            "multi-line prompt body collected"
        );

        assert_eq!(def.steps[1].depends_on, vec!["extract-jwt".to_string()]);
        assert_eq!(def.steps[2].depends_on, vec!["migrate-callers".to_string()]);
    }

    #[tokio::test]
    async fn stub_compile_prose_falls_back_to_default_name_without_h1() {
        let prose = "## a [agent: echo]\nhi\n";
        let def = StubPlanCompiler::new()
            .compile_prose(prose.into(), "proj".into())
            .await
            .unwrap()
            .definition
            .unwrap();
        assert_eq!(def.name, "prose-plan");
    }

    #[tokio::test]
    async fn stub_compile_prose_ignores_preamble_before_first_step_heading() {
        let prose =
            "# title\n\nSome narrative paragraph.\n\nAnother one.\n\n## a [agent: echo]\nhi\n";
        let def = StubPlanCompiler::new()
            .compile_prose(prose.into(), "proj".into())
            .await
            .unwrap()
            .definition
            .unwrap();
        assert_eq!(def.name, "title");
        assert_eq!(def.steps.len(), 1);
        assert_eq!(def.steps[0].prompt, "hi");
    }

    // -----------------------------------------------------------------
    // StubPlanCompiler -- failure paths
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn stub_compile_prose_returns_structure_question_for_freeform_prose() {
        let prose = "I want a plan that refactors the auth module and then runs the tests.";

        let out = StubPlanCompiler::new()
            .compile_prose(prose.to_string(), "proj".into())
            .await
            .unwrap();

        assert!(out.definition.is_none());
        assert_eq!(out.open_questions.len(), 1);
        let q = &out.open_questions[0];
        assert_eq!(q.id, STRUCTURE_QUESTION_ID);
        assert!(q.allow_explain_more);
        assert!(!q.allow_none);
        assert!(q.options.is_empty());
        assert!(
            q.prompt.contains("step headings found"),
            "parse error surfaced in prompt: {}",
            q.prompt
        );
    }

    #[tokio::test]
    async fn stub_rejects_duplicate_step_ids() {
        let prose = "## a [agent: echo]\none\n## a [agent: echo]\ntwo\n";
        let out = StubPlanCompiler::new()
            .compile_prose(prose.into(), "p".into())
            .await
            .unwrap();
        assert!(out.definition.is_none());
        assert!(out.open_questions[0].prompt.contains("duplicate step id"));
    }

    #[tokio::test]
    async fn stub_rejects_self_cycle() {
        let prose = "## a [agent: echo] depends_on: [a]\nbody\n";
        let out = StubPlanCompiler::new()
            .compile_prose(prose.into(), "p".into())
            .await
            .unwrap();
        assert!(out.definition.is_none());
        assert!(out.open_questions[0].prompt.contains("depends on itself"));
    }

    #[tokio::test]
    async fn stub_rejects_dangling_dependency() {
        let prose = "## a [agent: echo] depends_on: [missing]\nbody\n";
        let out = StubPlanCompiler::new()
            .compile_prose(prose.into(), "p".into())
            .await
            .unwrap();
        assert!(out.definition.is_none());
        assert!(out.open_questions[0].prompt.contains("not defined"));
    }

    #[tokio::test]
    async fn stub_rejects_step_with_empty_body() {
        let prose = "## a [agent: echo]\n\n## b [agent: echo]\nhi\n";
        let out = StubPlanCompiler::new()
            .compile_prose(prose.into(), "p".into())
            .await
            .unwrap();
        assert!(out.definition.is_none());
        assert!(out.open_questions[0].prompt.contains("no prompt body"));
    }

    // -----------------------------------------------------------------
    // StubPlanCompiler -- apply_answers / refine
    // -----------------------------------------------------------------

    #[tokio::test]
    async fn stub_apply_answers_consumes_explain_more_and_compiles() {
        let snap = snapshot(Some("not structured at all".to_string()));
        let answers = vec![(
            STRUCTURE_QUESTION_ID.to_string(),
            Answer::ExplainMore {
                note: "## a [agent: echo]\nhello\n".into(),
            },
        )];

        let out = StubPlanCompiler::new()
            .apply_answers(snap, answers, false)
            .await
            .unwrap();

        let def = out.definition.expect("explain_more body parsed");
        assert_eq!(def.steps.len(), 1);
        assert_eq!(def.steps[0].id, "a");
    }

    #[tokio::test]
    async fn stub_apply_answers_without_explain_more_re_parses_source_doc() {
        // No fresh prose supplied; same source_doc is still unparseable,
        // so we expect another pending output (not an error).
        let snap = snapshot(Some("just narrative".to_string()));
        let out = StubPlanCompiler::new()
            .apply_answers(snap, vec![], false)
            .await
            .unwrap();
        assert!(out.definition.is_none());
        assert_eq!(out.open_questions.len(), 1);
    }

    #[tokio::test]
    async fn stub_apply_answers_rejects_defer_remaining_when_unparseable() {
        let snap = snapshot(Some("nothing structured".to_string()));
        let err = StubPlanCompiler::new()
            .apply_answers(snap, vec![], true)
            .await
            .unwrap_err();
        match err {
            FlowdError::PlanValidation(m) => assert!(m.contains("defer_remaining")),
            other => panic!("expected PlanValidation, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stub_refine_treats_feedback_as_new_source() {
        let snap = snapshot(Some("# old\n\nfreeform".into()));
        let feedback = "# new\n\n## s [agent: echo]\ndo it\n";
        let out = StubPlanCompiler::new()
            .refine(snap, feedback.to_string())
            .await
            .unwrap();
        let def = out.definition.expect("refined feedback parsed");
        assert_eq!(def.name, "new");
        assert_eq!(def.steps[0].id, "s");
        assert_eq!(out.source_doc, feedback);
    }

    // -----------------------------------------------------------------
    // LlmPlanCompiler -- skeleton
    // -----------------------------------------------------------------

    /// Test callback that panics if invoked. Proves the
    /// `LlmPlanCompiler` skeleton never reaches the callback today.
    struct PanickingCallback;

    impl LlmCallback for PanickingCallback {
        async fn complete(&self, _prompt: String) -> Result<String> {
            panic!("LlmPlanCompiler must not invoke its callback in this PR");
        }
    }

    fn llm_compiler() -> LlmPlanCompiler<PanickingCallback> {
        LlmPlanCompiler::new(Arc::new(PanickingCallback))
    }

    #[tokio::test]
    async fn llm_compile_prose_returns_not_yet_wired_internal_error() {
        let err = llm_compiler()
            .compile_prose("anything".into(), "proj".into())
            .await
            .unwrap_err();
        match err {
            FlowdError::Internal(m) => {
                assert!(m.contains("LlmPlanCompiler::compile_prose"));
                assert!(m.contains("not yet wired"));
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn llm_apply_answers_returns_not_yet_wired_internal_error() {
        assert!(matches!(
            llm_compiler()
                .apply_answers(snapshot(None), vec![], false)
                .await,
            Err(FlowdError::Internal(m)) if m.contains("apply_answers")
        ));
    }

    #[tokio::test]
    async fn llm_refine_returns_not_yet_wired_internal_error() {
        assert!(matches!(
            llm_compiler()
                .refine(snapshot(None), "tweak".into())
                .await,
            Err(FlowdError::Internal(m)) if m.contains("refine")
        ));
    }

    #[test]
    fn llm_callback_accessor_exposes_arc_for_future_wiring() {
        let c = llm_compiler();
        assert_eq!(Arc::strong_count(c.callback()), 1);
    }

    // -----------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------

    fn snapshot(source_doc: Option<String>) -> PlanDraftSnapshot {
        PlanDraftSnapshot {
            plan_name: "p".into(),
            project: "proj".into(),
            source_doc,
            open_questions: vec![],
            decisions: vec![],
            previous_definition: None,
        }
    }
}
