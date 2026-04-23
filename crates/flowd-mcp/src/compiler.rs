//! [`PlanCompiler`] implementations that back the prose-first
//! `plan_create` / `plan_answer` / `plan_refine` MCP surface.
//!
//! `flowd-mcp` ships three [`PlanCompiler`]s, each with a distinct role:
//!
//! | Type                       | When to use                                                                 |
//! |----------------------------|-----------------------------------------------------------------------------|
//! | [`StubPlanCompiler`]       | Default. No LLM, no network. Parses structured-markdown prose into a DAG.   |
//! | [`LlmPlanCompiler`]        | Real LLM-backed compiler. Generic over [`LlmCallback`]; the daemon wires    |
//! |                            | [`crate::llm::OpenAiCompatibleCallback`] for local `mlx_lm.server` traffic. |
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
use std::fmt::Write as _;
use std::future::Future;
use std::sync::Arc;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{
    Answer, CompileOutput, DecisionRecord, OpenQuestion, PlanCompiler, PlanDefinition,
    PlanDraftSnapshot, QuestionOption, StepDefinition,
};
use serde::Deserialize;

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
// LlmPlanCompiler
// ---------------------------------------------------------------------------

/// Question id surfaced when the LLM repeatedly fails to produce a valid
/// JSON response. Stable across rounds so the executor can match on it
/// for invalidation, and deliberately distinct from
/// [`STRUCTURE_QUESTION_ID`] so callers can tell which compiler bailed.
const LLM_STRUCTURE_QUESTION_ID: &str = "llm.structure_required";

/// Callback shape the [`LlmPlanCompiler`] invokes to talk to a model.
/// Modelled after [`flowd_core::memory::Summarizer`] so that the MCP
/// layer can plug in any transport (HTTP via
/// [`crate::llm::OpenAiCompatibleCallback`], stdio CLI shellouts,
/// `sampling/createMessage` adapter) without changing the compiler's
/// public surface.
pub trait LlmCallback: Send + Sync {
    /// Send `prompt` to the underlying model and return the completion.
    ///
    /// The prompt is the full assistant input including any system /
    /// schema instructions; the callback is expected to be a pure
    /// "string in, string out" transport with no awareness of plan
    /// semantics.
    ///
    /// # Errors
    /// Implementations should return `FlowdError::Internal` for transport
    /// failures and `FlowdError::PlanValidation` for model-rejected
    /// prompts (e.g. content-policy rejections).
    fn complete(&self, prompt: String) -> impl Future<Output = Result<String>> + Send;
}

/// LLM-backed [`PlanCompiler`].
///
/// Drives the prose-first loop by sending tightly scoped prompts to a
/// model behind an [`LlmCallback`] and parsing the responses into
/// [`CompileOutput`]s. The compiler enforces a strict JSON output shape
/// (see the `LlmCompileResponse` wire type), retries once with a
/// corrective prompt when the first response fails to parse, and falls
/// back to a structured `llm.structure_required` open question rather
/// than bubbling raw model errors to the user.
///
/// ## Determinism
///
/// The compiler is deterministic given a deterministic callback. Tests
/// can therefore use a hand-rolled mock implementing [`LlmCallback`]
/// instead of calling out to the real model. The `--ignored` end-to-end
/// test in `tests/llm_e2e.rs` exercises a live `mlx_lm.server` once
/// per release.
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

    /// Borrow the callback. Exposed so tests and the daemon's startup
    /// banner can observe wiring without taking ownership.
    #[must_use]
    pub fn callback(&self) -> &Arc<C> {
        &self.callback
    }

    /// Run a prompt through the callback with at most one corrective
    /// retry on parse failure, then fall back to a structure-required
    /// open question. The fallback path is what keeps the user out of
    /// the "model just answered with prose" deadlock.
    ///
    /// Transport errors are propagated as-is (the user can't recover by
    /// rephrasing prose if MLX is down); only *parse* failures trigger
    /// the retry/fallback path.
    async fn invoke(&self, prompt: String, source_doc: &str) -> Result<CompileOutput> {
        let raw = self.callback.complete(prompt.clone()).await?;
        match parse_llm_response(&raw) {
            Ok(out) => Ok(out.into_compile_output(source_doc)),
            Err(parse_err) => {
                tracing::warn!(
                    error = %parse_err,
                    raw_preview = %preview(&raw, 256),
                    "LlmPlanCompiler: first response did not parse; retrying with corrective prompt"
                );
                let corrective = corrective_prompt(&prompt, &raw, &parse_err);
                let retry_raw = self.callback.complete(corrective).await?;
                match parse_llm_response(&retry_raw) {
                    Ok(out) => Ok(out.into_compile_output(source_doc)),
                    Err(retry_err) => {
                        tracing::warn!(
                            error = %retry_err,
                            raw_preview = %preview(&retry_raw, 256),
                            "LlmPlanCompiler: corrective retry still did not parse; falling back to structure-required question"
                        );
                        Ok(structure_required_fallback(
                            source_doc, &parse_err, &retry_err,
                        ))
                    }
                }
            }
        }
    }
}

impl<C: LlmCallback + 'static> PlanCompiler for LlmPlanCompiler<C> {
    async fn compile_prose(&self, prose: String, project: String) -> Result<CompileOutput> {
        let prompt = build_compile_prose_prompt(&prose, &project);
        self.invoke(prompt, &prose).await
    }

    async fn apply_answers(
        &self,
        snapshot: PlanDraftSnapshot,
        answers: Vec<(String, Answer)>,
        defer_remaining: bool,
    ) -> Result<CompileOutput> {
        let source_doc = snapshot.source_doc.clone().unwrap_or_default();
        let prompt = build_apply_answers_prompt(&snapshot, &answers, defer_remaining);
        self.invoke(prompt, &source_doc).await
    }

    async fn refine(&self, snapshot: PlanDraftSnapshot, feedback: String) -> Result<CompileOutput> {
        let source_doc = snapshot.source_doc.clone().unwrap_or_default();
        let prompt = build_refine_prompt(&snapshot, &feedback);
        self.invoke(prompt, &source_doc).await
    }
}

// ---------------------------------------------------------------------------
// LLM JSON wire types
// ---------------------------------------------------------------------------

/// JSON shape we ask the model to produce. Mirrors [`CompileOutput`] but
/// wire-friendlier:
///
/// * `definition` is `null` when questions remain.
/// * `decisions` (auto-fills) and `open_questions` are always present
///   as empty arrays rather than omitted, so the parser can validate
///   without a `serde(default)` per field.
///
/// The schema is documented inline in [`json_schema_string`] -- that
/// constant is what the prompts hand the model so the wire shape and
/// the parser stay in lockstep.
#[derive(Debug, Deserialize)]
struct LlmCompileResponse {
    #[serde(default)]
    plan_name: Option<String>,
    #[serde(default)]
    open_questions: Vec<LlmOpenQuestion>,
    #[serde(default)]
    decisions: Vec<LlmDecision>,
    #[serde(default)]
    definition: Option<LlmDefinition>,
}

#[derive(Debug, Deserialize)]
struct LlmOpenQuestion {
    id: String,
    prompt: String,
    #[serde(default)]
    rationale: String,
    #[serde(default)]
    options: Vec<LlmOption>,
    #[serde(default)]
    allow_explain_more: bool,
    #[serde(default)]
    allow_none: bool,
    #[serde(default)]
    depends_on_decisions: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct LlmOption {
    id: String,
    label: String,
    #[serde(default)]
    rationale: String,
}

#[derive(Debug, Deserialize)]
struct LlmDecision {
    question_id: String,
    chosen_option_id: String,
    #[serde(default)]
    depends_on_decisions: Vec<String>,
    /// Defaults to true on the wire because every decision the LLM
    /// emits is by definition a compiler-driven fill (the user-driven
    /// decisions are recorded by the handler before invocation). We
    /// still allow the model to explicitly mark `auto = false` for a
    /// caller who wants to record an "I would have picked this even if
    /// you'd asked" rationale; the executor doesn't currently
    /// distinguish, but the audit trail does.
    #[serde(default = "default_auto")]
    auto: bool,
}

const fn default_auto() -> bool {
    true
}

#[derive(Debug, Deserialize)]
struct LlmDefinition {
    #[serde(default)]
    name: Option<String>,
    steps: Vec<LlmStep>,
}

#[derive(Debug, Deserialize)]
struct LlmStep {
    id: String,
    #[serde(alias = "agent")]
    agent_type: String,
    prompt: String,
    #[serde(default)]
    depends_on: Vec<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    retry_count: u32,
}

impl LlmCompileResponse {
    /// Project the wire shape into a [`CompileOutput`] using
    /// `source_doc` for the source-of-truth prose. Validates the
    /// definition / question invariants of [`PlanCompiler`].
    ///
    /// We deliberately only run *local* invariants here -- full DAG
    /// validation (`validate_plan`) runs later in the executor when the
    /// definition is materialised, so we don't duplicate that logic.
    fn into_compile_output(self, source_doc: &str) -> CompileOutput {
        // Hard contract per `PlanCompiler`: definition.is_some() iff
        // open_questions.is_empty(). If the model produced both, the
        // open_questions win -- the user has more clarifications to do
        // and the partial definition can't be trusted.
        let drop_definition = !self.open_questions.is_empty();
        let definition = if drop_definition {
            None
        } else {
            self.definition.map(|d| d.into_definition(self.plan_name))
        };

        CompileOutput {
            source_doc: source_doc.to_string(),
            open_questions: self
                .open_questions
                .into_iter()
                .map(LlmOpenQuestion::into_open_question)
                .collect(),
            new_decisions: self
                .decisions
                .into_iter()
                .map(LlmDecision::into_decision_record)
                .collect(),
            definition,
        }
    }
}

impl LlmOpenQuestion {
    fn into_open_question(self) -> OpenQuestion {
        OpenQuestion {
            id: self.id,
            prompt: self.prompt,
            rationale: self.rationale,
            options: self
                .options
                .into_iter()
                .map(|o| QuestionOption {
                    id: o.id,
                    label: o.label,
                    rationale: o.rationale,
                })
                .collect(),
            allow_explain_more: self.allow_explain_more,
            allow_none: self.allow_none,
            depends_on_decisions: self.depends_on_decisions,
        }
    }
}

impl LlmDecision {
    fn into_decision_record(self) -> DecisionRecord {
        // We use `now()` for the timestamp because the wire shape doesn't
        // carry one -- the model can't possibly produce a meaningful
        // `decided_at`, and the handler/executor isn't in a position to
        // override it. This means the audit trail records the "compile
        // round when this auto-fill happened", which is exactly what a
        // human reviewer wants.
        DecisionRecord {
            question_id: self.question_id,
            chosen_option_id: self.chosen_option_id,
            depends_on_decisions: self.depends_on_decisions,
            auto: self.auto,
            decided_at: chrono::Utc::now(),
        }
    }
}

impl LlmDefinition {
    fn into_definition(self, top_level_name: Option<String>) -> PlanDefinition {
        PlanDefinition {
            name: self
                .name
                .or(top_level_name)
                .unwrap_or_else(|| "prose-plan".to_string()),
            project: None,
            steps: self
                .steps
                .into_iter()
                .map(|s| StepDefinition {
                    id: s.id,
                    agent_type: s.agent_type,
                    prompt: s.prompt,
                    depends_on: s.depends_on,
                    timeout_secs: s.timeout_secs,
                    retry_count: s.retry_count,
                })
                .collect(),
        }
    }
}

// ---------------------------------------------------------------------------
// Prompt assembly
// ---------------------------------------------------------------------------

/// Wire-shape JSON the model must emit. Embedded into every prompt.
///
/// We deliberately use a hand-rolled "shape doc" string rather than a
/// machine-readable JSON Schema:
///
/// 1. JSON Schema enforcement isn't available across all backends we
///    plan to target (`mlx_lm.server` in particular).
/// 2. A tightly scoped natural-language schema with a worked example
///    has empirically been the most reliable way to get small/mid
///    models to produce parseable JSON on the first try.
///
/// Keep this string and the `LlmCompileResponse` wire types in lockstep
/// -- the parse-error diagnostics here are the sole runtime check.
fn json_schema_string() -> &'static str {
    r#"You must reply with a single JSON object matching this exact shape:

{
  "plan_name": "string | null   // optional human label, falls back to 'prose-plan'",
  "open_questions": [             // empty array when the plan is ready to compile
    {
      "id": "stable-snake_case-id, used as the invalidation handle across rounds",
      "prompt": "the question text shown to the user",
      "rationale": "why this question matters; 1-3 sentences",
      "options": [
        { "id": "snake_case", "label": "short label", "rationale": "trade-off, 1-2 sentences" }
      ],
      "allow_explain_more": false,
      "allow_none": false,
      "depends_on_decisions": ["question_id_of_a_resolved_dependency"]
    }
  ],
  "decisions": [                  // empty array unless you are auto-filling answers
    {
      "question_id": "matches one of the questions you would have asked",
      "chosen_option_id": "matches that question's option id",
      "depends_on_decisions": [],
      "auto": true
    }
  ],
  "definition": null              // OR an object when open_questions is empty:
  // {
  //   "name": "optional, overrides plan_name",
  //   "steps": [
  //     {
  //       "id": "snake_case-step-id",
  //       "agent_type": "rust-engineer | tester | reviewer | ...",
  //       "prompt": "what this agent should do",
  //       "depends_on": ["other_step_id"],
  //       "timeout_secs": null,
  //       "retry_count": 0
  //     }
  //   ]
  // }
}

Strict rules:
- Output ONE JSON object, no leading/trailing prose, no markdown fences.
- "definition" must be null whenever "open_questions" is non-empty.
- "definition" must be a valid object whenever "open_questions" is empty.
- Reuse question ids across rounds when refining the same concern.
- Never refer to a step or decision id you didn't declare in this same response.
- Prefer snake_case for ids (`my_step`, `lib_choice`); kebab-case (`my-step`)
  is normalised to snake_case server-side for backwards compatibility with
  earlier rounds, so do not mix the two within one response."#
}

fn build_compile_prose_prompt(prose: &str, project: &str) -> String {
    format!(
        "{schema}\n\n\
         You are compiling a fresh prose plan into either an executable DAG \
         or a list of clarification questions.\n\n\
         Project: {project}\n\n\
         Prose plan to compile:\n```\n{prose}\n```\n\n\
         If the prose is unambiguous, emit a `definition` with no questions. \
         If material decisions are missing (library choice, architectural fork, \
         scope ambiguity), emit `open_questions` instead. Cap at 5 questions per \
         round; surface the highest-impact ones first.",
        schema = json_schema_string(),
    )
}

fn build_apply_answers_prompt(
    snapshot: &PlanDraftSnapshot,
    answers: &[(String, Answer)],
    defer_remaining: bool,
) -> String {
    let source_doc = snapshot.source_doc.as_deref().unwrap_or("");
    let answers_block = render_answers_block(answers);
    let still_open_block = render_open_questions_block(&snapshot.open_questions);
    let prior_decisions_block = render_decisions_block(&snapshot.decisions);

    let defer_clause = if defer_remaining {
        "The user has asked you to FILL IN best-effort answers for any still-open \
         questions rather than asking another round. For every still-open question \
         that you do not surface again, append a `decisions` entry with `auto: true` \
         and your best-guess option. Avoid emitting open_questions in this mode \
         unless a question raises a brand-new concern that the user must see."
    } else {
        "Surface only follow-up questions that are still material after applying \
         these answers. Drop any prior question whose answer is now implied."
    };

    format!(
        "{schema}\n\n\
         You are advancing an in-flight plan after the user submitted answers.\n\n\
         Project: {project}\n\
         Plan name: {plan_name}\n\n\
         Original prose:\n```\n{source_doc}\n```\n\n\
         Prior decisions (already recorded; do not re-emit):\n{prior_decisions}\n\n\
         Still-open questions before this round:\n{still_open}\n\n\
         User-submitted answers this round:\n{answers}\n\n\
         {defer_clause}",
        schema = json_schema_string(),
        project = snapshot.project,
        plan_name = snapshot.plan_name,
        prior_decisions = prior_decisions_block,
        still_open = still_open_block,
        answers = answers_block,
    )
}

fn build_refine_prompt(snapshot: &PlanDraftSnapshot, feedback: &str) -> String {
    let source_doc = snapshot.source_doc.as_deref().unwrap_or("");
    let still_open_block = render_open_questions_block(&snapshot.open_questions);
    let prior_decisions_block = render_decisions_block(&snapshot.decisions);

    format!(
        "{schema}\n\n\
         You are applying a freeform refinement instruction to an in-flight plan.\n\n\
         Project: {project}\n\
         Plan name: {plan_name}\n\n\
         Original prose:\n```\n{source_doc}\n```\n\n\
         Prior decisions:\n{prior_decisions}\n\n\
         Still-open questions:\n{still_open}\n\n\
         Refinement instruction:\n```\n{feedback}\n```\n\n\
         If the refinement raises new architectural concerns, you may RE-OPEN \
         questions (including ones whose decisions are now stale -- include a \
         `decisions` entry referring to the same question_id only if you are \
         keeping the prior choice). Otherwise, emit a fresh `definition` \
         reflecting the refined intent.",
        schema = json_schema_string(),
        project = snapshot.project,
        plan_name = snapshot.plan_name,
        prior_decisions = prior_decisions_block,
        still_open = still_open_block,
    )
}

fn render_answers_block(answers: &[(String, Answer)]) -> String {
    if answers.is_empty() {
        return "(none -- the user passed an empty answer list)".into();
    }
    let mut out = String::new();
    for (qid, a) in answers {
        // `writeln!` into a String never fails; the `let _ =` discards
        // the impossible Err arm without dragging in `unwrap()`.
        match a {
            Answer::Choose { option_id } => {
                let _ = writeln!(out, "- {qid}: choose `{option_id}`");
            }
            Answer::ExplainMore { note } => {
                let _ = writeln!(out, "- {qid}: explain_more (user note: {note:?})");
            }
            Answer::NoneOfThese => {
                let _ = writeln!(out, "- {qid}: none_of_these (propose new options)");
            }
        }
    }
    out
}

fn render_open_questions_block(qs: &[OpenQuestion]) -> String {
    if qs.is_empty() {
        return "(none)".into();
    }
    let mut out = String::new();
    for q in qs {
        let _ = writeln!(out, "- {} ({}): {}", q.id, q.prompt, q.rationale);
        for opt in &q.options {
            let _ = writeln!(out, "    * {} -- {} ({})", opt.id, opt.label, opt.rationale);
        }
    }
    out
}

fn render_decisions_block(ds: &[DecisionRecord]) -> String {
    if ds.is_empty() {
        return "(none)".into();
    }
    let mut out = String::new();
    for d in ds {
        let tag = if d.auto { " [auto]" } else { "" };
        let _ = writeln!(
            out,
            "- {}: chose `{}`{tag}",
            d.question_id, d.chosen_option_id
        );
    }
    out
}

// ---------------------------------------------------------------------------
// Response parsing + corrective retry
// ---------------------------------------------------------------------------

/// Try hard to parse `raw` as one of our wire shapes.
///
/// Strategy:
///   1. Strip leading/trailing whitespace.
///   2. Strip markdown code fences (```json ... ``` and ``` ... ```).
///   3. Best-effort: locate the first `{` and the matching `}` and try
///      that substring as JSON. This handles the "model added a polite
///      sentence before the JSON" failure mode.
///
/// Returns a structured `LlmCompileResponse` on success, or a
/// human-readable diagnostic string on failure (which the caller embeds
/// in the corrective prompt).
fn parse_llm_response(raw: &str) -> std::result::Result<LlmCompileResponse, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("response was empty".into());
    }

    let candidate = strip_code_fences(trimmed);
    let json_blob = locate_json_object(candidate).unwrap_or(candidate);

    let mut parsed: LlmCompileResponse = serde_json::from_str(json_blob)
        .map_err(|e| format!("response is not valid JSON for the schema: {e}"))?;

    // Normalise every id in the response to snake_case before we run
    // local invariants. Models in the wild (notably mid-sized open
    // coder models) routinely emit a mix of `kebab-case` and
    // `snake_case` ids within the same response, which the executor
    // accepts but which produces an inconsistent audit trail and
    // breaks ad-hoc string matching by callers. Doing the
    // normalisation here keeps the canonical wire shape consistent
    // with what the prompt asks for and means the invariant checks
    // below see ids that the executor will also see.
    normalize_response_ids(&mut parsed);

    // Local invariants -- the things we can check without dragging in
    // the executor's `validate_plan`. Anything caught here is something
    // the corrective prompt can teach the model to fix.
    if parsed.open_questions.is_empty() && parsed.definition.is_none() {
        return Err(
            "either `open_questions` must be non-empty OR `definition` must be present (got neither)".into(),
        );
    }
    if !parsed.open_questions.is_empty() && parsed.definition.is_some() {
        // Not technically a parse error -- `into_compile_output` drops
        // the definition in this case -- but flagging it during the
        // first pass nudges the model toward cleaner output on retry.
        return Err("set `definition` to null whenever `open_questions` is non-empty".into());
    }
    if let Some(def) = parsed.definition.as_ref() {
        if def.steps.is_empty() {
            return Err("`definition.steps` must be non-empty".into());
        }
        for s in &def.steps {
            if s.id.trim().is_empty() {
                return Err("every step needs a non-empty `id`".into());
            }
            if s.agent_type.trim().is_empty() {
                return Err(format!("step `{}` has empty `agent_type`", s.id));
            }
            if s.prompt.trim().is_empty() {
                return Err(format!("step `{}` has empty `prompt`", s.id));
            }
        }
        // Local uniqueness check on step ids -- cheap and catches a
        // common LLM failure mode (collision via lazy renaming).
        let mut seen: HashSet<&str> = HashSet::new();
        for s in &def.steps {
            if !seen.insert(&s.id) {
                return Err(format!("duplicate step id `{}` in definition", s.id));
            }
        }
    }
    for q in &parsed.open_questions {
        if q.id.trim().is_empty() {
            return Err("every open_question needs a non-empty `id`".into());
        }
        if q.prompt.trim().is_empty() {
            return Err(format!("question `{}` has empty `prompt`", q.id));
        }
    }

    Ok(parsed)
}

/// Strip a single `````` ... `````` (optionally `json`-tagged) wrapper
/// if the model added one. Idempotent on un-fenced input.
fn strip_code_fences(s: &str) -> &str {
    let stripped = s.trim();
    let Some(after_open) = stripped.strip_prefix("```") else {
        return stripped;
    };
    // Drop optional language tag on the same line.
    let after_lang = after_open
        .find('\n')
        .map_or(after_open, |i| &after_open[i + 1..]);
    after_lang.strip_suffix("```").unwrap_or(after_lang).trim()
}

/// Locate the first balanced top-level `{...}` block. Naive but
/// sufficient for our usecase since we control the schema and don't
/// expect deeply nested escaped braces in prose. Walks in O(n) and
/// honours JSON string boundaries so a `}` inside a string literal
/// doesn't terminate early.
fn locate_json_object(s: &str) -> Option<&str> {
    let bytes = s.as_bytes();
    let start = s.find('{')?;
    let mut depth = 0i32;
    let mut in_string = false;
    let mut escape = false;
    for (i, &b) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escape {
                escape = false;
            } else if b == b'\\' {
                escape = true;
            } else if b == b'"' {
                in_string = false;
            }
            continue;
        }
        match b {
            b'"' => in_string = true,
            b'{' => depth += 1,
            b'}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(&s[start..=i]);
                }
            }
            _ => {}
        }
    }
    None
}

/// Convert kebab-case identifiers to `snake_case` in place across every
/// id-bearing field of [`LlmCompileResponse`].
///
/// We only flip `-` to `_`; anything else is left alone (`camelCase`,
/// dotted ids like `auth.jwt`, etc.) so the change is conservative.
/// The reference fields (`depends_on*`, `chosen_option_id`,
/// `question_id`) are normalised the same way so an answer to a
/// kebab-case question still resolves correctly when the user (or the
/// executor) talks in `snake_case`.
fn normalize_response_ids(resp: &mut LlmCompileResponse) {
    for q in &mut resp.open_questions {
        normalize_id_in_place(&mut q.id);
        for opt in &mut q.options {
            normalize_id_in_place(&mut opt.id);
        }
        for dep in &mut q.depends_on_decisions {
            normalize_id_in_place(dep);
        }
    }
    for d in &mut resp.decisions {
        normalize_id_in_place(&mut d.question_id);
        normalize_id_in_place(&mut d.chosen_option_id);
        for dep in &mut d.depends_on_decisions {
            normalize_id_in_place(dep);
        }
    }
    if let Some(def) = resp.definition.as_mut() {
        for s in &mut def.steps {
            normalize_id_in_place(&mut s.id);
            for dep in &mut s.depends_on {
                normalize_id_in_place(dep);
            }
        }
    }
}

/// Replace every `-` in `s` with `_`. No-op when `s` contains no
/// hyphens, so callers can apply it unconditionally without paying for
/// an allocation in the common case.
fn normalize_id_in_place(s: &mut String) {
    if s.contains('-') {
        *s = s.replace('-', "_");
    }
}

/// Build a corrective prompt that re-shows the schema, surfaces the
/// previous response (truncated to keep the prompt bounded), and tells
/// the model exactly what was wrong.
fn corrective_prompt(original: &str, prior_raw: &str, parse_err: &str) -> String {
    format!(
        "{original}\n\n\
         ---\n\
         Your previous response could not be parsed:\n\
         > {parse_err}\n\n\
         The first 512 characters of that response were:\n```\n{preview}\n```\n\n\
         Reply again with a SINGLE JSON object matching the schema above. \
         No prose, no markdown fences, no commentary outside the object.",
        preview = preview(prior_raw, 512),
    )
}

/// Build a fallback [`CompileOutput`] when even the corrective retry
/// fails to parse. Produces an open question keyed on
/// [`LLM_STRUCTURE_QUESTION_ID`] with the diagnostics inlined so the
/// caller (and any audit log reader) can see what went wrong.
fn structure_required_fallback(
    source_doc: &str,
    first_err: &str,
    retry_err: &str,
) -> CompileOutput {
    CompileOutput::pending(
        source_doc,
        vec![OpenQuestion {
            id: LLM_STRUCTURE_QUESTION_ID.to_string(),
            prompt: format!(
                "The LLM compiler could not produce a parseable JSON plan after one corrective retry.\n\
                 First-pass diagnostic: {first_err}\n\
                 Retry diagnostic:      {retry_err}\n\n\
                 Restructure your prose (for example, list each step as `## <id> [agent: <type>]` \
                 followed by the prompt body) and resubmit via plan_refine, or paste a structured \
                 version through Answer::ExplainMore."
            ),
            rationale: "LLM responses must conform to flowd's plan-compiler JSON schema; \
                        when they don't, the daemon would otherwise loop on bad output."
                .to_string(),
            options: Vec::new(),
            allow_explain_more: true,
            allow_none: false,
            depends_on_decisions: Vec::new(),
        }],
    )
}

/// UTF-8-safe truncation used for embedding raw model output into
/// prompts and tracing messages.
fn preview(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map_or(s.len(), |(idx, _)| idx);
    format!("{}…", &s[..end])
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
    // LlmPlanCompiler -- pure unit tests with a hand-rolled mock
    // callback. End-to-end live-server tests live in tests/llm_e2e.rs.
    // -----------------------------------------------------------------

    use std::sync::Mutex;

    /// Hand-rolled mock callback that pops scripted responses in FIFO
    /// order and records every prompt it received. Lets us assert both
    /// "what did the compiler ask" and "how does it react to N
    /// responses" without spinning up an HTTP server.
    struct MockCallback {
        script: Mutex<std::collections::VecDeque<Result<String>>>,
        recorded: Mutex<Vec<String>>,
    }

    impl MockCallback {
        fn new(script: Vec<Result<String>>) -> Self {
            Self {
                script: Mutex::new(script.into()),
                recorded: Mutex::new(Vec::new()),
            }
        }

        fn calls(&self) -> Vec<String> {
            self.recorded.lock().unwrap().clone()
        }
    }

    impl LlmCallback for MockCallback {
        async fn complete(&self, prompt: String) -> Result<String> {
            self.recorded.lock().unwrap().push(prompt);
            self.script.lock().unwrap().pop_front().unwrap_or_else(|| {
                Err(FlowdError::Internal("MockCallback script exhausted".into()))
            })
        }
    }

    fn llm_with(script: Vec<Result<String>>) -> (LlmPlanCompiler<MockCallback>, Arc<MockCallback>) {
        let cb = Arc::new(MockCallback::new(script));
        (LlmPlanCompiler::new(cb.clone()), cb)
    }

    #[tokio::test]
    async fn llm_compile_prose_parses_definition_response_into_compile_output() {
        let body = r#"{
            "plan_name": "refactor-auth",
            "open_questions": [],
            "decisions": [],
            "definition": {
                "name": "refactor-auth",
                "steps": [
                    {"id": "extract", "agent_type": "rust-engineer", "prompt": "do extract", "depends_on": []},
                    {"id": "migrate", "agent_type": "rust-engineer", "prompt": "do migrate", "depends_on": ["extract"]}
                ]
            }
        }"#;
        let (c, cb) = llm_with(vec![Ok(body.into())]);
        let out = c
            .compile_prose("freeform prose".into(), "proj".into())
            .await
            .unwrap();
        let def = out.definition.expect("definition parsed");
        assert_eq!(def.name, "refactor-auth");
        assert_eq!(def.steps.len(), 2);
        assert_eq!(def.steps[1].depends_on, vec!["extract".to_string()]);
        assert_eq!(out.source_doc, "freeform prose");
        assert_eq!(cb.calls().len(), 1, "no retry needed for happy path");
    }

    #[tokio::test]
    async fn llm_compile_prose_handles_open_questions_response() {
        let body = r#"{
            "plan_name": null,
            "open_questions": [{
                "id": "lib_choice",
                "prompt": "Which JWT library?",
                "rationale": "Affects perf",
                "options": [
                    {"id": "jsonwebtoken", "label": "jsonwebtoken", "rationale": "popular"},
                    {"id": "biscuit", "label": "biscuit", "rationale": "richer"}
                ],
                "allow_explain_more": true
            }],
            "decisions": [],
            "definition": null
        }"#;
        let (c, _cb) = llm_with(vec![Ok(body.into())]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_none());
        assert_eq!(out.open_questions.len(), 1);
        assert_eq!(out.open_questions[0].id, "lib_choice");
        assert!(out.open_questions[0].allow_explain_more);
        assert_eq!(out.open_questions[0].options.len(), 2);
    }

    #[tokio::test]
    async fn llm_strips_markdown_fences_around_json() {
        let body = "```json\n{\"plan_name\":\"p\",\"open_questions\":[],\"decisions\":[],\"definition\":{\"steps\":[{\"id\":\"a\",\"agent_type\":\"echo\",\"prompt\":\"hi\"}]}}\n```";
        let (c, cb) = llm_with(vec![Ok(body.into())]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_some());
        assert_eq!(cb.calls().len(), 1);
    }

    #[tokio::test]
    async fn llm_extracts_json_object_from_chatty_prose() {
        let body = "Sure! Here's the plan:\n\n{\"plan_name\":\"p\",\"open_questions\":[],\"decisions\":[],\"definition\":{\"steps\":[{\"id\":\"a\",\"agent_type\":\"echo\",\"prompt\":\"hi\"}]}}\n\nLet me know if you need changes.";
        let (c, cb) = llm_with(vec![Ok(body.into())]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_some());
        assert_eq!(cb.calls().len(), 1, "no retry needed when json extractable");
    }

    #[tokio::test]
    async fn llm_retries_once_on_unparseable_response_and_succeeds() {
        let bad = "totally not json".to_string();
        let good = r#"{"plan_name":"p","open_questions":[],"decisions":[],"definition":{"steps":[{"id":"a","agent_type":"echo","prompt":"hi"}]}}"#.to_string();
        let (c, cb) = llm_with(vec![Ok(bad), Ok(good)]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_some());
        let calls = cb.calls();
        assert_eq!(calls.len(), 2, "should retry once");
        assert!(
            calls[1].contains("could not be parsed"),
            "corrective prompt embedded: {}",
            preview(&calls[1], 200)
        );
    }

    #[tokio::test]
    async fn llm_falls_back_to_structure_question_when_retry_also_fails() {
        let (c, cb) = llm_with(vec![Ok("garbage 1".into()), Ok("garbage 2".into())]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_none());
        assert_eq!(out.open_questions.len(), 1);
        assert_eq!(out.open_questions[0].id, LLM_STRUCTURE_QUESTION_ID);
        assert!(out.open_questions[0].allow_explain_more);
        assert!(
            out.open_questions[0]
                .prompt
                .contains("could not produce a parseable JSON plan")
        );
        assert_eq!(cb.calls().len(), 2);
    }

    #[tokio::test]
    async fn llm_propagates_transport_errors_without_retry() {
        let (c, cb) = llm_with(vec![Err(FlowdError::Internal("MLX down".into()))]);
        let err = c
            .compile_prose("p".into(), "proj".into())
            .await
            .unwrap_err();
        match err {
            FlowdError::Internal(m) => assert!(m.contains("MLX down"), "{m}"),
            other => panic!("expected Internal, got {other:?}"),
        }
        assert_eq!(cb.calls().len(), 1, "transport errors must not retry");
    }

    #[tokio::test]
    async fn llm_drops_definition_when_questions_present() {
        // Defends the PlanCompiler invariant: definition.is_some() iff
        // open_questions.is_empty(). If the model emits both, the
        // compiler keeps the questions and discards the definition.
        let body = r#"{
            "plan_name": "p",
            "open_questions": [{"id":"q","prompt":"?","rationale":"because"}],
            "decisions": [],
            "definition": {"steps":[{"id":"a","agent_type":"echo","prompt":"hi"}]}
        }"#;
        // This response will trip the `set definition to null` parse
        // diagnostic on the first pass; the second response keeps only
        // the question.
        let retry = r#"{
            "plan_name": "p",
            "open_questions": [{"id":"q","prompt":"?","rationale":"because"}],
            "decisions": [],
            "definition": null
        }"#;
        let (c, _) = llm_with(vec![Ok(body.into()), Ok(retry.into())]);
        let out = c.compile_prose("p".into(), "proj".into()).await.unwrap();
        assert!(out.definition.is_none());
        assert_eq!(out.open_questions.len(), 1);
    }

    #[tokio::test]
    async fn llm_apply_answers_includes_answers_block_in_prompt() {
        let body = r#"{"plan_name":"p","open_questions":[],"decisions":[],"definition":{"steps":[{"id":"a","agent_type":"echo","prompt":"hi"}]}}"#;
        let (c, cb) = llm_with(vec![Ok(body.into())]);
        let snap = snapshot(Some("# my prose".into()));
        let answers = vec![
            (
                "lib".into(),
                Answer::Choose {
                    option_id: "jsonwebtoken".into(),
                },
            ),
            ("scope".into(), Answer::NoneOfThese),
        ];
        let _ = c
            .apply_answers(snap, answers, /* defer */ true)
            .await
            .unwrap();
        let prompt = &cb.calls()[0];
        assert!(prompt.contains("lib: choose `jsonwebtoken`"), "{prompt}");
        assert!(prompt.contains("scope: none_of_these"), "{prompt}");
        assert!(prompt.contains("FILL IN"), "defer clause present");
    }

    #[tokio::test]
    async fn llm_refine_includes_feedback_block_in_prompt() {
        let body = r#"{"plan_name":"p","open_questions":[],"decisions":[],"definition":{"steps":[{"id":"a","agent_type":"echo","prompt":"hi"}]}}"#;
        let (c, cb) = llm_with(vec![Ok(body.into())]);
        let snap = snapshot(Some("# original".into()));
        let _ = c
            .refine(snap, "drop the smoke test step".into())
            .await
            .unwrap();
        let prompt = &cb.calls()[0];
        assert!(prompt.contains("drop the smoke test step"));
        assert!(prompt.contains("Refinement instruction"));
    }

    #[tokio::test]
    async fn llm_apply_answers_carries_decisions_through_to_compile_output() {
        let body = r#"{
            "plan_name":"p",
            "open_questions":[],
            "decisions":[{"question_id":"lib","chosen_option_id":"jsonwebtoken","auto":true}],
            "definition":{"steps":[{"id":"a","agent_type":"echo","prompt":"hi"}]}
        }"#;
        let (c, _) = llm_with(vec![Ok(body.into())]);
        let snap = snapshot(Some("# original".into()));
        let out = c.apply_answers(snap, vec![], true).await.unwrap();
        assert_eq!(out.new_decisions.len(), 1);
        assert_eq!(out.new_decisions[0].question_id, "lib");
        assert!(out.new_decisions[0].auto);
    }

    // -----------------------------------------------------------------
    // Pure-function tests for the parser / prompt helpers
    // -----------------------------------------------------------------

    #[test]
    fn parse_rejects_empty_response() {
        let err = parse_llm_response("").unwrap_err();
        assert!(err.contains("empty"));
    }

    #[test]
    fn parse_rejects_response_with_neither_questions_nor_definition() {
        let err = parse_llm_response(r#"{"open_questions":[],"decisions":[],"definition":null}"#)
            .unwrap_err();
        assert!(err.contains("either"));
    }

    #[test]
    fn parse_rejects_definition_with_empty_steps() {
        let err =
            parse_llm_response(r#"{"open_questions":[],"decisions":[],"definition":{"steps":[]}}"#)
                .unwrap_err();
        assert!(err.contains("non-empty"));
    }

    #[test]
    fn parse_rejects_duplicate_step_ids() {
        let err = parse_llm_response(
            r#"{"open_questions":[],"decisions":[],"definition":{"steps":[{"id":"a","agent_type":"e","prompt":"p"},{"id":"a","agent_type":"e","prompt":"p"}]}}"#,
        )
        .unwrap_err();
        assert!(err.contains("duplicate"));
    }

    #[test]
    fn parse_normalizes_kebab_case_ids_to_snake_case_across_every_field() {
        // Mixed kebab + snake input the model might produce. After
        // parse, every id should have been canonicalised so neither the
        // executor nor downstream callers have to tolerate the mix.
        let body = r#"{
            "plan_name": "p",
            "open_questions": [{
                "id": "lib-choice",
                "prompt": "?",
                "rationale": "r",
                "options": [{"id": "json-webtoken", "label": "x", "rationale": "y"}],
                "depends_on_decisions": ["scope-decision"]
            }],
            "decisions": [{
                "question_id": "scope-decision",
                "chosen_option_id": "narrow-scope",
                "depends_on_decisions": ["upstream-dep"]
            }],
            "definition": null
        }"#;
        let parsed = parse_llm_response(body).unwrap();
        let q = &parsed.open_questions[0];
        assert_eq!(q.id, "lib_choice");
        assert_eq!(q.options[0].id, "json_webtoken");
        assert_eq!(q.depends_on_decisions[0], "scope_decision");
        let d = &parsed.decisions[0];
        assert_eq!(d.question_id, "scope_decision");
        assert_eq!(d.chosen_option_id, "narrow_scope");
        assert_eq!(d.depends_on_decisions[0], "upstream_dep");
    }

    #[test]
    fn parse_normalizes_step_ids_and_dependencies() {
        let body = r#"{
            "plan_name": "p",
            "open_questions": [],
            "decisions": [],
            "definition": {
                "steps": [
                    {"id": "extract-jwt", "agent_type": "rust-engineer", "prompt": "do x", "depends_on": []},
                    {"id": "migrate-callers", "agent_type": "rust-engineer", "prompt": "do y", "depends_on": ["extract-jwt"]}
                ]
            }
        }"#;
        let parsed = parse_llm_response(body).unwrap();
        let def = parsed.definition.unwrap();
        assert_eq!(def.steps[0].id, "extract_jwt");
        assert_eq!(def.steps[1].id, "migrate_callers");
        assert_eq!(def.steps[1].depends_on, vec!["extract_jwt".to_string()]);
        // Note: `agent_type` is intentionally NOT normalised; agent types
        // are caller-defined identifiers (e.g. "rust-engineer") and the
        // executor matches them verbatim against its spawner registry.
        assert_eq!(def.steps[0].agent_type, "rust-engineer");
    }

    #[test]
    fn normalize_id_in_place_is_noop_for_already_snake_case() {
        let mut s = String::from("already_snake_case");
        normalize_id_in_place(&mut s);
        assert_eq!(s, "already_snake_case");
    }

    #[test]
    fn normalize_id_in_place_only_touches_hyphens() {
        let mut s = String::from("my-step.with.dots");
        normalize_id_in_place(&mut s);
        assert_eq!(s, "my_step.with.dots", "dots and other chars preserved");
    }

    #[test]
    fn locate_json_object_handles_braces_inside_string_literals() {
        let input = r#"prefix {"k":"v with } inside"} suffix"#;
        let extracted = locate_json_object(input).unwrap();
        assert_eq!(extracted, r#"{"k":"v with } inside"}"#);
    }

    #[test]
    fn strip_code_fences_removes_json_tagged_block() {
        let input = "```json\n{\"a\":1}\n```";
        assert_eq!(strip_code_fences(input), "{\"a\":1}");
    }

    #[test]
    fn strip_code_fences_is_noop_on_plain_input() {
        assert_eq!(strip_code_fences("{\"a\":1}"), "{\"a\":1}");
    }

    #[test]
    fn corrective_prompt_includes_diagnostic_and_truncated_preview() {
        let original = "ORIGINAL PROMPT";
        let raw = "x".repeat(2000);
        let p = corrective_prompt(original, &raw, "missing field foo");
        assert!(p.contains("ORIGINAL PROMPT"));
        assert!(p.contains("missing field foo"));
        // Truncated to 512 chars + ellipsis
        assert!(p.contains("…"));
    }

    #[test]
    fn fallback_question_records_both_diagnostics() {
        let out = structure_required_fallback("# prose", "first bad", "retry bad");
        assert_eq!(out.open_questions.len(), 1);
        let q = &out.open_questions[0];
        assert_eq!(q.id, LLM_STRUCTURE_QUESTION_ID);
        assert!(q.prompt.contains("first bad"));
        assert!(q.prompt.contains("retry bad"));
        assert_eq!(out.source_doc, "# prose");
    }

    #[test]
    fn preview_respects_utf8_boundaries() {
        let s = "αβγδε";
        assert_eq!(preview(s, 3), "αβγ…");
        assert_eq!(preview(s, 10), s);
    }

    #[test]
    fn render_helpers_handle_empty_inputs() {
        assert!(render_answers_block(&[]).contains("none"));
        assert_eq!(render_open_questions_block(&[]), "(none)");
        assert_eq!(render_decisions_block(&[]), "(none)");
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
