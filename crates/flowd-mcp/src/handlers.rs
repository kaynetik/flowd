//! MCP tool handlers.
//!
//! [`McpHandlers`] is the application-side interface the server dispatches to.
//! Concrete implementation [`FlowdHandlers`] wires the trait to the four
//! flowd-core services -- `MemoryService`, a `PlanExecutor`, a
//! `PlanCompiler`, and a `RuleEvaluator`. Tests can supply any other
//! implementation (see the integration test for a stub version).
//!
//! All handlers are `async` and return `serde_json::Value` payloads. The
//! server wraps these values into MCP tool-result envelopes; transport
//! details never leak into the handler layer.
//!
//! ## Prose-first plan creation
//!
//! `plan_create` is polymorphic: callers supply either `definition`
//! (DAG-first, legacy path) or `prose` (prose-first, routed through the
//! configured [`PlanCompiler`]). The two follow-on tools `plan_answer` and
//! `plan_refine` exist solely for the prose-first loop:
//!
//! 1. `plan_create` with `prose` => Draft plan with possibly-open questions.
//! 2. Repeat `plan_answer` (and optionally `plan_refine`) until the plan
//!    has no more open questions and a compiled DAG.
//! 3. `plan_confirm` to start execution.
//!
//! The handlers emit
//! [`flowd_core::orchestration::observer::PlanEvent::ClarificationOpened`],
//! [`PlanEvent::ClarificationResolved`][co], and
//! [`PlanEvent::RefinementApplied`][ra] at every transition so the
//! `plan_events` audit log captures the full clarification timeline.
//!
//! [co]: flowd_core::orchestration::observer::PlanEvent::ClarificationResolved
//! [ra]: flowd_core::orchestration::observer::PlanEvent::RefinementApplied

use std::future::Future;
use std::str::FromStr;
use std::sync::Arc;

use serde_json::{Value, json};
use uuid::Uuid;

use flowd_core::error::{FlowdError, Result};
use flowd_core::memory::compactor::ActivityMonitor;
use flowd_core::memory::context::AutoContextQuery;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, MemoryBackend, VectorIndex};
use flowd_core::orchestration::observer::{PlanEvent, SharedPlanObserver};
use flowd_core::orchestration::{
    Answer, Plan, PlanCompiler, PlanDraftSnapshot, PlanExecutor, PlanStatus, PlanSummary,
    loader::PlanDefinition, resolve_workspace_root,
};
use flowd_core::rules::{ProposedAction, RuleEvaluator};
use flowd_core::types::SearchQuery;

use crate::tools::{
    MemoryContextParams, MemorySearchParams, MemoryStoreParams, PlanAnswerParams, PlanCancelParams,
    PlanConfirmParams, PlanCreateParams, PlanListParams, PlanRecentParams, PlanRefineParams,
    PlanResumeParams, PlanShowParams, PlanStatusParams, RulesCheckParams, RulesListParams,
};

/// Maximum number of characters preserved for a derived plan name.
///
/// Plans authored from prose default to the first non-empty heading,
/// truncated to keep CLI output and event payloads tidy. The compiler
/// will overwrite this with whatever name it emits as soon as a DAG is
/// compiled, so the value only matters for the in-flight Draft window.
const MAX_DERIVED_PLAN_NAME_LEN: usize = 120;

/// Maximum length of the `feedback_summary` field on a `RefinementApplied`
/// event. Keeps the audit log from ballooning when callers paste long
/// instructions; the original feedback is still passed verbatim to the
/// compiler.
const MAX_REFINEMENT_SUMMARY_LEN: usize = 200;

/// Stable warning code surfaced in the JSON response when the
/// clarification budget is exceeded. Callers (CLI, agents) match on
/// this string to decide whether to render a "this plan needs to
/// converge" hint to the user.
pub const WARN_BUDGET_EXCEEDED: &str = "BudgetExceeded";

/// Stable warning code surfaced when the handler had to override the
/// caller's `defer_remaining = false` because the budget was already
/// exhausted at call time. The compiler is then asked to converge on
/// best-effort answers instead of opening yet another round.
pub const WARN_DEFER_COERCED: &str = "DeferRemainingCoerced";

/// Surface exposed by the MCP server. Return values are arbitrary JSON;
/// errors bubble up via `Result` and are mapped to `isError: true` by the
/// dispatcher in [`crate::server`].
pub trait McpHandlers: Send + Sync {
    fn memory_store(&self, params: MemoryStoreParams)
    -> impl Future<Output = Result<Value>> + Send;

    fn memory_search(
        &self,
        params: MemorySearchParams,
    ) -> impl Future<Output = Result<Value>> + Send;

    fn memory_context(
        &self,
        params: MemoryContextParams,
    ) -> impl Future<Output = Result<Value>> + Send;

    fn plan_create(&self, params: PlanCreateParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_answer(&self, params: PlanAnswerParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_refine(&self, params: PlanRefineParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_confirm(&self, params: PlanConfirmParams)
    -> impl Future<Output = Result<Value>> + Send;

    fn plan_cancel(&self, params: PlanCancelParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_status(&self, params: PlanStatusParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_resume(&self, params: PlanResumeParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_list(&self, params: PlanListParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_show(&self, params: PlanShowParams) -> impl Future<Output = Result<Value>> + Send;

    fn plan_recent(&self, params: PlanRecentParams) -> impl Future<Output = Result<Value>> + Send;

    fn rules_check(&self, params: RulesCheckParams) -> impl Future<Output = Result<Value>> + Send;

    fn rules_list(&self, params: RulesListParams) -> impl Future<Output = Result<Value>> + Send;
}

/// Concrete handlers that compose the four flowd-core services.
///
/// Generic over the backend types so the concrete `SQLite` / `Qdrant` / ONNX
/// implementations (or test doubles) are chosen at composition time --
/// `flowd-mcp` itself never pulls in those heavy dependencies. The
/// `PC: PlanCompiler` generic was added in HL-44 alongside the prose-first
/// plan creation flow; tests that don't exercise prose can pass the
/// no-deps [`flowd_core::orchestration::MockPlanCompiler`] with an empty
/// script.
pub struct FlowdHandlers<M, V, E, PE, PC, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    PC: PlanCompiler + 'static,
    R: RuleEvaluator + 'static,
{
    memory: Arc<MemoryService<M, V, E>>,
    executor: Arc<PE>,
    compiler: Arc<PC>,
    rules: Arc<R>,
    /// Plan-event sink for clarification / refinement transitions.
    ///
    /// Optional so test composers (and the existing daemon during the
    /// transition window) can omit it; when `None` the prose-first
    /// handlers simply skip event emission. The hot-path executor
    /// observer is wired separately via [`PlanExecutor::with_observer`]
    /// so callers can either share one observer between both surfaces
    /// (typical) or split them.
    observer: Option<SharedPlanObserver>,
    /// Idle-detector consulted by the background compactor.
    ///
    /// Touched on every incoming MCP request so the compactor's "is the
    /// daemon idle?" check sees real activity rather than wall-clock
    /// elapsed time. Optional so test/stub composers don't have to wire
    /// a monitor when they don't run a compactor.
    monitor: Option<ActivityMonitor>,
    /// Maximum allowed `decisions.len() + open_questions.len()` before
    /// the handlers start coercing `defer_remaining = true` and
    /// surfacing `BudgetExceeded` warnings in the response payload.
    ///
    /// `None` disables enforcement entirely (the pre-budget
    /// behaviour). The daemon binds this to the `[plan].max_questions`
    /// value from `flowd.toml`; in tests it is typically left unset so
    /// the budget machinery stays inert and existing assertions don't
    /// have to account for the warnings field.
    question_budget: Option<usize>,
}

impl<M, V, E, PE, PC, R> FlowdHandlers<M, V, E, PE, PC, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    PC: PlanCompiler + 'static,
    R: RuleEvaluator + 'static,
{
    pub fn new(
        memory: Arc<MemoryService<M, V, E>>,
        executor: Arc<PE>,
        compiler: Arc<PC>,
        rules: Arc<R>,
    ) -> Self {
        Self {
            memory,
            executor,
            compiler,
            rules,
            observer: None,
            monitor: None,
            question_budget: None,
        }
    }

    /// Attach an `ActivityMonitor` so each handler call resets the idle
    /// timer the background compactor watches. Returns `self` for chaining.
    #[must_use]
    pub fn with_activity_monitor(mut self, monitor: ActivityMonitor) -> Self {
        self.monitor = Some(monitor);
        self
    }

    /// Attach a [`SharedPlanObserver`] for clarification / refinement
    /// events. The same observer is typically also installed on the
    /// executor so all plan-event variants land in one store. Returns
    /// `self` for chaining.
    #[must_use]
    pub fn with_observer(mut self, observer: SharedPlanObserver) -> Self {
        self.observer = Some(observer);
        self
    }

    /// Cap the total clarification load (`decisions + open_questions`)
    /// the prose-first handlers will tolerate before they start
    /// coercing `defer_remaining = true` on `plan_answer` and surfacing
    /// `BudgetExceeded` warnings on every prose-first response.
    ///
    /// Set to `None` to disable enforcement (the default). The daemon
    /// passes `Some(flowd.toml plan.max_questions)`; tests typically
    /// leave it unset so existing fixtures stay focused on the loop.
    #[must_use]
    pub fn with_question_budget(mut self, budget: Option<usize>) -> Self {
        self.question_budget = budget;
        self
    }

    #[must_use]
    pub fn memory(&self) -> &Arc<MemoryService<M, V, E>> {
        &self.memory
    }

    #[must_use]
    pub fn executor(&self) -> &Arc<PE> {
        &self.executor
    }

    #[must_use]
    pub fn compiler(&self) -> &Arc<PC> {
        &self.compiler
    }

    #[must_use]
    pub fn rules(&self) -> &Arc<R> {
        &self.rules
    }

    /// Mark "we just did something". Cheap; safe to call on every handler.
    fn touch_activity(&self) {
        if let Some(m) = &self.monitor {
            m.touch();
        }
    }

    fn emit(&self, event: PlanEvent) {
        if let Some(obs) = &self.observer {
            obs.on_event(event);
        }
    }

    /// Emit `ClarificationOpened` and/or `ClarificationResolved` for any
    /// non-empty deltas surfaced in `plan` after a compile call. Centralised
    /// so `plan_create` / `plan_answer` / `plan_refine` all stay symmetric.
    fn emit_clarification_deltas(
        &self,
        plan: &Plan,
        new_question_ids: Vec<String>,
        new_decision_ids: Vec<String>,
    ) {
        if !new_question_ids.is_empty() {
            self.emit(PlanEvent::ClarificationOpened {
                plan_id: plan.id,
                project: plan.project.clone(),
                question_ids: new_question_ids,
            });
        }
        if !new_decision_ids.is_empty() {
            self.emit(PlanEvent::ClarificationResolved {
                plan_id: plan.id,
                project: plan.project.clone(),
                decision_ids: new_decision_ids,
            });
        }
    }
}

fn parse_uuid(raw: &str) -> Result<Uuid> {
    Uuid::from_str(raw).map_err(|e| FlowdError::Internal(format!("invalid uuid `{raw}`: {e}")))
}

/// Pull a short, human-readable plan name out of free-form prose.
///
/// Looks at the first non-empty line, strips up to six leading `#`
/// markers (treating any-level Markdown heading the same), and truncates
/// to [`MAX_DERIVED_PLAN_NAME_LEN`] characters. Falls back to
/// `"untitled draft"` so [`Plan::new`] always sees a non-empty string.
///
/// The compiler is free to overwrite this on the next round when it emits
/// a fully resolved [`PlanDefinition`]; the derived value only matters
/// for the in-flight Draft window.
fn derive_plan_name(prose: &str) -> String {
    let line = prose
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .unwrap_or("");
    let trimmed = line.trim_start_matches('#').trim();
    let candidate = if trimmed.is_empty() {
        "untitled draft"
    } else {
        trimmed
    };
    candidate.chars().take(MAX_DERIVED_PLAN_NAME_LEN).collect()
}

/// Truncate a refinement instruction to fit in the audit log.
fn truncate_summary(feedback: &str) -> String {
    feedback.chars().take(MAX_REFINEMENT_SUMMARY_LEN).collect()
}

/// One structured warning emitted alongside the prose-plan payload.
///
/// Stable shape so clients can match on `code` (see [`WARN_BUDGET_EXCEEDED`]
/// / [`WARN_DEFER_COERCED`]) and surface a localised message of their
/// own without parsing the human-readable `message` field.
#[derive(Debug, Clone, serde::Serialize)]
struct ProsePlanWarning {
    code: &'static str,
    message: String,
    /// Configured budget at the time the warning was emitted. Always
    /// present today (the only warnings we emit are budget-driven), but
    /// `Option` so future non-budget warnings can omit it without a
    /// shape break.
    #[serde(skip_serializing_if = "Option::is_none")]
    budget: Option<usize>,
    /// Observed `decisions + open_questions` when the warning fired.
    #[serde(skip_serializing_if = "Option::is_none")]
    current: Option<usize>,
}

/// Build the response payload returned by every prose-first plan handler.
///
/// Symmetric across `plan_create`, `plan_answer`, and `plan_refine` so MCP
/// clients can render the loop without branching on which tool produced
/// the result. `preview` is included only when the plan has compiled
/// (i.e. has steps and no open questions); otherwise it stays absent so
/// callers don't mistake an empty preview for a successful compile.
///
/// Per-tool handlers tack on two optional pieces:
///
/// * `warnings` -- structured budget / coercion notices (omitted when empty
///   so existing test fixtures keep matching).
/// * `clarification_reopened` -- only emitted by `plan_refine`, signals
///   that the refine call introduced (or kept) open questions so the
///   caller has to re-enter the clarification loop. Skipped when `None`.
fn prose_plan_payload_with(
    plan: &Plan,
    warnings: &[ProsePlanWarning],
    clarification_reopened: Option<bool>,
) -> Result<Value> {
    let mut payload = json!({
        "plan_id": plan.id.to_string(),
        "status": plan_status_label(plan.status),
        "name": plan.name,
        "project": plan.project,
        "definition_dirty": plan.definition_dirty,
        "open_questions": plan.open_questions,
        "decisions": plan.decisions,
    });
    let map = payload
        .as_object_mut()
        .expect("plan_payload is a json object");
    if !plan.has_open_questions() && !plan.steps.is_empty() {
        map.insert("preview".into(), build_preview_value(plan)?);
    }
    if !warnings.is_empty() {
        map.insert(
            "warnings".into(),
            serde_json::to_value(warnings).map_err(|e| FlowdError::Serialization(e.to_string()))?,
        );
    }
    if let Some(flag) = clarification_reopened {
        map.insert("clarification_reopened".into(), Value::Bool(flag));
    }
    Ok(payload)
}

fn filter_plan_summaries(
    mut summaries: Vec<PlanSummary>,
    status: Option<PlanStatus>,
    limit: Option<usize>,
) -> Vec<PlanSummary> {
    if let Some(status) = status {
        summaries.retain(|s| s.status == status);
    }
    if let Some(limit) = limit {
        summaries.truncate(limit.max(1));
    }
    summaries
}

fn plan_summaries_payload(summaries: &[PlanSummary]) -> Value {
    json!({
        "plans": summaries,
    })
}

fn parse_plan_status_filter(raw: Option<String>) -> Result<Option<PlanStatus>> {
    raw.map(|s| parse_plan_status(&s)).transpose()
}

fn parse_plan_status(raw: &str) -> Result<PlanStatus> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "draft" => Ok(PlanStatus::Draft),
        "confirmed" => Ok(PlanStatus::Confirmed),
        "running" => Ok(PlanStatus::Running),
        "interrupted" => Ok(PlanStatus::Interrupted),
        "completed" => Ok(PlanStatus::Completed),
        "failed" => Ok(PlanStatus::Failed),
        "cancelled" => Ok(PlanStatus::Cancelled),
        other => Err(FlowdError::PlanValidation(format!(
            "unknown plan status `{other}`; expected draft, confirmed, running, interrupted, completed, failed, or cancelled"
        ))),
    }
}

/// Sum of outstanding questions and recorded decisions on a plan.
///
/// This is the metric the question budget caps. Decisions count too
/// because every `Choose` answer that resolves a question still
/// represents a question we asked the user; without that, a chatty
/// compiler that resolves+reopens questions could circumvent the budget.
fn clarification_load(plan: &Plan) -> usize {
    plan.open_questions.len() + plan.decisions.len()
}

/// Build the `BudgetExceeded` warning payload. Centralised so the
/// message phrasing stays consistent between the pre- and post-compile
/// emission sites.
fn budget_exceeded_warning(budget: usize, current: usize) -> ProsePlanWarning {
    ProsePlanWarning {
        code: WARN_BUDGET_EXCEEDED,
        message: format!(
            "clarification budget exceeded: {current} questions+decisions outstanding (limit {budget}); \
             the next plan_answer call will coerce defer_remaining=true so the compiler converges"
        ),
        budget: Some(budget),
        current: Some(current),
    }
}

/// Build the `DeferRemainingCoerced` warning payload, used when the
/// pre-flight budget check overrode `defer_remaining=false`.
fn defer_coerced_warning(budget: usize, current: usize) -> ProsePlanWarning {
    ProsePlanWarning {
        code: WARN_DEFER_COERCED,
        message: format!(
            "defer_remaining was coerced to true: {current} questions+decisions exceed limit {budget}; \
             the compiler is now expected to fill remaining questions on a best-effort basis"
        ),
        budget: Some(budget),
        current: Some(current),
    }
}

/// Compute a preview for a plan that already passes structural validation.
/// Returns the JSON form so handlers don't have to repeat the
/// `serde_json::to_value` boilerplate.
fn build_preview_value(plan: &Plan) -> Result<Value> {
    let preview = flowd_core::orchestration::build_preview(plan)?;
    serde_json::to_value(&preview).map_err(|e| FlowdError::Serialization(e.to_string()))
}

impl<M, V, E, PE, PC, R> McpHandlers for FlowdHandlers<M, V, E, PE, PC, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    PC: PlanCompiler + 'static,
    R: RuleEvaluator + 'static,
{
    async fn memory_store(&self, p: MemoryStoreParams) -> Result<Value> {
        self.touch_activity();
        let session = parse_uuid(&p.session_id)?;
        // MCP clients hold the session UUID (Claude/Cursor assign it at
        // session start) but do not issue a separate start_session call.
        // Ensure the row exists before inserting an observation so the FK
        // on observations.session_id never trips.
        self.memory.ensure_session(&p.project, session).await?;
        let id = self
            .memory
            .record(&p.project, session, p.content, p.metadata)
            .await?;
        Ok(json!({ "id": id.to_string() }))
    }

    async fn memory_search(&self, p: MemorySearchParams) -> Result<Value> {
        self.touch_activity();
        let query = SearchQuery {
            text: p.query,
            project: p.project,
            since: None,
            limit: p.limit.unwrap_or(10),
        };
        let results = self.memory.search(&query).await?;
        serde_json::to_value(&results).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn memory_context(&self, p: MemoryContextParams) -> Result<Value> {
        self.touch_activity();
        let mut ctx = AutoContextQuery::new(&p.project);
        if let Some(f) = p.file_path.as_deref() {
            ctx = ctx.with_file(f);
        }
        if let Some(s) = p.session_id.as_deref() {
            ctx = ctx.with_session(parse_uuid(s)?);
        }
        if let Some(h) = p.hint {
            ctx = ctx.with_hint(h);
        }
        if let Some(lim) = p.limit {
            ctx = ctx.with_limit(lim);
        }

        let observations = self.memory.auto_context(&ctx).await?;
        let rules = self
            .rules
            .matching_rules(Some(&p.project), p.file_path.as_deref());

        // Rules are cloned rather than borrowed-serialised because the
        // caller may outlive the lock held by `matching_rules`.
        let rules: Vec<_> = rules.into_iter().cloned().collect();

        Ok(json!({
            "observations": observations,
            "rules": rules,
        }))
    }

    async fn plan_create(&self, p: PlanCreateParams) -> Result<Value> {
        self.touch_activity();
        if p.project.trim().is_empty() {
            return Err(FlowdError::PlanValidation(
                "plan_create: `project` must be a non-empty string".into(),
            ));
        }

        // Mutual exclusion is enforced at both the JSON-Schema layer (so
        // MCP clients see the constraint) and here (so direct API users
        // get a clear validation error rather than a confusing default).
        match (p.definition, p.prose) {
            (Some(_), Some(_)) => Err(FlowdError::PlanValidation(
                "plan_create: pass exactly one of `definition` or `prose`, not both".into(),
            )),
            (None, None) => Err(FlowdError::PlanValidation(
                "plan_create: must include either `definition` or `prose`".into(),
            )),
            (Some(definition), None) => {
                if p.compiler_override.is_some() {
                    // Refuse silently-ignoring the override on the DAG-first
                    // path; if the caller bothered to set it they almost
                    // certainly meant to use prose.
                    return Err(FlowdError::PlanValidation(
                        "plan_create: `compiler_override` only applies when `prose` is set; \
                         drop it or switch to the prose-first path"
                            .into(),
                    ));
                }
                self.plan_create_definition(p.project, definition, p.project_root.as_deref())
                    .await
            }
            (None, Some(prose)) => {
                self.plan_create_prose(
                    p.project,
                    prose,
                    p.compiler_override,
                    p.project_root.as_deref(),
                )
                .await
            }
        }
    }

    async fn plan_answer(&self, p: PlanAnswerParams) -> Result<Value> {
        self.touch_activity();
        if p.answers.is_empty() && !p.defer_remaining {
            return Err(FlowdError::PlanValidation(
                "plan_answer: must include at least one answer or set defer_remaining=true".into(),
            ));
        }
        let plan_id = parse_uuid(&p.plan_id)?;

        // Snapshot first so we can detect overwrites and decide what to
        // invalidate before re-invoking the compiler. Pulling the plan
        // also gates on `Draft` status -- the executor's apply path will
        // reject anything else but failing fast here gives a clearer
        // error and avoids speculative compiler work.
        let mut plan = self.executor.status(plan_id).await?;
        if plan.status != PlanStatus::Draft {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "plan_answer: plan `{plan_id}` is not in Draft state (currently {:?})",
                    plan.status
                ),
                metrics: None,
            });
        }

        // Step 1: invalidate any decision the user is overwriting. This
        // also drops downstream decisions whose `depends_on_decisions`
        // chain transitively includes the overwritten id, so the
        // compiler is asked to re-emit the right questions next round.
        let overwritten: Vec<&str> = p
            .answers
            .iter()
            .map(|a| a.question_id.as_str())
            .filter(|qid| plan.decisions.iter().any(|d| d.question_id == *qid))
            .collect();
        for qid in &overwritten {
            self.executor
                .invalidate_decision(plan_id, (*qid).to_owned())
                .await?;
        }
        if !overwritten.is_empty() {
            // Re-fetch so the snapshot we hand to the compiler reflects
            // the invalidations.
            plan = self.executor.status(plan_id).await?;
        }

        // Pre-flight budget check. If the load is already at/over the
        // budget, override `defer_remaining=false`: the compiler should
        // be asked to converge rather than open another round. We accumulate
        // the warning now so it lands in the response even if the compiler
        // hands back a clean payload.
        let mut warnings: Vec<ProsePlanWarning> = Vec::new();
        let mut effective_defer = p.defer_remaining;
        if let Some(budget) = self.question_budget {
            let load = clarification_load(&plan);
            if load >= budget && !effective_defer {
                warnings.push(defer_coerced_warning(budget, load));
                effective_defer = true;
            }
        }

        // Step 2: project to a snapshot and call the compiler. The
        // compiler is free to surface new questions in addition to the
        // ones the user just answered. The Draft plan carries the
        // trusted `project_root` we resolved on `plan_create`, so
        // follow-up rounds dispatch through the same project-scoped
        // config and never drift to the daemon's $FLOWD_HOME defaults.
        let snapshot = PlanDraftSnapshot::from_plan(&plan);
        let answers: Vec<(String, Answer)> = p
            .answers
            .into_iter()
            .map(|a| (a.question_id, a.answer))
            .collect();
        let project_root = plan.project_root.clone();
        let output = self
            .compiler
            .apply_answers_in_project(snapshot, answers, effective_defer, project_root)
            .await?;

        // Capture deltas before applying so we can emit them once the
        // executor has persisted the new state.
        let new_question_ids: Vec<String> =
            output.open_questions.iter().map(|q| q.id.clone()).collect();
        let new_decision_ids: Vec<String> = output
            .new_decisions
            .iter()
            .map(|d| d.question_id.clone())
            .collect();

        self.executor.apply_compile_output(plan_id, output).await?;
        let updated = self.executor.status(plan_id).await?;
        self.emit_clarification_deltas(&updated, new_question_ids, new_decision_ids);
        self.maybe_push_budget_warning(&updated, &mut warnings);
        prose_plan_payload_with(&updated, &warnings, None)
    }

    async fn plan_refine(&self, p: PlanRefineParams) -> Result<Value> {
        self.touch_activity();
        if p.feedback.trim().is_empty() {
            return Err(FlowdError::PlanValidation(
                "plan_refine: `feedback` must be a non-empty string".into(),
            ));
        }
        let plan_id = parse_uuid(&p.plan_id)?;
        let plan = self.executor.status(plan_id).await?;
        if plan.status != PlanStatus::Draft {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "plan_refine: plan `{plan_id}` is not in Draft state (currently {:?})",
                    plan.status
                ),
                metrics: None,
            });
        }

        // Track which questions were already on the plan so we can decide
        // whether the refine introduced *new* clarifications (the
        // `clarification_reopened = true` case) or merely passed through
        // pre-existing ones unchanged. The strict "reopened" semantic
        // matches what callers actually want: "do I need to render a
        // fresh question to the user as a result of this refine call?".
        let prior_question_ids: std::collections::HashSet<String> =
            plan.open_questions.iter().map(|q| q.id.clone()).collect();

        let snapshot = PlanDraftSnapshot::from_plan(&plan);
        let summary = truncate_summary(&p.feedback);
        // Same dispatch-by-project_root contract as plan_answer: the
        // Draft's persisted root drives compiler routing.
        let project_root = plan.project_root.clone();
        let output = self
            .compiler
            .refine_in_project(snapshot, p.feedback, project_root)
            .await?;

        let new_question_ids: Vec<String> =
            output.open_questions.iter().map(|q| q.id.clone()).collect();
        let new_decision_ids: Vec<String> = output
            .new_decisions
            .iter()
            .map(|d| d.question_id.clone())
            .collect();
        let reopened = output
            .open_questions
            .iter()
            .any(|q| !prior_question_ids.contains(&q.id));

        self.executor.apply_compile_output(plan_id, output).await?;
        let updated = self.executor.status(plan_id).await?;

        // Emit the refinement marker first so consumers see the trigger
        // ahead of the deltas it produced.
        self.emit(PlanEvent::RefinementApplied {
            plan_id: updated.id,
            project: updated.project.clone(),
            feedback_summary: summary,
        });
        self.emit_clarification_deltas(&updated, new_question_ids, new_decision_ids);
        let mut warnings: Vec<ProsePlanWarning> = Vec::new();
        self.maybe_push_budget_warning(&updated, &mut warnings);
        prose_plan_payload_with(&updated, &warnings, Some(reopened))
    }

    async fn plan_confirm(&self, p: PlanConfirmParams) -> Result<Value> {
        self.touch_activity();
        let id = parse_uuid(&p.plan_id)?;

        // Pre-flight: prose-first plans may still have outstanding
        // questions even after several rounds. Surface them as a
        // structured *successful* response with `status =
        // pending_clarification` rather than an error -- the caller
        // legitimately needs to render the questions and route them back
        // through `plan_answer`, which is a normal control-flow case,
        // not a tool failure.
        let plan = self.executor.status(id).await?;
        if plan.has_open_questions() {
            return Ok(json!({
                "plan_id": id.to_string(),
                "status": "pending_clarification",
                "reason": "open_questions_remain",
                "open_questions": plan.open_questions,
                "decisions": plan.decisions,
                "definition_dirty": plan.definition_dirty,
            }));
        }

        let preview = flowd_core::orchestration::build_preview(&plan)?;
        if preview.max_parallelism > 1 {
            if !self.executor.supports_worktree_isolation() {
                return Err(FlowdError::PlanExecution {
                    message: format!(
                        "plan `{id}` has max_parallelism={} but the configured agent spawner does not support git worktree isolation; refusing to run agents in one checkout",
                        preview.max_parallelism
                    ),
                    metrics: None,
                });
            }
            self.executor.prepare_plan(id).await?;
        }

        let preview = self.executor.confirm(id).await?;
        // Kick off execution in the background; plan_status is the polling
        // entry point. We intentionally do not propagate errors from the
        // detached task -- they land in the plan's recorded state instead.
        let exec = Arc::clone(&self.executor);
        tokio::spawn(async move {
            if let Err(e) = exec.execute(id).await {
                tracing::warn!(plan_id = %id, error = %e, "plan execution failed");
            }
        });
        Ok(json!({
            "plan_id": id.to_string(),
            "status": "running",
            "preview": preview,
        }))
    }

    async fn plan_cancel(&self, p: PlanCancelParams) -> Result<Value> {
        self.touch_activity();
        let id = parse_uuid(&p.plan_id)?;
        // Idempotent: terminal plans return ok without touching state.
        // Draft / Confirmed transition directly; Running flips the latch
        // and aborts in-flight steps. The Finished{Cancelled} event is
        // emitted by the executor itself.
        self.executor.cancel(id).await?;
        let plan = self.executor.status(id).await?;
        Ok(json!({
            "plan_id": id.to_string(),
            "status": plan_status_label(plan.status),
        }))
    }

    async fn plan_status(&self, p: PlanStatusParams) -> Result<Value> {
        // Deliberately do *not* touch_activity here: status polling should
        // not keep the daemon "busy" forever and starve the compactor.
        let id = parse_uuid(&p.plan_id)?;
        let plan = self.executor.status(id).await?;
        serde_json::to_value(&plan).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn plan_resume(&self, p: PlanResumeParams) -> Result<Value> {
        self.touch_activity();
        let id = parse_uuid(&p.plan_id)?;
        self.executor.resume_plan(id).await?;
        let exec = Arc::clone(&self.executor);
        tokio::spawn(async move {
            if let Err(e) = exec.execute(id).await {
                tracing::warn!(plan_id = %id, error = %e, "plan resume execution failed");
            }
        });
        Ok(json!({
            "plan_id": id.to_string(),
            "status": "running",
        }))
    }

    async fn plan_list(&self, p: PlanListParams) -> Result<Value> {
        self.touch_activity();
        let status = parse_plan_status_filter(p.status)?;
        let summaries = self.executor.list_plans(p.project).await?;
        let summaries = filter_plan_summaries(summaries, status, p.limit);
        Ok(plan_summaries_payload(&summaries))
    }

    async fn plan_show(&self, p: PlanShowParams) -> Result<Value> {
        self.touch_activity();
        let id = parse_uuid(&p.plan_id)?;
        let plan = self.executor.status(id).await?;
        serde_json::to_value(&plan).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn plan_recent(&self, p: PlanRecentParams) -> Result<Value> {
        self.touch_activity();
        let status = parse_plan_status_filter(p.status)?;
        let summaries = self.executor.list_plans(p.project).await?;
        let summaries = filter_plan_summaries(summaries, status, Some(p.limit.unwrap_or(5)));
        Ok(plan_summaries_payload(&summaries))
    }

    async fn rules_check(&self, p: RulesCheckParams) -> Result<Value> {
        self.touch_activity();
        let mut action = ProposedAction::new(p.tool);
        if let Some(f) = p.file_path {
            action = action.with_file(f);
        }
        if let Some(project) = p.project {
            action = action.with_project(project);
        }
        let result = self.rules.check(&action);
        serde_json::to_value(&result).map_err(|e| FlowdError::Serialization(e.to_string()))
    }

    async fn rules_list(&self, p: RulesListParams) -> Result<Value> {
        // rules_list is a read-only metadata query; treat like plan_status
        // and leave activity alone.
        let matches = self
            .rules
            .matching_rules(p.project.as_deref(), p.file_path.as_deref());
        let matches: Vec<_> = matches.into_iter().cloned().collect();
        Ok(json!({ "rules": matches }))
    }
}

impl<M, V, E, PE, PC, R> FlowdHandlers<M, V, E, PE, PC, R>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
    PE: PlanExecutor + 'static,
    PC: PlanCompiler + 'static,
    R: RuleEvaluator + 'static,
{
    /// Legacy DAG-first path: caller hands us a structured `PlanDefinition`.
    ///
    /// `client_root_hint` is the optional `project_root` field from
    /// [`PlanCreateParams`]: typically the workspace path the MCP
    /// client (or `flowd mcp` proxy) was launched in. The resolution
    /// chain (hint -> `FLOWD_WORKSPACE_ROOT` -> daemon cwd) lives in
    /// [`resolve_workspace_root`]; an explicit `project_root` embedded
    /// in the on-disk `definition` always wins because operators
    /// authoring a plan file are expected to know what they wrote.
    async fn plan_create_definition(
        &self,
        project: String,
        definition: Value,
        client_root_hint: Option<&str>,
    ) -> Result<Value> {
        let def: PlanDefinition = serde_json::from_value(definition)
            .map_err(|e| FlowdError::PlanValidation(format!("invalid PlanDefinition: {e}")))?;
        // Top-level `project` is authoritative: it overrides any value the
        // caller embedded in the definition payload, so the trusted handler
        // is the single source of truth for project scoping.
        let project_root = match def.project_root.clone() {
            Some(explicit) => Some(explicit),
            None => Some(resolve_workspace_root(client_root_hint)?.0),
        };
        let plan = def.into_plan_with_project_and_root(project, project_root);
        let preview = self.executor.preview(&plan)?;
        let plan_id = self.executor.submit(plan).await?;
        Ok(json!({
            "plan_id": plan_id.to_string(),
            "preview": preview,
        }))
    }

    /// Prose-first path: route `prose` through the configured compiler,
    /// submit a Draft plan with whatever the first round produced, emit
    /// clarification events, and return the unified prose-plan payload.
    ///
    /// `compiler_override` is forwarded to
    /// [`PlanCompiler::compile_prose_with_override`]; the trait's default
    /// impl ignores the override (single-backend compilers have no choice
    /// to make) so this is effectively a no-op for the Stub / Rejecting
    /// / Mock compilers and a routing hint for `DaemonPlanCompiler`.
    async fn plan_create_prose(
        &self,
        project: String,
        prose: String,
        compiler_override: Option<String>,
        client_root_hint: Option<&str>,
    ) -> Result<Value> {
        if prose.trim().is_empty() {
            return Err(FlowdError::PlanValidation(
                "plan_create: `prose` must be a non-empty string".into(),
            ));
        }
        // Resolve the workspace root *before* invoking the compiler so a
        // missing/garbage workspace fails fast and we don't burn an LLM
        // round on a plan that will never persist a useful project_root.
        let (resolved_root, _src) = resolve_workspace_root(client_root_hint)?;

        let derived_name = derive_plan_name(&prose);
        // Pass the resolved root through to the compiler so the daemon's
        // project-scoped wrapper can dispatch on
        // `<project_root>/.flowd/flowd.toml` rather than its global
        // `$FLOWD_HOME/flowd.toml`. Single-backend / test compilers ignore
        // it via the trait's default `_in_project` impl.
        let output = self
            .compiler
            .compile_prose_in_project(
                prose,
                project.clone(),
                Some(resolved_root.clone()),
                compiler_override,
            )
            .await?;

        // Build a Plan around whatever the compiler returned. If the
        // compiler resolved everything in one shot we still go through
        // apply_compile_output for symmetry: that way the only place
        // PlanDefinition -> PlanStep conversion happens is in
        // Plan::apply_compile_output, and source_doc / open_questions /
        // decisions live in exactly one update path. The execution root
        // captured above is stamped onto the Draft now so resume / list
        // / show paths can reach the original workspace even after the
        // daemon's CWD drifts.
        let mut plan =
            Plan::new(derived_name, project.clone(), Vec::new()).with_project_root(resolved_root);
        let new_question_ids: Vec<String> =
            output.open_questions.iter().map(|q| q.id.clone()).collect();
        let new_decision_ids: Vec<String> = output
            .new_decisions
            .iter()
            .map(|d| d.question_id.clone())
            .collect();
        plan.apply_compile_output(output);

        let plan_id = self.executor.submit(plan).await?;
        // Re-fetch through the executor so we return whatever the store
        // round-tripped (avoids drift between the in-handler clone and
        // the persisted snapshot).
        let stored = self.executor.status(plan_id).await?;
        self.emit_clarification_deltas(&stored, new_question_ids, new_decision_ids);
        let mut warnings: Vec<ProsePlanWarning> = Vec::new();
        self.maybe_push_budget_warning(&stored, &mut warnings);
        prose_plan_payload_with(&stored, &warnings, None)
    }

    /// Append a `BudgetExceeded` warning to `warnings` when the plan's
    /// current load (questions + decisions) crosses the configured
    /// budget. Inert when no budget is configured.
    fn maybe_push_budget_warning(&self, plan: &Plan, warnings: &mut Vec<ProsePlanWarning>) {
        if let Some(budget) = self.question_budget {
            let load = clarification_load(plan);
            if load > budget {
                warnings.push(budget_exceeded_warning(budget, load));
            }
        }
    }
}

/// Mark [`PlanStatus`] as MCP-visible; the JSON value is what callers see.
#[must_use]
pub fn plan_status_label(status: PlanStatus) -> &'static str {
    match status {
        PlanStatus::Draft => "draft",
        PlanStatus::Confirmed => "confirmed",
        PlanStatus::Running => "running",
        PlanStatus::Interrupted => "interrupted",
        PlanStatus::Completed => "completed",
        PlanStatus::Failed => "failed",
        PlanStatus::Cancelled => "cancelled",
    }
}
