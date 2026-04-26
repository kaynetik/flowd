//! Orchestration engine: plans, validation, execution.
//!
//! A `Plan` is a DAG of `PlanStep`s. The engine drives plans through a small
//! state machine: `Draft` → `Confirmed` → `Running` → `Completed` / `Failed`,
//! with `Cancelled` reachable from `Confirmed` or `Running`. A daemon restart
//! while work is in flight settles the plan as `Interrupted`, which is resumable.
//!
//! This module owns the trait surface and pure-data types. Concrete pieces:
//!
//! * Authored input format + parsers: [`loader`].
//! * In-process executor that supervises agents: [`executor`].
//!
//! `flowd-core` deliberately stays I/O-framework-free; spawning real agent
//! processes is done by an [`executor::AgentSpawner`] implementation supplied
//! by the CLI / MCP layer.

pub mod clarification;
pub mod compiler;
pub mod executor;
pub mod gate;
mod layer_runner;
pub mod loader;
pub mod observer;
pub mod plan_events;
pub mod store;
pub mod template;

use crate::error::{FlowdError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::future::Future;
use uuid::Uuid;

pub use clarification::{Answer, DecisionRecord, OpenQuestion, QuestionOption};
pub use compiler::{CompileOutput, MockPlanCompiler, PlanCompiler, PlanDraftSnapshot};
pub use executor::{AgentOutput, AgentSpawnContext, AgentSpawner, InMemoryPlanExecutor};
pub use loader::{PlanDefinition, StepDefinition, load_plan, load_plan_json, load_plan_yaml};
pub use store::{NoOpPlanStore, PlanStore, PlanSummary};

/// A complete orchestration plan (DAG of steps).
///
/// `project` is the central namespace key across plans, rules, and
/// observations: every rule's scope glob is matched against it, the plan
/// observer scopes its writes by it, and `flowd history` / `flowd search`
/// filter by it. Making it required (rather than `Option<String>`) removes
/// an entire class of "silent skip" bugs that previously bit rules with
/// `scope: "**"` and the memory plan observer when no project was set.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Plan {
    pub id: Uuid,
    pub name: String,
    /// Required namespace label (e.g. repo id). Used for rule scoping,
    /// observation grouping, and listing filters.
    ///
    /// Deserialisation tolerates a missing `project` for one specific
    /// reason: pre-`HL-38` plan JSON stored in the `plans.definition`
    /// column can be missing the field entirely (or `null`). Such rows
    /// are backfilled to `__legacy__` -- matching the `SQLite` migration
    /// `MIGRATION_003` -- so callers never see a partially-migrated
    /// Plan. New code paths supply a real value via [`Self::new`] and
    /// [`crate::orchestration::loader::PlanDefinition::into_plan`].
    #[serde(
        default = "default_legacy_project",
        deserialize_with = "deserialize_project_with_legacy_fallback"
    )]
    pub project: String,
    /// Canonical filesystem root the plan executes against (the
    /// "execution root"). Distinct from [`Self::project`]: `project` is
    /// the namespace label shared across plans / rules / observations
    /// and may be a short repo id, whereas `project_root` is an absolute
    /// path captured from the invoking workspace at submission time.
    ///
    /// `None` for plans persisted before this field existed; readers
    /// should fall back to per-call discovery in that case rather than
    /// assuming a value. New code paths populate it via
    /// [`infer_project_root_from_cwd`] (or an explicit override).
    #[serde(default)]
    pub project_root: Option<String>,
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
    /// Original prose the plan was authored from, when the prose-first
    /// front door was used. `None` for plans submitted directly as a
    /// [`PlanDefinition`]. Preserved post-confirm for audit and so the
    /// compiler can re-render on resume / refinement.
    #[serde(default)]
    pub source_doc: Option<String>,
    /// Outstanding clarifications surfaced by the compiler. While this
    /// list is non-empty the plan cannot transition out of
    /// [`PlanStatus::Draft`]; [`validate_plan`] will reject confirmation
    /// and the executor refuses to run.
    #[serde(default)]
    pub open_questions: Vec<OpenQuestion>,
    /// Resolved clarifications, in the order they were answered. Used to
    /// invalidate downstream questions when a prior answer is overwritten.
    #[serde(default)]
    pub decisions: Vec<DecisionRecord>,
    /// True when [`Self::source_doc`] / [`Self::open_questions`] /
    /// [`Self::decisions`] have changed since the last successful compile,
    /// signalling that [`PlanStep`]s in the wider [`Plan`] no longer
    /// reflect the authored intent and must be regenerated before
    /// confirmation. Always `false` for definition-first plans.
    #[serde(default)]
    pub definition_dirty: bool,
}

fn default_legacy_project() -> String {
    "__legacy__".to_owned()
}

/// Environment variable consulted by [`resolve_workspace_root`] when no
/// MCP client signal accompanies the request. Lets an operator pin the
/// workspace at daemon-launch (or per `flowd mcp` proxy invocation)
/// without rebuilding the binary.
pub const FLOWD_WORKSPACE_ROOT_ENV: &str = "FLOWD_WORKSPACE_ROOT";

/// Which signal a successful [`resolve_workspace_root`] picked up. Carried
/// alongside the resolved path so the caller can log `ClientHint` vs.
/// `EnvOverride` vs. `Cwd` without re-running the resolver.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WorkspaceRootSource {
    /// MCP client / proxy supplied the path in the request payload. The
    /// only signal that reflects the *invoking* workspace when the daemon
    /// was launched somewhere unrelated.
    ClientHint,
    /// `FLOWD_WORKSPACE_ROOT` env var on the resolving process.
    EnvOverride,
    /// `current_dir()` of the resolving process. Last resort; meaningful
    /// only when the daemon was launched from inside the workspace it is
    /// expected to serve (typical in single-user / test setups).
    Cwd,
}

/// Resolve a canonical, git-verified workspace root from the available
/// signals.
///
/// Resolution order (highest priority first):
/// 1. `client_hint` -- typically the MCP client or `flowd mcp` proxy
///    forwarding the workspace it was spawned from.
/// 2. `FLOWD_WORKSPACE_ROOT` env var on the resolving process.
/// 3. The resolving process's `current_dir()`.
///
/// Each candidate is canonicalised (resolves symlinks and `..`), then the
/// resulting path -- and its parents -- are scanned for a `.git` entry
/// (directory or file; the latter is what `git worktree` and submodules
/// emit). The first candidate that resolves and lives inside a git
/// repository wins; the returned path is the canonical form of the
/// candidate itself (NOT the repo root), so two plans created from the
/// same subdirectory compare equal even when the repo root sits a few
/// levels up.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` enumerating every candidate it
/// considered and why each was rejected, so an operator can tell at a
/// glance whether to set `FLOWD_WORKSPACE_ROOT`, fix the proxy, or run
/// `git init` in the workspace.
pub fn resolve_workspace_root(client_hint: Option<&str>) -> Result<(String, WorkspaceRootSource)> {
    let mut tried: Vec<String> = Vec::new();

    if let Some(raw) = client_hint {
        match try_workspace_candidate(raw) {
            Ok(canonical) => return Ok((canonical, WorkspaceRootSource::ClientHint)),
            Err(reason) => tried.push(format!("client hint `{raw}`: {reason}")),
        }
    }

    match std::env::var(FLOWD_WORKSPACE_ROOT_ENV) {
        Ok(raw) => match try_workspace_candidate(&raw) {
            Ok(canonical) => return Ok((canonical, WorkspaceRootSource::EnvOverride)),
            Err(reason) => tried.push(format!("{FLOWD_WORKSPACE_ROOT_ENV}=`{raw}`: {reason}")),
        },
        Err(_) => tried.push(format!("{FLOWD_WORKSPACE_ROOT_ENV} not set")),
    }

    match std::env::current_dir() {
        Ok(cwd) => match try_workspace_path(&cwd) {
            Ok(canonical) => return Ok((canonical, WorkspaceRootSource::Cwd)),
            Err(reason) => tried.push(format!("cwd `{}`: {reason}", cwd.display())),
        },
        Err(e) => tried.push(format!("cwd unavailable: {e}")),
    }

    Err(FlowdError::PlanValidation(format!(
        "could not resolve a trustworthy workspace root for the plan; \
         tried, in order: {}; set {FLOWD_WORKSPACE_ROOT_ENV} to a git-tracked \
         directory or have the MCP client supply `project_root` in the request",
        tried.join("; ")
    )))
}

fn try_workspace_candidate(raw: &str) -> std::result::Result<String, String> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err("path is empty / whitespace".into());
    }
    try_workspace_path(std::path::Path::new(trimmed))
}

fn try_workspace_path(path: &std::path::Path) -> std::result::Result<String, String> {
    let canonical =
        std::fs::canonicalize(path).map_err(|e| format!("canonicalize failed ({e})"))?;
    if !is_within_git_repo(&canonical) {
        return Err(
            "not inside a git repository (no .git found in this directory or any parent)".into(),
        );
    }
    Ok(canonical.display().to_string())
}

/// Walk `path` and its parents looking for a `.git` entry. Returns true
/// for plain repos (`.git/` directory), worktrees and submodules
/// (`.git` is a file pointing at the gitdir), and bare-repo subdirs the
/// user might be editing inside. We deliberately do not exec `git
/// rev-parse`: this is hot-path code and a filesystem probe is enough
/// for the trust check the resolver makes.
fn is_within_git_repo(path: &std::path::Path) -> bool {
    let mut cursor = Some(path);
    while let Some(dir) = cursor {
        if dir.join(".git").exists() {
            return true;
        }
        cursor = dir.parent();
    }
    false
}

/// Best-effort canonical workspace root from `current_dir()` only --
/// retained as a thin wrapper for callers (and persisted plan loaders)
/// that already trust their cwd, do not yet have an MCP client signal,
/// and still want the `Option`-returning shape. Newer call sites should
/// prefer [`resolve_workspace_root`] directly so the env-override and
/// git-verification paths are honoured.
#[must_use]
pub fn infer_project_root_from_cwd() -> Option<String> {
    resolve_workspace_root(None).ok().map(|(p, _)| p)
}

/// Re-canonicalise an externally-supplied execution root with the same
/// git-verification [`resolve_workspace_root`] applies. Returns `None`
/// for inputs that don't resolve to a git-tracked directory; the caller
/// should decide whether to fall back to inference or refuse.
#[must_use]
pub fn canonicalize_project_root(raw: &str) -> Option<String> {
    try_workspace_candidate(raw).ok()
}

fn deserialize_project_with_legacy_fallback<'de, D>(
    deserializer: D,
) -> std::result::Result<String, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let opt: Option<String> = Option::deserialize(deserializer)?;
    Ok(opt.map_or_else(default_legacy_project, |s| {
        if s.trim().is_empty() {
            default_legacy_project()
        } else {
            s
        }
    }))
}

/// An individual step within a plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanStep {
    pub id: String,
    pub agent_type: String,
    pub prompt: String,
    pub depends_on: Vec<String>,
    pub timeout_secs: Option<u64>,
    pub retry_count: u32,
    pub status: StepStatus,
    pub output: Option<String>,
    pub error: Option<String>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PlanStatus {
    Draft,
    Confirmed,
    Running,
    Interrupted,
    Completed,
    Failed,
    Cancelled,
}

impl PlanStatus {
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::Completed | Self::Failed | Self::Cancelled)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum StepStatus {
    Pending,
    Ready,
    Running,
    Completed,
    Failed,
    /// Executor-side refusal: the step was never spawned because something
    /// upstream of the agent (a rule gate, a missing precondition) declined
    /// to run it. Distinct from `Cancelled`, which is a user-initiated stop.
    Skipped,
    /// User-initiated stop while the step was pending or in flight. Reached
    /// via [`PlanExecutor::cancel`].
    Cancelled,
}

/// Preview of a plan before confirmation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanPreview {
    pub plan_id: Uuid,
    pub name: String,
    pub total_steps: usize,
    pub total_agents: usize,
    pub max_parallelism: usize,
    pub execution_order: Vec<Vec<String>>,
    pub dependency_graph: HashMap<String, Vec<String>>,
}

/// Executes orchestration plans.
///
/// Lifecycle:
/// 1. `submit(plan)` registers a `Draft` plan and returns its id.
/// 2. `confirm(id)` validates and transitions `Draft` → `Confirmed`,
///    returning the preview for display.
/// 3. `execute(id)` drives a `Confirmed` plan to completion.
/// 4. `status(id)` returns the current `Plan` snapshot at any time.
/// 5. `cancel(id)` aborts an in-flight plan.
pub trait PlanExecutor: Send + Sync {
    /// Validate plan structure (cycles, missing deps, duplicate ids).
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` if the plan is malformed.
    fn validate(&self, plan: &Plan) -> Result<()>;

    /// Generate a preview describing parallelism and execution order.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` if the plan structure is invalid.
    fn preview(&self, plan: &Plan) -> Result<PlanPreview>;

    /// Submit a new plan in the `Draft` state. Returns the assigned id.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` if the plan fails validation.
    fn submit(&self, plan: Plan) -> impl Future<Output = Result<Uuid>> + Send;

    /// Move a `Draft` plan to `Confirmed`. Returns the preview so the
    /// caller (CLI, MCP) can render it after the operator answers Y.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanNotFound` if the plan is unknown,
    /// `FlowdError::PlanExecution` if the plan is not in `Draft`.
    fn confirm(&self, plan_id: Uuid) -> impl Future<Output = Result<PlanPreview>> + Send;

    /// Drive a `Confirmed` plan to completion. Returns when the plan reaches
    /// a terminal state (`Completed`, `Failed`, or `Cancelled`).
    ///
    /// # Errors
    /// Returns `FlowdError::PlanExecution` for state-machine violations
    /// (e.g. plan not yet confirmed).
    fn execute(&self, plan_id: Uuid) -> impl Future<Output = Result<()>> + Send;

    /// Snapshot the current state of a known plan.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanNotFound` if the plan is unknown.
    fn status(&self, plan_id: Uuid) -> impl Future<Output = Result<Plan>> + Send;

    /// List persisted plan summaries, newest first, optionally scoped to
    /// one project.
    ///
    /// # Errors
    /// Propagates storage failures from the backing [`PlanStore`].
    fn list_plans(
        &self,
        project: Option<String>,
    ) -> impl Future<Output = Result<Vec<PlanSummary>>> + Send;

    /// Whether confirmed parallel plans can run in isolated git worktrees.
    #[must_use]
    fn supports_worktree_isolation(&self) -> bool;

    /// Validate plan-level execution prerequisites before confirmation.
    ///
    /// # Errors
    /// Returns a plan execution error when the plan cannot safely execute
    /// with the installed spawner.
    fn prepare_plan(&self, plan_id: Uuid) -> impl Future<Output = Result<PlanPreview>> + Send;

    /// Cancel a plan. Behaviour depends on current state:
    ///
    /// * `Draft` or `Confirmed` (no execution in flight): transitions
    ///   directly to `Cancelled`, persists, and emits the lifecycle
    ///   event. Used by the prose-first front door so a user can abandon
    ///   a plan mid-clarification.
    /// * `Running`: sets the cancellation latch and aborts in-flight
    ///   step tasks; the executing task settles the plan as `Cancelled`
    ///   in its finalisation step.
    /// * Terminal (`Completed`, `Failed`, `Cancelled`): no-op,
    ///   `Ok(())`. Cancellation is idempotent for already-finished
    ///   plans.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanNotFound` if the plan is unknown.
    fn cancel(&self, plan_id: Uuid) -> impl Future<Output = Result<()>> + Send;

    /// Reset every [`crate::orchestration::StepStatus::Failed`] step to
    /// [`StepStatus::Pending`], move the plan to [`PlanStatus::Confirmed`],
    /// and clear the cancellation latch so [`Self::execute`] can run again.
    ///
    /// Does not start execution; callers typically follow with `execute`.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanNotFound` if the plan is unknown, or
    /// `FlowdError::PlanExecution` if the plan is not `Failed` or `Interrupted`.
    fn resume_plan(&self, plan_id: Uuid) -> impl Future<Output = Result<()>> + Send;

    /// Apply a [`compiler::CompileOutput`] onto an in-flight `Draft`
    /// plan, replacing prose / open questions and (when the compiler
    /// emitted a [`PlanDefinition`]) the underlying steps. Persists the
    /// resulting snapshot.
    ///
    /// Used by the MCP plan handlers after every successful
    /// [`compiler::PlanCompiler`] invocation. Refuses to mutate plans
    /// that are not `Draft`.
    ///
    /// # Errors
    /// `FlowdError::PlanNotFound` if the plan is unknown,
    /// `FlowdError::PlanExecution` if the plan is not `Draft`.
    fn apply_compile_output(
        &self,
        plan_id: Uuid,
        output: compiler::CompileOutput,
    ) -> impl Future<Output = Result<()>> + Send;

    /// Drop the [`DecisionRecord`] for `question_id` and any decisions
    /// transitively dependent on it, marking the plan dirty. Returns the
    /// invalidated question ids in resolution order. Persists.
    ///
    /// Called by the MCP layer immediately before re-invoking the
    /// compiler with an overwriting answer; the compiler is then
    /// responsible for re-emitting the affected questions or
    /// re-establishing decisions.
    ///
    /// # Errors
    /// `FlowdError::PlanNotFound` if the plan is unknown,
    /// `FlowdError::PlanExecution` if the plan is not `Draft`.
    fn invalidate_decision(
        &self,
        plan_id: Uuid,
        question_id: String,
    ) -> impl Future<Output = Result<Vec<String>>> + Send;
}

impl Plan {
    /// Construct a `Draft` plan from authored data.
    ///
    /// `project` is required because every downstream subsystem (rule
    /// evaluator, plan observer, persistence layer) treats it as a
    /// non-null namespace key. The execution root is left unset; callers
    /// that know their invoking workspace should chain
    /// [`Self::with_project_root`].
    #[must_use]
    pub fn new(name: impl Into<String>, project: impl Into<String>, steps: Vec<PlanStep>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            project: project.into(),
            project_root: None,
            steps,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
            source_doc: None,
            open_questions: Vec::new(),
            decisions: Vec::new(),
            definition_dirty: false,
        }
    }

    /// Builder-style setter for the execution root. Trims and ignores
    /// empty / whitespace-only inputs so callers can pipe a possibly
    /// missing value (e.g. an env var) through without branching.
    #[must_use]
    pub fn with_project_root(mut self, root: impl Into<String>) -> Self {
        let root = root.into();
        if root.trim().is_empty() {
            self.project_root = None;
        } else {
            self.project_root = Some(root);
        }
        self
    }

    /// Compute which steps can run in parallel at each stage (topological layers).
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` if the dependency graph contains a cycle.
    #[allow(clippy::missing_panics_doc)]
    pub fn execution_layers(&self) -> Result<Vec<Vec<String>>> {
        let mut in_degree: HashMap<&str, usize> = HashMap::new();
        let mut dependents: HashMap<&str, Vec<&str>> = HashMap::new();

        for step in &self.steps {
            in_degree.entry(&step.id).or_insert(0);
            for dep in &step.depends_on {
                dependents.entry(dep.as_str()).or_default().push(&step.id);
                *in_degree.entry(&step.id).or_insert(0) += 1;
            }
        }

        let mut layers = Vec::new();
        let mut queue: Vec<&str> = in_degree
            .iter()
            .filter(|(_, deg)| **deg == 0)
            .map(|(id, _)| *id)
            .collect();
        queue.sort_unstable();

        let mut remaining = self.steps.len();

        while !queue.is_empty() {
            let layer: Vec<String> = queue.iter().map(|s| (*s).to_owned()).collect();
            remaining -= layer.len();

            let mut next_queue = Vec::new();
            for id in &queue {
                if let Some(deps) = dependents.get(id) {
                    for dep in deps {
                        let deg = in_degree
                            .get_mut(dep)
                            .expect("step must exist in degree map");
                        *deg -= 1;
                        if *deg == 0 {
                            next_queue.push(*dep);
                        }
                    }
                }
            }
            next_queue.sort_unstable();

            layers.push(layer);
            queue = next_queue;
        }

        if remaining > 0 {
            return Err(FlowdError::PlanValidation(
                "cycle detected in plan dependencies".into(),
            ));
        }

        Ok(layers)
    }

    /// True iff at least one [`OpenQuestion`] is still awaiting an answer.
    /// Plans in this state cannot be confirmed; the executor and
    /// [`validate_plan`] both refuse them.
    #[must_use]
    pub fn has_open_questions(&self) -> bool {
        !self.open_questions.is_empty()
    }

    /// Apply a [`compiler::CompileOutput`] in place.
    ///
    /// Replaces [`Self::source_doc`] and [`Self::open_questions`] wholesale
    /// (the compiler is the source of truth for both), appends new
    /// decisions, and -- when the compiler emitted a fully resolved
    /// [`PlanDefinition`] -- regenerates [`Self::steps`] from it and
    /// clears [`Self::definition_dirty`]. When no definition is supplied
    /// (questions still open), the dirty flag is set so a subsequent
    /// [`validate_plan`] / `confirm` knows the steps are stale.
    ///
    /// The plan's [`Self::project`] is preserved across recompiles; the
    /// compiler never gets to rewrite it.
    pub fn apply_compile_output(&mut self, output: compiler::CompileOutput) {
        self.source_doc = Some(output.source_doc);
        self.open_questions = output.open_questions;
        self.decisions.extend(output.new_decisions);
        if let Some(def) = output.definition {
            // Preserve per-step `timeout_secs` that the prose-first
            // compilers routinely lose across rounds. The structured
            // markdown stub emits `None` for any header without an
            // explicit `timeout_secs:` field, and the LLM-backed
            // compiler is similarly free to drop it on a refine. Without
            // this, a refined plan would silently fall back to the
            // daemon-wide default and quietly violate the operator's
            // authored deadline.
            let prior_timeouts: HashMap<String, u64> = self
                .steps
                .iter()
                .filter_map(|s| s.timeout_secs.map(|t| (s.id.clone(), t)))
                .collect();
            let recompiled = def.into_plan_with_project(self.project.clone());
            self.name = recompiled.name;
            self.steps = recompiled.steps;
            for s in &mut self.steps {
                if s.timeout_secs.is_none() {
                    if let Some(prior) = prior_timeouts.get(&s.id) {
                        s.timeout_secs = Some(*prior);
                    }
                }
            }
            self.definition_dirty = false;
        } else {
            self.definition_dirty = true;
        }
    }

    /// Drop the decision keyed on `question_id` and any decisions whose
    /// `depends_on_decisions` chain transitively contains it. Marks the
    /// plan dirty.
    ///
    /// Returns the question ids that were invalidated, in deterministic
    /// (input-order) form, so the caller can log or surface them. Does
    /// not re-emit [`OpenQuestion`]s -- the next compiler call (typically
    /// [`compiler::PlanCompiler::apply_answers`] with the new answer for
    /// `question_id`) is responsible for that.
    pub fn invalidate_decision(&mut self, question_id: &str) -> Vec<String> {
        // Build the transitive closure of dependents in the decision
        // graph. We iterate to a fixed point so a chain of length N is
        // captured even when decisions are stored out of resolution order
        // (which they shouldn't be, but defensively).
        let mut invalidated: HashSet<String> = HashSet::new();
        invalidated.insert(question_id.to_owned());
        loop {
            let before = invalidated.len();
            for d in &self.decisions {
                if invalidated.contains(&d.question_id) {
                    continue;
                }
                if d.depends_on_decisions
                    .iter()
                    .any(|dep| invalidated.contains(dep))
                {
                    invalidated.insert(d.question_id.clone());
                }
            }
            if invalidated.len() == before {
                break;
            }
        }

        // Preserve original order of removed entries so the caller's log
        // / response is stable.
        let mut removed_in_order: Vec<String> = Vec::new();
        self.decisions.retain(|d| {
            let drop_it = invalidated.contains(&d.question_id);
            if drop_it {
                removed_in_order.push(d.question_id.clone());
            }
            !drop_it
        });

        if !removed_in_order.is_empty() {
            self.definition_dirty = true;
        }
        removed_in_order
    }
}

/// Validate plan structure independently of any executor.
///
/// Checks, in order:
/// - No outstanding clarifications (the prose-first compiler must have
///   resolved every [`OpenQuestion`] before the plan can be executed).
/// - Required scalar fields are non-empty.
/// - At least one step.
/// - No duplicate step IDs.
/// - Every `depends_on` entry references a real step.
/// - No cycles (delegated to `Plan::execution_layers`).
/// - Every `{{steps.<id>.output}}` template token is well-formed and
///   refers to a transitive dependency (delegated to
///   [`template::validate_step_references`]).
///
/// In debug builds, additionally asserts that every
/// `depends_on_decisions` edge in [`Plan::open_questions`] /
/// [`Plan::decisions`] refers to a real recorded decision; this catches
/// compiler bugs where the dependency graph is left incoherent.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` describing the first violation.
pub fn validate_plan(plan: &Plan) -> Result<()> {
    if plan.has_open_questions() {
        // Compact, machine-friendly: callers (notably the MCP
        // `plan_confirm` handler) surface a richer payload by reading
        // [`Plan::open_questions`] directly. Embedding only ids here keeps
        // the error message bounded and avoids leaking long prompts into
        // log streams.
        let ids: Vec<&str> = plan.open_questions.iter().map(|q| q.id.as_str()).collect();
        return Err(FlowdError::PlanValidation(format!(
            "plan has {} unresolved clarification(s): {}; resolve via plan_answer/plan_refine before confirming",
            plan.open_questions.len(),
            serde_json::to_string(&ids).unwrap_or_else(|_| "[]".into()),
        )));
    }
    debug_assert!(
        decision_graph_is_coherent(plan),
        "compiler emitted depends_on_decisions edges referencing unknown decisions; \
         see Plan::open_questions / Plan::decisions"
    );

    if plan.name.trim().is_empty() {
        return Err(FlowdError::PlanValidation("plan name is empty".into()));
    }
    if plan.project.trim().is_empty() {
        return Err(FlowdError::PlanValidation("plan project is empty".into()));
    }
    if plan.steps.is_empty() {
        return Err(FlowdError::PlanValidation("plan has no steps".into()));
    }

    let mut seen: HashSet<&str> = HashSet::new();
    for step in &plan.steps {
        if step.id.trim().is_empty() {
            return Err(FlowdError::PlanValidation("step id is empty".into()));
        }
        if !seen.insert(&step.id) {
            return Err(FlowdError::PlanValidation(format!(
                "duplicate step id `{}`",
                step.id
            )));
        }
        if step.agent_type.trim().is_empty() {
            return Err(FlowdError::PlanValidation(format!(
                "step `{}` has empty agent_type",
                step.id
            )));
        }
    }

    let ids: HashSet<&str> = plan.steps.iter().map(|s| s.id.as_str()).collect();
    for step in &plan.steps {
        for dep in &step.depends_on {
            if !ids.contains(dep.as_str()) {
                return Err(FlowdError::PlanValidation(format!(
                    "step `{}` depends on unknown step `{}`",
                    step.id, dep
                )));
            }
            if dep == &step.id {
                return Err(FlowdError::PlanValidation(format!(
                    "step `{}` depends on itself",
                    step.id
                )));
            }
        }
    }

    plan.execution_layers()?;
    template::validate_step_references(plan)?;
    Ok(())
}

/// Lighter-touch validation used at plan-submission time for prose-first
/// plans that still have outstanding clarifications.
///
/// A freshly created prose-first [`Plan`] legitimately has no compiled
/// steps yet -- the compiler will fill them in once the user has
/// answered enough questions. Forcing it through the strict
/// [`validate_plan`] would reject every such plan at submission.
///
/// Behaviour:
/// * Always requires `name` and `project` to be non-empty.
/// * If [`Plan::has_open_questions`] is true, returns `Ok(())` without
///   inspecting steps. The strict check fires later, at confirmation.
/// * Otherwise delegates to [`validate_plan`], so plans submitted without
///   clarifications (the existing definition-first path) get the same
///   guarantees as before.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` describing the first violation.
pub fn validate_plan_pending(plan: &Plan) -> Result<()> {
    if plan.name.trim().is_empty() {
        return Err(FlowdError::PlanValidation("plan name is empty".into()));
    }
    if plan.project.trim().is_empty() {
        return Err(FlowdError::PlanValidation("plan project is empty".into()));
    }
    if plan.has_open_questions() {
        debug_assert!(
            decision_graph_is_coherent(plan),
            "compiler emitted depends_on_decisions edges referencing unknown decisions"
        );
        return Ok(());
    }
    validate_plan(plan)
}

/// Debug-only invariant: every id appearing in any
/// `depends_on_decisions` list (on either an [`OpenQuestion`] or a
/// [`DecisionRecord`]) must correspond to a recorded [`DecisionRecord`].
///
/// We deliberately do *not* require open-question dependencies to point
/// to *future* questions in the same payload -- the compiler is allowed
/// to lazily surface follow-ups across rounds. We only require that
/// every cited decision actually exists at the moment of validation.
fn decision_graph_is_coherent(plan: &Plan) -> bool {
    let known: HashSet<&str> = plan
        .decisions
        .iter()
        .map(|d| d.question_id.as_str())
        .collect();
    plan.decisions
        .iter()
        .flat_map(|d| d.depends_on_decisions.iter())
        .chain(
            plan.open_questions
                .iter()
                .flat_map(|q| q.depends_on_decisions.iter()),
        )
        .all(|id| known.contains(id.as_str()))
}

/// Build a `PlanPreview` for a validated plan.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` if the plan is invalid.
pub fn build_preview(plan: &Plan) -> Result<PlanPreview> {
    validate_plan(plan)?;
    let layers = plan.execution_layers()?;

    let max_parallelism = layers.iter().map(Vec::len).max().unwrap_or(0);
    let total_agents: HashSet<&str> = plan.steps.iter().map(|s| s.agent_type.as_str()).collect();
    let dependency_graph: HashMap<String, Vec<String>> = plan
        .steps
        .iter()
        .map(|s| (s.id.clone(), s.depends_on.clone()))
        .collect();

    Ok(PlanPreview {
        plan_id: plan.id,
        name: plan.name.clone(),
        total_steps: plan.steps.len(),
        total_agents: total_agents.len(),
        max_parallelism,
        execution_order: layers,
        dependency_graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(id: &str, deps: &[&str]) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "echo".into(),
            prompt: format!("do {id}"),
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
    fn validate_rejects_empty_plan() {
        let plan = Plan::new("p", "proj", vec![]);
        assert!(validate_plan(&plan).is_err());
    }

    #[test]
    fn validate_rejects_empty_project() {
        let plan = Plan::new("p", "", vec![step("a", &[])]);
        let err = validate_plan(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(msg) if msg.contains("project")));
    }

    #[test]
    fn validate_rejects_unknown_dependency() {
        let plan = Plan::new("p", "proj", vec![step("a", &["ghost"])]);
        let err = validate_plan(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(msg) if msg.contains("ghost")));
    }

    #[test]
    fn validate_rejects_duplicate_ids() {
        let plan = Plan::new("p", "proj", vec![step("a", &[]), step("a", &[])]);
        let err = validate_plan(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(msg) if msg.contains("duplicate")));
    }

    #[test]
    fn validate_rejects_self_dependency() {
        let plan = Plan::new("p", "proj", vec![step("a", &["a"])]);
        let err = validate_plan(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(msg) if msg.contains("itself")));
    }

    #[test]
    fn validate_rejects_cycle() {
        let plan = Plan::new(
            "p",
            "proj",
            vec![step("a", &["b"]), step("b", &["c"]), step("c", &["a"])],
        );
        let err = validate_plan(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(msg) if msg.contains("cycle")));
    }

    #[test]
    fn execution_layers_orders_dependencies() {
        // a -> c, b -> c, c -> d
        let plan = Plan::new(
            "p",
            "proj",
            vec![
                step("a", &[]),
                step("b", &[]),
                step("c", &["a", "b"]),
                step("d", &["c"]),
            ],
        );
        let layers = plan.execution_layers().unwrap();
        assert_eq!(layers.len(), 3);
        assert_eq!(layers[0], vec!["a".to_owned(), "b".to_owned()]);
        assert_eq!(layers[1], vec!["c".to_owned()]);
        assert_eq!(layers[2], vec!["d".to_owned()]);
    }

    #[test]
    fn build_preview_reports_parallelism() {
        let plan = Plan::new(
            "p",
            "proj",
            vec![
                step("a", &[]),
                step("b", &[]),
                step("c", &[]),
                step("d", &["a", "b", "c"]),
            ],
        );
        let preview = build_preview(&plan).unwrap();
        assert_eq!(preview.total_steps, 4);
        assert_eq!(preview.max_parallelism, 3);
        assert_eq!(preview.execution_order.len(), 2);
    }

    #[test]
    fn plan_new_defaults_clarification_fields_empty() {
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        assert!(plan.source_doc.is_none());
        assert!(plan.open_questions.is_empty());
        assert!(plan.decisions.is_empty());
        assert!(!plan.definition_dirty);
    }

    /// Pre-Phase-1 plan JSON (i.e. without any clarification fields) must
    /// still deserialise cleanly so the persisted `plans.definition` blobs
    /// keep loading after this crate is upgraded.
    #[test]
    fn plan_deserialises_legacy_json_without_clarification_fields() {
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        let mut json: serde_json::Value = serde_json::to_value(&plan).unwrap();
        let obj = json.as_object_mut().unwrap();
        obj.remove("source_doc");
        obj.remove("open_questions");
        obj.remove("decisions");
        obj.remove("definition_dirty");

        let restored: Plan = serde_json::from_value(json).unwrap();
        assert!(restored.source_doc.is_none());
        assert!(restored.open_questions.is_empty());
        assert!(restored.decisions.is_empty());
        assert!(!restored.definition_dirty);
        assert_eq!(restored.steps.len(), 1);
    }

    fn question(id: &str, deps: &[&str]) -> OpenQuestion {
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
            depends_on_decisions: deps.iter().map(|s| (*s).to_owned()).collect(),
        }
    }

    fn decision(question_id: &str, deps: &[&str]) -> DecisionRecord {
        DecisionRecord::new_user(
            question_id,
            "x",
            deps.iter().map(|s| (*s).to_owned()).collect(),
        )
    }

    #[test]
    fn validate_plan_rejects_open_questions_with_inline_id_list() {
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.open_questions.push(question("q1", &[]));
        plan.open_questions.push(question("q2", &[]));
        let err = validate_plan(&plan).unwrap_err();
        let msg = match err {
            FlowdError::PlanValidation(m) => m,
            other => panic!("unexpected variant: {other:?}"),
        };
        assert!(
            msg.contains("2 unresolved"),
            "missing count in message: {msg}"
        );
        assert!(
            msg.contains(r#"["q1","q2"]"#),
            "missing inline id list in message: {msg}"
        );
    }

    #[test]
    fn validate_plan_pending_accepts_no_steps_when_questions_open() {
        let mut plan = Plan::new("p", "proj", vec![]);
        plan.open_questions.push(question("q1", &[]));
        validate_plan_pending(&plan).expect("pending should accept clarification-stage plan");
    }

    #[test]
    fn validate_plan_pending_falls_back_to_strict_when_no_questions() {
        let plan = Plan::new("p", "proj", vec![]);
        let err = validate_plan_pending(&plan).unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("no steps")));
    }

    #[test]
    fn invalidate_decision_drops_target_and_transitive_dependents() {
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.decisions.push(decision("q1", &[]));
        plan.decisions.push(decision("q2", &["q1"]));
        plan.decisions.push(decision("q3", &["q2"]));
        plan.decisions.push(decision("q4", &[])); // unrelated; must survive

        let removed = plan.invalidate_decision("q1");
        assert_eq!(removed, vec!["q1", "q2", "q3"]);
        assert_eq!(plan.decisions.len(), 1);
        assert_eq!(plan.decisions[0].question_id, "q4");
        assert!(plan.definition_dirty);
    }

    #[test]
    fn invalidate_decision_unknown_id_is_noop_and_does_not_dirty() {
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.decisions.push(decision("q1", &[]));
        let removed = plan.invalidate_decision("ghost");
        assert!(removed.is_empty());
        assert_eq!(plan.decisions.len(), 1);
        assert!(!plan.definition_dirty);
    }

    #[test]
    fn apply_compile_output_with_definition_clears_dirty_and_replaces_steps() {
        use loader::{PlanDefinition, StepDefinition};
        let mut plan = Plan::new("p", "proj", vec![step("old", &[])]);
        plan.definition_dirty = true;
        plan.open_questions.push(question("q1", &[]));

        let def = PlanDefinition {
            name: "p".into(),
            project: Some("proj".into()),
            project_root: None,
            steps: vec![StepDefinition {
                id: "new".into(),
                agent_type: "echo".into(),
                prompt: "hi".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        };
        plan.apply_compile_output(compiler::CompileOutput::ready("# done", def));

        assert!(!plan.definition_dirty);
        assert!(plan.open_questions.is_empty());
        assert_eq!(plan.steps.len(), 1);
        assert_eq!(plan.steps[0].id, "new");
        assert_eq!(plan.source_doc.as_deref(), Some("# done"));
    }

    #[test]
    fn apply_compile_output_preserves_prior_timeout_for_matching_step_ids() {
        // Round 1: operator authored a step with a deadline, then a
        // refine round (the structured-markdown stub or LLM compiler)
        // re-emits the same id but drops `timeout_secs`. The plan must
        // keep the prior deadline so the executor doesn't silently fall
        // back to the daemon-wide default.
        use loader::{PlanDefinition, StepDefinition};

        let mut prior_step = step("keeper", &[]);
        prior_step.timeout_secs = Some(120);
        let mut plan = Plan::new("p", "proj", vec![prior_step, step("dropped", &[])]);

        let refined = PlanDefinition {
            name: "p".into(),
            project: Some("proj".into()),
            project_root: None,
            steps: vec![
                StepDefinition {
                    id: "keeper".into(),
                    agent_type: "echo".into(),
                    prompt: "still here".into(),
                    depends_on: vec![],
                    timeout_secs: None, // compiler dropped it
                    retry_count: 0,
                },
                StepDefinition {
                    id: "fresh".into(), // brand-new step, no prior to inherit
                    agent_type: "echo".into(),
                    prompt: "added".into(),
                    depends_on: vec![],
                    timeout_secs: None,
                    retry_count: 0,
                },
            ],
        };
        plan.apply_compile_output(compiler::CompileOutput::ready("# refined", refined));

        let keeper = plan
            .steps
            .iter()
            .find(|s| s.id == "keeper")
            .expect("keeper step preserved");
        assert_eq!(
            keeper.timeout_secs,
            Some(120),
            "prior timeout_secs survived the refine round"
        );
        let fresh = plan
            .steps
            .iter()
            .find(|s| s.id == "fresh")
            .expect("fresh step present");
        assert_eq!(
            fresh.timeout_secs, None,
            "new step keeps None when no prior timeout to inherit"
        );
    }

    #[test]
    fn apply_compile_output_lets_explicit_timeout_override_prior() {
        use loader::{PlanDefinition, StepDefinition};
        let mut prior = step("a", &[]);
        prior.timeout_secs = Some(30);
        let mut plan = Plan::new("p", "proj", vec![prior]);

        let def = PlanDefinition {
            name: "p".into(),
            project: Some("proj".into()),
            project_root: None,
            steps: vec![StepDefinition {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "hi".into(),
                depends_on: vec![],
                timeout_secs: Some(900), // explicit override wins
                retry_count: 0,
            }],
        };
        plan.apply_compile_output(compiler::CompileOutput::ready("# done", def));
        assert_eq!(plan.steps[0].timeout_secs, Some(900));
    }

    #[test]
    fn apply_compile_output_without_definition_sets_dirty() {
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.apply_compile_output(compiler::CompileOutput::pending(
            "# round 1",
            vec![question("q1", &[])],
        ));
        assert!(plan.definition_dirty);
        assert_eq!(plan.open_questions.len(), 1);
    }

    #[test]
    fn plan_round_trips_with_populated_clarification_fields() {
        let mut plan = Plan::new("p", "proj", vec![step("a", &[])]);
        plan.source_doc = Some("# Plan\n\n1. do a thing".into());
        plan.open_questions.push(OpenQuestion {
            id: "q1".into(),
            prompt: "Pick".into(),
            rationale: "why".into(),
            options: vec![QuestionOption {
                id: "o1".into(),
                label: "Option 1".into(),
                rationale: "small".into(),
            }],
            allow_explain_more: true,
            allow_none: false,
            depends_on_decisions: vec![],
        });
        plan.decisions
            .push(DecisionRecord::new_user("q0", "yes", vec![]));
        plan.definition_dirty = true;

        let json = serde_json::to_string(&plan).unwrap();
        let back: Plan = serde_json::from_str(&json).unwrap();

        assert_eq!(back.source_doc.as_deref(), Some("# Plan\n\n1. do a thing"));
        assert_eq!(back.open_questions.len(), 1);
        assert_eq!(back.open_questions[0].id, "q1");
        assert_eq!(back.decisions.len(), 1);
        assert_eq!(back.decisions[0].chosen_option_id, "yes");
        assert!(back.definition_dirty);
    }

    #[test]
    fn plan_new_leaves_project_root_unset() {
        // `project_root` is opt-in: callers that don't know the
        // execution root (tests, programmatic consumers) should not see
        // a fabricated value.
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        assert!(plan.project_root.is_none());
    }

    #[test]
    fn with_project_root_round_trips_through_serde() {
        let plan = Plan::new("p", "proj", vec![step("a", &[])]).with_project_root("/repos/flowd");

        let json = serde_json::to_string(&plan).unwrap();
        let back: Plan = serde_json::from_str(&json).unwrap();

        assert_eq!(plan.project_root.as_deref(), Some("/repos/flowd"));
        assert_eq!(back.project_root.as_deref(), Some("/repos/flowd"));
        assert_eq!(back.project, "proj");
    }

    #[test]
    fn with_project_root_treats_blank_as_unset() {
        // Pipe-through ergonomics: callers that route a possibly empty
        // env var through the builder should not poison the field with
        // whitespace.
        let plan = Plan::new("p", "proj", vec![step("a", &[])]).with_project_root("   \t\n");
        assert!(plan.project_root.is_none());
    }

    /// Pre-`project_root` plan JSON must still deserialise cleanly so
    /// existing rows in the `plans.definition` blob keep loading after
    /// the field is added. Mirrors the legacy-fallback test that guards
    /// the clarification fields.
    #[test]
    fn plan_deserialises_legacy_json_without_project_root_field() {
        let plan = Plan::new("p", "proj", vec![step("a", &[])]);
        let mut json: serde_json::Value = serde_json::to_value(&plan).unwrap();
        json.as_object_mut().unwrap().remove("project_root");

        let restored: Plan = serde_json::from_value(json).unwrap();
        assert!(
            restored.project_root.is_none(),
            "missing project_root must surface as None, the legacy fallback"
        );
        assert_eq!(restored.project, "proj");
        assert_eq!(restored.steps.len(), 1);
    }

    /// `project` (namespace label) and `project_root` (filesystem path)
    /// are independent: two plans can share a project label while
    /// pointing at different execution roots, and the same root can be
    /// reused across project labels. The validator only enforces a
    /// non-empty project; `project_root` has no such requirement.
    #[test]
    fn project_and_project_root_are_independent_fields() {
        let alpha =
            Plan::new("a", "shared", vec![step("a", &[])]).with_project_root("/repos/repo-a");
        let beta =
            Plan::new("b", "shared", vec![step("a", &[])]).with_project_root("/repos/repo-b");

        assert_eq!(alpha.project, beta.project);
        assert_ne!(alpha.project_root, beta.project_root);

        // The strict validator must accept both regardless of root --
        // it's an execution hint, not a structural constraint.
        validate_plan(&alpha).expect("plan validates without project_root constraint");
        validate_plan(&beta).expect("plan validates with a different project_root");

        // Conversely, identical roots across distinct project labels
        // also round-trip cleanly: the two fields don't collapse.
        let same_root_a =
            Plan::new("a", "alpha", vec![step("a", &[])]).with_project_root("/repos/shared");
        let same_root_b =
            Plan::new("b", "beta", vec![step("a", &[])]).with_project_root("/repos/shared");
        assert_eq!(same_root_a.project_root, same_root_b.project_root);
        assert_ne!(same_root_a.project, same_root_b.project);
    }

    #[test]
    fn validate_plan_ignores_project_root_value() {
        // `project_root = None` is the legacy-load case; the validator
        // must treat it as a perfectly valid plan rather than refusing
        // it the way it does for an empty `project` label.
        let no_root = Plan::new("p", "proj", vec![step("a", &[])]);
        validate_plan(&no_root).expect("missing project_root is allowed");

        let with_root =
            Plan::new("p", "proj", vec![step("a", &[])]).with_project_root("/repos/flowd");
        validate_plan(&with_root).expect("explicit project_root is allowed");
    }

    // ---- resolve_workspace_root ------------------------------------------

    /// Mutex serialising tests that touch the process-global env or cwd.
    /// `cargo test` runs cases in parallel within a binary, so two tests
    /// concurrently flipping `FLOWD_WORKSPACE_ROOT` would race.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    fn make_git_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("flowd-resolve-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::create_dir_all(dir.join(".git")).unwrap();
        dir
    }

    fn make_non_git_dir(tag: &str) -> std::path::PathBuf {
        let dir =
            std::env::temp_dir().join(format!("flowd-resolve-bare-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    /// Hand-roll an env-var guard: thread-local `Drop`s aren't enough
    /// because `std::env` is process-global. Restoring on drop keeps a
    /// failed assertion from leaking state into the next test.
    struct EnvGuard {
        key: &'static str,
        prior: Option<std::ffi::OsString>,
    }

    impl EnvGuard {
        // SAFETY: callers hold ENV_LOCK so no concurrent test mutates env.
        #[allow(unsafe_code)]
        fn set(key: &'static str, value: &std::path::Path) -> Self {
            let prior = std::env::var_os(key);
            unsafe { std::env::set_var(key, value) };
            Self { key, prior }
        }

        #[allow(unsafe_code)]
        fn unset(key: &'static str) -> Self {
            let prior = std::env::var_os(key);
            unsafe { std::env::remove_var(key) };
            Self { key, prior }
        }
    }

    impl Drop for EnvGuard {
        // SAFETY: same as set/unset; the test held ENV_LOCK for its lifetime.
        #[allow(unsafe_code)]
        fn drop(&mut self) {
            unsafe {
                match &self.prior {
                    Some(v) => std::env::set_var(self.key, v),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    #[test]
    fn resolve_workspace_root_prefers_client_hint_over_env_and_cwd() {
        let _g = ENV_LOCK.lock().unwrap();
        let hint_dir = make_git_dir("hint");
        let env_dir = make_git_dir("env");
        let _env = EnvGuard::set(FLOWD_WORKSPACE_ROOT_ENV, &env_dir);

        let (resolved, src) = resolve_workspace_root(Some(hint_dir.to_str().unwrap()))
            .expect("client hint should win");
        assert_eq!(src, WorkspaceRootSource::ClientHint);
        // The hint dir is in /tmp which symlinks to /private/tmp on macOS;
        // canonicalize both sides so the equality check matches what the
        // resolver returns.
        let expected = std::fs::canonicalize(&hint_dir)
            .unwrap()
            .display()
            .to_string();
        assert_eq!(resolved, expected);
    }

    #[test]
    fn resolve_workspace_root_falls_back_to_env_when_hint_missing() {
        let _g = ENV_LOCK.lock().unwrap();
        let env_dir = make_git_dir("env-only");
        let _env = EnvGuard::set(FLOWD_WORKSPACE_ROOT_ENV, &env_dir);

        let (resolved, src) = resolve_workspace_root(None).expect("env should win");
        assert_eq!(src, WorkspaceRootSource::EnvOverride);
        let expected = std::fs::canonicalize(&env_dir)
            .unwrap()
            .display()
            .to_string();
        assert_eq!(resolved, expected);
    }

    #[test]
    fn resolve_workspace_root_falls_back_to_cwd_when_no_hint_or_env() {
        let _g = ENV_LOCK.lock().unwrap();
        let _env = EnvGuard::unset(FLOWD_WORKSPACE_ROOT_ENV);
        // We don't touch cwd; the test binary runs inside the flowd
        // repo, which is itself a git repository, so the cwd fallback
        // must succeed and report the cwd source.
        let (resolved, src) = resolve_workspace_root(None).expect("cwd fallback inside flowd repo");
        assert_eq!(src, WorkspaceRootSource::Cwd);
        let cwd_canonical = std::fs::canonicalize(std::env::current_dir().unwrap())
            .unwrap()
            .display()
            .to_string();
        assert_eq!(resolved, cwd_canonical);
    }

    #[test]
    fn resolve_workspace_root_skips_non_git_hint_and_falls_through() {
        let _g = ENV_LOCK.lock().unwrap();
        let bare = make_non_git_dir("hint-no-git");
        let env_dir = make_git_dir("env-rescue");
        let _env = EnvGuard::set(FLOWD_WORKSPACE_ROOT_ENV, &env_dir);

        let (_resolved, src) = resolve_workspace_root(Some(bare.to_str().unwrap()))
            .expect("env should rescue the resolution");
        assert_eq!(
            src,
            WorkspaceRootSource::EnvOverride,
            "non-git hint must be rejected so the env override gets a turn"
        );
    }

    #[test]
    fn resolve_workspace_root_errors_when_every_candidate_rejects() {
        let _g = ENV_LOCK.lock().unwrap();
        let bare_hint = make_non_git_dir("err-hint");
        let bare_env = make_non_git_dir("err-env");
        let _env = EnvGuard::set(FLOWD_WORKSPACE_ROOT_ENV, &bare_env);

        // We can't trivially make the test process's cwd non-git (the
        // repo holds the binary), so use a hint + env where both are
        // outside any git tree. The cwd fallback will still succeed
        // because it lives inside the flowd repo, but the error path
        // we care about is exercised by passing `Err` candidates that
        // every prior signal rejects -- so swap that test to ensure
        // `try_workspace_candidate` rejects independently.
        let hint_err = try_workspace_candidate(bare_hint.to_str().unwrap()).unwrap_err();
        assert!(
            hint_err.contains("not inside a git repository"),
            "{hint_err}"
        );
        let env_err = try_workspace_candidate(bare_env.to_str().unwrap()).unwrap_err();
        assert!(env_err.contains("not inside a git repository"), "{env_err}");
    }

    #[test]
    fn resolve_workspace_root_returns_canonical_form_of_subdirectory() {
        // Subdirectories within a repo must resolve to the subdirectory's
        // canonical path, not the repo root: that way two plans created
        // from the same workspace path compare equal even when authors
        // are working several directories below the repo root.
        let _g = ENV_LOCK.lock().unwrap();
        let repo = make_git_dir("subdir-parent");
        let sub = repo.join("nested").join("workspace");
        std::fs::create_dir_all(&sub).unwrap();

        let (resolved, src) =
            resolve_workspace_root(Some(sub.to_str().unwrap())).expect("subdir resolves");
        assert_eq!(src, WorkspaceRootSource::ClientHint);
        let expected = std::fs::canonicalize(&sub).unwrap().display().to_string();
        assert_eq!(resolved, expected);
    }

    #[test]
    fn resolve_workspace_root_treats_blank_hint_as_absent() {
        // Whitespace-only hints must be skipped silently so callers can
        // forward an unset proxy variable through without branching.
        let _g = ENV_LOCK.lock().unwrap();
        let env_dir = make_git_dir("blank-hint");
        let _env = EnvGuard::set(FLOWD_WORKSPACE_ROOT_ENV, &env_dir);

        let (_, src) = resolve_workspace_root(Some("   ")).expect("env wins");
        assert_eq!(src, WorkspaceRootSource::EnvOverride);
    }

    #[test]
    fn canonicalize_project_root_round_trips_inside_git_repo() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = make_git_dir("canon");
        let canon = canonicalize_project_root(dir.to_str().unwrap()).unwrap();
        let expected = std::fs::canonicalize(&dir).unwrap().display().to_string();
        assert_eq!(canon, expected);
    }

    #[test]
    fn canonicalize_project_root_rejects_non_git_paths() {
        let _g = ENV_LOCK.lock().unwrap();
        let dir = make_non_git_dir("canon-bare");
        // The git-verification gate is the whole point of the new
        // resolver; the legacy-shaped helper inherits it so callers
        // cannot accidentally route around the check.
        assert!(canonicalize_project_root(dir.to_str().unwrap()).is_none());
    }
}
