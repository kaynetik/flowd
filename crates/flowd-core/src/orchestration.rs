//! Orchestration engine: plans, validation, execution.
//!
//! A `Plan` is a DAG of `PlanStep`s. The engine drives plans through a small
//! state machine: `Draft` → `Confirmed` → `Running` → `Completed` / `Failed`,
//! with `Cancelled` reachable from `Confirmed` or `Running`.
//!
//! This module owns the trait surface and pure-data types. Concrete pieces:
//!
//! * Authored input format + parsers: [`loader`].
//! * In-process executor that supervises agents: [`executor`].
//!
//! `flowd-core` deliberately stays I/O-framework-free; spawning real agent
//! processes is done by an [`executor::AgentSpawner`] implementation supplied
//! by the CLI / MCP layer.

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

pub use executor::{AgentOutput, AgentSpawner, InMemoryPlanExecutor};
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
    pub steps: Vec<PlanStep>,
    pub status: PlanStatus,
    pub created_at: DateTime<Utc>,
    pub started_at: Option<DateTime<Utc>>,
    pub completed_at: Option<DateTime<Utc>>,
}

fn default_legacy_project() -> String {
    "__legacy__".to_owned()
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

    /// Cancel an in-flight plan; in-flight steps are aborted.
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
    /// `FlowdError::PlanExecution` if the plan is in a terminal state.
    fn resume_plan(&self, plan_id: Uuid) -> impl Future<Output = Result<()>> + Send;
}

impl Plan {
    /// Construct a `Draft` plan from authored data.
    ///
    /// `project` is required because every downstream subsystem (rule
    /// evaluator, plan observer, persistence layer) treats it as a
    /// non-null namespace key.
    #[must_use]
    pub fn new(name: impl Into<String>, project: impl Into<String>, steps: Vec<PlanStep>) -> Self {
        Self {
            id: Uuid::new_v4(),
            name: name.into(),
            project: project.into(),
            steps,
            status: PlanStatus::Draft,
            created_at: Utc::now(),
            started_at: None,
            completed_at: None,
        }
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
}

/// Validate plan structure independently of any executor.
///
/// Checks:
/// - No duplicate step IDs.
/// - Every `depends_on` entry references a real step.
/// - No cycles (delegated to `Plan::execution_layers`).
///
/// # Errors
/// Returns `FlowdError::PlanValidation` describing the first violation.
pub fn validate_plan(plan: &Plan) -> Result<()> {
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
}
