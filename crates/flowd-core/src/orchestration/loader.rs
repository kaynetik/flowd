//! YAML / JSON parsing for orchestration plans.
//!
//! The on-disk format ([`PlanDefinition`]) carries only authored data;
//! runtime fields (`id`, `status`, timestamps, per-step state) are filled
//! in by [`PlanDefinition::into_plan`] when the plan is materialised.
//!
//! Example YAML:
//!
//! ```yaml
//! name: nightly-recap
//! steps:
//!   - id: gather
//!     agent: summarizer
//!     prompt: "Collect today's commits"
//!     timeout_secs: 60
//!   - id: review
//!     agent: reviewer
//!     prompt: "Critique the gathered notes"
//!     depends_on: [gather]
//!     retry_count: 2
//! ```

use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::error::{FlowdError, Result};

use super::{Plan, PlanStep, StepStatus};

/// Authored shape of a plan -- everything an operator types into a YAML/JSON
/// file. Distinct from [`Plan`] so authors don't have to fill in runtime
/// fields like `status` or `created_at`.
///
/// `project` is required at the type level so on-disk plans stay
/// self-describing. The MCP `plan_create` tool accepts the project
/// separately and overlays it onto a definition that omits the field, so
/// callers materialising a plan programmatically do not have to embed the
/// project in the YAML/JSON when their workflow already supplies it
/// out-of-band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanDefinition {
    pub name: String,
    /// Project namespace this plan belongs to. Optional in the on-disk
    /// schema only so callers (e.g. the MCP `plan_create` handler) may
    /// supply it via a separate parameter; [`Self::into_plan`] requires
    /// it to be present at materialisation time.
    #[serde(default)]
    pub project: Option<String>,
    pub steps: Vec<StepDefinition>,
}

/// Authored shape of a single step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepDefinition {
    pub id: String,
    /// Which agent type (binary / role) handles this step.
    #[serde(alias = "agent_type", alias = "agent")]
    pub agent_type: String,
    pub prompt: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub timeout_secs: Option<u64>,
    #[serde(default)]
    pub retry_count: u32,
}

impl PlanDefinition {
    /// Materialise the authored definition into a runtime [`Plan`] in the
    /// `Draft` state.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` if `project` was not supplied
    /// in the definition. Use [`Self::into_plan_with_project`] when the
    /// project is provided out-of-band (e.g. via `plan_create` MCP args).
    pub fn into_plan(self) -> Result<Plan> {
        let project = self.project.clone().ok_or_else(|| {
            FlowdError::PlanValidation("plan definition is missing required `project` field".into())
        })?;
        Ok(self.into_plan_with_project(project))
    }

    /// Materialise the authored definition with an externally-supplied
    /// project. Any `project` field embedded in the definition is
    /// overridden by the explicit argument so the call site (typically a
    /// trusted handler) is the source of truth.
    #[must_use]
    pub fn into_plan_with_project(self, project: impl Into<String>) -> Plan {
        let steps = self
            .steps
            .into_iter()
            .map(|s| PlanStep {
                id: s.id,
                agent_type: s.agent_type,
                prompt: s.prompt,
                depends_on: s.depends_on,
                timeout_secs: s.timeout_secs,
                retry_count: s.retry_count,
                status: StepStatus::Pending,
                output: None,
                error: None,
                started_at: None,
                completed_at: None,
            })
            .collect();
        Plan::new(self.name, project, steps)
    }
}

/// Parse a plan from a string, choosing format by sniffing the first
/// non-whitespace character: `{` or `[` → JSON, otherwise YAML.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` on parse failure.
pub fn parse_plan_str(text: &str) -> Result<Plan> {
    let trimmed = text.trim_start();
    let definition: PlanDefinition = if trimmed.starts_with('{') || trimmed.starts_with('[') {
        serde_json::from_str(text)
            .map_err(|e| FlowdError::PlanValidation(format!("invalid JSON plan: {e}")))?
    } else {
        serde_yaml::from_str(text)
            .map_err(|e| FlowdError::PlanValidation(format!("invalid YAML plan: {e}")))?
    };
    definition.into_plan()
}

/// Load a plan, dispatching on the file extension (`.json` → JSON,
/// `.yaml`/`.yml` → YAML, anything else → content sniffing).
///
/// # Errors
/// Returns `FlowdError::PlanValidation` if the file cannot be read or parsed.
pub fn load_plan(path: &Path) -> Result<Plan> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase);
    match ext.as_deref() {
        Some("json") => load_plan_json(path),
        Some("yaml" | "yml") => load_plan_yaml(path),
        _ => {
            let text = read_to_string(path)?;
            parse_plan_str(&text)
        }
    }
}

/// Load a plan from a YAML file.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` if the file cannot be read or parsed.
pub fn load_plan_yaml(path: &Path) -> Result<Plan> {
    let text = read_to_string(path)?;
    let definition: PlanDefinition = serde_yaml::from_str(&text)
        .map_err(|e| FlowdError::PlanValidation(format!("parse {}: {e}", path.display())))?;
    definition.into_plan()
}

/// Load a plan from a JSON file.
///
/// # Errors
/// Returns `FlowdError::PlanValidation` if the file cannot be read or parsed.
pub fn load_plan_json(path: &Path) -> Result<Plan> {
    let text = read_to_string(path)?;
    let definition: PlanDefinition = serde_json::from_str(&text)
        .map_err(|e| FlowdError::PlanValidation(format!("parse {}: {e}", path.display())))?;
    definition.into_plan()
}

fn read_to_string(path: &Path) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|e| FlowdError::PlanValidation(format!("read {}: {e}", path.display())))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_YAML: &str = r"
name: sample
project: demo
steps:
  - id: a
    agent: echo
    prompt: hi
  - id: b
    agent: echo
    prompt: hi
    depends_on: [a]
    timeout_secs: 5
    retry_count: 1
";

    const SAMPLE_JSON: &str = r#"{
  "name": "sample",
  "project": "demo",
  "steps": [
    {"id": "a", "agent_type": "echo", "prompt": "hi"},
    {"id": "b", "agent_type": "echo", "prompt": "hi", "depends_on": ["a"]}
  ]
}"#;

    #[test]
    fn parses_yaml_with_agent_alias() {
        let plan = parse_plan_str(SAMPLE_YAML).unwrap();
        assert_eq!(plan.project, "demo");
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[1].depends_on, vec!["a".to_owned()]);
        assert_eq!(plan.steps[1].timeout_secs, Some(5));
        assert_eq!(plan.steps[1].retry_count, 1);
    }

    #[test]
    fn parses_json() {
        let plan = parse_plan_str(SAMPLE_JSON).unwrap();
        assert_eq!(plan.steps.len(), 2);
        assert_eq!(plan.steps[0].agent_type, "echo");
    }

    #[test]
    fn into_plan_yields_draft_with_pending_steps() {
        let def = PlanDefinition {
            name: "p".into(),
            project: Some("demo".into()),
            steps: vec![StepDefinition {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "x".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        };
        let plan = def.into_plan().unwrap();
        assert_eq!(plan.status, super::super::PlanStatus::Draft);
        assert_eq!(plan.project, "demo");
        assert_eq!(plan.steps[0].status, StepStatus::Pending);
    }

    #[test]
    fn into_plan_rejects_missing_project() {
        let def = PlanDefinition {
            name: "p".into(),
            project: None,
            steps: vec![StepDefinition {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "x".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        };
        let err = def.into_plan().unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("project")));
    }

    #[test]
    fn into_plan_with_project_overrides_definition() {
        let def = PlanDefinition {
            name: "p".into(),
            project: Some("ignored".into()),
            steps: vec![StepDefinition {
                id: "a".into(),
                agent_type: "echo".into(),
                prompt: "x".into(),
                depends_on: vec![],
                timeout_secs: None,
                retry_count: 0,
            }],
        };
        let plan = def.into_plan_with_project("override");
        assert_eq!(plan.project, "override");
    }
}
