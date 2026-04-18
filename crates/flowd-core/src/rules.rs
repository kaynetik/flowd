//! Rules engine traits and types.
//!
//! Two enforcement mechanisms:
//! 1. Context injection: matching rules are appended to the auto-context
//!    returned by the memory subsystem so the LLM is reminded of relevant
//!    advisory rules before it acts.
//! 2. Gate checking: proposed actions are validated against loaded rules
//!    before execution. `warn` violations are advisory; `deny` violations
//!    block the action.
//!
//! Concrete implementation: [`evaluator::InMemoryRuleEvaluator`].
//! YAML loading helpers: [`loader`].

pub mod evaluator;
pub mod loader;

use crate::error::{Result, RuleLevel};
use serde::{Deserialize, Serialize};

pub use evaluator::InMemoryRuleEvaluator;
pub use loader::{StandardPaths, load_rules_file, load_standard_paths};

/// A rule definition as authored on disk or registered via MCP.
///
/// The `scope` field is a glob pattern (e.g. `**/*.rs`, `my-project/**`) that
/// determines which project/file contexts a rule applies to. The `match`
/// field is a regular expression evaluated against the proposed action's
/// tool name or file path during gate checking.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Rule {
    pub id: String,
    /// Glob pattern for file/project scope (e.g. `**/*.rs`, `my-project/**`).
    pub scope: String,
    pub level: RuleLevel,
    pub description: String,
    /// Regex pattern matched against tool names or file paths.
    #[serde(rename = "match")]
    pub match_pattern: String,
}

/// Outcome of a gate check against loaded rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateResult {
    /// `false` if at least one `deny`-level rule matched.
    pub allowed: bool,
    pub violations: Vec<RuleViolation>,
}

impl GateResult {
    #[must_use]
    pub fn has_warnings(&self) -> bool {
        self.violations.iter().any(|v| v.level == RuleLevel::Warn)
    }

    #[must_use]
    pub fn has_denials(&self) -> bool {
        self.violations.iter().any(|v| v.level == RuleLevel::Deny)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RuleViolation {
    pub rule_id: String,
    pub level: RuleLevel,
    pub description: String,
}

/// Action descriptor submitted for gate checking.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProposedAction {
    /// Tool name being invoked (e.g. `file_write`, `shell_exec`).
    pub tool: String,
    /// Target file path, if applicable.
    pub file_path: Option<String>,
    /// Current project context.
    pub project: Option<String>,
}

impl ProposedAction {
    #[must_use]
    pub fn new(tool: impl Into<String>) -> Self {
        Self {
            tool: tool.into(),
            file_path: None,
            project: None,
        }
    }

    #[must_use]
    pub fn with_file(mut self, file_path: impl Into<String>) -> Self {
        self.file_path = Some(file_path.into());
        self
    }

    #[must_use]
    pub fn with_project(mut self, project: impl Into<String>) -> Self {
        self.project = Some(project.into());
        self
    }
}

/// Evaluates rules against proposed actions and retrieves matching rules
/// for context injection.
pub trait RuleEvaluator: Send + Sync {
    /// Load every rule file (`*.yaml` / `*.yml`) in `dir`.
    ///
    /// Returns the number of rules registered on success.
    ///
    /// # Errors
    /// Returns `FlowdError::RuleLoad` if the directory cannot be read or any
    /// YAML file fails to parse / validate.
    fn load_rules_from_dir(&mut self, dir: &std::path::Path) -> Result<usize>;

    /// Register a single rule programmatically (e.g. via MCP tool).
    ///
    /// # Errors
    /// Returns `FlowdError::RuleLoad` if the rule ID conflicts with an
    /// existing rule or if the scope glob / match regex is malformed.
    fn register_rule(&mut self, rule: Rule) -> Result<()>;

    /// Find all rules whose scope glob matches the given project or file
    /// path. Used for context injection before gate evaluation.
    fn matching_rules(&self, project: Option<&str>, file_path: Option<&str>) -> Vec<&Rule>;

    /// Gate check: validate a proposed action against every rule currently
    /// in scope. The default implementation recompiles regexes per call;
    /// concrete types should override this for efficiency.
    fn check(&self, action: &ProposedAction) -> GateResult {
        let rules = self.matching_rules(action.project.as_deref(), action.file_path.as_deref());

        let mut violations = Vec::new();
        for rule in rules {
            let Ok(pattern) = regex::Regex::new(&rule.match_pattern) else {
                continue;
            };

            let matches = pattern.is_match(&action.tool)
                || action
                    .file_path
                    .as_deref()
                    .is_some_and(|fp| pattern.is_match(fp));

            if matches {
                violations.push(RuleViolation {
                    rule_id: rule.id.clone(),
                    level: rule.level,
                    description: rule.description.clone(),
                });
            }
        }

        let allowed = !violations.iter().any(|v| v.level == RuleLevel::Deny);

        GateResult {
            allowed,
            violations,
        }
    }
}
