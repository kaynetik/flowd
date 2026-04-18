//! YAML loading helpers for the rules engine.
//!
//! Two file layouts are accepted:
//!
//! ```yaml
//! # single-rule layout
//! id: no-shell-in-tests
//! scope: "**/*.test.rs"
//! level: deny
//! description: "Never shell out from test code"
//! match: "shell_exec|bash_command"
//! ```
//!
//! ```yaml
//! # list layout
//! - id: rule-a
//!   scope: "**/*.rs"
//!   level: warn
//!   description: "…"
//!   match: "…"
//! - id: rule-b
//!   scope: "docs/**"
//!   level: deny
//!   description: "…"
//!   match: "…"
//! ```
//!
//! The loader itself performs no scope/regex validation -- that responsibility
//! belongs to [`super::evaluator::InMemoryRuleEvaluator::register_rule`], which
//! pre-compiles both patterns once at registration time.

use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;

use crate::error::{FlowdError, Result};

use super::Rule;

/// Either a single rule or a sequence of rules, as authored on disk.
#[derive(Deserialize)]
#[serde(untagged)]
enum RuleFile {
    Single(Rule),
    List(Vec<Rule>),
}

/// Parse a single rule file, returning every rule it declares.
///
/// Accepts either a single rule mapping or a YAML sequence of rules.
///
/// # Errors
/// Returns `FlowdError::RuleLoad` if the file cannot be read or the YAML is
/// malformed.
pub fn load_rules_file(path: &Path) -> Result<Vec<Rule>> {
    let contents = fs::read_to_string(path)
        .map_err(|e| FlowdError::RuleLoad(format!("read {}: {e}", path.display())))?;

    let parsed: RuleFile = serde_yaml::from_str(&contents)
        .map_err(|e| FlowdError::RuleLoad(format!("parse {}: {e}", path.display())))?;

    Ok(match parsed {
        RuleFile::Single(rule) => vec![rule],
        RuleFile::List(rules) => rules,
    })
}

/// Standard locations from which rules are loaded.
///
/// By convention both directories may exist simultaneously; project-local
/// rules augment the global set rather than overriding it (rule-id uniqueness
/// is enforced by the evaluator).
#[derive(Debug, Clone)]
pub struct StandardPaths {
    /// `~/.flowd/rules/` -- global, user-level rules.
    pub global: Option<PathBuf>,
    /// `.flowd/rules/` resolved relative to a project root -- per-repo rules.
    pub project: Option<PathBuf>,
}

impl StandardPaths {
    /// Resolve the conventional rule directories for a given project root.
    ///
    /// The global directory is `$HOME/.flowd/rules/`; the project directory
    /// is `<project_root>/.flowd/rules/`. Entries that do not exist on disk
    /// are returned as `None` so callers can decide whether to treat the
    /// absence as an error.
    #[must_use]
    pub fn resolve(project_root: Option<&Path>) -> Self {
        let global = home_dir()
            .map(|h| h.join(".flowd").join("rules"))
            .filter(|p| p.is_dir());

        let project = project_root
            .map(|root| root.join(".flowd").join("rules"))
            .filter(|p| p.is_dir());

        Self { global, project }
    }

    /// Iterate over the resolved directories that actually exist.
    pub fn iter(&self) -> impl Iterator<Item = &Path> {
        self.global
            .as_deref()
            .into_iter()
            .chain(self.project.as_deref())
    }
}

/// Resolve standard paths and feed each into the supplied loader closure.
///
/// Returns the total number of rules loaded across all scopes. The callback
/// is invoked once per existing directory; typical usage is
/// `load_standard_paths(project_root, |dir| evaluator.load_rules_from_dir(dir))`.
///
/// # Errors
/// Propagates any error returned by the callback.
pub fn load_standard_paths<F>(project_root: Option<&Path>, mut loader: F) -> Result<usize>
where
    F: FnMut(&Path) -> Result<usize>,
{
    let paths = StandardPaths::resolve(project_root);
    let mut total = 0usize;
    for dir in paths.iter() {
        total = total.saturating_add(loader(dir)?);
    }
    Ok(total)
}

/// Cross-platform home-directory lookup without pulling in an extra crate.
///
/// Checks `$HOME` first (POSIX), then `$USERPROFILE` (Windows). Returns
/// `None` if neither variable is set.
fn home_dir() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .map(PathBuf::from)
}

#[cfg(test)]
#[allow(unsafe_code, clippy::undocumented_unsafe_blocks)]
mod tests {
    use super::*;
    use crate::error::RuleLevel;

    fn temp_dir_with_prefix(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "{prefix}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        );
        path.push(unique);
        fs::create_dir_all(&path).expect("temp dir");
        path
    }

    #[test]
    fn parses_single_rule_file() {
        let dir = temp_dir_with_prefix("flowd-rules-single");
        let file = dir.join("r.yaml");
        fs::write(
            &file,
            "id: r1\nscope: \"**/*.rs\"\nlevel: warn\ndescription: d\nmatch: tool\n",
        )
        .unwrap();

        let rules = load_rules_file(&file).unwrap();
        assert_eq!(rules.len(), 1);
        assert_eq!(rules[0].id, "r1");
        assert_eq!(rules[0].level, RuleLevel::Warn);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn parses_list_rule_file() {
        let dir = temp_dir_with_prefix("flowd-rules-list");
        let file = dir.join("r.yaml");
        fs::write(
            &file,
            "- id: a\n  scope: \"**/*.rs\"\n  level: warn\n  description: d\n  match: m\n\
             - id: b\n  scope: \"docs/**\"\n  level: deny\n  description: d\n  match: m\n",
        )
        .unwrap();

        let rules = load_rules_file(&file).unwrap();
        assert_eq!(rules.len(), 2);
        assert_eq!(rules[1].level, RuleLevel::Deny);

        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn malformed_yaml_is_error() {
        let dir = temp_dir_with_prefix("flowd-rules-bad");
        let file = dir.join("r.yaml");
        fs::write(&file, ":::not-yaml:::").unwrap();
        let err = load_rules_file(&file).unwrap_err();
        assert!(matches!(err, FlowdError::RuleLoad(_)));
        fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn resolve_ignores_missing_dirs() {
        let fake_home = temp_dir_with_prefix("flowd-rules-resolve-home");
        let fake_project = temp_dir_with_prefix("flowd-rules-resolve-project");

        let prev_home = std::env::var_os("HOME");
        // SAFETY: single-threaded test, we restore prior value before returning.
        unsafe {
            std::env::set_var("HOME", &fake_home);
        }
        let paths = StandardPaths::resolve(Some(&fake_project));
        assert!(paths.global.is_none(), "no .flowd/rules under fake home");
        assert!(
            paths.project.is_none(),
            "no .flowd/rules under fake project"
        );

        let global_rules = fake_home.join(".flowd").join("rules");
        fs::create_dir_all(&global_rules).unwrap();
        let project_rules = fake_project.join(".flowd").join("rules");
        fs::create_dir_all(&project_rules).unwrap();

        let paths = StandardPaths::resolve(Some(&fake_project));
        assert_eq!(paths.global.as_deref(), Some(global_rules.as_path()));
        assert_eq!(paths.project.as_deref(), Some(project_rules.as_path()));

        // SAFETY: restoring prior env var, still single-threaded.
        unsafe {
            match prev_home {
                Some(v) => std::env::set_var("HOME", v),
                None => std::env::remove_var("HOME"),
            }
        }

        fs::remove_dir_all(&fake_home).ok();
        fs::remove_dir_all(&fake_project).ok();
    }
}
