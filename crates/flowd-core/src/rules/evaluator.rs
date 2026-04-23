//! In-memory concrete implementation of [`RuleEvaluator`].
//!
//! Rules are validated and compiled once at registration: the scope glob
//! becomes a `glob::Pattern`, and the match regex becomes a `regex::Regex`.
//! Per-call evaluation reuses those compiled artifacts, avoiding repeated
//! compilation on every `matching_rules` / `check` invocation.
//!
//! Registration enforces rule-ID uniqueness; re-registering the same ID is
//! rejected rather than silently overwriting so that operators notice when
//! project-local and global rule files collide.

use std::path::Path;

use glob::Pattern as GlobPattern;
use regex::Regex;

use crate::error::{FlowdError, Result, RuleLevel};

use super::loader::{list_rule_yaml_files_under, load_rules_file};
use super::{GateResult, ProposedAction, Rule, RuleEvaluator, RuleViolation};

/// Internal store: the authored rule plus its compiled artifacts.
#[derive(Debug)]
struct CompiledRule {
    rule: Rule,
    scope: GlobPattern,
    matcher: Regex,
}

/// Default rules engine.
///
/// Hold-and-evaluate: loaded rules live in a `Vec`; matching is a linear
/// scan. The expected rule count is small (tens, not thousands) so a
/// hash-indexed dispatch would be premature.
#[derive(Debug, Default)]
pub struct InMemoryRuleEvaluator {
    rules: Vec<CompiledRule>,
}

impl InMemoryRuleEvaluator {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of rules currently loaded.
    #[must_use]
    pub fn len(&self) -> usize {
        self.rules.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.rules.is_empty()
    }

    /// Borrow every authored rule.
    pub fn rules(&self) -> impl Iterator<Item = &Rule> {
        self.rules.iter().map(|c| &c.rule)
    }

    /// Look up a rule by ID.
    #[must_use]
    pub fn get(&self, id: &str) -> Option<&Rule> {
        self.rules.iter().find(|c| c.rule.id == id).map(|c| &c.rule)
    }

    /// Remove a rule by ID. Returns `true` if a rule was removed.
    pub fn remove(&mut self, id: &str) -> bool {
        let len_before = self.rules.len();
        self.rules.retain(|c| c.rule.id != id);
        self.rules.len() != len_before
    }

    fn compile(rule: Rule) -> Result<CompiledRule> {
        let scope = GlobPattern::new(&rule.scope).map_err(|e| {
            FlowdError::RuleLoad(format!(
                "rule `{}`: invalid scope glob `{}`: {e}",
                rule.id, rule.scope
            ))
        })?;
        let matcher = Regex::new(&rule.match_pattern).map_err(|e| {
            FlowdError::RuleLoad(format!(
                "rule `{}`: invalid match regex `{}`: {e}",
                rule.id, rule.match_pattern
            ))
        })?;
        Ok(CompiledRule {
            rule,
            scope,
            matcher,
        })
    }

    fn is_in_scope(
        compiled: &CompiledRule,
        project: Option<&str>,
        file_path: Option<&str>,
    ) -> bool {
        let project_hit = project.is_some_and(|p| compiled.scope.matches(p));
        let file_hit = file_path.is_some_and(|f| compiled.scope.matches(f));
        project_hit || file_hit
    }
}

impl RuleEvaluator for InMemoryRuleEvaluator {
    fn load_rules_from_dir(&mut self, dir: &Path) -> Result<usize> {
        let paths = list_rule_yaml_files_under(dir)?;

        let mut loaded = 0usize;
        for path in paths {
            let rules = load_rules_file(&path)?;
            for rule in rules {
                self.register_rule(rule)?;
                loaded += 1;
            }
        }
        Ok(loaded)
    }

    fn register_rule(&mut self, rule: Rule) -> Result<()> {
        if self.rules.iter().any(|c| c.rule.id == rule.id) {
            return Err(FlowdError::RuleLoad(format!(
                "duplicate rule id `{}`",
                rule.id
            )));
        }
        let compiled = Self::compile(rule)?;
        self.rules.push(compiled);
        Ok(())
    }

    fn matching_rules(&self, project: Option<&str>, file_path: Option<&str>) -> Vec<&Rule> {
        self.rules
            .iter()
            .filter(|c| Self::is_in_scope(c, project, file_path))
            .map(|c| &c.rule)
            .collect()
    }

    fn check(&self, action: &ProposedAction) -> GateResult {
        let mut violations = Vec::new();

        for compiled in &self.rules {
            if !Self::is_in_scope(
                compiled,
                action.project.as_deref(),
                action.file_path.as_deref(),
            ) {
                continue;
            }

            let tool_hit = compiled.matcher.is_match(&action.tool);
            let file_hit = action
                .file_path
                .as_deref()
                .is_some_and(|fp| compiled.matcher.is_match(fp));

            if tool_hit || file_hit {
                violations.push(RuleViolation {
                    rule_id: compiled.rule.id.clone(),
                    level: compiled.rule.level,
                    description: compiled.rule.description.clone(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    fn rule(id: &str, scope: &str, level: RuleLevel, match_pattern: &str) -> Rule {
        Rule {
            id: id.to_owned(),
            scope: scope.to_owned(),
            level,
            description: format!("desc-{id}"),
            match_pattern: match_pattern.to_owned(),
        }
    }

    #[test]
    fn register_and_lookup() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("r1", "**/*.rs", RuleLevel::Warn, "shell_exec"))
            .unwrap();
        assert_eq!(ev.len(), 1);
        assert_eq!(ev.get("r1").unwrap().id, "r1");
    }

    #[test]
    fn rejects_duplicate_id() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("r1", "**/*.rs", RuleLevel::Warn, "x"))
            .unwrap();
        let err = ev
            .register_rule(rule("r1", "**/*.rs", RuleLevel::Warn, "x"))
            .unwrap_err();
        assert!(matches!(err, FlowdError::RuleLoad(_)));
    }

    #[test]
    fn rejects_bad_glob_and_regex() {
        let mut ev = InMemoryRuleEvaluator::new();
        let bad_glob = rule("bg", "[invalid", RuleLevel::Warn, "ok");
        assert!(matches!(
            ev.register_rule(bad_glob).unwrap_err(),
            FlowdError::RuleLoad(_)
        ));

        let bad_regex = rule("br", "**/*.rs", RuleLevel::Warn, "(unbalanced");
        assert!(matches!(
            ev.register_rule(bad_regex).unwrap_err(),
            FlowdError::RuleLoad(_)
        ));
    }

    #[test]
    fn matching_rules_filters_by_scope_glob() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("rust", "**/*.rs", RuleLevel::Warn, "x"))
            .unwrap();
        ev.register_rule(rule("docs", "docs/**", RuleLevel::Warn, "x"))
            .unwrap();
        ev.register_rule(rule("proj", "flowd", RuleLevel::Warn, "x"))
            .unwrap();

        let hits = ev.matching_rules(Some("flowd"), Some("src/lib.rs"));
        let ids: Vec<_> = hits.iter().map(|r| r.id.as_str()).collect();
        assert!(ids.contains(&"rust"));
        assert!(ids.contains(&"proj"));
        assert!(!ids.contains(&"docs"));
    }

    #[test]
    fn check_allows_when_no_match() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("r", "**/*.rs", RuleLevel::Deny, "shell_exec"))
            .unwrap();

        let result = ev.check(
            &ProposedAction::new("file_read")
                .with_file("src/lib.rs")
                .with_project("flowd"),
        );
        assert!(result.allowed);
        assert!(result.violations.is_empty());
    }

    #[test]
    fn check_denies_when_regex_matches_tool() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule(
            "no-shell",
            "**/*.rs",
            RuleLevel::Deny,
            "shell_exec|bash_command",
        ))
        .unwrap();

        let result = ev.check(
            &ProposedAction::new("shell_exec")
                .with_file("src/lib.rs")
                .with_project("flowd"),
        );
        assert!(!result.allowed);
        assert_eq!(result.violations.len(), 1);
        assert_eq!(result.violations[0].rule_id, "no-shell");
    }

    #[test]
    fn check_warns_do_not_block() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("advisory", "**/*.rs", RuleLevel::Warn, "write"))
            .unwrap();

        let result = ev.check(
            &ProposedAction::new("file_write")
                .with_file("src/lib.rs")
                .with_project("flowd"),
        );
        assert!(result.allowed, "warn-level must not block");
        assert!(result.has_warnings());
        assert!(!result.has_denials());
    }

    #[test]
    fn check_ignores_rules_outside_scope() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("ts-only", "**/*.ts", RuleLevel::Deny, "shell_exec"))
            .unwrap();

        let result = ev.check(
            &ProposedAction::new("shell_exec")
                .with_file("src/lib.rs")
                .with_project("flowd"),
        );
        assert!(result.allowed, "rule out of scope must not trigger");
    }

    #[test]
    fn remove_returns_true_once() {
        let mut ev = InMemoryRuleEvaluator::new();
        ev.register_rule(rule("r", "**/*.rs", RuleLevel::Warn, "x"))
            .unwrap();
        assert!(ev.remove("r"));
        assert!(!ev.remove("r"));
    }

    fn temp_rules_dir(prefix: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let unique = format!(
            "{prefix}-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        );
        path.push(unique);
        fs::create_dir_all(&path).expect("temp rules root");
        path
    }

    #[test]
    fn load_rules_nested_yaml_is_loaded() {
        let root = temp_rules_dir("flowd-rules-nested");
        let nested = root.join("a").join("b");
        fs::create_dir_all(&nested).unwrap();
        let file = nested.join("c.yaml");
        fs::write(
            &file,
            "id: nested-c\nscope: \"**\"\nlevel: warn\ndescription: d\nmatch: m\n",
        )
        .unwrap();

        let mut ev = InMemoryRuleEvaluator::new();
        let n = ev.load_rules_from_dir(&root).unwrap();
        assert_eq!(n, 1);
        assert_eq!(ev.get("nested-c").unwrap().id, "nested-c");

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn load_rules_lexicographic_order_across_subdirs() {
        let root = temp_rules_dir("flowd-rules-lex");
        let b = root.join("b");
        let a = root.join("a");
        fs::create_dir_all(&b).unwrap();
        fs::create_dir_all(&a).unwrap();
        fs::write(
            b.join("r.yaml"),
            "id: rule-b\nscope: \"**\"\nlevel: warn\ndescription: d\nmatch: m\n",
        )
        .unwrap();
        fs::write(
            a.join("r.yaml"),
            "id: rule-a\nscope: \"**\"\nlevel: warn\ndescription: d\nmatch: m\n",
        )
        .unwrap();

        let mut ev = InMemoryRuleEvaluator::new();
        ev.load_rules_from_dir(&root).unwrap();
        let ids: Vec<_> = ev.rules().map(|r| r.id.as_str()).collect();
        assert_eq!(ids, vec!["rule-a", "rule-b"]);

        fs::remove_dir_all(&root).ok();
    }

    #[cfg(unix)]
    #[test]
    fn load_rules_symlink_dir_not_followed() {
        use std::os::unix::fs::symlink;

        let mut base = std::env::temp_dir();
        base.push(format!(
            "flowd-rules-symlink-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        ));
        let rules_root = base.join("rules");
        let outside = base.join("outside_target");
        fs::create_dir_all(&rules_root).unwrap();
        fs::create_dir_all(&outside).unwrap();
        fs::write(
            outside.join("inside.yaml"),
            "id: inside\nscope: \"**\"\nlevel: warn\ndescription: d\nmatch: m\n",
        )
        .unwrap();
        symlink(&outside, rules_root.join("via_link")).unwrap();

        let mut ev = InMemoryRuleEvaluator::new();
        let n = ev.load_rules_from_dir(&rules_root).unwrap();
        assert_eq!(n, 0, "must not traverse symlinked directory");
        assert!(ev.get("inside").is_none());

        fs::remove_dir_all(&base).ok();
    }

    #[test]
    fn load_rules_malformed_nested_yaml_errors_with_path() {
        let root = temp_rules_dir("flowd-rules-bad-nested");
        let nested = root.join("x").join("y");
        fs::create_dir_all(&nested).unwrap();
        let bad = nested.join("bad.yaml");
        fs::write(&bad, ":::not-yaml:::").unwrap();

        let mut ev = InMemoryRuleEvaluator::new();
        let err = ev.load_rules_from_dir(&root).unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("parse") && msg.contains(&*bad.to_string_lossy()),
            "expected path-pointing parse error, got {msg}"
        );

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn load_rules_skips_non_yaml_at_depth() {
        let root = temp_rules_dir("flowd-rules-skip-txt");
        let deep = root.join("deep");
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("notes.txt"), "noise").unwrap();
        fs::write(
            deep.join("keep.yaml"),
            "id: keep\nscope: \"**\"\nlevel: warn\ndescription: d\nmatch: m\n",
        )
        .unwrap();

        let mut ev = InMemoryRuleEvaluator::new();
        let n = ev.load_rules_from_dir(&root).unwrap();
        assert_eq!(n, 1);
        assert!(ev.get("keep").is_some());

        fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn load_repo_dot_flowd_rules_flat_layout_regression() {
        let rules_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../.flowd/rules");
        assert!(
            rules_dir.is_dir(),
            "expected {} (repo .flowd/rules)",
            rules_dir.display()
        );

        let mut ev = InMemoryRuleEvaluator::new();
        let n = ev.load_rules_from_dir(&rules_dir).unwrap();
        assert!(
            n >= 7,
            "expected all rules from flat .flowd/rules yaml files, got {n}"
        );
        assert!(ev.get("cargo-no-build").is_some());
        assert!(ev.get("flowd-core-no-io-deps").is_some());
        assert!(ev.get("forward-only-migrations").is_some());
        assert!(ev.get("mcp-wire-discipline").is_some());
        assert!(ev.get("hooks-swallow-errors-by-design").is_some());
    }
}
