//! `flowd rules list` -- enumerate active rules for a given scope.
//!
//! Rules are loaded the same way `flowd start` loads them: global first
//! (`$FLOWD_HOME/rules/`), then project-local (`<project>/.flowd/rules/`
//! when a project root is detectable from cwd).

use anyhow::{Context, Result};

use flowd_core::rules::{InMemoryRuleEvaluator, RuleEvaluator};

use crate::output::{Style, banner};
use crate::paths::FlowdPaths;

pub fn list(
    paths: &FlowdPaths,
    style: Style,
    project: Option<&str>,
    file: Option<&str>,
) -> Result<()> {
    let mut eval = InMemoryRuleEvaluator::new();

    if paths.rules_dir().is_dir() {
        eval.load_rules_from_dir(&paths.rules_dir())
            .with_context(|| format!("load global rules from {}", paths.rules_dir().display()))?;
    }
    let cwd = std::env::current_dir().context("determine cwd")?;
    if let Some(root) = FlowdPaths::detect_project_root(&cwd) {
        let dir = root.join(".flowd").join("rules");
        if dir.is_dir() {
            eval.load_rules_from_dir(&dir)
                .with_context(|| format!("load project rules from {}", dir.display()))?;
        }
    }

    let matches = eval.matching_rules(project, file);

    let scope_label = match (project, file) {
        (Some(p), Some(f)) => format!("project=`{p}` file=`{f}`"),
        (Some(p), None) => format!("project=`{p}`"),
        (None, Some(f)) => format!("file=`{f}`"),
        (None, None) => "(all scopes)".to_owned(),
    };
    print!(
        "{}",
        banner(
            &format!(
                "rules matching {} ({} rule{})",
                scope_label,
                matches.len(),
                if matches.len() == 1 { "" } else { "s" }
            ),
            style,
        )
    );

    if matches.is_empty() {
        println!("  {}", style.dim("(no rules)"));
        return Ok(());
    }

    for rule in matches {
        let level = match rule.level {
            flowd_core::error::RuleLevel::Warn => style.yellow("warn"),
            flowd_core::error::RuleLevel::Deny => style.red("deny"),
        };
        println!(
            "  {id}  [{level}]  scope=`{scope}`  match=`{pat}`",
            id = style.bold(&rule.id),
            scope = rule.scope,
            pat = rule.match_pattern,
        );
        println!("    {}", rule.description);
    }
    Ok(())
}
