//! End-to-end tests for the rules engine.
//!
//! Covers YAML file loading, glob scope matching, gate checking semantics
//! (warn vs deny), and context-injection via `matching_rules`.

#![allow(unsafe_code, clippy::undocumented_unsafe_blocks)]

use std::fs;
use std::path::PathBuf;

use flowd_core::error::{FlowdError, RuleLevel};
use flowd_core::rules::{
    InMemoryRuleEvaluator, ProposedAction, Rule, RuleEvaluator, StandardPaths,
};

fn temp_dir(prefix: &str) -> PathBuf {
    let mut path = std::env::temp_dir();
    path.push(format!(
        "{prefix}-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    fs::create_dir_all(&path).expect("create temp dir");
    path
}

#[test]
fn loads_multiple_yaml_files_from_dir() {
    let dir = temp_dir("flowd-rules-multi");

    fs::write(
        dir.join("a.yaml"),
        "id: a\nscope: \"**/*.rs\"\nlevel: warn\ndescription: a\nmatch: write\n",
    )
    .unwrap();
    fs::write(
        dir.join("b.yml"),
        "- id: b1\n  scope: \"**/*.rs\"\n  level: deny\n  description: b1\n  match: shell\n\
         - id: b2\n  scope: docs/**\n  level: warn\n  description: b2\n  match: publish\n",
    )
    .unwrap();
    fs::write(dir.join("README.md"), "ignored").unwrap();

    let mut ev = InMemoryRuleEvaluator::new();
    let loaded = ev.load_rules_from_dir(&dir).unwrap();
    assert_eq!(loaded, 3, "3 rules across 2 files");
    assert_eq!(ev.len(), 3);

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn duplicate_id_across_files_is_rejected() {
    let dir = temp_dir("flowd-rules-dup");

    fs::write(
        dir.join("one.yaml"),
        "id: same\nscope: \"**/*.rs\"\nlevel: warn\ndescription: one\nmatch: x\n",
    )
    .unwrap();
    fs::write(
        dir.join("two.yaml"),
        "id: same\nscope: \"**/*.rs\"\nlevel: deny\ndescription: two\nmatch: y\n",
    )
    .unwrap();

    let mut ev = InMemoryRuleEvaluator::new();
    let err = ev.load_rules_from_dir(&dir).unwrap_err();
    assert!(matches!(err, FlowdError::RuleLoad(msg) if msg.contains("duplicate")));

    fs::remove_dir_all(&dir).ok();
}

#[test]
fn gate_deny_blocks_and_warn_passes() {
    let mut ev = InMemoryRuleEvaluator::new();
    ev.register_rule(Rule {
        id: "no-shell-in-rs".into(),
        scope: "**/*.rs".into(),
        level: RuleLevel::Deny,
        description: "no shell from rust".into(),
        match_pattern: "shell_exec".into(),
    })
    .unwrap();
    ev.register_rule(Rule {
        id: "advise-write".into(),
        scope: "**/*.rs".into(),
        level: RuleLevel::Warn,
        description: "think before writing".into(),
        match_pattern: "file_write".into(),
    })
    .unwrap();

    let shell = ev.check(
        &ProposedAction::new("shell_exec")
            .with_file("src/main.rs")
            .with_project("flowd"),
    );
    assert!(!shell.allowed, "deny must block");
    assert!(shell.has_denials());

    let write = ev.check(
        &ProposedAction::new("file_write")
            .with_file("src/main.rs")
            .with_project("flowd"),
    );
    assert!(write.allowed, "warn must not block");
    assert!(write.has_warnings());
}

#[test]
fn matching_rules_feeds_context_injection() {
    let mut ev = InMemoryRuleEvaluator::new();
    ev.register_rule(Rule {
        id: "rust-style".into(),
        scope: "**/*.rs".into(),
        level: RuleLevel::Warn,
        description: "use ? over unwrap".into(),
        match_pattern: ".*".into(),
    })
    .unwrap();
    ev.register_rule(Rule {
        id: "docs-tone".into(),
        scope: "docs/**".into(),
        level: RuleLevel::Warn,
        description: "no marketing fluff".into(),
        match_pattern: ".*".into(),
    })
    .unwrap();

    let ctx = ev.matching_rules(Some("flowd"), Some("crates/flowd-core/src/lib.rs"));
    let ids: Vec<_> = ctx.iter().map(|r| r.id.as_str()).collect();
    assert_eq!(ids, vec!["rust-style"]);
}

#[test]
fn standard_paths_load_global_and_project() {
    let fake_home = temp_dir("flowd-rules-home");
    let fake_project = temp_dir("flowd-rules-project");

    let global = fake_home.join(".flowd").join("rules");
    fs::create_dir_all(&global).unwrap();
    fs::write(
        global.join("global.yaml"),
        "id: global-rule\nscope: \"**/*\"\nlevel: warn\ndescription: g\nmatch: x\n",
    )
    .unwrap();

    let project = fake_project.join(".flowd").join("rules");
    fs::create_dir_all(&project).unwrap();
    fs::write(
        project.join("local.yaml"),
        "id: local-rule\nscope: \"**/*\"\nlevel: deny\ndescription: l\nmatch: x\n",
    )
    .unwrap();

    let prev_home = std::env::var_os("HOME");
    // SAFETY: test is single-threaded.
    unsafe {
        std::env::set_var("HOME", &fake_home);
    }

    let paths = StandardPaths::resolve(Some(&fake_project));
    assert!(paths.global.is_some());
    assert!(paths.project.is_some());

    let mut ev = InMemoryRuleEvaluator::new();
    let loaded = flowd_core::rules::load_standard_paths(Some(&fake_project), |dir| {
        ev.load_rules_from_dir(dir)
    })
    .unwrap();
    assert_eq!(loaded, 2);
    assert!(ev.get("global-rule").is_some());
    assert!(ev.get("local-rule").is_some());

    // SAFETY: restoring.
    unsafe {
        match prev_home {
            Some(v) => std::env::set_var("HOME", v),
            None => std::env::remove_var("HOME"),
        }
    }

    fs::remove_dir_all(&fake_home).ok();
    fs::remove_dir_all(&fake_project).ok();
}
