//! `flowd plan` -- inspect orchestration plans.
//!
//! Two subcommands:
//!
//! * `flowd plan preview <file>` -- load a plan file, render a preview,
//!   and optionally prompt for Y/N confirmation.
//! * `flowd plan events <plan_id>` -- print the persisted lifecycle event
//!   log for a plan, read straight from `SQLite` (HL-39).
//!
//! Execution itself is deferred: until a concrete `AgentSpawner` is wired
//! up (future issue) "confirming" can only print the plan id a daemon
//! would accept. We surface this clearly so users don't think plans run.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use flowd_core::orchestration::plan_events::{PlanEventQuery, PlanEventStore, StoredPlanEvent};
use flowd_core::orchestration::{build_preview, load_plan};
use flowd_storage::plan_event_store::SqlitePlanEventStore;
use uuid::Uuid;

use crate::output::{Style, banner};
use crate::paths::FlowdPaths;

pub async fn preview(
    _paths: &FlowdPaths,
    style: Style,
    file: PathBuf,
    dry_run: bool,
) -> Result<()> {
    let plan = load_plan(&file).with_context(|| format!("load plan {}", file.display()))?;
    let preview = build_preview(&plan).context("build plan preview")?;

    print!("{}", banner(&format!("plan: {}", preview.name), style));
    println!("  id:          {}", style.dim(&preview.plan_id.to_string()));
    println!("  steps:       {}", preview.total_steps);
    println!("  agents:      {}", preview.total_agents);
    println!("  parallelism: {}", preview.max_parallelism);

    render_layers(&preview.execution_order, style);
    render_dependencies(&preview.dependency_graph, style);

    if dry_run {
        println!("\n{} dry-run -- skipping confirmation", style.dim("note:"));
        return Ok(());
    }

    // Confirmation prompt. stdin is blocking, so we hop onto the blocking
    // thread pool to avoid stalling the tokio reactor.
    let confirmed = tokio::task::spawn_blocking(|| prompt_yes_no("proceed? [y/N] "))
        .await
        .context("confirmation prompt panicked")??;

    if confirmed {
        println!(
            "\n{} plan execution is not yet wired up in this CLI cut.",
            style.yellow("note:")
        );
        println!("       the daemon's `plan_confirm` MCP tool will execute plans once a concrete");
        println!("       `AgentSpawner` is provided -- tracked separately.");
    } else {
        println!("{}", style.dim("aborted."));
    }
    Ok(())
}

pub async fn events(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    limit: usize,
    kinds: Vec<String>,
) -> Result<()> {
    let plan_id = Uuid::parse_str(plan_id_arg.trim())
        .with_context(|| format!("parse plan id `{plan_id_arg}` as UUID"))?;

    // Open the same SQLite file the daemon writes to. WAL mode keeps
    // this read-side safe even while `flowd start` is live.
    let db_path = paths.db_file();
    if !db_path.exists() {
        anyhow::bail!(
            "no flowd database at {}; start the daemon at least once to initialise it",
            db_path.display()
        );
    }
    let store = SqlitePlanEventStore::open(&db_path)
        .with_context(|| format!("open plan event store at {}", db_path.display()))?;

    let kinds = normalise_kinds(kinds);
    let query = PlanEventQuery {
        kinds,
        limit: limit.max(1),
    };
    let rows = store
        .list_for_plan(plan_id, query)
        .await
        .with_context(|| format!("list events for plan {plan_id}"))?;

    print!("{}", banner(&format!("events: {plan_id}"), style));
    if rows.is_empty() {
        println!("  {}", style.dim("(no events recorded)"));
        return Ok(());
    }
    for evt in &rows {
        render_event(evt, style);
    }
    Ok(())
}

/// Flatten `--kind a,b --kind c` and `--kind a --kind b,c` into a single
/// trimmed-and-deduped list. The clap `value_delimiter = ','` handles
/// the splitting; this layer drops empty entries and surrounding
/// whitespace from operator typos.
fn normalise_kinds(kinds: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in kinds {
        let trimmed = k.trim();
        if trimmed.is_empty() {
            continue;
        }
        let owned = trimmed.to_owned();
        if !out.contains(&owned) {
            out.push(owned);
        }
    }
    out
}

fn render_event(evt: &StoredPlanEvent, style: Style) {
    let ts = evt.created_at.format("%Y-%m-%d %H:%M:%SZ");
    let header = format!(
        "{ts}  {kind}",
        ts = style.dim(&ts.to_string()),
        kind = style.cyan(&evt.kind),
    );
    let suffix = match (evt.step_id.as_deref(), evt.agent_type.as_deref()) {
        (Some(step), Some(agent)) => format!("  step={step}  agent={agent}"),
        (Some(step), None) => format!("  step={step}"),
        _ => String::new(),
    };
    println!("  {header}{suffix}");
    if let Some(detail) = describe_payload(&evt.kind, &evt.payload) {
        println!("    {}", style.dim(&detail));
    }
}

/// One-line human summary of the event payload. Falls back to the raw
/// JSON for anything we don't recognise.
fn describe_payload(kind: &str, payload: &serde_json::Value) -> Option<String> {
    use flowd_core::orchestration::plan_events::kind as k;

    match kind {
        k::SUBMITTED => payload
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(|n| format!("name: {n}")),
        k::STEP_COMPLETED => payload
            .get("output")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("output: {}", truncate_one_line(s, 120))),
        k::STEP_FAILED => payload
            .get("error")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("error: {s}")),
        k::STEP_REFUSED => payload
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("reason: {s}")),
        k::FINISHED => payload
            .get("status")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("status: {s}")),
        _ => None,
    }
}

fn truncate_one_line(s: &str, max: usize) -> String {
    let single = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.len() <= max {
        single
    } else {
        // Use a char-boundary safe slice: walk char_indices to find the
        // largest cut <= max - 1 (room for the ellipsis).
        let target = max.saturating_sub(1);
        let mut cut = 0usize;
        for (i, _) in single.char_indices() {
            if i > target {
                break;
            }
            cut = i;
        }
        format!("{}…", &single[..cut])
    }
}

fn render_layers(layers: &[Vec<String>], style: Style) {
    println!("\n{}", style.bold("execution layers:"));
    if layers.is_empty() {
        println!("  {}", style.dim("(none)"));
        return;
    }
    for (i, layer) in layers.iter().enumerate() {
        println!(
            "  {label} {steps}",
            label = style.cyan(&format!("L{i}:")),
            steps = layer.join(", "),
        );
    }
}

fn render_dependencies(graph: &HashMap<String, Vec<String>>, style: Style) {
    println!("\n{}", style.bold("dependencies:"));
    // Stable ordering: keys in lexical order, so output is deterministic
    // across hash-map iteration variance.
    let mut entries: Vec<_> = graph.iter().filter(|(_, v)| !v.is_empty()).collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    if entries.is_empty() {
        println!("  {}", style.dim("(none)"));
        return;
    }
    for (id, deps) in entries {
        println!("  {id} ← {}", deps.join(", "));
    }
}

fn prompt_yes_no(prompt: &str) -> Result<bool> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(prompt.as_bytes())?;
    out.flush()?;

    let stdin = std::io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    let answer = line.trim().to_ascii_lowercase();
    Ok(matches!(answer.as_str(), "y" | "yes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_empty_layers_without_panicking() {
        render_layers(&[], Style::plain());
    }

    #[test]
    fn renders_empty_graph_without_panicking() {
        render_dependencies(&HashMap::new(), Style::plain());
    }

    #[test]
    fn normalise_kinds_drops_empty_and_dedupes() {
        let out = normalise_kinds(vec![
            "step_failed".into(),
            " ".into(),
            "step_failed".into(),
            "step_completed".into(),
            String::new(),
        ]);
        assert_eq!(out, vec!["step_failed", "step_completed"]);
    }

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate_one_line("hello", 10), "hello");
    }

    #[test]
    fn truncate_collapses_whitespace_then_cuts() {
        let out = truncate_one_line("a\nb   c", 32);
        assert_eq!(out, "a b c");
    }

    #[test]
    fn truncate_appends_ellipsis_when_over_limit() {
        let out = truncate_one_line(&"x".repeat(200), 10);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 10);
    }
}
