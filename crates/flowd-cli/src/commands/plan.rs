//! `flowd plan` -- load a plan file, render a preview, and optionally
//! prompt the operator for Y/N confirmation.
//!
//! Execution itself is deferred: until a concrete `AgentSpawner` is wired
//! up (future issue) "confirming" can only print the plan id a daemon
//! would accept. We surface this clearly so users don't think plans run.

use std::collections::HashMap;
use std::io::{BufRead, Write};
use std::path::PathBuf;

use anyhow::{Context, Result};

use flowd_core::orchestration::{build_preview, load_plan};

use crate::output::{Style, banner};
use crate::paths::FlowdPaths;

pub async fn run(_paths: &FlowdPaths, style: Style, file: PathBuf, dry_run: bool) -> Result<()> {
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
}
