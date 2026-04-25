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
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use flowd_core::orchestration::plan_events::{PlanEventQuery, PlanEventStore, StoredPlanEvent};
use flowd_core::orchestration::{
    Answer, Plan, PlanCompiler, PlanDraftSnapshot, PlanStatus, PlanStore, PlanSummary,
    build_preview, load_plan,
};
use flowd_storage::plan_event_store::SqlitePlanEventStore;
use flowd_storage::plan_store::SqlitePlanStore;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::FlowdConfig;
use crate::daemon;
use crate::output::{Style, banner};
use crate::paths::FlowdPaths;
use crate::plan_compiler::DaemonPlanCompiler;

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

pub async fn list(
    paths: &FlowdPaths,
    style: Style,
    project: Option<String>,
    status: Option<String>,
    limit: Option<usize>,
) -> Result<()> {
    let store = open_plan_store(paths)?;
    let status = parse_plan_status_filter(status)?;
    let mut rows = store
        .list_plans(project.as_deref())
        .await
        .context("list persisted plans")?;
    rows = filter_plan_summaries(rows, status, limit);
    render_plan_summaries(&rows, style, "plans");
    Ok(())
}

pub async fn show(paths: &FlowdPaths, style: Style, plan_id_arg: String) -> Result<()> {
    let store = open_plan_store(paths)?;
    let plan_id = parse_plan_id(&plan_id_arg)?;
    let plan = store
        .load_plan(plan_id)
        .await
        .with_context(|| format!("load plan {plan_id}"))?
        .ok_or_else(|| anyhow!("plan {plan_id} not found"))?;
    print_plan_summary(&plan, style, "snapshot");
    Ok(())
}

pub async fn recent(
    paths: &FlowdPaths,
    style: Style,
    project: Option<String>,
    status: Option<String>,
    limit: usize,
) -> Result<()> {
    let store = open_plan_store(paths)?;
    let status = parse_plan_status_filter(status)?;
    let mut rows = store
        .list_plans(project.as_deref())
        .await
        .context("list recent persisted plans")?;
    rows = filter_plan_summaries(rows, status, Some(limit));
    render_plan_summaries(&rows, style, "recent plans");
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
    use flowd_core::orchestration::plan_events::kind as k;

    let ts = evt.created_at.format("%Y-%m-%d %H:%M:%SZ");
    let header = format!(
        "{ts}  {kind}",
        ts = style.dim(&ts.to_string()),
        kind = style.cyan(&evt.kind),
    );
    // `finished` hoists its status into the header line so the per-plan
    // rollup reads as a single self-contained block; per-step events keep
    // the id/agent suffix they had before.
    let suffix = if evt.kind == k::FINISHED {
        evt.payload
            .get("status")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("  status={s}"))
            .unwrap_or_default()
    } else {
        match (evt.step_id.as_deref(), evt.agent_type.as_deref()) {
            (Some(step), Some(agent)) => format!("  step={step}  agent={agent}"),
            (Some(step), None) => format!("  step={step}"),
            _ => String::new(),
        }
    };
    println!("  {header}{suffix}");
    // Finished events render their own purpose-built rollup below; skip
    // the generic `describe_payload` fallback that would duplicate the
    // status we already put on the header line.
    if evt.kind != k::FINISHED {
        if let Some(detail) = describe_payload(&evt.kind, &evt.payload) {
            println!("    {}", style.dim(&detail));
        }
    }
    if matches!(evt.kind.as_str(), k::STEP_COMPLETED | k::STEP_FAILED) {
        if let Some(lines) = format_metrics_block(&evt.payload) {
            for line in lines {
                println!("    {}", style.dim(&line));
            }
        }
    }
    if evt.kind == k::FINISHED {
        for line in format_finished_rollup(&evt.payload) {
            println!("    {}", style.dim(&line));
        }
    }
}

/// Two-line summary of an agent's reported `metrics` payload: cost + token
/// breakdown on the first line, wall/api duration on the second. Returns
/// `None` when the payload has no `metrics` object at all -- older events
/// predate metrics capture and shouldn't render a bogus `$0.0000` line.
// Durations are u64 milliseconds. Loss only matters past 2^53 ms (~285k
// years); the per-step values we render are seconds-to-minutes.
#[allow(clippy::cast_precision_loss)]
fn format_metrics_block(payload: &serde_json::Value) -> Option<Vec<String>> {
    let m = payload.get("metrics")?;
    let u64_of = |k: &str| m.get(k).and_then(serde_json::Value::as_u64).unwrap_or(0);
    let f64_of = |k: &str| m.get(k).and_then(serde_json::Value::as_f64).unwrap_or(0.0);

    let input = u64_of("input_tokens");
    let output = u64_of("output_tokens");
    let cache_read = u64_of("cache_read_input_tokens");
    let cache_creation = u64_of("cache_creation_input_tokens");
    let cost = f64_of("total_cost_usd");
    let duration_ms = u64_of("duration_ms");
    let duration_api_ms = u64_of("duration_api_ms");

    let line1 = format!(
        "cost: ${cost:.4}   tokens: in {in_}   out {out}   cache_read {cr}   cache_creation {cc}",
        in_ = format_thousands(input),
        out = format_thousands(output),
        cr = format_thousands(cache_read),
        cc = format_thousands(cache_creation),
    );
    let line2 = format!(
        "duration: api {:.1}s / total {:.1}s",
        duration_api_ms as f64 / 1000.0,
        duration_ms as f64 / 1000.0,
    );
    Some(vec![line1, line2])
}

/// Rollup rendered beneath the `finished` event header: cost + tokens +
/// cache hit-rate + per-outcome counts on the first line, duration on the
/// second. Returns an empty `Vec` when neither `metrics` nor a
/// `step_count` block was persisted (pre-rollup event rows). Payload
/// shape is owned by `flowd_core::orchestration::plan_events::event_payload`
/// for the `Finished` variant.
//
// Cast lints, all ranged-safe:
// - `cast_precision_loss`: u64 ms / token counts well below 2^53.
// - `cast_possible_truncation` and `cast_sign_loss` on the `pct` cast:
//   the expression is `cache_read / cache_denom * 100` with
//   `cache_read <= cache_denom`, so the rounded value lies in `[0, 100]`
//   and fits trivially in u64 without sign loss.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]
fn format_finished_rollup(payload: &serde_json::Value) -> Vec<String> {
    use std::fmt::Write as _;

    let metrics = payload.get("metrics");
    let step_count = payload.get("step_count");
    if metrics.is_none() && step_count.is_none() {
        return Vec::new();
    }
    let u64_of = |v: Option<&serde_json::Value>, k: &str| -> u64 {
        v.and_then(|m| m.get(k))
            .and_then(serde_json::Value::as_u64)
            .unwrap_or(0)
    };
    let f64_of = |v: Option<&serde_json::Value>, k: &str| -> f64 {
        v.and_then(|m| m.get(k))
            .and_then(serde_json::Value::as_f64)
            .unwrap_or(0.0)
    };

    let cost = f64_of(metrics, "total_cost_usd");
    let input = u64_of(metrics, "input_tokens");
    let output = u64_of(metrics, "output_tokens");
    let cache_read = u64_of(metrics, "cache_read_input_tokens");
    let cache_creation = u64_of(metrics, "cache_creation_input_tokens");
    let duration_ms = u64_of(metrics, "duration_ms");
    let duration_api_ms = u64_of(metrics, "duration_api_ms");
    let completed = u64_of(step_count, "completed");
    let failed = u64_of(step_count, "failed");

    let mut first = format!(
        "total: ${cost:.2}   in {in_}   out {out}",
        in_ = format_tokens_compact(input),
        out = format_tokens_compact(output),
    );
    let cache_denom = cache_read + cache_creation;
    if cache_denom > 0 {
        // Integer percent to keep output stable: a trailing `.0` on a
        // dashboard number is visual noise, and fractional precision
        // changes nothing about the signal the operator acts on.
        let pct = (cache_read as f64 / cache_denom as f64 * 100.0).round() as u64;
        write!(first, "   cache hit-rate {pct}%").expect("write to String is infallible");
    }
    write!(first, "   completed {completed}   failed {failed}")
        .expect("write to String is infallible");

    let second = format!(
        "duration: api {:.1}s / total {:.1}s",
        duration_api_ms as f64 / 1000.0,
        duration_ms as f64 / 1000.0,
    );

    vec![first, second]
}

/// Token-count formatter that switches to a compact `k` suffix once the
/// value crosses 10,000 -- high-cardinality rollup numbers stay readable
/// without widening the line, while per-step events (which usually stay
/// well under that cap) keep their thousands separators.
// `n` is a token count; loss only matters past 2^53 tokens, which no
// realistic agent run will reach.
#[allow(clippy::cast_precision_loss)]
fn format_tokens_compact(n: u64) -> String {
    if n > 9_999 {
        // Always one decimal place, e.g. 12_400 -> "12.4k", 100_000 -> "100.0k".
        format!("{:.1}k", n as f64 / 1000.0)
    } else {
        format_thousands(n)
    }
}

/// Render `n` with ASCII thousands separators (e.g. `15_820 -> "15,820"`).
/// Hand-rolled rather than pulling in `num-format` -- cheap enough for a
/// single-digit-count per event, and keeps the CLI's dependency graph flat.
fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let len = bytes.len();
    let mut out = String::with_capacity(len + (len.saturating_sub(1)) / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (len - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
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

/// `flowd plan answer` -- offline-mode counterpart to the `plan_answer`
/// MCP tool. Reads answers (a JSON array of `PlanAnswerEntry`-shaped
/// objects) from `file` (or stdin when `file == Some("-")`), invokes
/// the configured compiler, and persists the resulting draft.
///
/// Refuses to run when the daemon holds a live PID file: the only safe
/// way to mutate plan state with the daemon up is through the same
/// MCP wiring every other client uses, so we exit with a clear pointer
/// rather than racing against the in-memory executor.
pub async fn answer(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    file: Option<PathBuf>,
    defer_remaining: bool,
) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan answer")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let entries: Vec<CliAnswerEntry> = if let Some(path) = file.as_ref() {
        let raw = read_text_input(path).context("read --file payload")?;
        if raw.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&raw).context("parse --file as a JSON answers array")?
        }
    } else {
        Vec::new()
    };
    if entries.is_empty() && !defer_remaining {
        bail!(
            "plan answer: pass --file with at least one answer or --defer-remaining; both omitted"
        );
    }

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_draft(plan_id).await?;
    let snapshot = PlanDraftSnapshot::from_plan(&plan);
    let answers: Vec<(String, Answer)> = entries
        .into_iter()
        .map(|e| (e.question_id, e.answer))
        .collect();

    let output = runner
        .compiler
        .apply_answers(snapshot, answers, defer_remaining)
        .await
        .context("compile answers")?;

    plan.apply_compile_output(output);
    runner.save(&plan).await?;
    print_plan_summary(&plan, style, "answers applied");
    Ok(())
}

/// `flowd plan refine` -- offline-mode counterpart to the `plan_refine`
/// MCP tool. Same daemon-liveness guard as [`answer`].
pub async fn refine(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    feedback: Option<String>,
    file: Option<PathBuf>,
) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan refine")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let feedback = match (feedback, file) {
        (Some(s), None) => s,
        (None, Some(p)) => read_text_input(&p).context("read --file feedback")?,
        (None, None) => bail!("plan refine: pass either --feedback <STRING> or --file <PATH>"),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this combination"),
    };
    if feedback.trim().is_empty() {
        bail!("plan refine: feedback must be non-empty");
    }

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_draft(plan_id).await?;
    let prior_open: std::collections::HashSet<String> =
        plan.open_questions.iter().map(|q| q.id.clone()).collect();
    let snapshot = PlanDraftSnapshot::from_plan(&plan);

    let output = runner
        .compiler
        .refine(snapshot, feedback)
        .await
        .context("compile refinement")?;
    plan.apply_compile_output(output);
    runner.save(&plan).await?;

    let reopened = plan
        .open_questions
        .iter()
        .any(|q| !prior_open.contains(&q.id))
        || plan.open_questions.len() > prior_open.len();
    let header = if reopened {
        "refinement applied -- clarification reopened"
    } else {
        "refinement applied"
    };
    print_plan_summary(&plan, style, header);
    Ok(())
}

/// `flowd plan cancel` -- offline-mode counterpart to the `plan_cancel`
/// MCP tool. Same daemon-liveness guard. Cancellation of a plan that
/// is already terminal (Completed / Failed / Cancelled) is a silent
/// no-op so scripts can call this idempotently.
pub async fn cancel(paths: &FlowdPaths, style: Style, plan_id_arg: String) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan cancel")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_existing(plan_id).await?;

    match plan.status {
        PlanStatus::Completed | PlanStatus::Failed | PlanStatus::Cancelled => {
            println!(
                "{} plan {} already in {:?}; nothing to do",
                style.dim("noop:"),
                plan.id,
                plan.status,
            );
            return Ok(());
        }
        PlanStatus::Running => {
            // Without the daemon, no executor is holding handles to
            // abort -- the running task lives in the dead daemon's
            // address space. Refuse rather than silently flipping to
            // Cancelled and leaving the operator wondering why the
            // events log doesn't show any abort.
            bail!(
                "plan cancel: plan {} is Running but the daemon is offline; \
                 start the daemon and use the plan_cancel MCP tool, or \
                 wait for the executor task to exit before retrying",
                plan.id
            );
        }
        PlanStatus::Draft | PlanStatus::Confirmed | PlanStatus::Interrupted => {
            plan.status = PlanStatus::Cancelled;
            plan.completed_at = Some(chrono::Utc::now());
        }
    }

    runner.save(&plan).await?;
    println!("{} plan {} -> Cancelled", style.green("cancelled"), plan.id);
    Ok(())
}

/// Wire shape for `--file` payloads to `flowd plan answer`. Mirrors
/// `flowd_mcp::tools::PlanAnswerEntry` (flattened `Answer` enum), but
/// duplicated here so the CLI does not pull in `flowd-mcp` types just
/// for serde.
#[derive(Debug, Deserialize)]
struct CliAnswerEntry {
    question_id: String,
    #[serde(flatten)]
    answer: Answer,
}

/// Bundle of the dependencies the offline plan commands share.
struct OfflinePlanRunner {
    store: SqlitePlanStore,
    compiler: Arc<DaemonPlanCompiler>,
}

impl OfflinePlanRunner {
    fn open(paths: &FlowdPaths) -> Result<Self> {
        let store = open_plan_store(paths)?;

        let cfg_path = FlowdConfig::default_path(&paths.home);
        let config = FlowdConfig::load(&cfg_path)
            .with_context(|| format!("load flowd config from {}", cfg_path.display()))?;
        let compiler = DaemonPlanCompiler::from_selection(config.plan.compiler, &config.plan.llm)
            .map_err(|e| anyhow!("instantiate plan compiler: {e}"))?;

        Ok(Self {
            store,
            compiler: Arc::new(compiler),
        })
    }

    async fn load_existing(&self, plan_id: Uuid) -> Result<Plan> {
        self.store
            .load_plan(plan_id)
            .await
            .with_context(|| format!("load plan {plan_id}"))?
            .ok_or_else(|| anyhow!("plan {plan_id} not found"))
    }

    async fn load_draft(&self, plan_id: Uuid) -> Result<Plan> {
        let plan = self.load_existing(plan_id).await?;
        if plan.status != PlanStatus::Draft {
            bail!(
                "plan {plan_id} is not in Draft state (currently {:?}); \
                 only Draft plans accept clarification updates",
                plan.status
            );
        }
        Ok(plan)
    }

    async fn save(&self, plan: &Plan) -> Result<()> {
        self.store
            .save_plan(plan)
            .await
            .with_context(|| format!("save plan {}", plan.id))
    }
}

fn open_plan_store(paths: &FlowdPaths) -> Result<SqlitePlanStore> {
    let db_path = paths.db_file();
    if !db_path.exists() {
        bail!(
            "no flowd database at {}; start the daemon at least once to initialise it",
            db_path.display()
        );
    }
    SqlitePlanStore::open(&db_path)
        .with_context(|| format!("open plan store at {}", db_path.display()))
}

/// Refuse to operate on shared `SQLite` state when the daemon is alive.
/// The check is best-effort -- the PID file could go stale between
/// the probe and the first write -- but covers the common operator
/// mistake of running both surfaces against the same `$FLOWD_HOME`.
fn ensure_daemon_offline(paths: &FlowdPaths, style: Style, op: &str) -> Result<()> {
    let pid_path = paths.pid_file();
    let Some(pid) = daemon::read_pid(&pid_path)? else {
        return Ok(());
    };
    if !daemon::is_alive(pid) {
        return Ok(());
    }
    eprintln!(
        "{} the flowd daemon is running (pid {pid}, {pidfile}).",
        style.yellow(&format!("{op}: refused")),
        pidfile = pid_path.display(),
    );
    eprintln!("       offline `flowd plan` mutations would race the daemon's in-memory executor.");
    eprintln!(
        "       use the corresponding MCP tool (`{}`) instead, or run `flowd stop` first.",
        op.replace(' ', "_"),
    );
    bail!("{op}: daemon is alive");
}

fn parse_plan_id(raw: &str) -> Result<Uuid> {
    Uuid::parse_str(raw.trim()).with_context(|| format!("parse plan id `{raw}` as UUID"))
}

fn parse_plan_status_filter(raw: Option<String>) -> Result<Option<PlanStatus>> {
    raw.map(|s| parse_plan_status(&s)).transpose()
}

fn parse_plan_status(raw: &str) -> Result<PlanStatus> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "draft" => Ok(PlanStatus::Draft),
        "confirmed" => Ok(PlanStatus::Confirmed),
        "running" => Ok(PlanStatus::Running),
        "interrupted" => Ok(PlanStatus::Interrupted),
        "completed" => Ok(PlanStatus::Completed),
        "failed" => Ok(PlanStatus::Failed),
        "cancelled" => Ok(PlanStatus::Cancelled),
        other => bail!(
            "unknown plan status `{other}`; expected draft, confirmed, running, interrupted, completed, failed, or cancelled"
        ),
    }
}

fn filter_plan_summaries(
    mut rows: Vec<PlanSummary>,
    status: Option<PlanStatus>,
    limit: Option<usize>,
) -> Vec<PlanSummary> {
    if let Some(status) = status {
        rows.retain(|p| p.status == status);
    }
    if let Some(limit) = limit {
        rows.truncate(limit.max(1));
    }
    rows
}

/// Read text from a path; treat `-` as stdin so callers can pipe.
fn read_text_input(path: &std::path::Path) -> Result<String> {
    if path == std::path::Path::new("-") {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("read from stdin")?;
        return Ok(buf);
    }
    std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

fn render_plan_summaries(rows: &[PlanSummary], style: Style, title: &str) {
    print!("{}", banner(title, style));
    if rows.is_empty() {
        println!("  {}", style.dim("(no plans found)"));
        return;
    }
    for p in rows {
        println!(
            "  {}  {:<11}  {:<18}  {}",
            style.dim(&p.id.to_string()),
            plan_status_label(p.status),
            p.project,
            p.name,
        );
        println!(
            "    created: {}",
            style.dim(&p.created_at.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        );
    }
}

fn print_plan_summary(plan: &Plan, style: Style, header: &str) {
    print!(
        "{}",
        banner(&format!("plan: {} -- {}", plan.name, header), style)
    );
    println!("  id:               {}", style.dim(&plan.id.to_string()));
    println!("  status:           {}", plan_status_label(plan.status));
    println!("  project:          {}", plan.project);
    println!("  steps:            {}", plan.steps.len());
    println!("  open_questions:   {}", plan.open_questions.len());
    println!("  decisions:        {}", plan.decisions.len());
    println!("  definition_dirty: {}", plan.definition_dirty);

    if !plan.open_questions.is_empty() {
        println!("\n{}", style.bold("open questions:"));
        for q in &plan.open_questions {
            println!("  {} -- {}", style.cyan(&q.id), q.prompt);
        }
    }
    if !plan.decisions.is_empty() {
        println!("\n{}", style.bold("decisions:"));
        for d in &plan.decisions {
            let tag = if d.auto { " (auto)" } else { "" };
            println!(
                "  {} = {}{tag}",
                style.cyan(&d.question_id),
                d.chosen_option_id,
            );
        }
    }
    if !plan.steps.is_empty() && plan.open_questions.is_empty() {
        match build_preview(plan) {
            Ok(preview) => render_layers(&preview.execution_order, style),
            Err(e) => println!(
                "\n{} could not build preview: {e}",
                style.yellow("warning:")
            ),
        }
    }
}

fn plan_status_label(status: PlanStatus) -> &'static str {
    match status {
        PlanStatus::Draft => "draft",
        PlanStatus::Confirmed => "confirmed",
        PlanStatus::Running => "running",
        PlanStatus::Interrupted => "interrupted",
        PlanStatus::Completed => "completed",
        PlanStatus::Failed => "failed",
        PlanStatus::Cancelled => "cancelled",
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

    #[test]
    fn parse_plan_id_trims_whitespace() {
        let id = Uuid::new_v4();
        let raw = format!("  {id}  \n");
        let parsed = parse_plan_id(&raw).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn parse_plan_id_errors_on_garbage() {
        let err = parse_plan_id("not-a-uuid").unwrap_err();
        assert!(format!("{err:#}").contains("UUID"), "{err:#}");
    }

    #[test]
    fn read_text_input_reads_file() {
        let tmp =
            std::env::temp_dir().join(format!("flowd-plan-input-{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&tmp, "hello world\n").unwrap();
        let got = read_text_input(&tmp).unwrap();
        assert_eq!(got, "hello world\n");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn ensure_daemon_offline_passes_when_no_pid_file() {
        let dir = std::env::temp_dir().join(format!("flowd-plan-offline-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        ensure_daemon_offline(&paths, Style::plain(), "plan answer").unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_daemon_offline_passes_when_pid_is_stale() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-stale-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // PID 2^31-2 is essentially never live; the helper should treat
        // it as a stale pid file and let the command proceed.
        std::fs::write(dir.join("flowd.pid"), "2147483646\n").unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        ensure_daemon_offline(&paths, Style::plain(), "plan refine").unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_daemon_offline_refuses_when_pid_is_self() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-live-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("flowd.pid"), format!("{}\n", std::process::id())).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        let err = ensure_daemon_offline(&paths, Style::plain(), "plan cancel").unwrap_err();
        assert!(format!("{err:#}").contains("daemon is alive"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn offline_plan_runner_open_errors_when_db_absent() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-nodb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        let Err(err) = OfflinePlanRunner::open(&paths) else {
            panic!("expected error when no flowd.db is present");
        };
        assert!(format!("{err:#}").contains("no flowd database"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn format_thousands_inserts_commas_every_three_digits() {
        assert_eq!(format_thousands(0), "0");
        assert_eq!(format_thousands(7), "7");
        assert_eq!(format_thousands(42), "42");
        assert_eq!(format_thousands(999), "999");
        assert_eq!(format_thousands(1_000), "1,000");
        assert_eq!(format_thousands(2_148), "2,148");
        assert_eq!(format_thousands(15_820), "15,820");
        assert_eq!(format_thousands(1_234_567), "1,234,567");
        assert_eq!(format_thousands(u64::MAX), "18,446,744,073,709,551,615");
    }

    #[test]
    fn format_metrics_block_returns_none_when_metrics_absent() {
        let payload = serde_json::json!({ "output": "ok" });
        assert!(format_metrics_block(&payload).is_none());
    }

    #[test]
    fn format_metrics_block_formats_cost_tokens_and_duration() {
        let payload = serde_json::json!({
            "output": "ok",
            "metrics": {
                "input_tokens": 2_148,
                "output_tokens": 8_392,
                "cache_creation_input_tokens": 4_216,
                "cache_read_input_tokens": 15_820,
                "total_cost_usd": 0.4231,
                "duration_ms": 4_500,
                "duration_api_ms": 4_100,
            }
        });
        let lines = format_metrics_block(&payload).expect("metrics block");
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            "cost: $0.4231   tokens: in 2,148   out 8,392   cache_read 15,820   cache_creation 4,216"
        );
        assert_eq!(lines[1], "duration: api 4.1s / total 4.5s");
    }

    #[test]
    fn format_tokens_compact_keeps_thousands_separator_below_ten_k() {
        assert_eq!(format_tokens_compact(0), "0");
        assert_eq!(format_tokens_compact(999), "999");
        assert_eq!(format_tokens_compact(9_999), "9,999");
    }

    #[test]
    fn format_tokens_compact_switches_to_k_suffix_above_ten_k() {
        assert_eq!(format_tokens_compact(10_000), "10.0k");
        assert_eq!(format_tokens_compact(12_400), "12.4k");
        assert_eq!(format_tokens_compact(21_300), "21.3k");
        assert_eq!(format_tokens_compact(100_000), "100.0k");
    }

    #[test]
    fn format_finished_rollup_returns_empty_when_payload_has_no_rollup_fields() {
        let payload = serde_json::json!({ "status": "completed" });
        assert!(format_finished_rollup(&payload).is_empty());
    }

    #[test]
    fn format_finished_rollup_renders_full_layout() {
        let payload = serde_json::json!({
            "status": "completed",
            "step_count": { "completed": 4, "failed": 1 },
            "metrics": {
                "input_tokens": 12_400,
                "output_tokens": 21_300,
                "cache_creation_input_tokens": 5_000,
                "cache_read_input_tokens": 18_000,
                "total_cost_usd": 1.42,
                "duration_ms": 268_100,
                "duration_api_ms": 24_700,
            }
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(lines.len(), 2);
        // 18_000 / (18_000 + 5_000) = 78.26% -> rounds to 78%.
        assert_eq!(
            lines[0],
            "total: $1.42   in 12.4k   out 21.3k   cache hit-rate 78%   completed 4   failed 1"
        );
        assert_eq!(lines[1], "duration: api 24.7s / total 268.1s");
    }

    #[test]
    fn format_finished_rollup_omits_cache_hit_rate_when_denominator_zero() {
        let payload = serde_json::json!({
            "status": "completed",
            "step_count": { "completed": 1, "failed": 0 },
            "metrics": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_cost_usd": 0.01,
                "duration_ms": 500,
                "duration_api_ms": 400,
            }
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(
            lines[0],
            "total: $0.01   in 100   out 50   completed 1   failed 0"
        );
    }

    #[test]
    fn format_finished_rollup_tolerates_step_count_without_metrics() {
        // Plan finished with zero metrics (e.g. cancelled draft) but
        // still carries a step_count block; render the counts and a
        // zero-cost/zero-token block rather than skipping the rollup
        // entirely.
        let payload = serde_json::json!({
            "status": "cancelled",
            "step_count": { "completed": 0, "failed": 0 },
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            "total: $0.00   in 0   out 0   completed 0   failed 0"
        );
        assert_eq!(lines[1], "duration: api 0.0s / total 0.0s");
    }

    #[test]
    fn format_metrics_block_defaults_missing_fields_to_zero() {
        // An agent that only reports cost still gets a well-formed block;
        // we don't want a crash or `null` formatting when newer fields
        // are absent from older event rows.
        let payload = serde_json::json!({
            "error": "boom",
            "metrics": { "total_cost_usd": 0.0009 }
        });
        let lines = format_metrics_block(&payload).expect("metrics block");
        assert_eq!(
            lines[0],
            "cost: $0.0009   tokens: in 0   out 0   cache_read 0   cache_creation 0"
        );
        assert_eq!(lines[1], "duration: api 0.0s / total 0.0s");
    }

    #[test]
    fn cli_answer_entry_parses_choose_payload() {
        let raw = r#"[{"question_id": "q1", "kind": "choose", "option_id": "opt-a"}]"#;
        let parsed: Vec<CliAnswerEntry> = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].question_id, "q1");
        assert!(matches!(
            parsed[0].answer,
            Answer::Choose { ref option_id } if option_id == "opt-a"
        ));
    }
}
