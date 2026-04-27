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

use std::collections::{BTreeMap, HashMap};
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use chrono::{DateTime, Utc};
use flowd_core::orchestration::observer::PlanEvent;
use flowd_core::orchestration::plan_events::{PlanEventQuery, PlanEventStore, StoredPlanEvent};
use flowd_core::orchestration::{
    Answer, CleanupPolicy, IntegrationConfig, IntegrationMetadata, IntegrationMode,
    IntegrationStatus, Plan, PlanCompiler, PlanDraftSnapshot, PlanIntegrateOutcome,
    PlanIntegrateRequest, PlanStatus, PlanStore, PlanSummary, VerificationConfig, build_preview,
    load_plan,
};
use flowd_storage::plan_event_store::SqlitePlanEventStore;
use flowd_storage::plan_store::SqlitePlanStore;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::config::FlowdConfig;
use crate::daemon;
use crate::integration::{IntegrateError, PlanIntegrator};
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

/// `flowd plan usage <plan_id>` -- concise audit rollup of token/cost
/// spend for a single plan. Reads the same persisted event log as
/// [`events`] and aggregates step-event metrics in process so it works
/// against a live daemon (WAL-safe) without needing the executor.
///
/// `--json` emits the same [`UsageReport`] struct that backs the human
/// renderer, so scripted callers and the human surface share a schema.
pub async fn usage(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    json: bool,
) -> Result<()> {
    let plan_id = parse_plan_id(&plan_id_arg)?;
    let report = prepare_usage_report(paths, plan_id).await?;

    if json {
        let s = serde_json::to_string_pretty(&report).context("serialize usage report")?;
        println!("{s}");
    } else {
        render_usage_report(&report, style);
    }
    Ok(())
}

/// DB-read half of [`usage`]: open the event store, pull the plan's
/// event tail, and aggregate. Split out so command-level tests can
/// drive end-to-end through a temp `SQLite` file without capturing
/// stdout.
async fn prepare_usage_report(paths: &FlowdPaths, plan_id: Uuid) -> Result<UsageReport> {
    let db_path = paths.db_file();
    if !db_path.exists() {
        bail!(
            "no flowd database at {}; start the daemon at least once to initialise it",
            db_path.display()
        );
    }
    let store = SqlitePlanEventStore::open(&db_path)
        .with_context(|| format!("open plan event store at {}", db_path.display()))?;

    // Pull the entire event tail for this plan; bounded by the per-plan
    // event count, which is small (one row per step plus a handful of
    // lifecycle events). usize::MAX maps to i64::MAX inside the store.
    let rows = store
        .list_for_plan(plan_id, PlanEventQuery::new(usize::MAX))
        .await
        .with_context(|| format!("list events for plan {plan_id}"))?;

    if rows.is_empty() {
        bail!("plan {plan_id}: no events recorded; nothing to summarise");
    }
    Ok(build_usage_report(plan_id, &rows))
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

/// Rollup rendered beneath the `finished` event header. Two lines:
///
/// 1. `total: $… in … out … cache_read … cache_create … reuse …% completed … failed …`
///    -- per-token totals plus a one-decimal `reuse` rate, so an
///    integer-rounded `78%` no longer hides the difference between
///    78.1% and 78.9%. The `cache_read` and `cache_create` totals are
///    surfaced directly so the operator can sanity-check the rate.
/// 2. `runtime sum: api …s   agent …s   elapsed …s` -- API and agent
///    durations are explicitly summed across steps; with parallel layers
///    that sum can exceed `elapsed`, which is the wall-clock from
///    `Plan.started_at` to `completed_at`. Older events without
///    `elapsed_ms` and "never executed" transitions (cancel from `Draft`,
///    rehydrate-as-`Interrupted`) drop the `elapsed` segment rather than
///    rendering a misleading `0.0s`.
///
/// Returns an empty `Vec` when neither `metrics` nor a `step_count` block
/// was persisted (pre-rollup event rows). Payload shape is owned by
/// `flowd_core::orchestration::plan_events::event_payload` for the
/// `Finished` variant.
//
// Cast lints, all ranged-safe:
// - `cast_precision_loss`: u64 ms / token counts well below 2^53.
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
    let elapsed_ms = payload
        .get("elapsed_ms")
        .and_then(serde_json::Value::as_u64);

    let mut first = format!(
        "total: ${cost:.2}   in {in_}   out {out}",
        in_ = format_tokens_compact(input),
        out = format_tokens_compact(output),
    );
    let cache_denom = cache_read + cache_creation;
    if cache_denom > 0 {
        write!(
            first,
            "   cache_read {read}   cache_create {create}   reuse {pct}",
            read = format_tokens_compact(cache_read),
            create = format_tokens_compact(cache_creation),
            pct = format_one_decimal_percent(cache_read, cache_denom),
        )
        .expect("write to String is infallible");
    }
    write!(first, "   completed {completed}   failed {failed}")
        .expect("write to String is infallible");

    #[allow(clippy::cast_precision_loss)]
    let mut second = format!(
        "runtime sum: api {:.1}s   agent {:.1}s",
        duration_api_ms as f64 / 1000.0,
        duration_ms as f64 / 1000.0,
    );
    if let Some(ms) = elapsed_ms {
        #[allow(clippy::cast_precision_loss)]
        let elapsed_secs = ms as f64 / 1000.0;
        write!(second, "   elapsed {elapsed_secs:.1}s").expect("write to String is infallible");
    }

    vec![first, second]
}

/// Format `numerator / denominator` as a one-decimal percent like `78.3%`.
/// Caller must ensure `denominator > 0`; we still guard against it
/// defensively because the rollup feeds in u64 sums that can theoretically
/// be zero in malformed payloads.
//
// `cast_precision_loss` is fine: cache totals fit comfortably in f64.
#[allow(clippy::cast_precision_loss)]
fn format_one_decimal_percent(numerator: u64, denominator: u64) -> String {
    if denominator == 0 {
        return "0.0%".to_owned();
    }
    let pct = numerator as f64 / denominator as f64 * 100.0;
    format!("{pct:.1}%")
}

/// JSON-friendly per-plan usage rollup powering both the human render
/// and `--json` output. Field names mirror the on-disk metrics keys so
/// scripted callers can correlate without consulting docs; counters that
/// would always be `null` for plans without metrics (cache reuse rate,
/// wall-clock window) are skipped via `Option`-elision rather than
/// rendered as `0` or `null`.
#[derive(Debug, Default, Serialize, PartialEq)]
pub(crate) struct UsageReport {
    pub plan_id: Uuid,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub plan_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub status: Option<String>,
    pub steps: u64,
    pub total_cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
    /// `cache_read / (cache_read + cache_creation)`, in `[0.0, 1.0]`.
    /// `None` when both counters are zero -- there was no cache activity
    /// to compute a ratio over, and rendering `0.0%` would mislead.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cache_hit_rate: Option<f64>,
    pub duration_api_ms: u64,
    pub duration_total_ms: u64,
    /// Wall-clock span between the first and last persisted event for
    /// this plan. `None` when only a single event was recorded.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub wall_clock_ms: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub first_event_at: Option<DateTime<Utc>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub last_event_at: Option<DateTime<Utc>>,
    pub model_breakdown: BTreeMap<String, ModelBreakdown>,
}

/// Per-model rollup inside [`UsageReport::model_breakdown`].
///
/// Cache fields use the short alias (`cache_read_tokens`) the report
/// surfaces for `UsageTotals` consumers; the corresponding storage
/// keys are `cache_read_input_tokens` / `cache_creation_input_tokens`,
/// which the aggregator translates inline.
#[derive(Debug, Default, Serialize, PartialEq)]
pub(crate) struct ModelBreakdown {
    pub cost_usd: f64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_read_tokens: u64,
    pub cache_creation_tokens: u64,
}

/// Walk `events` and aggregate the metrics blocks attached to step
/// outcomes. Pure function over the event slice so the unit tests can
/// drive it without booting `SQLite`.
//
// Token / duration counters and the cache ratio cast match the rationale
// in `format_finished_rollup`: well below the f64 precision ceiling, and
// the ratio is bounded to `[0, 1]` before any cast to a percent value
// happens at the render layer.
#[allow(clippy::cast_precision_loss)]
fn build_usage_report(plan_id: Uuid, events: &[StoredPlanEvent]) -> UsageReport {
    use flowd_core::orchestration::plan_events::kind as k;

    let mut report = UsageReport {
        plan_id,
        ..UsageReport::default()
    };

    if let (Some(first), Some(last)) = (events.first(), events.last()) {
        report.first_event_at = Some(first.created_at);
        report.last_event_at = Some(last.created_at);
        // Distinct events: derive an actual span. A single event leaves
        // `wall_clock_ms` as `None` -- a 0ms wall-clock is misleading
        // because there's no real span to report on.
        if events.len() > 1 {
            let delta = (last.created_at - first.created_at).num_milliseconds();
            report.wall_clock_ms = Some(u64::try_from(delta.max(0)).unwrap_or(0));
        }
    }

    for evt in events {
        if report.project.is_none() {
            report.project = Some(evt.project.clone());
        }
        match evt.kind.as_str() {
            k::SUBMITTED => {
                report.plan_name = evt
                    .payload
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
            k::FINISHED => {
                report.status = evt
                    .payload
                    .get("status")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned);
            }
            k::STEP_COMPLETED | k::STEP_FAILED => {
                let Some(metrics) = evt.payload.get("metrics") else {
                    continue;
                };
                let u64_of = |key: &str| {
                    metrics
                        .get(key)
                        .and_then(serde_json::Value::as_u64)
                        .unwrap_or(0)
                };
                let f64_of = |key: &str| {
                    metrics
                        .get(key)
                        .and_then(serde_json::Value::as_f64)
                        .unwrap_or(0.0)
                };

                report.steps += 1;
                report.input_tokens += u64_of("input_tokens");
                report.output_tokens += u64_of("output_tokens");
                report.cache_read_tokens += u64_of("cache_read_input_tokens");
                report.cache_creation_tokens += u64_of("cache_creation_input_tokens");
                report.total_cost_usd += f64_of("total_cost_usd");
                report.duration_api_ms += u64_of("duration_api_ms");
                report.duration_total_ms += u64_of("duration_ms");

                if let Some(model_usage) = metrics
                    .get("model_usage")
                    .and_then(serde_json::Value::as_object)
                {
                    for (model, m) in model_usage {
                        let entry = report.model_breakdown.entry(model.clone()).or_default();
                        entry.input_tokens += m
                            .get("input_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        entry.output_tokens += m
                            .get("output_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        entry.cache_read_tokens += m
                            .get("cache_read_input_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        entry.cache_creation_tokens += m
                            .get("cache_creation_input_tokens")
                            .and_then(serde_json::Value::as_u64)
                            .unwrap_or(0);
                        entry.cost_usd += m
                            .get("cost_usd")
                            .and_then(serde_json::Value::as_f64)
                            .unwrap_or(0.0);
                    }
                }
            }
            _ => {}
        }
    }

    let cache_denom = report.cache_read_tokens + report.cache_creation_tokens;
    if cache_denom > 0 {
        report.cache_hit_rate = Some(report.cache_read_tokens as f64 / cache_denom as f64);
    }
    report
}

/// Sort the model breakdown by descending cost (with a name tiebreaker
/// so the order is deterministic when two models report the same spend).
fn model_rows_by_cost(report: &UsageReport) -> Vec<(&String, &ModelBreakdown)> {
    let mut rows: Vec<_> = report.model_breakdown.iter().collect();
    rows.sort_by(|a, b| {
        b.1.cost_usd
            .partial_cmp(&a.1.cost_usd)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(b.0))
    });
    rows
}

/// Format a duration in milliseconds for human display:
/// * `<1s`   -> `Xms`
/// * `<60s`  -> `S.Ds` (one decimal)
/// * `<60m`  -> `Mm Ss`
/// * else    -> `Hh Mm Ss`
//
// The seconds branch crosses an `as f64` boundary; we are always under
// `60_000` here, well within precision, and the result is bounded to
// `[0.0, 60.0)`. The same applies to the integer branches.
#[allow(clippy::cast_precision_loss)]
fn format_duration_ms(ms: u64) -> String {
    if ms < 1_000 {
        return format!("{ms}ms");
    }
    let total_secs = ms / 1_000;
    if total_secs < 60 {
        return format!("{:.1}s", ms as f64 / 1_000.0);
    }
    let secs = total_secs % 60;
    let mins = (total_secs / 60) % 60;
    let hours = total_secs / 3_600;
    if hours > 0 {
        format!("{hours}h {mins}m {secs}s")
    } else {
        format!("{mins}m {secs}s")
    }
}

/// Render `report` to a `String` so tests can inspect the layout
/// without capturing stdout. The `println!` wrapper ([`render_usage_report`])
/// is a one-liner over this.
fn format_usage_report(report: &UsageReport, style: Style) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str(&banner(&format!("usage: {}", report.plan_id), style));
    if let Some(name) = &report.plan_name {
        let _ = writeln!(out, "  plan:        {name}");
    }
    if let Some(project) = &report.project {
        let _ = writeln!(out, "  project:     {project}");
    }
    if let Some(status) = &report.status {
        let _ = writeln!(out, "  status:      {status}");
    }
    let _ = writeln!(out, "  steps:       {}", report.steps);
    let _ = writeln!(
        out,
        "  total cost:  {}",
        style.bold(&format!("${:.2}", report.total_cost_usd)),
    );
    let _ = writeln!(
        out,
        "  tokens:      in {}   out {}",
        format_tokens_compact(report.input_tokens),
        format_tokens_compact(report.output_tokens),
    );
    let _ = writeln!(
        out,
        "  cache:       read {}   creation {}",
        format_thousands(report.cache_read_tokens),
        format_thousands(report.cache_creation_tokens),
    );
    match report.cache_hit_rate {
        // One-decimal cache reuse rate per the audit spec.
        Some(rate) => {
            let _ = writeln!(out, "  cache reuse: {:.1}%", rate * 100.0);
        }
        None => {
            let _ = writeln!(out, "  cache reuse: {}", style.dim("(no cache activity)"));
        }
    }
    let _ = writeln!(
        out,
        "  runtime:     api {}   agent {}",
        format_duration_ms(report.duration_api_ms),
        format_duration_ms(report.duration_total_ms),
    );
    if let Some(wall) = report.wall_clock_ms {
        let _ = writeln!(out, "  wall-clock:  {}", format_duration_ms(wall));
    }

    if !report.model_breakdown.is_empty() {
        let _ = writeln!(out, "\n{}", style.bold("models:"));
        for (name, m) in model_rows_by_cost(report) {
            let _ = writeln!(
                out,
                "  {name:<20}  ${cost:>7.2}   in {in_:>8}   out {out:>8}   cache r/c {cr}/{cc}",
                cost = m.cost_usd,
                in_ = format_tokens_compact(m.input_tokens),
                out = format_tokens_compact(m.output_tokens),
                cr = format_thousands(m.cache_read_tokens),
                cc = format_thousands(m.cache_creation_tokens),
            );
        }
    }
    out
}

fn render_usage_report(report: &UsageReport, style: Style) {
    print!("{}", format_usage_report(report, style));
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
        k::INTEGRATION_STARTED => Some(format_integration_started(payload)),
        k::INTEGRATION_SUCCEEDED => Some(format_integration_succeeded(payload)),
        k::INTEGRATION_FAILED => Some(format_integration_failed(payload)),
        _ => None,
    }
}

/// Squeeze the structured integration event payloads into one
/// `key=value` line per event. Keeps the renderer aligned with the
/// per-step event style and avoids leaking raw JSON onto the terminal.
fn format_integration_started(payload: &serde_json::Value) -> String {
    let branch = payload
        .get("integration_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let base = payload
        .get("base_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let mode = payload
        .get("mode")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    format!("integration_branch={branch}  base={base}  mode={mode}")
}

fn format_integration_succeeded(payload: &serde_json::Value) -> String {
    use std::fmt::Write as _;

    let branch = payload
        .get("integration_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let base = payload
        .get("base_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let status = payload
        .get("status")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let tip = payload
        .get("promoted_tip")
        .and_then(serde_json::Value::as_str);
    let mut out = format!("integration_branch={branch}  base={base}  status={status}");
    if let Some(t) = tip {
        let _ = write!(out, "  promoted_tip={t}");
    }
    out
}

fn format_integration_failed(payload: &serde_json::Value) -> String {
    let branch = payload
        .get("integration_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let base = payload
        .get("base_branch")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    let reason = payload
        .get("reason")
        .and_then(serde_json::Value::as_str)
        .unwrap_or("?");
    format!("integration_branch={branch}  base={base}  reason={reason}")
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

/// Bundle of arguments forwarded from clap to [`integrate`]. Pulled out so
/// the dispatcher in `main.rs` does not exceed the
/// `clippy::too_many_arguments` ceiling and so a future MCP front door can
/// reuse the same parsing helpers without re-deriving the call shape.
///
/// Lint allowances:
/// * `struct_excessive_bools`: the four operator-facing toggles
///   (`promote`, `dry_run`, `discard`, `json`) intentionally each carry
///   their own flag at the CLI surface so scripts can pin behaviour
///   without parsing a state-machine value. Clap already enforces the
///   mutually-exclusive set upstream, so the lint's "use a state
///   machine" suggestion would just shuffle the validation boilerplate
///   inward.
#[allow(clippy::struct_excessive_bools)]
#[derive(Debug, Clone)]
pub struct IntegrateArgs {
    pub plan_id: String,
    pub base: String,
    pub promote: bool,
    pub dry_run: bool,
    /// Tear down a previously-staged integration without promoting.
    /// Removes the integration worktree+branch and (per `cleanup`) the
    /// per-step branches and worktrees. Mutually exclusive with
    /// `promote` and `dry_run` -- enforced upstream by clap.
    pub discard: bool,
    pub cleanup: String,
    pub strategy: String,
    /// Optional verification command run inside the integration worktree
    /// before promote. Supplied as a single shell-style string (e.g.
    /// `cargo nextest run -p flowd-cli`); split on whitespace at parse
    /// time. Empty / missing skips verification.
    pub verify: Option<String>,
    pub json: bool,
}

/// `flowd plan integrate <plan_id>` -- drive the `plan_integrate`
/// contract against a real git repo and persist the resulting integration
/// metadata.
///
/// Three modes:
///
/// * Default (no flags): stage. Runs eligibility, creates the integration
///   worktree+branch, cherry-picks each tip step's range, and persists
///   [`IntegrationStatus::Staged`] on the plan.
/// * `--dry-run`: preview only. No git mutation; renders the planned
///   cherry-picks. Does not persist anything to the plan store.
/// * `--promote`: fast-forward the staged integration branch onto
///   `--base`, persisting [`IntegrationStatus::Promoted`] on success.
///
/// Refuses while the daemon is alive (mutates shared `SQLite` state); the
/// equivalent MCP tool is the alternative when the daemon owns the
/// in-memory executor.
///
/// Never pushes -- remote propagation is the operator's job. The
/// integrator only ever invokes local `git` plumbing (`worktree add`,
/// `cherry-pick`, `update-ref`).
pub async fn integrate(paths: &FlowdPaths, style: Style, args: IntegrateArgs) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan integrate")?;

    let mode = parse_integrate_mode(args.dry_run, args.promote);
    let cleanup = parse_cleanup_policy(&args.cleanup)?;
    parse_integrate_strategy(&args.strategy)?;

    let plan_id = parse_plan_id(&args.plan_id)?;

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_existing(plan_id).await?;

    let verify = parse_verify_argv(args.verify.as_deref());
    let request = PlanIntegrateRequest::new(
        plan_id,
        mode,
        IntegrationConfig {
            base_branch: args.base.clone(),
            cleanup,
            verify,
        },
    )
    .map_err(|refusal| anyhow!("{refusal}"))?;

    let event_store = open_plan_event_store(paths)?;
    let integrator = PlanIntegrator::new(
        plan.project_root
            .as_deref()
            .map_or_else(|| paths.home.clone(), Into::into),
        paths.integrate_worktrees_dir(),
        paths.home.join("worktrees"),
        None,
    );

    if args.discard {
        run_discard(&integrator, &runner, &mut plan, &request, style, args.json).await
    } else if args.promote {
        run_promote(
            &integrator,
            &runner,
            &event_store,
            &mut plan,
            &request,
            style,
            args.json,
        )
        .await
    } else {
        run_stage(
            &integrator,
            &runner,
            &event_store,
            &mut plan,
            &request,
            style,
            args.json,
        )
        .await
    }
}

/// Tear down a previously-staged integration without touching the base.
/// Resets the persisted integration metadata to `Pending` so a follow-up
/// stage starts from a clean slate. Refusals from the pure contract
/// surface verbatim; missing artefacts on disk are treated as
/// already-cleaned (the integrator is idempotent).
async fn run_discard(
    integrator: &PlanIntegrator,
    runner: &OfflinePlanRunner,
    plan: &mut Plan,
    request: &PlanIntegrateRequest,
    style: Style,
    json: bool,
) -> Result<()> {
    match integrator.discard(plan, request).await {
        Ok(()) => {
            // Operator-driven discard: drop the integration metadata so a
            // subsequent stage starts fresh. We deliberately do not write
            // a `Failed` status -- nothing failed, the operator chose to
            // throw away the work.
            plan.integration = None;
            runner.save(plan).await?;
            if json {
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "kind": "discarded",
                        "plan_id": plan.id,
                        "base_branch": request.config.base_branch,
                    }))
                    .unwrap_or_else(|_| "{}".into())
                );
            } else {
                print!("{}", banner("integrate: discarded", style));
                println!("  plan_id:     {}", style.dim(&plan.id.to_string()));
                println!("  base_branch: {}", request.config.base_branch);
                println!(
                    "\n{}",
                    style.dim(
                        "integration worktree + branch removed; per-step artefacts honour --cleanup",
                    ),
                );
            }
            Ok(())
        }
        Err(err) => {
            render_integrate_error(&err, style, json);
            Err(integrate_error_into_anyhow(err))
        }
    }
}

async fn run_stage(
    integrator: &PlanIntegrator,
    runner: &OfflinePlanRunner,
    events: &SqlitePlanEventStore,
    plan: &mut Plan,
    request: &PlanIntegrateRequest,
    style: Style,
    json: bool,
) -> Result<()> {
    if request.mode != IntegrationMode::DryRun {
        let now = Utc::now();
        let mut meta = IntegrationMetadata::pending(plan.id, &plan.project, request);
        meta.status = IntegrationStatus::InProgress;
        meta.started_at = Some(now);
        plan.integration = Some(meta.clone());
        runner.save(plan).await?;

        record_event(
            events,
            &PlanEvent::IntegrationStarted {
                plan_id: plan.id,
                project: plan.project.clone(),
                integration_branch: meta.integration_branch.clone(),
                base_branch: meta.base_branch.clone(),
                mode: request.mode,
            },
        )
        .await;
    }

    match integrator.integrate(plan, request).await {
        Ok(outcome) => {
            apply_outcome_to_plan(plan, request, &outcome);
            // Dry-runs leave the plan store untouched: there is no
            // persisted state change to record (no branch, no status
            // transition), so a save would just rewrite the row with
            // identical contents.
            if request.mode != IntegrationMode::DryRun {
                runner.save(plan).await?;
                if let Some(meta) = &plan.integration {
                    record_event(
                        events,
                        &PlanEvent::IntegrationSucceeded {
                            plan_id: plan.id,
                            project: plan.project.clone(),
                            integration_branch: meta.integration_branch.clone(),
                            base_branch: meta.base_branch.clone(),
                            status: meta.status,
                            promoted_tip: None,
                        },
                    )
                    .await;
                }
            }
            render_integrate_success(&outcome, plan, style, json);
            Ok(())
        }
        Err(err) => {
            mark_integration_failure(plan, &err);
            // Persist the failure regardless of dry-run: a refusal that
            // surfaces during `assess_eligibility` is information the
            // operator wants captured. For dry-runs we skip the row
            // write only to avoid clobbering a concurrently-staged
            // metadata block.
            if request.mode != IntegrationMode::DryRun {
                let _ = runner.save(plan).await;
                record_event(
                    events,
                    &PlanEvent::IntegrationFailed {
                        plan_id: plan.id,
                        project: plan.project.clone(),
                        integration_branch: plan
                            .integration
                            .as_ref()
                            .map(|m| m.integration_branch.clone())
                            .unwrap_or_default(),
                        base_branch: request.config.base_branch.clone(),
                        reason: err.to_string(),
                    },
                )
                .await;
            }
            render_integrate_error(&err, style, json);
            Err(integrate_error_into_anyhow(err))
        }
    }
}

async fn run_promote(
    integrator: &PlanIntegrator,
    runner: &OfflinePlanRunner,
    events: &SqlitePlanEventStore,
    plan: &mut Plan,
    request: &PlanIntegrateRequest,
    style: Style,
    json: bool,
) -> Result<()> {
    let now = Utc::now();
    if let Some(meta) = plan.integration.as_mut() {
        meta.status = IntegrationStatus::InProgress;
        meta.started_at.get_or_insert(now);
    } else {
        // Promote without prior stage: the integrator will refuse with a
        // typed Plan-execution error since the integration branch will
        // not resolve. We still emit a started event so the audit trail
        // shows the operator's attempt.
        plan.integration = Some(IntegrationMetadata::pending(
            plan.id,
            &plan.project,
            request,
        ));
    }
    let started_branch = plan
        .integration
        .as_ref()
        .map(|m| m.integration_branch.clone())
        .unwrap_or_default();
    runner.save(plan).await?;
    record_event(
        events,
        &PlanEvent::IntegrationStarted {
            plan_id: plan.id,
            project: plan.project.clone(),
            integration_branch: started_branch.clone(),
            base_branch: request.config.base_branch.clone(),
            mode: request.mode,
        },
    )
    .await;

    match integrator.promote(plan, request).await {
        Ok(outcome) => {
            apply_outcome_to_plan(plan, request, &outcome);
            runner.save(plan).await?;
            if let (PlanIntegrateOutcome::Promoted { promoted_tip, .. }, Some(meta)) =
                (&outcome, plan.integration.as_ref())
            {
                record_event(
                    events,
                    &PlanEvent::IntegrationSucceeded {
                        plan_id: plan.id,
                        project: plan.project.clone(),
                        integration_branch: meta.integration_branch.clone(),
                        base_branch: meta.base_branch.clone(),
                        status: meta.status,
                        promoted_tip: Some(promoted_tip.clone()),
                    },
                )
                .await;
            }
            render_integrate_success(&outcome, plan, style, json);
            Ok(())
        }
        Err(err) => {
            mark_integration_failure(plan, &err);
            let _ = runner.save(plan).await;
            record_event(
                events,
                &PlanEvent::IntegrationFailed {
                    plan_id: plan.id,
                    project: plan.project.clone(),
                    integration_branch: started_branch,
                    base_branch: request.config.base_branch.clone(),
                    reason: err.to_string(),
                },
            )
            .await;
            render_integrate_error(&err, style, json);
            Err(integrate_error_into_anyhow(err))
        }
    }
}

/// Translate the integrator's outcome into a fresh
/// [`IntegrationMetadata`] block on `plan`. Pure -- the caller persists
/// the plan via the runner separately so dry-runs can skip the I/O.
fn apply_outcome_to_plan(
    plan: &mut Plan,
    request: &PlanIntegrateRequest,
    outcome: &PlanIntegrateOutcome,
) {
    let now = Utc::now();
    match outcome {
        PlanIntegrateOutcome::DryRun { intended } => {
            // Dry-runs report eligibility but should not advertise an
            // attempted run on the plan: leave any prior metadata intact
            // and only synthesise a fresh pending block when none exists.
            if plan.integration.is_none() {
                let mut meta = IntegrationMetadata::pending(plan.id, &plan.project, request);
                meta.integration_branch
                    .clone_from(&intended.integration_branch);
                plan.integration = Some(meta);
            }
        }
        PlanIntegrateOutcome::Staged { intended, .. } => {
            let mut meta = plan
                .integration
                .clone()
                .unwrap_or_else(|| IntegrationMetadata::pending(plan.id, &plan.project, request));
            meta.status = IntegrationStatus::Staged;
            meta.integration_branch
                .clone_from(&intended.integration_branch);
            meta.base_branch.clone_from(&intended.base_branch);
            meta.mode = intended.mode;
            meta.cleanup = intended.cleanup;
            meta.completed_at = Some(now);
            meta.failure = None;
            meta.refusal = None;
            plan.integration = Some(meta);
        }
        PlanIntegrateOutcome::Promoted { intended, .. } => {
            let mut meta = plan
                .integration
                .clone()
                .unwrap_or_else(|| IntegrationMetadata::pending(plan.id, &plan.project, request));
            meta.status = IntegrationStatus::Promoted;
            meta.integration_branch
                .clone_from(&intended.integration_branch);
            meta.base_branch.clone_from(&intended.base_branch);
            meta.mode = intended.mode;
            meta.cleanup = intended.cleanup;
            meta.completed_at = Some(now);
            meta.failure = None;
            meta.refusal = None;
            plan.integration = Some(meta);
        }
    }
}

/// Stamp the structured cause from an [`IntegrateError`] onto the plan's
/// integration metadata so a follow-up `plan show` surfaces the typed
/// failure without consulting the event log.
fn mark_integration_failure(plan: &mut Plan, err: &IntegrateError) {
    let now = Utc::now();
    let meta = plan.integration.get_or_insert_with(|| IntegrationMetadata {
        status: IntegrationStatus::Failed,
        integration_branch: String::new(),
        base_branch: String::new(),
        mode: IntegrationMode::Confirm,
        cleanup: CleanupPolicy::default(),
        started_at: Some(now),
        completed_at: Some(now),
        failure: None,
        refusal: None,
    });
    meta.status = IntegrationStatus::Failed;
    meta.completed_at = Some(now);
    meta.refusal = None;
    meta.failure = None;
    match err {
        IntegrateError::Refusal(r) => meta.refusal = Some(r.clone()),
        IntegrateError::Failure(f) => meta.failure = Some(f.clone()),
        IntegrateError::Plan(_) => {}
    }
}

/// Best-effort event write: the integrate flow is operator-facing, and a
/// failure to persist a telemetry row should not abort the run after the
/// git work has already landed. Logged at warn for diagnostics.
async fn record_event(events: &SqlitePlanEventStore, event: &PlanEvent) {
    if let Err(e) = PlanEventStore::record(events, event).await {
        tracing::warn!(target: "flowd::cli::integrate", error = %e, "record plan event");
    }
}

fn integrate_error_into_anyhow(err: IntegrateError) -> anyhow::Error {
    match err {
        IntegrateError::Refusal(r) => anyhow!(r),
        IntegrateError::Failure(f) => anyhow!(f),
        IntegrateError::Plan(p) => anyhow!(p),
    }
}

fn render_integrate_success(outcome: &PlanIntegrateOutcome, plan: &Plan, style: Style, json: bool) {
    if json {
        let value = serde_json::to_value(outcome).unwrap_or_else(|_| {
            serde_json::json!({
                "kind": "unknown",
                "error": "failed to serialise outcome",
            })
        });
        if let Ok(s) = serde_json::to_string_pretty(&value) {
            println!("{s}");
        }
        return;
    }
    let header = match outcome {
        PlanIntegrateOutcome::DryRun { .. } => "dry-run -- no git mutation",
        PlanIntegrateOutcome::Staged { .. } => "staged",
        PlanIntegrateOutcome::Promoted { .. } => "promoted",
    };
    print!("{}", banner(&format!("integrate: {header}"), style));
    match outcome {
        PlanIntegrateOutcome::DryRun { intended } => {
            println!(
                "  plan_id:            {}",
                style.dim(&intended.plan_id.to_string())
            );
            println!("  base_branch:        {}", intended.base_branch);
            println!("  integration_branch: {}", intended.integration_branch);
            println!(
                "  cleanup:            {}",
                cleanup_policy_label(intended.cleanup)
            );
            println!("\n{}", style.bold("planned cherry-picks:"));
            for (i, pick) in intended.cherry_picks.iter().enumerate() {
                println!(
                    "  {idx} step={step}  branch={branch}",
                    idx = style.cyan(&format!("{:>2}.", i + 1)),
                    step = pick.step_id,
                    branch = pick.source_branch,
                );
            }
        }
        PlanIntegrateOutcome::Staged {
            intended,
            integration_tip,
        } => {
            println!(
                "  plan_id:            {}",
                style.dim(&intended.plan_id.to_string())
            );
            println!("  base_branch:        {}", intended.base_branch);
            println!("  integration_branch: {}", intended.integration_branch);
            println!("  integration_tip:    {}", style.dim(integration_tip));
            println!("  cherry_picks:       {}", intended.cherry_picks.len());
            println!(
                "\n{}",
                style.dim(
                    "next: review the integration branch, then re-run with `--promote --base <BRANCH>`"
                ),
            );
        }
        PlanIntegrateOutcome::Promoted {
            intended,
            promoted_tip,
        } => {
            println!(
                "  plan_id:            {}",
                style.dim(&intended.plan_id.to_string())
            );
            println!("  base_branch:        {}", intended.base_branch);
            println!("  promoted_tip:       {}", style.green(promoted_tip));
            println!(
                "\n{}",
                style.dim("note: nothing has been pushed; propagate to a remote yourself"),
            );
        }
    }
    if let Some(meta) = &plan.integration {
        render_integration_metadata(meta, style);
    }
}

fn render_integrate_error(err: &IntegrateError, style: Style, json: bool) {
    if json {
        let value = match err {
            IntegrateError::Refusal(r) => serde_json::json!({
                "kind": "refusal",
                "cause": r,
                "message": r.to_string(),
            }),
            IntegrateError::Failure(f) => serde_json::json!({
                "kind": "failure",
                "cause": f,
                "message": f.to_string(),
            }),
            IntegrateError::Plan(p) => serde_json::json!({
                "kind": "plan_execution",
                "message": p.to_string(),
            }),
        };
        if let Ok(s) = serde_json::to_string_pretty(&value) {
            eprintln!("{s}");
        }
        return;
    }
    let (label, message) = match err {
        IntegrateError::Refusal(r) => ("refusal", r.to_string()),
        IntegrateError::Failure(f) => ("failure", f.to_string()),
        IntegrateError::Plan(p) => ("error", p.to_string()),
    };
    eprintln!("{} {message}", style.red(&format!("integrate {label}:")));
}

fn parse_integrate_mode(dry_run: bool, promote: bool) -> IntegrationMode {
    if dry_run {
        IntegrationMode::DryRun
    } else if promote {
        // Promote replays eligibility against the same `Confirm` shape;
        // the integrator distinguishes promote-vs-stage by which method
        // the caller invokes, not by mode.
        IntegrationMode::Confirm
    } else {
        IntegrationMode::Confirm
    }
}

fn parse_cleanup_policy(raw: &str) -> Result<CleanupPolicy> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "keep_on_failure" | "keep-on-failure" => Ok(CleanupPolicy::KeepOnFailure),
        "keep_always" | "keep-always" => Ok(CleanupPolicy::KeepAlways),
        "drop_always" | "drop-always" => Ok(CleanupPolicy::DropAlways),
        other => bail!(
            "unknown cleanup policy `{other}`; expected keep_on_failure, keep_always, or drop_always"
        ),
    }
}

fn parse_integrate_strategy(raw: &str) -> Result<()> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "tip-cherry-pick" | "tip_cherry_pick" => Ok(()),
        other => {
            bail!("unknown integrate strategy `{other}`; only `tip-cherry-pick` is supported in v1")
        }
    }
}

/// Parse the `--verify "<cmd>"` argument into a [`VerificationConfig`].
///
/// V1 keeps this simple: whitespace-split the command into argv. Quoted
/// arguments and shell metacharacters are explicitly *not* honoured -- a
/// real shell never enters the picture, and a future revision can graft
/// `shlex` on top without changing the call sites if operators need it.
/// An empty / `None` value disables verification.
fn parse_verify_argv(raw: Option<&str>) -> VerificationConfig {
    let Some(raw) = raw else {
        return VerificationConfig::default();
    };
    let argv: Vec<String> = raw.split_whitespace().map(str::to_owned).collect();
    VerificationConfig::from_argv(argv)
}

fn integration_status_label(status: IntegrationStatus) -> &'static str {
    match status {
        IntegrationStatus::Pending => "pending",
        IntegrationStatus::InProgress => "in_progress",
        IntegrationStatus::Staged => "staged",
        IntegrationStatus::Promoted => "promoted",
        IntegrationStatus::Failed => "failed",
    }
}

fn integration_mode_label(mode: IntegrationMode) -> &'static str {
    match mode {
        IntegrationMode::Confirm => "confirm",
        IntegrationMode::DryRun => "dry_run",
    }
}

fn cleanup_policy_label(policy: CleanupPolicy) -> &'static str {
    match policy {
        CleanupPolicy::KeepOnFailure => "keep_on_failure",
        CleanupPolicy::KeepAlways => "keep_always",
        CleanupPolicy::DropAlways => "drop_always",
    }
}

fn open_plan_event_store(paths: &FlowdPaths) -> Result<SqlitePlanEventStore> {
    let db_path = paths.db_file();
    if !db_path.exists() {
        bail!(
            "no flowd database at {}; start the daemon at least once to initialise it",
            db_path.display()
        );
    }
    SqlitePlanEventStore::open(&db_path)
        .with_context(|| format!("open plan event store at {}", db_path.display()))
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
    println!(
        "  project_root:     {}",
        plan.project_root
            .as_deref()
            .unwrap_or("(legacy: not recorded)")
    );
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
    if let Some(meta) = &plan.integration {
        render_integration_metadata(meta, style);
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

/// Render the integration block on `plan show` / post-integrate output.
/// Lifted out of [`print_plan_summary`] so the integrate handler can call
/// it without going through the full plan summary header.
fn render_integration_metadata(meta: &IntegrationMetadata, style: Style) {
    println!("\n{}", style.bold("integration:"));
    println!(
        "  status:             {}",
        integration_status_label(meta.status)
    );
    println!("  integration_branch: {}", meta.integration_branch);
    println!("  base_branch:        {}", meta.base_branch);
    println!(
        "  mode:               {}",
        integration_mode_label(meta.mode)
    );
    println!(
        "  cleanup:            {}",
        cleanup_policy_label(meta.cleanup),
    );
    if let Some(ts) = meta.started_at {
        println!(
            "  started_at:         {}",
            style.dim(&ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        );
    }
    if let Some(ts) = meta.completed_at {
        println!(
            "  completed_at:       {}",
            style.dim(&ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()),
        );
    }
    if let Some(refusal) = &meta.refusal {
        println!("  refusal:            {}", style.red(&refusal.to_string()));
    }
    if let Some(failure) = &meta.failure {
        println!("  failure:            {}", style.red(&failure.to_string()));
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
                // Both cache totals chosen >9_999 so the test exercises
                // the compact `k` suffix path in both fields -- summed
                // rollup numbers are routinely well above that floor.
                "cache_creation_input_tokens": 50_000,
                "cache_read_input_tokens": 180_000,
                "total_cost_usd": 1.42,
                "duration_ms": 268_100,
                "duration_api_ms": 24_700,
            },
            "elapsed_ms": 268_100u64,
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(lines.len(), 2);
        // 180_000 / (180_000 + 50_000) = 78.26% -> 78.3% (one-decimal
        // reuse, not the integer-rounded 78% the old layout collapsed to).
        assert_eq!(
            lines[0],
            "total: $1.42   in 12.4k   out 21.3k   cache_read 180.0k   cache_create 50.0k   reuse 78.3%   completed 4   failed 1"
        );
        // Sequential plan: summed agent runtime ≈ wall-clock elapsed.
        assert_eq!(
            lines[1],
            "runtime sum: api 24.7s   agent 268.1s   elapsed 268.1s"
        );
    }

    #[test]
    fn format_finished_rollup_omits_cache_segment_when_denominator_zero() {
        // No cache reads or creates means the reuse rate is undefined;
        // dropping the cache_read/cache_create/reuse segment is the
        // honest signal -- a `reuse 0.0%` line would wrongly imply the
        // agent saw cache traffic.
        let payload = serde_json::json!({
            "status": "completed",
            "step_count": { "completed": 1, "failed": 0 },
            "metrics": {
                "input_tokens": 100,
                "output_tokens": 50,
                "total_cost_usd": 0.01,
                "duration_ms": 500,
                "duration_api_ms": 400,
            },
            "elapsed_ms": 500u64,
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(
            lines[0],
            "total: $0.01   in 100   out 50   completed 1   failed 0"
        );
        assert_eq!(
            lines[1],
            "runtime sum: api 0.4s   agent 0.5s   elapsed 0.5s"
        );
    }

    #[test]
    fn format_finished_rollup_tolerates_step_count_without_metrics() {
        // Plan finished with zero metrics (e.g. cancelled draft) but
        // still carries a step_count block; render the counts and a
        // zero-cost/zero-token block rather than skipping the rollup
        // entirely. With no `elapsed_ms` the runtime line drops the
        // `elapsed` segment instead of rendering a misleading `0.0s`.
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
        assert_eq!(lines[1], "runtime sum: api 0.0s   agent 0.0s");
    }

    #[test]
    fn format_finished_rollup_distinguishes_summed_runtime_from_wall_clock_for_parallel_steps() {
        // Regression guard for the parallel-execution case the new
        // layout exists to surface: two 30-second steps running side
        // by side report a 60s summed agent runtime even though the
        // operator only waited ~30s. The pre-rename layout said
        // `total 60.0s` and silently dropped the wall-clock difference.
        let payload = serde_json::json!({
            "status": "completed",
            "step_count": { "completed": 2, "failed": 0 },
            "metrics": {
                "input_tokens": 1_000,
                "output_tokens": 500,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0,
                "total_cost_usd": 0.20,
                "duration_ms": 60_000,
                "duration_api_ms": 50_000,
            },
            "elapsed_ms": 30_500u64,
        });
        let lines = format_finished_rollup(&payload);
        assert_eq!(lines.len(), 2);
        assert_eq!(
            lines[0],
            "total: $0.20   in 1,000   out 500   completed 2   failed 0"
        );
        // Summed `agent 60.0s` is twice `elapsed 30.5s` -- the layout
        // makes the parallel speed-up visible at a glance.
        assert_eq!(
            lines[1],
            "runtime sum: api 50.0s   agent 60.0s   elapsed 30.5s"
        );
    }

    #[test]
    fn format_one_decimal_percent_keeps_one_fractional_digit() {
        assert_eq!(format_one_decimal_percent(18_000, 23_000), "78.3%");
        assert_eq!(format_one_decimal_percent(1, 3), "33.3%");
        assert_eq!(format_one_decimal_percent(1, 1), "100.0%");
        assert_eq!(format_one_decimal_percent(0, 100), "0.0%");
        // Defensive: zero denominator must not divide-by-zero or NaN
        // its way into the rendered output.
        assert_eq!(format_one_decimal_percent(0, 0), "0.0%");
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

    // ---------- usage rollup ----------

    use flowd_core::orchestration::executor::{AgentMetrics, ModelUsage};
    use flowd_core::orchestration::observer::{PlanEvent, PlanStepCounts};

    fn ts(secs_from_epoch: i64) -> DateTime<Utc> {
        DateTime::<Utc>::from_timestamp(secs_from_epoch, 0).expect("valid timestamp")
    }

    fn step_event(
        plan_id: Uuid,
        step_id: &str,
        kind_str: &str,
        metrics: Option<AgentMetrics>,
        created_at: DateTime<Utc>,
    ) -> StoredPlanEvent {
        // Mirror what `event_payload` writes for step variants so tests
        // exercise the same JSON shape `build_usage_report` reads in
        // production. Mismatches here would silently mask real bugs.
        let mut payload = match kind_str {
            "step_completed" => serde_json::json!({ "output": "ok" }),
            "step_failed" => serde_json::json!({ "error": "boom" }),
            other => panic!("step_event helper does not support kind {other}"),
        };
        if let Some(m) = metrics {
            payload
                .as_object_mut()
                .unwrap()
                .insert("metrics".into(), serde_json::to_value(m).unwrap());
        }
        StoredPlanEvent {
            id: 0,
            plan_id,
            project: "demo".into(),
            kind: kind_str.into(),
            step_id: Some(step_id.into()),
            agent_type: Some("claude".into()),
            payload,
            created_at,
        }
    }

    fn submitted_event(plan_id: Uuid, name: &str, created_at: DateTime<Utc>) -> StoredPlanEvent {
        StoredPlanEvent {
            id: 0,
            plan_id,
            project: "demo".into(),
            kind: "submitted".into(),
            step_id: None,
            agent_type: None,
            payload: serde_json::json!({ "name": name }),
            created_at,
        }
    }

    fn finished_event(plan_id: Uuid, status: &str, created_at: DateTime<Utc>) -> StoredPlanEvent {
        StoredPlanEvent {
            id: 0,
            plan_id,
            project: "demo".into(),
            kind: "finished".into(),
            step_id: None,
            agent_type: None,
            payload: serde_json::json!({
                "status": status,
                "step_count": { "completed": 1, "failed": 0 },
                "metrics": { "input_tokens": 999, "output_tokens": 999, "total_cost_usd": 9.99 },
            }),
            created_at,
        }
    }

    #[test]
    fn format_duration_ms_branches() {
        assert_eq!(format_duration_ms(0), "0ms");
        assert_eq!(format_duration_ms(750), "750ms");
        assert_eq!(format_duration_ms(1_500), "1.5s");
        assert_eq!(format_duration_ms(59_900), "59.9s");
        assert_eq!(format_duration_ms(60_000), "1m 0s");
        assert_eq!(format_duration_ms(125_000), "2m 5s");
        assert_eq!(format_duration_ms(3_725_000), "1h 2m 5s");
    }

    #[test]
    fn build_usage_report_sums_step_metrics_and_skips_finished() {
        let plan_id = Uuid::new_v4();
        let mut step_a = AgentMetrics {
            input_tokens: 100,
            output_tokens: 50,
            cache_read_input_tokens: 10,
            cache_creation_input_tokens: 5,
            total_cost_usd: 0.0123,
            duration_ms: 1_000,
            duration_api_ms: 800,
            ..AgentMetrics::default()
        };
        step_a.model_usage.insert(
            "sonnet".into(),
            ModelUsage {
                input_tokens: 100,
                output_tokens: 50,
                cache_read_input_tokens: 10,
                cache_creation_input_tokens: 5,
                cost_usd: 0.0123,
            },
        );
        let step_b_failed = AgentMetrics {
            input_tokens: 30,
            output_tokens: 0,
            total_cost_usd: 0.0010,
            duration_ms: 200,
            duration_api_ms: 180,
            ..AgentMetrics::default()
        };
        let events = vec![
            submitted_event(plan_id, "demo plan", ts(1_000_000)),
            step_event(plan_id, "a", "step_completed", Some(step_a), ts(1_000_010)),
            step_event(
                plan_id,
                "b",
                "step_failed",
                Some(step_b_failed),
                ts(1_000_020),
            ),
            // Pre-metrics step row -- payload has no `metrics`. Must be
            // ignored without poisoning the sums.
            step_event(plan_id, "legacy", "step_completed", None, ts(1_000_030)),
            // Finished row with its own metrics; including it would
            // double-count.
            finished_event(plan_id, "completed", ts(1_000_040)),
        ];

        let r = build_usage_report(plan_id, &events);
        assert_eq!(r.plan_id, plan_id);
        assert_eq!(r.plan_name.as_deref(), Some("demo plan"));
        assert_eq!(r.project.as_deref(), Some("demo"));
        assert_eq!(r.status.as_deref(), Some("completed"));
        // Two step rows contributed metrics; the legacy row had `metrics: None`
        // and the finished row is filtered by kind.
        assert_eq!(r.steps, 2);
        assert_eq!(r.input_tokens, 130);
        assert_eq!(r.output_tokens, 50);
        assert_eq!(r.cache_read_tokens, 10);
        assert_eq!(r.cache_creation_tokens, 5);
        assert!((r.total_cost_usd - 0.0133).abs() < 1e-9);
        assert_eq!(r.duration_api_ms, 980);
        assert_eq!(r.duration_total_ms, 1_200);
        assert_eq!(r.wall_clock_ms, Some(40_000));
        // 10 / (10 + 5) = 0.6666...
        assert!((r.cache_hit_rate.unwrap() - (10.0 / 15.0)).abs() < 1e-9);
        // Per-model: only sonnet had a model_usage entry.
        assert_eq!(r.model_breakdown.len(), 1);
        let sonnet = r.model_breakdown.get("sonnet").expect("sonnet bucket");
        assert_eq!(sonnet.input_tokens, 100);
        assert_eq!(sonnet.output_tokens, 50);
    }

    #[test]
    fn build_usage_report_no_metrics_yields_zeros_and_no_cache_rate() {
        let plan_id = Uuid::new_v4();
        let events = vec![submitted_event(plan_id, "p", ts(1_000_000))];
        let r = build_usage_report(plan_id, &events);
        assert_eq!(r.steps, 0);
        assert_eq!(r.input_tokens, 0);
        assert_eq!(r.output_tokens, 0);
        // Single event -> no wall-clock span (a 0ms wall-clock would
        // be misleading here).
        assert_eq!(r.wall_clock_ms, None);
        assert_eq!(r.cache_hit_rate, None);
        assert!(r.model_breakdown.is_empty());
    }

    #[test]
    fn build_usage_report_cache_hit_rate_is_none_when_no_cache_activity() {
        let plan_id = Uuid::new_v4();
        let m = AgentMetrics {
            input_tokens: 100,
            output_tokens: 50,
            total_cost_usd: 0.5,
            ..AgentMetrics::default()
        };
        let events = vec![step_event(
            plan_id,
            "a",
            "step_completed",
            Some(m),
            ts(2_000_000),
        )];
        let r = build_usage_report(plan_id, &events);
        assert_eq!(r.cache_hit_rate, None);
    }

    #[test]
    fn model_rows_by_cost_orders_descending_with_name_tiebreaker() {
        let mut report = UsageReport {
            plan_id: Uuid::nil(),
            ..UsageReport::default()
        };
        report.model_breakdown.insert(
            "haiku".into(),
            ModelBreakdown {
                cost_usd: 0.10,
                ..ModelBreakdown::default()
            },
        );
        report.model_breakdown.insert(
            "sonnet".into(),
            ModelBreakdown {
                cost_usd: 0.50,
                ..ModelBreakdown::default()
            },
        );
        report.model_breakdown.insert(
            "opus".into(),
            ModelBreakdown {
                cost_usd: 0.10,
                ..ModelBreakdown::default()
            },
        );
        let names: Vec<&str> = model_rows_by_cost(&report)
            .iter()
            .map(|(name, _)| name.as_str())
            .collect();
        // sonnet first (highest cost); opus before haiku as a name tie-break.
        assert_eq!(names, vec!["sonnet", "haiku", "opus"]);
    }

    #[test]
    fn format_usage_report_includes_required_fields() {
        let mut report = UsageReport {
            plan_id: Uuid::nil(),
            plan_name: Some("widget rollout".into()),
            project: Some("demo".into()),
            status: Some("completed".into()),
            steps: 4,
            total_cost_usd: 1.4231,
            input_tokens: 12_400,
            output_tokens: 21_300,
            cache_read_tokens: 18_000,
            cache_creation_tokens: 5_000,
            cache_hit_rate: Some(18_000.0 / 23_000.0),
            duration_api_ms: 24_700,
            duration_total_ms: 268_100,
            wall_clock_ms: Some(310_000),
            first_event_at: Some(ts(1_000_000)),
            last_event_at: Some(ts(1_000_310)),
            ..UsageReport::default()
        };
        report.model_breakdown.insert(
            "sonnet".into(),
            ModelBreakdown {
                cost_usd: 1.20,
                input_tokens: 10_000,
                output_tokens: 18_000,
                cache_read_tokens: 16_000,
                cache_creation_tokens: 4_000,
            },
        );
        let out = format_usage_report(&report, Style::plain());

        assert!(out.contains("=== usage:"));
        assert!(out.contains("plan:        widget rollout"));
        assert!(out.contains("project:     demo"));
        assert!(out.contains("status:      completed"));
        assert!(out.contains("steps:       4"));
        // Two-decimal cost rounding.
        assert!(out.contains("total cost:  $1.42"), "got:\n{out}");
        // Token compaction kicks in above 9_999.
        assert!(out.contains("tokens:      in 12.4k   out 21.3k"));
        // Raw cache totals with thousands separators.
        assert!(out.contains("cache:       read 18,000   creation 5,000"));
        // One-decimal cache reuse rate.
        assert!(out.contains("cache reuse: 78.3%"), "got:\n{out}");
        // Both runtime sums rendered.
        assert!(out.contains("runtime:     api 24.7s   agent 4m 28s"));
        assert!(out.contains("wall-clock:  5m 10s"));
        assert!(out.contains("models:"));
        assert!(out.contains("sonnet"));
    }

    #[test]
    fn format_usage_report_marks_no_cache_activity_when_rate_absent() {
        let report = UsageReport {
            plan_id: Uuid::nil(),
            steps: 1,
            total_cost_usd: 0.01,
            input_tokens: 10,
            output_tokens: 5,
            ..UsageReport::default()
        };
        let out = format_usage_report(&report, Style::plain());
        assert!(out.contains("cache reuse: (no cache activity)"));
        // No "models:" section when the breakdown is empty -- the
        // operator would otherwise see a dangling header.
        assert!(!out.contains("models:"));
        // No wall-clock line either.
        assert!(!out.contains("wall-clock:"));
    }

    #[test]
    fn usage_report_json_omits_optional_fields_when_absent() {
        // Empty rollup must not serialise `cache_hit_rate: null` or
        // `wall_clock_ms: null` -- callers should distinguish "unset"
        // from "explicitly zero" and we lean on `Option`-elision.
        let report = UsageReport {
            plan_id: Uuid::nil(),
            steps: 1,
            total_cost_usd: 0.01,
            input_tokens: 10,
            output_tokens: 5,
            ..UsageReport::default()
        };
        let s = serde_json::to_string(&report).unwrap();
        assert!(!s.contains("cache_hit_rate"));
        assert!(!s.contains("wall_clock_ms"));
        assert!(!s.contains("plan_name"));
        assert!(!s.contains("project"));
        assert!(!s.contains("status"));
        // Required scalars are always present.
        assert!(s.contains("\"plan_id\""));
        assert!(s.contains("\"steps\":1"));
        assert!(s.contains("\"total_cost_usd\":0.01"));
    }

    #[tokio::test]
    async fn usage_command_reads_recorded_events_end_to_end() {
        let dir = std::env::temp_dir().join(format!("flowd-plan-usage-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());

        // SqlitePlanEventStore::open writes to the same path the CLI
        // command will subsequently read from.
        let db_path = paths.db_file();
        let store = SqlitePlanEventStore::open(&db_path).expect("open store");

        let plan_id = Uuid::new_v4();
        let metrics = AgentMetrics {
            input_tokens: 200,
            output_tokens: 80,
            cache_read_input_tokens: 50,
            cache_creation_input_tokens: 50,
            total_cost_usd: 0.42,
            duration_ms: 12_000,
            duration_api_ms: 9_500,
            ..AgentMetrics::default()
        };
        store
            .record(&PlanEvent::Submitted {
                plan_id,
                name: "rollout".into(),
                project: "demo".into(),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::StepCompleted {
                plan_id,
                project: "demo".into(),
                step_id: "build".into(),
                agent_type: "claude".into(),
                output: "ok".into(),
                metrics: Some(metrics),
            })
            .await
            .unwrap();
        store
            .record(&PlanEvent::Finished {
                plan_id,
                project: "demo".into(),
                status: PlanStatus::Completed,
                total_metrics: None,
                step_count: PlanStepCounts {
                    completed: 1,
                    failed: 0,
                },
                elapsed_ms: None,
            })
            .await
            .unwrap();

        let report = prepare_usage_report(&paths, plan_id)
            .await
            .expect("prepare_usage_report");
        assert_eq!(report.steps, 1);
        assert_eq!(report.input_tokens, 200);
        assert_eq!(report.output_tokens, 80);
        assert_eq!(report.cache_read_tokens, 50);
        assert_eq!(report.cache_creation_tokens, 50);
        assert!((report.total_cost_usd - 0.42).abs() < 1e-9);
        assert_eq!(report.duration_total_ms, 12_000);
        assert_eq!(report.duration_api_ms, 9_500);
        // 50 / (50 + 50) = 0.5 -> renders 50.0%.
        assert!((report.cache_hit_rate.unwrap() - 0.5).abs() < 1e-9);
        assert_eq!(report.plan_name.as_deref(), Some("rollout"));
        assert_eq!(report.status.as_deref(), Some("completed"));

        // Run the user-facing entry point as well to make sure the
        // dispatch path doesn't regress (output goes to stdout; we just
        // assert the call resolves).
        usage(&paths, Style::plain(), plan_id.to_string(), false)
            .await
            .expect("usage human");
        usage(&paths, Style::plain(), plan_id.to_string(), true)
            .await
            .expect("usage json");

        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn usage_command_errors_when_plan_has_no_events() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-usage-empty-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        // Initialise the schema by opening the store without writing.
        let _ = SqlitePlanEventStore::open(&paths.db_file()).expect("open store");

        let err = prepare_usage_report(&paths, Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("no events recorded"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn usage_command_errors_when_db_missing() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-usage-nodb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        let err = prepare_usage_report(&paths, Uuid::new_v4())
            .await
            .unwrap_err();
        assert!(format!("{err:#}").contains("no flowd database"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    // -------- `plan integrate` argument parsing -------------------------

    fn parse_integrate(args: &[&str]) -> Result<crate::cli::PlanAction, clap::Error> {
        use crate::cli::{Cli, Command};
        use clap::Parser;
        let mut full: Vec<&str> = vec!["flowd", "plan", "integrate"];
        full.extend_from_slice(args);
        let cli = Cli::try_parse_from(full)?;
        match cli.command {
            Command::Plan { action } => Ok(action),
            other => unreachable!("expected Command::Plan, got {other:?}"),
        }
    }

    #[test]
    fn integrate_minimal_args_parse_with_defaults() {
        let id = Uuid::new_v4();
        let action = parse_integrate(&[&id.to_string(), "--base", "main"]).expect("parse");
        match action {
            crate::cli::PlanAction::Integrate {
                plan_id,
                base,
                promote,
                dry_run,
                discard,
                cleanup,
                strategy,
                verify,
                json,
            } => {
                assert_eq!(plan_id, id.to_string());
                assert_eq!(base, "main");
                assert!(!promote);
                assert!(!dry_run);
                assert!(!discard);
                assert_eq!(cleanup, "keep_on_failure");
                assert_eq!(strategy, "tip-cherry-pick");
                assert!(verify.is_none(), "verify must default to None");
                assert!(!json);
            }
            other => panic!("expected Integrate, got {other:?}"),
        }
    }

    #[test]
    fn integrate_verify_flag_round_trips_as_string() {
        let id = Uuid::new_v4();
        let action = parse_integrate(&[
            &id.to_string(),
            "--base",
            "main",
            "--verify",
            "cargo nextest run -p flowd-cli",
        ])
        .expect("parse");
        let crate::cli::PlanAction::Integrate { verify, .. } = action else {
            panic!("expected Integrate");
        };
        assert_eq!(
            verify.as_deref(),
            Some("cargo nextest run -p flowd-cli"),
            "raw verify string must round-trip; whitespace splitting happens in parse_verify_argv"
        );
    }

    #[test]
    fn parse_verify_argv_splits_on_whitespace_and_handles_empty() {
        let cfg = parse_verify_argv(Some("cargo nextest run -p flowd-cli"));
        assert!(cfg.is_enabled());
        assert_eq!(
            cfg.command.as_deref().unwrap(),
            ["cargo", "nextest", "run", "-p", "flowd-cli"]
        );
        // Empty string => disabled, not an empty argv. Same for None.
        assert!(!parse_verify_argv(Some("")).is_enabled());
        assert!(!parse_verify_argv(Some("   \t  ")).is_enabled());
        assert!(!parse_verify_argv(None).is_enabled());
    }

    #[test]
    fn integrate_promote_flag_round_trips() {
        let id = Uuid::new_v4();
        let action =
            parse_integrate(&[&id.to_string(), "--base", "main", "--promote"]).expect("parse");
        let crate::cli::PlanAction::Integrate {
            promote, dry_run, ..
        } = action
        else {
            panic!("expected Integrate");
        };
        assert!(promote);
        assert!(!dry_run);
    }

    #[test]
    fn integrate_dry_run_flag_round_trips() {
        let id = Uuid::new_v4();
        let action =
            parse_integrate(&[&id.to_string(), "--base", "main", "--dry-run"]).expect("parse");
        let crate::cli::PlanAction::Integrate {
            dry_run, promote, ..
        } = action
        else {
            panic!("expected Integrate");
        };
        assert!(dry_run);
        assert!(!promote);
    }

    #[test]
    fn integrate_dry_run_and_promote_are_mutually_exclusive() {
        let id = Uuid::new_v4();
        let err = parse_integrate(&[&id.to_string(), "--base", "main", "--promote", "--dry-run"])
            .unwrap_err();
        // clap's conflict error is rendered to stderr; the typed kind
        // is `ArgumentConflict`. Regardless, parsing must fail.
        assert!(
            err.kind() == clap::error::ErrorKind::ArgumentConflict,
            "expected ArgumentConflict, got {:?}",
            err.kind()
        );
    }

    #[test]
    fn integrate_requires_base_branch() {
        let id = Uuid::new_v4();
        let err = parse_integrate(&[&id.to_string()]).unwrap_err();
        assert_eq!(err.kind(), clap::error::ErrorKind::MissingRequiredArgument);
    }

    #[test]
    fn integrate_accepts_cleanup_alias_forms() {
        let id = Uuid::new_v4();
        let action = parse_integrate(&[
            &id.to_string(),
            "--base",
            "main",
            "--cleanup",
            "drop_always",
        ])
        .expect("parse");
        let crate::cli::PlanAction::Integrate { cleanup, .. } = action else {
            panic!("expected Integrate");
        };
        assert_eq!(cleanup, "drop_always");
    }

    // -------- Cleanup / strategy / mode parsers -------------------------

    #[test]
    fn parse_cleanup_policy_accepts_both_separators_per_variant() {
        assert!(matches!(
            parse_cleanup_policy("keep_on_failure").unwrap(),
            CleanupPolicy::KeepOnFailure
        ));
        assert!(matches!(
            parse_cleanup_policy("KEEP-ON-FAILURE").unwrap(),
            CleanupPolicy::KeepOnFailure
        ));
        assert!(matches!(
            parse_cleanup_policy("keep_always").unwrap(),
            CleanupPolicy::KeepAlways
        ));
        assert!(matches!(
            parse_cleanup_policy("drop-always").unwrap(),
            CleanupPolicy::DropAlways
        ));
    }

    #[test]
    fn parse_cleanup_policy_rejects_unknown_value() {
        let err = parse_cleanup_policy("yolo").unwrap_err();
        assert!(format!("{err:#}").contains("yolo"), "{err:#}");
    }

    #[test]
    fn parse_integrate_strategy_only_accepts_tip_cherry_pick() {
        parse_integrate_strategy("tip-cherry-pick").unwrap();
        parse_integrate_strategy("tip_cherry_pick").unwrap();
        let err = parse_integrate_strategy("rebase-merge").unwrap_err();
        assert!(format!("{err:#}").contains("rebase-merge"), "{err:#}");
    }

    #[test]
    fn parse_integrate_mode_picks_dry_run_over_promote_when_both_set() {
        // The clap layer rejects `--dry-run --promote` together (see
        // `integrate_dry_run_and_promote_are_mutually_exclusive`), but
        // the helper still pins a deterministic policy if it is ever
        // called with both bits set programmatically.
        assert_eq!(parse_integrate_mode(true, true), IntegrationMode::DryRun);
        assert_eq!(parse_integrate_mode(true, false), IntegrationMode::DryRun);
        assert_eq!(parse_integrate_mode(false, true), IntegrationMode::Confirm);
        assert_eq!(parse_integrate_mode(false, false), IntegrationMode::Confirm);
    }

    // -------- Event payload renderers -----------------------------------

    #[test]
    fn integration_started_payload_renders_all_fields() {
        let payload = serde_json::json!({
            "integration_branch": "flowd-integrate/proj/abc",
            "base_branch": "main",
            "mode": "confirm",
        });
        let line = format_integration_started(&payload);
        assert_eq!(
            line,
            "integration_branch=flowd-integrate/proj/abc  base=main  mode=confirm"
        );
    }

    #[test]
    fn integration_succeeded_payload_omits_promoted_tip_when_absent() {
        let payload = serde_json::json!({
            "integration_branch": "flowd-integrate/proj/abc",
            "base_branch": "main",
            "status": "staged",
        });
        let line = format_integration_succeeded(&payload);
        assert_eq!(
            line,
            "integration_branch=flowd-integrate/proj/abc  base=main  status=staged"
        );
    }

    #[test]
    fn integration_succeeded_payload_appends_promoted_tip_when_present() {
        let payload = serde_json::json!({
            "integration_branch": "flowd-integrate/proj/abc",
            "base_branch": "main",
            "status": "promoted",
            "promoted_tip": "deadbeef",
        });
        let line = format_integration_succeeded(&payload);
        assert!(line.contains("status=promoted"), "{line}");
        assert!(line.contains("promoted_tip=deadbeef"), "{line}");
    }

    #[test]
    fn integration_failed_payload_carries_reason() {
        let payload = serde_json::json!({
            "integration_branch": "flowd-integrate/proj/abc",
            "base_branch": "main",
            "reason": "cherry-pick conflict on step `b`",
        });
        let line = format_integration_failed(&payload);
        assert!(
            line.contains("reason=cherry-pick conflict on step `b`"),
            "{line}"
        );
    }

    #[test]
    fn describe_payload_dispatches_integration_kinds() {
        use flowd_core::orchestration::plan_events::kind as k;

        let started = serde_json::json!({
            "integration_branch": "i",
            "base_branch": "main",
            "mode": "confirm",
        });
        assert!(describe_payload(k::INTEGRATION_STARTED, &started).is_some());

        let succeeded = serde_json::json!({
            "integration_branch": "i",
            "base_branch": "main",
            "status": "promoted",
        });
        assert!(describe_payload(k::INTEGRATION_SUCCEEDED, &succeeded).is_some());

        let failed = serde_json::json!({
            "integration_branch": "i",
            "base_branch": "main",
            "reason": "boom",
        });
        assert!(describe_payload(k::INTEGRATION_FAILED, &failed).is_some());
    }

    // -------- Outcome → plan metadata mapping ---------------------------

    fn fixture_request(plan_id: Uuid, base: &str) -> PlanIntegrateRequest {
        PlanIntegrateRequest::new(
            plan_id,
            IntegrationMode::Confirm,
            IntegrationConfig {
                base_branch: base.into(),
                cleanup: CleanupPolicy::DropAlways,
                verify: VerificationConfig::default(),
            },
        )
        .expect("fixture request")
    }

    fn fixture_completed_plan() -> Plan {
        use flowd_core::orchestration::{PlanStep, StepStatus};
        let step = PlanStep {
            id: "a".into(),
            agent_type: "echo".into(),
            prompt: "do a".into(),
            depends_on: Vec::new(),
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Completed,
            output: Some("ok".into()),
            error: None,
            started_at: None,
            completed_at: None,
        };
        let mut p = Plan::new("p", "proj", vec![step]);
        p.status = PlanStatus::Completed;
        p
    }

    #[test]
    fn apply_outcome_dry_run_only_synthesises_metadata_when_absent() {
        use flowd_core::orchestration::IntegrationPlan;

        let mut plan = fixture_completed_plan();
        let req = fixture_request(plan.id, "main");
        let outcome = PlanIntegrateOutcome::DryRun {
            intended: IntegrationPlan {
                plan_id: plan.id,
                project: plan.project.clone(),
                base_branch: "main".into(),
                integration_branch: "flowd-integrate/proj/abc".into(),
                cherry_picks: Vec::new(),
                mode: IntegrationMode::DryRun,
                cleanup: CleanupPolicy::DropAlways,
                verify: VerificationConfig::default(),
            },
        };
        apply_outcome_to_plan(&mut plan, &req, &outcome);
        let meta = plan.integration.as_ref().expect("metadata");
        assert_eq!(meta.status, IntegrationStatus::Pending);
        assert_eq!(meta.integration_branch, "flowd-integrate/proj/abc");

        // A pre-existing `Staged` block must not be downgraded by a
        // subsequent dry-run -- preview never overwrites prior state.
        plan.integration.as_mut().unwrap().status = IntegrationStatus::Staged;
        apply_outcome_to_plan(&mut plan, &req, &outcome);
        assert_eq!(
            plan.integration.as_ref().unwrap().status,
            IntegrationStatus::Staged
        );
    }

    #[test]
    fn apply_outcome_staged_marks_plan_metadata() {
        use flowd_core::orchestration::IntegrationPlan;

        let mut plan = fixture_completed_plan();
        let req = fixture_request(plan.id, "main");
        let outcome = PlanIntegrateOutcome::Staged {
            intended: IntegrationPlan {
                plan_id: plan.id,
                project: plan.project.clone(),
                base_branch: "main".into(),
                integration_branch: "flowd-integrate/proj/abc".into(),
                cherry_picks: Vec::new(),
                mode: IntegrationMode::Confirm,
                cleanup: CleanupPolicy::DropAlways,
                verify: VerificationConfig::default(),
            },
            integration_tip: "deadbeef".into(),
        };
        apply_outcome_to_plan(&mut plan, &req, &outcome);
        let meta = plan.integration.expect("metadata");
        assert_eq!(meta.status, IntegrationStatus::Staged);
        assert_eq!(meta.cleanup, CleanupPolicy::DropAlways);
        assert!(meta.completed_at.is_some());
        assert!(meta.failure.is_none());
        assert!(meta.refusal.is_none());
    }

    #[test]
    fn apply_outcome_promoted_marks_plan_metadata() {
        use flowd_core::orchestration::IntegrationPlan;

        let mut plan = fixture_completed_plan();
        let req = fixture_request(plan.id, "main");
        let outcome = PlanIntegrateOutcome::Promoted {
            intended: IntegrationPlan {
                plan_id: plan.id,
                project: plan.project.clone(),
                base_branch: "main".into(),
                integration_branch: "flowd-integrate/proj/abc".into(),
                cherry_picks: Vec::new(),
                mode: IntegrationMode::Confirm,
                cleanup: CleanupPolicy::KeepOnFailure,
                verify: VerificationConfig::default(),
            },
            promoted_tip: "cafef00d".into(),
        };
        apply_outcome_to_plan(&mut plan, &req, &outcome);
        let meta = plan.integration.expect("metadata");
        assert_eq!(meta.status, IntegrationStatus::Promoted);
    }

    #[test]
    fn mark_integration_failure_records_refusal_and_failure_separately() {
        use flowd_core::orchestration::{IntegrationFailure, IntegrationRefusal};

        let mut plan = fixture_completed_plan();
        let refusal = IntegrationRefusal::PlanStatus {
            observed: PlanStatus::Failed,
        };
        mark_integration_failure(&mut plan, &IntegrateError::Refusal(refusal.clone()));
        let meta = plan.integration.as_ref().unwrap();
        assert_eq!(meta.status, IntegrationStatus::Failed);
        assert!(meta.refusal.is_some());
        assert!(meta.failure.is_none());

        let failure = IntegrationFailure::DirtyBase;
        mark_integration_failure(&mut plan, &IntegrateError::Failure(failure.clone()));
        let meta = plan.integration.as_ref().unwrap();
        assert!(meta.failure.is_some());
        assert!(meta.refusal.is_none());
    }

    #[test]
    fn render_integration_metadata_does_not_panic_on_terminal_states() {
        let meta = IntegrationMetadata {
            status: IntegrationStatus::Promoted,
            integration_branch: "flowd-integrate/proj/abc".into(),
            base_branch: "main".into(),
            mode: IntegrationMode::Confirm,
            cleanup: CleanupPolicy::DropAlways,
            started_at: Some(Utc::now()),
            completed_at: Some(Utc::now()),
            failure: None,
            refusal: None,
        };
        render_integration_metadata(&meta, Style::plain());
    }

    #[test]
    fn integration_status_label_covers_every_variant() {
        // Compile-time guard: a new `IntegrationStatus` variant must be
        // wired into the renderer or this match panics on the Default.
        for status in [
            IntegrationStatus::Pending,
            IntegrationStatus::InProgress,
            IntegrationStatus::Staged,
            IntegrationStatus::Promoted,
            IntegrationStatus::Failed,
        ] {
            let label = integration_status_label(status);
            assert!(!label.is_empty(), "{status:?} -> empty label");
        }
    }
}
