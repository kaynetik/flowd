//! `flowd status` -- daemon liveness + DB stats.
//!
//! DB stats are computed with a single read on `SQLite`; we deliberately
//! use raw counts per tier rather than the `MemoryService` wrapper so no
//! embedder / vector index is required for an offline status view.

use anyhow::{Context, Result};
use chrono::{DateTime, TimeZone, Utc};

use flowd_core::memory::MemoryBackend;
use flowd_core::types::MemoryTier;
use flowd_mcp::observer::ObserverHealth;
use flowd_storage::plan_event_store::{SqlitePlanEventStore, UsageTotals};
use flowd_storage::sqlite::SqliteBackend;

use crate::daemon::{is_alive, read_pid};
use crate::output::{Style, banner};
use crate::paths::FlowdPaths;

/// Snapshots older than this are flagged as stale (the daemon is most
/// likely stopped). Picked at 3x the default 1s health-write interval
/// plus a fudge factor so a momentarily-busy daemon doesn't show stale.
const HEALTH_STALE_AFTER_SECS: i64 = 30;

pub async fn run(paths: &FlowdPaths, style: Style) -> Result<()> {
    print!("{}", banner("flowd status", style));

    print_home(paths, style);
    print_daemon(paths, style)?;
    print_db(paths, style).await?;
    print_token_usage(paths, style).await?;
    print_plan_event_observer(paths, style);

    Ok(())
}

fn print_home(paths: &FlowdPaths, style: Style) {
    println!("{} {}", style.bold("home:"), paths.home.display());
    let exists = paths.home.is_dir();
    println!(
        "  dir:       {}",
        if exists {
            style.green("present")
        } else {
            style.yellow("missing")
        }
    );
    println!("  db:        {}", paths.db_file().display());
    println!("  rules:     {}", paths.rules_dir().display());
    println!("  models:    {}", paths.model_dir().display());
}

fn print_daemon(paths: &FlowdPaths, style: Style) -> Result<()> {
    println!("\n{}", style.bold("daemon:"));
    let pid_path = paths.pid_file();
    match read_pid(&pid_path)? {
        None => {
            println!(
                "  {}  no pid file at {}",
                style.dim("stopped"),
                pid_path.display()
            );
        }
        Some(pid) => {
            if is_alive(pid) {
                println!(
                    "  {}  pid {pid} (from {})",
                    style.green("running"),
                    pid_path.display()
                );
            } else {
                println!(
                    "  {}  stale pid {pid} in {} -- run `flowd stop` to clean up",
                    style.yellow("stale"),
                    pid_path.display()
                );
            }
        }
    }
    Ok(())
}

async fn print_db(paths: &FlowdPaths, style: Style) -> Result<()> {
    println!("\n{}", style.bold("database:"));
    if !paths.db_file().exists() {
        println!(
            "  {}  not initialised; run `flowd start` once",
            style.dim("missing")
        );
        return Ok(());
    }

    let db = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    let sessions = db.list_sessions(None).await.context("list sessions")?;
    let now = Utc::now();
    let hot = db
        .list_by_tier_and_age(MemoryTier::Hot, now)
        .await
        .context("count hot tier")?
        .len();
    let warm = db
        .list_by_tier_and_age(MemoryTier::Warm, now)
        .await
        .context("count warm tier")?
        .len();
    let cold = db
        .list_by_tier_and_age(MemoryTier::Cold, now)
        .await
        .context("count cold tier")?
        .len();

    println!("  sessions:  {}", sessions.len());
    println!("  obs hot:   {hot}");
    println!("  obs warm:  {warm}");
    println!("  obs cold:  {cold}");
    println!("  obs total: {}", hot + warm + cold);

    if let Ok(meta) = std::fs::metadata(paths.db_file()) {
        println!("  size:      {} bytes", meta.len());
    }
    Ok(())
}

/// Render cross-plan token + cost totals, summed straight from the
/// `plan_events` table via `SqlitePlanEventStore::usage_totals`.
///
/// Token counts are split (input / output / `cache_read` / `cache_creation`)
/// to surface where spend actually lives -- output tokens are roughly
/// 5x the price of input on Anthropic's rate card, and `cache_read` is
/// a fraction of either, so the distribution matters more than the bare
/// total. The cost line is a single bundled `$X.XXXX` because that is
/// what the provider returns: Anthropic's `total_cost_usd` is per
/// request, not per direction. A per-direction USD split would require
/// a client-side rate card (deferred work; multi-provider).
async fn print_token_usage(paths: &FlowdPaths, style: Style) -> Result<()> {
    println!("\n{}", style.bold("tokens:"));
    if !paths.db_file().exists() {
        println!("  {}  no database yet", style.dim("absent"));
        return Ok(());
    }

    let store = SqlitePlanEventStore::open(&paths.db_file())
        .with_context(|| format!("open plan_events at {}", paths.db_file().display()))?;
    let totals = store
        .usage_totals()
        .await
        .context("aggregate usage totals")?;

    if totals == UsageTotals::default() {
        println!(
            "  {}  no metrics-bearing step events yet",
            style.dim("none")
        );
        return Ok(());
    }

    println!(
        "  input:          {}",
        format_thousands(totals.input_tokens)
    );
    println!(
        "  output:         {}",
        format_thousands(totals.output_tokens)
    );
    println!(
        "  cache_read:     {}",
        format_thousands(totals.cache_read_tokens)
    );
    println!(
        "  cache_creation: {}",
        format_thousands(totals.cache_creation_tokens)
    );
    println!("  cost:           ${:.4}", totals.total_cost_usd);
    println!("  step events:    {}", totals.step_events);
    Ok(())
}

/// Render `n` with ASCII thousands separators (e.g. `15_820 -> "15,820"`).
/// Hand-rolled to keep `flowd-cli`'s dependency graph flat -- the same
/// helper exists in `commands::plan` for per-event rendering; duplicated
/// here on purpose to avoid a new shared `output::format` module that
/// would have exactly two callers.
fn format_thousands(n: u64) -> String {
    let s = n.to_string();
    let bytes = s.as_bytes();
    let mut out = String::with_capacity(s.len() + s.len() / 3);
    for (i, b) in bytes.iter().enumerate() {
        if i > 0 && (bytes.len() - i) % 3 == 0 {
            out.push(',');
        }
        out.push(*b as char);
    }
    out
}

/// Render the plan-event observer health snapshot the daemon writes to
/// disk every ~1s (HL-40). The CLI is a separate process from the
/// daemon, so the file is the simplest IPC: present + fresh => daemon
/// is healthy; missing or stale => daemon is stopped.
fn print_plan_event_observer(paths: &FlowdPaths, style: Style) {
    println!("\n{}", style.bold("plan_events:"));
    let path = paths.plan_event_health_file();
    let raw = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            println!(
                "  {}  no health snapshot at {} (daemon never started?)",
                style.dim("absent"),
                path.display()
            );
            return;
        }
        Err(e) => {
            println!(
                "  {}  failed to read {}: {e}",
                style.yellow("error"),
                path.display()
            );
            return;
        }
    };
    let snapshot: ObserverHealth = match serde_json::from_slice(&raw) {
        Ok(s) => s,
        Err(e) => {
            println!(
                "  {}  could not parse {}: {e}",
                style.yellow("error"),
                path.display()
            );
            return;
        }
    };

    let now = Utc::now();
    #[allow(clippy::cast_possible_wrap)]
    let updated_at_secs = snapshot.updated_at as i64;
    let updated_at = Utc.timestamp_opt(updated_at_secs, 0).single();
    let snapshot_stale =
        updated_at.is_none_or(|ts| (now - ts).num_seconds() > HEALTH_STALE_AFTER_SECS);

    let dropped_label = if snapshot.dropped == 0 {
        style.green("0")
    } else {
        style.yellow(&format!("{}", snapshot.dropped))
    };

    println!("  capacity:  {}", snapshot.capacity);
    println!(
        "  in_flight: {} ({:.1}% utilised)",
        snapshot.in_flight,
        utilisation_pct(snapshot.in_flight, snapshot.capacity),
    );
    println!("  dropped:   {dropped_label}");
    match (updated_at, snapshot_stale) {
        (Some(ts), false) => println!("  updated:   {} (fresh)", format_ts(ts)),
        (Some(ts), true) => println!(
            "  updated:   {} ({}; daemon likely stopped)",
            format_ts(ts),
            style.dim("stale")
        ),
        (None, _) => println!("  updated:   <invalid timestamp>"),
    }
}

fn format_ts(ts: DateTime<Utc>) -> String {
    ts.format("%Y-%m-%dT%H:%M:%SZ").to_string()
}

#[allow(clippy::cast_precision_loss)]
fn utilisation_pct(in_flight: usize, capacity: usize) -> f64 {
    if capacity == 0 {
        return 0.0;
    }
    (in_flight as f64 / capacity as f64) * 100.0
}
