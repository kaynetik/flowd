//! `flowd status` -- daemon liveness + DB stats.
//!
//! DB stats are computed with a single read on `SQLite`; we deliberately
//! use raw counts per tier rather than the `MemoryService` wrapper so no
//! embedder / vector index is required for an offline status view.

use anyhow::{Context, Result};
use chrono::{DateTime, Datelike, Days, NaiveDate, TimeZone, Utc};

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
/// Costs are rendered to two decimals -- this is operator-facing summary
/// output, and `$0.0001` precision is visual noise here. The detailed
/// `flowd plan events <id>` view keeps four-decimal precision on the
/// per-step `cost:` and `total:` lines for spot-check work.
///
/// Period rollups (week prominently, day + month compactly, all-time
/// last) are computed against the indexed `created_at` column via
/// [`SqlitePlanEventStore::usage_totals_for_period`]. Each is a single
/// covered SQL query, so the four lines together stay cheap even on
/// large `plan_events` tables. Token counts come from the all-time
/// totals only -- per-period token splits would dilute the spend
/// signal and add lines without changing operator decisions.
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

    let periods = compute_periods(Utc::now());
    let week = store
        .usage_totals_for_period(periods.week.0, periods.week.1)
        .await
        .context("aggregate week usage totals")?;
    let day = store
        .usage_totals_for_period(periods.day.0, periods.day.1)
        .await
        .context("aggregate day usage totals")?;
    let month = store
        .usage_totals_for_period(periods.month.0, periods.month.1)
        .await
        .context("aggregate month usage totals")?;

    print!(
        "{}",
        format_period_block(&PeriodSpend {
            day,
            week,
            month,
            all_time: totals,
        })
    );
    Ok(())
}

/// Snapshot of the four spend windows status reports on, in the order
/// they're rendered. Bundled into a struct so the renderer can be tested
/// without hitting a database.
#[derive(Debug, Clone, PartialEq)]
struct PeriodSpend {
    day: UsageTotals,
    week: UsageTotals,
    month: UsageTotals,
    all_time: UsageTotals,
}

/// Half-open `[start, end)` ranges, all UTC, lined up with the rows
/// `SQLite` writes via `datetime('now')` (which is itself UTC).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Periods {
    day: (DateTime<Utc>, DateTime<Utc>),
    week: (DateTime<Utc>, DateTime<Utc>),
    month: (DateTime<Utc>, DateTime<Utc>),
}

/// Compute the day / ISO-week / month windows that contain `now`.
///
/// All three windows are half-open `[start, end)`: a step event recorded
/// exactly at `start` belongs to the period; one recorded exactly at
/// `end` belongs to the next period. ISO-week starts on Monday so the
/// rollup matches what most operators (and Anthropic's own dashboard)
/// expect; computed in UTC so the boundaries match the storage column
/// and don't drift with the daemon's local TZ.
fn compute_periods(now: DateTime<Utc>) -> Periods {
    let date = now.date_naive();
    let day_start = naive_to_utc_midnight(date);
    let next_day = naive_to_utc_midnight(
        date.checked_add_days(Days::new(1))
            .unwrap_or(NaiveDate::MAX),
    );

    let days_from_monday = i64::from(date.weekday().num_days_from_monday());
    let week_start_date = date
        .checked_sub_days(Days::new(u64::try_from(days_from_monday).unwrap_or(0)))
        .unwrap_or(date);
    let week_start = naive_to_utc_midnight(week_start_date);
    let next_week = naive_to_utc_midnight(
        week_start_date
            .checked_add_days(Days::new(7))
            .unwrap_or(NaiveDate::MAX),
    );

    let month_start = naive_to_utc_midnight(
        NaiveDate::from_ymd_opt(date.year(), date.month(), 1).unwrap_or(date),
    );
    let (next_year, next_month) = if date.month() == 12 {
        (date.year() + 1, 1)
    } else {
        (date.year(), date.month() + 1)
    };
    let next_month_start = naive_to_utc_midnight(
        NaiveDate::from_ymd_opt(next_year, next_month, 1).unwrap_or(NaiveDate::MAX),
    );

    Periods {
        day: (day_start, next_day),
        week: (week_start, next_week),
        month: (month_start, next_month_start),
    }
}

fn naive_to_utc_midnight(date: NaiveDate) -> DateTime<Utc> {
    // `and_hms_opt(0, 0, 0)` only returns `None` for out-of-range
    // wall-clock times; midnight is always valid for every NaiveDate
    // chrono can construct, so the expect is unreachable in practice.
    let naive = date.and_hms_opt(0, 0, 0).expect("midnight is always valid");
    DateTime::<Utc>::from_naive_utc_and_offset(naive, Utc)
}

/// Build the `tokens:` body block. Returned as a `String` rather than
/// streamed to stdout so the layout can be unit-tested without capturing
/// the process's stdout. Costs round to two decimals (summary view); the
/// per-step detail in `flowd plan events` keeps four-decimal precision.
fn format_period_block(spend: &PeriodSpend) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "  this week:      ${:.2}  ({} step events)",
        spend.week.total_cost_usd, spend.week.step_events
    );
    let _ = writeln!(out, "  today:          ${:.2}", spend.day.total_cost_usd);
    let _ = writeln!(out, "  this month:     ${:.2}", spend.month.total_cost_usd);
    let _ = writeln!(
        out,
        "  all-time:       ${:.2}  ({} step events)",
        spend.all_time.total_cost_usd, spend.all_time.step_events
    );

    let _ = writeln!(
        out,
        "  input:          {}",
        format_thousands(spend.all_time.input_tokens)
    );
    let _ = writeln!(
        out,
        "  output:         {}",
        format_thousands(spend.all_time.output_tokens)
    );
    let _ = writeln!(
        out,
        "  cache_read:     {}",
        format_thousands(spend.all_time.cache_read_tokens)
    );
    let _ = writeln!(
        out,
        "  cache_creation: {}",
        format_thousands(spend.all_time.cache_creation_tokens)
    );
    out
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

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn ymd_hms(
        year: i32,
        month: u32,
        day: u32,
        hour: u32,
        minute: u32,
        second: u32,
    ) -> DateTime<Utc> {
        Utc.with_ymd_and_hms(year, month, day, hour, minute, second)
            .single()
            .expect("valid timestamp")
    }

    #[test]
    fn compute_periods_anchors_day_to_utc_midnight() {
        // Mid-afternoon Wednesday: today's window is the full UTC day,
        // not "now minus 24h".
        let now = ymd_hms(2026, 4, 15, 14, 30, 0);
        let p = compute_periods(now);
        assert_eq!(p.day.0, ymd_hms(2026, 4, 15, 0, 0, 0));
        assert_eq!(p.day.1, ymd_hms(2026, 4, 16, 0, 0, 0));
    }

    #[test]
    fn compute_periods_anchors_week_to_monday() {
        // Wednesday 2026-04-15 -> ISO week starts Monday 2026-04-13.
        let now = ymd_hms(2026, 4, 15, 14, 30, 0);
        let p = compute_periods(now);
        assert_eq!(p.week.0, ymd_hms(2026, 4, 13, 0, 0, 0));
        assert_eq!(p.week.1, ymd_hms(2026, 4, 20, 0, 0, 0));
    }

    #[test]
    fn compute_periods_week_on_monday_starts_today() {
        // Monday: weekday().num_days_from_monday() == 0, so the window
        // starts at today's UTC midnight rather than rolling back seven
        // days into the previous week.
        let monday = ymd_hms(2026, 4, 13, 9, 0, 0);
        let p = compute_periods(monday);
        assert_eq!(p.week.0, ymd_hms(2026, 4, 13, 0, 0, 0));
        assert_eq!(p.week.1, ymd_hms(2026, 4, 20, 0, 0, 0));
    }

    #[test]
    fn compute_periods_week_on_sunday_includes_full_week() {
        // Sunday 2026-04-19 is the *last* day of an ISO week starting
        // Monday 2026-04-13. The window must cover the whole week,
        // not collapse to a one-day Sunday slice.
        let sunday = ymd_hms(2026, 4, 19, 23, 59, 30);
        let p = compute_periods(sunday);
        assert_eq!(p.week.0, ymd_hms(2026, 4, 13, 0, 0, 0));
        assert_eq!(p.week.1, ymd_hms(2026, 4, 20, 0, 0, 0));
    }

    #[test]
    fn compute_periods_anchors_month_to_first() {
        let now = ymd_hms(2026, 4, 15, 14, 30, 0);
        let p = compute_periods(now);
        assert_eq!(p.month.0, ymd_hms(2026, 4, 1, 0, 0, 0));
        assert_eq!(p.month.1, ymd_hms(2026, 5, 1, 0, 0, 0));
    }

    #[test]
    fn compute_periods_month_wraps_at_year_boundary() {
        // December must roll into January of the next year, not
        // month 13 of the same year (which would panic in chrono).
        let dec = ymd_hms(2026, 12, 20, 14, 0, 0);
        let p = compute_periods(dec);
        assert_eq!(p.month.0, ymd_hms(2026, 12, 1, 0, 0, 0));
        assert_eq!(p.month.1, ymd_hms(2027, 1, 1, 0, 0, 0));
    }

    #[test]
    fn compute_periods_handles_first_of_month() {
        // 1 April: month_start is today, next_month_start is 1 May.
        // The week computation must keep working too even though it's
        // a Wednesday in the test calendar.
        let first = ymd_hms(2026, 4, 1, 0, 0, 1);
        let p = compute_periods(first);
        assert_eq!(p.month.0, ymd_hms(2026, 4, 1, 0, 0, 0));
        assert_eq!(p.month.1, ymd_hms(2026, 5, 1, 0, 0, 0));
        assert_eq!(p.day.0, ymd_hms(2026, 4, 1, 0, 0, 0));
        assert_eq!(p.day.1, ymd_hms(2026, 4, 2, 0, 0, 0));
    }

    #[test]
    fn format_period_block_rounds_costs_to_two_decimals() {
        // 0.12345 must round to 0.12; 0.005 to 0.01 (Rust's round-half
        // -to-even on f64::format produces 0.01 here, but the salient
        // assertion is just "two decimals, no fourth-place noise").
        let spend = PeriodSpend {
            day: UsageTotals {
                total_cost_usd: 0.12345,
                step_events: 1,
                ..UsageTotals::default()
            },
            week: UsageTotals {
                total_cost_usd: 1.4321,
                step_events: 12,
                ..UsageTotals::default()
            },
            month: UsageTotals {
                total_cost_usd: 5.0789,
                step_events: 45,
                ..UsageTotals::default()
            },
            all_time: UsageTotals {
                input_tokens: 12_345,
                output_tokens: 6_789,
                cache_read_tokens: 1_111,
                cache_creation_tokens: 222,
                total_cost_usd: 12.3456,
                step_events: 100,
            },
        };
        let body = format_period_block(&spend);
        assert!(
            body.contains("this week:      $1.43  (12 step events)"),
            "{body}"
        );
        assert!(body.contains("today:          $0.12"), "{body}");
        assert!(body.contains("this month:     $5.08"), "{body}");
        assert!(
            body.contains("all-time:       $12.35  (100 step events)"),
            "{body}"
        );
        assert!(body.contains("input:          12,345"), "{body}");
        assert!(body.contains("output:         6,789"), "{body}");
        assert!(body.contains("cache_read:     1,111"), "{body}");
        assert!(body.contains("cache_creation: 222"), "{body}");
        // Two decimals everywhere -- guard against a regression that
        // smuggled the old `${:.4}` line back into the summary.
        assert!(!body.contains("$12.3456"), "{body}");
        assert!(!body.contains("$0.1234"), "{body}");
    }

    #[test]
    fn format_period_block_handles_all_zeros() {
        // Even when every period is empty (e.g. the `print_token_usage`
        // early-return branch isn't taken because some other window
        // had spend), the block must render without panicking and
        // still show the labelled $0.00 lines.
        let spend = PeriodSpend {
            day: UsageTotals::default(),
            week: UsageTotals::default(),
            month: UsageTotals::default(),
            all_time: UsageTotals::default(),
        };
        let body = format_period_block(&spend);
        assert!(
            body.contains("this week:      $0.00  (0 step events)"),
            "{body}"
        );
        assert!(body.contains("today:          $0.00"), "{body}");
        assert!(body.contains("this month:     $0.00"), "{body}");
        assert!(
            body.contains("all-time:       $0.00  (0 step events)"),
            "{body}"
        );
    }

    /// Empty-database integration check: with no `plan_events` rows
    /// at all, the storage helpers used by `print_token_usage` must
    /// each return `UsageTotals::default()` so the CLI takes the
    /// `none -- no metrics-bearing step events yet` branch instead
    /// of rendering a misleading $0.00 block. Exercises the same
    /// queries the status command will run against a fresh on-disk
    /// `SQLite` file (the `open` constructor the production path uses).
    #[tokio::test]
    async fn empty_database_yields_default_usage_totals_across_periods() {
        let dir = tempfile::tempdir().expect("tempdir");
        let db_path = dir.path().join("flowd.db");
        let store = SqlitePlanEventStore::open(&db_path).expect("open store");

        let now = ymd_hms(2026, 4, 15, 14, 30, 0);
        let p = compute_periods(now);

        let all = store.usage_totals().await.expect("usage_totals");
        let week = store
            .usage_totals_for_period(p.week.0, p.week.1)
            .await
            .expect("week");
        let day = store
            .usage_totals_for_period(p.day.0, p.day.1)
            .await
            .expect("day");
        let month = store
            .usage_totals_for_period(p.month.0, p.month.1)
            .await
            .expect("month");

        assert_eq!(all, UsageTotals::default());
        assert_eq!(week, UsageTotals::default());
        assert_eq!(day, UsageTotals::default());
        assert_eq!(month, UsageTotals::default());
    }
}
