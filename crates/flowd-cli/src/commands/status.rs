//! `flowd status` -- daemon liveness + DB stats.
//!
//! DB stats are computed with a single read on `SQLite`; we deliberately
//! use raw counts per tier rather than the `MemoryService` wrapper so no
//! embedder / vector index is required for an offline status view.

use anyhow::{Context, Result};
use chrono::Utc;

use flowd_core::memory::MemoryBackend;
use flowd_core::types::MemoryTier;
use flowd_storage::sqlite::SqliteBackend;

use crate::daemon::{is_alive, read_pid};
use crate::output::{Style, banner};
use crate::paths::FlowdPaths;

pub async fn run(paths: &FlowdPaths, style: Style) -> Result<()> {
    print!("{}", banner("flowd status", style));

    print_home(paths, style);
    print_daemon(paths, style)?;
    print_db(paths, style).await?;

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
