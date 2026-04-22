//! Command-line interface definitions (clap types only).
//!
//! Kept separate from [`crate::main`] so tests and tooling can reference
//! the argument surface without depending on the runtime.

use std::path::PathBuf;

use clap::{Parser, Subcommand};

/// flowd -- memory, orchestration, and rules engine for AI coding agents.
#[derive(Debug, Parser)]
#[command(name = "flowd", version, about)]
pub struct Cli {
    /// Override the flowd home directory. When absent, `$FLOWD_HOME` is
    /// consulted, falling back to `~/.flowd/`.
    #[arg(long, global = true)]
    pub home: Option<PathBuf>,

    /// Disable colourised output (equivalent to setting `NO_COLOR=1`).
    #[arg(long, global = true)]
    pub no_color: bool,

    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Start the flowd daemon (runs the MCP stdio server).
    Start {
        /// Qdrant URL (overrides the default <http://localhost:6334>).
        #[arg(long)]
        qdrant_url: Option<String>,

        /// Bounded mpsc capacity for the plan-event observer (HL-40).
        ///
        /// Plan-lifecycle events are buffered between the executor's
        /// hot path and the `SQLite` writer. When the buffer is full,
        /// new events are dropped (counted, not silently lost) so the
        /// executor never stalls on slow storage. Increase if your
        /// workload is bursty; decrease for tighter back-pressure.
        #[arg(long, default_value_t = 1024)]
        plan_event_buffer: usize,
    },

    /// Stop the running flowd daemon (sends SIGTERM).
    Stop,

    /// Search stored memory.
    Search {
        /// The search query.
        query: String,

        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,

        /// Max results to return.
        #[arg(short, long, default_value_t = 10)]
        limit: usize,
    },

    /// Browse session history.
    History {
        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,

        /// Show sessions since this RFC3339/ISO-8601 timestamp.
        #[arg(short, long)]
        since: Option<String>,

        /// Max sessions to return.
        #[arg(short, long, default_value_t = 50)]
        limit: usize,
    },

    /// Inspect orchestration plans.
    Plan {
        #[command(subcommand)]
        action: PlanAction,
    },

    /// Manage rules.
    Rules {
        #[command(subcommand)]
        action: RulesAction,
    },

    /// Record a single observation read from stdin.
    ///
    /// Designed for hook scripts (Claude Code session-start, post-tool-use,
    /// session-end). Writes directly to `SQLite` without invoking the vector
    /// index -- keyword search works immediately; vector search picks up
    /// the row after the daemon's next reindex pass.
    Observe {
        /// Project scope for the observation.
        #[arg(short, long)]
        project: String,

        /// Attach to an existing session UUID. Omit (or pass `-`) to start
        /// a new session for this observation.
        #[arg(short, long)]
        session: Option<String>,

        /// Optional JSON object to store under `metadata`.
        #[arg(short, long)]
        metadata: Option<String>,
    },

    /// Show daemon status and database stats.
    Status,

    /// Export memory as browsable markdown.
    Export {
        /// Output directory (created if missing).
        #[arg(short, long, default_value = "flowd-export")]
        output: PathBuf,

        /// Filter to a single project.
        #[arg(short, long)]
        project: Option<String>,
    },
}

#[derive(Debug, Subcommand)]
pub enum PlanAction {
    /// Load a plan file, render its preview, and optionally prompt
    /// the operator for Y/N confirmation.
    Preview {
        /// Path to the plan YAML or JSON file.
        file: PathBuf,

        /// Only show preview, skip the confirmation prompt.
        #[arg(long)]
        dry_run: bool,
    },

    /// Print the persisted lifecycle event log for a plan.
    ///
    /// Reads directly from `SQLite` (WAL-safe) so it works even while
    /// `flowd start` is running.
    Events {
        /// Plan UUID.
        plan_id: String,

        /// Maximum events to return (oldest first).
        #[arg(short, long, default_value_t = 100)]
        limit: usize,

        /// Restrict to one or more event kinds. Pass either repeatedly
        /// (`--kind step_failed --kind step_completed`) or as a single
        /// comma-separated value. Allowed: `submitted`, `started`,
        /// `step_completed`, `step_failed`, `step_refused`,
        /// `step_cancelled`, `finished`.
        #[arg(short, long, value_delimiter = ',')]
        kind: Vec<String>,
    },

    /// Submit answers to a Draft plan's open questions (offline mode).
    ///
    /// Refuses to run when the daemon is alive; in that case use the
    /// `plan_answer` MCP tool so you exercise the same wiring as
    /// every other client. Reads a JSON array of
    /// `[{"question_id": "...", "kind": "choose", "option_id": "..."}]`
    /// entries from `--file` (or stdin when `--file -`).
    Answer {
        /// Plan UUID.
        plan_id: String,

        /// JSON file containing the answers array. Pass `-` to read
        /// from stdin. Optional when `--defer-remaining` is set.
        #[arg(short, long)]
        file: Option<PathBuf>,

        /// Ask the compiler to invent best-effort answers for any
        /// still-open questions (mirrors `plan_answer`'s
        /// `defer_remaining` flag).
        #[arg(long)]
        defer_remaining: bool,
    },

    /// Submit a freeform refinement instruction to a Draft plan
    /// (offline mode). Refuses to run when the daemon is alive.
    Refine {
        /// Plan UUID.
        plan_id: String,

        /// Refinement instruction text. Mutually exclusive with
        /// `--file`.
        #[arg(short = 'm', long, conflicts_with = "file")]
        feedback: Option<String>,

        /// Read the refinement instruction from a file (or `-` for
        /// stdin). Mutually exclusive with `--feedback`.
        #[arg(short, long)]
        file: Option<PathBuf>,
    },

    /// Idempotently abandon a Draft / Confirmed plan (offline mode).
    /// Refuses to run when the daemon is alive.
    Cancel {
        /// Plan UUID.
        plan_id: String,
    },
}

#[derive(Debug, Subcommand)]
pub enum RulesAction {
    /// List active rules.
    List {
        /// Filter rules by project scope.
        #[arg(short, long)]
        project: Option<String>,

        /// Filter rules by file path scope.
        #[arg(short, long)]
        file: Option<String>,
    },
}
