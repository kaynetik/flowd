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

    /// Load and preview an orchestration plan.
    Plan {
        /// Path to the plan YAML or JSON file.
        file: PathBuf,

        /// Only show preview, skip the confirmation prompt.
        #[arg(long)]
        dry_run: bool,
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
