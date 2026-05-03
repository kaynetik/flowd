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
    /// Start the central flowd daemon (runs the local socket server).
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

    /// Bridge MCP stdio to the central flowd daemon.
    Mcp,

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

    /// Install flowd integration snippets for external tools.
    Init {
        #[command(subcommand)]
        target: InitTarget,
    },

    /// Claude Code hook receiver. Reads the hook JSON payload on stdin
    /// and records a structured observation. Replaces the shell hops in
    /// `integrations/claude-code/hooks/*.sh` so Claude Code's
    /// `settings.json` no longer needs `/ABSOLUTE/PATH` placeholders or
    /// `bash`, `jq`, `uuidgen` on `$PATH`.
    ///
    /// All error paths inside the handler are swallowed -- see
    /// `.flowd/rules/hook-error-swallowing.yaml`.
    Hook {
        #[command(subcommand)]
        event: HookEvent,
    },
}

#[derive(Debug, Subcommand)]
pub enum HookEvent {
    /// Claude Code `SessionStart` hook. Records a session-open marker
    /// so the session appears in `flowd history` even if no tool fires.
    SessionStart,
    /// Claude Code `PostToolUse` hook. Records tool name, a truncated
    /// input/response summary as content, and the full input/response
    /// JSON under metadata.
    PostToolUse,
    /// Claude Code `SessionEnd` hook. Records a terminator row and
    /// drops the Claude->flowd session mapping so a restart starts
    /// a clean session.
    SessionEnd,
}

#[derive(Debug, Subcommand)]
pub enum InitTarget {
    /// Install the flowd MCP entry into Cursor's `mcp.json`.
    ///
    /// Writes (or deep-merges) the canonical server stanza that points at
    /// the currently-running `flowd` binary. Existing keys in the target
    /// file are preserved; only the `mcpServers.flowd` entry is replaced.
    Cursor {
        /// Write to `$HOME/.cursor/mcp.json` (user-global Cursor config).
        /// Mutually exclusive with `--project`.
        #[arg(long, conflicts_with = "project")]
        global: bool,

        /// Write to `<path>/.cursor/mcp.json` for a single project
        /// (the `.cursor/` directory is created if missing).
        #[arg(long, value_name = "PATH")]
        project: Option<PathBuf>,
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
        /// `step_started`, `step_completed`, `step_failed`,
        /// `step_refused`, `step_cancelled`, `finished`.
        #[arg(short, long, value_delimiter = ',')]
        kind: Vec<String>,
    },

    /// List persisted plan summaries, newest first.
    List {
        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,

        /// Filter by status.
        #[arg(short, long)]
        status: Option<String>,

        /// Max rows to return.
        #[arg(short, long)]
        limit: Option<usize>,
    },

    /// Show the full persisted snapshot for one plan.
    Show {
        /// Plan UUID.
        plan_id: String,
    },

    /// Show recent persisted plan summaries. Defaults to five rows.
    Recent {
        /// Filter by project.
        #[arg(short, long)]
        project: Option<String>,

        /// Filter by status.
        #[arg(short, long)]
        status: Option<String>,

        /// Max rows to return.
        #[arg(short, long, default_value_t = 5)]
        limit: usize,
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

    /// Concise audit rollup: total cost, raw cache read/create totals,
    /// cache reuse rate, input/output tokens, summed API + agent
    /// runtime, wall-clock span (when derivable), step count, and a
    /// per-model breakdown. Reads the persisted event log -- WAL-safe
    /// against a live daemon.
    Usage {
        /// Plan UUID.
        plan_id: String,

        /// Emit the same rollup as machine-readable JSON instead of the
        /// human-formatted block.
        #[arg(long)]
        json: bool,
    },

    /// Stage (and optionally promote) a `Completed` plan onto a base
    /// branch via the `plan_integrate` contract.
    ///
    /// Default mode stages an integration branch -- a fast-forward
    /// promotion to the configured base requires `--promote` on a
    /// follow-up call. `--dry-run` prints the planned cherry-picks
    /// without touching the repo. Never pushes to a remote; the
    /// operator propagates results themselves.
    ///
    /// Refuses while the daemon is alive: the CLI mutates the same
    /// `SQLite` plan / event tables the daemon owns. Stop the daemon
    /// (or use the equivalent MCP tool, when wired up) first.
    Integrate {
        /// Plan UUID.
        plan_id: String,

        /// Base branch to promote onto. Required: v1 has no inferred
        /// default; the operator must declare the target so a
        /// misconfigured workspace fails fast rather than quietly
        /// merging into the wrong branch.
        #[arg(short, long)]
        base: String,

        /// Stage / promote toggle. With `--promote`, fast-forwards the
        /// previously-staged integration branch onto `--base`. Mutually
        /// exclusive with `--dry-run` and `--discard`.
        #[arg(long, conflicts_with_all = ["dry_run", "discard"])]
        promote: bool,

        /// Compute and render the planned operations without touching
        /// the repo. Mutually exclusive with `--promote` and `--discard`.
        #[arg(long, conflicts_with_all = ["discard"])]
        dry_run: bool,

        /// Explicitly discard a previously-staged integration: removes
        /// the integration worktree+branch and (per `--cleanup`) drops
        /// the per-step branches and worktrees. The base ref is never
        /// touched. Mutually exclusive with `--promote` and `--dry-run`.
        #[arg(long)]
        discard: bool,

        /// What to do with the integration branch and per-step branches
        /// once the run finishes. Persisted with the integration metadata
        /// so the daemon honours it on the cleanup pass. Allowed values:
        /// `keep_on_failure` (default), `keep_always`, `drop_always`.
        #[arg(long, default_value = "keep_on_failure", value_name = "POLICY")]
        cleanup: String,

        /// Cherry-pick strategy. Reserved for future variants; v1 only
        /// supports the topological-tip cherry-pick. Anything else is
        /// rejected with a structured error so scripts can pin behaviour.
        #[arg(long, default_value = "tip-cherry-pick", value_name = "STRATEGY")]
        strategy: String,

        /// Optional verification command run inside the integration
        /// worktree before fast-forward promotion. A non-zero exit
        /// blocks the promote and leaves the configured base ref
        /// untouched. Whitespace-split into argv at parse time -- pass
        /// quoting-free invocations like `cargo nextest run -p flowd-cli`.
        /// Omit (or pass an empty string) to skip verification.
        #[arg(long, value_name = "CMD")]
        verify: Option<String>,

        /// Emit the run outcome (or refusal/failure) as JSON instead of
        /// the human-formatted block. Useful for scripted callers that
        /// need to switch on the typed cause.
        #[arg(long)]
        json: bool,
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
