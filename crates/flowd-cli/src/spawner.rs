//! Concrete `AgentSpawner` implementations for the `flowd` daemon.
//!
//! `flowd-core` does not spawn OS processes itself; that is the CLI's job.
//! This module wires the orchestration engine to a local agent CLI binary
//! (Claude Code, `cursor-agent`, `OpenAI` Codex, Aider, ...).
//!
//! ## Selection order
//!
//! When a [`LocalShellSpawner`] is built via [`LocalShellSpawner::detect`]:
//!
//! 1. `$FLOWD_AGENT_BIN` if set (operator override; absolute path or a name
//!    on `$PATH`).
//! 2. The first entry from [`DEFAULT_CANDIDATES`] found on `$PATH`.
//! 3. Otherwise `None`, so the caller can fall back to [`UnconfiguredSpawner`]
//!    which fails loudly on `spawn` rather than silently.
//!
//! ## Invocation
//!
//! Each step is invoked as
//!
//! ```text
//! <bin> [default_args_for(bin)...] [extra_args...] <prompt_flag> "<step.prompt>"
//! ```
//!
//! `prompt_flag` defaults to `-p` (Claude Code, `cursor-agent`); override
//! it via `$FLOWD_AGENT_PROMPT_FLAG` for CLIs that use a different name
//! (e.g. `--message` for Aider). Extra leading arguments come from
//! `$FLOWD_AGENT_EXTRA_ARGS` (whitespace-split).
//!
//! ### Auto-injected defaults for Claude-like CLIs
//!
//! When the resolved binary's basename is `claude` or `cursor-agent`,
//! the spawner prepends two flags before any user-supplied
//! `extra_args`:
//!
//! - `--dangerously-skip-permissions` -- headless `-p` mode cannot
//!   prompt the operator for tool-use approval. Without this flag,
//!   every `Write` / `Edit` / `Bash` invocation by the spawned step
//!   is silently denied while the model still exits 0 with a
//!   chatty "I cannot do this" reply. The daemon would then record
//!   `step_completed` against an empty working tree -- the worst
//!   possible failure (green status, zero work).
//! - `--output-format=json` -- exposes a structured envelope with
//!   `is_error`, `permission_denials`, and a clean `result` field,
//!   so the spawner can detect post-skip refusals (the model still
//!   has policy gates beyond the workspace-trust prompt) and API
//!   errors that exit code 0 cannot signal.
//!
//! Operators who need the opposite (sandbox runs, audit replays)
//! must wrap the binary -- e.g. point `$FLOWD_AGENT_BIN` at a
//! shell script that strips the flags. There is no opt-out env
//! var, on purpose; the default has to make the common case
//! (headless dispatch) safe.
//!
//! Stdout is captured into [`AgentOutput::stdout`]. For Claude-like
//! bins the JSON envelope is parsed and the human-readable `result`
//! field is what downstream consumers see; the raw JSON is logged at
//! `tracing::trace!` only. The same envelope is also mined for token
//! counts and USD cost (top-level `usage` + `total_cost_usd`, plus
//! the per-model `modelUsage` block) and surfaced as
//! [`AgentMetrics`] on the returned [`AgentOutput`]. Metrics are
//! captured on both the success path *and* the failure paths
//! (`is_error: true`, lingering `permission_denials`); the failure
//! variants attach them to [`FlowdError::PlanExecution`] so an
//! expensive refusal still lands in the audit log with its true
//! spend instead of flying under the radar. Stderr is emitted via
//! `tracing` (warn on non-zero exit, debug otherwise) so it surfaces
//! in the daemon log without polluting the JSON-RPC channel.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::executor::{
    AgentMetrics, AgentSpawnContext, AgentSpawner, ModelUsage,
};
use flowd_core::orchestration::{AgentOutput, Plan, PlanPreview, PlanStep};
use flowd_storage::step_branch_store::{SqliteStepBranchStore, StepBranchRecord};
use serde::Deserialize;
use tokio::process::Command;

/// Default candidate binaries searched on `$PATH` when `$FLOWD_AGENT_BIN`
/// is unset. Order matters: first match wins.
const DEFAULT_CANDIDATES: &[&str] = &["claude", "cursor-agent"];
// Temporarily disable codex and aider until we have a way to test them.
// const DEFAULT_CANDIDATES: &[&str] = &["claude", "codex", "aider", "cursor-agent"];

/// Default flag passed before the prompt argument. Suitable for Claude Code
/// and `cursor-agent`; override via `$FLOWD_AGENT_PROMPT_FLAG` for others.
const DEFAULT_PROMPT_FLAG: &str = "-p";

/// Returns the basename of `bin` lowercased, with `.exe` stripped on
/// Windows-style names. Used to gate Claude-specific behavior on the
/// resolved binary regardless of its directory.
fn bin_basename(bin: &OsStr) -> Option<String> {
    let name = std::path::Path::new(bin).file_name()?.to_str()?;
    let trimmed = name.strip_suffix(".exe").unwrap_or(name);
    Some(trimmed.to_ascii_lowercase())
}

/// Whether the resolved spawner binary is a Claude Code-flavored CLI
/// (Anthropic's `claude` or `cursor-agent`). These share the
/// `-p` / `--dangerously-skip-permissions` / `--output-format=json`
/// surface; other CLIs (`codex`, `aider`) have their own conventions
/// and are left alone.
fn is_claude_like(bin: &OsStr) -> bool {
    matches!(
        bin_basename(bin).as_deref(),
        Some("claude" | "cursor-agent")
    )
}

/// Flags prepended automatically when the resolved binary is
/// Claude-like. See the module docstring for why each is mandatory
/// in headless dispatch.
fn default_args_for(bin: &OsStr) -> &'static [&'static str] {
    if is_claude_like(bin) {
        &["--dangerously-skip-permissions", "--output-format=json"]
    } else {
        &[]
    }
}

/// Subset of Claude Code's `--output-format=json` envelope that the
/// spawner cares about. Every field is `#[serde(default)]` so the
/// parser tolerates schema drift from upstream Claude releases --
/// missing fields degrade to "no signal", never "decode error".
#[derive(Debug, Default, Deserialize)]
struct ClaudeJsonEnvelope {
    /// True when the run terminated in an error state (API failure,
    /// model error, etc.). Independent of process exit code.
    #[serde(default)]
    is_error: bool,
    /// Human-readable model output. What downstream
    /// [`AgentOutput::stdout`] should carry.
    #[serde(default)]
    result: String,
    /// Tools the model wanted to use that were denied. Should be
    /// empty when `--dangerously-skip-permissions` is in effect; a
    /// non-empty value means a deeper policy gate (Anthropic-side
    /// safety filter, MCP server refusal, etc.) blocked the work.
    /// Treated as a hard failure so the audit log does not lie.
    #[serde(default)]
    permission_denials: Vec<ClaudePermissionDenial>,
    /// Aggregate token counts for the whole run (`snake_case` upstream).
    #[serde(default)]
    usage: ClaudeUsage,
    /// Aggregate USD cost for the run.
    #[serde(default)]
    total_cost_usd: f64,
    /// Wall-clock duration of the run in milliseconds.
    #[serde(default)]
    duration_ms: u64,
    /// Cumulative time spent in Anthropic API calls, in milliseconds.
    #[serde(default)]
    duration_api_ms: u64,
    /// Per-model breakdown, keyed by model name. Note the camelCase
    /// key (`modelUsage`) and camelCase inner fields -- the rest of
    /// the envelope is `snake_case`, so upstream is inconsistent and
    /// this one block needs explicit `rename`s.
    #[serde(default, rename = "modelUsage")]
    model_usage: BTreeMap<String, ClaudeModelUsage>,
}

#[derive(Debug, Deserialize)]
struct ClaudePermissionDenial {
    #[serde(default)]
    tool_name: String,
}

/// Top-level `usage` block. Upstream uses `snake_case` here, so no
/// per-field renames are needed.
#[derive(Debug, Default, Deserialize)]
#[allow(clippy::struct_field_names)]
struct ClaudeUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

/// Per-model entry under `modelUsage`. Upstream emits camelCase keys
/// here (unlike the top-level `usage` block), so every field needs
/// an explicit rename.
#[derive(Debug, Default, Deserialize)]
struct ClaudeModelUsage {
    #[serde(default, rename = "inputTokens")]
    input_tokens: u64,
    #[serde(default, rename = "outputTokens")]
    output_tokens: u64,
    #[serde(default, rename = "cacheCreationInputTokens")]
    cache_creation_input_tokens: u64,
    #[serde(default, rename = "cacheReadInputTokens")]
    cache_read_input_tokens: u64,
    #[serde(default, rename = "costUSD")]
    cost_usd: f64,
}

/// Single conversion site from the raw Claude envelope to the
/// transport-agnostic [`AgentMetrics`] type consumed by the rest of
/// the daemon. Keeping this as the only translation point means
/// schema-drift fixes land in one place.
impl From<&ClaudeJsonEnvelope> for AgentMetrics {
    fn from(e: &ClaudeJsonEnvelope) -> Self {
        let model_usage = e
            .model_usage
            .iter()
            .map(|(name, u)| {
                (
                    name.clone(),
                    ModelUsage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_creation_input_tokens: u.cache_creation_input_tokens,
                        cache_read_input_tokens: u.cache_read_input_tokens,
                        cost_usd: u.cost_usd,
                    },
                )
            })
            .collect();
        Self {
            input_tokens: e.usage.input_tokens,
            output_tokens: e.usage.output_tokens,
            cache_creation_input_tokens: e.usage.cache_creation_input_tokens,
            cache_read_input_tokens: e.usage.cache_read_input_tokens,
            total_cost_usd: e.total_cost_usd,
            duration_ms: e.duration_ms,
            duration_api_ms: e.duration_api_ms,
            model_usage,
        }
    }
}

/// `AgentSpawner` that shells out to a local CLI binary with the step's
/// prompt.
#[derive(Debug, Clone)]
pub struct LocalShellSpawner {
    bin: OsString,
    prompt_flag: String,
    cwd: Option<PathBuf>,
    extra_args: Vec<String>,
}

impl LocalShellSpawner {
    /// Build a spawner from environment variables, or return `None` if no
    /// usable agent CLI is discoverable.
    #[must_use]
    pub fn detect() -> Option<Self> {
        let bin = std::env::var_os("FLOWD_AGENT_BIN")
            .filter(|s| !s.is_empty())
            .or_else(|| {
                DEFAULT_CANDIDATES
                    .iter()
                    .find_map(|name| which_in_path(name).map(OsString::from))
            })?;

        let prompt_flag = std::env::var("FLOWD_AGENT_PROMPT_FLAG")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_PROMPT_FLAG.to_owned());

        let extra_args = std::env::var("FLOWD_AGENT_EXTRA_ARGS")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.split_whitespace().map(str::to_owned).collect())
            .unwrap_or_default();

        let cwd = std::env::var_os("FLOWD_AGENT_CWD")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        Some(Self {
            bin,
            prompt_flag,
            cwd,
            extra_args,
        })
    }

    /// Construct an explicit spawner. Used by tests; not gated on `cfg(test)`
    /// so embedders that depend on the crate can call it directly.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(bin: impl Into<OsString>) -> Self {
        Self {
            bin: bin.into(),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: None,
            extra_args: Vec::new(),
        }
    }

    /// Display the configured binary; for logging only.
    #[must_use]
    pub fn bin_display(&self) -> std::path::Display<'_> {
        std::path::Path::new(&self.bin).display()
    }

    fn effective_cwd(&self) -> Result<PathBuf> {
        match &self.cwd {
            Some(cwd) => Ok(cwd.clone()),
            None => std::env::current_dir().map_err(|e| FlowdError::PlanExecution {
                message: format!("resolve current directory for agent spawner: {e}"),
                metrics: None,
            }),
        }
    }

    async fn spawn_in_cwd(
        &self,
        step: &PlanStep,
        cwd_override: Option<&Path>,
    ) -> Result<AgentOutput> {
        let claude_like = is_claude_like(&self.bin);
        let injected = default_args_for(&self.bin);

        let mut cmd = Command::new(&self.bin);
        for arg in injected {
            cmd.arg(arg);
        }
        for arg in &self.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(&self.prompt_flag).arg(&step.prompt);
        if let Some(cwd) = cwd_override {
            cmd.current_dir(cwd);
        } else if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        // Detach stdin so an agent that tries to prompt the operator errors
        // immediately instead of blocking on a closed terminal.
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // kill_on_drop ensures the executor's `JoinHandle::abort` (used by
        // `cancel`) actually terminates the spawned process tree.
        cmd.kill_on_drop(true);

        tracing::info!(
            step_id = %step.id,
            agent_type = %step.agent_type,
            bin = %self.bin_display(),
            cwd = ?cwd_override.or(self.cwd.as_deref()),
            injected_args = ?injected,
            claude_like,
            "spawning agent step"
        );

        let output = cmd.output().await.map_err(|e| FlowdError::PlanExecution {
            message: format!(
                "failed to spawn agent `{bin}` for step `{step}`: {e}",
                bin = self.bin_display(),
                step = step.id,
            ),
            metrics: None,
        })?;

        let raw_stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code();

        if !output.status.success() {
            tracing::warn!(
                step_id = %step.id,
                exit_code = ?code,
                stderr = %truncate_for_log(&stderr, 4096),
                "agent step exited non-zero"
            );
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "step `{step}` failed (exit={code}): {stderr}",
                    step = step.id,
                    code = code.map_or_else(|| "signal".into(), |c| c.to_string()),
                    stderr = truncate_for_log(stderr.trim(), 1024),
                ),
                metrics: None,
            });
        }

        if !stderr.is_empty() {
            tracing::debug!(
                step_id = %step.id,
                stderr = %truncate_for_log(&stderr, 2048),
                "agent step stderr"
            );
        }

        let (stdout, metrics) = if claude_like {
            interpret_claude_stdout(&step.id, &raw_stdout)?
        } else {
            (raw_stdout, None)
        };

        Ok(AgentOutput {
            stdout,
            exit_code: code,
            metrics,
        })
    }
}

impl AgentSpawner for LocalShellSpawner {
    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        // The plan's persisted `project_root` wins over the spawner's
        // own `cwd` (which is a daemon-process-wide fallback set from
        // `$FLOWD_AGENT_CWD` or, ultimately, `current_dir()`). Without
        // this, every plan would silently anchor to wherever the
        // operator launched flowd from -- fine for one workspace,
        // wrong the moment a daemon services more than one plan.
        self.spawn_in_cwd(step, ctx.project_root.as_deref()).await
    }
}

#[derive(Debug, Clone)]
pub struct GitWorktreeSpawner {
    inner: LocalShellSpawner,
    manager: Arc<GitWorktreeManager>,
}

impl GitWorktreeSpawner {
    fn try_new(
        inner: LocalShellSpawner,
        flowd_home: &Path,
        branch_store: Option<Arc<SqliteStepBranchStore>>,
    ) -> Option<Self> {
        let cwd = inner.effective_cwd().ok()?;
        let repo = discover_git_root(&cwd).ok()?;
        Some(Self {
            inner,
            manager: Arc::new(GitWorktreeManager::new(
                repo,
                flowd_home.join("worktrees"),
                branch_store,
            )),
        })
    }

    fn repo_display(&self) -> std::path::Display<'_> {
        self.manager.repo.display()
    }
}

impl AgentSpawner for GitWorktreeSpawner {
    fn supports_worktree_isolation(&self) -> bool {
        true
    }

    async fn prepare_plan(&self, plan: &Plan, preview: &PlanPreview) -> Result<()> {
        if preview.max_parallelism <= 1 {
            return Ok(());
        }
        self.manager.prepare_plan(plan).await
    }

    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        if !ctx.plan_parallel {
            // Sequential plans skip the worktree machinery but must
            // still honour the plan's `project_root` over the wrapped
            // spawner's daemon-cwd fallback -- same precedence rule
            // `LocalShellSpawner::spawn` enforces directly.
            return self
                .inner
                .spawn_in_cwd(step, ctx.project_root.as_deref())
                .await;
        }
        let worktree = self.manager.prepare_step(&ctx, step).await?;
        let output = self.inner.spawn_in_cwd(step, Some(&worktree)).await?;
        self.manager.finish_step(&ctx, step, &worktree).await?;
        Ok(output)
    }
}

#[derive(Debug)]
struct GitWorktreeManager {
    repo: PathBuf,
    root: PathBuf,
    /// In-process cache of the durable `(plan_id, step_id) -> branch`
    /// mapping. Populated lazily from `branch_store` on first access for
    /// a given plan; written through on `finish_step`. The DB is the
    /// source of truth across daemon restarts -- this map exists so
    /// `dependency_branches` stays a sync hot path.
    branches: Mutex<HashMap<uuid::Uuid, HashMap<String, String>>>,
    /// Plan ids whose state has already been pulled out of
    /// `branch_store`. Distinct from `branches.contains_key(plan_id)`
    /// because an in-process insert from `finish_step` would
    /// otherwise look indistinguishable from "rehydrated, nothing
    /// recorded yet" and skip a needed reload.
    loaded_plans: Mutex<HashSet<uuid::Uuid>>,
    /// Durable backing store. `None` only in unit-test setups that
    /// don't need to assert restart behaviour; the daemon always
    /// supplies one.
    branch_store: Option<Arc<SqliteStepBranchStore>>,
}

impl GitWorktreeManager {
    fn new(
        repo: PathBuf,
        root: PathBuf,
        branch_store: Option<Arc<SqliteStepBranchStore>>,
    ) -> Self {
        Self {
            repo,
            root,
            branches: Mutex::new(HashMap::new()),
            loaded_plans: Mutex::new(HashSet::new()),
            branch_store,
        }
    }

    async fn prepare_plan(&self, plan: &Plan) -> Result<()> {
        // Plans persisted with `project_root` win over the manager's
        // construction-time `self.repo` (discovered once from the
        // daemon cwd by `try_new`). Without this, a parallel plan
        // submitted from a workspace different from the daemon's
        // launch dir would dirty-check and `git worktree add` against
        // the wrong repo. Legacy plans (`project_root = None`) fall
        // back to `self.repo` -- same behaviour as before this wiring.
        let repo = self.repo_for(plan.project_root.as_deref().map(Path::new));
        let dirty = git_output(repo, ["status", "--porcelain"]).await?;
        if !dirty.trim().is_empty() {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "parallel plan `{}` requires a clean git base at {}; refusing to use stashes. Commit or discard these changes first:\n{}",
                    plan.id,
                    repo.display(),
                    dirty.trim()
                ),
                metrics: None,
            });
        }
        tokio::fs::create_dir_all(self.plan_root(plan))
            .await
            .map_err(|e| FlowdError::PlanExecution {
                message: format!("create worktree root for plan `{}`: {e}", plan.id),
                metrics: None,
            })?;
        Ok(())
    }

    async fn prepare_step(&self, ctx: &AgentSpawnContext, step: &PlanStep) -> Result<PathBuf> {
        let branch = step_branch(ctx, step);
        let path = self
            .root
            .join(sanitize_path(&ctx.project))
            .join(ctx.plan_id.to_string())
            .join(sanitize_path(&step.id));
        if path.exists() {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "worktree path already exists for step `{}`: {}; remove it before resuming this parallel plan",
                    step.id,
                    path.display()
                ),
                metrics: None,
            });
        }

        // Pull the plan's durable step→branch state into the in-memory
        // cache before the sync `dependency_branches` consults it.
        // After a daemon restart this is the only thing that lets a
        // dependent step find its parent branch -- the in-process map
        // is empty until this call seeds it.
        self.ensure_branches_loaded(ctx.plan_id).await?;
        let dep_branches = self.dependency_branches(ctx.plan_id, step)?;
        let base = dep_branches.first().map_or("HEAD", String::as_str);
        let path_arg = path.to_string_lossy().into_owned();
        // Same precedence rule as `prepare_plan`: use `ctx.project_root`
        // when the plan persisted one, fall back to `self.repo` for
        // legacy plans. The dependent `git merge` calls below run
        // inside the freshly-added worktree, not the base repo, so they
        // already operate on the right tree without needing the override.
        let repo = self.repo_for(ctx.project_root.as_deref());
        git_output(repo, ["worktree", "add", "-b", &branch, &path_arg, base]).await?;
        for dep in dep_branches.iter().skip(1) {
            git_output(&path, ["merge", "--no-edit", dep]).await?;
        }
        Ok(path)
    }

    /// Resolve the git repo path to drive `git worktree add` / `git status`
    /// from. The plan's persisted `project_root` (when present) overrides
    /// `self.repo`, which is only meaningful as a fallback for plans that
    /// pre-date the field. Returning `&Path` (not `PathBuf`) keeps the
    /// borrow tied to the input lifetimes -- callers pass either an owned
    /// or borrowed override and never need to clone.
    fn repo_for<'a>(&'a self, override_root: Option<&'a Path>) -> &'a Path {
        override_root.unwrap_or(self.repo.as_path())
    }

    async fn finish_step(
        &self,
        ctx: &AgentSpawnContext,
        step: &PlanStep,
        worktree: &Path,
    ) -> Result<()> {
        git_output(worktree, ["add", "-A"]).await?;
        let has_changes = git_status(worktree, ["diff", "--cached", "--quiet"]).await?;
        if has_changes != 0 {
            git_output(
                worktree,
                [
                    "-c",
                    "user.name=flowd",
                    "-c",
                    "user.email=flowd@local",
                    "commit",
                    "-m",
                    &format!("flowd step {}", step.id),
                ],
            )
            .await?;
        }
        let branch = step_branch(ctx, step);
        // Capture HEAD after the (maybe-)commit so the durable record
        // points at the exact tree future integration must
        // fast-forward to. A no-change step still records the inherited
        // tip so dependents can ff-merge to it without re-running.
        let tip_sha = git_output(worktree, ["rev-parse", "HEAD"])
            .await?
            .trim()
            .to_owned();

        {
            let mut guard = self
                .branches
                .lock()
                .map_err(|_| FlowdError::PlanExecution {
                    message: "worktree branch map poisoned".into(),
                    metrics: None,
                })?;
            guard
                .entry(ctx.plan_id)
                .or_default()
                .insert(step.id.clone(), branch.clone());
        }

        // Write through to the durable store after the in-process
        // cache update so the cache stays usable even if the DB write
        // fails (the executor will surface the error and the next
        // restart re-derives from the remaining rows).
        if let Some(store) = &self.branch_store {
            store
                .upsert(&StepBranchRecord {
                    plan_id: ctx.plan_id,
                    step_id: step.id.clone(),
                    branch,
                    tip_sha,
                    worktree_path: Some(worktree.to_string_lossy().into_owned()),
                })
                .await?;
        }
        Ok(())
    }

    /// Pull every recorded `(step_id, branch)` for `plan_id` out of the
    /// durable store into the in-memory cache. No-op once a plan has
    /// been loaded -- subsequent calls within the same process trust
    /// the cache, since `finish_step` writes through to both sides.
    ///
    /// Existing in-memory entries win over DB entries on a key clash,
    /// because an in-flight `finish_step` reflects the latest commit
    /// while the DB row could be one retry behind.
    async fn ensure_branches_loaded(&self, plan_id: uuid::Uuid) -> Result<()> {
        {
            let loaded = self
                .loaded_plans
                .lock()
                .map_err(|_| FlowdError::PlanExecution {
                    message: "worktree loaded-plans set poisoned".into(),
                    metrics: None,
                })?;
            if loaded.contains(&plan_id) {
                return Ok(());
            }
        }

        let records = if let Some(store) = &self.branch_store {
            store.list_for_plan(plan_id).await?
        } else {
            Vec::new()
        };

        let mut branches = self
            .branches
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree branch map poisoned".into(),
                metrics: None,
            })?;
        let entry = branches.entry(plan_id).or_default();
        for record in records {
            // `or_insert`: do not clobber an in-memory write recorded
            // by a concurrent `finish_step`.
            entry.entry(record.step_id).or_insert(record.branch);
        }
        drop(branches);

        let mut loaded = self
            .loaded_plans
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree loaded-plans set poisoned".into(),
                metrics: None,
            })?;
        loaded.insert(plan_id);
        Ok(())
    }

    fn dependency_branches(&self, plan_id: uuid::Uuid, step: &PlanStep) -> Result<Vec<String>> {
        let guard = self
            .branches
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree branch map poisoned".into(),
                metrics: None,
            })?;
        let Some(plan_branches) = guard.get(&plan_id) else {
            return Ok(Vec::new());
        };
        step.depends_on
            .iter()
            .map(|dep| {
                plan_branches
                    .get(dep)
                    .cloned()
                    .ok_or_else(|| FlowdError::PlanExecution {
                        message: format!(
                            "dependency branch for step `{dep}` is missing before running `{}`",
                            step.id
                        ),
                        metrics: None,
                    })
            })
            .collect()
    }

    fn plan_root(&self, plan: &Plan) -> PathBuf {
        self.root
            .join(sanitize_path(&plan.project))
            .join(plan.id.to_string())
    }
}

fn discover_git_root(cwd: &Path) -> Result<PathBuf> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .arg("rev-parse")
        .arg("--show-toplevel")
        .output()
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("probe git repository at {}: {e}", cwd.display()),
            metrics: None,
        })?;
    if !output.status.success() {
        return Err(FlowdError::PlanExecution {
            message: format!("agent cwd {} is not inside a git repository", cwd.display()),
            metrics: None,
        });
    }
    let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    Ok(PathBuf::from(root))
}

async fn git_output<const N: usize>(cwd: &Path, args: [&str; N]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .await
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("run git in {}: {e}", cwd.display()),
            metrics: None,
        })?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    Err(FlowdError::PlanExecution {
        message: format!(
            "git command failed in {}: {}",
            cwd.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        metrics: None,
    })
}

async fn git_status<const N: usize>(cwd: &Path, args: [&str; N]) -> Result<i32> {
    let status = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .status()
        .await
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("run git in {}: {e}", cwd.display()),
            metrics: None,
        })?;
    Ok(status.code().unwrap_or(1))
}

fn step_branch(ctx: &AgentSpawnContext, step: &PlanStep) -> String {
    format!(
        "flowd/{}/{}/{}",
        sanitize_ref(&ctx.project),
        ctx.plan_id.simple(),
        sanitize_ref(&step.id),
    )
}

fn sanitize_path(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn sanitize_ref(raw: &str) -> String {
    let cleaned = sanitize_path(raw).trim_matches('-').to_owned();
    if cleaned.is_empty() {
        "unnamed".into()
    } else {
        cleaned
    }
}

/// Parse a Claude `--output-format=json` envelope, fail loudly on
/// signals that exit-code zero would otherwise hide, and return the
/// human-readable `result` field alongside the metrics block the
/// envelope carries. Falls back to the raw payload (and no metrics)
/// if the bytes do not parse as JSON -- common when an operator
/// overrides `$FLOWD_AGENT_EXTRA_ARGS` to force text output.
///
/// Failure paths (`is_error: true`, non-empty `permission_denials`)
/// still attach the envelope's metrics to the returned `Err`: failed
/// steps still cost money, and the audit log must reflect the spend.
/// Returning `Err` here marks the step `failed` in the orchestration log.
fn interpret_claude_stdout(step_id: &str, raw: &str) -> Result<(String, Option<AgentMetrics>)> {
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        return Err(FlowdError::PlanExecution {
            message: format!(
                "step `{step_id}` produced empty stdout (claude exited 0 with no JSON envelope; \
                 likely a tool-use timeout or a config that suppressed --output-format=json)"
            ),
            metrics: None,
        });
    }

    let envelope: ClaudeJsonEnvelope = match serde_json::from_str(trimmed) {
        Ok(v) => v,
        Err(e) => {
            tracing::debug!(
                step_id,
                error = %e,
                "claude-like spawner: stdout was not parseable JSON; passing through raw"
            );
            return Ok((raw.to_owned(), None));
        }
    };

    let metrics: AgentMetrics = (&envelope).into();

    if envelope.is_error {
        return Err(FlowdError::PlanExecution {
            message: format!(
                "step `{step_id}` reported is_error=true: {result}",
                result = truncate_for_log(envelope.result.trim(), 1024),
            ),
            metrics: Some(metrics),
        });
    }

    if !envelope.permission_denials.is_empty() {
        let denied: Vec<&str> = envelope
            .permission_denials
            .iter()
            .map(|d| d.tool_name.as_str())
            .collect();
        return Err(FlowdError::PlanExecution {
            message: format!(
                "step `{step_id}` had {n} tool invocation(s) denied despite \
                 --dangerously-skip-permissions ({denied:?}); the daemon is refusing \
                 to record this as success. result: {result}",
                n = denied.len(),
                result = truncate_for_log(envelope.result.trim(), 1024),
            ),
            metrics: Some(metrics),
        });
    }

    Ok((envelope.result, Some(metrics)))
}

/// Spawner used as the fallback when no agent binary is discoverable. Fails
/// every `spawn` with an actionable error so misconfiguration surfaces the
/// first time a plan is confirmed, instead of silently succeeding.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnconfiguredSpawner;

impl AgentSpawner for UnconfiguredSpawner {
    async fn spawn(&self, _ctx: AgentSpawnContext, _step: &PlanStep) -> Result<AgentOutput> {
        Err(FlowdError::PlanExecution {
            message: "no agent CLI configured. Set $FLOWD_AGENT_BIN to an absolute \
                      path, or install one of: cursor-agent, claude, codex, aider \
                      into $PATH, then restart the daemon."
                .into(),
            metrics: None,
        })
    }
}

/// Type-erased spawner so the executor can be parametrised at runtime
/// without leaking concrete types into the rest of `start.rs`.
pub enum BoxedSpawner {
    Local(LocalShellSpawner),
    Worktree(GitWorktreeSpawner),
    Unconfigured(UnconfiguredSpawner),
}

impl BoxedSpawner {
    /// Build the best available spawner for the current environment.
    ///
    /// `branch_store`, when supplied, makes finished step → branch
    /// state survive a daemon restart. Pass `None` only when the
    /// caller explicitly does not want durable worktree state (the
    /// integration tests do this; the daemon never does).
    pub fn auto(
        flowd_home: &Path,
        branch_store: Option<Arc<SqliteStepBranchStore>>,
    ) -> Self {
        let Some(local) = LocalShellSpawner::detect() else {
            return Self::Unconfigured(UnconfiguredSpawner);
        };
        GitWorktreeSpawner::try_new(local.clone(), flowd_home, branch_store)
            .map_or(Self::Local(local), Self::Worktree)
    }

    pub fn description(&self) -> String {
        match self {
            Self::Local(s) => format!("local-shell ({})", s.bin_display()),
            Self::Worktree(s) => format!(
                "git-worktree ({}, repo {})",
                s.inner.bin_display(),
                s.repo_display()
            ),
            Self::Unconfigured(_) => "unconfigured (plan execution will fail)".into(),
        }
    }
}

impl AgentSpawner for BoxedSpawner {
    fn supports_worktree_isolation(&self) -> bool {
        matches!(self, Self::Worktree(_))
    }

    async fn prepare_plan(&self, plan: &Plan, preview: &PlanPreview) -> Result<()> {
        match self {
            Self::Worktree(s) => s.prepare_plan(plan, preview).await,
            Self::Local(_) | Self::Unconfigured(_) => Ok(()),
        }
    }

    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        match self {
            Self::Local(s) => s.spawn(ctx, step).await,
            Self::Worktree(s) => s.spawn(ctx, step).await,
            Self::Unconfigured(s) => s.spawn(ctx, step).await,
        }
    }
}

/// First executable match for `name` on `$PATH`. Returns `None` if the name
/// is not found or `$PATH` is unset.
fn which_in_path(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| is_executable(p))
        .and_then(|p| p.into_os_string().into_string().ok())
}

#[cfg(unix)]
fn is_executable(p: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && (m.permissions().mode() & 0o111 != 0))
}

#[cfg(not(unix))]
fn is_executable(p: &std::path::Path) -> bool {
    // Best-effort on non-unix: any regular file in PATH is treated as
    // executable. The supported agent CLIs are unix-only in practice.
    p.is_file()
}

/// Truncate `s` to at most `max` bytes on a UTF-8 boundary, appending a
/// marker noting how many bytes were elided.
fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let cut = s
        .char_indices()
        .take_while(|(i, _)| *i <= max)
        .last()
        .map_or(0, |(i, _)| i);
    format!("{}…[+{} bytes]", &s[..cut], s.len() - cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::StepStatus;

    fn step(prompt: &str) -> PlanStep {
        step_with_id("t", prompt, &[])
    }

    fn step_with_id(id: &str, prompt: &str, depends_on: &[&str]) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "test".into(),
            prompt: prompt.into(),
            depends_on: depends_on.iter().map(|s| (*s).to_owned()).collect(),
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn ctx() -> AgentSpawnContext {
        AgentSpawnContext {
            plan_id: uuid::Uuid::new_v4(),
            project: "test".into(),
            plan_parallel: false,
            layer_width: 1,
            project_root: None,
        }
    }

    fn git_sync(repo: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .status()
            .expect("spawn git");
        assert!(
            status.success(),
            "git {args:?} failed in {}",
            repo.display()
        );
    }

    /// Smoke test: `echo` stands in for an agent that just prints its
    /// argv. With the default `-p` flag the invocation is
    /// `echo -p "<prompt>"`, which exits 0 and contains the prompt.
    #[tokio::test]
    async fn echo_spawner_returns_stdout() {
        let s = LocalShellSpawner::new("echo");
        let out = s.spawn(ctx(), &step("hello flowd")).await.unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout.contains("hello flowd"));
    }

    #[tokio::test]
    async fn missing_binary_yields_plan_execution_error() {
        let s = LocalShellSpawner::new("/does/not/exist/flowd-agent");
        let err = s.spawn(ctx(), &step("anything")).await.unwrap_err();
        assert!(matches!(err, FlowdError::PlanExecution { .. }));
    }

    /// `LocalShellSpawner::spawn` must take its cwd from
    /// `ctx.project_root` (the plan's persisted workspace) and override
    /// the daemon-set `LocalShellSpawner.cwd` field. Without this,
    /// every step silently inherits the directory the operator launched
    /// flowd from -- fine for one workspace, wrong the moment a daemon
    /// services more than one plan. We assert against real subprocess
    /// state (`pwd`) so a future refactor that "passes the value
    /// through" but never reaches `Command::current_dir` would still
    /// fail this test.
    #[cfg(unix)]
    #[tokio::test]
    async fn local_shell_spawn_uses_ctx_project_root_over_daemon_cwd() {
        let daemon_cwd = tempfile::tempdir().expect("daemon cwd");
        let plan_root = tempfile::tempdir().expect("plan root");

        let s = LocalShellSpawner {
            bin: OsString::from("/bin/sh"),
            prompt_flag: "-c".to_owned(),
            cwd: Some(daemon_cwd.path().to_path_buf()),
            extra_args: Vec::new(),
        };

        let mut ctx = ctx();
        ctx.project_root = Some(plan_root.path().to_path_buf());

        let out = s
            .spawn(ctx, &step("pwd"))
            .await
            .expect("pwd subprocess must succeed");
        assert_eq!(out.exit_code, Some(0));

        // Canonicalise both sides: macOS resolves `/var/...` -> `/private/var/...`
        // when a process chdirs through the symlink, and `tempdir()` may
        // return either form. Compare canonical paths to dodge that.
        let expected = std::fs::canonicalize(plan_root.path()).expect("canonicalize plan root");
        let observed_raw = std::path::PathBuf::from(out.stdout.trim());
        let observed = std::fs::canonicalize(&observed_raw).expect("canonicalize pwd stdout");
        assert_eq!(
            observed,
            expected,
            "subprocess cwd resolved to {observed:?}; expected plan_root {expected:?}. \
             Daemon cwd was {daemon:?} -- ctx.project_root must win.",
            daemon = daemon_cwd.path()
        );
    }

    /// Sequential branch of `GitWorktreeSpawner::spawn` skips the
    /// worktree machinery and delegates to the wrapped
    /// `LocalShellSpawner`. Pin that the delegation still threads
    /// `ctx.project_root` through -- the historical bug had this
    /// branch hardcode `None`, silently anchoring sequential plans to
    /// the daemon cwd even after the executor learned to carry the
    /// plan root.
    #[cfg(unix)]
    #[tokio::test]
    async fn worktree_spawner_sequential_uses_ctx_project_root_over_daemon_cwd() {
        let daemon_cwd = tempfile::tempdir().expect("daemon cwd");
        let plan_root = tempfile::tempdir().expect("plan root");
        let flowd_home = tempfile::tempdir().expect("flowd home");

        let inner = LocalShellSpawner {
            bin: OsString::from("/bin/sh"),
            prompt_flag: "-c".to_owned(),
            cwd: Some(daemon_cwd.path().to_path_buf()),
            extra_args: Vec::new(),
        };
        let s = GitWorktreeSpawner {
            inner,
            // Manager `repo` is irrelevant on the sequential branch
            // (no `git worktree add` is called); pick any path so
            // construction succeeds.
            manager: Arc::new(GitWorktreeManager::new(
                daemon_cwd.path().to_path_buf(),
                flowd_home.path().join("worktrees"),
                None,
            )),
        };

        let mut ctx = ctx();
        ctx.plan_parallel = false;
        ctx.project_root = Some(plan_root.path().to_path_buf());

        let out = s
            .spawn(ctx, &step("pwd"))
            .await
            .expect("pwd subprocess must succeed");
        assert_eq!(out.exit_code, Some(0));

        let expected = std::fs::canonicalize(plan_root.path()).unwrap();
        let observed = std::fs::canonicalize(std::path::PathBuf::from(out.stdout.trim())).unwrap();
        assert_eq!(
            observed,
            expected,
            "GitWorktreeSpawner sequential delegation lost ctx.project_root: \
             observed {observed:?}, expected {expected:?}, daemon was {:?}",
            daemon_cwd.path()
        );
    }

    #[tokio::test]
    async fn unconfigured_spawner_reports_actionable_error() {
        let s = UnconfiguredSpawner;
        let err = s.spawn(ctx(), &step("x")).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("FLOWD_AGENT_BIN"));
        assert!(msg.contains("cursor-agent"));
    }

    #[test]
    fn truncate_for_log_passes_through_short_strings() {
        assert_eq!(truncate_for_log("abc", 10), "abc");
    }

    /// `αβγδε` is 10 bytes (5 chars × 2). With `max = 5` we can fit
    /// `αβ` (4 bytes) but not `αβγ` (6 bytes); the cut must land on a
    /// UTF-8 boundary and the elided-byte marker must follow.
    #[test]
    fn truncate_for_log_clips_on_utf8_boundary() {
        let out = truncate_for_log("αβγδε", 5);
        assert_eq!(out, "αβ…[+6 bytes]");
    }

    #[test]
    fn is_claude_like_recognises_canonical_names() {
        assert!(is_claude_like(OsStr::new("claude")));
        assert!(is_claude_like(OsStr::new("/usr/local/bin/claude")));
        assert!(is_claude_like(OsStr::new("cursor-agent")));
        assert!(is_claude_like(OsStr::new("/opt/cursor-agent")));
        // case-insensitive on the basename
        assert!(is_claude_like(OsStr::new("/x/Claude")));
        assert!(is_claude_like(OsStr::new("CLAUDE.exe")));
    }

    #[test]
    fn is_claude_like_rejects_unrelated_bins() {
        assert!(!is_claude_like(OsStr::new("echo")));
        assert!(!is_claude_like(OsStr::new("codex")));
        assert!(!is_claude_like(OsStr::new("aider")));
        assert!(!is_claude_like(OsStr::new("/usr/bin/claude-like-but-not")));
    }

    #[test]
    fn default_args_for_only_injects_for_claude_family() {
        assert_eq!(
            default_args_for(OsStr::new("claude")),
            &["--dangerously-skip-permissions", "--output-format=json"]
        );
        assert_eq!(
            default_args_for(OsStr::new("cursor-agent")),
            &["--dangerously-skip-permissions", "--output-format=json"]
        );
        assert!(default_args_for(OsStr::new("echo")).is_empty());
        assert!(default_args_for(OsStr::new("aider")).is_empty());
    }

    #[test]
    fn interpret_claude_stdout_extracts_result_on_success() {
        let raw = r#"{"is_error":false,"result":"OK","permission_denials":[]}"#;
        let (out, _) = interpret_claude_stdout("step-1", raw).unwrap();
        assert_eq!(out, "OK");
    }

    /// Empty stdout from a Claude-like bin is suspicious enough to
    /// fail the step rather than silently record success against a
    /// blank tree.
    #[test]
    fn interpret_claude_stdout_fails_on_empty_payload() {
        let err = interpret_claude_stdout("step-1", "").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("empty stdout"), "got: {msg}");
    }

    /// Permission denials persisting past `--dangerously-skip-permissions`
    /// are the canary the spawner is built to catch. The model returns
    /// exit 0 with `is_error: false` but `permission_denials` is non-empty;
    /// this used to land as `step_completed` against an empty working tree.
    #[test]
    fn interpret_claude_stdout_fails_on_permission_denials() {
        let raw = r#"{
            "is_error": false,
            "result": "File creation was denied. Let me know if you'd like to approve.",
            "permission_denials": [{"tool_name": "Write"}]
        }"#;
        let err = interpret_claude_stdout("step-1", raw).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("denied despite --dangerously-skip-permissions"),
            "got: {msg}"
        );
        assert!(msg.contains("Write"), "got: {msg}");
    }

    #[test]
    fn interpret_claude_stdout_fails_on_is_error_true() {
        let raw = r#"{"is_error":true,"result":"upstream API error"}"#;
        let err = interpret_claude_stdout("step-1", raw).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is_error=true"), "got: {msg}");
        assert!(msg.contains("upstream API error"), "got: {msg}");
    }

    /// Operators may override `--output-format` via `FLOWD_AGENT_EXTRA_ARGS`,
    /// which leaves stdout as plain text. Pass it through rather than
    /// erroring -- the override is intentional.
    #[test]
    fn interpret_claude_stdout_passes_through_non_json_payload() {
        let raw = "just a plain text reply\n";
        let (out, metrics) = interpret_claude_stdout("step-1", raw).unwrap();
        assert_eq!(out, raw);
        assert!(
            metrics.is_none(),
            "plain-text passthrough yields no metrics"
        );
    }

    /// Schema-drift defence: tolerate unknown keys without choking,
    /// and treat absent keys as their defaults (`is_error=false`,
    /// no denials, empty result).
    #[test]
    fn interpret_claude_stdout_tolerates_unknown_fields() {
        let raw = r#"{"is_error":false,"result":"x","future_field":42,"another":"y"}"#;
        let (out, _) = interpret_claude_stdout("step-1", raw).unwrap();
        assert_eq!(out, "x");
    }

    /// Happy-path metric extraction: a full envelope populates every
    /// numeric field the daemon bills against.
    #[test]
    fn interpret_claude_stdout_extracts_metrics_on_success() {
        let raw = r#"{
            "is_error": false,
            "result": "done",
            "permission_denials": [],
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20
            },
            "total_cost_usd": 0.0123,
            "duration_ms": 1500,
            "duration_api_ms": 1200,
            "modelUsage": {
                "claude-sonnet-4-5": {
                    "inputTokens": 100,
                    "outputTokens": 50,
                    "cacheCreationInputTokens": 10,
                    "cacheReadInputTokens": 20,
                    "costUSD": 0.0123
                }
            }
        }"#;
        let (stdout, metrics) = interpret_claude_stdout("step-1", raw).unwrap();
        assert_eq!(stdout, "done");
        let m = metrics.expect("metrics must be present on success");
        assert_eq!(m.input_tokens, 100);
        assert_eq!(m.output_tokens, 50);
        assert_eq!(m.cache_creation_input_tokens, 10);
        assert_eq!(m.cache_read_input_tokens, 20);
        assert!((m.total_cost_usd - 0.0123).abs() < f64::EPSILON);
        assert_eq!(m.duration_ms, 1500);
        assert_eq!(m.duration_api_ms, 1200);
        let model = m
            .model_usage
            .get("claude-sonnet-4-5")
            .expect("per-model entry must be decoded");
        assert_eq!(model.input_tokens, 100);
        assert_eq!(model.output_tokens, 50);
        assert_eq!(model.cache_creation_input_tokens, 10);
        assert_eq!(model.cache_read_input_tokens, 20);
        assert!((model.cost_usd - 0.0123).abs() < f64::EPSILON);
    }

    /// Failed steps still cost money: the `is_error: true` path must
    /// surface the envelope's metrics through the `Err` so accounting
    /// stays honest.
    #[test]
    fn interpret_claude_stdout_attaches_metrics_to_is_error_failure() {
        let raw = r#"{
            "is_error": true,
            "result": "upstream API error",
            "usage": {"input_tokens": 7, "output_tokens": 3},
            "total_cost_usd": 0.004,
            "duration_ms": 250,
            "duration_api_ms": 240
        }"#;
        let err = interpret_claude_stdout("step-1", raw).unwrap_err();
        match err {
            FlowdError::PlanExecution { metrics, .. } => {
                let m = metrics.expect("is_error failure must carry metrics");
                assert_eq!(m.input_tokens, 7);
                assert_eq!(m.output_tokens, 3);
                assert!((m.total_cost_usd - 0.004).abs() < f64::EPSILON);
                assert_eq!(m.duration_ms, 250);
                assert_eq!(m.duration_api_ms, 240);
            }
            other => panic!("expected PlanExecution, got {other:?}"),
        }
    }

    /// Same contract as `is_error`: a denied-tool failure must still
    /// report the cost incurred getting to the refusal.
    #[test]
    fn interpret_claude_stdout_attaches_metrics_to_permission_denial_failure() {
        let raw = r#"{
            "is_error": false,
            "result": "refused",
            "permission_denials": [{"tool_name": "Write"}],
            "usage": {"input_tokens": 12, "output_tokens": 4},
            "total_cost_usd": 0.002
        }"#;
        let err = interpret_claude_stdout("step-1", raw).unwrap_err();
        match err {
            FlowdError::PlanExecution { metrics, .. } => {
                let m = metrics.expect("permission_denials failure must carry metrics");
                assert_eq!(m.input_tokens, 12);
                assert_eq!(m.output_tokens, 4);
                assert!((m.total_cost_usd - 0.002).abs() < f64::EPSILON);
            }
            other => panic!("expected PlanExecution, got {other:?}"),
        }
    }

    /// Schema-drift defence: an envelope without `usage` (or any other
    /// new field) must decode into zero-valued metrics rather than
    /// erroring. Upstream can drop or rename a field between releases
    /// and the spawner keeps running.
    #[test]
    fn interpret_claude_stdout_handles_envelope_without_usage_block() {
        let raw = r#"{"is_error":false,"result":"minimal"}"#;
        let (stdout, metrics) = interpret_claude_stdout("step-1", raw).unwrap();
        assert_eq!(stdout, "minimal");
        let m = metrics.expect("metrics envelope is constructed unconditionally on parse");
        assert_eq!(m.input_tokens, 0);
        assert_eq!(m.output_tokens, 0);
        assert_eq!(m.cache_creation_input_tokens, 0);
        assert_eq!(m.cache_read_input_tokens, 0);
        assert!(m.total_cost_usd.abs() < f64::EPSILON);
        assert_eq!(m.duration_ms, 0);
        assert_eq!(m.duration_api_ms, 0);
        assert!(m.model_usage.is_empty());
    }

    #[tokio::test]
    async fn worktree_manager_rejects_dirty_base_and_records_step_branch() {
        let repo = tempfile::tempdir().expect("repo");
        git_sync(repo.path(), &["init"]);
        std::fs::write(repo.path().join("README.md"), "base\n").unwrap();
        git_sync(repo.path(), &["add", "README.md"]);
        git_sync(
            repo.path(),
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "commit",
                "-m",
                "base",
            ],
        );

        let worktrees = tempfile::tempdir().expect("worktrees");
        let manager = GitWorktreeManager::new(
            repo.path().to_path_buf(),
            worktrees.path().join("worktrees"),
            None,
        );
        let plan = flowd_core::orchestration::Plan::new(
            "p",
            "proj",
            vec![
                step_with_id("a", "do a", &[]),
                step_with_id("b", "do b", &[]),
            ],
        );

        std::fs::write(repo.path().join("dirty.txt"), "dirty\n").unwrap();
        let dirty = manager.prepare_plan(&plan).await.unwrap_err();
        assert!(dirty.to_string().contains("requires a clean git base"));
        std::fs::remove_file(repo.path().join("dirty.txt")).unwrap();

        manager.prepare_plan(&plan).await.unwrap();
        let ctx = AgentSpawnContext {
            plan_id: plan.id,
            project: plan.project.clone(),
            plan_parallel: true,
            layer_width: 2,
            project_root: None,
        };
        let step = step_with_id("a", "do a", &[]);
        let worktree = manager.prepare_step(&ctx, &step).await.unwrap();
        assert!(worktree.is_dir());
        std::fs::write(worktree.join("a.txt"), "a\n").unwrap();
        manager.finish_step(&ctx, &step, &worktree).await.unwrap();

        let branches = manager.branches.lock().unwrap();
        assert_eq!(
            branches
                .get(&plan.id)
                .and_then(|plan_branches| plan_branches.get("a"))
                .map(String::as_str),
            Some(step_branch(&ctx, &step).as_str()),
        );
    }

    /// End-to-end persistence: `finish_step` must write the durable
    /// record, and a *fresh* `GitWorktreeManager` (modelling a daemon
    /// restart) must rehydrate that state via `ensure_branches_loaded`
    /// so dependency resolution still works -- this is the scenario
    /// the in-process map could never satisfy.
    #[tokio::test]
    async fn worktree_manager_persists_branch_and_rehydrates_after_restart() {
        let repo = tempfile::tempdir().expect("repo");
        git_sync(repo.path(), &["init"]);
        std::fs::write(repo.path().join("README.md"), "base\n").unwrap();
        git_sync(repo.path(), &["add", "README.md"]);
        git_sync(
            repo.path(),
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "commit",
                "-m",
                "base",
            ],
        );

        let worktrees = tempfile::tempdir().expect("worktrees");
        let db_dir = tempfile::tempdir().expect("db dir");
        let db_path = db_dir.path().join("flowd.sqlite");

        // ---- First process: open store, run a step, drop the manager.
        let plan = flowd_core::orchestration::Plan::new(
            "p",
            "proj",
            vec![
                step_with_id("a", "do a", &[]),
                step_with_id("b", "do b", &["a"]),
            ],
        );
        let step_a = step_with_id("a", "do a", &[]);
        let ctx = AgentSpawnContext {
            plan_id: plan.id,
            project: plan.project.clone(),
            plan_parallel: true,
            layer_width: 2,
            project_root: None,
        };

        let recorded_branch;
        let recorded_tip_sha;
        {
            let store = Arc::new(
                flowd_storage::step_branch_store::SqliteStepBranchStore::open(&db_path)
                    .expect("open store"),
            );
            let manager = GitWorktreeManager::new(
                repo.path().to_path_buf(),
                worktrees.path().join("worktrees"),
                Some(Arc::clone(&store)),
            );
            manager.prepare_plan(&plan).await.unwrap();
            let worktree_a = manager.prepare_step(&ctx, &step_a).await.unwrap();
            std::fs::write(worktree_a.join("a.txt"), "a\n").unwrap();
            manager.finish_step(&ctx, &step_a, &worktree_a).await.unwrap();

            // Read what got persisted so the second pass can assert
            // the durable record carried the same branch + tip sha
            // that the in-process map saw.
            let rows = store.list_for_plan(plan.id).await.unwrap();
            assert_eq!(rows.len(), 1, "exactly one row written for step a");
            assert_eq!(rows[0].step_id, "a");
            assert_eq!(rows[0].branch, step_branch(&ctx, &step_a));
            assert!(
                !rows[0].tip_sha.is_empty(),
                "tip_sha must be captured from the worktree HEAD"
            );
            assert_eq!(
                rows[0].worktree_path.as_deref(),
                Some(worktree_a.to_string_lossy().as_ref()),
            );
            recorded_branch = rows[0].branch.clone();
            recorded_tip_sha = rows[0].tip_sha.clone();
        }

        // ---- Second process: brand-new manager, brand-new in-memory
        // map, same on-disk DB. `ensure_branches_loaded` must pull
        // step a's branch back so a step b that depends on it can
        // resolve its parent without the original manager ever having
        // existed in this process. Without MIGRATION_007 + the
        // rehydration hook this would error with
        // "dependency branch for step `a` is missing".
        let store2 = Arc::new(
            flowd_storage::step_branch_store::SqliteStepBranchStore::open(&db_path)
                .expect("reopen store"),
        );
        let manager2 = GitWorktreeManager::new(
            repo.path().to_path_buf(),
            worktrees.path().join("worktrees"),
            Some(Arc::clone(&store2)),
        );

        manager2.ensure_branches_loaded(plan.id).await.unwrap();
        let cached_branch = {
            let guard = manager2.branches.lock().unwrap();
            guard
                .get(&plan.id)
                .and_then(|m| m.get("a"))
                .cloned()
                .expect("branch must be rehydrated from durable store")
        };
        assert_eq!(
            cached_branch, recorded_branch,
            "rehydrated branch name must match the persisted record"
        );

        // Sanity: a step that depends on `a` must now resolve through
        // the rehydrated map -- this is the production read site we
        // care about, not just the cache contents.
        let step_b = step_with_id("b", "do b", &["a"]);
        let dep_branches = manager2.dependency_branches(plan.id, &step_b).unwrap();
        assert_eq!(dep_branches, vec![recorded_branch.clone()]);

        // The tip sha is preserved verbatim across the boundary --
        // future integration uses it to fast-forward instead of
        // walking the worktree.
        let rows2 = store2.list_for_plan(plan.id).await.unwrap();
        assert_eq!(rows2.len(), 1);
        assert_eq!(rows2[0].tip_sha, recorded_tip_sha);
    }

    /// `ensure_branches_loaded` must be safe to call even when the
    /// plan has nothing recorded yet -- this is the common case before
    /// any step finishes (and the only case for sequential plans). It
    /// must not error and must not poison the cache against a
    /// subsequent `finish_step`.
    #[tokio::test]
    async fn ensure_branches_loaded_is_noop_for_legacy_empty_plan_state() {
        let dir = tempfile::tempdir().expect("dir");
        let store = Arc::new(
            flowd_storage::step_branch_store::SqliteStepBranchStore::open(
                &dir.path().join("flowd.sqlite"),
            )
            .expect("open store"),
        );
        let manager = GitWorktreeManager::new(
            dir.path().to_path_buf(),
            dir.path().join("worktrees"),
            Some(Arc::clone(&store)),
        );

        let plan_id = uuid::Uuid::new_v4();
        // No prior writes -- DB has the table but no rows for this
        // plan. `ensure_branches_loaded` should treat that as
        // "nothing to rehydrate" and succeed.
        manager.ensure_branches_loaded(plan_id).await.unwrap();
        assert!(
            manager
                .branches
                .lock()
                .unwrap()
                .get(&plan_id)
                .is_none_or(std::collections::HashMap::is_empty),
            "no rows recorded -> in-memory map stays empty for this plan"
        );

        // Calling again is also a no-op: the loaded set guards
        // against a redundant DB read on every subsequent prepare.
        manager.ensure_branches_loaded(plan_id).await.unwrap();
    }

    /// The `modelUsage` block is the one place upstream swaps to
    /// camelCase. Verify the serde renames actually bite: if any of
    /// them regress to `snake_case`, the inner fields silently zero out.
    #[test]
    #[allow(non_snake_case)]
    fn interpret_claude_stdout_parses_modelUsage_camelcase() {
        let raw = r#"{
            "is_error": false,
            "result": "ok",
            "modelUsage": {
                "claude-opus-4-7": {
                    "inputTokens": 11,
                    "outputTokens": 22,
                    "cacheCreationInputTokens": 33,
                    "cacheReadInputTokens": 44,
                    "costUSD": 0.55
                }
            }
        }"#;
        let (_, metrics) = interpret_claude_stdout("step-1", raw).unwrap();
        let m = metrics.expect("metrics present on success");
        let entry = m
            .model_usage
            .get("claude-opus-4-7")
            .expect("camelCase model-usage key must decode");
        assert_eq!(entry.input_tokens, 11);
        assert_eq!(entry.output_tokens, 22);
        assert_eq!(entry.cache_creation_input_tokens, 33);
        assert_eq!(entry.cache_read_input_tokens, 44);
        assert!((entry.cost_usd - 0.55).abs() < f64::EPSILON);
    }

    // ----------------------------------------------------------------------
    // Live smoke tests against a real `claude` binary.
    //
    // These are gated with `#[ignore]` so default `cargo test` runs do not
    // make API calls. Run explicitly with:
    //
    //     cargo test -p flowd-cli --bin flowd live_spawner -- --ignored
    //
    // They skip with a printed warning (and return `Ok`) when `claude` is
    // not on `$PATH`, so CI hosts without the binary do not flag spurious
    // failures. Each costs ~$0.05-0.15 of real API spend; keep them tight.
    // ----------------------------------------------------------------------

    /// Returns `Some(path)` if `claude` is callable; `None` if the binary
    /// is missing. Used by every live test as the skip predicate.
    fn claude_path_or_skip(test_name: &str) -> Option<String> {
        which_in_path("claude").or_else(|| {
            eprintln!(
                "skipping {test_name}: `claude` not found on $PATH; \
                 install Claude Code or unset FLOWD_AGENT_BIN to enable"
            );
            None
        })
    }

    /// Live: a no-tool-required prompt should round-trip cleanly --
    /// JSON envelope parsed, `result` extracted, no permission denials.
    /// Validates the happy path of `--output-format=json` injection.
    #[tokio::test]
    #[ignore = "live: hits real claude API, costs money, skipped by default"]
    async fn live_spawner_claude_simple_reply() {
        let Some(claude) = claude_path_or_skip("live_spawner_claude_simple_reply") else {
            return;
        };
        let s = LocalShellSpawner::new(claude);
        let out = s
            .spawn(ctx(), &step("Reply with exactly the two characters: OK"))
            .await
            .expect("live claude smoke: simple reply must succeed");

        assert_eq!(out.exit_code, Some(0), "claude exited non-zero: {out:?}");
        // The interpreter strips the JSON envelope and returns just the
        // model's `result` field. If we still see raw JSON it means the
        // claude-like detection or the parse path silently regressed.
        assert!(
            !out.stdout.contains("\"is_error\""),
            "stdout still wrapped in JSON envelope; interpret_claude_stdout did not run: {out:?}"
        );
        assert!(
            out.stdout.to_uppercase().contains("OK"),
            "expected 'OK' in result, got: {stdout}",
            stdout = out.stdout
        );

        // Live wire-up check: a real claude run must populate metrics
        // and report non-zero spend. If either fails, the JSON-envelope
        // contract with upstream has drifted and `From<&ClaudeJsonEnvelope>`
        // needs a look.
        assert!(
            out.metrics.is_some(),
            "expected metrics on live claude run; got: {out:?}"
        );
        let m = out.metrics.clone().unwrap();
        assert!(
            m.total_cost_usd > 0.0,
            "expected non-zero total_cost_usd on live claude run; got: {m:?}"
        );
    }

    /// Live: THE smoke test for the `--dangerously-skip-permissions`
    /// fix. Asks claude to use the `Write` tool to create a file in a
    /// tempdir. Without the injected flag, headless mode silently
    /// denies the write while exit-code stays 0 -- the very bug this
    /// patch exists to kill. We assert on the post-spawn filesystem
    /// state, not just the exit code, so the test cannot be fooled by
    /// a chatty refusal.
    #[tokio::test]
    #[ignore = "live: hits real claude API, performs disk write, skipped by default"]
    async fn live_spawner_claude_writes_file_in_cwd() {
        let Some(claude) = claude_path_or_skip("live_spawner_claude_writes_file_in_cwd") else {
            return;
        };
        let workdir = tempfile::tempdir().expect("tempdir");
        let target_rel = "flowd-spawner-smoketest.txt";
        let target_abs = workdir.path().join(target_rel);

        // Construct the spawner with cwd pinned to the tempdir. `new()`
        // does not expose cwd; use struct-literal access (legal from a
        // child module of the crate).
        let s = LocalShellSpawner {
            bin: OsString::from(claude),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: Some(workdir.path().to_path_buf()),
            extra_args: Vec::new(),
        };

        let prompt = format!(
            "Use the Write tool to create exactly one file at the relative path \
             `{target_rel}` containing the literal text `OK`. \
             Do not list directories. Do not run shell commands. Do not edit any \
             other file. Reply with the single word `done` when complete."
        );
        let out = s
            .spawn(ctx(), &step(&prompt))
            .await
            .expect("live claude smoke: write step must succeed");

        assert_eq!(out.exit_code, Some(0), "claude exited non-zero: {out:?}");

        // The contractual assertion: the file is on disk. If this fails,
        // either the spawner is not injecting --dangerously-skip-permissions,
        // or claude refused for another reason and the JSON-envelope failure
        // detection should already have surfaced an Err above.
        assert!(
            target_abs.exists(),
            "claude reported success but did not create {target_abs:?}; \
             stdout was: {stdout}",
            stdout = out.stdout
        );
        let body = std::fs::read_to_string(&target_abs).expect("read created file");
        assert!(
            body.contains("OK"),
            "file exists but body does not contain 'OK': {body:?}"
        );
    }

    /// Live: when an operator forces a non-JSON output format via
    /// `extra_args`, the JSON parse should fall back to passthrough
    /// rather than fail. Validates the escape hatch documented in
    /// the module header.
    #[tokio::test]
    #[ignore = "live: hits real claude API, skipped by default"]
    async fn live_spawner_claude_passthrough_when_extra_args_force_text() {
        let Some(claude) =
            claude_path_or_skip("live_spawner_claude_passthrough_when_extra_args_force_text")
        else {
            return;
        };
        // The injected default is `--output-format=json`. Operator-supplied
        // `extra_args` come AFTER injected args, so a later
        // `--output-format=text` wins per claude's last-flag-wins parsing.
        let s = LocalShellSpawner {
            bin: OsString::from(claude),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: None,
            extra_args: vec!["--output-format=text".to_owned()],
        };
        let out = s
            .spawn(ctx(), &step("Reply with exactly the two characters: OK"))
            .await
            .expect("live claude passthrough: must not error on plain-text output");

        assert_eq!(out.exit_code, Some(0));
        // No JSON envelope -> passthrough -> stdout is raw text.
        assert!(
            !out.stdout.contains("\"is_error\""),
            "stdout looks like JSON despite --output-format=text override: {out:?}"
        );
        assert!(
            out.stdout.to_uppercase().contains("OK"),
            "expected 'OK' in plain-text result, got: {stdout}",
            stdout = out.stdout
        );
    }
}
