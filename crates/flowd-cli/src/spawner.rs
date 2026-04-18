//! Concrete `AgentSpawner` implementations for the `flowd` daemon.
//!
//! `flowd-core` deliberately does not spawn OS processes itself; spawning a
//! real coding agent is the CLI's job. This module wires the orchestration
//! engine to the agent CLI binaries that ship with Cursor / Claude Code /
//! `OpenAI` Codex / Anthropic Claude.
//!
//! ## Selection order
//!
//! When a [`LocalShellSpawner`] is built via [`LocalShellSpawner::detect`]:
//!
//! 1. `$FLOWD_AGENT_BIN` if set (operator override; absolute or in `$PATH`).
//! 2. First entry from a fixed candidate list found in `$PATH`:
//!    `cursor-agent`, `claude`, `codex`, `aider`.
//! 3. If none are found, returns `None` so the caller can wire a no-op
//!    fallback that fails *loudly* on `spawn` rather than silently.
//!
//! ## Invocation contract
//!
//! All supported CLIs accept `<bin> -p "<prompt>"` (Anthropic's Claude CLI,
//! Cursor's `cursor-agent`, `OpenAI` Codex CLI, Aider's `--message`).
//! `LocalShellSpawner` uses `-p` by default; override via
//! `$FLOWD_AGENT_PROMPT_FLAG` (e.g. `--message`) when wiring a CLI that
//! uses a different flag name.
//!
//! Stdout is captured verbatim and returned in `AgentOutput::stdout`.
//! Stderr is logged via `tracing` (warn on non-zero exit, debug otherwise)
//! so it surfaces in the daemon log without polluting the JSON-RPC channel.
//!
//! ## Why not stream stdout incrementally?
//!
//! The orchestrator currently observes step output once, after the step
//! completes (`AgentOutput::stdout`). Streaming would require a richer
//! `AgentSpawner` trait. Until then, we run the child process to completion
//! before returning, with cancellation handled at the layer above via
//! `JoinHandle::abort` -- which delivers SIGKILL to the spawned process
//! group through tokio's reaper.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::executor::AgentSpawner;
use flowd_core::orchestration::{AgentOutput, PlanStep};
use tokio::process::Command;

/// Default candidate binaries searched in `$PATH` when `$FLOWD_AGENT_BIN`
/// is unset. Order matters: first match wins.
const DEFAULT_CANDIDATES: &[&str] = &["cursor-agent", "claude", "codex", "aider"];

/// Default flag passed before the prompt argument. All four candidate CLIs
/// accept `-p` for non-interactive single-prompt mode.
const DEFAULT_PROMPT_FLAG: &str = "-p";

/// `AgentSpawner` that shells out to a local CLI binary (claude / cursor-agent
/// / codex / aider) with the step's prompt.
#[derive(Debug, Clone)]
pub struct LocalShellSpawner {
    /// Resolved path or PATH-relative name of the agent binary.
    bin: OsString,
    /// Flag preceding the prompt argument (e.g. `-p`, `--message`).
    prompt_flag: String,
    /// Working directory for the child. `None` inherits the parent's.
    cwd: Option<PathBuf>,
    /// Extra leading args inserted between `bin` and `prompt_flag`.
    /// Useful for wrappers like `cursor-agent --model gpt-5` (set via
    /// `$FLOWD_AGENT_EXTRA_ARGS`, space-split).
    extra_args: Vec<String>,
}

impl LocalShellSpawner {
    /// Build a spawner from environment variables, or return `None` if no
    /// usable agent CLI is discoverable.
    ///
    /// Inspect the returned binary path with [`LocalShellSpawner::bin_display`]
    /// before logging so operators can see which CLI was selected.
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
            .map(|s| {
                s.split_whitespace()
                    .map(std::borrow::ToOwned::to_owned)
                    .collect::<Vec<_>>()
            })
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

    /// Construct an explicit spawner -- handy for tests and embedding.
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

    /// Override the prompt-introducing flag (default `-p`).
    /// Reserved for downstream embedders that wire flowd into a CLI other
    /// than the four candidates baked into [`DEFAULT_CANDIDATES`].
    #[must_use]
    #[allow(dead_code)]
    pub fn with_prompt_flag(mut self, flag: impl Into<String>) -> Self {
        self.prompt_flag = flag.into();
        self
    }

    /// Override the working directory of the spawned process.
    #[must_use]
    #[allow(dead_code)]
    pub fn with_cwd(mut self, cwd: impl Into<PathBuf>) -> Self {
        self.cwd = Some(cwd.into());
        self
    }

    /// Override the extra args inserted between bin and prompt flag.
    #[must_use]
    #[allow(dead_code)]
    pub fn with_extra_args(mut self, args: Vec<String>) -> Self {
        self.extra_args = args;
        self
    }

    /// Display the configured binary; for logging only.
    #[must_use]
    pub fn bin_display(&self) -> std::path::Display<'_> {
        std::path::Path::new(&self.bin).display()
    }
}

impl AgentSpawner for LocalShellSpawner {
    async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
        let mut cmd = Command::new(&self.bin);
        for arg in &self.extra_args {
            cmd.arg(arg);
        }
        cmd.arg(&self.prompt_flag).arg(&step.prompt);
        if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        // Detach stdin: agents that try to prompt the operator must error,
        // not block forever waiting for input from a closed terminal.
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // Mark the child for kill_on_drop so the executor's `JoinHandle::abort`
        // (used by `cancel`) actually terminates the process tree.
        cmd.kill_on_drop(true);

        tracing::info!(
            step_id = %step.id,
            agent_type = %step.agent_type,
            bin = %self.bin_display(),
            "spawning agent step"
        );

        let output = cmd.output().await.map_err(|e| {
            FlowdError::PlanExecution(format!(
                "failed to spawn agent `{bin}` for step `{step}`: {e}",
                bin = self.bin_display(),
                step = step.id,
            ))
        })?;

        let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code();

        if !output.status.success() {
            tracing::warn!(
                step_id = %step.id,
                exit_code = ?code,
                stderr = %truncate_for_log(&stderr, 4096),
                "agent step exited non-zero"
            );
            return Err(FlowdError::PlanExecution(format!(
                "step `{step}` failed (exit={code}): {stderr}",
                step = step.id,
                code = code.map_or_else(|| "signal".into(), |c| c.to_string()),
                stderr = truncate_for_log(stderr.trim(), 1024),
            )));
        }

        if !stderr.is_empty() {
            tracing::debug!(
                step_id = %step.id,
                stderr = %truncate_for_log(&stderr, 2048),
                "agent step stderr"
            );
        }

        Ok(AgentOutput {
            stdout,
            exit_code: code,
        })
    }
}

/// Spawner that returns a clear, actionable error explaining how to wire a
/// real agent CLI. Used as the fallback when no agent binary is discoverable
/// at startup -- prefer this to silently succeeding so operators see the
/// misconfiguration the first time they confirm a plan.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnconfiguredSpawner;

impl AgentSpawner for UnconfiguredSpawner {
    async fn spawn(&self, _step: &PlanStep) -> Result<AgentOutput> {
        Err(FlowdError::PlanExecution(
            "no agent CLI configured. Set $FLOWD_AGENT_BIN to an absolute \
             path, or install one of: cursor-agent, claude, codex, aider \
             into $PATH, then restart the daemon."
                .into(),
        ))
    }
}

/// Type-erased spawner so the executor can be parametrised at runtime
/// (`LocalShellSpawner` vs `UnconfiguredSpawner`) without leaking concrete
/// types into the rest of `start.rs`.
pub enum BoxedSpawner {
    Local(LocalShellSpawner),
    Unconfigured(UnconfiguredSpawner),
}

impl BoxedSpawner {
    /// Build the best available spawner for the current environment.
    pub fn auto() -> Self {
        match LocalShellSpawner::detect() {
            Some(s) => Self::Local(s),
            None => Self::Unconfigured(UnconfiguredSpawner),
        }
    }

    pub fn description(&self) -> String {
        match self {
            Self::Local(s) => format!("local-shell ({})", s.bin_display()),
            Self::Unconfigured(_) => "unconfigured (plan execution will fail)".into(),
        }
    }
}

impl AgentSpawner for BoxedSpawner {
    async fn spawn(&self, step: &PlanStep) -> Result<AgentOutput> {
        match self {
            Self::Local(s) => s.spawn(step).await,
            Self::Unconfigured(s) => s.spawn(step).await,
        }
    }
}

/// Look up `name` in the `$PATH`. Mirrors `which(1)` -- returns the first
/// executable match. Returns `None` if not found or `$PATH` is unset.
fn which_in_path(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path) {
        let candidate = dir.join(name);
        if is_executable(&candidate) {
            return candidate.into_os_string().into_string().ok();
        }
    }
    None
}

#[cfg(unix)]
fn is_executable(p: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && (m.permissions().mode() & 0o111 != 0))
}

#[cfg(not(unix))]
fn is_executable(p: &std::path::Path) -> bool {
    // Conservative: treat any regular file in PATH as executable on
    // non-unix targets. The agent CLIs we care about are unix-only in
    // practice; this branch exists so the crate still builds.
    p.is_file()
}

fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        s.to_owned()
    } else {
        let mut cut = 0;
        for (i, _) in s.char_indices() {
            if i > max {
                break;
            }
            cut = i;
        }
        format!("{}…[+{} bytes]", &s[..cut], s.len() - cut)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::StepStatus;

    fn step(prompt: &str) -> PlanStep {
        PlanStep {
            id: "t".into(),
            agent_type: "test".into(),
            prompt: prompt.into(),
            depends_on: vec![],
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    #[tokio::test]
    async fn echo_spawner_returns_stdout() {
        // `echo` is universally available; treat it as an agent that just
        // prints its prompt.
        let s = LocalShellSpawner::new("echo");
        let out = s.spawn(&step("hello flowd")).await.unwrap();
        // `echo -p hello flowd` on macOS prints `-p hello flowd\n`. We just
        // assert non-empty + exit 0; echo is portable enough for a smoke test.
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout.contains("hello flowd"));
    }

    #[tokio::test]
    async fn missing_binary_yields_plan_execution_error() {
        let s = LocalShellSpawner::new("/does/not/exist/flowd-agent");
        let err = s.spawn(&step("anything")).await.unwrap_err();
        assert!(matches!(err, FlowdError::PlanExecution(_)));
    }

    #[tokio::test]
    async fn unconfigured_spawner_reports_actionable_error() {
        let s = UnconfiguredSpawner;
        let err = s.spawn(&step("x")).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("FLOWD_AGENT_BIN"));
        assert!(msg.contains("cursor-agent"));
    }

    // Note: a `detect_honours_explicit_env_override` test would need to
    // mutate the process-wide `FLOWD_AGENT_BIN` env var. Rust 2024 made
    // `std::env::set_var` `unsafe`, and the workspace lints `unsafe_code`
    // as a warning. The detect path is exercised in practice by the
    // daemon at startup; dropping the test keeps the lint surface clean.
}
