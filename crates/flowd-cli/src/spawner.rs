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
//! <bin> [extra_args...] <prompt_flag> "<step.prompt>"
//! ```
//!
//! `prompt_flag` defaults to `-p` (Claude Code, `cursor-agent`); override
//! it via `$FLOWD_AGENT_PROMPT_FLAG` for CLIs that use a different name
//! (e.g. `--message` for Aider). Extra leading arguments come from
//! `$FLOWD_AGENT_EXTRA_ARGS` (whitespace-split).
//!
//! Stdout is captured verbatim into [`AgentOutput::stdout`]. Stderr is
//! emitted via `tracing` (warn on non-zero exit, debug otherwise) so it
//! surfaces in the daemon log without polluting the JSON-RPC channel.

use std::ffi::OsString;
use std::path::PathBuf;
use std::process::Stdio;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::executor::AgentSpawner;
use flowd_core::orchestration::{AgentOutput, PlanStep};
use tokio::process::Command;

/// Default candidate binaries searched on `$PATH` when `$FLOWD_AGENT_BIN`
/// is unset. Order matters: first match wins.
const DEFAULT_CANDIDATES: &[&str] = &["claude", "codex", "aider", "cursor-agent"];

/// Default flag passed before the prompt argument. Suitable for Claude Code
/// and `cursor-agent`; override via `$FLOWD_AGENT_PROMPT_FLAG` for others.
const DEFAULT_PROMPT_FLAG: &str = "-p";

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

/// Spawner used as the fallback when no agent binary is discoverable. Fails
/// every `spawn` with an actionable error so misconfiguration surfaces the
/// first time a plan is confirmed, instead of silently succeeding.
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
/// without leaking concrete types into the rest of `start.rs`.
pub enum BoxedSpawner {
    Local(LocalShellSpawner),
    Unconfigured(UnconfiguredSpawner),
}

impl BoxedSpawner {
    /// Build the best available spawner for the current environment.
    pub fn auto() -> Self {
        LocalShellSpawner::detect().map_or(Self::Unconfigured(UnconfiguredSpawner), Self::Local)
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

    /// Smoke test: `echo` stands in for an agent that just prints its
    /// argv. With the default `-p` flag the invocation is
    /// `echo -p "<prompt>"`, which exits 0 and contains the prompt.
    #[tokio::test]
    async fn echo_spawner_returns_stdout() {
        let s = LocalShellSpawner::new("echo");
        let out = s.spawn(&step("hello flowd")).await.unwrap();
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
}
