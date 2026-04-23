//! Subprocess backend that shells out to the local `claude` CLI.
//!
//! The CLI is the recommended quality-first transport for the prose
//! compiler: it routes through whatever Anthropic credentials the
//! `claude` binary already manages (login token, MAX subscription, OAuth
//! flow, ...) so `flowd` never has to ingest, persist, or rotate an
//! `ANTHROPIC_API_KEY`. The trade-off is that latency is bounded by
//! whatever the user's `claude` install does -- typically a single
//! `claude -p` invocation per compile round, typically a few seconds
//! for the Sonnet tier and climbing into the tens of seconds when the
//! operator pins an Opus tier on a long prompt.
//!
//! ## Wire format
//!
//! We invoke the binary as
//!
//! ```text
//! <binary> -p --model <model> --output-format text
//! ```
//!
//! and pipe the prompt to its stdin. Stdin (rather than `--prompt
//! <arg>`) is mandatory: compile-time prompts include the full JSON
//! schema, prior decisions, open questions, and the user's prose --
//! easily tens of KB on a busy plan -- which would blow past `ARG_MAX`
//! on macOS / Linux. Stdin has no such limit.
//!
//! `--output-format text` keeps the CLI in non-streaming mode and
//! suppresses the JSON envelope it would otherwise emit, so the
//! callback's caller (the compiler's response parser) sees the model
//! response verbatim and can apply the same fence-stripping +
//! `{...}` extraction it already runs on every backend.
//!
//! ## Errors and timeouts
//!
//! The callback honours a per-request timeout (configured in
//! `flowd.toml`'s `[plan.llm.claude_cli].timeout_secs`). A timeout
//! kills the child process and returns
//! [`flowd_core::error::FlowdError::Internal`] with a human-readable
//! message; non-zero exit codes also become `Internal`, with the first
//! 512 bytes of stderr surfaced in the message so the daemon log is
//! self-explanatory. Empty stdout (i.e. the CLI succeeded but said
//! nothing) is treated as a transport failure for the same reason.

use std::future::Future;
use std::path::PathBuf;
use std::process::Stdio;
use std::time::Duration;

use flowd_core::error::{FlowdError, Result};
use tokio::io::AsyncWriteExt;
use tokio::process::Command;

use crate::compiler::LlmCallback;

/// Configuration for [`ClaudeCliCallback`].
///
/// All fields come straight from `[plan.llm.claude_cli]` in
/// `flowd.toml`; defaults live in the CLI crate's config layer so this
/// struct mirrors the on-disk shape without baking in policy.
#[derive(Debug, Clone)]
pub struct ClaudeCliConfig {
    /// Path or bare name of the executable. Bare names are resolved on
    /// `$PATH` by the OS when [`Command::spawn`] is invoked; absolute
    /// paths are used verbatim. The startup probe in
    /// [`ClaudeCliCallback::probe_binary`] runs the same resolution
    /// logic so wiring failures fail fast.
    pub binary: PathBuf,
    /// Model identifier passed via `--model`. Anything `claude -p`
    /// accepts is fair game: tier aliases (`sonnet`, `opus`, `haiku`)
    /// auto-resolve to the latest build of that tier, while
    /// fully-pinned identifiers (e.g. `claude-sonnet-4-5`) reproduce
    /// byte-for-byte across hosts.
    pub model: String,
    /// Per-request timeout. Caps the entire shell-out (start + stdin
    /// write + stdout read + reap), not just the model's first byte.
    pub timeout: Duration,
}

/// Subprocess-backed [`LlmCallback`] talking to the local `claude` CLI.
///
/// Stateless beyond its [`ClaudeCliConfig`]; spawning is per-request so
/// the daemon never holds a long-running child between compiles.
#[derive(Debug, Clone)]
pub struct ClaudeCliCallback {
    cfg: ClaudeCliConfig,
}

impl ClaudeCliCallback {
    /// Build a callback over `cfg`.
    ///
    /// Cheap; the heavy lifting (binary resolution + sanity check) lives
    /// in [`Self::probe_binary`] which the daemon should call at
    /// startup so missing CLIs fail fast rather than at first request.
    #[must_use]
    pub const fn new(cfg: ClaudeCliConfig) -> Self {
        Self { cfg }
    }

    /// Borrow the underlying config. Useful for the daemon's startup
    /// banner and for tests that want to assert wiring without
    /// snapshotting the whole struct.
    #[must_use]
    pub const fn config(&self) -> &ClaudeCliConfig {
        &self.cfg
    }

    /// Resolve the configured binary on `$PATH` (when bare) or check
    /// that the configured absolute path exists and is executable
    /// (when not). Returns the resolved absolute path on success.
    ///
    /// # Errors
    /// Returns `FlowdError::Internal` with a message naming the
    /// configured binary and the search strategy that failed, so the
    /// startup error tells the operator exactly what to install or
    /// where to point the config.
    pub fn probe_binary(cfg: &ClaudeCliConfig) -> Result<PathBuf> {
        if has_separator(&cfg.binary) {
            // Treat as a literal path. We do not consult `$PATH` in
            // this case because the operator was explicit.
            let exists = std::fs::metadata(&cfg.binary).is_ok();
            if !exists {
                return Err(FlowdError::Internal(format!(
                    "[plan.llm.claude_cli] binary `{}` does not exist; \
                     install the Claude CLI or update flowd.toml to point at it",
                    cfg.binary.display()
                )));
            }
            Ok(cfg.binary.clone())
        } else {
            // Bare name -- walk $PATH ourselves so the diagnostic is
            // helpful (the OS-level "no such file or directory" from
            // Command::spawn drops the binary name).
            resolve_on_path(&cfg.binary).ok_or_else(|| {
                FlowdError::Internal(format!(
                    "[plan.llm.claude_cli] binary `{}` not found on $PATH; \
                     install the Claude CLI (https://docs.anthropic.com/en/docs/claude-code) \
                     or set [plan.llm.claude_cli].binary to a full path",
                    cfg.binary.display()
                ))
            })
        }
    }
}

/// True when the path contains a directory separator (`/`, or `\\` on
/// Windows). Used to decide whether to consult `$PATH` or treat the
/// value as a literal location.
fn has_separator(p: &std::path::Path) -> bool {
    let s = p.to_string_lossy();
    s.contains(std::path::MAIN_SEPARATOR) || s.contains('/')
}

/// Walk `$PATH` looking for `name`. Mirrors what `which(1)` does, minus
/// platform-specific extensions handling -- adequate for our use case
/// since the `claude` CLI ships as a single executable on every
/// platform Anthropic supports.
fn resolve_on_path(name: &std::path::Path) -> Option<PathBuf> {
    let path_var = std::env::var_os("PATH")?;
    for dir in std::env::split_paths(&path_var) {
        let candidate = dir.join(name);
        if std::fs::metadata(&candidate).is_ok() {
            return Some(candidate);
        }
    }
    None
}

impl LlmCallback for ClaudeCliCallback {
    fn complete(&self, prompt: String) -> impl Future<Output = Result<String>> + Send {
        let cfg = self.cfg.clone();
        async move {
            let mut child = Command::new(&cfg.binary)
                .arg("-p")
                .arg("--model")
                .arg(&cfg.model)
                .arg("--output-format")
                .arg("text")
                .stdin(Stdio::piped())
                .stdout(Stdio::piped())
                .stderr(Stdio::piped())
                .spawn()
                .map_err(|e| {
                    FlowdError::Internal(format!(
                        "spawn `{} -p --model {}`: {e}; \
                         is the Claude CLI installed and on $PATH?",
                        cfg.binary.display(),
                        cfg.model
                    ))
                })?;

            // Pipe the prompt to stdin. Closing stdin signals EOF to the
            // CLI so it stops waiting for more input. We bind the handle
            // first because `child.stdin` is `take()`-only.
            let mut stdin = child
                .stdin
                .take()
                .ok_or_else(|| FlowdError::Internal("claude CLI: stdin pipe missing".into()))?;
            stdin.write_all(prompt.as_bytes()).await.map_err(|e| {
                FlowdError::Internal(format!("write prompt to claude CLI stdin: {e}"))
            })?;
            stdin
                .flush()
                .await
                .map_err(|e| FlowdError::Internal(format!("flush claude CLI stdin: {e}")))?;
            // Drop the handle to send EOF; an explicit drop makes the
            // intent obvious to anyone reading the code.
            drop(stdin);

            // Race the timeout against the wait. On timeout we kill the
            // child explicitly so it doesn't linger past the Future.
            let out = match tokio::time::timeout(cfg.timeout, child.wait_with_output()).await {
                Ok(Ok(out)) => out,
                Ok(Err(e)) => {
                    return Err(FlowdError::Internal(format!("wait for claude CLI: {e}")));
                }
                Err(_elapsed) => {
                    return Err(FlowdError::Internal(format!(
                        "claude CLI timed out after {:?}; consider raising \
                         [plan.llm.claude_cli].timeout_secs or shrinking the prompt",
                        cfg.timeout
                    )));
                }
            };

            if !out.status.success() {
                let stderr = String::from_utf8_lossy(&out.stderr);
                let stdout = String::from_utf8_lossy(&out.stdout);
                return Err(FlowdError::Internal(format!(
                    "claude CLI exited with status {}: stderr=`{}` stdout=`{}`",
                    out.status,
                    truncate(&stderr, 512),
                    truncate(&stdout, 256),
                )));
            }

            let body = String::from_utf8(out.stdout).map_err(|e| {
                FlowdError::Internal(format!("claude CLI stdout was not valid UTF-8: {e}"))
            })?;

            if body.trim().is_empty() {
                return Err(FlowdError::Internal(
                    "claude CLI returned 0 with empty stdout; the upstream may have produced \
                     no completion -- inspect the binary directly to confirm"
                        .into(),
                ));
            }

            Ok(body)
        }
    }
}

/// UTF-8-safe truncation used for embedding child-process output into
/// error messages. Mirrors the helper in [`crate::llm::openai`] -- we
/// duplicate rather than share to keep the modules independently
/// re-orderable.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let end = s
        .char_indices()
        .nth(max_chars)
        .map_or(s.len(), |(idx, _)| idx);
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::TempDir;

    fn cfg_for(binary: PathBuf) -> ClaudeCliConfig {
        ClaudeCliConfig {
            binary,
            model: "sonnet".into(),
            timeout: Duration::from_secs(5),
        }
    }

    /// Write an executable shell script at `dir/<name>` and return the
    /// Mutex shared by every test that mutates `$PATH` (or any other
    /// process-global env var). Rust runs unit tests in parallel by
    /// default; without this guard a `set_var("PATH", ...)` race
    /// against a sibling test that spawns a child process (which
    /// inherits PATH and resolves `cat` etc.) produces flaky 127
    /// "command not found" failures.
    ///
    /// Lives at module scope so any future test that touches the
    /// environment can opt in by acquiring the same lock.
    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// path. Used to fake the `claude` CLI in callback unit tests so
    /// nothing depends on a real install being present.
    fn write_fake_claude(dir: &TempDir, name: &str, body: &str) -> PathBuf {
        let path = dir.path().join(name);
        std::fs::write(&path, body).unwrap();
        let mut perms = std::fs::metadata(&path).unwrap().permissions();
        perms.set_mode(0o755);
        std::fs::set_permissions(&path, perms).unwrap();
        path
    }

    // ---- probe_binary --------------------------------------------------

    #[test]
    fn probe_binary_resolves_absolute_path_when_present() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fake_claude(&dir, "claude", "#!/bin/sh\nexit 0\n");
        let cfg = cfg_for(path.clone());
        let resolved = ClaudeCliCallback::probe_binary(&cfg).unwrap();
        assert_eq!(resolved, path);
    }

    #[test]
    fn probe_binary_errors_on_missing_absolute_path() {
        let cfg = cfg_for(PathBuf::from("/definitely/does/not/exist/claude"));
        let err = ClaudeCliCallback::probe_binary(&cfg).unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("does not exist"), "{s}");
        assert!(s.contains("/definitely/does/not/exist/claude"), "{s}");
    }

    #[test]
    #[allow(
        unsafe_code,
        clippy::undocumented_unsafe_blocks,
        clippy::multiple_unsafe_ops_per_block
    )]
    fn probe_binary_resolves_bare_name_via_path_var() {
        // Serialise with every other test that touches $PATH so a
        // sibling test can't observe an empty PATH while we hold the
        // override (and vice versa).
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let dir = tempfile::tempdir().unwrap();
        let _ = write_fake_claude(&dir, "fake-flowd-claude", "#!/bin/sh\nexit 0\n");
        // Override $PATH so the test is hermetic.
        let original = std::env::var_os("PATH");
        // SAFETY: ENV_LOCK guarantees no other test reads/writes PATH
        // for the duration of this block.
        unsafe { std::env::set_var("PATH", dir.path()) };

        let cfg = cfg_for(PathBuf::from("fake-flowd-claude"));
        let result = ClaudeCliCallback::probe_binary(&cfg);

        // Restore PATH before any assertion so a panic doesn't leak
        // the override into other tests in this binary.
        // SAFETY: same single-thread assumption as above.
        unsafe {
            match original {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }

        let resolved = result.unwrap();
        assert_eq!(resolved, dir.path().join("fake-flowd-claude"));
    }

    #[test]
    #[allow(
        unsafe_code,
        clippy::undocumented_unsafe_blocks,
        clippy::multiple_unsafe_ops_per_block
    )]
    fn probe_binary_errors_with_helpful_hint_on_path_miss() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        // Empty PATH guarantees a miss without polluting the
        // user's environment.
        let original = std::env::var_os("PATH");
        // SAFETY: ENV_LOCK guards against parallel readers/writers.
        unsafe { std::env::set_var("PATH", "") };

        let cfg = cfg_for(PathBuf::from("claude"));
        let result = ClaudeCliCallback::probe_binary(&cfg);

        // SAFETY: same single-thread assumption as above.
        unsafe {
            match original {
                Some(v) => std::env::set_var("PATH", v),
                None => std::env::remove_var("PATH"),
            }
        }

        let err = result.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("not found on $PATH"), "{s}");
        assert!(
            s.contains("claude.com") || s.contains("anthropic.com"),
            "{s}"
        );
    }

    // ---- complete() over a fake CLI -----------------------------------

    #[tokio::test]
    async fn complete_returns_stdout_on_zero_exit() {
        let dir = tempfile::tempdir().unwrap();
        // The fake echoes a fixed JSON body to stdout regardless of
        // input -- enough to verify the callback round-trips.
        let path = write_fake_claude(
            &dir,
            "claude",
            "#!/bin/sh\n/bin/cat > /dev/null\nprintf '%s' '{\"plan_name\":\"x\"}'\n",
        );
        let cb = ClaudeCliCallback::new(cfg_for(path));
        let out = cb.complete("anything".into()).await.unwrap();
        assert_eq!(out, "{\"plan_name\":\"x\"}");
    }

    #[tokio::test]
    async fn complete_propagates_nonzero_exit_with_stderr_in_message() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fake_claude(
            &dir,
            "claude",
            "#!/bin/sh\n/bin/cat > /dev/null\necho 'boom: rate limited' 1>&2\nexit 7\n",
        );
        let cb = ClaudeCliCallback::new(cfg_for(path));
        let err = cb.complete("anything".into()).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("status"), "{s}");
        assert!(s.contains("boom: rate limited"), "{s}");
    }

    #[tokio::test]
    async fn complete_empty_stdout_is_classified_as_internal_failure() {
        let dir = tempfile::tempdir().unwrap();
        let path = write_fake_claude(&dir, "claude", "#!/bin/sh\n/bin/cat > /dev/null\n");
        let cb = ClaudeCliCallback::new(cfg_for(path));
        let err = cb.complete("anything".into()).await.unwrap_err();
        assert!(format!("{err}").contains("empty stdout"));
    }

    #[tokio::test]
    async fn complete_kills_child_on_timeout() {
        let dir = tempfile::tempdir().unwrap();
        // Sleep longer than the timeout so the wait races out.
        let path = write_fake_claude(
            &dir,
            "claude",
            "#!/bin/sh\n/bin/cat > /dev/null\n/bin/sleep 60\n",
        );
        let cfg = ClaudeCliConfig {
            binary: path,
            model: "sonnet".into(),
            timeout: Duration::from_millis(150),
        };
        let cb = ClaudeCliCallback::new(cfg);
        let err = cb.complete("hi".into()).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("timed out"), "{s}");
    }

    #[tokio::test]
    async fn complete_pipes_prompt_to_stdin() {
        let dir = tempfile::tempdir().unwrap();
        // Echo stdin to stdout so we can see what was piped through.
        let path = write_fake_claude(&dir, "claude", "#!/bin/sh\n/bin/cat\n");
        let cb = ClaudeCliCallback::new(cfg_for(path));
        let out = cb.complete("hello via stdin".into()).await.unwrap();
        assert_eq!(out, "hello via stdin");
    }

    // ---- helpers -------------------------------------------------------

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_appends_ellipsis_when_clipping() {
        let out = truncate("abcdefghij", 5);
        assert_eq!(out, "abcde…");
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        let s = "αβγδε";
        let out = truncate(s, 3);
        assert_eq!(out, "αβγ…");
    }

    #[test]
    fn has_separator_distinguishes_path_from_bare_name() {
        assert!(!has_separator(std::path::Path::new("claude")));
        assert!(has_separator(std::path::Path::new("/usr/local/bin/claude")));
        assert!(has_separator(std::path::Path::new("./claude")));
    }
}
