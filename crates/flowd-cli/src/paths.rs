//! Resolve filesystem paths used by the CLI.
//!
//! Layout (XDG-like; no actual XDG dep):
//!
//! ```text
//! $FLOWD_HOME | ~/.flowd/
//!   flowd.pid               -- daemon PID file
//!   flowd.sock              -- local MCP proxy socket
//!   memory.db               -- SQLite + FTS5
//!   memory.db-wal           -- WAL file (SQLite managed)
//!   rules/                  -- global rules
//!   models/                 -- ONNX model + tokenizer.json
//! ```
//!
//! `$FLOWD_HOME` overrides the root for tests and unusual deployments.

use std::env;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};

/// Root directory for flowd data. Honours `$FLOWD_HOME`, falls back to
/// `~/.flowd/`.
#[derive(Debug, Clone)]
pub struct FlowdPaths {
    pub home: PathBuf,
}

impl FlowdPaths {
    /// Resolve paths from the environment. Does not create any directories.
    ///
    /// # Errors
    /// Returns an error if neither `$FLOWD_HOME` nor `$HOME` is set.
    pub fn from_env() -> Result<Self> {
        let home = if let Some(explicit) = env::var_os("FLOWD_HOME") {
            PathBuf::from(explicit)
        } else {
            let home_dir = env::var_os("HOME")
                .context("neither $FLOWD_HOME nor $HOME is set; cannot locate flowd data dir")?;
            PathBuf::from(home_dir).join(".flowd")
        };
        Ok(Self { home })
    }

    /// Override the root directly (primarily for tests).
    #[must_use]
    pub fn with_home(home: PathBuf) -> Self {
        Self { home }
    }

    #[must_use]
    pub fn pid_file(&self) -> PathBuf {
        self.home.join("flowd.pid")
    }

    #[must_use]
    pub fn socket_file(&self) -> PathBuf {
        self.home.join("flowd.sock")
    }

    #[must_use]
    pub fn db_file(&self) -> PathBuf {
        self.home.join("memory.db")
    }

    #[must_use]
    pub fn rules_dir(&self) -> PathBuf {
        self.home.join("rules")
    }

    #[must_use]
    pub fn model_dir(&self) -> PathBuf {
        self.home.join("models")
    }

    /// Path the running daemon writes the plan-event observer health
    /// snapshot to (HL-40). Read by `flowd status` out-of-band; absent
    /// when the daemon was never started or is currently stopped.
    #[must_use]
    pub fn plan_event_health_file(&self) -> PathBuf {
        self.home.join("plan-event-observer.health")
    }

    /// Directory holding the Claude Code session-id -> flowd session
    /// UUID mapping files, one file per Claude session id. Used by the
    /// `flowd hook` subcommand to keep session-start, post-tool-use, and
    /// session-end observations pinned to the same flowd session.
    #[must_use]
    pub fn hook_sessions_dir(&self) -> PathBuf {
        self.home.join("hook-sessions")
    }

    /// Root where `plan_integrate` materialises its temporary worktrees.
    /// Lives under a distinct subdir from the spawner's `worktrees/` so a
    /// per-plan cleanup loop keyed on the spawner's path does not sweep
    /// integration state by accident.
    #[must_use]
    pub fn integrate_worktrees_dir(&self) -> PathBuf {
        self.home.join("integrate-worktrees")
    }

    /// Create the home dir (and parents) if missing. Idempotent.
    ///
    /// # Errors
    /// Propagates filesystem errors other than "already exists".
    pub fn ensure_home(&self) -> Result<()> {
        std::fs::create_dir_all(&self.home)
            .with_context(|| format!("create flowd home: {}", self.home.display()))
    }

    /// Best-effort project root detection: if `cwd` contains a `.flowd/`
    /// directory, return it; else climb parents until a project marker is
    /// found. Returns `None` when no project scope can be inferred.
    #[must_use]
    pub fn detect_project_root(cwd: &Path) -> Option<PathBuf> {
        let mut cursor = Some(cwd);
        while let Some(dir) = cursor {
            if dir.join(".flowd").is_dir() || dir.join(".git").exists() {
                return Some(dir.to_path_buf());
            }
            cursor = dir.parent();
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_home_composes_paths() {
        let p = FlowdPaths::with_home(PathBuf::from("/tmp/flowd-test"));
        assert_eq!(p.pid_file(), PathBuf::from("/tmp/flowd-test/flowd.pid"));
        assert_eq!(p.socket_file(), PathBuf::from("/tmp/flowd-test/flowd.sock"));
        assert_eq!(p.db_file(), PathBuf::from("/tmp/flowd-test/memory.db"));
        assert_eq!(p.rules_dir(), PathBuf::from("/tmp/flowd-test/rules"));
        assert_eq!(p.model_dir(), PathBuf::from("/tmp/flowd-test/models"));
    }

    #[test]
    fn detect_project_root_finds_git_marker() {
        let tmp = tempdir_for("detect-git");
        std::fs::create_dir_all(tmp.join(".git")).unwrap();
        let sub = tmp.join("a").join("b");
        std::fs::create_dir_all(&sub).unwrap();
        let root = FlowdPaths::detect_project_root(&sub).unwrap();
        assert_eq!(root, tmp);
    }

    #[test]
    fn detect_project_root_returns_none_when_no_markers() {
        let tmp = tempdir_for("detect-none");
        assert!(FlowdPaths::detect_project_root(&tmp).is_none());
    }

    fn tempdir_for(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("flowd-cli-test-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }
}
