//! Daemon lifecycle primitives: PID file + signalling.
//!
//! We intentionally do not double-fork or detach the controlling terminal.
//! Modern service managers (systemd, launchd) expect processes to run in
//! the foreground and handle detachment themselves. For ad-hoc use the
//! shell's `&` is sufficient.
//!
//! The PID file is an advisory artifact, not a lock: two `flowd start`
//! invocations can race. We mitigate with a "stale check" (is the PID
//! alive?) on acquire; a truly-robust lock would use `flock(2)` on the
//! file, which is deferred to a later issue.

use std::io::Write;
use std::path::{Path, PathBuf};
use std::process;

use anyhow::{Context, Result, bail};

/// RAII guard for the PID file. Creating a [`PidFile`] writes the current
/// process PID; dropping it removes the file. On graceful shutdown, drop
/// runs via normal scope exit; on SIGTERM, our signal handler is expected
/// to let `main` return so drop still fires.
#[derive(Debug)]
pub struct PidFile {
    path: PathBuf,
    owned: bool,
}

impl PidFile {
    /// Acquire the PID file at `path`. If an existing file points to a
    /// live process, this fails to avoid clobbering a running daemon; if
    /// it points to a dead PID it is overwritten.
    ///
    /// # Errors
    /// Returns an error if a live daemon already holds the PID file or if
    /// the filesystem rejects the write.
    pub fn acquire(path: PathBuf) -> Result<Self> {
        if let Some(existing) = read_pid(&path)? {
            if is_alive(existing) {
                bail!(
                    "pid file {} held by live process {existing}; run `flowd stop` first",
                    path.display()
                );
            }
            tracing::warn!(
                pid = existing,
                file = %path.display(),
                "stale pid file -- replacing"
            );
        }
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create pid-file dir: {}", parent.display()))?;
        }
        let tmp = path.with_extension("pid.tmp");
        {
            let mut f = std::fs::File::create(&tmp)
                .with_context(|| format!("create pid file: {}", tmp.display()))?;
            writeln!(f, "{}", process::id())
                .with_context(|| format!("write pid to {}", tmp.display()))?;
            f.sync_all().ok();
        }
        std::fs::rename(&tmp, &path)
            .with_context(|| format!("install pid file: {}", path.display()))?;
        Ok(Self { path, owned: true })
    }

    #[must_use]
    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for PidFile {
    fn drop(&mut self) {
        if self.owned {
            let _ = std::fs::remove_file(&self.path);
        }
    }
}

/// Read the PID recorded in `path`. Returns `Ok(None)` when the file does
/// not exist; malformed content maps to an error so operators notice.
///
/// # Errors
/// I/O errors other than "not found"; unparseable content.
pub fn read_pid(path: &Path) -> Result<Option<i32>> {
    match std::fs::read_to_string(path) {
        Ok(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return Ok(None);
            }
            let pid: i32 = trimmed.parse().with_context(|| {
                format!("malformed pid file {}: content `{trimmed}`", path.display())
            })?;
            Ok(Some(pid))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("read pid file {}", path.display())),
    }
}

/// Liveness probe: `kill(pid, 0)` returns 0 when the process exists and we
/// have permission to signal it; ESRCH ⇒ dead.
#[must_use]
#[allow(unsafe_code)]
pub fn is_alive(pid: i32) -> bool {
    // SAFETY: kill(2) with signal 0 performs an existence check without
    // actually delivering a signal. It takes two plain `int` arguments by
    // value and returns an `int` -- there are no memory-safety invariants
    // to uphold. Thread-safety is guaranteed by POSIX.
    let rc = unsafe { libc::kill(pid, 0) };
    rc == 0
}

/// Send SIGTERM to `pid`. Errors if the PID is invalid or we lack
/// permission.
///
/// # Errors
/// Propagates errno from `kill(2)`.
#[allow(unsafe_code)]
pub fn send_sigterm(pid: i32) -> Result<()> {
    // SAFETY: kill(2) is async-signal-safe and takes two plain `int`
    // arguments by value. It mutates no caller-owned memory and has no
    // aliasing or lifetime requirements for the calling code.
    let rc = unsafe { libc::kill(pid, libc::SIGTERM) };
    if rc == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error()).with_context(|| format!("SIGTERM to pid {pid}"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tempdir_for(tag: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!("flowd-cli-pidfile-{tag}-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn my_pid() -> i32 {
        i32::try_from(process::id()).expect("test pid fits in i32")
    }

    #[test]
    fn acquire_writes_current_pid_and_cleans_up_on_drop() {
        let dir = tempdir_for("rw");
        let path = dir.join("flowd.pid");
        {
            let guard = PidFile::acquire(path.clone()).unwrap();
            let recorded = read_pid(guard.path()).unwrap().unwrap();
            assert_eq!(recorded, my_pid());
        }
        assert!(!path.exists(), "pid file should be removed on drop");
    }

    #[test]
    fn acquire_rejects_live_existing_pid() {
        let dir = tempdir_for("live");
        let path = dir.join("flowd.pid");
        std::fs::write(&path, format!("{}", process::id())).unwrap();
        let err = PidFile::acquire(path.clone()).unwrap_err();
        assert!(err.to_string().contains("live process"));
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn acquire_replaces_stale_pid_file() {
        let dir = tempdir_for("stale");
        let path = dir.join("flowd.pid");
        // PID 2^31-1 is effectively never a live process on any sane system.
        std::fs::write(&path, "2147483646").unwrap();
        let _guard = PidFile::acquire(path.clone()).unwrap();
        let recorded = read_pid(&path).unwrap().unwrap();
        assert_eq!(recorded, my_pid());
    }

    #[test]
    fn is_alive_reports_self() {
        assert!(is_alive(my_pid()));
    }

    #[test]
    fn is_alive_reports_dead_pid() {
        assert!(!is_alive(2_147_483_646));
    }

    #[test]
    fn read_pid_returns_none_for_missing_file() {
        let dir = tempdir_for("missing");
        let path = dir.join("no.pid");
        assert!(read_pid(&path).unwrap().is_none());
    }

    #[test]
    fn read_pid_errors_on_garbage() {
        let dir = tempdir_for("garbage");
        let path = dir.join("bad.pid");
        std::fs::write(&path, "not-a-number").unwrap();
        assert!(read_pid(&path).is_err());
    }
}
