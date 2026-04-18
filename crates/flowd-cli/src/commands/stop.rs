//! `flowd stop` -- send SIGTERM to the daemon recorded in the PID file.

use anyhow::{Result, bail};

use crate::daemon::{is_alive, read_pid, send_sigterm};
use crate::output::Style;
use crate::paths::FlowdPaths;

pub fn run(paths: &FlowdPaths, style: Style) -> Result<()> {
    let pid_path = paths.pid_file();
    let Some(pid) = read_pid(&pid_path)? else {
        bail!(
            "no pid file at {} -- is the daemon running?",
            pid_path.display()
        );
    };

    if !is_alive(pid) {
        eprintln!(
            "{} pid {pid} is not alive -- cleaning up stale {}",
            style.yellow("warn"),
            pid_path.display()
        );
        let _ = std::fs::remove_file(&pid_path);
        return Ok(());
    }

    send_sigterm(pid)?;
    eprintln!("{} SIGTERM sent to pid {pid}", style.green("ok"));
    Ok(())
}
