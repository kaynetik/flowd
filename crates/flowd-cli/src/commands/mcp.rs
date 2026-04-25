//! `flowd mcp` -- per-client stdio bridge to the central daemon.
//!
//! Cursor and Claude expect an MCP server on stdio. The real flowd daemon now
//! owns state behind a local Unix socket; this command is intentionally tiny:
//! connect to that socket, then proxy bytes in both directions.

use anyhow::{Context, Result};
use tokio::io::{AsyncWriteExt, copy};
use tokio::net::UnixStream;

use crate::paths::FlowdPaths;

pub async fn run(paths: &FlowdPaths) -> Result<()> {
    let socket = paths.socket_file();
    let stream = UnixStream::connect(&socket)
        .await
        .with_context(|| format!("connect to flowd daemon socket at {}", socket.display()))?;
    let (mut daemon_read, mut daemon_write) = stream.into_split();
    let mut stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();

    let to_daemon = async {
        copy(&mut stdin, &mut daemon_write).await?;
        daemon_write.shutdown().await
    };
    let to_client = async {
        copy(&mut daemon_read, &mut stdout).await?;
        stdout.shutdown().await
    };

    tokio::try_join!(to_daemon, to_client)?;
    Ok(())
}
