//! `flowd` CLI entry point.
//!
//! The binary is intentionally thin: it parses clap args, resolves paths,
//! installs tracing, and hands off to a subcommand handler. All real
//! behaviour lives in [`commands`]; all formatting in [`output`];
//! path/resource resolution in [`paths`] and [`daemon`].

use anyhow::Result;
use clap::Parser;

mod cli;
mod commands;
mod daemon;
mod output;
mod paths;

use cli::{Cli, Command, RulesAction};
use output::Style;
use paths::FlowdPaths;

#[tokio::main]
async fn main() -> Result<()> {
    // Tracing goes to stderr so stdout stays reserved for structured
    // CLI output (and, in `flowd start`, MCP JSON-RPC frames).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "flowd=info".into()),
        )
        .init();

    let cli = Cli::parse();

    let paths = match cli.home {
        Some(h) => FlowdPaths::with_home(h),
        None => FlowdPaths::from_env()?,
    };
    let style = if cli.no_color {
        Style::plain()
    } else {
        Style::from_env()
    };

    match cli.command {
        Command::Start { qdrant_url } => commands::start::run(&paths, style, qdrant_url).await,
        Command::Stop => commands::stop::run(&paths, style),
        Command::Search {
            query,
            project,
            limit,
        } => commands::search::run(&paths, style, query, project, limit).await,
        Command::History {
            project,
            since,
            limit,
        } => commands::history::run(&paths, style, project, since, limit).await,
        Command::Plan { file, dry_run } => commands::plan::run(&paths, style, file, dry_run).await,
        Command::Rules { action } => match action {
            RulesAction::List { project, file } => {
                commands::rules::list(&paths, style, project.as_deref(), file.as_deref())
            }
        },
        Command::Observe {
            project,
            session,
            metadata,
        } => commands::observe::run(&paths, style, project, session, metadata).await,
        Command::Status => commands::status::run(&paths, style).await,
        Command::Export { output, project } => {
            commands::export::run(&paths, style, output, project).await
        }
    }
}
