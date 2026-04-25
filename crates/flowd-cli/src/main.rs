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
mod config;
mod daemon;
mod output;
mod paths;
mod plan_compiler;
mod spawner;

use cli::{Cli, Command, PlanAction, RulesAction};
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
        Command::Start {
            qdrant_url,
            plan_event_buffer,
        } => commands::start::run(&paths, style, qdrant_url, plan_event_buffer).await,
        Command::Mcp => commands::mcp::run(&paths).await,
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
        Command::Plan { action } => match action {
            PlanAction::Preview { file, dry_run } => {
                commands::plan::preview(&paths, style, file, dry_run).await
            }
            PlanAction::Events {
                plan_id,
                limit,
                kind,
            } => commands::plan::events(&paths, style, plan_id, limit, kind).await,
            PlanAction::List {
                project,
                status,
                limit,
            } => commands::plan::list(&paths, style, project, status, limit).await,
            PlanAction::Show { plan_id } => commands::plan::show(&paths, style, plan_id).await,
            PlanAction::Recent {
                project,
                status,
                limit,
            } => commands::plan::recent(&paths, style, project, status, limit).await,
            PlanAction::Answer {
                plan_id,
                file,
                defer_remaining,
            } => commands::plan::answer(&paths, style, plan_id, file, defer_remaining).await,
            PlanAction::Refine {
                plan_id,
                feedback,
                file,
            } => commands::plan::refine(&paths, style, plan_id, feedback, file).await,
            PlanAction::Cancel { plan_id } => commands::plan::cancel(&paths, style, plan_id).await,
        },
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
        Command::Init { target } => commands::init::run(target).await,
        Command::Hook { event } => commands::hook::run(event).await,
    }
}
