//! `flowd start` -- compose all backends, write the PID file, run the MCP
//! stdio server, and shut down gracefully on SIGTERM / Ctrl+C.
//!
//! This command intentionally runs foreground. Service managers
//! (systemd / launchd) or the shell's `&` handle detachment.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::signal::unix::{SignalKind, signal};

use flowd_core::memory::EmbeddingProvider;
use flowd_core::memory::service::MemoryService;
use flowd_core::orchestration::executor::InMemoryPlanExecutor;
use flowd_core::orchestration::{AgentOutput, PlanStep};
use flowd_core::rules::{InMemoryRuleEvaluator, RuleEvaluator};
use flowd_mcp::{FlowdHandlers, McpServer, McpServerConfig};
use flowd_onnx::provider::OnnxEmbedder;
use flowd_storage::sqlite::SqliteBackend;
use flowd_vector::qdrant::{QdrantConfig, QdrantIndex};

use crate::daemon::PidFile;
use crate::output::Style;
use crate::paths::FlowdPaths;

/// Placeholder `AgentSpawner` used until a real implementation lands in a
/// later issue. Every step returns an immediate "not configured" error so
/// any plan confirmation fails loudly rather than silently no-opping.
#[derive(Debug, Default, Clone)]
struct StubAgentSpawner;

impl flowd_core::orchestration::executor::AgentSpawner for StubAgentSpawner {
    async fn spawn(&self, _step: &PlanStep) -> flowd_core::error::Result<AgentOutput> {
        Err(flowd_core::error::FlowdError::PlanExecution(
            "no agent spawner configured -- plan execution is not wired yet".into(),
        ))
    }
}

pub async fn run(paths: &FlowdPaths, style: Style, qdrant_url: Option<String>) -> Result<()> {
    paths.ensure_home()?;

    // Acquire the PID file before we start any I/O so concurrent starts
    // fail fast rather than corrupt the DB / spam Qdrant.
    let pid = PidFile::acquire(paths.pid_file())?;
    eprintln!(
        "{} pid file at {}",
        style.green("started"),
        pid.path().display()
    );

    // Compose backends. Failures here bubble up; the PidFile guard takes
    // care of cleanup on drop.
    let memory_service = compose_memory(paths, qdrant_url.as_deref()).await?;
    let executor = Arc::new(InMemoryPlanExecutor::new(StubAgentSpawner));
    let rules = compose_rules(paths, style)?;
    let handlers = Arc::new(FlowdHandlers::new(
        Arc::new(memory_service),
        executor,
        Arc::new(rules),
    ));

    let server = McpServer::new(handlers, McpServerConfig::default());

    eprintln!(
        "{} mcp stdio server ready -- press Ctrl+C or send SIGTERM to stop",
        style.cyan("running")
    );

    // Run the server against stdio; race it against termination signals.
    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let ctrl_c = tokio::signal::ctrl_c();

    tokio::select! {
        result = server.run_stdio() => {
            if let Err(e) = result {
                tracing::error!(error = %e, "mcp server exited with error");
            }
        }
        _ = ctrl_c => {
            eprintln!("\n{} received Ctrl+C", style.yellow("shutdown"));
        }
        _ = sigterm.recv() => {
            eprintln!("{} received SIGTERM", style.yellow("shutdown"));
        }
    }

    drop(pid); // explicit for clarity; would run anyway at scope exit
    eprintln!("{} daemon stopped", style.green("done"));
    Ok(())
}

async fn compose_memory(
    paths: &FlowdPaths,
    qdrant_url_override: Option<&str>,
) -> Result<MemoryService<SqliteBackend, QdrantIndex, OnnxEmbedder>> {
    let sqlite = SqliteBackend::open(&paths.db_file())
        .with_context(|| format!("open sqlite at {}", paths.db_file().display()))?;

    let embedder = OnnxEmbedder::load(&paths.model_dir()).with_context(|| {
        format!(
            "load onnx embedder from {} (expects model.onnx + tokenizer.json)",
            paths.model_dir().display()
        )
    })?;

    let mut qcfg = QdrantConfig::default();
    if let Some(url) = qdrant_url_override {
        qcfg.url = url.to_owned();
    }
    qcfg.dimensions = embedder.dimensions();
    let qdrant = QdrantIndex::open(&qcfg)
        .await
        .with_context(|| format!("connect to qdrant at {}", qcfg.url))?;

    Ok(MemoryService::new(sqlite, qdrant, embedder))
}

fn compose_rules(paths: &FlowdPaths, style: Style) -> Result<InMemoryRuleEvaluator> {
    let mut eval = InMemoryRuleEvaluator::new();

    // Global rules live under $FLOWD_HOME/rules/; project-scoped rules are
    // resolved relative to cwd (standard convention).
    if paths.rules_dir().is_dir() {
        let n = eval
            .load_rules_from_dir(&paths.rules_dir())
            .with_context(|| format!("load global rules from {}", paths.rules_dir().display()))?;
        eprintln!(
            "{} {n} global rule(s) from {}",
            style.cyan("loaded"),
            paths.rules_dir().display()
        );
    }

    let cwd = std::env::current_dir().context("determine cwd for project rule resolution")?;
    if let Some(project_root) = FlowdPaths::detect_project_root(&cwd) {
        let project_rules = project_root.join(".flowd").join("rules");
        if project_rules.is_dir() {
            let n = eval
                .load_rules_from_dir(&project_rules)
                .with_context(|| format!("load project rules from {}", project_rules.display()))?;
            eprintln!(
                "{} {n} project rule(s) from {}",
                style.cyan("loaded"),
                project_rules.display()
            );
        }
    }

    Ok(eval)
}
