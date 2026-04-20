//! `flowd start` -- compose all backends, write the PID file, run the MCP
//! stdio server, and shut down gracefully on SIGTERM / Ctrl+C.
//!
//! This command intentionally runs foreground. Service managers
//! (systemd / launchd) or the shell's `&` handle detachment.
//!
//! ## Wiring overview
//!
//! `start` is the integration point where every flowd subsystem meets:
//!
//! ```text
//!  SqliteBackend ──┐
//!  QdrantIndex   ──┼─→ MemoryService::from_shared ──→ FlowdHandlers ──→ McpServer
//!  OnnxEmbedder  ──┘                                       │
//!                  │                                       │
//!                  └─→ Compactor (background tokio task) ←─┴─ ActivityMonitor
//!                          │                                  (touched on every
//!                          └─→ NoopSummarizer (Hot → Warm)     handler call)
//!
//!  LocalShellSpawner / UnconfiguredSpawner ─→ InMemoryPlanExecutor + SqlitePlanStore ─→ FlowdHandlers
//! ```
//!
//! Sharing the three backends as `Arc` is what unlocks the compactor:
//! the same `SQLite` connection / Qdrant client / ONNX session powers both
//! the live request path and the background sweep, so a summarized
//! observation is immediately visible to the next `memory_context` call.

use std::sync::Arc;

use anyhow::{Context, Result};
use tokio::signal::unix::{SignalKind, signal};

use flowd_core::memory::EmbeddingProvider;
use flowd_core::memory::compactor::{ActivityMonitor, Compactor, CompactorConfig};
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::tier::TieringPolicy;
use flowd_core::orchestration::executor::InMemoryPlanExecutor;
use flowd_core::orchestration::gate::RuleGate;
use flowd_core::orchestration::observer::PlanObserver;
use flowd_core::rules::{InMemoryRuleEvaluator, RuleEvaluator};
use flowd_mcp::summarizer::NoopSummarizer;
use flowd_mcp::{FlowdHandlers, McpServer, McpServerConfig, MemoryPlanObserver};
use flowd_onnx::provider::OnnxEmbedder;
use flowd_storage::sqlite::SqliteBackend;
use flowd_vector::qdrant::{QdrantConfig, QdrantIndex};

use crate::daemon::PidFile;
use crate::output::Style;
use crate::paths::FlowdPaths;
use crate::spawner::BoxedSpawner;

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

    // ---- Memory backends (shared between live handlers & compactor) ----
    let backends = compose_backends(paths, qdrant_url.as_deref()).await?;
    let memory_service = Arc::new(MemoryService::from_shared(
        Arc::clone(&backends.sqlite),
        Arc::clone(&backends.qdrant),
        Arc::clone(&backends.embedder),
    ));

    // ---- Background compactor: Hot → Warm summaries, Warm → Cold demotion.
    let monitor = ActivityMonitor::new();
    let compactor_handle = spawn_compactor(&backends, monitor.clone(), style);

    // ---- Plan execution wiring: select an agent CLI from $PATH or env.
    let spawner = Arc::new(BoxedSpawner::auto());
    eprintln!(
        "{} agent spawner: {}",
        style.cyan("using"),
        spawner.description()
    );
    let plan_store = backends.sqlite.plan_store();

    // Build the rules evaluator once and share it between the MCP handlers
    // (`rules_check` / `rules_list` tools) and the executor's step-boundary
    // gate, so an agent that ignores `rules_check` is still blocked by the
    // executor before its step is spawned.
    let rules = Arc::new(compose_rules(paths, style)?);
    let rule_gate: Arc<dyn RuleGate> = Arc::clone(&rules) as Arc<dyn RuleGate>;

    // The plan observer mirrors every executor lifecycle event into flowd's
    // memory under a deterministic per-plan session, so `flowd history` and
    // semantic search both surface what the executor actually did -- no
    // longer dependent on the agent voluntarily calling `memory_store`.
    let plan_observer: Arc<dyn PlanObserver> =
        Arc::new(MemoryPlanObserver::new(Arc::clone(&memory_service)));

    let executor = Arc::new(
        InMemoryPlanExecutor::from_shared_with_store(spawner, plan_store)
            .with_rule_gate(rule_gate)
            .with_observer(plan_observer),
    );
    executor
        .rehydrate()
        .await
        .map_err(|e| anyhow::anyhow!("rehydrate orchestration plans: {e}"))?;

    let handlers = Arc::new(
        FlowdHandlers::new(memory_service, executor, rules).with_activity_monitor(monitor),
    );
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

    // Stop the compactor *before* dropping the PID file so any in-flight
    // compaction completes (or aborts cleanly) while the daemon is still
    // marked alive. `stop` is idempotent and short-circuits on Drop.
    compactor_handle.stop().await;

    drop(pid); // explicit for clarity; would run anyway at scope exit
    eprintln!("{} daemon stopped", style.green("done"));
    Ok(())
}

/// Backends needed by both the live request path and the background
/// compactor. Held as `Arc`s so cloning is cheap and refcount-managed.
struct SharedBackends {
    sqlite: Arc<SqliteBackend>,
    qdrant: Arc<QdrantIndex>,
    embedder: Arc<OnnxEmbedder>,
}

async fn compose_backends(
    paths: &FlowdPaths,
    qdrant_url_override: Option<&str>,
) -> Result<SharedBackends> {
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

    Ok(SharedBackends {
        sqlite: Arc::new(sqlite),
        qdrant: Arc::new(qdrant),
        embedder: Arc::new(embedder),
    })
}

/// Construct and spawn the background compactor. Returns the handle whose
/// `stop()` must be awaited at shutdown so any in-flight pass drains cleanly.
fn spawn_compactor(
    backends: &SharedBackends,
    monitor: ActivityMonitor,
    style: Style,
) -> flowd_core::memory::compactor::CompactorHandle {
    let summarizer = Arc::new(NoopSummarizer::new());
    let policy = TieringPolicy::standard();
    let config = CompactorConfig::default();
    let compactor = Compactor::new(
        Arc::clone(&backends.sqlite),
        Arc::clone(&backends.qdrant),
        Arc::clone(&backends.embedder),
        summarizer,
        policy,
        monitor,
        config,
    );

    eprintln!(
        "{} compactor (idle threshold: {:?}, hot→warm: {:?}, warm→cold: {:?})",
        style.cyan("spawned"),
        config.idle_threshold,
        policy.hot_max_age().to_std().unwrap_or_default(),
        policy.warm_max_age().to_std().unwrap_or_default(),
    );
    compactor.spawn()
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
