//! `flowd start` -- compose all backends, write the PID file, run the central
//! local MCP socket server, and shut down gracefully on SIGTERM / Ctrl+C.
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

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio::net::UnixListener;
use tokio::signal::unix::{SignalKind, signal};

use flowd_core::memory::EmbeddingProvider;
use flowd_core::memory::compactor::{ActivityMonitor, Compactor, CompactorConfig};
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::tier::TieringPolicy;
use flowd_core::orchestration::PlanExecutor;
use flowd_core::orchestration::executor::InMemoryPlanExecutor;
use flowd_core::orchestration::gate::RuleGate;
use flowd_core::orchestration::observer::PlanObserver;
use flowd_core::rules::{InMemoryRuleEvaluator, RuleEvaluator};
use flowd_mcp::observer::{DEFAULT_HEALTH_INTERVAL, PlanEventObserver, PlanEventObserverConfig};
use flowd_mcp::summarizer::NoopSummarizer;
use flowd_mcp::{FlowdHandlers, McpServer, McpServerConfig};
use flowd_onnx::provider::OnnxEmbedder;
use flowd_storage::sqlite::SqliteBackend;
use flowd_vector::qdrant::{QdrantConfig, QdrantIndex};

use crate::config::{FlowdConfig, LlmConfig, LlmProvider};
use crate::daemon::PidFile;
use crate::output::Style;
use crate::paths::FlowdPaths;
use crate::plan_compiler::DaemonPlanCompiler;
use crate::spawner::BoxedSpawner;
use flowd_mcp::ClaudeCliCallback;
use flowd_mcp::ClaudeCliConfig as McpClaudeCliConfig;

#[allow(clippy::too_many_lines)]
pub async fn run(
    paths: &FlowdPaths,
    style: Style,
    qdrant_url: Option<String>,
    plan_event_buffer: usize,
) -> Result<()> {
    paths.ensure_home()?;

    // Acquire the PID file before we start any I/O so concurrent starts
    // fail fast rather than corrupt the DB / spam Qdrant.
    let pid = PidFile::acquire(paths.pid_file())?;
    eprintln!(
        "{} pid file at {}",
        style.green("started"),
        pid.path().display()
    );

    // Load `flowd.toml` (or fall back to defaults if absent) before any
    // other wiring so a malformed file fails fast, before the daemon
    // burns a Qdrant connection or rehydrates the executor.
    let config_path = FlowdConfig::default_path(&paths.home);
    let config = FlowdConfig::load(&config_path)
        .with_context(|| format!("load flowd config from {}", config_path.display()))?;
    eprintln!(
        "{} config (compiler: {}, max_questions: {})",
        style.cyan("loaded"),
        config.plan.compiler.wire_name(),
        config.plan.max_questions,
    );
    eprintln!(
        "{} default step timeout: {}",
        style.cyan("using"),
        config
            .plan
            .step_timeout_secs
            .map_or("none (steps may run forever)".into(), |s| format!("{s}s")),
    );
    if matches!(config.plan.compiler, crate::config::CompilerSelection::Llm) {
        log_llm_startup(style, &config.plan.llm)?;
    }

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
    let spawner = Arc::new(BoxedSpawner::auto(&paths.home));
    eprintln!(
        "{} agent spawner: {}",
        style.cyan("using"),
        spawner.description()
    );
    let plan_store = backends.sqlite.plan_store();
    // Plan-lifecycle event log shares the same SQLite connection as plan
    // persistence so a single transaction-coherent file backs every
    // orchestration write (HL-39).
    let plan_event_store = Arc::new(backends.sqlite.plan_event_store());

    // Build the rules evaluator once and share it between the MCP handlers
    // (`rules_check` / `rules_list` tools) and the executor's step-boundary
    // gate, so an agent that ignores `rules_check` is still blocked by the
    // executor before its step is spawned.
    let rules = Arc::new(compose_rules(paths, style)?);
    let rule_gate: Arc<dyn RuleGate> = Arc::clone(&rules) as Arc<dyn RuleGate>;

    // The plan observer fans every executor lifecycle event into the
    // dedicated `plan_events` table. `flowd plan events <id>` reads it
    // back without booting the daemon (WAL-safe).
    let plan_event_observer =
        spawn_plan_event_observer(paths, style, plan_event_store, plan_event_buffer);
    let plan_observer: Arc<dyn PlanObserver> = Arc::clone(&plan_event_observer) as _;

    let executor = Arc::new(
        InMemoryPlanExecutor::from_shared_with_store(spawner, plan_store)
            .with_rule_gate(rule_gate)
            .with_observer(Arc::clone(&plan_observer))
            .with_default_step_timeout(config.plan.step_timeout_secs),
    );
    executor
        .rehydrate()
        .await
        .map_err(|e| anyhow::anyhow!("rehydrate orchestration plans: {e}"))?;

    // Prose-first plan creation (HL-44). The compiler is selected at
    // startup from `flowd.toml`'s `[plan].compiler` key; a sum-type
    // wrapper ([`DaemonPlanCompiler`]) keeps `FlowdHandlers` statically
    // dispatched while still letting the operator switch implementations
    // without recompiling. Three variants today:
    //   * `stub`      -- deterministic markdown parser (default).
    //   * `rejecting` -- prose-first disabled at deployment time.
    //   * `llm`       -- `LlmPlanCompiler` over the `[plan.llm]`
    //                    backend selected by `provider` (claude-cli
    //                    subprocess shell-out today, OpenAI-compatible
    //                    HTTP for the `mlx` provider, claude-http
    //                    reserved for a follow-up). Construction can
    //                    fail (binary missing on $PATH, HTTP client
    //                    init failure, unsupported provider); the
    //                    daemon refuses to start in that case so the
    //                    operator notices before requests pile up.
    let plan_compiler = Arc::new(
        DaemonPlanCompiler::from_selection(config.plan.compiler, &config.plan.llm)
            .map_err(|e| anyhow::anyhow!("instantiate plan compiler: {e}"))?,
    );
    log_llm_router(style, &plan_compiler);

    let handlers = Arc::new(
        FlowdHandlers::new(memory_service, Arc::clone(&executor), plan_compiler, rules)
            .with_question_budget(Some(config.plan.max_questions))
            .with_activity_monitor(monitor)
            // Share the plan-event sink with the executor so clarification
            // and refinement transitions land in the same audit log as
            // step-level events.
            .with_observer(plan_observer),
    );
    let server = Arc::new(McpServer::new(handlers, McpServerConfig::default()));
    let socket = BoundSocket::bind(paths.socket_file())?;

    eprintln!(
        "{} mcp socket server ready at {} -- press Ctrl+C or send SIGTERM to stop",
        style.cyan("running"),
        socket.path().display(),
    );

    // Run the central socket server; per-client `flowd mcp` processes proxy
    // stdio frames to this listener.
    let mut sigterm = signal(SignalKind::terminate()).context("install SIGTERM handler")?;
    let ctrl_c = tokio::signal::ctrl_c();

    tokio::select! {
        result = run_socket_server(server, socket.listener()) => {
            if let Err(e) = result {
                tracing::error!(error = %e, "mcp socket server exited with error");
            }
        }
        _ = ctrl_c => {
            eprintln!("\n{} received Ctrl+C", style.yellow("shutdown"));
        }
        _ = sigterm.recv() => {
            eprintln!("{} received SIGTERM", style.yellow("shutdown"));
        }
    }

    // Drain any still-`Running` plans so their executing tasks commit a
    // terminal status to SQLite. Without this the next daemon start will
    // settle them as Interrupted via the rehydrate safety net -- correct,
    // but noisier than a graceful Cancelled.
    let running = executor.list_running();
    if !running.is_empty() {
        eprintln!(
            "{} draining {} running plan(s)",
            style.yellow("shutdown"),
            running.len()
        );
        let mut join_set = tokio::task::JoinSet::new();
        for id in &running {
            let exec = Arc::clone(&executor);
            let id = *id;
            join_set.spawn(async move {
                let _ = exec.cancel(id).await;
            });
        }
        let drain = async { while join_set.join_next().await.is_some() {} };
        match tokio::time::timeout(Duration::from_secs(10), drain).await {
            Ok(()) => {
                eprintln!("{} drained {} plan(s)", style.cyan("done"), running.len());
            }
            Err(_) => {
                eprintln!(
                    "{} plan drain timed out after 10s; the rehydrate path will mark stragglers Interrupted on next start",
                    style.yellow("warning")
                );
            }
        }
    }

    shutdown_plan_event_observer(paths, style, &plan_event_observer).await;

    // Stop the compactor *before* dropping the PID file so any in-flight
    // compaction completes (or aborts cleanly) while the daemon is still
    // marked alive. `stop` is idempotent and short-circuits on Drop.
    compactor_handle.stop().await;

    drop(pid); // explicit for clarity; would run anyway at scope exit
    drop(socket); // remove the Unix socket before reporting the daemon stopped
    eprintln!("{} daemon stopped", style.green("done"));
    Ok(())
}

struct BoundSocket {
    path: PathBuf,
    listener: UnixListener,
}

impl BoundSocket {
    fn bind(path: PathBuf) -> Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .with_context(|| format!("create socket dir: {}", parent.display()))?;
        }
        match std::fs::remove_file(&path) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => {
                return Err(e).with_context(|| format!("remove stale socket {}", path.display()));
            }
        }
        let listener = UnixListener::bind(&path)
            .with_context(|| format!("bind daemon socket at {}", path.display()))?;
        Ok(Self { path, listener })
    }

    fn path(&self) -> &std::path::Path {
        &self.path
    }

    fn listener(&self) -> &UnixListener {
        &self.listener
    }
}

impl Drop for BoundSocket {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.path);
    }
}

async fn run_socket_server<H>(
    server: Arc<McpServer<H>>,
    listener: &UnixListener,
) -> std::io::Result<()>
where
    H: flowd_mcp::McpHandlers + 'static,
{
    loop {
        let (stream, _addr) = listener.accept().await?;
        let (read, write) = stream.into_split();
        let server = Arc::clone(&server);
        tokio::spawn(async move {
            if let Err(e) = server.run(read, write).await {
                tracing::debug!(error = %e, "mcp socket client disconnected with error");
            }
        });
    }
}

/// Build the bounded-queue plan-event observer (HL-40).
///
/// `capacity == 0` is rejected by [`PlanEventObserver::new`]; we
/// translate it to the documented default here so a typo on the CLI
/// just emits a warning rather than crashing the daemon.
fn spawn_plan_event_observer(
    paths: &FlowdPaths,
    style: Style,
    store: Arc<flowd_storage::plan_event_store::SqlitePlanEventStore>,
    requested_capacity: usize,
) -> Arc<PlanEventObserver> {
    let capacity = if requested_capacity == 0 {
        eprintln!(
            "{} --plan-event-buffer 0 is invalid; falling back to default 1024",
            style.yellow("warning")
        );
        1024
    } else {
        requested_capacity
    };
    let observer = PlanEventObserver::new(
        store,
        PlanEventObserverConfig {
            capacity,
            health_file: Some(paths.plan_event_health_file()),
            health_interval: DEFAULT_HEALTH_INTERVAL,
        },
    );
    eprintln!(
        "{} plan-event observer (capacity: {}, health: {})",
        style.cyan("spawned"),
        capacity,
        paths.plan_event_health_file().display()
    );
    Arc::new(observer)
}

/// Flush buffered events to the store, log the drop count, and remove
/// the stale health snapshot. 5s deadline matches systemd's default
/// `TimeoutStopSec`.
async fn shutdown_plan_event_observer(
    paths: &FlowdPaths,
    style: Style,
    observer: &PlanEventObserver,
) {
    let report = observer.shutdown(Duration::from_secs(5)).await;
    if report.dropped > 0 {
        eprintln!(
            "{} plan-event observer dropped {} event(s) over the lifetime of this daemon (channel saturated)",
            style.yellow("note"),
            report.dropped
        );
    }
    if report.timed_out {
        eprintln!(
            "{} plan-event observer drain timed out; some buffered events may be lost",
            style.yellow("warning")
        );
    }
    let health_path = paths.plan_event_health_file();
    if health_path.exists() {
        let _ = std::fs::remove_file(&health_path);
    }
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

/// Print the LLM compiler banner and probe any provider that requires
/// an out-of-process dependency (today: the `claude` CLI binary).
///
/// The probe is asymmetric on purpose:
///
/// * `claude-cli` is fail-fast -- if the binary is missing, the daemon
///   refuses to start so the operator sees a clear error before the
///   first `plan_create` call. We probe both the primary tier *and*
///   the optional refine tier, since either can be invoked during a
///   single plan's lifetime.
/// * `mlx` is lazy -- the local OpenAI-compatible server may legally
///   not be running yet (the daemon and `mlx_lm.server` are typically
///   started in either order). The first compile attempt surfaces a
///   clean transport error if the server never comes up.
/// * `claude-http` is currently rejected at compiler-construction time
///   (the transport is unimplemented), so no startup probe is needed.
fn log_llm_startup(style: Style, cfg: &LlmConfig) -> Result<()> {
    log_provider_line(style, "primary", cfg.provider, cfg);
    if let Some(refine) = cfg.refine.as_ref() {
        // Reuse the same banner shape but flag this as the refine tier
        // so an operator scanning the log sees both lines distinctly.
        eprintln!(
            "{} llm refine tier configured ({} model={})",
            style.cyan("loaded"),
            refine.provider.wire_name(),
            describe_provider_model(refine.provider, &refine_as_llm_view(refine)),
        );
    } else {
        eprintln!(
            "{} llm refine tier not configured (refine() falls back to primary)",
            style.dim("note"),
        );
    }

    // Fail-fast probe for the `claude` CLI binary. We only probe the
    // tiers that actually use it; an MLX-only deployment skips this
    // entirely.
    if needs_claude_cli_probe(cfg) {
        let cli_cfg = McpClaudeCliConfig {
            binary: cfg.claude_cli.binary.clone(),
            model: cfg.claude_cli.model.clone(),
            effort: cfg.claude_cli.effort,
            timeout: Duration::from_secs(cfg.claude_cli.timeout_secs),
        };
        let resolved = ClaudeCliCallback::probe_binary(&cli_cfg).with_context(
            || "probe claude CLI binary at startup (set provider = \"mlx\" to skip the probe)",
        )?;
        eprintln!(
            "{} claude CLI at {}",
            style.cyan("found"),
            resolved.display()
        );
    }
    Ok(())
}

/// Print the per-tier "loaded llm backend" line. Keeps formatting
/// consistent across the primary and refine tiers.
fn log_provider_line(style: Style, label: &str, provider: LlmProvider, cfg: &LlmConfig) {
    eprintln!(
        "{} llm {} backend ({} model={})",
        style.cyan("loaded"),
        label,
        provider.wire_name(),
        describe_provider_model(provider, cfg),
    );
}

/// Render the model identifier for a provider, plus whatever
/// supplementary detail makes sense for that transport. Kept here
/// rather than on `LlmProvider` because the format is banner-policy,
/// not data.
fn describe_provider_model(provider: LlmProvider, cfg: &LlmConfig) -> String {
    match provider {
        LlmProvider::ClaudeCli => {
            let effort = cfg
                .claude_cli
                .effort
                .map(|l| format!(" effort={}", l.wire_name()))
                .unwrap_or_default();
            format!(
                "{} binary={}{effort}",
                cfg.claude_cli.model,
                cfg.claude_cli.binary.display(),
            )
        }
        LlmProvider::Mlx => format!("{} url={}", cfg.mlx.model, cfg.mlx.base_url),
        LlmProvider::ClaudeHttp => format!(
            "{} url={} (transport not yet implemented)",
            cfg.claude_http.model, cfg.claude_http.base_url
        ),
    }
}

/// Build a synthetic `LlmConfig` view from the refine tier's overrides.
///
/// The refine block carries a fully-resolved set of per-provider
/// subsections (defaulted at parse time, just like the primary), so we
/// can hand its values straight to `describe_provider_model` without
/// merging anything in from the primary. Kept as a thin shim purely so
/// the banner formatting stays uniform across the two tiers.
fn refine_as_llm_view(refine: &crate::config::RefineConfig) -> LlmConfig {
    LlmConfig {
        provider: refine.provider,
        claude_cli: refine.claude_cli.clone(),
        mlx: refine.mlx.clone(),
        claude_http: refine.claude_http.clone(),
        refine: None,
    }
}

/// Print the runtime LLM router shape so an operator can verify which
/// providers a `compiler_override` request can target. Pulled out of
/// `run` to keep that function under the pedantic-clippy line ceiling.
fn log_llm_router(style: Style, plan_compiler: &DaemonPlanCompiler) {
    let Some(router) = plan_compiler.llm_router() else {
        return;
    };
    let primary = router.primary_provider().wire_name();
    let refine = router
        .refine_provider()
        .map_or("(falls back to primary)", LlmProvider::wire_name);
    eprintln!(
        "{} llm router: primary={primary} refine={refine} overrides=[{}]",
        style.dim("note"),
        router.configured_overrides().join(", "),
    );
}

fn needs_claude_cli_probe(cfg: &LlmConfig) -> bool {
    cfg.provider == LlmProvider::ClaudeCli
        || cfg
            .refine
            .as_ref()
            .is_some_and(|r| r.provider == LlmProvider::ClaudeCli)
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
