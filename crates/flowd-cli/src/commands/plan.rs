//! `flowd plan` -- inspect orchestration plans.
//!
//! Two subcommands:
//!
//! * `flowd plan preview <file>` -- load a plan file, render a preview,
//!   and optionally prompt for Y/N confirmation.
//! * `flowd plan events <plan_id>` -- print the persisted lifecycle event
//!   log for a plan, read straight from `SQLite` (HL-39).
//!
//! Execution itself is deferred: until a concrete `AgentSpawner` is wired
//! up (future issue) "confirming" can only print the plan id a daemon
//! would accept. We surface this clearly so users don't think plans run.

use std::collections::HashMap;
use std::io::{BufRead, Read, Write};
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow, bail};

use flowd_core::orchestration::plan_events::{PlanEventQuery, PlanEventStore, StoredPlanEvent};
use flowd_core::orchestration::{
    Answer, Plan, PlanCompiler, PlanDraftSnapshot, PlanStatus, PlanStore, build_preview, load_plan,
};
use flowd_storage::plan_event_store::SqlitePlanEventStore;
use flowd_storage::plan_store::SqlitePlanStore;
use serde::Deserialize;
use uuid::Uuid;

use crate::config::FlowdConfig;
use crate::daemon;
use crate::output::{Style, banner};
use crate::paths::FlowdPaths;
use crate::plan_compiler::DaemonPlanCompiler;

pub async fn preview(
    _paths: &FlowdPaths,
    style: Style,
    file: PathBuf,
    dry_run: bool,
) -> Result<()> {
    let plan = load_plan(&file).with_context(|| format!("load plan {}", file.display()))?;
    let preview = build_preview(&plan).context("build plan preview")?;

    print!("{}", banner(&format!("plan: {}", preview.name), style));
    println!("  id:          {}", style.dim(&preview.plan_id.to_string()));
    println!("  steps:       {}", preview.total_steps);
    println!("  agents:      {}", preview.total_agents);
    println!("  parallelism: {}", preview.max_parallelism);

    render_layers(&preview.execution_order, style);
    render_dependencies(&preview.dependency_graph, style);

    if dry_run {
        println!("\n{} dry-run -- skipping confirmation", style.dim("note:"));
        return Ok(());
    }

    // Confirmation prompt. stdin is blocking, so we hop onto the blocking
    // thread pool to avoid stalling the tokio reactor.
    let confirmed = tokio::task::spawn_blocking(|| prompt_yes_no("proceed? [y/N] "))
        .await
        .context("confirmation prompt panicked")??;

    if confirmed {
        println!(
            "\n{} plan execution is not yet wired up in this CLI cut.",
            style.yellow("note:")
        );
        println!("       the daemon's `plan_confirm` MCP tool will execute plans once a concrete");
        println!("       `AgentSpawner` is provided -- tracked separately.");
    } else {
        println!("{}", style.dim("aborted."));
    }
    Ok(())
}

pub async fn events(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    limit: usize,
    kinds: Vec<String>,
) -> Result<()> {
    let plan_id = Uuid::parse_str(plan_id_arg.trim())
        .with_context(|| format!("parse plan id `{plan_id_arg}` as UUID"))?;

    // Open the same SQLite file the daemon writes to. WAL mode keeps
    // this read-side safe even while `flowd start` is live.
    let db_path = paths.db_file();
    if !db_path.exists() {
        anyhow::bail!(
            "no flowd database at {}; start the daemon at least once to initialise it",
            db_path.display()
        );
    }
    let store = SqlitePlanEventStore::open(&db_path)
        .with_context(|| format!("open plan event store at {}", db_path.display()))?;

    let kinds = normalise_kinds(kinds);
    let query = PlanEventQuery {
        kinds,
        limit: limit.max(1),
    };
    let rows = store
        .list_for_plan(plan_id, query)
        .await
        .with_context(|| format!("list events for plan {plan_id}"))?;

    print!("{}", banner(&format!("events: {plan_id}"), style));
    if rows.is_empty() {
        println!("  {}", style.dim("(no events recorded)"));
        return Ok(());
    }
    for evt in &rows {
        render_event(evt, style);
    }
    Ok(())
}

/// Flatten `--kind a,b --kind c` and `--kind a --kind b,c` into a single
/// trimmed-and-deduped list. The clap `value_delimiter = ','` handles
/// the splitting; this layer drops empty entries and surrounding
/// whitespace from operator typos.
fn normalise_kinds(kinds: Vec<String>) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    for k in kinds {
        let trimmed = k.trim();
        if trimmed.is_empty() {
            continue;
        }
        let owned = trimmed.to_owned();
        if !out.contains(&owned) {
            out.push(owned);
        }
    }
    out
}

fn render_event(evt: &StoredPlanEvent, style: Style) {
    let ts = evt.created_at.format("%Y-%m-%d %H:%M:%SZ");
    let header = format!(
        "{ts}  {kind}",
        ts = style.dim(&ts.to_string()),
        kind = style.cyan(&evt.kind),
    );
    let suffix = match (evt.step_id.as_deref(), evt.agent_type.as_deref()) {
        (Some(step), Some(agent)) => format!("  step={step}  agent={agent}"),
        (Some(step), None) => format!("  step={step}"),
        _ => String::new(),
    };
    println!("  {header}{suffix}");
    if let Some(detail) = describe_payload(&evt.kind, &evt.payload) {
        println!("    {}", style.dim(&detail));
    }
}

/// One-line human summary of the event payload. Falls back to the raw
/// JSON for anything we don't recognise.
fn describe_payload(kind: &str, payload: &serde_json::Value) -> Option<String> {
    use flowd_core::orchestration::plan_events::kind as k;

    match kind {
        k::SUBMITTED => payload
            .get("name")
            .and_then(serde_json::Value::as_str)
            .map(|n| format!("name: {n}")),
        k::STEP_COMPLETED => payload
            .get("output")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("output: {}", truncate_one_line(s, 120))),
        k::STEP_FAILED => payload
            .get("error")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("error: {s}")),
        k::STEP_REFUSED => payload
            .get("reason")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("reason: {s}")),
        k::FINISHED => payload
            .get("status")
            .and_then(serde_json::Value::as_str)
            .map(|s| format!("status: {s}")),
        _ => None,
    }
}

fn truncate_one_line(s: &str, max: usize) -> String {
    let single = s.split_whitespace().collect::<Vec<_>>().join(" ");
    if single.len() <= max {
        single
    } else {
        // Use a char-boundary safe slice: walk char_indices to find the
        // largest cut <= max - 1 (room for the ellipsis).
        let target = max.saturating_sub(1);
        let mut cut = 0usize;
        for (i, _) in single.char_indices() {
            if i > target {
                break;
            }
            cut = i;
        }
        format!("{}…", &single[..cut])
    }
}

fn render_layers(layers: &[Vec<String>], style: Style) {
    println!("\n{}", style.bold("execution layers:"));
    if layers.is_empty() {
        println!("  {}", style.dim("(none)"));
        return;
    }
    for (i, layer) in layers.iter().enumerate() {
        println!(
            "  {label} {steps}",
            label = style.cyan(&format!("L{i}:")),
            steps = layer.join(", "),
        );
    }
}

fn render_dependencies(graph: &HashMap<String, Vec<String>>, style: Style) {
    println!("\n{}", style.bold("dependencies:"));
    // Stable ordering: keys in lexical order, so output is deterministic
    // across hash-map iteration variance.
    let mut entries: Vec<_> = graph.iter().filter(|(_, v)| !v.is_empty()).collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));
    if entries.is_empty() {
        println!("  {}", style.dim("(none)"));
        return;
    }
    for (id, deps) in entries {
        println!("  {id} ← {}", deps.join(", "));
    }
}

/// `flowd plan answer` -- offline-mode counterpart to the `plan_answer`
/// MCP tool. Reads answers (a JSON array of `PlanAnswerEntry`-shaped
/// objects) from `file` (or stdin when `file == Some("-")`), invokes
/// the configured compiler, and persists the resulting draft.
///
/// Refuses to run when the daemon holds a live PID file: the only safe
/// way to mutate plan state with the daemon up is through the same
/// MCP wiring every other client uses, so we exit with a clear pointer
/// rather than racing against the in-memory executor.
pub async fn answer(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    file: Option<PathBuf>,
    defer_remaining: bool,
) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan answer")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let entries: Vec<CliAnswerEntry> = if let Some(path) = file.as_ref() {
        let raw = read_text_input(path).context("read --file payload")?;
        if raw.trim().is_empty() {
            Vec::new()
        } else {
            serde_json::from_str(&raw).context("parse --file as a JSON answers array")?
        }
    } else {
        Vec::new()
    };
    if entries.is_empty() && !defer_remaining {
        bail!(
            "plan answer: pass --file with at least one answer or --defer-remaining; both omitted"
        );
    }

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_draft(plan_id).await?;
    let snapshot = PlanDraftSnapshot::from_plan(&plan);
    let answers: Vec<(String, Answer)> = entries
        .into_iter()
        .map(|e| (e.question_id, e.answer))
        .collect();

    let output = runner
        .compiler
        .apply_answers(snapshot, answers, defer_remaining)
        .await
        .context("compile answers")?;

    plan.apply_compile_output(output);
    runner.save(&plan).await?;
    print_plan_summary(&plan, style, "answers applied");
    Ok(())
}

/// `flowd plan refine` -- offline-mode counterpart to the `plan_refine`
/// MCP tool. Same daemon-liveness guard as [`answer`].
pub async fn refine(
    paths: &FlowdPaths,
    style: Style,
    plan_id_arg: String,
    feedback: Option<String>,
    file: Option<PathBuf>,
) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan refine")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let feedback = match (feedback, file) {
        (Some(s), None) => s,
        (None, Some(p)) => read_text_input(&p).context("read --file feedback")?,
        (None, None) => bail!("plan refine: pass either --feedback <STRING> or --file <PATH>"),
        (Some(_), Some(_)) => unreachable!("clap conflicts_with prevents this combination"),
    };
    if feedback.trim().is_empty() {
        bail!("plan refine: feedback must be non-empty");
    }

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_draft(plan_id).await?;
    let prior_open: std::collections::HashSet<String> =
        plan.open_questions.iter().map(|q| q.id.clone()).collect();
    let snapshot = PlanDraftSnapshot::from_plan(&plan);

    let output = runner
        .compiler
        .refine(snapshot, feedback)
        .await
        .context("compile refinement")?;
    plan.apply_compile_output(output);
    runner.save(&plan).await?;

    let reopened = plan
        .open_questions
        .iter()
        .any(|q| !prior_open.contains(&q.id))
        || plan.open_questions.len() > prior_open.len();
    let header = if reopened {
        "refinement applied -- clarification reopened"
    } else {
        "refinement applied"
    };
    print_plan_summary(&plan, style, header);
    Ok(())
}

/// `flowd plan cancel` -- offline-mode counterpart to the `plan_cancel`
/// MCP tool. Same daemon-liveness guard. Cancellation of a plan that
/// is already terminal (Completed / Failed / Cancelled) is a silent
/// no-op so scripts can call this idempotently.
pub async fn cancel(paths: &FlowdPaths, style: Style, plan_id_arg: String) -> Result<()> {
    ensure_daemon_offline(paths, style, "plan cancel")?;
    let plan_id = parse_plan_id(&plan_id_arg)?;

    let runner = OfflinePlanRunner::open(paths)?;
    let mut plan = runner.load_existing(plan_id).await?;

    match plan.status {
        PlanStatus::Completed | PlanStatus::Failed | PlanStatus::Cancelled => {
            println!(
                "{} plan {} already in {:?}; nothing to do",
                style.dim("noop:"),
                plan.id,
                plan.status,
            );
            return Ok(());
        }
        PlanStatus::Running => {
            // Without the daemon, no executor is holding handles to
            // abort -- the running task lives in the dead daemon's
            // address space. Refuse rather than silently flipping to
            // Cancelled and leaving the operator wondering why the
            // events log doesn't show any abort.
            bail!(
                "plan cancel: plan {} is Running but the daemon is offline; \
                 start the daemon and use the plan_cancel MCP tool, or \
                 wait for the executor task to exit before retrying",
                plan.id
            );
        }
        PlanStatus::Draft | PlanStatus::Confirmed => {
            plan.status = PlanStatus::Cancelled;
            plan.completed_at = Some(chrono::Utc::now());
        }
    }

    runner.save(&plan).await?;
    println!("{} plan {} -> Cancelled", style.green("cancelled"), plan.id);
    Ok(())
}

/// Wire shape for `--file` payloads to `flowd plan answer`. Mirrors
/// `flowd_mcp::tools::PlanAnswerEntry` (flattened `Answer` enum), but
/// duplicated here so the CLI does not pull in `flowd-mcp` types just
/// for serde.
#[derive(Debug, Deserialize)]
struct CliAnswerEntry {
    question_id: String,
    #[serde(flatten)]
    answer: Answer,
}

/// Bundle of the dependencies the offline plan commands share.
struct OfflinePlanRunner {
    store: SqlitePlanStore,
    compiler: Arc<DaemonPlanCompiler>,
}

impl OfflinePlanRunner {
    fn open(paths: &FlowdPaths) -> Result<Self> {
        let db_path = paths.db_file();
        if !db_path.exists() {
            bail!(
                "no flowd database at {}; start the daemon at least once to initialise it",
                db_path.display()
            );
        }
        let store = SqlitePlanStore::open(&db_path)
            .with_context(|| format!("open plan store at {}", db_path.display()))?;

        let cfg_path = FlowdConfig::default_path(&paths.home);
        let config = FlowdConfig::load(&cfg_path)
            .with_context(|| format!("load flowd config from {}", cfg_path.display()))?;
        let compiler = DaemonPlanCompiler::from_selection(config.plan.compiler, &config.plan.llm)
            .map_err(|e| anyhow!("instantiate plan compiler: {e}"))?;

        Ok(Self {
            store,
            compiler: Arc::new(compiler),
        })
    }

    async fn load_existing(&self, plan_id: Uuid) -> Result<Plan> {
        self.store
            .load_plan(plan_id)
            .await
            .with_context(|| format!("load plan {plan_id}"))?
            .ok_or_else(|| anyhow!("plan {plan_id} not found"))
    }

    async fn load_draft(&self, plan_id: Uuid) -> Result<Plan> {
        let plan = self.load_existing(plan_id).await?;
        if plan.status != PlanStatus::Draft {
            bail!(
                "plan {plan_id} is not in Draft state (currently {:?}); \
                 only Draft plans accept clarification updates",
                plan.status
            );
        }
        Ok(plan)
    }

    async fn save(&self, plan: &Plan) -> Result<()> {
        self.store
            .save_plan(plan)
            .await
            .with_context(|| format!("save plan {}", plan.id))
    }
}

/// Refuse to operate on shared `SQLite` state when the daemon is alive.
/// The check is best-effort -- the PID file could go stale between
/// the probe and the first write -- but covers the common operator
/// mistake of running both surfaces against the same `$FLOWD_HOME`.
fn ensure_daemon_offline(paths: &FlowdPaths, style: Style, op: &str) -> Result<()> {
    let pid_path = paths.pid_file();
    let Some(pid) = daemon::read_pid(&pid_path)? else {
        return Ok(());
    };
    if !daemon::is_alive(pid) {
        return Ok(());
    }
    eprintln!(
        "{} the flowd daemon is running (pid {pid}, {pidfile}).",
        style.yellow(&format!("{op}: refused")),
        pidfile = pid_path.display(),
    );
    eprintln!("       offline `flowd plan` mutations would race the daemon's in-memory executor.");
    eprintln!(
        "       use the corresponding MCP tool (`{}`) instead, or run `flowd stop` first.",
        op.replace(' ', "_"),
    );
    bail!("{op}: daemon is alive");
}

fn parse_plan_id(raw: &str) -> Result<Uuid> {
    Uuid::parse_str(raw.trim()).with_context(|| format!("parse plan id `{raw}` as UUID"))
}

/// Read text from a path; treat `-` as stdin so callers can pipe.
fn read_text_input(path: &std::path::Path) -> Result<String> {
    if path == std::path::Path::new("-") {
        let mut buf = String::new();
        std::io::stdin()
            .read_to_string(&mut buf)
            .context("read from stdin")?;
        return Ok(buf);
    }
    std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))
}

fn print_plan_summary(plan: &Plan, style: Style, header: &str) {
    print!(
        "{}",
        banner(&format!("plan: {} -- {}", plan.name, header), style)
    );
    println!("  id:               {}", style.dim(&plan.id.to_string()));
    println!("  status:           {:?}", plan.status);
    println!("  project:          {}", plan.project);
    println!("  steps:            {}", plan.steps.len());
    println!("  open_questions:   {}", plan.open_questions.len());
    println!("  decisions:        {}", plan.decisions.len());
    println!("  definition_dirty: {}", plan.definition_dirty);

    if !plan.open_questions.is_empty() {
        println!("\n{}", style.bold("open questions:"));
        for q in &plan.open_questions {
            println!("  {} -- {}", style.cyan(&q.id), q.prompt);
        }
    }
    if !plan.decisions.is_empty() {
        println!("\n{}", style.bold("decisions:"));
        for d in &plan.decisions {
            let tag = if d.auto { " (auto)" } else { "" };
            println!(
                "  {} = {}{tag}",
                style.cyan(&d.question_id),
                d.chosen_option_id,
            );
        }
    }
    if !plan.steps.is_empty() && plan.open_questions.is_empty() {
        match build_preview(plan) {
            Ok(preview) => render_layers(&preview.execution_order, style),
            Err(e) => println!(
                "\n{} could not build preview: {e}",
                style.yellow("warning:")
            ),
        }
    }
}

fn prompt_yes_no(prompt: &str) -> Result<bool> {
    let stdout = std::io::stdout();
    let mut out = stdout.lock();
    out.write_all(prompt.as_bytes())?;
    out.flush()?;

    let stdin = std::io::stdin();
    let mut line = String::new();
    stdin.lock().read_line(&mut line)?;
    let answer = line.trim().to_ascii_lowercase();
    Ok(matches!(answer.as_str(), "y" | "yes"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn renders_empty_layers_without_panicking() {
        render_layers(&[], Style::plain());
    }

    #[test]
    fn renders_empty_graph_without_panicking() {
        render_dependencies(&HashMap::new(), Style::plain());
    }

    #[test]
    fn normalise_kinds_drops_empty_and_dedupes() {
        let out = normalise_kinds(vec![
            "step_failed".into(),
            " ".into(),
            "step_failed".into(),
            "step_completed".into(),
            String::new(),
        ]);
        assert_eq!(out, vec!["step_failed", "step_completed"]);
    }

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate_one_line("hello", 10), "hello");
    }

    #[test]
    fn truncate_collapses_whitespace_then_cuts() {
        let out = truncate_one_line("a\nb   c", 32);
        assert_eq!(out, "a b c");
    }

    #[test]
    fn truncate_appends_ellipsis_when_over_limit() {
        let out = truncate_one_line(&"x".repeat(200), 10);
        assert!(out.ends_with('…'));
        assert!(out.chars().count() <= 10);
    }

    #[test]
    fn parse_plan_id_trims_whitespace() {
        let id = Uuid::new_v4();
        let raw = format!("  {id}  \n");
        let parsed = parse_plan_id(&raw).unwrap();
        assert_eq!(parsed, id);
    }

    #[test]
    fn parse_plan_id_errors_on_garbage() {
        let err = parse_plan_id("not-a-uuid").unwrap_err();
        assert!(format!("{err:#}").contains("UUID"), "{err:#}");
    }

    #[test]
    fn read_text_input_reads_file() {
        let tmp =
            std::env::temp_dir().join(format!("flowd-plan-input-{}.txt", uuid::Uuid::new_v4()));
        std::fs::write(&tmp, "hello world\n").unwrap();
        let got = read_text_input(&tmp).unwrap();
        assert_eq!(got, "hello world\n");
        std::fs::remove_file(&tmp).ok();
    }

    #[test]
    fn ensure_daemon_offline_passes_when_no_pid_file() {
        let dir = std::env::temp_dir().join(format!("flowd-plan-offline-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        ensure_daemon_offline(&paths, Style::plain(), "plan answer").unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_daemon_offline_passes_when_pid_is_stale() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-stale-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        // PID 2^31-2 is essentially never live; the helper should treat
        // it as a stale pid file and let the command proceed.
        std::fs::write(dir.join("flowd.pid"), "2147483646\n").unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        ensure_daemon_offline(&paths, Style::plain(), "plan refine").unwrap();
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn ensure_daemon_offline_refuses_when_pid_is_self() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-live-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("flowd.pid"), format!("{}\n", std::process::id())).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        let err = ensure_daemon_offline(&paths, Style::plain(), "plan cancel").unwrap_err();
        assert!(format!("{err:#}").contains("daemon is alive"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn offline_plan_runner_open_errors_when_db_absent() {
        let dir =
            std::env::temp_dir().join(format!("flowd-plan-offline-nodb-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let paths = FlowdPaths::with_home(dir.clone());
        let Err(err) = OfflinePlanRunner::open(&paths) else {
            panic!("expected error when no flowd.db is present");
        };
        assert!(format!("{err:#}").contains("no flowd database"), "{err:#}");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn cli_answer_entry_parses_choose_payload() {
        let raw = r#"[{"question_id": "q1", "kind": "choose", "option_id": "opt-a"}]"#;
        let parsed: Vec<CliAnswerEntry> = serde_json::from_str(raw).unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].question_id, "q1");
        assert!(matches!(
            parsed[0].answer,
            Answer::Choose { ref option_id } if option_id == "opt-a"
        ));
    }
}
