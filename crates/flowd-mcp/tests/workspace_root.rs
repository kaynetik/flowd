//! End-to-end coverage of the workspace-root inference path on
//! `plan_create`.
//!
//! These tests stand in for "Cursor/Claude just opened workspace X and
//! issued `plan_create` against the flowd daemon" without spinning up
//! the real proxy + daemon: they construct the same `FlowdHandlers` the
//! daemon wires together, then drive `plan_create` with a `project_root`
//! hint pointing at a temporary git checkout that is *not* the flowd
//! repo. The persisted plan must record that workspace, not the test
//! binary's `current_dir()`.
//!
//! Two scenarios:
//!
//! 1. Prose-first plan -- the path Cursor/Claude actually exercises when
//!    the operator types prose into the IDE chat.
//! 2. DAG-first plan -- the older path, which still has to honour the
//!    `project_root` hint so deployment scripts that hand-write a
//!    `PlanDefinition` behave the same way.
//!
//! Negative coverage: a hint that points at a non-git directory must
//! cause the handler to fall back to the env / cwd chain, and -- if
//! every signal misses -- to surface a clear `PlanValidation` error.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tempfile::TempDir;
use uuid::Uuid;

use flowd_core::error::Result as FlowdResult;
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, VectorIndex};
use flowd_core::orchestration::{
    AgentOutput, AgentSpawnContext, AgentSpawner, InMemoryPlanExecutor, PlanStep,
};
use flowd_core::rules::InMemoryRuleEvaluator;
use flowd_core::types::Embedding;
use flowd_mcp::tools::{PlanCreateParams, PlanShowParams};
use flowd_mcp::{FlowdHandlers, McpHandlers, StubPlanCompiler};
use flowd_storage::plan_store::SqlitePlanStore;
use flowd_storage::sqlite::SqliteBackend;

// ---------- Test doubles (mirror tests/prose_first_stub.rs) ---------------

struct HashEmbedder;

impl EmbeddingProvider for HashEmbedder {
    fn embed(&self, text: &str) -> FlowdResult<Vec<f32>> {
        #[allow(clippy::cast_precision_loss)]
        let sum = text.bytes().map(u32::from).sum::<u32>() as f32;
        Ok(vec![sum, sum * 0.5, sum * 0.25, sum * 0.125])
    }
    fn embed_batch(&self, texts: &[&str]) -> FlowdResult<Vec<Vec<f32>>> {
        texts.iter().map(|t| self.embed(t)).collect()
    }
    fn dimensions(&self) -> usize {
        4
    }
}

#[derive(Default)]
struct MemVectors {
    inner: Mutex<HashMap<Uuid, String>>,
}

impl VectorIndex for MemVectors {
    async fn upsert(&self, embedding: &Embedding) -> FlowdResult<()> {
        self.inner
            .lock()
            .unwrap()
            .insert(embedding.observation_id, embedding.project.clone());
        Ok(())
    }
    async fn search(
        &self,
        _query_vector: &[f32],
        _limit: usize,
        _project_filter: Option<&str>,
    ) -> FlowdResult<Vec<(Uuid, f64)>> {
        Ok(Vec::new())
    }
    async fn delete(&self, observation_id: Uuid) -> FlowdResult<()> {
        self.inner.lock().unwrap().remove(&observation_id);
        Ok(())
    }
}

#[derive(Clone, Default)]
struct EchoSpawner;

impl AgentSpawner for EchoSpawner {
    fn spawn(
        &self,
        _ctx: AgentSpawnContext,
        step: &PlanStep,
    ) -> impl Future<Output = FlowdResult<AgentOutput>> + Send {
        let id = step.id.clone();
        async move { Ok(AgentOutput::success(format!("ran:{id}"))) }
    }
}

// ---------- Harness -------------------------------------------------------

type Handlers = FlowdHandlers<
    SqliteBackend,
    MemVectors,
    HashEmbedder,
    InMemoryPlanExecutor<EchoSpawner, SqlitePlanStore>,
    StubPlanCompiler,
    InMemoryRuleEvaluator,
>;

struct Harness {
    _tmp: TempDir,
    handlers: Arc<Handlers>,
}

fn build_harness() -> Harness {
    let tmp = tempfile::tempdir().expect("tempdir");
    let db_path = tmp.path().join("flowd.db");
    let backend = SqliteBackend::open(&db_path).expect("open sqlite");
    let plan_store = backend.plan_store();
    let memory = Arc::new(MemoryService::new(
        backend,
        MemVectors::default(),
        HashEmbedder,
    ));
    let executor = Arc::new(InMemoryPlanExecutor::with_plan_store(
        EchoSpawner,
        plan_store,
    ));
    let rules = Arc::new(InMemoryRuleEvaluator::new());
    let compiler = Arc::new(StubPlanCompiler::new());
    let handlers = Arc::new(FlowdHandlers::new(memory, executor, compiler, rules));
    Harness {
        _tmp: tmp,
        handlers,
    }
}

/// Fabricate a temporary directory that *looks* like a freshly-cloned
/// git repo from the resolver's perspective: it has a `.git` directory
/// in its root, so the parent walk stops there. We don't run real git
/// because the resolver only does a filesystem probe and we want the
/// test to be fast and offline.
fn make_workspace(tag: &str) -> TempDir {
    let tmp = tempfile::Builder::new()
        .prefix(&format!("flowd-ws-{tag}-"))
        .tempdir()
        .expect("workspace tempdir");
    std::fs::create_dir_all(tmp.path().join(".git")).expect("seed .git directory");
    tmp
}

fn canonical(p: &std::path::Path) -> String {
    std::fs::canonicalize(p).unwrap().display().to_string()
}

// ---------- Scenarios -----------------------------------------------------

/// Cursor/Claude-style flow: the MCP client (or `flowd mcp` proxy
/// speaking on its behalf) supplies `project_root` in the request so the
/// daemon -- whose own cwd is unrelated -- still records the invoking
/// workspace. The `prose` path is what real users hit.
#[tokio::test]
async fn prose_plan_records_client_supplied_workspace_as_project_root() {
    let h = build_harness();
    let workspace = make_workspace("prose-cursor");

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("# refactor-auth\n\n## extract [agent: echo]\nDo the thing.\n".into()),
            compiler_override: None,
            project_root: Some(workspace.path().display().to_string()),
        })
        .await
        .expect("plan_create accepts client-supplied project_root");

    let plan_id = created["plan_id"]
        .as_str()
        .expect("plan_id present")
        .to_owned();
    assert_eq!(created["status"], "draft", "stub compiles structured prose");

    // Round-trip through the persistence layer to confirm the resolver
    // ran inside `plan_create` -- not just the in-memory return value.
    let plan: Value = h
        .handlers
        .plan_show(PlanShowParams {
            plan_id: plan_id.clone(),
        })
        .await
        .expect("plan_show after create");

    assert_eq!(
        plan["project_root"].as_str(),
        Some(canonical(workspace.path()).as_str()),
        "project_root must be the canonical client-supplied workspace, not the daemon's cwd"
    );
}

/// Same coverage for the legacy DAG-first path: deployment scripts that
/// hand-construct a `definition` payload still benefit from the proxy
/// hint when the operator hasn't pinned `project_root` in the YAML.
#[tokio::test]
async fn definition_plan_records_client_supplied_workspace_as_project_root() {
    let h = build_harness();
    let workspace = make_workspace("def-cursor");

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: Some(json!({
                "name": "deploy",
                "steps": [{ "id": "a", "agent_type": "echo", "prompt": "go" }]
            })),
            prose: None,
            compiler_override: None,
            project_root: Some(workspace.path().display().to_string()),
        })
        .await
        .expect("plan_create(definition) accepts project_root hint");

    let plan_id = created["plan_id"].as_str().unwrap().to_owned();
    let plan: Value = h
        .handlers
        .plan_show(PlanShowParams { plan_id })
        .await
        .expect("plan_show after definition create");

    assert_eq!(
        plan["project_root"].as_str(),
        Some(canonical(workspace.path()).as_str()),
        "DAG-first path must honour the same hint as the prose path"
    );
}

/// `definition.project_root` (operator-authored) wins over the request
/// hint -- the on-disk file is the source of truth for plans that ship
/// with one. This protects deployment scripts that explicitly target a
/// different workspace from being silently overridden by the IDE that
/// happened to launch the proxy.
#[tokio::test]
async fn explicit_definition_project_root_overrides_client_hint() {
    let h = build_harness();
    let definition_workspace = make_workspace("def-explicit");
    let proxy_workspace = make_workspace("proxy-other");

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: Some(json!({
                "name": "deploy",
                "project_root": definition_workspace.path().display().to_string(),
                "steps": [{ "id": "a", "agent_type": "echo", "prompt": "go" }]
            })),
            prose: None,
            compiler_override: None,
            project_root: Some(proxy_workspace.path().display().to_string()),
        })
        .await
        .expect("explicit definition.project_root wins");

    let plan_id = created["plan_id"].as_str().unwrap().to_owned();
    let plan: Value = h
        .handlers
        .plan_show(PlanShowParams { plan_id })
        .await
        .expect("plan_show");

    // Note: the on-disk path is stored verbatim (no canonicalisation),
    // matching the existing behaviour for explicit project_root values
    // in PlanDefinition.
    assert_eq!(
        plan["project_root"].as_str(),
        Some(definition_workspace.path().display().to_string().as_str()),
        "definition.project_root must override the client hint"
    );
}

/// Negative path: when the hint points at a directory that is not a
/// git checkout the resolver's git gate rejects it, and -- absent any
/// other signal that resolves -- we want a structured `PlanValidation`
/// error rather than a silently-mis-rooted plan.
///
/// We can't assert the *full* failure path here without isolating the
/// process's `current_dir()` (which lives inside the flowd repo and so
/// would always succeed). Instead, rely on the resolver's unit tests
/// (in `flowd_core::orchestration`) for the all-candidates-fail error
/// shape, and use this test to confirm a non-git hint is gracefully
/// skipped rather than poisoning the persisted record.
#[tokio::test]
async fn non_git_client_hint_is_rejected_and_falls_through_to_cwd() {
    let h = build_harness();
    let bare = tempfile::Builder::new()
        .prefix("flowd-ws-no-git-")
        .tempdir()
        .expect("non-git tempdir");

    let created = h
        .handlers
        .plan_create(PlanCreateParams {
            project: "rnd".into(),
            definition: None,
            prose: Some("# refactor-auth\n\n## extract [agent: echo]\nDo the thing.\n".into()),
            compiler_override: None,
            project_root: Some(bare.path().display().to_string()),
        })
        .await
        .expect("non-git hint should fall through to cwd, not error");

    let plan_id = created["plan_id"].as_str().unwrap().to_owned();
    let plan: Value = h
        .handlers
        .plan_show(PlanShowParams { plan_id })
        .await
        .expect("plan_show");

    let stored = plan["project_root"].as_str().expect("project_root set");
    let bare_canonical = canonical(bare.path());
    assert_ne!(
        stored, bare_canonical,
        "non-git hint must NOT be persisted; resolver should fall through"
    );
    // The fallback is the test binary's cwd, which is the flowd repo
    // -- a valid git checkout. Confirming the value is non-empty and
    // distinct from the rejected hint is enough; we don't pin it to
    // the cwd literal because tests can run from a worktree path.
    assert!(!stored.trim().is_empty());
}
