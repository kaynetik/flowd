//! End-to-end MCP integration test.
//!
//! Exercises `FlowdHandlers` wired to:
//!   * real `SqliteBackend` (temp file, migrations run)
//!   * in-crate stub `VectorIndex` (`HashMap`)
//!   * in-crate stub `EmbeddingProvider` (deterministic hash)
//!   * real `InMemoryRuleEvaluator` with one seeded rule
//!   * real `InMemoryPlanExecutor` with a deterministic `EchoSpawner`
//!
//! Then drives the real `McpServer` over an in-memory byte stream to assert
//! the full agent-facing contract: `initialize` -> `tools/list` ->
//! `memory_store` -> `memory_search` -> `plan_create` -> `plan_confirm` ->
//! `plan_status` -> `rules_check`.
//!
//! Rationale: `flowd-mcp/tests/integration.rs` already covers the wire
//! protocol with stub handlers. This test covers the composition path --
//! what breaks when real `SQLite` meets the real executor meets real rules.

use std::collections::HashMap;
use std::future::Future;
use std::sync::{Arc, Mutex};

use serde_json::{Value, json};
use tempfile::TempDir;
use uuid::Uuid;

use flowd_core::error::{Result as FlowdResult, RuleLevel};
use flowd_core::memory::service::MemoryService;
use flowd_core::memory::{EmbeddingProvider, VectorIndex};
use flowd_core::orchestration::{AgentOutput, AgentSpawner, InMemoryPlanExecutor, PlanStep};
use flowd_core::rules::{InMemoryRuleEvaluator, Rule, RuleEvaluator};
use flowd_core::types::Embedding;
use flowd_mcp::protocol::JsonRpcResponse;
use flowd_mcp::{FlowdHandlers, McpServer, McpServerConfig, RejectingPlanCompiler};
use flowd_storage::plan_store::SqlitePlanStore;
use flowd_storage::sqlite::SqliteBackend;

/// Full concrete handler type. Pins the generics so `McpServer<Handlers>`
/// is a single type the harness and helpers can share.
type Handlers = FlowdHandlers<
    SqliteBackend,
    MemVectors,
    HashEmbedder,
    InMemoryPlanExecutor<EchoSpawner, SqlitePlanStore>,
    RejectingPlanCompiler,
    InMemoryRuleEvaluator,
>;

// ---------- Test doubles ---------------------------------------------------

/// Deterministic low-dim embedder: each output is a 4-wide vector seeded by
/// the text's bytes. Not semantically meaningful, but stable across runs.
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

/// In-memory vector index. Stores the last `upsert`; `search` returns all
/// entries with score=0.5 so RRF still places them in the hybrid result set.
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
        limit: usize,
        project_filter: Option<&str>,
    ) -> FlowdResult<Vec<(Uuid, f64)>> {
        let guard = self.inner.lock().unwrap();
        let hits: Vec<(Uuid, f64)> = guard
            .iter()
            .filter(|(_, p)| project_filter.is_none_or(|f| *p == f))
            .take(limit)
            .map(|(id, _)| (*id, 0.5))
            .collect();
        Ok(hits)
    }
    async fn delete(&self, observation_id: Uuid) -> FlowdResult<()> {
        self.inner.lock().unwrap().remove(&observation_id);
        Ok(())
    }
}

/// Spawner that records every step it saw in execution order and returns
/// a success envelope. Cloneable: internal state lives behind an `Arc`, so
/// the test can keep a handle to the call log after moving one clone into
/// the executor.
#[derive(Clone, Default)]
struct EchoSpawner {
    calls: Arc<Mutex<Vec<String>>>,
}

impl AgentSpawner for EchoSpawner {
    fn spawn(&self, step: &PlanStep) -> impl Future<Output = FlowdResult<AgentOutput>> + Send {
        let id = step.id.clone();
        self.calls.lock().unwrap().push(id.clone());
        async move { Ok(AgentOutput::success(format!("ran:{id}"))) }
    }
}

// ---------- Harness --------------------------------------------------------

struct Harness {
    _tmp: TempDir,
    server: McpServer<Handlers>,
    spawner: EchoSpawner,
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

    let spawner = EchoSpawner::default();
    let executor = Arc::new(InMemoryPlanExecutor::with_plan_store(
        spawner.clone(),
        plan_store,
    ));

    let mut rules = InMemoryRuleEvaluator::new();
    rules
        .register_rule(Rule {
            id: "no-shell".into(),
            scope: "**".into(),
            level: RuleLevel::Deny,
            description: "forbid shell_exec during the test".into(),
            match_pattern: "^shell_exec$".into(),
        })
        .expect("register rule");
    let rules = Arc::new(rules);

    let handlers = Arc::new(FlowdHandlers::new(
        memory,
        executor,
        Arc::new(RejectingPlanCompiler::new()),
        rules,
    ));
    let server = McpServer::new(handlers, McpServerConfig::default());

    Harness {
        _tmp: tmp,
        server,
        spawner,
    }
}

fn parse_responses(raw: &[u8]) -> Vec<JsonRpcResponse> {
    String::from_utf8_lossy(raw)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<JsonRpcResponse>(l)
                .unwrap_or_else(|e| panic!("bad response line `{l}`: {e}"))
        })
        .collect()
}

fn tool_payload(resp: &JsonRpcResponse) -> Value {
    let result = resp.result.as_ref().expect("tool call result");
    assert_eq!(result["isError"], false, "tool errored: {result:?}");
    let text = result["content"][0]["text"].as_str().expect("text content");
    serde_json::from_str::<Value>(text).unwrap_or_else(|_| Value::String(text.to_owned()))
}

// ---------- The test ------------------------------------------------------

#[tokio::test]
#[allow(clippy::too_many_lines)]
async fn full_composed_session_roundtrip() {
    let harness = build_harness();

    // A known session so memory_store/memory_search line up.
    let session_id = Uuid::new_v4().to_string();

    // The plan exercises a fan-out: a,b parallel -> c depends on both.
    let plan_def = json!({
        "name": "e2e",
        "steps": [
            { "id": "a", "agent_type": "echo", "prompt": "do a" },
            { "id": "b", "agent_type": "echo", "prompt": "do b" },
            { "id": "c", "agent_type": "echo", "prompt": "do c",
              "depends_on": ["a", "b"] }
        ]
    });

    let script = [
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name": "memory_store",
            "arguments": {
                "project": "rnd",
                "session_id": session_id,
                "content": "planted observation for the e2e test"
            }
        }}),
        json!({"jsonrpc":"2.0","id":4,"method":"tools/call","params":{
            "name": "memory_search",
            "arguments": { "query": "planted", "project": "rnd", "limit": 5 }
        }}),
        json!({"jsonrpc":"2.0","id":5,"method":"tools/call","params":{
            "name": "rules_check",
            "arguments": { "tool": "shell_exec", "project": "rnd" }
        }}),
        json!({"jsonrpc":"2.0","id":6,"method":"tools/call","params":{
            "name": "plan_create",
            "arguments": { "project": "rnd", "definition": plan_def }
        }}),
    ];

    let input = script
        .iter()
        .map(ToString::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";

    let mut output = Vec::new();
    harness
        .server
        .run(input.as_bytes(), &mut output)
        .await
        .expect("run mcp");

    let responses = parse_responses(&output);
    assert_eq!(
        responses.len(),
        6,
        "expected 6 responses (one notif was dropped)"
    );

    // initialize: protocol handshake sanity
    assert!(responses[0].result.as_ref().unwrap()["protocolVersion"].is_string());

    // tools/list: we ship exactly the 9 MCP tools
    let tools = responses[1].result.as_ref().unwrap()["tools"]
        .as_array()
        .unwrap();
    assert_eq!(tools.len(), 12);

    // memory_store: returns the new UUID
    let stored = tool_payload(&responses[2]);
    assert!(Uuid::parse_str(stored["id"].as_str().unwrap()).is_ok());

    // memory_search: the planted observation is recoverable
    let hits = tool_payload(&responses[3]);
    let arr = hits.as_array().expect("search returns array");
    assert!(!arr.is_empty(), "search returned no hits: {hits:?}");
    let contents: Vec<&str> = arr
        .iter()
        .map(|r| r["observation"]["content"].as_str().unwrap_or_default())
        .collect();
    assert!(
        contents.iter().any(|c| c.contains("planted observation")),
        "expected content not found: {contents:?}"
    );

    // rules_check: shell_exec is denied
    let gate = tool_payload(&responses[4]);
    assert_eq!(
        gate["allowed"], false,
        "rule should deny shell_exec: {gate:?}"
    );
    assert!(
        gate["violations"][0]["rule_id"]
            .as_str()
            .unwrap_or("")
            .contains("no-shell")
    );

    // plan_create: returns a plan_id and a preview with 2 layers
    let created = tool_payload(&responses[5]);
    let plan_id = created["plan_id"].as_str().expect("plan_id string");
    let layers = created["preview"]["execution_order"].as_array().unwrap();
    assert_eq!(layers.len(), 2);
    assert_eq!(layers[0].as_array().unwrap().len(), 2);
    assert_eq!(layers[1].as_array().unwrap()[0], "c");

    // Now confirm + drive the plan through the same server in a follow-up
    // session. McpServer is stateless across `run` calls, so we reuse it.
    let step_two = [
        json!({"jsonrpc":"2.0","id":7,"method":"tools/call","params":{
            "name": "plan_confirm",
            "arguments": { "plan_id": plan_id }
        }}),
    ];
    let input2 = step_two
        .iter()
        .map(Value::to_string)
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    let mut output2 = Vec::new();
    harness
        .server
        .run(input2.as_bytes(), &mut output2)
        .await
        .expect("confirm");
    let confirmed = tool_payload(&parse_responses(&output2)[0]);
    assert_eq!(confirmed["status"], "running");

    // Poll plan_status until the plan reaches a terminal state.
    let terminal = poll_plan_until_done(&harness.server, plan_id).await;
    assert_eq!(
        terminal["status"], "completed",
        "plan did not complete: {terminal:?}"
    );

    let calls = harness.spawner.calls.lock().unwrap().clone();
    assert_eq!(
        calls.len(),
        3,
        "spawner saw {} calls: {:?}",
        calls.len(),
        calls
    );
    assert!(calls[0] == "a" || calls[0] == "b");
    assert!(calls[1] == "a" || calls[1] == "b");
    assert_ne!(calls[0], calls[1]);
    assert_eq!(calls[2], "c");
}

async fn poll_plan_until_done(server: &McpServer<Handlers>, plan_id: &str) -> Value {
    for attempt in 0..40u32 {
        let req = json!({"jsonrpc":"2.0","id": 100 + attempt, "method":"tools/call",
            "params":{ "name": "plan_status", "arguments": { "plan_id": plan_id } }});
        let input = format!("{req}\n");
        let mut output = Vec::new();
        server
            .run(input.as_bytes(), &mut output)
            .await
            .expect("status");
        let payload = tool_payload(&parse_responses(&output)[0]);
        let status = payload["status"].as_str().unwrap_or("");
        if matches!(status, "completed" | "failed" | "cancelled") {
            return payload;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    panic!("plan did not reach terminal state within timeout");
}
