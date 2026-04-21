//! End-to-end MCP server tests.
//!
//! Drives the real transport: we feed line-delimited JSON-RPC requests
//! through `McpServer::run` and assert the emitted responses. Uses an
//! in-crate stub handler so the test doesn't need to compose `SQLite` /
//! `Qdrant` / ONNX backends.

use std::sync::Arc;

use flowd_core::error::Result;
use flowd_mcp::protocol::{JSON_RPC_VERSION, JsonRpcResponse};
use flowd_mcp::tools::{
    MemoryContextParams, MemorySearchParams, MemoryStoreParams, PlanAnswerParams, PlanCancelParams,
    PlanConfirmParams, PlanCreateParams, PlanRefineParams, PlanResumeParams, PlanStatusParams,
    RulesCheckParams, RulesListParams,
};
use flowd_mcp::{McpHandlers, McpServer, McpServerConfig};
use serde_json::{Value, json};

/// Handler stub that echoes the method name in its payload so each response
/// is individually identifiable.
struct EchoHandlers;

impl McpHandlers for EchoHandlers {
    async fn memory_store(&self, _: MemoryStoreParams) -> Result<Value> {
        Ok(json!({"ok": "memory_store"}))
    }
    async fn memory_search(&self, _: MemorySearchParams) -> Result<Value> {
        Ok(json!({"ok": "memory_search"}))
    }
    async fn memory_context(&self, _: MemoryContextParams) -> Result<Value> {
        Ok(json!({"ok": "memory_context"}))
    }
    async fn plan_create(&self, _: PlanCreateParams) -> Result<Value> {
        Ok(json!({"ok": "plan_create"}))
    }
    async fn plan_answer(&self, _: PlanAnswerParams) -> Result<Value> {
        Ok(json!({"ok": "plan_answer"}))
    }
    async fn plan_refine(&self, _: PlanRefineParams) -> Result<Value> {
        Ok(json!({"ok": "plan_refine"}))
    }
    async fn plan_confirm(&self, _: PlanConfirmParams) -> Result<Value> {
        Ok(json!({"ok": "plan_confirm"}))
    }
    async fn plan_cancel(&self, _: PlanCancelParams) -> Result<Value> {
        Ok(json!({"ok": "plan_cancel"}))
    }
    async fn plan_status(&self, _: PlanStatusParams) -> Result<Value> {
        Ok(json!({"ok": "plan_status"}))
    }
    async fn plan_resume(&self, _: PlanResumeParams) -> Result<Value> {
        Ok(json!({"ok": "plan_resume"}))
    }
    async fn rules_check(&self, _: RulesCheckParams) -> Result<Value> {
        Ok(json!({"ok": "rules_check"}))
    }
    async fn rules_list(&self, _: RulesListParams) -> Result<Value> {
        Ok(json!({"ok": "rules_list"}))
    }
}

fn parse_responses(raw: &[u8]) -> Vec<JsonRpcResponse> {
    String::from_utf8_lossy(raw)
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            serde_json::from_str::<JsonRpcResponse>(l)
                .unwrap_or_else(|e| panic!("bad line `{l}`: {e}"))
        })
        .collect()
}

#[tokio::test]
async fn full_mcp_session_roundtrip() {
    let server = McpServer::new(Arc::new(EchoHandlers), McpServerConfig::default());

    // Simulated client session:
    // 1. initialize
    // 2. notifications/initialized (no response)
    // 3. tools/list
    // 4. tools/call rules_check
    // 5. tools/call memory_store with bad params
    let input = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
        r#"{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"rules_check","arguments":{"tool":"shell_exec"}}}"#,
        r#"{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"memory_store","arguments":{"project":"p"}}}"#,
        "",
    ]
    .join("\n");

    let mut output = Vec::new();
    server.run(input.as_bytes(), &mut output).await.unwrap();

    let responses = parse_responses(&output);
    // No response for the notification, so 4 responses total.
    assert_eq!(
        responses.len(),
        4,
        "expected 4 responses, got {}",
        responses.len()
    );

    // initialize
    assert_eq!(responses[0].jsonrpc, JSON_RPC_VERSION);
    assert_eq!(responses[0].id, json!(1));
    assert!(
        responses[0]
            .result
            .as_ref()
            .unwrap()
            .get("protocolVersion")
            .is_some()
    );

    // tools/list has 12 tools (memory_*, plan_create/answer/refine/confirm/cancel/status/resume, rules_*).
    let tools = responses[1].result.as_ref().unwrap()["tools"]
        .as_array()
        .unwrap();
    assert_eq!(tools.len(), 12);

    // tools/call rules_check -> success envelope
    let ok_result = responses[2].result.as_ref().unwrap();
    assert_eq!(ok_result["isError"], false);
    assert!(
        ok_result["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("rules_check")
    );

    // tools/call with missing session_id -> -32602 invalid params
    let err = responses[3].error.as_ref().unwrap();
    assert_eq!(err.code, -32_602);
}

#[tokio::test]
async fn parse_error_returns_null_id_response() {
    let server = McpServer::new(Arc::new(EchoHandlers), McpServerConfig::default());
    let input = b"this is not json\n";
    let mut output = Vec::new();
    server.run(&input[..], &mut output).await.unwrap();

    let responses = parse_responses(&output);
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].id, Value::Null);
    assert_eq!(responses[0].error.as_ref().unwrap().code, -32_700);
}
