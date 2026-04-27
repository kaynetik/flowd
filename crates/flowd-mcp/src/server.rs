//! MCP server: dispatch + stdio transport.
//!
//! Two independent halves live here so each is testable alone:
//!
//! * [`dispatch`] -- given a parsed [`JsonRpcRequest`] and an [`McpHandlers`]
//!   implementation, produce the result payload (or JSON-RPC error).
//!   Pure logic, no I/O.
//! * [`McpServer::run`] -- line-delimited JSON over stdin/stdout, calling
//!   `dispatch` for each inbound message. One message per line; responses
//!   are emitted in the order their requests were received.
//!
//! Notifications (`id` absent) are processed but never produce a response,
//! as required by JSON-RPC 2.0.

use std::sync::Arc;

use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::handlers::McpHandlers;
use crate::protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
use crate::tools::{
    MemoryContextParams, MemorySearchParams, MemoryStoreParams, PlanAnswerParams, PlanCancelParams,
    PlanConfirmParams, PlanCreateParams, PlanIntegrateParams, PlanListParams, PlanRecentParams,
    PlanRefineParams, PlanResumeParams, PlanShowParams, PlanStatusParams, RulesCheckParams,
    RulesListParams, ToolResult, all_tool_schemas,
};

/// MCP server configuration.
#[derive(Debug, Clone)]
pub struct McpServerConfig {
    /// Protocol version advertised in the `initialize` response.
    pub protocol_version: String,
    /// Server name advertised in the `initialize` response.
    pub server_name: String,
    /// Server version advertised in the `initialize` response.
    pub server_version: String,
}

impl Default for McpServerConfig {
    fn default() -> Self {
        Self {
            protocol_version: "2024-11-05".into(),
            server_name: "flowd".into(),
            server_version: env!("CARGO_PKG_VERSION").to_string(),
        }
    }
}

/// Stateful MCP server. One instance per process; transport runs on the
/// task it was spawned from.
pub struct McpServer<H: McpHandlers> {
    handlers: Arc<H>,
    config: McpServerConfig,
}

impl<H: McpHandlers + 'static> McpServer<H> {
    pub fn new(handlers: Arc<H>, config: McpServerConfig) -> Self {
        Self { handlers, config }
    }

    /// Run the server on the supplied async reader / writer. Most callers
    /// pass `tokio::io::stdin()` / `tokio::io::stdout()`.
    ///
    /// Returns when the reader yields EOF.
    ///
    /// # Errors
    /// Propagates I/O errors from the transport.
    pub async fn run<R, W>(&self, reader: R, mut writer: W) -> std::io::Result<()>
    where
        R: tokio::io::AsyncRead + Unpin,
        W: tokio::io::AsyncWrite + Unpin,
    {
        let mut lines = BufReader::new(reader).lines();
        while let Some(line) = lines.next_line().await? {
            if line.trim().is_empty() {
                continue;
            }

            let response = match serde_json::from_str::<JsonRpcRequest>(&line) {
                Ok(req) => {
                    if req.is_notification() {
                        handle_notification(&req);
                        continue;
                    }
                    let id = req.id.clone().unwrap_or(Value::Null);
                    match dispatch(&*self.handlers, &self.config, req).await {
                        Ok(result) => JsonRpcResponse::success(id, result),
                        Err(err) => JsonRpcResponse::failure(id, err),
                    }
                }
                Err(e) => {
                    JsonRpcResponse::failure(Value::Null, JsonRpcError::parse_error(e.to_string()))
                }
            };

            write_message(&mut writer, &response).await?;
        }
        Ok(())
    }

    /// Run the server on process stdio. Convenience wrapper over
    /// [`McpServer::run`].
    ///
    /// # Errors
    /// Propagates I/O errors from stdin / stdout.
    pub async fn run_stdio(&self) -> std::io::Result<()> {
        self.run(tokio::io::stdin(), tokio::io::stdout()).await
    }
}

async fn write_message<W>(writer: &mut W, response: &JsonRpcResponse) -> std::io::Result<()>
where
    W: tokio::io::AsyncWrite + Unpin,
{
    let text = serde_json::to_string(response).map_err(std::io::Error::other)?;
    writer.write_all(text.as_bytes()).await?;
    writer.write_all(b"\n").await?;
    writer.flush().await
}

fn handle_notification(req: &JsonRpcRequest) {
    tracing::debug!(method = %req.method, "mcp notification");
}

/// Pure dispatch: method + params → result payload or JSON-RPC error.
///
/// Exposed for testing so the protocol layer can be driven without stdio.
///
/// # Errors
/// Returns a JSON-RPC error for unknown methods, malformed params, or
/// handler failures.
pub async fn dispatch<H: McpHandlers>(
    handlers: &H,
    config: &McpServerConfig,
    req: JsonRpcRequest,
) -> std::result::Result<Value, JsonRpcError> {
    let params = req.params.unwrap_or(Value::Null);

    match req.method.as_str() {
        "initialize" => Ok(json!({
            "protocolVersion": config.protocol_version,
            "serverInfo": {
                "name": config.server_name,
                "version": config.server_version,
            },
            "capabilities": {
                "tools": { "listChanged": false }
            },
        })),

        "ping" => Ok(json!({})),

        "tools/list" => Ok(json!({ "tools": all_tool_schemas() })),

        "tools/call" => dispatch_tool_call(handlers, params).await,

        other => Err(JsonRpcError::method_not_found(other)),
    }
}

async fn dispatch_tool_call<H: McpHandlers>(
    handlers: &H,
    params: Value,
) -> std::result::Result<Value, JsonRpcError> {
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or_else(|| JsonRpcError::invalid_params("missing `name`"))?
        .to_owned();
    let arguments = params
        .get("arguments")
        .cloned()
        .unwrap_or(Value::Object(serde_json::Map::new()));

    let result: std::result::Result<Value, flowd_core::error::FlowdError> = match name.as_str() {
        "memory_store" => match parse_params::<MemoryStoreParams>(arguments) {
            Ok(p) => handlers.memory_store(p).await,
            Err(e) => return Err(e),
        },
        "memory_search" => match parse_params::<MemorySearchParams>(arguments) {
            Ok(p) => handlers.memory_search(p).await,
            Err(e) => return Err(e),
        },
        "memory_context" => match parse_params::<MemoryContextParams>(arguments) {
            Ok(p) => handlers.memory_context(p).await,
            Err(e) => return Err(e),
        },
        "plan_create" => match parse_params::<PlanCreateParams>(arguments) {
            Ok(p) => handlers.plan_create(p).await,
            Err(e) => return Err(e),
        },
        "plan_answer" => match parse_params::<PlanAnswerParams>(arguments) {
            Ok(p) => handlers.plan_answer(p).await,
            Err(e) => return Err(e),
        },
        "plan_refine" => match parse_params::<PlanRefineParams>(arguments) {
            Ok(p) => handlers.plan_refine(p).await,
            Err(e) => return Err(e),
        },
        "plan_confirm" => match parse_params::<PlanConfirmParams>(arguments) {
            Ok(p) => handlers.plan_confirm(p).await,
            Err(e) => return Err(e),
        },
        "plan_cancel" => match parse_params::<PlanCancelParams>(arguments) {
            Ok(p) => handlers.plan_cancel(p).await,
            Err(e) => return Err(e),
        },
        "plan_status" => match parse_params::<PlanStatusParams>(arguments) {
            Ok(p) => handlers.plan_status(p).await,
            Err(e) => return Err(e),
        },
        "plan_resume" => match parse_params::<PlanResumeParams>(arguments) {
            Ok(p) => handlers.plan_resume(p).await,
            Err(e) => return Err(e),
        },
        "plan_list" => match parse_params::<PlanListParams>(arguments) {
            Ok(p) => handlers.plan_list(p).await,
            Err(e) => return Err(e),
        },
        "plan_show" => match parse_params::<PlanShowParams>(arguments) {
            Ok(p) => handlers.plan_show(p).await,
            Err(e) => return Err(e),
        },
        "plan_recent" => match parse_params::<PlanRecentParams>(arguments) {
            Ok(p) => handlers.plan_recent(p).await,
            Err(e) => return Err(e),
        },
        "plan_integrate" => match parse_params::<PlanIntegrateParams>(arguments) {
            Ok(p) => handlers.plan_integrate(p).await,
            Err(e) => return Err(e),
        },
        "rules_check" => match parse_params::<RulesCheckParams>(arguments) {
            Ok(p) => handlers.rules_check(p).await,
            Err(e) => return Err(e),
        },
        "rules_list" => match parse_params::<RulesListParams>(arguments) {
            Ok(p) => handlers.rules_list(p).await,
            Err(e) => return Err(e),
        },
        other => {
            return Err(JsonRpcError::invalid_params(format!(
                "unknown tool `{other}`"
            )));
        }
    };

    // Map handler outcomes into the MCP `ToolResult` envelope; handler
    // errors surface as `isError: true` payloads rather than JSON-RPC
    // errors, so the protocol frame always carries a structured result.
    let envelope = match result {
        Ok(payload) => ToolResult::ok(&payload)
            .map_err(|e| JsonRpcError::internal_error(format!("serialise tool result: {e}")))?,
        Err(e) => ToolResult::error(e.to_string()),
    };
    serde_json::to_value(envelope)
        .map_err(|e| JsonRpcError::internal_error(format!("serialise tool envelope: {e}")))
}

fn parse_params<T: serde::de::DeserializeOwned>(value: Value) -> Result<T, JsonRpcError> {
    serde_json::from_value(value).map_err(|e| JsonRpcError::invalid_params(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::JSON_RPC_VERSION;
    use serde_json::json;

    /// No-op handlers that fail every call; lets us test dispatch envelopes
    /// without wiring flowd-core backends.
    struct StubHandlers;

    impl McpHandlers for StubHandlers {
        async fn memory_store(
            &self,
            _: crate::tools::MemoryStoreParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"id": "00000000-0000-0000-0000-000000000000"}))
        }
        async fn memory_search(
            &self,
            _: crate::tools::MemorySearchParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!([]))
        }
        async fn memory_context(
            &self,
            _: crate::tools::MemoryContextParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"observations": [], "rules": []}))
        }
        async fn plan_create(
            &self,
            _: crate::tools::PlanCreateParams,
        ) -> flowd_core::error::Result<Value> {
            Err(flowd_core::error::FlowdError::PlanValidation("stub".into()))
        }
        async fn plan_answer(
            &self,
            _: crate::tools::PlanAnswerParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "draft", "open_questions": []}))
        }
        async fn plan_refine(
            &self,
            _: crate::tools::PlanRefineParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "draft", "open_questions": []}))
        }
        async fn plan_confirm(
            &self,
            _: crate::tools::PlanConfirmParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "running"}))
        }
        async fn plan_cancel(
            &self,
            _: crate::tools::PlanCancelParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "cancelled"}))
        }
        async fn plan_status(
            &self,
            _: crate::tools::PlanStatusParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "running"}))
        }
        async fn plan_resume(
            &self,
            _: crate::tools::PlanResumeParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "running"}))
        }
        async fn plan_list(
            &self,
            _: crate::tools::PlanListParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"plans": []}))
        }
        async fn plan_show(
            &self,
            _: crate::tools::PlanShowParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"status": "draft"}))
        }
        async fn plan_recent(
            &self,
            _: crate::tools::PlanRecentParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"plans": []}))
        }
        async fn plan_integrate(
            &self,
            _: crate::tools::PlanIntegrateParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"kind": "dry_run"}))
        }
        async fn rules_check(
            &self,
            _: crate::tools::RulesCheckParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"allowed": true, "violations": []}))
        }
        async fn rules_list(
            &self,
            _: crate::tools::RulesListParams,
        ) -> flowd_core::error::Result<Value> {
            Ok(json!({"rules": []}))
        }
    }

    fn req(id: i32, method: &str, params: Value) -> JsonRpcRequest {
        JsonRpcRequest {
            jsonrpc: JSON_RPC_VERSION.into(),
            id: Some(json!(id)),
            method: method.into(),
            params: Some(params),
        }
    }

    #[tokio::test]
    async fn initialize_returns_capabilities() {
        let cfg = McpServerConfig::default();
        let result = dispatch(&StubHandlers, &cfg, req(1, "initialize", Value::Null))
            .await
            .unwrap();
        assert_eq!(result["protocolVersion"], cfg.protocol_version);
        assert_eq!(result["serverInfo"]["name"], "flowd");
        assert!(result["capabilities"]["tools"].is_object());
    }

    #[tokio::test]
    async fn tools_list_enumerates_all_tools() {
        let result = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(2, "tools/list", Value::Null),
        )
        .await
        .unwrap();
        let tools = result["tools"].as_array().unwrap();
        assert_eq!(tools.len(), 16);
    }

    #[tokio::test]
    async fn unknown_method_returns_method_not_found() {
        let err = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(3, "frobnicate", Value::Null),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, -32_601);
    }

    #[tokio::test]
    async fn tools_call_wraps_success_in_tool_result() {
        let params = json!({
            "name": "rules_check",
            "arguments": { "tool": "shell_exec" }
        });
        let result = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(4, "tools/call", params),
        )
        .await
        .unwrap();
        assert_eq!(result["isError"], false);
        let text = result["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("\"allowed\": true"));
    }

    #[tokio::test]
    async fn tools_call_maps_handler_error_to_is_error() {
        let params = json!({
            "name": "plan_create",
            "arguments": { "project": "p", "definition": {"name": "x", "steps": []} }
        });
        let result = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(5, "tools/call", params),
        )
        .await
        .unwrap();
        assert_eq!(result["isError"], true);
        assert!(
            result["content"][0]["text"]
                .as_str()
                .unwrap()
                .contains("stub")
        );
    }

    #[tokio::test]
    async fn tools_call_rejects_unknown_tool() {
        let params = json!({"name": "nope", "arguments": {}});
        let err = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(6, "tools/call", params),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, -32_602);
    }

    #[tokio::test]
    async fn tools_call_rejects_bad_params() {
        let params = json!({"name": "memory_store", "arguments": {"project": "p"}});
        let err = dispatch(
            &StubHandlers,
            &McpServerConfig::default(),
            req(7, "tools/call", params),
        )
        .await
        .unwrap_err();
        assert_eq!(err.code, -32_602);
    }
}
