//! JSON-RPC 2.0 envelope types used by the MCP transport.
//!
//! MCP is JSON-RPC 2.0 over a transport that frames one message per line of
//! stdio. This module is transport-agnostic; it only knows how to serialise
//! and deserialise the envelope. Method dispatch lives in [`crate::server`].
//!
//! Request / response shapes are intentionally narrow: the MCP spec allows
//! batched requests and streaming, but flowd speaks a single-message
//! sequential dialect that is easier to reason about and sufficient for the
//! tools we export.

use serde::{Deserialize, Serialize};
use serde_json::Value;

/// JSON-RPC 2.0 version tag.
pub const JSON_RPC_VERSION: &str = "2.0";

/// Incoming message envelope. A message is either a request (has `id`) or a
/// notification (no `id`, no response expected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcRequest {
    pub jsonrpc: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<Value>,
    pub method: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub params: Option<Value>,
}

impl JsonRpcRequest {
    #[must_use]
    pub fn is_notification(&self) -> bool {
        self.id.is_none()
    }
}

/// Outgoing response envelope. Exactly one of `result` or `error` is present.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<JsonRpcError>,
}

impl JsonRpcResponse {
    #[must_use]
    pub fn success(id: Value, result: Value) -> Self {
        Self {
            jsonrpc: JSON_RPC_VERSION.into(),
            id,
            result: Some(result),
            error: None,
        }
    }

    #[must_use]
    pub fn failure(id: Value, error: JsonRpcError) -> Self {
        Self {
            jsonrpc: JSON_RPC_VERSION.into(),
            id,
            result: None,
            error: Some(error),
        }
    }
}

/// Standard JSON-RPC 2.0 error object.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JsonRpcError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

impl JsonRpcError {
    #[must_use]
    pub fn new(code: i32, message: impl Into<String>) -> Self {
        Self {
            code,
            message: message.into(),
            data: None,
        }
    }

    #[must_use]
    pub fn parse_error(detail: impl Into<String>) -> Self {
        Self::new(-32_700, format!("parse error: {}", detail.into()))
    }

    #[must_use]
    pub fn invalid_request(detail: impl Into<String>) -> Self {
        Self::new(-32_600, format!("invalid request: {}", detail.into()))
    }

    #[must_use]
    pub fn method_not_found(method: &str) -> Self {
        Self::new(-32_601, format!("method not found: {method}"))
    }

    #[must_use]
    pub fn invalid_params(detail: impl Into<String>) -> Self {
        Self::new(-32_602, format!("invalid params: {}", detail.into()))
    }

    #[must_use]
    pub fn internal_error(detail: impl Into<String>) -> Self {
        Self::new(-32_603, detail)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn request_roundtrip() {
        let raw = r#"{"jsonrpc":"2.0","id":1,"method":"tools/list","params":null}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert_eq!(req.method, "tools/list");
        assert!(!req.is_notification());
    }

    #[test]
    fn notification_has_no_id() {
        let raw = r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#;
        let req: JsonRpcRequest = serde_json::from_str(raw).unwrap();
        assert!(req.is_notification());
    }

    #[test]
    fn response_serialises_without_null_fields() {
        let resp = JsonRpcResponse::success(json!(7), json!({"ok": true}));
        let text = serde_json::to_string(&resp).unwrap();
        assert!(text.contains(r#""result":{"ok":true}"#));
        assert!(!text.contains("\"error\""));
    }

    #[test]
    fn error_codes_follow_spec() {
        assert_eq!(JsonRpcError::parse_error("x").code, -32_700);
        assert_eq!(JsonRpcError::invalid_request("x").code, -32_600);
        assert_eq!(JsonRpcError::method_not_found("m").code, -32_601);
        assert_eq!(JsonRpcError::invalid_params("x").code, -32_602);
        assert_eq!(JsonRpcError::internal_error("x").code, -32_603);
    }
}
