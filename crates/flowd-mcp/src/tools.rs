//! Tool parameter structs, schemas, and the MCP tool-result envelope.
//!
//! Each tool exposed by the MCP server has a typed parameter struct here,
//! which doubles as its JSON Schema source of truth (via hand-written
//! schemas rather than a derive macro, to stay dep-light and to keep full
//! control over the wire format). The dispatch layer in [`crate::server`]
//! deserialises the `arguments` field of a `tools/call` into one of these
//! structs before invoking the corresponding handler.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

// -------- Parameter structs -------------------------------------------------

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryStoreParams {
    pub project: String,
    pub session_id: String,
    pub content: String,
    #[serde(default)]
    pub metadata: Option<Value>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemorySearchParams {
    pub query: String,
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct MemoryContextParams {
    pub project: String,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub session_id: Option<String>,
    #[serde(default)]
    pub hint: Option<String>,
    #[serde(default)]
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanCreateParams {
    /// Project namespace this plan belongs to. Required: every Plan in
    /// flowd is project-scoped (see HL-38). Overrides any `project` field
    /// embedded in `definition`.
    pub project: String,
    /// Plan definition matching [`flowd_core::orchestration::PlanDefinition`].
    /// May omit the `project` field; the top-level `project` parameter
    /// is authoritative.
    pub definition: Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanConfirmParams {
    pub plan_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanStatusParams {
    pub plan_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanResumeParams {
    pub plan_id: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RulesCheckParams {
    pub tool: String,
    #[serde(default)]
    pub file_path: Option<String>,
    #[serde(default)]
    pub project: Option<String>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct RulesListParams {
    #[serde(default)]
    pub project: Option<String>,
    #[serde(default)]
    pub file_path: Option<String>,
}

// -------- MCP tool-result envelope -----------------------------------------

/// MCP-spec-compatible content block. The server emits exactly one text
/// block per call, carrying a JSON-serialised payload so structured consumers
/// can round-trip while the text form stays human-readable.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolContent {
    #[serde(rename = "type")]
    pub kind: String,
    pub text: String,
}

impl ToolContent {
    #[must_use]
    pub fn text(text: impl Into<String>) -> Self {
        Self {
            kind: "text".into(),
            text: text.into(),
        }
    }
}

/// MCP `tools/call` result envelope.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolResult {
    pub content: Vec<ToolContent>,
    #[serde(rename = "isError", default)]
    pub is_error: bool,
}

impl ToolResult {
    /// Success: pretty-print the JSON payload into a single text block.
    ///
    /// # Errors
    /// Returns an error if `payload` cannot be serialised to JSON.
    pub fn ok(payload: &Value) -> serde_json::Result<Self> {
        let text = serde_json::to_string_pretty(payload)?;
        Ok(Self {
            content: vec![ToolContent::text(text)],
            is_error: false,
        })
    }

    #[must_use]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            content: vec![ToolContent::text(message)],
            is_error: true,
        }
    }
}

// -------- Tool schema (for `tools/list`) -----------------------------------

/// JSON Schema describing a tool, as emitted by MCP `tools/list`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolSchema {
    pub name: String,
    pub description: String,
    #[serde(rename = "inputSchema")]
    pub input_schema: Value,
}

/// Every tool flowd exposes over MCP, with its JSON Schema.
#[must_use]
#[allow(clippy::too_many_lines)] // one schema per tool; splitting adds no clarity.
pub fn all_tool_schemas() -> Vec<ToolSchema> {
    vec![
        ToolSchema {
            name: "memory_store".into(),
            description: "Persist an observation in the memory subsystem (SQLite + vector index)."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["project", "session_id", "content"],
                "properties": {
                    "project":    { "type": "string" },
                    "session_id": { "type": "string", "description": "UUID of the active session" },
                    "content":    { "type": "string" },
                    "metadata":   { "type": "object", "description": "Arbitrary JSON metadata" }
                }
            }),
        },
        ToolSchema {
            name: "memory_search".into(),
            description: "Hybrid keyword + vector search across stored observations (RRF fused)."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query":   { "type": "string" },
                    "project": { "type": "string" },
                    "limit":   { "type": "integer", "minimum": 1 }
                }
            }),
        },
        ToolSchema {
            name: "memory_context".into(),
            description: "Auto-retrieve relevant observations for the current session/file, \
                          plus any rules that apply to that scope."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["project"],
                "properties": {
                    "project":    { "type": "string" },
                    "file_path":  { "type": "string" },
                    "session_id": { "type": "string" },
                    "hint":       { "type": "string" },
                    "limit":      { "type": "integer", "minimum": 1 }
                }
            }),
        },
        ToolSchema {
            name: "plan_create".into(),
            description: "Register a new orchestration plan from its definition. \
                          Returns the plan id plus a preview of execution order and parallelism."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["project", "definition"],
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project namespace for this plan; overrides any project in `definition`."
                    },
                    "definition": {
                        "type": "object",
                        "description": "PlanDefinition as defined by flowd-core"
                    }
                }
            }),
        },
        ToolSchema {
            name: "plan_confirm".into(),
            description: "Approve a Draft plan and begin execution asynchronously.".into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id"],
                "properties": { "plan_id": { "type": "string" } }
            }),
        },
        ToolSchema {
            name: "plan_status".into(),
            description: "Return the current status of a registered plan.".into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id"],
                "properties": { "plan_id": { "type": "string" } }
            }),
        },
        ToolSchema {
            name: "plan_resume".into(),
            description: "Reset failed steps to pending, confirm the plan, and re-run execution in the background."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id"],
                "properties": { "plan_id": { "type": "string", "description": "UUID of the plan to resume" } }
            }),
        },
        ToolSchema {
            name: "rules_check".into(),
            description: "Validate a proposed tool invocation against loaded rules (warn / deny)."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["tool"],
                "properties": {
                    "tool":      { "type": "string" },
                    "file_path": { "type": "string" },
                    "project":   { "type": "string" }
                }
            }),
        },
        ToolSchema {
            name: "rules_list".into(),
            description: "List rules whose scope glob matches the given project / file.".into(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "project":   { "type": "string" },
                    "file_path": { "type": "string" }
                }
            }),
        },
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_tool_has_unique_name_and_schema() {
        let schemas = all_tool_schemas();
        let mut seen = std::collections::HashSet::new();
        for s in &schemas {
            assert!(seen.insert(s.name.clone()), "duplicate tool: {}", s.name);
            assert!(!s.description.trim().is_empty());
            assert_eq!(s.input_schema["type"], "object");
        }
        assert_eq!(schemas.len(), 9);
    }

    #[test]
    fn tool_result_ok_pretty_prints_payload() {
        let result = ToolResult::ok(&json!({"a": 1})).unwrap();
        assert!(!result.is_error);
        assert_eq!(result.content.len(), 1);
        assert!(result.content[0].text.contains("\"a\": 1"));
    }

    #[test]
    fn tool_result_error_sets_flag() {
        let result = ToolResult::error("boom");
        assert!(result.is_error);
        assert_eq!(result.content[0].text, "boom");
    }
}
