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

/// Parameters for the polymorphic `plan_create` MCP tool.
///
/// Two mutually exclusive ways to author a plan:
///
/// * `definition` -- the legacy DAG-first path. The caller supplies a
///   fully-formed [`flowd_core::orchestration::PlanDefinition`] and the
///   handler submits it as-is.
/// * `prose` -- the prose-first path introduced in HL-44. The caller
///   supplies a freeform description; the handler invokes the configured
///   [`flowd_core::orchestration::PlanCompiler`] to surface clarifications
///   and (eventually) compile a DAG.
///
/// Exactly one of the two must be set. The handler enforces this; the
/// JSON-Schema also encodes it via `oneOf` so MCP clients see the
/// constraint up front. The fields are kept as separate `Option`s rather
/// than a serde-tagged enum so `tools/call` `arguments` payloads stay flat
/// and easy for MCP clients to construct dynamically.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanCreateParams {
    /// Project namespace this plan belongs to. Required: every Plan in
    /// flowd is project-scoped (see HL-38). Overrides any `project` field
    /// embedded in `definition`.
    pub project: String,
    /// DAG-first authored plan. Mutually exclusive with `prose`.
    #[serde(default)]
    pub definition: Option<Value>,
    /// Prose-first authored plan. Mutually exclusive with `definition`.
    /// The handler calls the configured compiler and returns any open
    /// clarifications alongside the new plan id.
    #[serde(default)]
    pub prose: Option<String>,
}

/// Parameters for `plan_answer`: the user submits answers to one or more
/// outstanding [`flowd_core::orchestration::OpenQuestion`]s on a `Draft`
/// plan, optionally instructing the compiler to fill in any remaining
/// questions on a best-effort basis.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanAnswerParams {
    pub plan_id: String,
    /// Map of question id -> [`flowd_core::orchestration::Answer`]. At
    /// least one entry must be present unless `defer_remaining` is true.
    pub answers: Vec<PlanAnswerEntry>,
    /// When true the compiler is asked to invent best-effort answers for
    /// any still-open questions and emit them as auto-decisions. Useful
    /// when the user wants to converge fast and trusts the compiler's
    /// defaults.
    #[serde(default)]
    pub defer_remaining: bool,
}

/// Single (`question_id`, answer) pair from the `plan_answer` payload. The
/// inner `answer` is the serde-flattened
/// [`flowd_core::orchestration::Answer`] so the wire shape stays
/// `{"question_id": "...", "kind": "choose", "option_id": "..."}`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanAnswerEntry {
    pub question_id: String,
    #[serde(flatten)]
    pub answer: flowd_core::orchestration::Answer,
}

/// Parameters for `plan_refine`: the user submits a freeform refinement
/// instruction (e.g. "make step c idempotent and add a rollback") and the
/// compiler returns an updated draft.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanRefineParams {
    pub plan_id: String,
    pub feedback: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanConfirmParams {
    pub plan_id: String,
}

/// Parameters for `plan_cancel`: idempotently abandon a plan in any
/// non-terminal state. Distinguished from `plan_resume` because the
/// prose-first front door needs a way to discard half-clarified Draft
/// plans that the user no longer wants to pursue (the executor already
/// supports this; HL-44 just wires it through MCP).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PlanCancelParams {
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
            description: "Register a new orchestration plan. Supply EITHER `definition` (DAG-first) \
                          OR `prose` (prose-first; routed through the configured plan compiler, \
                          may surface clarification questions before a DAG can be compiled). \
                          Returns the plan id plus -- for DAG-first plans -- a preview of execution \
                          order and parallelism."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["project"],
                "properties": {
                    "project": {
                        "type": "string",
                        "description": "Project namespace for this plan; overrides any project in `definition`."
                    },
                    "definition": {
                        "type": "object",
                        "description": "PlanDefinition as defined by flowd-core. Mutually exclusive with `prose`."
                    },
                    "prose": {
                        "type": "string",
                        "description": "Freeform plan description. Routed through the plan compiler; \
                                        may return open clarification questions in the response. \
                                        Mutually exclusive with `definition`."
                    }
                },
                "oneOf": [
                    { "required": ["definition"] },
                    { "required": ["prose"] }
                ]
            }),
        },
        ToolSchema {
            name: "plan_answer".into(),
            description: "Submit answers to outstanding clarification questions on a Draft plan. \
                          The plan compiler is re-invoked with the merged decisions and either \
                          returns more questions, returns a fully compiled DAG, or both. \
                          Set `defer_remaining=true` to ask the compiler to auto-fill any \
                          questions left unanswered in this batch."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id", "answers"],
                "properties": {
                    "plan_id": { "type": "string" },
                    "answers": {
                        "type": "array",
                        "description": "Each entry pairs a question_id with one of three Answer kinds.",
                        "items": {
                            "type": "object",
                            "required": ["question_id", "kind"],
                            "properties": {
                                "question_id": { "type": "string" },
                                "kind": { "type": "string", "enum": ["choose", "explain_more", "none_of_these"] },
                                "option_id": { "type": "string", "description": "Required when kind=choose." },
                                "note": { "type": "string", "description": "Optional context when kind=explain_more." }
                            }
                        }
                    },
                    "defer_remaining": {
                        "type": "boolean",
                        "default": false,
                        "description": "Ask the compiler to auto-answer any still-open questions."
                    }
                }
            }),
        },
        ToolSchema {
            name: "plan_refine".into(),
            description: "Apply a freeform refinement instruction to a Draft plan. The compiler may \
                          return new clarifications if the change introduces new architectural \
                          decisions; otherwise it returns an updated draft (or a fully compiled DAG)."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id", "feedback"],
                "properties": {
                    "plan_id": { "type": "string" },
                    "feedback": {
                        "type": "string",
                        "description": "Natural-language instruction. The compiler echoes a truncated form into the audit log."
                    }
                }
            }),
        },
        ToolSchema {
            name: "plan_confirm".into(),
            description: "Approve a Draft plan and begin execution asynchronously. Refuses plans \
                          that still have open clarification questions; the error payload includes \
                          the outstanding question list so the caller can route them back through \
                          `plan_answer`."
                .into(),
            input_schema: json!({
                "type": "object",
                "required": ["plan_id"],
                "properties": { "plan_id": { "type": "string" } }
            }),
        },
        ToolSchema {
            name: "plan_cancel".into(),
            description: "Idempotently cancel a plan. Draft and Confirmed plans transition \
                          directly to Cancelled; Running plans set the cancellation latch and \
                          abort in-flight steps. Terminal plans are no-ops."
                .into(),
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
        assert_eq!(schemas.len(), 12);
    }

    #[test]
    fn plan_create_schema_enforces_exactly_one_of_definition_or_prose() {
        let schema = all_tool_schemas()
            .into_iter()
            .find(|s| s.name == "plan_create")
            .expect("plan_create schema");
        let one_of = schema.input_schema["oneOf"]
            .as_array()
            .expect("oneOf array");
        assert_eq!(one_of.len(), 2);
        let required: Vec<&str> = one_of
            .iter()
            .filter_map(|v| v["required"][0].as_str())
            .collect();
        assert!(required.contains(&"definition"));
        assert!(required.contains(&"prose"));
    }

    #[test]
    fn new_clarification_tools_are_registered() {
        let names: Vec<String> = all_tool_schemas().into_iter().map(|s| s.name).collect();
        assert!(names.contains(&"plan_answer".to_owned()));
        assert!(names.contains(&"plan_refine".to_owned()));
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
