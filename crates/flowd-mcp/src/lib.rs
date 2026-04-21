//! MCP server for flowd.
//!
//! Exposes flowd's memory, orchestration, and rules capabilities as MCP
//! tools consumable by Claude Code and Cursor.
//!
//! The crate is organised into three layers:
//!
//! * [`protocol`] -- JSON-RPC 2.0 envelope types, transport-agnostic.
//! * [`handlers`] -- the [`handlers::McpHandlers`] trait plus a concrete
//!   [`handlers::FlowdHandlers`] that wires flowd-core services.
//! * [`server`] -- dispatch logic and a line-delimited stdio transport.
//! * [`tools`] -- parameter structs, MCP tool-result envelope, and the
//!   JSON-Schema definitions emitted by `tools/list`.
//!
//! Most callers only need to `use flowd_mcp::{McpServer, McpServerConfig,
//! FlowdHandlers}` and hand the server their composed backends.

pub mod compiler;
pub mod handlers;
pub mod observer;
pub mod protocol;
pub mod server;
pub mod summarizer;
pub mod tools;

pub use compiler::{LlmCallback, LlmPlanCompiler, RejectingPlanCompiler, StubPlanCompiler};
pub use handlers::{FlowdHandlers, McpHandlers};
pub use observer::{
    DEFAULT_CAPACITY as PLAN_EVENT_DEFAULT_CAPACITY, ObserverHealth, PlanEventObserver,
    PlanEventObserverConfig, ShutdownReport,
};
pub use protocol::{JsonRpcError, JsonRpcRequest, JsonRpcResponse};
pub use server::{McpServer, McpServerConfig, dispatch};
pub use summarizer::NoopSummarizer;
pub use tools::{ToolContent, ToolResult, ToolSchema};
