//! Transport-side implementations of [`crate::compiler::LlmCallback`].
//!
//! [`crate::compiler::LlmPlanCompiler`] is generic over `LlmCallback`, so
//! a concrete transport must satisfy the trait but not the compiler's
//! prompt-shaping logic. Splitting transport code from `compiler.rs`
//! keeps that file purely about prompt assembly + JSON parsing, and
//! makes it cheap to swap transports in tests via a hand-rolled mock
//! callback.
//!
//! ## Shipped transports
//!
//! | Module                    | Wire shape                              | Auth                |
//! |---------------------------|-----------------------------------------|---------------------|
//! | [`openai`]                | `POST /v1/chat/completions`             | none / bearer-passthrough |
//! | [`claude_cli`]            | `claude -p ... < prompt` (subprocess)   | delegated to `claude` CLI |
//!
//! A direct Anthropic Messages API path (`claude-http`) is reserved for
//! a follow-up; today the daemon refuses to start with `provider =
//! "claude-http"` so operators see the gap up front rather than at
//! request time.

pub mod claude_cli;
pub mod openai;

pub use claude_cli::{ClaudeCliCallback, ClaudeCliConfig, ClaudeEffort};
pub use openai::{OpenAiCompatibleCallback, OpenAiCompatibleConfig};
