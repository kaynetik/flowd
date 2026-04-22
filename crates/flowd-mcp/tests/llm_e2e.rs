//! End-to-end smoke test against a *live* OpenAI-compatible LLM server.
//!
//! Always `#[ignore]`d so the default `cargo test` run does not need the
//! daemon nor a model loaded into memory. Run on demand:
//!
//! ```bash
//! # Start the server in another shell first. Either backend works as long
//! # as it speaks OpenAI `/v1/chat/completions`. The daemon's documented
//! # MLX default is `qwen3-coder:30b` because Ollama's OpenAI shim ignores
//! # `think:false` and Qwen3-thinking models will burn the entire token
//! # budget reasoning before emitting JSON.
//! mlx_lm.server --model mlx-community/qwen3-coder:30b --port 8080
//! # or:  ollama serve  (then pull `qwen3-coder:30b` once)
//!
//! # Then exercise the round-trip:
//! cargo test -p flowd-mcp --test llm_e2e -- --ignored --nocapture
//! ```
//!
//! Override the target endpoint and model with environment variables:
//!
//! * `FLOWD_LLM_BASE_URL` -- defaults to `http://127.0.0.1:11434/v1`
//! * `FLOWD_LLM_MODEL` -- defaults to `qwen3-coder:30b`
//! * `FLOWD_LLM_TIMEOUT_SECS` -- defaults to `120`
//!
//! ## What we assert
//!
//! 1. `compile_prose` over a deliberately ambiguous brief produces
//!    *something* parseable -- either a definition (model felt
//!    confident) or at least one open question (model wanted
//!    clarification). The strict invariant
//!    `definition.is_some() iff open_questions.is_empty()` is enforced
//!    by the compiler itself; we just sanity-check the surface.
//! 2. A second call with a structured brief should produce a
//!    definition with at least one step (sanity for "happy path"
//!    output).
//!
//! These tests deliberately do not assert step counts or names: small
//! models are non-deterministic, so we only require *shape* compliance,
//! which is what the prose-first loop actually depends on at runtime.

use std::env;
use std::sync::Arc;
use std::time::Duration;

use flowd_core::orchestration::PlanCompiler;
use flowd_mcp::compiler::LlmPlanCompiler;
use flowd_mcp::llm::{OpenAiCompatibleCallback, OpenAiCompatibleConfig};

fn build_compiler() -> LlmPlanCompiler<OpenAiCompatibleCallback> {
    let base_url =
        env::var("FLOWD_LLM_BASE_URL").unwrap_or_else(|_| "http://127.0.0.1:11434/v1".to_string());
    let model = env::var("FLOWD_LLM_MODEL").unwrap_or_else(|_| "qwen3-coder:30b".to_string());
    let timeout_secs = env::var("FLOWD_LLM_TIMEOUT_SECS")
        .ok()
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(120);

    let cfg = OpenAiCompatibleConfig {
        base_url,
        model,
        timeout: Duration::from_secs(timeout_secs),
        // Plan responses are bounded JSON; 2k is comfortable for a 5-7
        // step plan and keeps slow models from hitting their default
        // generation cap.
        max_tokens: 2048,
        // Determinism over creativity. The model is shaping JSON, not
        // writing prose -- 0.0 was unstable on Qwen during pilot runs,
        // 0.2 is the documented sweet spot.
        temperature: 0.2,
    };

    let cb = OpenAiCompatibleCallback::new(cfg).expect("build OpenAI-compatible callback");
    LlmPlanCompiler::new(Arc::new(cb))
}

#[tokio::test]
#[ignore = "requires a running mlx_lm.server / OpenAI-compatible endpoint -- run with --ignored"]
async fn live_compile_prose_handles_ambiguous_brief() {
    let compiler = build_compiler();
    let prose = "Refactor the auth module so we can swap the JWT signing algorithm \
                 without touching the session layer. Then run the integration tests.";

    let out = compiler
        .compile_prose(prose.into(), "live-e2e".into())
        .await
        .expect("compile_prose returns ok");

    assert_eq!(out.source_doc, prose, "source_doc preserved verbatim");

    // Either path is valid; we just want the model to commit to one.
    if out.definition.is_some() {
        assert!(
            out.open_questions.is_empty(),
            "definition + questions is an invariant violation"
        );
    } else {
        assert!(
            !out.open_questions.is_empty(),
            "no definition AND no questions; compiler bailed"
        );
    }
}

#[tokio::test]
#[ignore = "requires a running mlx_lm.server / OpenAI-compatible endpoint -- run with --ignored"]
async fn live_compile_prose_handles_structured_brief() {
    let compiler = build_compiler();
    let prose = "Plan: rename `Foo` to `Bar` across the codebase.\n\
                 Steps:\n\
                 1. Update the Rust source files.\n\
                 2. Update the documentation.\n\
                 3. Run `cargo test` to verify.";

    let out = compiler
        .compile_prose(prose.into(), "live-e2e".into())
        .await
        .expect("compile_prose returns ok");

    if let Some(def) = out.definition {
        assert!(!def.steps.is_empty(), "structured brief should yield steps");
    } else {
        // Acceptable if the model wanted to clarify (e.g. asked which
        // crates to scope to). We still want the surface to be sane.
        assert!(
            !out.open_questions.is_empty(),
            "no definition AND no questions; compiler bailed"
        );
    }
}
