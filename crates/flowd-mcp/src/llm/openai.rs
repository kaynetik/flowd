//! `OpenAI`-compatible HTTP backend that satisfies
//! [`crate::compiler::LlmCallback`].
//!
//! Speaks the `OpenAI` `/v1/chat/completions` shape that `mlx_lm.server`,
//! `vllm`, `ollama`'s OpenAI-compatible mode, and most cloud providers
//! use. The sibling [`crate::llm::claude_cli`] module ships an alternate
//! transport that shells out to the local `claude` CLI for auth-free
//! Anthropic access; see that module for the quality-first default.
//!
//! ## Why separate from `compiler.rs`
//!
//! [`crate::compiler::LlmPlanCompiler`] is generic over `LlmCallback`, so
//! any concrete transport must satisfy the trait but not the compiler's
//! prompt-shaping logic. Splitting the HTTP code out keeps
//! `compiler.rs` purely about prompt assembly and JSON parsing, and
//! makes it cheap to swap transports in tests via a hand-rolled mock
//! callback.
//!
//! ## Wire format we expect
//!
//! ```json
//! POST {base_url}/chat/completions
//! {
//!   "model": "{model}",
//!   "messages": [
//!     {"role": "system", "content": "..."},
//!     {"role": "user", "content": "..."}
//!   ],
//!   "temperature": 0.2,
//!   "max_tokens": 4096,
//!   "stream": false
//! }
//! ```
//!
//! Response (only `choices[0].message.content` is consulted; everything
//! else is ignored so a future provider that adds extra metadata won't
//! break us):
//!
//! ```json
//! {
//!   "choices": [
//!     {"message": {"role": "assistant", "content": "..."}}
//!   ]
//! }
//! ```

use std::future::Future;
use std::time::Duration;

use flowd_core::error::{FlowdError, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::compiler::LlmCallback;

/// Configuration for [`OpenAiCompatibleCallback`].
///
/// All fields are required because callers nearly always have them
/// available (read off `[plan.llm]` in `flowd.toml`); defaulting them
/// here would just push validation down by a layer.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleConfig {
    /// Base URL including the `/v1` segment, e.g.
    /// `http://127.0.0.1:11434/v1`. We append `/chat/completions` to it
    /// when sending requests.
    pub base_url: String,
    /// Model id passed in the request body. For `mlx_lm.server` this
    /// must match the model the server was launched with.
    pub model: String,
    /// Per-request timeout. Plan compilation is a single-shot completion
    /// today, so this caps the entire round-trip.
    pub timeout: Duration,
    /// Cap on completion tokens. Forwarded as-is to the server.
    pub max_tokens: u32,
    /// Sampling temperature. 0.2 is what we recommend for plan
    /// compilation (see [`crate::compiler`] docs).
    pub temperature: f32,
}

/// HTTP callback that talks to any OpenAI-compatible
/// `/v1/chat/completions` endpoint.
///
/// Stateless beyond the [`reqwest::Client`] it owns -- the client is
/// cheap to clone (it's `Arc`-backed) but we keep one per callback so
/// connection pooling persists across the daemon's lifetime. Use
/// [`OpenAiCompatibleCallback::with_client`] to inject a pre-built
/// client for tests that want to override TLS / proxy behaviour.
#[derive(Debug, Clone)]
pub struct OpenAiCompatibleCallback {
    client: Client,
    cfg: OpenAiCompatibleConfig,
}

impl OpenAiCompatibleCallback {
    /// Build a callback with a freshly constructed [`Client`] that
    /// honours `cfg.timeout`. Most callers want this.
    ///
    /// # Errors
    /// Returns `FlowdError::Internal` if the [`Client`] cannot be
    /// constructed (TLS backend init failure on exotic platforms).
    pub fn new(cfg: OpenAiCompatibleConfig) -> Result<Self> {
        let client = Client::builder()
            .timeout(cfg.timeout)
            .build()
            .map_err(|e| FlowdError::Internal(format!("build reqwest client: {e}")))?;
        Ok(Self::with_client(client, cfg))
    }

    /// Like [`Self::new`] but uses an externally supplied client. Useful
    /// for tests (e.g. to point at a `wiremock` server that uses
    /// `http://` only and therefore doesn't need TLS).
    #[must_use]
    pub fn with_client(client: Client, cfg: OpenAiCompatibleConfig) -> Self {
        Self { client, cfg }
    }
}

/// Wire-shape we send. The `OpenAI` spec accepts many more fields; we
/// keep ours minimal so the same body works against `mlx_lm.server`,
/// `vllm`, and any compliant cloud endpoint without per-provider
/// special-casing.
#[derive(Debug, Serialize)]
struct ChatRequest<'a> {
    model: &'a str,
    messages: Vec<ChatMessage<'a>>,
    temperature: f32,
    max_tokens: u32,
    /// Always false: the compiler needs the full body before it can
    /// parse JSON, so streaming would just buffer and add latency.
    stream: bool,
}

#[derive(Debug, Serialize)]
struct ChatMessage<'a> {
    role: &'a str,
    content: &'a str,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    /// Empty in practice means the upstream served an error JSON shape;
    /// we treat that as a transport failure with a helpful message.
    #[serde(default)]
    choices: Vec<ChatChoice>,
    /// OpenAI-style error payload. Surfaced as `Internal` so the daemon
    /// log shows the upstream message verbatim.
    #[serde(default)]
    error: Option<ChatErrorPayload>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageOwned,
}

#[derive(Debug, Deserialize)]
struct ChatMessageOwned {
    #[allow(dead_code)] // We keep the role for symmetry / future filtering.
    #[serde(default)]
    role: String,
    #[serde(default)]
    content: String,
}

#[derive(Debug, Deserialize)]
struct ChatErrorPayload {
    #[serde(default)]
    message: String,
    #[serde(default, rename = "type")]
    kind: String,
}

impl LlmCallback for OpenAiCompatibleCallback {
    fn complete(&self, prompt: String) -> impl Future<Output = Result<String>> + Send {
        let url = format!(
            "{}/chat/completions",
            self.cfg.base_url.trim_end_matches('/')
        );
        let cfg = self.cfg.clone();
        let client = self.client.clone();
        async move {
            // The compiler hands us a single text blob. The simplest
            // wire-faithful split is to use it verbatim as the user
            // message and ship a tiny system message that pins the
            // assistant to JSON-only output. The compiler itself bakes
            // the full instructions into `prompt`, so the system
            // message is mostly defensive padding.
            let req = ChatRequest {
                model: &cfg.model,
                messages: vec![
                    ChatMessage {
                        role: "system",
                        content: "You are flowd's plan compiler. \
                                  Reply with a single JSON object that matches the \
                                  schema in the user message. \
                                  Do not include any prose, markdown fences, or commentary \
                                  outside that JSON object.",
                    },
                    ChatMessage {
                        role: "user",
                        content: &prompt,
                    },
                ],
                temperature: cfg.temperature,
                max_tokens: cfg.max_tokens,
                stream: false,
            };

            let resp = client.post(&url).json(&req).send().await.map_err(|e| {
                // Network / DNS / connect failure. We classify as
                // `Internal` rather than `PlanValidation` because the
                // user can't fix their prose to recover from "MLX
                // server is down".
                FlowdError::Internal(format!(
                    "POST {url}: {e}; \
                     check that mlx_lm.server (or the configured backend) is reachable"
                ))
            })?;

            let status = resp.status();
            let body = resp.text().await.map_err(|e| {
                FlowdError::Internal(format!("read body from {url} ({status}): {e}"))
            })?;

            if !status.is_success() {
                // Try to surface the OpenAI-style error message; fall
                // back to a truncated raw body if the response isn't
                // JSON.
                let detail = serde_json::from_str::<ChatResponse>(&body)
                    .ok()
                    .and_then(|r| r.error)
                    .map_or_else(
                        || truncate(&body, 512),
                        |e| {
                            if e.kind.is_empty() {
                                e.message
                            } else {
                                format!("{}: {}", e.kind, e.message)
                            }
                        },
                    );
                return Err(FlowdError::Internal(format!(
                    "{url} returned {status}: {detail}"
                )));
            }

            let parsed: ChatResponse = serde_json::from_str(&body).map_err(|e| {
                FlowdError::Internal(format!(
                    "parse JSON response from {url}: {e}; first 256 bytes: {}",
                    truncate(&body, 256)
                ))
            })?;

            if let Some(err) = parsed.error {
                // 200 OK with an `error` block -- some OpenAI-compatible
                // servers (notably older mlx_lm.server builds) do this
                // when the model fails to load. Surface as Internal.
                let label = if err.kind.is_empty() {
                    err.message
                } else {
                    format!("{}: {}", err.kind, err.message)
                };
                return Err(FlowdError::Internal(format!(
                    "{url} returned 200 with embedded error: {label}"
                )));
            }

            let first = parsed.choices.into_iter().next().ok_or_else(|| {
                FlowdError::Internal(format!(
                    "{url} returned 200 with no choices; backend likely returned an empty completion"
                ))
            })?;
            Ok(first.message.content)
        }
    }
}

/// Truncate `s` to at most `max_chars` characters, preserving UTF-8
/// boundaries by walking char indices. Used only for error logging, so
/// performance is not a concern.
fn truncate(s: &str, max_chars: usize) -> String {
    if s.chars().count() <= max_chars {
        return s.to_string();
    }
    let mut end = s.len();
    for (i, _) in s.char_indices().take(max_chars + 1).enumerate() {
        if i == max_chars {
            end = s
                .char_indices()
                .nth(max_chars)
                .map_or(s.len(), |(idx, _)| idx);
            break;
        }
    }
    format!("{}…", &s[..end])
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    fn cfg_for(server: &MockServer) -> OpenAiCompatibleConfig {
        OpenAiCompatibleConfig {
            base_url: format!("{}/v1", server.uri()),
            model: "test/model".into(),
            timeout: Duration::from_secs(5),
            max_tokens: 256,
            temperature: 0.0,
        }
    }

    #[tokio::test]
    async fn happy_path_returns_message_content() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(header("content-type", "application/json"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{
                    "message": {"role": "assistant", "content": "{\"hello\":\"world\"}"}
                }]
            })))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let out = cb.complete("anything".into()).await.unwrap();
        assert_eq!(out, "{\"hello\":\"world\"}");
    }

    #[tokio::test]
    async fn http_error_status_is_classified_as_internal_with_upstream_detail() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(503).set_body_json(json!({
                "error": {"type": "model_unavailable", "message": "MLX worker is loading"}
            })))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        match err {
            FlowdError::Internal(m) => {
                assert!(m.contains("503"), "{m}");
                assert!(m.contains("model_unavailable"), "{m}");
                assert!(m.contains("MLX worker is loading"), "{m}");
            }
            other => panic!("expected Internal, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn http_error_with_non_json_body_falls_back_to_raw_text() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(500).set_body_string("nginx upstream gone"))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        assert!(format!("{err}").contains("nginx upstream gone"));
    }

    #[tokio::test]
    async fn ok_with_embedded_error_block_is_treated_as_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "error": {"type": "load_failure", "message": "weights missing"}
            })))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("embedded error"), "{s}");
        assert!(s.contains("weights missing"), "{s}");
    }

    #[tokio::test]
    async fn empty_choices_array_is_treated_as_failure() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({"choices": []})))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        assert!(format!("{err}").contains("no choices"));
    }

    #[tokio::test]
    async fn non_json_body_on_2xx_is_classified_as_internal_with_preview() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .respond_with(ResponseTemplate::new(200).set_body_string("hello not json"))
            .mount(&server)
            .await;

        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("parse JSON response"), "{s}");
        assert!(s.contains("hello not json"), "{s}");
    }

    #[tokio::test]
    async fn request_body_carries_configured_model_and_sampling_params() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/v1/chat/completions"))
            .and(wiremock::matchers::body_partial_json(json!({
                "model": "test/model",
                "temperature": 0.0,
                "max_tokens": 256,
                "stream": false,
            })))
            .respond_with(ResponseTemplate::new(200).set_body_json(json!({
                "choices": [{"message": {"role": "assistant", "content": "ok"}}]
            })))
            .mount(&server)
            .await;
        let cb = OpenAiCompatibleCallback::new(cfg_for(&server)).unwrap();
        let out = cb.complete("user prompt".into()).await.unwrap();
        assert_eq!(out, "ok");
    }

    #[tokio::test]
    async fn connect_failure_surfaces_helpful_internal_error() {
        // 127.0.0.1:1 is reserved-by-convention and should refuse
        // immediately on most platforms. We don't strictly need a
        // specific kind; we only need to confirm the code path
        // produces a useful error message rather than panicking.
        let cfg = OpenAiCompatibleConfig {
            base_url: "http://127.0.0.1:1/v1".into(),
            model: "x".into(),
            timeout: Duration::from_millis(250),
            max_tokens: 16,
            temperature: 0.0,
        };
        let cb = OpenAiCompatibleCallback::new(cfg).unwrap();
        let err = cb.complete("hi".into()).await.unwrap_err();
        let s = format!("{err}");
        assert!(s.contains("POST"), "{s}");
        assert!(s.contains("mlx_lm.server"), "{s}");
    }

    #[test]
    fn truncate_keeps_short_strings_intact() {
        assert_eq!(truncate("hello", 10), "hello");
    }

    #[test]
    fn truncate_appends_ellipsis_when_clipping() {
        let out = truncate("abcdefghij", 5);
        assert_eq!(out, "abcde…");
    }

    #[test]
    fn truncate_respects_utf8_boundaries() {
        let s = "αβγδε";
        let out = truncate(s, 3);
        assert_eq!(out, "αβγ…");
    }
}
