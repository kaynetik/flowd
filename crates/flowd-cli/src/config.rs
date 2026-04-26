//! `flowd.toml` -- runtime configuration for the daemon and CLI.
//!
//! ## Layout
//!
//! The config file lives at `$FLOWD_HOME/flowd.toml`. It is optional:
//! when absent, every section falls back to documented defaults so a
//! fresh installation works zero-config. When present, only the keys
//! the operator overrides are honoured -- absent keys keep their
//! defaults rather than resetting to zero.
//!
//! ```toml
//! [plan]
//! # Maximum number of clarification questions the prose-first compiler
//! # may have outstanding before the daemon coerces `defer_remaining = true`
//! # on the next `plan_answer` call and surfaces a `BudgetExceeded`
//! # warning. Counts both `open_questions` and resolved `decisions` so
//! # long clarification chains converge instead of running forever.
//! max_questions = 12
//!
//! # Which `PlanCompiler` implementation the daemon wires:
//! #   * "stub"      -- StubPlanCompiler (deterministic markdown parser).
//! #   * "rejecting" -- RejectingPlanCompiler (every prose-first call errors;
//! #                    use this when prose-first must be disabled at the
//! #                    deployment level).
//! #   * "llm"       -- LlmPlanCompiler. Requires the [plan.llm] block below.
//! compiler = "llm"
//!
//! # Only consulted when [plan].compiler = "llm".
//! #
//! # The provider key picks which subsection the daemon reads; the other
//! # subsections still parse (and apply defaults) so an operator can flip
//! # `provider` without re-authoring their config.
//! [plan.llm]
//! # Backend that fronts the model:
//! #   * "claude-cli"  -- shells out to the local `claude` CLI for auth-free
//! #                       Anthropic access (default; quality-first).
//! #   * "mlx"         -- talks OpenAI `/v1/chat/completions` to a local
//! #                       mlx_lm.server / vMLX / ollama-compat endpoint
//! #                       (offline fallback).
//! #   * "claude-http" -- direct Anthropic Messages API over HTTPS;
//! #                       requires an API key in $ANTHROPIC_API_KEY (or the
//! #                       env var named in `[plan.llm.claude_http].api_key_env`).
//! provider = "claude-cli"
//!
//! # Subsection consumed when provider = "claude-cli".
//! [plan.llm.claude_cli]
//! # The `claude` CLI accepts both tier aliases (`sonnet`, `opus`,
//! # `haiku`) -- which auto-resolve to the current latest build of
//! # that tier -- and fully-pinned model identifiers (see
//! # `claude --help`). Operators who need byte-for-byte reproducible
//! # plans should pin a specific identifier (e.g. `claude-opus-4-7`).
//! model        = "opus"
//! binary       = "claude"   # path or executable name resolved on $PATH
//! timeout_secs = 240
//! # Optional reasoning-effort override forwarded as `--effort <level>`.
//! # Accepts: "low" | "medium" | "high" | "xhigh" | "max". Omit (or
//! # comment out) to let the CLI's own resolution order win
//! # (env -> ~/.claude/settings.json -> model default), which is what
//! # operators who already pin `effortLevel` interactively want.
//! # effort     = "high"
//!
//! # Subsection consumed when provider = "mlx".
//! #
//! # Despite the historical name, this section configures *any* OpenAI
//! # `/v1/chat/completions`-compatible local server (Ollama, mlx_lm.server,
//! # vLLM, llama.cpp's server, ...). Defaults target Ollama because it
//! # is the most common local setup; MLX users typically override
//! # `base_url` to `http://127.0.0.1:8080/v1` and `model` to an MLX
//! # community identifier (e.g. `mlx-community/Qwen3-Coder-30B-...`).
//! [plan.llm.mlx]
//! model        = "qwen3-coder:30b"
//! base_url     = "http://127.0.0.1:11434/v1"
//! timeout_secs = 60
//! max_tokens   = 4096
//! temperature  = 0.2
//!
//! # Subsection consumed when provider = "claude-http".
//! # NOTE: the direct Anthropic Messages API transport is not yet
//! # implemented; the daemon refuses to start with provider = "claude-http"
//! # today. The block below documents the on-disk shape so operators
//! # can stage the config now and flip `provider` once it lands.
//! [plan.llm.claude_http]
//! model        = "claude-sonnet-4-5"
//! api_key_env  = "ANTHROPIC_API_KEY"
//! base_url     = "https://api.anthropic.com"
//! timeout_secs = 120
//! max_tokens   = 4096
//! temperature  = 0.2
//!
//! # Optional escalation tier: when present, plan_refine() routes through
//! # this backend instead of the primary above. compile_prose / apply_answers
//! # still go through the primary. Same shape as [plan.llm] (with its own
//! # per-provider subsections); a refine block may not nest a further
//! # refine of its own.
//! [plan.llm.refine]
//! provider = "claude-cli"
//! [plan.llm.refine.claude_cli]
//! model        = "opus"
//! timeout_secs = 240
//! ```
//!
//! ## Why a hand-rolled struct instead of `serde(default)` everywhere
//!
//! `toml::Value::try_into` with `#[serde(default)]` would work, but we
//! intentionally split parse-time validation from defaulting so that
//! malformed values (`compiler = "gibberish"`) produce a clear error
//! pointing at the file rather than silently selecting a default. The
//! defaults therefore live on the typed struct, not on the raw
//! `Deserialize` impl.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow};
use flowd_mcp::ClaudeEffort;
use serde::Deserialize;

/// Default question budget. Matches the value documented in the
/// HL-44 spec; chosen to be large enough for typical refactors yet
/// small enough that runaway compilers visibly trip the warning.
pub const DEFAULT_MAX_QUESTIONS: usize = 12;

/// Default daemon-wide fallback timeout (seconds) applied to steps
/// whose per-step `timeout_secs` is unset. Prose-compiled plans always
/// leave that field `None` (the LLM prompt doesn't surface it and the
/// structured-stub markdown grammar can't express it), so without this
/// bound a wedged step would block its execution layer indefinitely.
/// 1 hour is generous enough to accommodate a long Opus high-effort
/// round (multi-tool reasoning, refine passes, slow CLI hand-offs)
/// without misfiring, yet still keeps a stuck agent from wedging a
/// layer overnight before the operator notices. Set
/// `step_timeout_secs = 0` in `[plan]` to disable the fallback
/// entirely.
pub const DEFAULT_STEP_TIMEOUT_SECS: u64 = 3600;

// -- Claude CLI defaults ----------------------------------------------------

/// Default model id passed to the local `claude` CLI when
/// `provider = "claude-cli"`.
///
/// We default to the `"opus"` alias -- plan compilation is
/// quality-sensitive (a botched plan cascades into wasted tool calls
/// downstream), and operators on Anthropic's MAX subscription pay for
/// Opus access whether they use it or not, so spending the latency on
/// the strongest tier by default is the cheap call. Aliases auto-resolve
/// to the latest build of that tier so this default does not bit-rot
/// when Anthropic ships a new Opus snapshot. Operators who want
/// byte-for-byte reproducible plans pin a specific identifier (e.g.
/// `claude-opus-4-7`); those who want lower latency / cost set this to
/// `"sonnet"` or `"haiku"`.
pub const DEFAULT_CLAUDE_CLI_MODEL: &str = "opus";

/// Default `binary` for the Claude CLI backend. Resolved on `$PATH`
/// when the value contains no path separator; treated as a literal
/// path otherwise.
pub const DEFAULT_CLAUDE_CLI_BINARY: &str = "claude";

/// Default per-request timeout for the Claude CLI shell-out. Latency
/// varies with prompt size and the model's reasoning depth; 240s
/// leaves headroom for Opus-tier responses (especially with high
/// `effort`) while still failing fast enough to be useful in an
/// interactive loop. Operators on Sonnet/Haiku can comfortably halve
/// this if they want tighter feedback on hung calls.
pub const DEFAULT_CLAUDE_CLI_TIMEOUT_SECS: u64 = 240;

// -- MLX defaults -----------------------------------------------------------
//
// Despite the `MLX` naming (preserved for config-file compatibility),
// these defaults target *any* OpenAI `/v1/chat/completions`-compatible
// local server. The Ollama-shaped defaults below are the most common
// local setup; MLX users typically override both the URL (port 8080)
// and the model id (HuggingFace-style, e.g. `mlx-community/...`).

/// Default model id for the OpenAI-compatible local backend. The value
/// is in Ollama tag format (`<name>:<size-tag>`) because Ollama is the
/// most common local server we see in the wild; users running
/// `mlx_lm.server`, vLLM, or llama.cpp will override this with whatever
/// id their server exposes.
pub const DEFAULT_MLX_MODEL: &str = "qwen3-coder:30b";

/// Default `base_url` for the OpenAI-compatible local backend. Matches
/// the port Ollama listens on by default; `mlx_lm.server` defaults to
/// 8080 and operators on that stack should override this.
pub const DEFAULT_MLX_BASE_URL: &str = "http://127.0.0.1:11434/v1";

/// Default request timeout for the MLX backend.
pub const DEFAULT_MLX_TIMEOUT_SECS: u64 = 60;

/// Default `max_tokens` for the MLX backend. Plan responses are small
/// JSON blobs; 4 KB of completion is plenty for a 10-step plan with a
/// couple of open questions.
pub const DEFAULT_MLX_MAX_TOKENS: u32 = 4096;

/// Default sampling temperature for the MLX backend. Plan compilation
/// wants determinism, not creativity.
pub const DEFAULT_MLX_TEMPERATURE: f32 = 0.2;

// -- Claude HTTP defaults ---------------------------------------------------

/// Default model for the direct Anthropic HTTP backend.
///
/// The HTTP transport is not yet implemented (the daemon refuses to
/// start with `provider = "claude-http"`), so this value is purely
/// cosmetic today. When the transport lands, operators must set this
/// to a model identifier accepted by the Anthropic Messages API
/// (the API does not honour tier aliases the way the CLI does).
pub const DEFAULT_CLAUDE_HTTP_MODEL: &str = "claude-sonnet-4-5";

/// Default env-var name the daemon reads at startup to obtain the
/// Anthropic API key. Operators who already use a different env var
/// (`CLAUDE_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, ...) can rename it here
/// rather than rewriting their secrets manager.
pub const DEFAULT_CLAUDE_HTTP_API_KEY_ENV: &str = "ANTHROPIC_API_KEY";

/// Default Anthropic Messages API base.
pub const DEFAULT_CLAUDE_HTTP_BASE_URL: &str = "https://api.anthropic.com";

/// Default Claude HTTP timeout.
pub const DEFAULT_CLAUDE_HTTP_TIMEOUT_SECS: u64 = 120;

/// Default Claude HTTP `max_tokens`.
pub const DEFAULT_CLAUDE_HTTP_MAX_TOKENS: u32 = 4096;

/// Default Claude HTTP temperature.
pub const DEFAULT_CLAUDE_HTTP_TEMPERATURE: f32 = 0.2;

/// Compiler implementation the daemon should instantiate.
///
/// Kept deliberately small. Adding a new variant here means the
/// `try_from_str` parser, the `wire_name` table, and the daemon's
/// composition site (`DaemonPlanCompiler::from_selection`) all need an
/// arm; the compiler will tell you about every one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerSelection {
    Stub,
    Rejecting,
    Llm,
}

impl CompilerSelection {
    fn try_from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "stub" => Ok(Self::Stub),
            "rejecting" => Ok(Self::Rejecting),
            "llm" => Ok(Self::Llm),
            other => Err(anyhow!(
                "[plan] compiler = \"{other}\" is not a known compiler; \
                 expected one of: \"stub\", \"rejecting\", \"llm\""
            )),
        }
    }

    /// Stable wire name -- matches the strings the parser accepts.
    /// Used for log messages so the daemon's startup banner shows the
    /// same token an operator would write in `flowd.toml`.
    #[must_use]
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::Stub => "stub",
            Self::Rejecting => "rejecting",
            Self::Llm => "llm",
        }
    }
}

/// LLM backend selector.
///
/// `ClaudeCli` is the default and the recommended path: it shells out
/// to the local `claude` CLI for auth-free Anthropic access (plan
/// compilation is quality-sensitive, and the CLI routes through
/// whatever Anthropic credentials the binary already manages so flowd
/// never has to ingest, persist, or rotate an API key). `Mlx` is the
/// offline / local-server fallback (any OpenAI-compatible endpoint).
/// `ClaudeHttp` is reserved for a follow-up that will call the
/// Anthropic Messages API directly with an operator-supplied key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LlmProvider {
    ClaudeCli,
    Mlx,
    ClaudeHttp,
}

impl LlmProvider {
    fn try_from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            // Accept both kebab-case (idiomatic on the wire) and
            // snake_case (matches the subsection name) so operators who
            // copy-paste from one to the other are not surprised.
            "claude-cli" | "claude_cli" => Ok(Self::ClaudeCli),
            "mlx" => Ok(Self::Mlx),
            "claude-http" | "claude_http" => Ok(Self::ClaudeHttp),
            other => Err(anyhow!(
                "[plan.llm] provider = \"{other}\" is not a known provider; \
                 expected one of: \"claude-cli\", \"mlx\", \"claude-http\""
            )),
        }
    }

    /// Stable wire name; used for startup banners, tracing fields, and
    /// the `compiler_override` argument on `plan_create`.
    #[must_use]
    pub fn wire_name(self) -> &'static str {
        match self {
            Self::ClaudeCli => "claude-cli",
            Self::Mlx => "mlx",
            Self::ClaudeHttp => "claude-http",
        }
    }
}

// ---------------------------------------------------------------------------
// Per-provider config: raw (on-disk) and resolved (in-memory) shapes
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Deserialize)]
struct RawClaudeCliConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    binary: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    effort: Option<String>,
}

/// Resolved `[plan.llm.claude_cli]` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeCliConfig {
    pub model: String,
    /// Either a bare executable name (resolved on `$PATH` at startup) or
    /// an absolute / relative path to the binary.
    pub binary: PathBuf,
    pub timeout_secs: u64,
    /// Optional reasoning-effort override forwarded to `claude -p` as
    /// `--effort <level>`. `None` (the default, and the value used when
    /// the operator omits the key) leaves the flag off the command line
    /// so the CLI's own resolution order applies (env, then
    /// `~/.claude/settings.json`'s `effortLevel`, then the model's
    /// built-in default). `Some` pins the tier for every daemon-issued
    /// request, overriding interactive defaults.
    pub effort: Option<ClaudeEffort>,
}

impl Default for ClaudeCliConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_CLAUDE_CLI_MODEL.to_string(),
            binary: PathBuf::from(DEFAULT_CLAUDE_CLI_BINARY),
            timeout_secs: DEFAULT_CLAUDE_CLI_TIMEOUT_SECS,
            effort: None,
        }
    }
}

impl ClaudeCliConfig {
    fn from_raw(raw: &RawClaudeCliConfig) -> Result<Self> {
        let model = raw
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_CLAUDE_CLI_MODEL.to_string());
        if model.trim().is_empty() {
            return Err(anyhow!(
                "[plan.llm.claude_cli] model must be a non-empty string"
            ));
        }
        let binary = match raw.binary.as_deref() {
            Some(s) if !s.trim().is_empty() => PathBuf::from(s),
            Some(_) => {
                return Err(anyhow!(
                    "[plan.llm.claude_cli] binary must be a non-empty string"
                ));
            }
            None => PathBuf::from(DEFAULT_CLAUDE_CLI_BINARY),
        };
        let timeout_secs = raw.timeout_secs.unwrap_or(DEFAULT_CLAUDE_CLI_TIMEOUT_SECS);
        if timeout_secs == 0 {
            return Err(anyhow!("[plan.llm.claude_cli] timeout_secs must be > 0"));
        }
        let effort = parse_optional_effort(raw.effort.as_deref(), "[plan.llm.claude_cli]")?;
        Ok(Self {
            model,
            binary,
            timeout_secs,
            effort,
        })
    }
}

/// Parse a `[plan.llm.*.claude_cli].effort` value. `None` and the
/// empty string both resolve to "no override"; anything else must
/// match a `ClaudeEffort` variant or we error with a section pointer
/// so the operator knows which block to fix.
fn parse_optional_effort(raw: Option<&str>, section: &str) -> Result<Option<ClaudeEffort>> {
    let Some(s) = raw else { return Ok(None) };
    if s.trim().is_empty() {
        return Ok(None);
    }
    ClaudeEffort::try_from_str(s).map(Some).map_err(|bad| {
        anyhow!(
            "{section} effort = \"{bad}\" is not a known reasoning tier; \
             expected one of: \"low\", \"medium\", \"high\", \"xhigh\", \"max\""
        )
    })
}

#[derive(Debug, Default, Clone, Deserialize)]
struct RawMlxConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

/// Resolved `[plan.llm.mlx]` block.
#[derive(Debug, Clone, PartialEq)]
pub struct MlxConfig {
    pub model: String,
    pub base_url: String,
    pub timeout_secs: u64,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for MlxConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_MLX_MODEL.to_string(),
            base_url: DEFAULT_MLX_BASE_URL.to_string(),
            timeout_secs: DEFAULT_MLX_TIMEOUT_SECS,
            max_tokens: DEFAULT_MLX_MAX_TOKENS,
            temperature: DEFAULT_MLX_TEMPERATURE,
        }
    }
}

impl MlxConfig {
    fn from_raw(raw: &RawMlxConfig) -> Result<Self> {
        let model = raw
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_MLX_MODEL.to_string());
        if model.trim().is_empty() {
            return Err(anyhow!("[plan.llm.mlx] model must be a non-empty string"));
        }
        let base_url = raw
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_MLX_BASE_URL.to_string());
        if base_url.trim().is_empty() {
            return Err(anyhow!(
                "[plan.llm.mlx] base_url must be a non-empty string"
            ));
        }
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(anyhow!(
                "[plan.llm.mlx] base_url must start with http:// or https:// (got `{base_url}`)"
            ));
        }
        let timeout_secs = raw.timeout_secs.unwrap_or(DEFAULT_MLX_TIMEOUT_SECS);
        if timeout_secs == 0 {
            return Err(anyhow!("[plan.llm.mlx] timeout_secs must be > 0"));
        }
        let max_tokens = raw.max_tokens.unwrap_or(DEFAULT_MLX_MAX_TOKENS);
        if max_tokens == 0 {
            return Err(anyhow!("[plan.llm.mlx] max_tokens must be > 0"));
        }
        let temperature = raw.temperature.unwrap_or(DEFAULT_MLX_TEMPERATURE);
        if !(0.0..=2.0).contains(&temperature) || !temperature.is_finite() {
            return Err(anyhow!(
                "[plan.llm.mlx] temperature must be a finite value in [0.0, 2.0] (got {temperature})"
            ));
        }
        Ok(Self {
            model,
            base_url,
            timeout_secs,
            max_tokens,
            temperature,
        })
    }
}

#[derive(Debug, Default, Clone, Deserialize)]
struct RawClaudeHttpConfig {
    #[serde(default)]
    model: Option<String>,
    #[serde(default)]
    api_key_env: Option<String>,
    #[serde(default)]
    base_url: Option<String>,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    max_tokens: Option<u32>,
    #[serde(default)]
    temperature: Option<f32>,
}

/// Resolved `[plan.llm.claude_http]` block.
#[derive(Debug, Clone, PartialEq)]
pub struct ClaudeHttpConfig {
    pub model: String,
    /// Name of the environment variable the daemon reads to obtain the
    /// Anthropic API key. The daemon does not store the secret in the
    /// config struct; it is looked up at request time.
    pub api_key_env: String,
    pub base_url: String,
    pub timeout_secs: u64,
    pub max_tokens: u32,
    pub temperature: f32,
}

impl Default for ClaudeHttpConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_CLAUDE_HTTP_MODEL.to_string(),
            api_key_env: DEFAULT_CLAUDE_HTTP_API_KEY_ENV.to_string(),
            base_url: DEFAULT_CLAUDE_HTTP_BASE_URL.to_string(),
            timeout_secs: DEFAULT_CLAUDE_HTTP_TIMEOUT_SECS,
            max_tokens: DEFAULT_CLAUDE_HTTP_MAX_TOKENS,
            temperature: DEFAULT_CLAUDE_HTTP_TEMPERATURE,
        }
    }
}

impl ClaudeHttpConfig {
    fn from_raw(raw: &RawClaudeHttpConfig) -> Result<Self> {
        let model = raw
            .model
            .clone()
            .unwrap_or_else(|| DEFAULT_CLAUDE_HTTP_MODEL.to_string());
        if model.trim().is_empty() {
            return Err(anyhow!(
                "[plan.llm.claude_http] model must be a non-empty string"
            ));
        }
        let api_key_env = raw
            .api_key_env
            .clone()
            .unwrap_or_else(|| DEFAULT_CLAUDE_HTTP_API_KEY_ENV.to_string());
        if api_key_env.trim().is_empty() {
            return Err(anyhow!(
                "[plan.llm.claude_http] api_key_env must be a non-empty environment variable name"
            ));
        }
        let base_url = raw
            .base_url
            .clone()
            .unwrap_or_else(|| DEFAULT_CLAUDE_HTTP_BASE_URL.to_string());
        if !(base_url.starts_with("http://") || base_url.starts_with("https://")) {
            return Err(anyhow!(
                "[plan.llm.claude_http] base_url must start with http:// or https:// (got `{base_url}`)"
            ));
        }
        let timeout_secs = raw.timeout_secs.unwrap_or(DEFAULT_CLAUDE_HTTP_TIMEOUT_SECS);
        if timeout_secs == 0 {
            return Err(anyhow!("[plan.llm.claude_http] timeout_secs must be > 0"));
        }
        let max_tokens = raw.max_tokens.unwrap_or(DEFAULT_CLAUDE_HTTP_MAX_TOKENS);
        if max_tokens == 0 {
            return Err(anyhow!("[plan.llm.claude_http] max_tokens must be > 0"));
        }
        let temperature = raw.temperature.unwrap_or(DEFAULT_CLAUDE_HTTP_TEMPERATURE);
        if !(0.0..=2.0).contains(&temperature) || !temperature.is_finite() {
            return Err(anyhow!(
                "[plan.llm.claude_http] temperature must be a finite value in [0.0, 2.0] (got {temperature})"
            ));
        }
        Ok(Self {
            model,
            api_key_env,
            base_url,
            timeout_secs,
            max_tokens,
            temperature,
        })
    }
}

// ---------------------------------------------------------------------------
// `[plan.llm]` (top-level + the optional `[plan.llm.refine]` escalation)
// ---------------------------------------------------------------------------

#[derive(Debug, Default, Clone, Deserialize)]
struct RawLlmConfig {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    claude_cli: Option<RawClaudeCliConfig>,
    #[serde(default)]
    mlx: Option<RawMlxConfig>,
    #[serde(default)]
    claude_http: Option<RawClaudeHttpConfig>,
    #[serde(default)]
    refine: Option<RawRefineConfig>,
}

/// `[plan.llm.refine]` mirrors `[plan.llm]` but cannot itself nest a
/// further refine block (we reject that during validation).
#[derive(Debug, Default, Clone, Deserialize)]
struct RawRefineConfig {
    #[serde(default)]
    provider: Option<String>,
    #[serde(default)]
    claude_cli: Option<RawClaudeCliConfig>,
    #[serde(default)]
    mlx: Option<RawMlxConfig>,
    #[serde(default)]
    claude_http: Option<RawClaudeHttpConfig>,
    /// Reserved so an accidental `[plan.llm.refine.refine]` block produces
    /// a clear validation error instead of being silently ignored.
    #[serde(default)]
    refine: Option<toml::Value>,
}

/// Resolved `[plan.llm]` section. Always carries fully-populated
/// per-provider subsections (defaulted) so the daemon can switch
/// `provider` at runtime via `compiler_override` without re-parsing.
#[derive(Debug, Clone, PartialEq)]
pub struct LlmConfig {
    pub provider: LlmProvider,
    pub claude_cli: ClaudeCliConfig,
    pub mlx: MlxConfig,
    pub claude_http: ClaudeHttpConfig,
    /// Optional escalation tier consulted only by `plan_refine`.
    /// When `None`, `refine()` routes through the primary provider.
    pub refine: Option<RefineConfig>,
}

impl Default for LlmConfig {
    fn default() -> Self {
        Self {
            provider: LlmProvider::ClaudeCli,
            claude_cli: ClaudeCliConfig::default(),
            mlx: MlxConfig::default(),
            claude_http: ClaudeHttpConfig::default(),
            refine: None,
        }
    }
}

impl LlmConfig {
    fn from_raw(raw: &RawLlmConfig) -> Result<Self> {
        let provider = match raw.provider.as_deref() {
            Some(s) => LlmProvider::try_from_str(s)?,
            None => LlmProvider::ClaudeCli,
        };
        let claude_cli = match raw.claude_cli.as_ref() {
            Some(r) => ClaudeCliConfig::from_raw(r)?,
            None => ClaudeCliConfig::default(),
        };
        let mlx = match raw.mlx.as_ref() {
            Some(r) => MlxConfig::from_raw(r)?,
            None => MlxConfig::default(),
        };
        let claude_http = match raw.claude_http.as_ref() {
            Some(r) => ClaudeHttpConfig::from_raw(r)?,
            None => ClaudeHttpConfig::default(),
        };
        let refine = match raw.refine.as_ref() {
            Some(r) => Some(RefineConfig::from_raw(r)?),
            None => None,
        };
        Ok(Self {
            provider,
            claude_cli,
            mlx,
            claude_http,
            refine,
        })
    }
}

/// Resolved `[plan.llm.refine]` escalation tier.
///
/// Same shape as [`LlmConfig`] minus the recursive `refine` field --
/// nesting a refine inside a refine is rejected at parse time so the
/// dispatch story stays one layer deep.
#[derive(Debug, Clone, PartialEq)]
pub struct RefineConfig {
    pub provider: LlmProvider,
    pub claude_cli: ClaudeCliConfig,
    pub mlx: MlxConfig,
    pub claude_http: ClaudeHttpConfig,
}

impl RefineConfig {
    fn from_raw(raw: &RawRefineConfig) -> Result<Self> {
        if raw.refine.is_some() {
            return Err(anyhow!(
                "[plan.llm.refine] may not nest a further `refine = ...` block; \
                 escalation is one layer deep by design"
            ));
        }
        let provider = match raw.provider.as_deref() {
            Some(s) => LlmProvider::try_from_str(s)?,
            None => LlmProvider::ClaudeCli,
        };
        let claude_cli = match raw.claude_cli.as_ref() {
            Some(r) => ClaudeCliConfig::from_raw(r)?,
            None => ClaudeCliConfig::default(),
        };
        let mlx = match raw.mlx.as_ref() {
            Some(r) => MlxConfig::from_raw(r)?,
            None => MlxConfig::default(),
        };
        let claude_http = match raw.claude_http.as_ref() {
            Some(r) => ClaudeHttpConfig::from_raw(r)?,
            None => ClaudeHttpConfig::default(),
        };
        Ok(Self {
            provider,
            claude_cli,
            mlx,
            claude_http,
        })
    }
}

// ---------------------------------------------------------------------------
// `[plan]` + top-level
// ---------------------------------------------------------------------------

/// `[plan]` section.
#[derive(Debug, Default, Clone, Deserialize)]
struct RawPlanConfig {
    #[serde(default)]
    max_questions: Option<usize>,
    #[serde(default)]
    compiler: Option<String>,
    #[serde(default)]
    llm: Option<RawLlmConfig>,
    #[serde(default)]
    step_timeout_secs: Option<u64>,
}

/// Resolved `[plan]` section, all fields populated.
#[derive(Debug, Clone)]
pub struct PlanConfig {
    pub max_questions: usize,
    pub compiler: CompilerSelection,
    /// Settings consumed only when `compiler == Llm`. Always populated
    /// (defaults applied) so the daemon can log them on startup
    /// regardless of selection.
    pub llm: LlmConfig,
    /// Daemon-wide fallback timeout applied to steps whose per-step
    /// `timeout_secs` is unset. `Some(n)` installs an `n`-second bound;
    /// `None` leaves such steps unbounded (the pre-HL behaviour). An
    /// operator explicitly writing `step_timeout_secs = 0` resolves to
    /// `None` -- the escape hatch for deployments that need to disable
    /// the fallback without picking an arbitrarily large number.
    pub step_timeout_secs: Option<u64>,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_questions: DEFAULT_MAX_QUESTIONS,
            compiler: CompilerSelection::Stub,
            llm: LlmConfig::default(),
            step_timeout_secs: Some(DEFAULT_STEP_TIMEOUT_SECS),
        }
    }
}

impl PlanConfig {
    fn from_raw(raw: &RawPlanConfig) -> Result<Self> {
        let max_questions = raw.max_questions.unwrap_or(DEFAULT_MAX_QUESTIONS);
        if max_questions == 0 {
            return Err(anyhow!(
                "[plan] max_questions must be > 0 (got 0); use a large number to disable budget enforcement"
            ));
        }
        let compiler = match raw.compiler.as_deref() {
            Some(s) => CompilerSelection::try_from_str(s)?,
            None => CompilerSelection::Stub,
        };
        let llm = match raw.llm.as_ref() {
            Some(r) => LlmConfig::from_raw(r)?,
            None => LlmConfig::default(),
        };
        // An absent key keeps the 1-hour default; an explicit `0`
        // disables the fallback so operators have an escape hatch
        // without picking an arbitrarily large number.
        let step_timeout_secs = match raw.step_timeout_secs {
            None => Some(DEFAULT_STEP_TIMEOUT_SECS),
            Some(0) => None,
            Some(s) => Some(s),
        };
        Ok(Self {
            max_questions,
            compiler,
            llm,
            step_timeout_secs,
        })
    }
}

/// Top-level on-disk shape. Every section is optional.
#[derive(Debug, Default, Deserialize)]
struct RawFlowdConfig {
    #[serde(default)]
    plan: RawPlanConfig,
}

/// Resolved daemon / CLI configuration.
///
/// Constructed via [`FlowdConfig::load`] (fallible -- file I/O + parse)
/// or [`FlowdConfig::default`] (infallible -- ships defaults without
/// touching disk; useful for tests).
#[derive(Debug, Default, Clone)]
pub struct FlowdConfig {
    pub plan: PlanConfig,
}

impl FlowdConfig {
    /// Path the daemon and CLI consult by default.
    #[must_use]
    pub fn default_path(home: &Path) -> PathBuf {
        home.join("flowd.toml")
    }

    /// Load the config from `path`. A missing file is treated as
    /// "use defaults" rather than an error so a fresh installation
    /// works without ever running an `init` command.
    ///
    /// # Errors
    /// Returns an error when the file exists but cannot be read,
    /// parsed, or validated (e.g. unknown compiler value).
    pub fn load(path: &Path) -> Result<Self> {
        match std::fs::read_to_string(path) {
            Ok(s) => {
                let raw: RawFlowdConfig = toml::from_str(&s)
                    .with_context(|| format!("parse flowd config: {}", path.display()))?;
                Ok(Self {
                    plan: PlanConfig::from_raw(&raw.plan)
                        .with_context(|| format!("validate flowd config: {}", path.display()))?,
                })
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => {
                Err(anyhow::Error::new(e).context(format!("read flowd config: {}", path.display())))
            }
        }
    }

    /// Path the project-scoped resolver consults: `<project_root>/.flowd/flowd.toml`.
    ///
    /// The `.flowd/` directory is the same one that already houses
    /// project-local rules, so a project that has opted into rules can
    /// adopt config without introducing a second marker.
    // Allowed-dead until the daemon entry point switches from the
    // global `$FLOWD_HOME/flowd.toml` load to project-scoped
    // resolution; the contract + tests land first so wiring follow-ups
    // can rely on it.
    #[allow(dead_code)]
    #[must_use]
    pub fn project_path(project_root: &Path) -> PathBuf {
        project_root.join(".flowd").join("flowd.toml")
    }

    /// Resolve the effective runtime config for an MCP / project session.
    ///
    /// Contract:
    ///
    /// * If `project_root` is `Some` and `<project_root>/.flowd/flowd.toml`
    ///   exists, that file is the **only** source of overrides for this
    ///   project. The per-machine `$FLOWD_HOME/flowd.toml` is never
    ///   layered or inherited -- `$FLOWD_HOME` stays a state home for
    ///   the socket, memory DB, models, and global rules.
    /// * Otherwise (no project root, or the project file is absent) the
    ///   result is `FlowdConfig::default()`.
    ///
    /// A project file that is present but malformed surfaces as an
    /// error, matching `FlowdConfig::load`'s loud-validation policy --
    /// silently falling back to defaults on a typo would mask the
    /// operator's intent.
    ///
    /// # Errors
    /// Propagates parse / validation errors from `FlowdConfig::load`.
    #[allow(dead_code)] // see note on `project_path`
    pub fn resolve(project_root: Option<&Path>) -> Result<Self> {
        let Some(root) = project_root else {
            return Ok(Self::default());
        };
        let path = Self::project_path(root);
        if !path.exists() {
            return Ok(Self::default());
        }
        Self::load(&path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_cfg(contents: &str) -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, contents).unwrap();
        (dir, p)
    }

    #[test]
    fn missing_file_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FlowdConfig::load(&dir.path().join("absent.toml")).unwrap();
        assert_eq!(cfg.plan.max_questions, DEFAULT_MAX_QUESTIONS);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
        assert_eq!(cfg.plan.llm, LlmConfig::default());
        // The default LLM provider is claude-cli (auth-free local CLI).
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeCli);
        assert_eq!(cfg.plan.llm.claude_cli.model, DEFAULT_CLAUDE_CLI_MODEL);
        assert_eq!(cfg.plan.llm.mlx.model, DEFAULT_MLX_MODEL);
        assert!(cfg.plan.llm.refine.is_none());
    }

    #[test]
    fn empty_table_keeps_all_defaults() {
        let (_d, p) = write_cfg("[plan]\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.max_questions, DEFAULT_MAX_QUESTIONS);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
        assert_eq!(cfg.plan.llm, LlmConfig::default());
    }

    #[test]
    fn explicit_overrides_are_honoured() {
        let (_d, p) = write_cfg("[plan]\nmax_questions = 4\ncompiler = \"rejecting\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.max_questions, 4);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Rejecting);
    }

    #[test]
    fn unknown_compiler_value_fails_loudly() {
        let (_d, p) = write_cfg("[plan]\ncompiler = \"gpt\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not a known compiler"), "{msg}");
    }

    #[test]
    fn llm_compiler_value_resolves_with_claude_cli_default() {
        let (_d, p) = write_cfg("[plan]\ncompiler = \"llm\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.compiler, CompilerSelection::Llm);
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeCli);
        assert_eq!(cfg.plan.llm.claude_cli.model, DEFAULT_CLAUDE_CLI_MODEL);
        assert_eq!(
            cfg.plan.llm.claude_cli.binary,
            PathBuf::from(DEFAULT_CLAUDE_CLI_BINARY)
        );
        // The mlx subsection still applies its defaults so a later
        // compiler_override = "mlx" works without re-parsing the file.
        assert_eq!(cfg.plan.llm.mlx.model, DEFAULT_MLX_MODEL);
        assert_eq!(cfg.plan.llm.mlx.base_url, DEFAULT_MLX_BASE_URL);
    }

    #[test]
    fn zero_max_questions_is_rejected() {
        let (_d, p) = write_cfg("[plan]\nmax_questions = 0\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("max_questions"));
    }

    // -------- [plan] step_timeout_secs -- fallback bound contract -------
    //
    // The fallback timeout is load-bearing: prose-compiled plans never
    // populate per-step `timeout_secs`, so PlanConfig::step_timeout_secs
    // is the only thing standing between a wedged step and an indefinite
    // block on its execution layer. Pin the three contract points here
    // so a refactor that flips a default or drops the explicit-zero
    // escape hatch trips a test rather than silently changing prod
    // behaviour.

    #[test]
    fn step_timeout_missing_key_resolves_to_default_fallback() {
        // No file at all -> defaults; the daemon-wide fallback applies.
        let dir = tempfile::tempdir().unwrap();
        let cfg = FlowdConfig::load(&dir.path().join("absent.toml")).unwrap();
        assert_eq!(cfg.plan.step_timeout_secs, Some(DEFAULT_STEP_TIMEOUT_SECS));

        // Present `[plan]` with the key omitted -> same default; an
        // operator overriding `max_questions` must not lose the bound.
        let (_d, p) = write_cfg("[plan]\nmax_questions = 4\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.step_timeout_secs, Some(DEFAULT_STEP_TIMEOUT_SECS));
    }

    #[test]
    fn step_timeout_zero_disables_the_fallback() {
        // `0` is the documented escape hatch -- it resolves to `None`
        // (unbounded) rather than a 0-second timeout that would fire
        // immediately. Operators who need to disable the bound rely on
        // this without picking an arbitrarily large number.
        let (_d, p) = write_cfg("[plan]\nstep_timeout_secs = 0\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.step_timeout_secs, None);
    }

    #[test]
    fn step_timeout_explicit_value_is_honoured() {
        let (_d, p) = write_cfg("[plan]\nstep_timeout_secs = 3600\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.step_timeout_secs, Some(3600));
    }

    #[test]
    fn default_path_lives_under_home() {
        let p = FlowdConfig::default_path(Path::new("/tmp/flowd"));
        assert_eq!(p, PathBuf::from("/tmp/flowd/flowd.toml"));
    }

    #[test]
    fn compiler_wire_names_are_stable() {
        assert_eq!(CompilerSelection::Stub.wire_name(), "stub");
        assert_eq!(CompilerSelection::Rejecting.wire_name(), "rejecting");
        assert_eq!(CompilerSelection::Llm.wire_name(), "llm");
    }

    // -------- [plan.llm] -- provider selection --------------------------

    #[test]
    fn provider_kebab_and_snake_case_are_both_accepted() {
        let (_d, p) = write_cfg("[plan.llm]\nprovider = \"claude_cli\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeCli);

        let (_d, p) = write_cfg("[plan.llm]\nprovider = \"claude-http\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeHttp);

        let (_d, p) = write_cfg("[plan.llm]\nprovider = \"mlx\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::Mlx);
    }

    #[test]
    fn unknown_provider_fails_loudly() {
        let (_d, p) = write_cfg("[plan.llm]\nprovider = \"openai\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("not a known provider"));
    }

    // -------- [plan.llm.claude_cli] -------------------------------------

    #[test]
    fn claude_cli_subsection_overrides_are_honoured() {
        let (_d, p) = write_cfg(
            "[plan.llm]\nprovider = \"claude-cli\"\n\n\
             [plan.llm.claude_cli]\n\
             model        = \"opus\"\n\
             binary       = \"/usr/local/bin/claude\"\n\
             timeout_secs = 90\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeCli);
        assert_eq!(cfg.plan.llm.claude_cli.model, "opus");
        assert_eq!(
            cfg.plan.llm.claude_cli.binary,
            PathBuf::from("/usr/local/bin/claude")
        );
        assert_eq!(cfg.plan.llm.claude_cli.timeout_secs, 90);
    }

    #[test]
    fn claude_cli_zero_timeout_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\ntimeout_secs = 0\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("timeout_secs"));
    }

    #[test]
    fn claude_cli_empty_binary_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\nbinary = \"\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("binary"));
    }

    #[test]
    fn claude_cli_default_model_is_opus() {
        // Belt-and-braces: the default exists for a reason (see the
        // doc on DEFAULT_CLAUDE_CLI_MODEL), and a silent flip back to
        // sonnet would change real users' bills, so pin it here.
        let cfg = FlowdConfig::default();
        assert_eq!(cfg.plan.llm.claude_cli.model, "opus");
        assert_eq!(DEFAULT_CLAUDE_CLI_MODEL, "opus");
    }

    #[test]
    fn claude_cli_effort_defaults_to_none() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert!(cfg.plan.llm.claude_cli.effort.is_none());
    }

    #[test]
    fn claude_cli_effort_canonical_token_resolves() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\neffort = \"high\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.claude_cli.effort, Some(ClaudeEffort::High));
    }

    #[test]
    fn claude_cli_effort_alias_token_resolves() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\neffort = \"extra-high\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.claude_cli.effort, Some(ClaudeEffort::Xhigh));
    }

    #[test]
    fn claude_cli_effort_unknown_token_is_rejected_with_section_pointer() {
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\neffort = \"turbo\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("[plan.llm.claude_cli]"), "{msg}");
        assert!(msg.contains("turbo"), "{msg}");
        assert!(msg.contains("\"high\""), "{msg}");
    }

    #[test]
    fn claude_cli_effort_empty_string_resolves_to_none() {
        // An empty string is what serde produces for `effort = ""` --
        // we treat it the same as the key being absent so operators
        // can blank-out a setting without commenting the whole line.
        let (_d, p) = write_cfg("[plan.llm.claude_cli]\neffort = \"\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        assert!(cfg.plan.llm.claude_cli.effort.is_none());
    }

    #[test]
    fn claude_cli_effort_round_trips_through_refine_block() {
        let (_d, p) = write_cfg(
            "[plan.llm.refine]\nprovider = \"claude-cli\"\n\n\
             [plan.llm.refine.claude_cli]\n\
             model  = \"claude-opus-4-7\"\n\
             effort = \"max\"\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        let refine = cfg.plan.llm.refine.unwrap();
        assert_eq!(refine.claude_cli.model, "claude-opus-4-7");
        assert_eq!(refine.claude_cli.effort, Some(ClaudeEffort::Max));
    }

    // -------- [plan.llm.mlx] -- carries over from the old flat layout ----

    #[test]
    fn mlx_subsection_overrides_are_honoured() {
        let (_d, p) = write_cfg(
            "[plan.llm]\nprovider = \"mlx\"\n\n\
             [plan.llm.mlx]\n\
             model        = \"my/custom-model\"\n\
             base_url     = \"http://localhost:9999/v1\"\n\
             timeout_secs = 30\n\
             max_tokens   = 1024\n\
             temperature  = 0.5\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::Mlx);
        assert_eq!(cfg.plan.llm.mlx.model, "my/custom-model");
        assert_eq!(cfg.plan.llm.mlx.base_url, "http://localhost:9999/v1");
        assert_eq!(cfg.plan.llm.mlx.timeout_secs, 30);
        assert_eq!(cfg.plan.llm.mlx.max_tokens, 1024);
        assert!((cfg.plan.llm.mlx.temperature - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn mlx_zero_timeout_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.mlx]\ntimeout_secs = 0\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("timeout_secs"));
    }

    #[test]
    fn mlx_zero_max_tokens_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.mlx]\nmax_tokens = 0\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("max_tokens"));
    }

    #[test]
    fn mlx_temperature_out_of_range_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.mlx]\ntemperature = 5.0\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("temperature"));
    }

    #[test]
    fn mlx_invalid_base_url_scheme_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.mlx]\nbase_url = \"localhost:11434\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("http://"));
    }

    // -------- [plan.llm.claude_http] ------------------------------------

    #[test]
    fn claude_http_subsection_overrides_are_honoured() {
        let (_d, p) = write_cfg(
            "[plan.llm]\nprovider = \"claude-http\"\n\n\
             [plan.llm.claude_http]\n\
             model        = \"claude-haiku-4-5\"\n\
             api_key_env  = \"CLAUDE_API_KEY\"\n\
             base_url     = \"https://api.example.com\"\n\
             timeout_secs = 30\n\
             max_tokens   = 2048\n\
             temperature  = 0.0\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeHttp);
        assert_eq!(cfg.plan.llm.claude_http.model, "claude-haiku-4-5");
        assert_eq!(cfg.plan.llm.claude_http.api_key_env, "CLAUDE_API_KEY");
        assert_eq!(cfg.plan.llm.claude_http.base_url, "https://api.example.com");
    }

    #[test]
    fn claude_http_invalid_base_url_scheme_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.claude_http]\nbase_url = \"api.anthropic.com\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("http"));
    }

    #[test]
    fn claude_http_empty_api_key_env_is_rejected() {
        let (_d, p) = write_cfg("[plan.llm.claude_http]\napi_key_env = \"\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("api_key_env"));
    }

    // -------- [plan.llm.refine] -- two-tier escalation -------------------

    #[test]
    fn refine_block_resolves_with_independent_provider() {
        let (_d, p) = write_cfg(
            "[plan.llm]\nprovider = \"mlx\"\n\n\
             [plan.llm.refine]\nprovider = \"claude-cli\"\n\n\
             [plan.llm.refine.claude_cli]\n\
             model        = \"opus\"\n\
             timeout_secs = 240\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::Mlx);
        let refine = cfg.plan.llm.refine.expect("refine block parsed");
        assert_eq!(refine.provider, LlmProvider::ClaudeCli);
        assert_eq!(refine.claude_cli.model, "opus");
        assert_eq!(refine.claude_cli.timeout_secs, 240);
    }

    #[test]
    fn refine_block_with_no_overrides_uses_provider_defaults() {
        let (_d, p) = write_cfg("[plan.llm.refine]\nprovider = \"claude-cli\"\n");
        let cfg = FlowdConfig::load(&p).unwrap();
        let refine = cfg.plan.llm.refine.unwrap();
        assert_eq!(refine.provider, LlmProvider::ClaudeCli);
        assert_eq!(refine.claude_cli.model, DEFAULT_CLAUDE_CLI_MODEL);
    }

    #[test]
    fn refine_block_rejects_nested_refine() {
        let (_d, p) = write_cfg(
            "[plan.llm.refine]\nprovider = \"claude-cli\"\nrefine = \"recursion-not-allowed\"\n",
        );
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("nest a further"));
    }

    #[test]
    fn refine_block_invalid_provider_is_rejected_with_section_pointer() {
        let (_d, p) = write_cfg("[plan.llm.refine]\nprovider = \"openai\"\n");
        let err = FlowdConfig::load(&p).unwrap_err();
        assert!(format!("{err:#}").contains("not a known provider"));
    }

    // -------- Wire names are stable across Anthropic CLI / docs ----------

    #[test]
    fn llm_provider_wire_names_are_stable() {
        assert_eq!(LlmProvider::ClaudeCli.wire_name(), "claude-cli");
        assert_eq!(LlmProvider::Mlx.wire_name(), "mlx");
        assert_eq!(LlmProvider::ClaudeHttp.wire_name(), "claude-http");
    }

    // -------- FlowdConfig::resolve -- project-scoped resolution contract -----
    //
    // Each test below isolates state in its own `tempfile::TempDir`,
    // touches no environment variables, and never changes `cwd`, so
    // `cargo test` can execute them on parallel worker threads without
    // serialising on shared state. The `within_deadline` helper bounds
    // every resolution: pure file I/O on a tempdir should finish in
    // milliseconds, so a regression that introduces blocking work
    // surfaces as a fast test failure rather than a hung CI job.
    mod resolve {
        use super::*;
        use std::path::Path;
        use std::sync::mpsc;
        use std::thread;
        use std::time::Duration;

        const RESOLVE_DEADLINE: Duration = Duration::from_secs(5);

        fn within_deadline<T, F>(f: F) -> T
        where
            F: FnOnce() -> T + Send + 'static,
            T: Send + 'static,
        {
            let (tx, rx) = mpsc::channel();
            thread::spawn(move || {
                let _ = tx.send(f());
            });
            match rx.recv_timeout(RESOLVE_DEADLINE) {
                Ok(v) => v,
                Err(mpsc::RecvTimeoutError::Timeout) => {
                    panic!("FlowdConfig::resolve exceeded {RESOLVE_DEADLINE:?}")
                }
                Err(mpsc::RecvTimeoutError::Disconnected) => {
                    panic!("FlowdConfig::resolve worker panicked (see thread output)")
                }
            }
        }

        fn write_project_config(root: &Path, body: &str) {
            let dir = root.join(".flowd");
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(dir.join("flowd.toml"), body).unwrap();
        }

        #[test]
        fn project_path_composition_is_stable() {
            let p = FlowdConfig::project_path(Path::new("/tmp/proj"));
            assert_eq!(p, PathBuf::from("/tmp/proj/.flowd/flowd.toml"));
        }

        #[test]
        fn no_project_root_returns_defaults() {
            let cfg = within_deadline(|| FlowdConfig::resolve(None).unwrap());
            let defaults = FlowdConfig::default();
            assert_eq!(cfg.plan.max_questions, defaults.plan.max_questions);
            assert_eq!(cfg.plan.compiler, defaults.plan.compiler);
        }

        #[test]
        fn missing_project_file_returns_defaults() {
            let dir = tempfile::tempdir().unwrap();
            let root = dir.path().to_path_buf();
            let cfg = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap());
            let defaults = FlowdConfig::default();
            assert_eq!(cfg.plan.max_questions, defaults.plan.max_questions);
            assert_eq!(cfg.plan.compiler, defaults.plan.compiler);
        }

        #[test]
        fn flowd_dir_without_toml_returns_defaults() {
            // `.flowd/` may exist for rules even when no `flowd.toml` is
            // authored. The resolver must still treat that as "no
            // project config" rather than erroring on the directory.
            let dir = tempfile::tempdir().unwrap();
            std::fs::create_dir_all(dir.path().join(".flowd")).unwrap();
            let root = dir.path().to_path_buf();
            let cfg = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap());
            assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
        }

        #[test]
        fn present_project_file_overrides_defaults() {
            let dir = tempfile::tempdir().unwrap();
            write_project_config(
                dir.path(),
                "[plan]\nmax_questions = 7\ncompiler = \"rejecting\"\n",
            );
            let root = dir.path().to_path_buf();
            let cfg = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap());
            assert_eq!(cfg.plan.max_questions, 7);
            assert_eq!(cfg.plan.compiler, CompilerSelection::Rejecting);
        }

        #[test]
        fn malformed_project_file_surfaces_loud_error() {
            let dir = tempfile::tempdir().unwrap();
            write_project_config(dir.path(), "[plan]\ncompiler = \"gpt\"\n");
            let root = dir.path().to_path_buf();
            let err = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap_err());
            let msg = format!("{err:#}");
            assert!(msg.contains("not a known compiler"), "{msg}");
        }

        // The two invariants below are the load-bearing assertions of
        // this contract: `$FLOWD_HOME/flowd.toml` must never bleed into
        // a project's effective config, in either branch of the
        // resolver. We materialise a deliberately-conflicting
        // `$FLOWD_HOME/flowd.toml` in a *separate* tempdir and prove
        // that none of its values reach the resolved config.

        #[test]
        fn flowd_home_is_not_layered_when_project_file_present() {
            let project = tempfile::tempdir().unwrap();
            let home = tempfile::tempdir().unwrap();

            write_project_config(project.path(), "[plan]\nmax_questions = 3\n");
            std::fs::write(
                home.path().join("flowd.toml"),
                "[plan]\nmax_questions = 99\ncompiler = \"rejecting\"\n",
            )
            .unwrap();

            let root = project.path().to_path_buf();
            let cfg = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap());

            // Project-local override wins for `max_questions`, and the
            // compiler stays at the project's *implicit* default
            // (`Stub`) -- not the home file's `Rejecting`. If layering
            // ever sneaks in, the compiler assertion catches it.
            assert_eq!(cfg.plan.max_questions, 3);
            assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
        }

        #[test]
        fn flowd_home_is_not_layered_when_project_file_absent() {
            let project = tempfile::tempdir().unwrap();
            let home = tempfile::tempdir().unwrap();
            std::fs::write(
                home.path().join("flowd.toml"),
                "[plan]\nmax_questions = 99\ncompiler = \"rejecting\"\n",
            )
            .unwrap();

            let root = project.path().to_path_buf();
            let cfg = within_deadline(move || FlowdConfig::resolve(Some(&root)).unwrap());
            let defaults = FlowdConfig::default();
            assert_eq!(cfg.plan.max_questions, defaults.plan.max_questions);
            assert_eq!(cfg.plan.compiler, defaults.plan.compiler);
        }
    }
}
