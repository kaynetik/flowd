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
//! model        = "claude-opus-4-7"
//! binary       = "claude"   # path or executable name resolved on $PATH
//! timeout_secs = 120
//!
//! # Subsection consumed when provider = "mlx".
//! [plan.llm.mlx]
//! model        = "qwen3-coder:30b"
//! base_url     = "http://127.0.0.1:11434/v1"
//! timeout_secs = 60
//! max_tokens   = 4096
//! temperature  = 0.2
//!
//! # Subsection consumed when provider = "claude-http".
//! [plan.llm.claude_http]
//! model        = "claude-opus-4-7"
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
//! model        = "claude-opus-4-7-thinking-high"
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
use serde::Deserialize;

/// Default question budget. Matches the value documented in the
/// HL-44 spec; chosen to be large enough for typical refactors yet
/// small enough that runaway compilers visibly trip the warning.
pub const DEFAULT_MAX_QUESTIONS: usize = 12;

// -- Claude CLI defaults ----------------------------------------------------

/// Default model id passed to the local `claude` CLI when
/// `provider = "claude-cli"`. The CLI accepts any model the operator's
/// Anthropic plan exposes; we pin Opus 4.7 because plan compilation
/// needs strong instruction-following on JSON shape.
pub const DEFAULT_CLAUDE_CLI_MODEL: &str = "claude-opus-4-7";

/// Default `binary` for the Claude CLI backend. Resolved on `$PATH`
/// when the value contains no path separator; treated as a literal
/// path otherwise.
pub const DEFAULT_CLAUDE_CLI_BINARY: &str = "claude";

/// Default per-request timeout for the Claude CLI shell-out. Opus 4.7
/// has variable latency depending on prompt size and reasoning depth;
/// 120s leaves headroom while still failing fast enough to be useful
/// in an interactive loop.
pub const DEFAULT_CLAUDE_CLI_TIMEOUT_SECS: u64 = 120;

// -- MLX defaults -----------------------------------------------------------

/// Default model id for the MLX (OpenAI-compatible) backend. Matches the
/// model we recommend in the docs (qwen3-coder:30b via ollama or
/// `mlx_lm.server`); operators who run a different model just override
/// `[plan.llm.mlx].model`.
pub const DEFAULT_MLX_MODEL: &str = "qwen3-coder:30b";

/// Default `base_url` for the OpenAI-compatible MLX backend. Matches
/// the port `ollama` and `mlx_lm.server` listen on by default.
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
pub const DEFAULT_CLAUDE_HTTP_MODEL: &str = "claude-opus-4-7";

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
/// `ClaudeCli` is the default and the recommended path: it shells out to
/// the local `claude` CLI for auth-free Anthropic access (plan compilation
/// is quality-sensitive and Opus 4.7 is the strongest model we can route
/// through without managing secrets). `Mlx` is the offline fallback.
/// `ClaudeHttp` calls the Anthropic Messages API directly with an
/// operator-supplied API key.
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
}

/// Resolved `[plan.llm.claude_cli]` block.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaudeCliConfig {
    pub model: String,
    /// Either a bare executable name (resolved on `$PATH` at startup) or
    /// an absolute / relative path to the binary.
    pub binary: PathBuf,
    pub timeout_secs: u64,
}

impl Default for ClaudeCliConfig {
    fn default() -> Self {
        Self {
            model: DEFAULT_CLAUDE_CLI_MODEL.to_string(),
            binary: PathBuf::from(DEFAULT_CLAUDE_CLI_BINARY),
            timeout_secs: DEFAULT_CLAUDE_CLI_TIMEOUT_SECS,
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
        Ok(Self {
            model,
            binary,
            timeout_secs,
        })
    }
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
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_questions: DEFAULT_MAX_QUESTIONS,
            compiler: CompilerSelection::Stub,
            llm: LlmConfig::default(),
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
        Ok(Self {
            max_questions,
            compiler,
            llm,
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
        // The default LLM provider is claude-cli (Opus quality-first).
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
             model        = \"claude-sonnet-4-6\"\n\
             binary       = \"/usr/local/bin/claude\"\n\
             timeout_secs = 90\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeCli);
        assert_eq!(cfg.plan.llm.claude_cli.model, "claude-sonnet-4-6");
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
             model        = \"claude-sonnet-4-6\"\n\
             api_key_env  = \"CLAUDE_API_KEY\"\n\
             base_url     = \"https://api.example.com\"\n\
             timeout_secs = 30\n\
             max_tokens   = 2048\n\
             temperature  = 0.0\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::ClaudeHttp);
        assert_eq!(cfg.plan.llm.claude_http.model, "claude-sonnet-4-6");
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
             model        = \"claude-opus-4-7-thinking-high\"\n\
             timeout_secs = 240\n",
        );
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.llm.provider, LlmProvider::Mlx);
        let refine = cfg.plan.llm.refine.expect("refine block parsed");
        assert_eq!(refine.provider, LlmProvider::ClaudeCli);
        assert_eq!(refine.claude_cli.model, "claude-opus-4-7-thinking-high");
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
}
