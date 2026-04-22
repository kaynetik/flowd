//! Runtime [`PlanCompiler`] selection for the daemon.
//!
//! `flowd-mcp` ships several concrete compilers ([`StubPlanCompiler`],
//! [`RejectingPlanCompiler`], and [`LlmPlanCompiler`]).
//! `FlowdHandlers` is generic over the compiler so each compiler stays
//! statically dispatched in tests, but the daemon needs to pick one at
//! startup based on `flowd.toml`. The trait uses `impl Future` returns
//! which means a plain `Arc<dyn PlanCompiler>` is impossible; we
//! enum-wrap the concrete implementations and forward each method
//! by hand instead.
//!
//! The enum lives in the CLI crate (rather than `flowd-mcp`) on
//! purpose: it owns the deployment policy ("which compiler ships in
//! the binary"), and `flowd-mcp` should keep exposing the individual
//! compilers without picking one for callers.
//!
//! ## Two-tier dispatch
//!
//! When `compiler == Llm`, the wrapper hosts up to **two** independently
//! configured [`LlmPlanCompiler`]s:
//!
//! * `primary` -- handles `compile_prose` and `apply_answers`.
//! * `refine`  -- optional escalation tier from `[plan.llm.refine]`;
//!   handles `refine` only. Falls back to `primary` when unset, so
//!   config without a refine block keeps the old single-tier behaviour.
//!
//! Both tiers are built eagerly so a startup probe (e.g. for the
//! `claude` CLI binary) covers every transport that will run during
//! the daemon's lifetime. A small registry of *override* compilers --
//! one per provider configured in `[plan.llm.*]` -- is also built
//! eagerly to back the `compiler_override` field on `plan_create`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::{Answer, CompileOutput, PlanCompiler, PlanDraftSnapshot};
use flowd_mcp::{
    ClaudeCliCallback, ClaudeCliConfig as McpClaudeCliConfig, LlmCallback, LlmPlanCompiler,
    OpenAiCompatibleCallback, OpenAiCompatibleConfig, RejectingPlanCompiler, StubPlanCompiler,
};

use crate::config::{
    ClaudeCliConfig, ClaudeHttpConfig, CompilerSelection, LlmConfig, LlmProvider, MlxConfig,
    RefineConfig,
};

// ---------------------------------------------------------------------------
// DaemonLlmCallback -- runtime sum-type over every LlmCallback the daemon
//                      can host. The daemon binds `LlmPlanCompiler` to this
//                      single concrete type so a `compile_prose_with_override`
//                      call can pick a different transport at request time
//                      without needing `dyn` or per-variant compiler clones.
// ---------------------------------------------------------------------------

/// Runtime sum-type over the [`LlmCallback`] implementations the daemon
/// hosts.
///
/// `LlmCallback::complete` returns `impl Future`, which makes a plain
/// `Arc<dyn LlmCallback>` impossible. We enum-wrap the concrete callbacks
/// and unify their futures via `Pin<Box<dyn Future + Send>>` so a single
/// `LlmPlanCompiler<DaemonLlmCallback>` can host *either* the Claude CLI
/// transport *or* the OpenAI-compatible MLX transport.
///
/// `ClaudeHttp` is intentionally absent: the direct Anthropic Messages
/// HTTP path is not yet implemented (see `LlmRouter::build_callback`).
/// Configuring `provider = "claude-http"` while the variant is missing
/// surfaces a clear startup error instead of being routed to a stub.
#[derive(Debug, Clone)]
pub enum DaemonLlmCallback {
    ClaudeCli(ClaudeCliCallback),
    Mlx(OpenAiCompatibleCallback),
}

impl LlmCallback for DaemonLlmCallback {
    // Trait method uses 2024 RPIT capture which already binds `&self`;
    // adding `+ '_` here is a *narrowing* refinement (more permissive
    // for the caller) but the lint flags any return-type difference.
    // We're intentionally explicit so the boxing strategy is readable.
    #[allow(refining_impl_trait_reachable)]
    fn complete(
        &self,
        prompt: String,
    ) -> impl std::future::Future<Output = Result<String>> + Send + '_ {
        // Boxing into a single `Pin<Box<_>>` is the cheapest way to unify
        // the two arms' future types without introducing a `futures`
        // crate dependency. We borrow `&self` for the call duration; the
        // inner futures already own their config (each arm's
        // `complete()` clones eagerly), so the future itself does not
        // hold any reference once it is returned -- but we keep the
        // `'_` lifetime in the signature to satisfy the trait's
        // implicit `&self` capture in Rust 2024.
        let fut: std::pin::Pin<Box<dyn std::future::Future<Output = Result<String>> + Send + '_>> =
            match self {
                Self::ClaudeCli(cb) => Box::pin(cb.complete(prompt)),
                Self::Mlx(cb) => Box::pin(cb.complete(prompt)),
            };
        fut
    }
}

// ---------------------------------------------------------------------------
// LlmRouter -- two-tier (primary + refine) routing plus a small registry
//              of all configured backends to back `compiler_override`.
// ---------------------------------------------------------------------------

/// Routing layer for the LLM compiler.
///
/// Holds:
///
/// * `primary` -- handles `compile_prose` and `apply_answers`. This is
///   the backend selected by `[plan.llm].provider`.
/// * `refine` -- handles `refine` (escalation tier). When unset, the
///   wrapper forwards `refine` to `primary`, preserving the
///   single-tier behaviour.
/// * `overrides` -- table of all *configured* backends keyed by
///   [`LlmProvider`]. Lets `plan_create(compiler_override = "...")`
///   re-route the *initial* compile through a non-default backend
///   without spinning up a fresh transport per request.
///
/// Both `primary` and `refine` references point into the `overrides`
/// registry rather than holding their own `Arc`, so the daemon's banner
/// and any future introspection observes a single source of truth for
/// "which backends does this daemon know about".
pub struct LlmRouter {
    primary_provider: LlmProvider,
    refine_provider: Option<LlmProvider>,
    /// All backends built at startup, keyed by provider. Fully owns
    /// each compiler -- the `provider` fields above are just lookup keys.
    overrides: HashMap<LlmProvider, Arc<LlmPlanCompiler<DaemonLlmCallback>>>,
}

impl std::fmt::Debug for LlmRouter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LlmRouter")
            .field("primary_provider", &self.primary_provider)
            .field("refine_provider", &self.refine_provider)
            .field(
                "overrides",
                &self
                    .overrides
                    .keys()
                    .map(|p| p.wire_name())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl LlmRouter {
    /// Build the router from a fully-resolved [`LlmConfig`].
    ///
    /// Eagerly constructs:
    ///
    /// * The `primary` backend (always).
    /// * The `refine` backend (when `[plan.llm.refine]` is present).
    /// * A registry entry for *every* provider whose subsection is
    ///   present in the config so `compiler_override` lookups succeed
    ///   in O(1) without re-parsing or re-constructing transports.
    ///
    /// # Errors
    /// Returns `FlowdError::Internal` when any requested transport
    /// fails to construct (e.g. `reqwest::Client` build failure for the
    /// MLX backend), or `FlowdError::PlanValidation` when the config
    /// asks for an unimplemented provider (today: `claude-http`).
    pub fn build(cfg: &LlmConfig) -> Result<Self> {
        let mut overrides: HashMap<LlmProvider, Arc<LlmPlanCompiler<DaemonLlmCallback>>> =
            HashMap::new();

        // Build the primary unconditionally so the most common path
        // ("compile a fresh prose plan") always has a transport ready.
        overrides.insert(
            cfg.provider,
            Arc::new(build_compiler_for(cfg.provider, cfg)?),
        );

        // Pre-build the other two providers' backends so the override
        // registry is complete after one pass. Build failures for these
        // are swallowed at startup (logged) rather than aborting the
        // daemon: if the operator never invokes `compiler_override`,
        // the failure has no observable effect; if they do, the request
        // path returns the same error their startup probe would have.
        for provider in [
            LlmProvider::ClaudeCli,
            LlmProvider::Mlx,
            LlmProvider::ClaudeHttp,
        ] {
            if overrides.contains_key(&provider) {
                continue;
            }
            match build_compiler_for(provider, cfg) {
                Ok(c) => {
                    overrides.insert(provider, Arc::new(c));
                }
                Err(e) => {
                    // Skip the entry; the request-time path will surface
                    // the same error if anyone names this provider via
                    // `compiler_override`. Tracing as `debug` keeps the
                    // banner uncluttered for the common case.
                    tracing::debug!(
                        provider = provider.wire_name(),
                        error = %e,
                        "skipping unconfigured LLM backend in override registry"
                    );
                }
            }
        }

        // Refine tier overrides nothing in the registry; it just picks
        // a different key. Build it eagerly so any startup probe (e.g.
        // for a thinking-mode CLI) runs at startup, not at first
        // refine() call.
        let refine_provider = match cfg.refine.as_ref() {
            Some(refine) => {
                let c = build_compiler_for_refine(refine)?;
                // Replace the registry entry with the refine-specific
                // compiler so a `compiler_override` request that names
                // the same provider still routes through the same
                // (possibly stronger) model the operator configured.
                //
                // This is intentional: when an operator says "use
                // claude-cli with the thinking-high model for refine",
                // they almost certainly want the same upgrade for any
                // explicit override. If a deployment ever needs the
                // two to diverge we can split the registry, but YAGNI.
                overrides.insert(refine.provider, Arc::new(c));
                Some(refine.provider)
            }
            None => None,
        };

        Ok(Self {
            primary_provider: cfg.provider,
            refine_provider,
            overrides,
        })
    }

    /// Borrow the primary compiler (used for `compile_prose` /
    /// `apply_answers`). Always present.
    fn primary(&self) -> &LlmPlanCompiler<DaemonLlmCallback> {
        // Safe by construction: `build` always inserts the primary
        // provider's compiler into `overrides`.
        self.overrides
            .get(&self.primary_provider)
            .expect("LlmRouter invariant: primary backend present")
    }

    /// Borrow the refine compiler when a refine tier is configured,
    /// otherwise fall back to the primary so the call still succeeds.
    fn refine_or_primary(&self) -> &LlmPlanCompiler<DaemonLlmCallback> {
        let key = self.refine_provider.unwrap_or(self.primary_provider);
        self.overrides
            .get(&key)
            .expect("LlmRouter invariant: refine backend present when refine_provider is set")
    }

    /// Look up an override compiler by stable wire name.
    ///
    /// Returns a `PlanValidation` error (rather than `Internal`) so the
    /// MCP layer reports a 400-equivalent "unknown override" to the
    /// caller -- this is a user-controllable input, not a daemon bug.
    fn for_override(&self, wire_name: &str) -> Result<&LlmPlanCompiler<DaemonLlmCallback>> {
        let provider = parse_provider_wire_name(wire_name)?;
        self.overrides
            .get(&provider)
            .map(AsRef::as_ref)
            .ok_or_else(|| {
                FlowdError::PlanValidation(format!(
                    "compiler_override = `{wire_name}` is not configured at runtime; \
                 add a `[plan.llm.{}]` block to flowd.toml or pick a configured backend",
                    provider_subsection_name(provider)
                ))
            })
    }

    /// The provider that handles `compile_prose` / `apply_answers`.
    /// Used for the startup banner.
    #[must_use]
    pub fn primary_provider(&self) -> LlmProvider {
        self.primary_provider
    }

    /// The optional refine override provider. Used for the startup banner.
    #[must_use]
    pub fn refine_provider(&self) -> Option<LlmProvider> {
        self.refine_provider
    }

    /// Stable wire names of every configured override (sorted for
    /// deterministic banner output). Used by the startup banner.
    #[must_use]
    pub fn configured_overrides(&self) -> Vec<&'static str> {
        let mut v: Vec<&'static str> = self
            .overrides
            .keys()
            .copied()
            .map(LlmProvider::wire_name)
            .collect();
        v.sort_unstable();
        v
    }
}

/// Map a wire name (from `compiler_override`) to a [`LlmProvider`],
/// accepting both kebab-case and `snake_case` forms (mirrors the parser
/// in [`crate::config`]).
fn parse_provider_wire_name(name: &str) -> Result<LlmProvider> {
    match name.trim().to_ascii_lowercase().as_str() {
        "claude-cli" | "claude_cli" => Ok(LlmProvider::ClaudeCli),
        "mlx" => Ok(LlmProvider::Mlx),
        "claude-http" | "claude_http" => Ok(LlmProvider::ClaudeHttp),
        other => Err(FlowdError::PlanValidation(format!(
            "compiler_override = `{other}` is not a known LLM provider; \
             expected one of: \"claude-cli\", \"mlx\", \"claude-http\""
        ))),
    }
}

/// Stable subsection name used in error messages so the operator can
/// jump straight to the right TOML block.
fn provider_subsection_name(provider: LlmProvider) -> &'static str {
    match provider {
        LlmProvider::ClaudeCli => "claude_cli",
        LlmProvider::Mlx => "mlx",
        LlmProvider::ClaudeHttp => "claude_http",
    }
}

/// Build a compiler for `provider` using `cfg`'s primary subsections.
fn build_compiler_for(
    provider: LlmProvider,
    cfg: &LlmConfig,
) -> Result<LlmPlanCompiler<DaemonLlmCallback>> {
    match provider {
        LlmProvider::ClaudeCli => build_claude_cli(&cfg.claude_cli),
        LlmProvider::Mlx => build_mlx(&cfg.mlx),
        LlmProvider::ClaudeHttp => build_claude_http(&cfg.claude_http),
    }
}

/// Build a compiler for the refine tier using its dedicated subsections.
fn build_compiler_for_refine(refine: &RefineConfig) -> Result<LlmPlanCompiler<DaemonLlmCallback>> {
    match refine.provider {
        LlmProvider::ClaudeCli => build_claude_cli(&refine.claude_cli),
        LlmProvider::Mlx => build_mlx(&refine.mlx),
        LlmProvider::ClaudeHttp => build_claude_http(&refine.claude_http),
    }
}

// Returns `Result` for symmetry with `build_mlx` / `build_claude_http`
// so `build_compiler_for(_for_refine)` can dispatch through a single
// match arm. The CLI callback construction itself is infallible today.
#[allow(clippy::unnecessary_wraps)]
fn build_claude_cli(cfg: &ClaudeCliConfig) -> Result<LlmPlanCompiler<DaemonLlmCallback>> {
    let cb = ClaudeCliCallback::new(McpClaudeCliConfig {
        binary: cfg.binary.clone(),
        model: cfg.model.clone(),
        timeout: Duration::from_secs(cfg.timeout_secs),
    });
    Ok(LlmPlanCompiler::new(Arc::new(
        DaemonLlmCallback::ClaudeCli(cb),
    )))
}

fn build_mlx(cfg: &MlxConfig) -> Result<LlmPlanCompiler<DaemonLlmCallback>> {
    let api_cfg = OpenAiCompatibleConfig {
        base_url: cfg.base_url.clone(),
        model: cfg.model.clone(),
        timeout: Duration::from_secs(cfg.timeout_secs),
        max_tokens: cfg.max_tokens,
        temperature: cfg.temperature,
    };
    let cb = OpenAiCompatibleCallback::new(api_cfg)
        .map_err(|e| FlowdError::Internal(format!("build OpenAI-compatible callback: {e}")))?;
    Ok(LlmPlanCompiler::new(Arc::new(DaemonLlmCallback::Mlx(cb))))
}

/// Direct Anthropic Messages API HTTP path. Reserved for a follow-up
/// commit; for now the daemon refuses to wire this provider so the
/// failure is loud at startup rather than silent at request time.
fn build_claude_http(_cfg: &ClaudeHttpConfig) -> Result<LlmPlanCompiler<DaemonLlmCallback>> {
    Err(FlowdError::PlanValidation(
        "[plan.llm].provider = \"claude-http\" is recognised by the config layer \
         but the direct HTTP transport is not yet implemented. \
         Use \"claude-cli\" (local CLI shell-out, no API key needed) or \"mlx\" \
         (local OpenAI-compatible server) for now."
            .into(),
    ))
}

// ---------------------------------------------------------------------------
// DaemonPlanCompiler -- top-level enum that the handler is generic over.
// ---------------------------------------------------------------------------

/// Sum-type wrapper over every [`PlanCompiler`] the daemon can host.
///
/// One variant per [`CompilerSelection`]. The [`PlanCompiler`] impl
/// pattern-matches and forwards. The `Llm` variant carries an
/// [`LlmRouter`] that does its own primary/refine/override dispatch.
#[derive(Debug)]
pub enum DaemonPlanCompiler {
    Stub(StubPlanCompiler),
    Rejecting(RejectingPlanCompiler),
    Llm(LlmRouter),
}

impl DaemonPlanCompiler {
    /// Build the configured compiler. The `Stub` and `Rejecting`
    /// variants are infallible (zero-sized); the `Llm` variant builds
    /// every configured backend eagerly and can fail if the primary or
    /// refine transports cannot be constructed.
    ///
    /// # Errors
    /// Returns `FlowdError::PlanValidation` when the LLM provider is
    /// not implemented (claude-http today), or `FlowdError::Internal`
    /// when an HTTP client cannot be constructed.
    pub fn from_selection(selection: CompilerSelection, llm_cfg: &LlmConfig) -> Result<Self> {
        Ok(match selection {
            CompilerSelection::Stub => Self::Stub(StubPlanCompiler::new()),
            CompilerSelection::Rejecting => Self::Rejecting(RejectingPlanCompiler::new()),
            CompilerSelection::Llm => Self::Llm(LlmRouter::build(llm_cfg)?),
        })
    }

    /// Borrow the LLM router when this variant is `Llm`. Used by the
    /// startup banner to surface provider/model details.
    #[must_use]
    pub fn llm_router(&self) -> Option<&LlmRouter> {
        match self {
            Self::Llm(r) => Some(r),
            _ => None,
        }
    }
}

impl PlanCompiler for DaemonPlanCompiler {
    async fn compile_prose(&self, prose: String, project: String) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.compile_prose(prose, project).await,
            Self::Rejecting(c) => c.compile_prose(prose, project).await,
            Self::Llm(r) => r.primary().compile_prose(prose, project).await,
        }
    }

    async fn apply_answers(
        &self,
        snapshot: PlanDraftSnapshot,
        answers: Vec<(String, Answer)>,
        defer_remaining: bool,
    ) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.apply_answers(snapshot, answers, defer_remaining).await,
            Self::Rejecting(c) => c.apply_answers(snapshot, answers, defer_remaining).await,
            Self::Llm(r) => {
                r.primary()
                    .apply_answers(snapshot, answers, defer_remaining)
                    .await
            }
        }
    }

    async fn refine(&self, snapshot: PlanDraftSnapshot, feedback: String) -> Result<CompileOutput> {
        match self {
            Self::Stub(c) => c.refine(snapshot, feedback).await,
            Self::Rejecting(c) => c.refine(snapshot, feedback).await,
            // Refine routes through the escalation tier when configured.
            Self::Llm(r) => r.refine_or_primary().refine(snapshot, feedback).await,
        }
    }

    async fn compile_prose_with_override(
        &self,
        prose: String,
        project: String,
        compiler_override: Option<String>,
    ) -> Result<CompileOutput> {
        // No override + non-LLM variants: identical to compile_prose.
        let Some(name) = compiler_override
            .as_deref()
            .map(str::trim)
            .filter(|s| !s.is_empty())
        else {
            return self.compile_prose(prose, project).await;
        };
        match self {
            Self::Stub(_) | Self::Rejecting(_) => Err(FlowdError::PlanValidation(format!(
                "compiler_override = `{name}` is only meaningful when [plan].compiler = \"llm\"; \
                 the active compiler is `{}`",
                match self {
                    Self::Stub(_) => "stub",
                    Self::Rejecting(_) => "rejecting",
                    Self::Llm(_) => unreachable!(),
                }
            ))),
            Self::Llm(r) => r.for_override(name)?.compile_prose(prose, project).await,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::error::FlowdError;

    fn snapshot() -> PlanDraftSnapshot {
        PlanDraftSnapshot {
            plan_name: "p".into(),
            project: "proj".into(),
            source_doc: None,
            open_questions: vec![],
            decisions: vec![],
            previous_definition: None,
        }
    }

    fn default_llm_cfg() -> LlmConfig {
        LlmConfig::default()
    }

    #[tokio::test]
    async fn stub_variant_forwards_to_stub_parser() {
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Stub, &default_llm_cfg())
            .unwrap();
        let out = c
            .compile_prose("## a [agent: echo]\nhi\n".into(), "proj".into())
            .await
            .unwrap();
        assert!(out.definition.is_some());
    }

    #[tokio::test]
    async fn rejecting_variant_returns_disabled_error() {
        let c =
            DaemonPlanCompiler::from_selection(CompilerSelection::Rejecting, &default_llm_cfg())
                .unwrap();
        let err = c
            .compile_prose("anything".into(), "proj".into())
            .await
            .unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(m) if m.contains("disabled")));
        let err = c.refine(snapshot(), "x".into()).await.unwrap_err();
        assert!(matches!(err, FlowdError::PlanValidation(_)));
    }

    #[test]
    fn llm_variant_constructs_with_default_config() {
        // Smoke-only: confirms that wiring an `LlmRouter` over the default
        // (claude-cli primary, no refine) config does not blow up at
        // construction time. Network behaviour is verified by
        // `tests/llm_e2e.rs` (gated behind `--ignored`) and by the unit
        // tests in `flowd-mcp::compiler`.
        let c =
            DaemonPlanCompiler::from_selection(CompilerSelection::Llm, &default_llm_cfg()).unwrap();
        let router = c.llm_router().expect("Llm variant exposes a router");
        assert_eq!(router.primary_provider(), LlmProvider::ClaudeCli);
        assert_eq!(router.refine_provider(), None);
        // The MLX backend is also pre-built in the override registry so a
        // `compiler_override = "mlx"` plan_create call does not need to
        // construct anything at request time.
        assert!(router.configured_overrides().contains(&"mlx"));
        assert!(router.configured_overrides().contains(&"claude-cli"));
        // claude-http is intentionally absent (transport not implemented).
        assert!(!router.configured_overrides().contains(&"claude-http"));
    }

    #[tokio::test]
    async fn override_on_non_llm_variant_returns_validation_error() {
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Stub, &default_llm_cfg())
            .unwrap();
        let err = c
            .compile_prose_with_override("p".into(), "proj".into(), Some("mlx".into()))
            .await
            .unwrap_err();
        assert!(matches!(&err, FlowdError::PlanValidation(m) if m.contains("only meaningful")));
    }

    #[tokio::test]
    async fn override_with_unknown_provider_returns_validation_error() {
        let c =
            DaemonPlanCompiler::from_selection(CompilerSelection::Llm, &default_llm_cfg()).unwrap();
        let err = c
            .compile_prose_with_override("p".into(), "proj".into(), Some("openai".into()))
            .await
            .unwrap_err();
        assert!(matches!(&err, FlowdError::PlanValidation(m) if m.contains("not a known")));
    }

    #[tokio::test]
    async fn override_with_unconfigured_claude_http_returns_validation_error() {
        // claude-http transport is not implemented; the override entry is
        // therefore intentionally missing from the registry. Selecting it
        // explicitly should produce a clear "not configured" error.
        let c =
            DaemonPlanCompiler::from_selection(CompilerSelection::Llm, &default_llm_cfg()).unwrap();
        let err = c
            .compile_prose_with_override("p".into(), "proj".into(), Some("claude-http".into()))
            .await
            .unwrap_err();
        assert!(
            matches!(&err, FlowdError::PlanValidation(m) if m.contains("not configured at runtime"))
        );
    }

    #[tokio::test]
    async fn empty_or_whitespace_override_is_ignored() {
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Stub, &default_llm_cfg())
            .unwrap();
        // Stub accepts any structured markdown; an empty override should
        // route through compile_prose, so the same input parses fine.
        let out = c
            .compile_prose_with_override(
                "## a [agent: echo]\nhi\n".into(),
                "proj".into(),
                Some("   ".into()),
            )
            .await
            .unwrap();
        assert!(out.definition.is_some());
    }

    #[test]
    fn refine_block_promotes_alternative_provider() {
        // Build a config where the primary is mlx and refine escalates
        // to claude-cli. Verify the router exposes both keys correctly.
        let cfg = LlmConfig {
            provider: LlmProvider::Mlx,
            refine: Some(RefineConfig {
                provider: LlmProvider::ClaudeCli,
                claude_cli: ClaudeCliConfig::default(),
                mlx: MlxConfig::default(),
                claude_http: ClaudeHttpConfig::default(),
            }),
            ..LlmConfig::default()
        };
        let c = DaemonPlanCompiler::from_selection(CompilerSelection::Llm, &cfg).unwrap();
        let router = c.llm_router().unwrap();
        assert_eq!(router.primary_provider(), LlmProvider::Mlx);
        assert_eq!(router.refine_provider(), Some(LlmProvider::ClaudeCli));
    }

    #[test]
    fn parse_provider_wire_name_accepts_both_cases() {
        assert_eq!(
            parse_provider_wire_name("claude-cli").unwrap(),
            LlmProvider::ClaudeCli
        );
        assert_eq!(
            parse_provider_wire_name("CLAUDE_CLI").unwrap(),
            LlmProvider::ClaudeCli
        );
        assert_eq!(
            parse_provider_wire_name(" mlx  ").unwrap(),
            LlmProvider::Mlx
        );
        assert!(parse_provider_wire_name("openai").is_err());
    }
}
