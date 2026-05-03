//! Concrete `AgentSpawner` implementations for the `flowd` daemon.
//!
//! `flowd-core` does not spawn OS processes itself; that is the CLI's job.
//! This module wires the orchestration engine to a local agent CLI binary
//! (Claude Code, `cursor-agent`, `OpenAI` Codex, Aider, ...).
//!
//! ## Selection order
//!
//! When a [`LocalShellSpawner`] is built via [`LocalShellSpawner::detect`]:
//!
//! 1. `$FLOWD_AGENT_BIN` if set (operator override; absolute path or a name
//!    on `$PATH`).
//! 2. The first entry from [`DEFAULT_CANDIDATES`] found on `$PATH`.
//! 3. Otherwise `None`, so the caller can fall back to [`UnconfiguredSpawner`]
//!    which fails loudly on `spawn` rather than silently.
//!
//! ## Invocation
//!
//! Each step is invoked as
//!
//! ```text
//! <bin> [LocalShellSpawner::default_args()...] [extra_args...] <prompt_flag> "<step.prompt>"
//! ```
//!
//! `prompt_flag` defaults to `-p` (Claude Code, `cursor-agent`); override
//! it via `$FLOWD_AGENT_PROMPT_FLAG` for CLIs that use a different name
//! (e.g. `--message` for Aider). Extra leading arguments come from
//! `$FLOWD_AGENT_EXTRA_ARGS` (whitespace-split).
//!
//! ### Auto-injected defaults for Claude-like CLIs
//!
//! When the resolved binary's basename is `claude` or `cursor-agent`,
//! the spawner prepends three flags before any user-supplied
//! `extra_args`:
//!
//! - `--bare` -- skips the Node-side auto-discovery pass that
//!   otherwise loads `~/.claude`, project `CLAUDE.md`, MCP servers,
//!   hooks, plugins, skills, auto-memory, OAuth, and the OS
//!   keychain on every invocation. In headless dispatch we usually
//!   want none of that: the prompt comes from `step.prompt`, the
//!   workspace trust gate is already bypassed, and reproducibility
//!   matters more than convenience. Anthropic has stated `--bare`
//!   will become the `-p` default in a future release; we adopt
//!   it where we can.
//!
//!   `FLOWD_AGENT_CLAUDE_BARE` is tri-state:
//!     - `1` -- force ON (legacy default; pinned reproducibility).
//!     - `0` -- force OFF (full auto-discovery).
//!     - unset / anything else -- *Auto*: at construction time the
//!       spawner probes for a Claude Code credential source the
//!       upstream CLI can read on its own (macOS keychain entry
//!       `Claude Code-credentials` or `~/.claude/.credentials.json`
//!       on Linux). If credentials are reachable AND
//!       `ANTHROPIC_API_KEY` is unset, bare mode auto-disables so
//!       the spawned `claude` reuses the operator's existing
//!       OAuth/keychain auth. If `ANTHROPIC_API_KEY` is set, or
//!       no credential source is detected, bare mode stays ON.
//!
//!   Bare mode requires `ANTHROPIC_API_KEY` to be set in the
//!   environment (or an `apiKeyHelper` declared via `--settings`)
//!   because keychain lookup is skipped. The detection only checks
//!   for the *presence* of the keychain entry (via `security
//!   find-generic-password -s ...`); it never reads the secret, so
//!   no authorisation prompt is triggered.
//! - `--dangerously-skip-permissions` -- headless `-p` mode cannot
//!   prompt the operator for tool-use approval. Without this flag,
//!   every `Write` / `Edit` / `Bash` invocation by the spawned step
//!   is silently denied while the model still exits 0 with a
//!   chatty "I cannot do this" reply. The daemon would then record
//!   `step_completed` against an empty working tree -- the worst
//!   possible failure (green status, zero work).
//! - `--output-format=stream-json` (paired with the mandatory
//!   `--verbose`) -- emits one NDJSON event per line. The leading
//!   `system/init` event carries `session_id`, which the daemon
//!   captures and threads into the next step's `--resume` so
//!   prompt cache and conversation context survive across steps
//!   in the same plan. The final `result` event carries the same
//!   `is_error`, `permission_denials`, `usage`, `total_cost_usd`,
//!   and `result` fields that single-blob `--output-format=json`
//!   used to emit, so the spawner can still detect post-skip
//!   refusals (the model has policy gates beyond the
//!   workspace-trust prompt) and API errors that exit code 0
//!   cannot signal. Intermediate events (assistant deltas, tool
//!   uses, retries) are logged at `tracing::debug!` and
//!   otherwise discarded; future PRs may surface them through
//!   the plan observer.
//!
//! Operators who need the opposite (sandbox runs, audit replays)
//! must wrap the binary -- e.g. point `$FLOWD_AGENT_BIN` at a
//! shell script that strips the flags. The `--bare` injection is
//! the only one with an env opt-out (`FLOWD_AGENT_CLAUDE_BARE=0`);
//! the other two are non-negotiable because the default has to
//! make the common case (headless dispatch) safe.
//!
//! Stdout is captured into [`AgentOutput::stdout`]. For Claude-like
//! bins the JSON envelope is parsed and the human-readable `result`
//! field is what downstream consumers see; the raw JSON is logged at
//! `tracing::trace!` only. The same envelope is also mined for token
//! counts and USD cost (top-level `usage` + `total_cost_usd`, plus
//! the per-model `modelUsage` block) and surfaced as
//! [`AgentMetrics`] on the returned [`AgentOutput`]. Metrics are
//! captured on both the success path *and* the failure paths
//! (`is_error: true`, lingering `permission_denials`); the failure
//! variants attach them to [`FlowdError::PlanExecution`] so an
//! expensive refusal still lands in the audit log with its true
//! spend instead of flying under the radar. Stderr is emitted via
//! `tracing` (warn on non-zero exit, debug otherwise) so it surfaces
//! in the daemon log without polluting the JSON-RPC channel.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::ffi::{OsStr, OsString};
use std::path::{Path, PathBuf};
use std::process::Stdio;
use std::sync::{Arc, Mutex};

use flowd_core::error::{FlowdError, Result};
use flowd_core::orchestration::executor::{
    AgentMetrics, AgentSpawnContext, AgentSpawner, ModelUsage,
};
use flowd_core::orchestration::{AgentOutput, Plan, PlanPreview, PlanStep};
use flowd_storage::step_branch_store::{SqliteStepBranchStore, StepBranchRecord};
use serde::Deserialize;
use tokio::process::Command;

/// Default candidate binaries searched on `$PATH` when `$FLOWD_AGENT_BIN`
/// is unset. Order matters: first match wins.
const DEFAULT_CANDIDATES: &[&str] = &["claude", "cursor-agent"];
// Temporarily disable codex and aider until we have a way to test them.
// const DEFAULT_CANDIDATES: &[&str] = &["claude", "codex", "aider", "cursor-agent"];

/// Default flag passed before the prompt argument. Suitable for Claude Code
/// and `cursor-agent`; override via `$FLOWD_AGENT_PROMPT_FLAG` for others.
const DEFAULT_PROMPT_FLAG: &str = "-p";

/// Returns the basename of `bin` lowercased, with `.exe` stripped on
/// Windows-style names. Used to gate Claude-specific behavior on the
/// resolved binary regardless of its directory.
fn bin_basename(bin: &OsStr) -> Option<String> {
    let name = std::path::Path::new(bin).file_name()?.to_str()?;
    let trimmed = name.strip_suffix(".exe").unwrap_or(name);
    Some(trimmed.to_ascii_lowercase())
}

/// Whether the resolved spawner binary is a Claude Code-flavored CLI
/// (Anthropic's `claude` or `cursor-agent`). These share the
/// `-p` / `--dangerously-skip-permissions` / `--output-format=json`
/// surface; other CLIs (`codex`, `aider`) have their own conventions
/// and are left alone.
fn is_claude_like(bin: &OsStr) -> bool {
    matches!(
        bin_basename(bin).as_deref(),
        Some("claude" | "cursor-agent")
    )
}

/// Build the full argv (excluding the binary itself) for one spawn.
///
/// Order is:
///
/// 1. `--resume <id>` when `resume_id` is `Some` (the caller has
///    already filtered to claude-like binaries; passing a foreign
///    `resume_id` is a programmer error).
/// 2. The `injected` defaults (e.g. `--bare`, the stream-json pair).
/// 3. The operator-supplied `extra_args` -- Claude's last-flag-wins
///    parsing means putting these AFTER our defaults lets operators
///    override an injected flag without us having to know which
///    flags can be overridden.
/// 4. `prompt_flag prompt`.
///
/// Pure / no I/O so the test suite can pin the exact placement of
/// `--resume` without spawning a real process. Centralising this
/// logic is the only safe way to keep the resume flag's position
/// from drifting under future refactors of `spawn_in_cwd`.
fn assemble_args(
    resume_id: Option<&str>,
    injected: &[&str],
    extra_args: &[String],
    prompt_flag: &str,
    prompt: &str,
) -> Vec<String> {
    // Capacity is exact for the common path (no resume) and 2-over for
    // the resume path -- not worth a precise sum.
    let mut out = Vec::with_capacity(injected.len() + extra_args.len() + 4);
    if let Some(sid) = resume_id {
        out.push("--resume".to_owned());
        out.push(sid.to_owned());
    }
    out.extend(injected.iter().map(|s| (*s).to_owned()));
    out.extend(extra_args.iter().cloned());
    out.push(prompt_flag.to_owned());
    out.push(prompt.to_owned());
    out
}

/// Env var Anthropic looks for when bare mode is active and OAuth/keychain
/// are skipped. Used by [`LocalShellSpawner::detect`] both for the
/// auto-decision and the one-shot warning -- the spawner never reads
/// the value during the spawn hot path.
const ENV_ANTHROPIC_API_KEY: &str = "ANTHROPIC_API_KEY";

/// Tri-state operator override for the `--bare` injection.
///
/// `Auto` means "let `LocalShellSpawner::detect` decide based on
/// available auth": if Claude Code's OAuth/keychain credentials are
/// reachable from the daemon's user, default OFF (parity with running
/// `claude -p` directly); otherwise default ON (the
/// reproducibility-favouring legacy default).
///
/// `On` and `Off` are explicit operator overrides; the resolver never
/// re-evaluates them against the environment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeBareToggle {
    Auto,
    On,
    Off,
}

/// Pure parser for the `FLOWD_AGENT_CLAUDE_BARE` env var. Only the
/// literal strings `"0"` and `"1"` flip the toggle to a hard
/// `Off` / `On`; anything else (unset, empty, `"true"`, typos)
/// resolves to `Auto` so the credential probe gets the final say.
///
/// Backward compatibility: the previous parser treated *anything but
/// `"0"`* as ON. Operators who relied on that to force ON (the legacy
/// default behaviour for unset env) should now set the value to `"1"`
/// explicitly. Stray non-empty values that previously mapped to ON
/// now map to Auto, which on a credential-equipped workstation also
/// resolves to OFF -- a deliberate convenience improvement, not a
/// silent regression: a missing key path still surfaces via the
/// `ANTHROPIC_API_KEY` warning below.
fn parse_claude_bare_toggle(raw: Option<&str>) -> ClaudeBareToggle {
    match raw {
        Some("0") => ClaudeBareToggle::Off,
        Some("1") => ClaudeBareToggle::On,
        _ => ClaudeBareToggle::Auto,
    }
}

/// Read the toggle from the live process environment. Called once per
/// spawner construction; the hot path stays env-free.
fn read_claude_bare_toggle() -> ClaudeBareToggle {
    parse_claude_bare_toggle(std::env::var("FLOWD_AGENT_CLAUDE_BARE").ok().as_deref())
}

/// End-to-end live resolution of the `--bare` flag from the current
/// process environment. Reads the toggle, probes for credentials only
/// when the answer can swing the decision, and applies
/// [`resolve_claude_bare`]. Logging is suppressed -- live callers go
/// through [`LocalShellSpawner::detect`] which decides when to log.
fn live_claude_bare() -> bool {
    let toggle = read_claude_bare_toggle();
    let env_api_key_set = std::env::var_os(ENV_ANTHROPIC_API_KEY).is_some_and(|v| !v.is_empty());
    let creds = if matches!(toggle, ClaudeBareToggle::Auto) && !env_api_key_set {
        detect_claude_credentials()
    } else {
        ClaudeCredentialState::Missing
    };
    resolve_claude_bare(toggle, creds, env_api_key_set)
}

/// Outcome of the Claude Code credential probe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ClaudeCredentialState {
    /// A credential source the upstream Claude Code CLI can read on
    /// its own (macOS keychain entry or `~/.claude/.credentials.json`)
    /// is reachable. Auto-resolution prefers OFF in this case so the
    /// daemon-spawned `claude` behaves the same as an interactive run.
    Available,
    /// No credential source detectable from this process. Auto stays
    /// ON, which forces the operator to provide `ANTHROPIC_API_KEY`
    /// (or an `apiKeyHelper` via `--settings`) -- the legacy default.
    Missing,
}

/// Probe for a Claude Code credential source the upstream CLI would
/// pick up on its own. Synchronous and fail-closed: any unexpected
/// error (missing `security` binary, broken FS, permission denials)
/// yields `Missing` so the resolver keeps the safer legacy default.
///
/// Platform layout (matches what the upstream CLI looks for):
/// - macOS: `security find-generic-password -s "Claude Code-credentials"`.
///   We only inspect the exit code -- never the secret itself -- so
///   this never prompts the user with an authorisation dialog.
/// - Other platforms: `~/.claude/.credentials.json` (the cross-platform
///   fallback Claude Code uses where keychain integration isn't
///   available).
fn detect_claude_credentials() -> ClaudeCredentialState {
    #[cfg(target_os = "macos")]
    {
        // `security find-generic-password` exits 0 when the entry
        // exists, regardless of whether the caller has access to
        // its data. We pass `/dev/null` for stdout/stderr so the
        // probe never pollutes the daemon log with keychain noise.
        let probe = std::process::Command::new("security")
            .args(["find-generic-password", "-s", "Claude Code-credentials"])
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status();
        if matches!(probe, Ok(s) if s.success()) {
            return ClaudeCredentialState::Available;
        }
    }

    if let Some(home) = std::env::var_os("HOME") {
        let creds = PathBuf::from(home)
            .join(".claude")
            .join(".credentials.json");
        if creds.is_file() {
            return ClaudeCredentialState::Available;
        }
    }

    ClaudeCredentialState::Missing
}

/// Resolve the operator toggle plus environment signals into the
/// final boolean handed to the spawner. Pure -- so the unit test
/// suite can exhaustively pin the policy without mutating the
/// process environment or mocking the keychain probe.
///
/// The policy:
/// - `On` and `Off` always win (explicit operator intent).
/// - `Auto` + creds available + no `ANTHROPIC_API_KEY`: default OFF
///   so OAuth/keychain authenticated workstations work without
///   ceremony.
/// - `Auto` + `ANTHROPIC_API_KEY` set: default ON. The operator
///   has provided bare-mode-compatible auth, so we keep the
///   reproducibility / speed benefits of bare mode.
/// - `Auto` + no creds + no env key: default ON, matching the
///   legacy behaviour. The downstream warning still fires because
///   neither auth source is present, telling the operator how to
///   recover.
fn resolve_claude_bare(
    toggle: ClaudeBareToggle,
    creds: ClaudeCredentialState,
    env_api_key_set: bool,
) -> bool {
    match toggle {
        ClaudeBareToggle::On => true,
        ClaudeBareToggle::Off => false,
        ClaudeBareToggle::Auto => {
            if env_api_key_set {
                true
            } else {
                !matches!(creds, ClaudeCredentialState::Available)
            }
        }
    }
}

/// Emit a one-shot tracing event explaining why bare mode is on or off.
/// Split out from [`LocalShellSpawner::detect`] so the resolver test
/// suite stays pure -- and so the format lives next to the policy it
/// describes.
fn log_claude_bare_decision(
    toggle: ClaudeBareToggle,
    creds: ClaudeCredentialState,
    env_api_key_set: bool,
    claude_bare: bool,
) {
    let auto = matches!(toggle, ClaudeBareToggle::Auto);
    let creds_available = matches!(creds, ClaudeCredentialState::Available);

    if claude_bare && !env_api_key_set && !creds_available {
        tracing::warn!(
            "Claude bare mode is ON but neither ANTHROPIC_API_KEY nor a \
             local Claude Code credential source is detectable. Bare mode \
             skips OAuth and the OS keychain, so spawned `claude` \
             invocations will fail unless an apiKeyHelper is provided via \
             `--settings` (in FLOWD_AGENT_EXTRA_ARGS). Set ANTHROPIC_API_KEY \
             or export FLOWD_AGENT_CLAUDE_BARE=0 to disable bare mode."
        );
        return;
    }

    if auto {
        if claude_bare {
            tracing::info!(
                "Claude bare mode auto-enabled (ANTHROPIC_API_KEY is set). \
                 Override with FLOWD_AGENT_CLAUDE_BARE=0."
            );
        } else {
            tracing::info!(
                "Claude bare mode auto-disabled (local Claude Code \
                 credentials detected, ANTHROPIC_API_KEY unset). \
                 Spawned `claude` will reuse the operator's OAuth/keychain \
                 auth. Override with FLOWD_AGENT_CLAUDE_BARE=1."
            );
        }
    }
}

/// Subset of Claude Code's final `result` event payload (in
/// `stream-json`) -- equivalent to the single-blob envelope
/// `--output-format=json` used to emit. Every field is
/// `#[serde(default)]` so the parser tolerates schema drift from
/// upstream Claude releases -- missing fields degrade to "no signal",
/// never "decode error".
#[derive(Debug, Default, Deserialize)]
struct ClaudeJsonEnvelope {
    /// True when the run terminated in an error state (API failure,
    /// model error, etc.). Independent of process exit code.
    #[serde(default)]
    is_error: bool,
    /// Human-readable model output. What downstream
    /// [`AgentOutput::stdout`] should carry.
    #[serde(default)]
    result: String,
    /// Tools the model wanted to use that were denied. Should be
    /// empty when `--dangerously-skip-permissions` is in effect; a
    /// non-empty value means a deeper policy gate (Anthropic-side
    /// safety filter, MCP server refusal, etc.) blocked the work.
    /// Treated as a hard failure so the audit log does not lie.
    #[serde(default)]
    permission_denials: Vec<ClaudePermissionDenial>,
    /// Aggregate token counts for the whole run (`snake_case` upstream).
    #[serde(default)]
    usage: ClaudeUsage,
    /// Aggregate USD cost for the run.
    #[serde(default)]
    total_cost_usd: f64,
    /// Wall-clock duration of the run in milliseconds.
    #[serde(default)]
    duration_ms: u64,
    /// Cumulative time spent in Anthropic API calls, in milliseconds.
    #[serde(default)]
    duration_api_ms: u64,
    /// Per-model breakdown, keyed by model name. Note the camelCase
    /// key (`modelUsage`) and camelCase inner fields -- the rest of
    /// the envelope is `snake_case`, so upstream is inconsistent and
    /// this one block needs explicit `rename`s.
    #[serde(default, rename = "modelUsage")]
    model_usage: BTreeMap<String, ClaudeModelUsage>,
    /// Conversation handle. Both the `system/init` event (always
    /// first) and the `result` event (always last) carry one, and
    /// they are the same value for a given run. The init event is
    /// authoritative and captured separately by the stream parser;
    /// keeping the field here too means a stream that omits init
    /// (e.g. a non-bare CLI without `--verbose`, or a future schema
    /// drift) still surfaces *some* session id from the result
    /// event rather than none.
    #[serde(default)]
    session_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ClaudePermissionDenial {
    #[serde(default)]
    tool_name: String,
}

/// Top-level `usage` block. Upstream uses `snake_case` here, so no
/// per-field renames are needed.
#[derive(Debug, Default, Deserialize)]
#[allow(clippy::struct_field_names)]
struct ClaudeUsage {
    #[serde(default)]
    input_tokens: u64,
    #[serde(default)]
    output_tokens: u64,
    #[serde(default)]
    cache_creation_input_tokens: u64,
    #[serde(default)]
    cache_read_input_tokens: u64,
}

/// Per-model entry under `modelUsage`. Upstream emits camelCase keys
/// here (unlike the top-level `usage` block), so every field needs
/// an explicit rename.
#[derive(Debug, Default, Deserialize)]
struct ClaudeModelUsage {
    #[serde(default, rename = "inputTokens")]
    input_tokens: u64,
    #[serde(default, rename = "outputTokens")]
    output_tokens: u64,
    #[serde(default, rename = "cacheCreationInputTokens")]
    cache_creation_input_tokens: u64,
    #[serde(default, rename = "cacheReadInputTokens")]
    cache_read_input_tokens: u64,
    #[serde(default, rename = "costUSD")]
    cost_usd: f64,
}

/// Single conversion site from the raw Claude envelope to the
/// transport-agnostic [`AgentMetrics`] type consumed by the rest of
/// the daemon. Keeping this as the only translation point means
/// schema-drift fixes land in one place.
impl From<&ClaudeJsonEnvelope> for AgentMetrics {
    fn from(e: &ClaudeJsonEnvelope) -> Self {
        let model_usage = e
            .model_usage
            .iter()
            .map(|(name, u)| {
                (
                    name.clone(),
                    ModelUsage {
                        input_tokens: u.input_tokens,
                        output_tokens: u.output_tokens,
                        cache_creation_input_tokens: u.cache_creation_input_tokens,
                        cache_read_input_tokens: u.cache_read_input_tokens,
                        cost_usd: u.cost_usd,
                    },
                )
            })
            .collect();
        Self {
            input_tokens: e.usage.input_tokens,
            output_tokens: e.usage.output_tokens,
            cache_creation_input_tokens: e.usage.cache_creation_input_tokens,
            cache_read_input_tokens: e.usage.cache_read_input_tokens,
            total_cost_usd: e.total_cost_usd,
            duration_ms: e.duration_ms,
            duration_api_ms: e.duration_api_ms,
            model_usage,
        }
    }
}

/// `AgentSpawner` that shells out to a local CLI binary with the step's
/// prompt.
#[derive(Debug, Clone)]
pub struct LocalShellSpawner {
    bin: OsString,
    prompt_flag: String,
    cwd: Option<PathBuf>,
    extra_args: Vec<String>,
    /// Whether to prepend `--bare` for Claude-like bins. Resolved once at
    /// construction time from `FLOWD_AGENT_CLAUDE_BARE` so per-spawn cost
    /// stays env-free. Has no effect for non-Claude-like bins.
    claude_bare: bool,
}

impl LocalShellSpawner {
    /// Build a spawner from environment variables, or return `None` if no
    /// usable agent CLI is discoverable.
    #[must_use]
    pub fn detect() -> Option<Self> {
        let bin = std::env::var_os("FLOWD_AGENT_BIN")
            .filter(|s| !s.is_empty())
            .or_else(|| {
                DEFAULT_CANDIDATES
                    .iter()
                    .find_map(|name| which_in_path(name).map(OsString::from))
            })?;

        let prompt_flag = std::env::var("FLOWD_AGENT_PROMPT_FLAG")
            .ok()
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| DEFAULT_PROMPT_FLAG.to_owned());

        let extra_args = std::env::var("FLOWD_AGENT_EXTRA_ARGS")
            .ok()
            .filter(|s| !s.is_empty())
            .map(|s| s.split_whitespace().map(str::to_owned).collect())
            .unwrap_or_default();

        let cwd = std::env::var_os("FLOWD_AGENT_CWD")
            .filter(|s| !s.is_empty())
            .map(PathBuf::from);

        let toggle = read_claude_bare_toggle();
        let env_api_key_set =
            std::env::var_os(ENV_ANTHROPIC_API_KEY).is_some_and(|v| !v.is_empty());

        // Probe the keychain / credentials file only when we might
        // actually need the answer: a Claude-like bin in `Auto` mode
        // without a process-level API key. This keeps the keychain
        // probe off the hot path for non-Claude bins and for operators
        // who pinned the toggle explicitly.
        let claude_like = is_claude_like(&bin);
        let creds = if claude_like && matches!(toggle, ClaudeBareToggle::Auto) && !env_api_key_set {
            detect_claude_credentials()
        } else {
            ClaudeCredentialState::Missing
        };

        let claude_bare = resolve_claude_bare(toggle, creds, env_api_key_set);

        // One-shot diagnostic so operators understand why they're in
        // bare or non-bare mode. Tracing level is chosen by severity:
        // - `info` for an Auto-resolved decision (informational, the
        //   common path on a fresh dev box).
        // - `warn` when bare is ON but neither auth source is present;
        //   the spawned `claude` is going to fail at run time and the
        //   operator needs to act.
        if claude_like {
            log_claude_bare_decision(toggle, creds, env_api_key_set, claude_bare);
        }

        Some(Self {
            bin,
            prompt_flag,
            cwd,
            extra_args,
            claude_bare,
        })
    }

    /// Construct an explicit spawner. Used by tests; not gated on `cfg(test)`
    /// so embedders that depend on the crate can call it directly.
    ///
    /// The `claude_bare` toggle is read from `FLOWD_AGENT_CLAUDE_BARE` here
    /// too -- programmatic embedders that want a different default should
    /// build the struct directly or set the env var before calling.
    #[must_use]
    #[allow(dead_code)]
    pub fn new(bin: impl Into<OsString>) -> Self {
        Self {
            bin: bin.into(),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: None,
            extra_args: Vec::new(),
            claude_bare: live_claude_bare(),
        }
    }

    /// Display the configured binary; for logging only.
    #[must_use]
    pub fn bin_display(&self) -> std::path::Display<'_> {
        std::path::Path::new(&self.bin).display()
    }

    /// Flags prepended automatically before any user-supplied
    /// `extra_args`. Only Claude-like bins receive injected flags;
    /// everything else gets an empty slice. The exact set is gated on
    /// `self.claude_bare` so operators can opt out of the bare-mode
    /// speedup without touching the binary path.
    ///
    /// `--output-format=stream-json` plus the mandatory `--verbose`
    /// pair makes the CLI emit one NDJSON event per line: a leading
    /// `system/init` (carrying `session_id`), a stream of
    /// `assistant`/`user`/`rate_limit_event` events, and a final
    /// `result` event with the same payload the old single-blob
    /// `--output-format=json` produced. We need the `init` event to
    /// capture `session_id` for cross-step resume; `--verbose` is
    /// mandatory because Claude omits the init event without it.
    ///
    /// See the module docstring for why each flag is here.
    fn default_args(&self) -> &'static [&'static str] {
        if !is_claude_like(&self.bin) {
            return &[];
        }
        if self.claude_bare {
            &[
                "--bare",
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        } else {
            &[
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        }
    }

    fn effective_cwd(&self) -> Result<PathBuf> {
        match &self.cwd {
            Some(cwd) => Ok(cwd.clone()),
            None => std::env::current_dir().map_err(|e| FlowdError::PlanExecution {
                message: format!("resolve current directory for agent spawner: {e}"),
                metrics: None,
            }),
        }
    }

    async fn spawn_in_cwd(
        &self,
        step: &PlanStep,
        cwd_override: Option<&Path>,
        prior_session_id: Option<&str>,
    ) -> Result<AgentOutput> {
        let claude_like = is_claude_like(&self.bin);
        let injected = self.default_args();
        // Pure arg assembly is factored out so the test suite can pin
        // the exact ordering (resume flag -> default flags -> operator
        // extras -> prompt) without spawning a real process.
        let resume_id = prior_session_id.filter(|_| claude_like);
        let assembled = assemble_args(
            resume_id,
            injected,
            &self.extra_args,
            &self.prompt_flag,
            &step.prompt,
        );

        let mut cmd = Command::new(&self.bin);
        for arg in &assembled {
            cmd.arg(arg);
        }
        if let Some(cwd) = cwd_override {
            cmd.current_dir(cwd);
        } else if let Some(cwd) = &self.cwd {
            cmd.current_dir(cwd);
        }
        // Detach stdin so an agent that tries to prompt the operator errors
        // immediately instead of blocking on a closed terminal.
        cmd.stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // kill_on_drop ensures the executor's `JoinHandle::abort` (used by
        // `cancel`) actually terminates the spawned process tree.
        cmd.kill_on_drop(true);

        tracing::info!(
            step_id = %step.id,
            agent_type = %step.agent_type,
            bin = %self.bin_display(),
            cwd = ?cwd_override.or(self.cwd.as_deref()),
            injected_args = ?injected,
            resume = ?resume_id,
            claude_like,
            "spawning agent step"
        );

        if claude_like {
            self.spawn_claude_streaming(step, cmd).await
        } else {
            self.spawn_simple(step, cmd).await
        }
    }

    /// Non-Claude-like spawners (cursor-agent's text mode, aider, codex,
    /// shells used as test fixtures) pre-date stream-json and have no
    /// `session_id` concept. Keep the simple "wait for exit, capture
    /// stdout, passthrough" path so we don't regress them.
    async fn spawn_simple(&self, step: &PlanStep, mut cmd: Command) -> Result<AgentOutput> {
        let output = cmd.output().await.map_err(|e| FlowdError::PlanExecution {
            message: format!(
                "failed to spawn agent `{bin}` for step `{step}`: {e}",
                bin = self.bin_display(),
                step = step.id,
            ),
            metrics: None,
        })?;

        let raw_stdout = String::from_utf8_lossy(&output.stdout).into_owned();
        let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
        let code = output.status.code();

        if !output.status.success() {
            tracing::warn!(
                step_id = %step.id,
                exit_code = ?code,
                stderr = %truncate_for_log(&stderr, 4096),
                "agent step exited non-zero"
            );
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "step `{step}` failed (exit={code}): {stderr}",
                    step = step.id,
                    code = code.map_or_else(|| "signal".into(), |c| c.to_string()),
                    stderr = truncate_for_log(stderr.trim(), 1024),
                ),
                metrics: None,
            });
        }

        if !stderr.is_empty() {
            tracing::debug!(
                step_id = %step.id,
                stderr = %truncate_for_log(&stderr, 2048),
                "agent step stderr"
            );
        }

        Ok(AgentOutput {
            stdout: raw_stdout,
            exit_code: code,
            metrics: None,
            session_id: None,
        })
    }

    /// Spawn a Claude-like CLI in `--output-format=stream-json` mode.
    ///
    /// Drains stdout line-by-line through [`parse_claude_stream`]
    /// while concurrently draining stderr into a buffer for logging.
    /// On a successful exit we still need a `result` event in the
    /// stream; absence is treated as a hard failure (matching the
    /// upstream bug pattern in `claude-code#1920` where a missing
    /// `result` would otherwise hang or silently report success
    /// against zero work).
    async fn spawn_claude_streaming(
        &self,
        step: &PlanStep,
        mut cmd: Command,
    ) -> Result<AgentOutput> {
        let mut child = cmd.spawn().map_err(|e| FlowdError::PlanExecution {
            message: format!(
                "failed to spawn agent `{bin}` for step `{step}`: {e}",
                bin = self.bin_display(),
                step = step.id,
            ),
            metrics: None,
        })?;

        let drained = drain_child_pipes(&mut child).await;

        let status = child.wait().await.map_err(|e| FlowdError::PlanExecution {
            message: format!(
                "failed to wait on agent `{bin}` for step `{step}`: {e}",
                bin = self.bin_display(),
                step = step.id,
            ),
            metrics: None,
        })?;

        let DrainedPipes {
            stdout: stdout_bytes,
            stderr: stderr_string,
        } = drained.map_err(|e| FlowdError::PlanExecution {
            message: format!(
                "failed to drain stdio for step `{step}`: {e}",
                step = step.id
            ),
            metrics: None,
        })?;

        let code = status.code();

        if !status.success() {
            tracing::warn!(
                step_id = %step.id,
                exit_code = ?code,
                stderr = %truncate_for_log(&stderr_string, 4096),
                "agent step exited non-zero"
            );
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "step `{step}` failed (exit={code}): {stderr}",
                    step = step.id,
                    code = code.map_or_else(|| "signal".into(), |c| c.to_string()),
                    stderr = truncate_for_log(stderr_string.trim(), 1024),
                ),
                metrics: None,
            });
        }

        if !stderr_string.is_empty() {
            tracing::debug!(
                step_id = %step.id,
                stderr = %truncate_for_log(&stderr_string, 2048),
                "agent step stderr"
            );
        }

        let summary =
            parse_claude_stream(&step.id, std::io::Cursor::new(&stdout_bytes)).map_err(|e| {
                FlowdError::PlanExecution {
                    message: format!(
                        "failed to parse stream-json output for step `{step}`: {e}",
                        step = step.id
                    ),
                    metrics: None,
                }
            })?;

        let Some(envelope) = summary.final_envelope else {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "step `{step}` produced no `result` event in its stream-json output \
                     (claude exited 0 but the daemon cannot record success without the \
                     final envelope; see claude-code#1920 for the upstream bug pattern)",
                    step = step.id,
                ),
                metrics: None,
            });
        };

        let (stdout, metrics) = validate_claude_envelope(&step.id, &envelope)?;

        Ok(AgentOutput {
            stdout,
            exit_code: code,
            metrics,
            session_id: summary.session_id,
        })
    }
}

impl AgentSpawner for LocalShellSpawner {
    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        // The plan's persisted `project_root` wins over the spawner's
        // own `cwd` (which is a daemon-process-wide fallback set from
        // `$FLOWD_AGENT_CWD` or, ultimately, `current_dir()`). Without
        // this, every plan would silently anchor to wherever the
        // operator launched flowd from -- fine for one workspace,
        // wrong the moment a daemon services more than one plan.
        self.spawn_in_cwd(
            step,
            ctx.project_root.as_deref(),
            ctx.prior_session_id.as_deref(),
        )
        .await
    }
}

#[derive(Debug, Clone)]
pub struct GitWorktreeSpawner {
    inner: LocalShellSpawner,
    manager: Arc<GitWorktreeManager>,
}

impl GitWorktreeSpawner {
    fn try_new(
        inner: LocalShellSpawner,
        flowd_home: &Path,
        branch_store: Option<Arc<SqliteStepBranchStore>>,
    ) -> Option<Self> {
        let cwd = inner.effective_cwd().ok()?;
        let repo = discover_git_root(&cwd).ok()?;
        Some(Self {
            inner,
            manager: Arc::new(GitWorktreeManager::new(
                repo,
                flowd_home.join("worktrees"),
                branch_store,
            )),
        })
    }

    fn repo_display(&self) -> std::path::Display<'_> {
        self.manager.repo.display()
    }
}

impl AgentSpawner for GitWorktreeSpawner {
    fn supports_worktree_isolation(&self) -> bool {
        true
    }

    async fn prepare_plan(&self, plan: &Plan, preview: &PlanPreview) -> Result<()> {
        if preview.max_parallelism <= 1 {
            return Ok(());
        }
        self.manager.prepare_plan(plan).await
    }

    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        if !ctx.plan_parallel {
            // Sequential plans skip the worktree machinery but must
            // still honour the plan's `project_root` over the wrapped
            // spawner's daemon-cwd fallback -- same precedence rule
            // `LocalShellSpawner::spawn` enforces directly. Session
            // resume is also forwarded here because purely sequential
            // plans hit this branch end-to-end and benefit from the
            // hot-cache behaviour.
            return self
                .inner
                .spawn_in_cwd(
                    step,
                    ctx.project_root.as_deref(),
                    ctx.prior_session_id.as_deref(),
                )
                .await;
        }
        let worktree = self.manager.prepare_step(&ctx, step).await?;
        // Parallel layers run inside isolated git worktrees; they
        // never receive a resume id (the layer runner clears the
        // session slot on entry), so we pass `None` unconditionally
        // -- guards against a future caller that hands us a stale
        // ctx.
        let output = self.inner.spawn_in_cwd(step, Some(&worktree), None).await?;
        self.manager.finish_step(&ctx, step, &worktree).await?;
        Ok(output)
    }
}

#[derive(Debug)]
struct GitWorktreeManager {
    repo: PathBuf,
    root: PathBuf,
    /// In-process cache of the durable `(plan_id, step_id) -> branch`
    /// mapping. Populated lazily from `branch_store` on first access for
    /// a given plan; written through on `finish_step`. The DB is the
    /// source of truth across daemon restarts -- this map exists so
    /// `dependency_branches` stays a sync hot path.
    branches: Mutex<HashMap<uuid::Uuid, HashMap<String, String>>>,
    /// Plan ids whose state has already been pulled out of
    /// `branch_store`. Distinct from `branches.contains_key(plan_id)`
    /// because an in-process insert from `finish_step` would
    /// otherwise look indistinguishable from "rehydrated, nothing
    /// recorded yet" and skip a needed reload.
    loaded_plans: Mutex<HashSet<uuid::Uuid>>,
    /// Durable backing store. `None` only in unit-test setups that
    /// don't need to assert restart behaviour; the daemon always
    /// supplies one.
    branch_store: Option<Arc<SqliteStepBranchStore>>,
}

impl GitWorktreeManager {
    fn new(repo: PathBuf, root: PathBuf, branch_store: Option<Arc<SqliteStepBranchStore>>) -> Self {
        Self {
            repo,
            root,
            branches: Mutex::new(HashMap::new()),
            loaded_plans: Mutex::new(HashSet::new()),
            branch_store,
        }
    }

    async fn prepare_plan(&self, plan: &Plan) -> Result<()> {
        // Plans persisted with `project_root` win over the manager's
        // construction-time `self.repo` (discovered once from the
        // daemon cwd by `try_new`). Without this, a parallel plan
        // submitted from a workspace different from the daemon's
        // launch dir would dirty-check and `git worktree add` against
        // the wrong repo. Legacy plans (`project_root = None`) fall
        // back to `self.repo` -- same behaviour as before this wiring.
        let repo = self.repo_for(plan.project_root.as_deref().map(Path::new));
        let dirty = git_output(repo, ["status", "--porcelain"]).await?;
        if !dirty.trim().is_empty() {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "parallel plan `{}` requires a clean git base at {}; refusing to use stashes. Commit or discard these changes first:\n{}",
                    plan.id,
                    repo.display(),
                    dirty.trim()
                ),
                metrics: None,
            });
        }
        tokio::fs::create_dir_all(self.plan_root(plan))
            .await
            .map_err(|e| FlowdError::PlanExecution {
                message: format!("create worktree root for plan `{}`: {e}", plan.id),
                metrics: None,
            })?;
        Ok(())
    }

    async fn prepare_step(&self, ctx: &AgentSpawnContext, step: &PlanStep) -> Result<PathBuf> {
        let branch = step_branch(ctx, step);
        let path = self
            .root
            .join(sanitize_path(&ctx.project))
            .join(ctx.plan_id.to_string())
            .join(sanitize_path(&step.id));
        if path.exists() {
            return Err(FlowdError::PlanExecution {
                message: format!(
                    "worktree path already exists for step `{}`: {}; remove it before resuming this parallel plan",
                    step.id,
                    path.display()
                ),
                metrics: None,
            });
        }

        // Pull the plan's durable step→branch state into the in-memory
        // cache before the sync `dependency_branches` consults it.
        // After a daemon restart this is the only thing that lets a
        // dependent step find its parent branch -- the in-process map
        // is empty until this call seeds it.
        self.ensure_branches_loaded(ctx.plan_id).await?;
        let dep_branches = self.dependency_branches(ctx.plan_id, step)?;
        let base = dep_branches.first().map_or("HEAD", String::as_str);
        let path_arg = path.to_string_lossy().into_owned();
        // Same precedence rule as `prepare_plan`: use `ctx.project_root`
        // when the plan persisted one, fall back to `self.repo` for
        // legacy plans. The dependent `git merge` calls below run
        // inside the freshly-added worktree, not the base repo, so they
        // already operate on the right tree without needing the override.
        let repo = self.repo_for(ctx.project_root.as_deref());
        git_output(repo, ["worktree", "add", "-b", &branch, &path_arg, base]).await?;

        // Repo identity guard. `git worktree add` can succeed against the
        // wrong repo when the resolved path is a symlink that lands inside
        // a different checkout, when the target collides with a stale
        // linked worktree, or when a nested checkout intercepts it. Compare
        // `--git-common-dir` of the freshly-added worktree against
        // `--git-common-dir` of the base repo; on mismatch, rip out the
        // worktree and abort before the agent prompt is dispatched.
        if let Err(e) = assert_worktree_identity(repo, &path).await {
            cleanup_rejected_worktree(repo, &path, &path_arg).await;
            return Err(e);
        }

        for dep in dep_branches.iter().skip(1) {
            git_output(&path, ["merge", "--no-edit", dep]).await?;
        }
        Ok(path)
    }

    /// Resolve the git repo path to drive `git worktree add` / `git status`
    /// from. The plan's persisted `project_root` (when present) overrides
    /// `self.repo`, which is only meaningful as a fallback for plans that
    /// pre-date the field. Returning `&Path` (not `PathBuf`) keeps the
    /// borrow tied to the input lifetimes -- callers pass either an owned
    /// or borrowed override and never need to clone.
    fn repo_for<'a>(&'a self, override_root: Option<&'a Path>) -> &'a Path {
        override_root.unwrap_or(self.repo.as_path())
    }

    async fn finish_step(
        &self,
        ctx: &AgentSpawnContext,
        step: &PlanStep,
        worktree: &Path,
    ) -> Result<()> {
        git_output(worktree, ["add", "-A"]).await?;
        let has_changes = git_status(worktree, ["diff", "--cached", "--quiet"]).await?;
        if has_changes != 0 {
            git_output(
                worktree,
                [
                    "-c",
                    "user.name=flowd",
                    "-c",
                    "user.email=flowd@local",
                    "commit",
                    "-m",
                    &format!("flowd step {}", step.id),
                ],
            )
            .await?;
        }
        let branch = step_branch(ctx, step);
        // Capture HEAD after the (maybe-)commit so the durable record
        // points at the exact tree future integration must
        // fast-forward to. A no-change step still records the inherited
        // tip so dependents can ff-merge to it without re-running.
        let tip_sha = git_output(worktree, ["rev-parse", "HEAD"])
            .await?
            .trim()
            .to_owned();

        {
            let mut guard = self
                .branches
                .lock()
                .map_err(|_| FlowdError::PlanExecution {
                    message: "worktree branch map poisoned".into(),
                    metrics: None,
                })?;
            guard
                .entry(ctx.plan_id)
                .or_default()
                .insert(step.id.clone(), branch.clone());
        }

        // Write through to the durable store after the in-process
        // cache update so the cache stays usable even if the DB write
        // fails (the executor will surface the error and the next
        // restart re-derives from the remaining rows).
        if let Some(store) = &self.branch_store {
            store
                .upsert(&StepBranchRecord {
                    plan_id: ctx.plan_id,
                    step_id: step.id.clone(),
                    branch,
                    tip_sha,
                    worktree_path: Some(worktree.to_string_lossy().into_owned()),
                })
                .await?;
        }
        Ok(())
    }

    /// Pull every recorded `(step_id, branch)` for `plan_id` out of the
    /// durable store into the in-memory cache. No-op once a plan has
    /// been loaded -- subsequent calls within the same process trust
    /// the cache, since `finish_step` writes through to both sides.
    ///
    /// Existing in-memory entries win over DB entries on a key clash,
    /// because an in-flight `finish_step` reflects the latest commit
    /// while the DB row could be one retry behind.
    async fn ensure_branches_loaded(&self, plan_id: uuid::Uuid) -> Result<()> {
        {
            let loaded = self
                .loaded_plans
                .lock()
                .map_err(|_| FlowdError::PlanExecution {
                    message: "worktree loaded-plans set poisoned".into(),
                    metrics: None,
                })?;
            if loaded.contains(&plan_id) {
                return Ok(());
            }
        }

        let records = if let Some(store) = &self.branch_store {
            store.list_for_plan(plan_id).await?
        } else {
            Vec::new()
        };

        let mut branches = self
            .branches
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree branch map poisoned".into(),
                metrics: None,
            })?;
        let entry = branches.entry(plan_id).or_default();
        for record in records {
            // `or_insert`: do not clobber an in-memory write recorded
            // by a concurrent `finish_step`.
            entry.entry(record.step_id).or_insert(record.branch);
        }
        drop(branches);

        let mut loaded = self
            .loaded_plans
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree loaded-plans set poisoned".into(),
                metrics: None,
            })?;
        loaded.insert(plan_id);
        Ok(())
    }

    fn dependency_branches(&self, plan_id: uuid::Uuid, step: &PlanStep) -> Result<Vec<String>> {
        let guard = self
            .branches
            .lock()
            .map_err(|_| FlowdError::PlanExecution {
                message: "worktree branch map poisoned".into(),
                metrics: None,
            })?;
        let Some(plan_branches) = guard.get(&plan_id) else {
            return Ok(Vec::new());
        };
        step.depends_on
            .iter()
            .map(|dep| {
                plan_branches
                    .get(dep)
                    .cloned()
                    .ok_or_else(|| FlowdError::PlanExecution {
                        message: format!(
                            "dependency branch for step `{dep}` is missing before running `{}`",
                            step.id
                        ),
                        metrics: None,
                    })
            })
            .collect()
    }

    fn plan_root(&self, plan: &Plan) -> PathBuf {
        self.root
            .join(sanitize_path(&plan.project))
            .join(plan.id.to_string())
    }
}

fn discover_git_root(cwd: &Path) -> Result<PathBuf> {
    let output = std::process::Command::new("git")
        .arg("-C")
        .arg(cwd)
        .arg("rev-parse")
        .arg("--show-toplevel")
        .output()
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("probe git repository at {}: {e}", cwd.display()),
            metrics: None,
        })?;
    if !output.status.success() {
        return Err(FlowdError::PlanExecution {
            message: format!("agent cwd {} is not inside a git repository", cwd.display()),
            metrics: None,
        });
    }
    let root = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    Ok(PathBuf::from(root))
}

async fn git_output<const N: usize>(cwd: &Path, args: [&str; N]) -> Result<String> {
    let output = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .output()
        .await
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("run git in {}: {e}", cwd.display()),
            metrics: None,
        })?;
    if output.status.success() {
        return Ok(String::from_utf8_lossy(&output.stdout).into_owned());
    }
    Err(FlowdError::PlanExecution {
        message: format!(
            "git command failed in {}: {}",
            cwd.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        ),
        metrics: None,
    })
}

/// Resolve `git rev-parse --git-common-dir` from `cwd` and return it as a
/// canonical absolute path. `--git-common-dir` returns a relative path
/// (e.g. `.git`) when invoked from a primary worktree and an absolute path
/// from a linked worktree, so we always join with `cwd` and canonicalize
/// before comparing — otherwise two paths that point at the same on-disk
/// `.git` would compare unequal just because one came back relative.
async fn git_common_dir(cwd: &Path) -> Result<PathBuf> {
    let raw = git_output(cwd, ["rev-parse", "--git-common-dir"]).await?;
    let trimmed = raw.trim();
    let candidate = PathBuf::from(trimmed);
    let resolved = if candidate.is_absolute() {
        candidate
    } else {
        cwd.join(candidate)
    };
    std::fs::canonicalize(&resolved).map_err(|e| FlowdError::PlanExecution {
        message: format!(
            "canonicalize git common dir {} (from cwd {}): {e}",
            resolved.display(),
            cwd.display(),
        ),
        metrics: None,
    })
}

/// README-promised post-`git worktree add` repository-identity guard:
/// confirm the freshly-added `worktree` actually belongs to the same
/// repository as `repo`. A mismatch happens when `git worktree add`
/// succeeds against a stale linked worktree, a symlinked target inside a
/// different checkout, or a nested checkout that intercepts the path.
/// Returning `Err` here is the signal for the caller to remove the
/// worktree and abort the step before the agent prompt is dispatched —
/// otherwise the run anchors against the wrong tree and every artefact
/// (commits, branches, observations) lands in the wrong repo.
async fn assert_worktree_identity(repo: &Path, worktree: &Path) -> Result<()> {
    let repo_common = git_common_dir(repo).await?;
    let wt_common = git_common_dir(worktree).await?;
    if repo_common == wt_common {
        return Ok(());
    }
    Err(FlowdError::PlanExecution {
        message: format!(
            "worktree {} resolved to git-common-dir {} but project_root {} resolved to {}; \
             refusing to dispatch step against the wrong repository",
            worktree.display(),
            wt_common.display(),
            repo.display(),
            repo_common.display(),
        ),
        metrics: None,
    })
}

/// Tear down a worktree the identity guard rejected. Best-effort: try
/// `git worktree remove --force` first so git's own bookkeeping (the
/// `worktrees/<name>` admin dir under the common dir, the registered
/// path) is cleaned up. If that fails — or if `git` refuses because the
/// path it sees is not the worktree it expected — fall back to a raw
/// `remove_dir_all` so we at least do not leave the directory on disk
/// to collide with the next step. Failures are logged, not propagated:
/// the caller is already returning the underlying guard error and the
/// operator's signal is that error, not the cleanup outcome.
async fn cleanup_rejected_worktree(repo: &Path, worktree: &Path, path_arg: &str) {
    if let Err(e) = git_output(repo, ["worktree", "remove", "--force", path_arg]).await {
        tracing::warn!(
            repo = %repo.display(),
            worktree = %worktree.display(),
            error = %e,
            "git worktree remove failed during identity-guard cleanup; falling back to rmdir"
        );
    }
    if worktree.exists() {
        if let Err(e) = tokio::fs::remove_dir_all(worktree).await {
            tracing::warn!(
                worktree = %worktree.display(),
                error = %e,
                "remove_dir_all failed during identity-guard cleanup; manual removal may be required"
            );
        }
    }
}

async fn git_status<const N: usize>(cwd: &Path, args: [&str; N]) -> Result<i32> {
    let status = Command::new("git")
        .arg("-C")
        .arg(cwd)
        .args(args)
        .status()
        .await
        .map_err(|e| FlowdError::PlanExecution {
            message: format!("run git in {}: {e}", cwd.display()),
            metrics: None,
        })?;
    Ok(status.code().unwrap_or(1))
}

fn step_branch(ctx: &AgentSpawnContext, step: &PlanStep) -> String {
    format!(
        "flowd/{}/{}/{}",
        sanitize_ref(&ctx.project),
        ctx.plan_id.simple(),
        sanitize_ref(&step.id),
    )
}

fn sanitize_path(raw: &str) -> String {
    raw.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') {
                c
            } else {
                '-'
            }
        })
        .collect()
}

fn sanitize_ref(raw: &str) -> String {
    let cleaned = sanitize_path(raw).trim_matches('-').to_owned();
    if cleaned.is_empty() {
        "unnamed".into()
    } else {
        cleaned
    }
}

/// Bytes drained from a child's stdout/stderr pipes by
/// [`drain_child_pipes`]. Both fields are populated on success;
/// stderr is decoded with `from_utf8_lossy` since it is only used
/// for logging and the daemon must never reject a step on a
/// malformed log line.
#[derive(Debug, Default)]
struct DrainedPipes {
    stdout: Vec<u8>,
    stderr: String,
}

/// Concurrently drain a child's stdout and stderr pipes into memory.
///
/// Both drains run as `tokio::spawn`'d tasks so a slow producer on
/// either pipe cannot deadlock us by filling the kernel buffer (the
/// classic `cmd.output()` antipattern that `tokio::process` already
/// avoids by reading both pipes in parallel under the hood; we
/// replicate it here because we own the `Child` to also call
/// `wait()` and `kill_on_drop`).
///
/// `Stdio::piped` must have been set on the `Command` that produced
/// `child`; if it was not, the `expect()` below panics, which is
/// strictly a programmer error.
async fn drain_child_pipes(child: &mut tokio::process::Child) -> std::io::Result<DrainedPipes> {
    let stdout = child
        .stdout
        .take()
        .expect("child stdout was piped at command construction time");
    let stderr = child
        .stderr
        .take()
        .expect("child stderr was piped at command construction time");

    let stdout_task = tokio::spawn(async move {
        let mut buf = Vec::with_capacity(4096);
        let mut reader = tokio::io::BufReader::new(stdout);
        tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buf)
            .await
            .map(|_| buf)
    });

    let stderr_task = tokio::spawn(async move {
        let mut buf = Vec::with_capacity(1024);
        let mut reader = tokio::io::BufReader::new(stderr);
        let _ = tokio::io::AsyncReadExt::read_to_end(&mut reader, &mut buf).await;
        String::from_utf8_lossy(&buf).into_owned()
    });

    // Surface drain-task panics as IO errors rather than letting them
    // unwind into the executor; same panic-isolation contract the
    // executor already enforces around the spawner itself.
    let stderr_string = stderr_task
        .await
        .map_err(|e| std::io::Error::other(format!("stderr drain task panicked: {e}")))?;
    let stdout_bytes = stdout_task
        .await
        .map_err(|e| std::io::Error::other(format!("stdout drain task panicked: {e}")))??;

    Ok(DrainedPipes {
        stdout: stdout_bytes,
        stderr: stderr_string,
    })
}

/// Aggregated state extracted from a `--output-format=stream-json`
/// run: the `session_id` lifted from the leading `system/init` event
/// (or, as a fallback, from the trailing `result` event) and the
/// final `result` envelope itself. Either field may be `None` if the
/// stream was malformed or the run died before completing; callers
/// distinguish "no session" from "no result" because they have
/// different recovery strategies.
#[derive(Debug, Default)]
struct ClaudeStreamSummary {
    session_id: Option<String>,
    final_envelope: Option<ClaudeJsonEnvelope>,
}

/// Validate a fully-assembled `result` envelope against the daemon's
/// invariants and project it down to the transport-agnostic
/// `(stdout, metrics)` pair the rest of the executor consumes.
///
/// Failure paths (`is_error: true`, non-empty `permission_denials`)
/// still attach the envelope's metrics to the returned `Err`: failed
/// steps still cost money, and the audit log must reflect the spend.
/// Returning `Err` here marks the step `failed` in the orchestration
/// log.
///
/// This function is pure -- no I/O, no env reads -- so the failure
/// cases can be exercised by table-driven unit tests against
/// hand-crafted envelopes, independent of the stream parser.
fn validate_claude_envelope(
    step_id: &str,
    envelope: &ClaudeJsonEnvelope,
) -> Result<(String, Option<AgentMetrics>)> {
    let metrics: AgentMetrics = envelope.into();

    if envelope.is_error {
        return Err(FlowdError::PlanExecution {
            message: format!(
                "step `{step_id}` reported is_error=true: {result}",
                result = truncate_for_log(envelope.result.trim(), 1024),
            ),
            metrics: Some(metrics),
        });
    }

    if !envelope.permission_denials.is_empty() {
        let denied: Vec<&str> = envelope
            .permission_denials
            .iter()
            .map(|d| d.tool_name.as_str())
            .collect();
        return Err(FlowdError::PlanExecution {
            message: format!(
                "step `{step_id}` had {n} tool invocation(s) denied despite \
                 --dangerously-skip-permissions ({denied:?}); the daemon is refusing \
                 to record this as success. result: {result}",
                n = denied.len(),
                result = truncate_for_log(envelope.result.trim(), 1024),
            ),
            metrics: Some(metrics),
        });
    }

    Ok((envelope.result.clone(), Some(metrics)))
}

/// Parse a Claude `--output-format=stream-json --verbose` byte stream
/// into a [`ClaudeStreamSummary`].
///
/// We only require two lines: the leading `system/init` (carrying
/// `session_id`) and the trailing `result` event (carrying the final
/// envelope). Everything else -- assistant text deltas, tool uses,
/// rate-limit pings, `system/api_retry` -- is logged at `debug` and
/// discarded. Parse errors on a single line are tolerated (logged at
/// `debug`) so a stray non-JSON line in stderr-tunneled-through-stdout
/// or a schema drift on an event we do not consume cannot fail the
/// whole step.
///
/// If the stream emits multiple `result` lines (defensive: spec says
/// exactly one, but real-world bugs have shipped duplicates), the
/// last wins.
///
/// `step_id` flows into the trace context so the daemon log can
/// correlate noisy event streams with a specific plan step.
fn parse_claude_stream<R: std::io::BufRead>(
    step_id: &str,
    reader: R,
) -> std::io::Result<ClaudeStreamSummary> {
    let mut summary = ClaudeStreamSummary::default();
    let mut event_count: u64 = 0;

    for line_res in reader.lines() {
        let line = line_res?;
        if line.trim().is_empty() {
            continue;
        }
        event_count += 1;

        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                tracing::debug!(
                    step_id,
                    error = %e,
                    line = %truncate_for_log(&line, 256),
                    "claude stream-json: dropping unparseable line"
                );
                continue;
            }
        };

        let ty = value.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let subtype = value.get("subtype").and_then(|v| v.as_str()).unwrap_or("");

        match ty {
            "system" if subtype == "init" => {
                if summary.session_id.is_none()
                    && let Some(sid) = value.get("session_id").and_then(|v| v.as_str())
                {
                    summary.session_id = Some(sid.to_owned());
                    tracing::debug!(step_id, session_id = sid, "claude stream-json: init");
                }
            }
            "system" if subtype == "api_retry" => {
                tracing::debug!(
                    step_id,
                    attempt = value.get("attempt").and_then(serde_json::Value::as_u64),
                    max_retries = value.get("max_retries").and_then(serde_json::Value::as_u64),
                    error_kind = value.get("error").and_then(|v| v.as_str()),
                    "claude stream-json: api retry"
                );
            }
            "result" => match serde_json::from_value::<ClaudeJsonEnvelope>(value) {
                Ok(envelope) => {
                    summary.final_envelope = Some(envelope);
                }
                Err(e) => {
                    tracing::warn!(
                        step_id,
                        error = %e,
                        "claude stream-json: result event failed to deserialize as envelope; \
                         dropping. Schema drift?"
                    );
                }
            },
            _ => {
                // assistant deltas, tool uses, rate_limit_event, stream_event, ...
                // Quiet at debug; the `result` event is the source of truth for metrics.
                tracing::trace!(
                    step_id,
                    event_type = ty,
                    "claude stream-json: ignored event"
                );
            }
        }
    }

    if summary.session_id.is_none()
        && let Some(env) = summary.final_envelope.as_ref()
        && let Some(sid) = env.session_id.as_deref()
    {
        // Fallback: missing/disabled init event but result still has the
        // session id. Better to pick it up here than thread None forward
        // and lose the resume opportunity.
        summary.session_id = Some(sid.to_owned());
    }

    tracing::debug!(
        step_id,
        events = event_count,
        captured_session = summary.session_id.is_some(),
        captured_result = summary.final_envelope.is_some(),
        "claude stream-json: parse complete"
    );

    Ok(summary)
}

/// Spawner used as the fallback when no agent binary is discoverable. Fails
/// every `spawn` with an actionable error so misconfiguration surfaces the
/// first time a plan is confirmed, instead of silently succeeding.
#[derive(Debug, Default, Clone, Copy)]
pub struct UnconfiguredSpawner;

impl AgentSpawner for UnconfiguredSpawner {
    async fn spawn(&self, _ctx: AgentSpawnContext, _step: &PlanStep) -> Result<AgentOutput> {
        Err(FlowdError::PlanExecution {
            message: "no agent CLI configured. Set $FLOWD_AGENT_BIN to an absolute \
                      path, or install one of: cursor-agent, claude, codex, aider \
                      into $PATH, then restart the daemon."
                .into(),
            metrics: None,
        })
    }
}

/// Type-erased spawner so the executor can be parametrised at runtime
/// without leaking concrete types into the rest of `start.rs`.
pub enum BoxedSpawner {
    Local(LocalShellSpawner),
    Worktree(GitWorktreeSpawner),
    Unconfigured(UnconfiguredSpawner),
}

impl BoxedSpawner {
    /// Build the best available spawner for the current environment.
    ///
    /// `branch_store`, when supplied, makes finished step → branch
    /// state survive a daemon restart. Pass `None` only when the
    /// caller explicitly does not want durable worktree state (the
    /// integration tests do this; the daemon never does).
    pub fn auto(flowd_home: &Path, branch_store: Option<Arc<SqliteStepBranchStore>>) -> Self {
        let Some(local) = LocalShellSpawner::detect() else {
            return Self::Unconfigured(UnconfiguredSpawner);
        };
        GitWorktreeSpawner::try_new(local.clone(), flowd_home, branch_store)
            .map_or(Self::Local(local), Self::Worktree)
    }

    pub fn description(&self) -> String {
        match self {
            Self::Local(s) => format!("local-shell ({})", s.bin_display()),
            Self::Worktree(s) => format!(
                "git-worktree ({}, repo {})",
                s.inner.bin_display(),
                s.repo_display()
            ),
            Self::Unconfigured(_) => "unconfigured (plan execution will fail)".into(),
        }
    }
}

impl AgentSpawner for BoxedSpawner {
    fn supports_worktree_isolation(&self) -> bool {
        matches!(self, Self::Worktree(_))
    }

    async fn prepare_plan(&self, plan: &Plan, preview: &PlanPreview) -> Result<()> {
        match self {
            Self::Worktree(s) => s.prepare_plan(plan, preview).await,
            Self::Local(_) | Self::Unconfigured(_) => Ok(()),
        }
    }

    async fn spawn(&self, ctx: AgentSpawnContext, step: &PlanStep) -> Result<AgentOutput> {
        match self {
            Self::Local(s) => s.spawn(ctx, step).await,
            Self::Worktree(s) => s.spawn(ctx, step).await,
            Self::Unconfigured(s) => s.spawn(ctx, step).await,
        }
    }
}

/// First executable match for `name` on `$PATH`. Returns `None` if the name
/// is not found or `$PATH` is unset.
fn which_in_path(name: &str) -> Option<String> {
    let path = std::env::var_os("PATH")?;
    std::env::split_paths(&path)
        .map(|dir| dir.join(name))
        .find(|p| is_executable(p))
        .and_then(|p| p.into_os_string().into_string().ok())
}

#[cfg(unix)]
fn is_executable(p: &std::path::Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata()
        .is_ok_and(|m| m.is_file() && (m.permissions().mode() & 0o111 != 0))
}

#[cfg(not(unix))]
fn is_executable(p: &std::path::Path) -> bool {
    // Best-effort on non-unix: any regular file in PATH is treated as
    // executable. The supported agent CLIs are unix-only in practice.
    p.is_file()
}

/// Truncate `s` to at most `max` bytes on a UTF-8 boundary, appending a
/// marker noting how many bytes were elided.
fn truncate_for_log(s: &str, max: usize) -> String {
    if s.len() <= max {
        return s.to_owned();
    }
    let cut = s
        .char_indices()
        .take_while(|(i, _)| *i <= max)
        .last()
        .map_or(0, |(i, _)| i);
    format!("{}…[+{} bytes]", &s[..cut], s.len() - cut)
}

#[cfg(test)]
mod tests {
    use super::*;
    use flowd_core::orchestration::StepStatus;

    fn step(prompt: &str) -> PlanStep {
        step_with_id("t", prompt, &[])
    }

    fn step_with_id(id: &str, prompt: &str, depends_on: &[&str]) -> PlanStep {
        PlanStep {
            id: id.into(),
            agent_type: "test".into(),
            prompt: prompt.into(),
            depends_on: depends_on.iter().map(|s| (*s).to_owned()).collect(),
            timeout_secs: None,
            retry_count: 0,
            status: StepStatus::Pending,
            output: None,
            error: None,
            started_at: None,
            completed_at: None,
        }
    }

    fn ctx() -> AgentSpawnContext {
        AgentSpawnContext {
            plan_id: uuid::Uuid::new_v4(),
            project: "test".into(),
            plan_parallel: false,
            layer_width: 1,
            project_root: None,
            prior_session_id: None,
        }
    }

    fn git_sync(repo: &Path, args: &[&str]) {
        let status = std::process::Command::new("git")
            .arg("-C")
            .arg(repo)
            .args(args)
            .status()
            .expect("spawn git");
        assert!(
            status.success(),
            "git {args:?} failed in {}",
            repo.display()
        );
    }

    /// Smoke test: `echo` stands in for an agent that just prints its
    /// argv. With the default `-p` flag the invocation is
    /// `echo -p "<prompt>"`, which exits 0 and contains the prompt.
    #[tokio::test]
    async fn echo_spawner_returns_stdout() {
        let s = LocalShellSpawner::new("echo");
        let out = s.spawn(ctx(), &step("hello flowd")).await.unwrap();
        assert_eq!(out.exit_code, Some(0));
        assert!(out.stdout.contains("hello flowd"));
    }

    #[tokio::test]
    async fn missing_binary_yields_plan_execution_error() {
        let s = LocalShellSpawner::new("/does/not/exist/flowd-agent");
        let err = s.spawn(ctx(), &step("anything")).await.unwrap_err();
        assert!(matches!(err, FlowdError::PlanExecution { .. }));
    }

    /// `LocalShellSpawner::spawn` must take its cwd from
    /// `ctx.project_root` (the plan's persisted workspace) and override
    /// the daemon-set `LocalShellSpawner.cwd` field. Without this,
    /// every step silently inherits the directory the operator launched
    /// flowd from -- fine for one workspace, wrong the moment a daemon
    /// services more than one plan. We assert against real subprocess
    /// state (`pwd`) so a future refactor that "passes the value
    /// through" but never reaches `Command::current_dir` would still
    /// fail this test.
    #[cfg(unix)]
    #[tokio::test]
    async fn local_shell_spawn_uses_ctx_project_root_over_daemon_cwd() {
        let daemon_cwd = tempfile::tempdir().expect("daemon cwd");
        let plan_root = tempfile::tempdir().expect("plan root");

        let s = LocalShellSpawner {
            bin: OsString::from("/bin/sh"),
            prompt_flag: "-c".to_owned(),
            cwd: Some(daemon_cwd.path().to_path_buf()),
            extra_args: Vec::new(),
            claude_bare: false,
        };

        let mut ctx = ctx();
        ctx.project_root = Some(plan_root.path().to_path_buf());

        let out = s
            .spawn(ctx, &step("pwd"))
            .await
            .expect("pwd subprocess must succeed");
        assert_eq!(out.exit_code, Some(0));

        // Canonicalise both sides: macOS resolves `/var/...` -> `/private/var/...`
        // when a process chdirs through the symlink, and `tempdir()` may
        // return either form. Compare canonical paths to dodge that.
        let expected = std::fs::canonicalize(plan_root.path()).expect("canonicalize plan root");
        let observed_raw = std::path::PathBuf::from(out.stdout.trim());
        let observed = std::fs::canonicalize(&observed_raw).expect("canonicalize pwd stdout");
        assert_eq!(
            observed,
            expected,
            "subprocess cwd resolved to {observed:?}; expected plan_root {expected:?}. \
             Daemon cwd was {daemon:?} -- ctx.project_root must win.",
            daemon = daemon_cwd.path()
        );
    }

    /// Sequential branch of `GitWorktreeSpawner::spawn` skips the
    /// worktree machinery and delegates to the wrapped
    /// `LocalShellSpawner`. Pin that the delegation still threads
    /// `ctx.project_root` through -- the historical bug had this
    /// branch hardcode `None`, silently anchoring sequential plans to
    /// the daemon cwd even after the executor learned to carry the
    /// plan root.
    #[cfg(unix)]
    #[tokio::test]
    async fn worktree_spawner_sequential_uses_ctx_project_root_over_daemon_cwd() {
        let daemon_cwd = tempfile::tempdir().expect("daemon cwd");
        let plan_root = tempfile::tempdir().expect("plan root");
        let flowd_home = tempfile::tempdir().expect("flowd home");

        let inner = LocalShellSpawner {
            bin: OsString::from("/bin/sh"),
            prompt_flag: "-c".to_owned(),
            cwd: Some(daemon_cwd.path().to_path_buf()),
            extra_args: Vec::new(),
            claude_bare: false,
        };
        let s = GitWorktreeSpawner {
            inner,
            // Manager `repo` is irrelevant on the sequential branch
            // (no `git worktree add` is called); pick any path so
            // construction succeeds.
            manager: Arc::new(GitWorktreeManager::new(
                daemon_cwd.path().to_path_buf(),
                flowd_home.path().join("worktrees"),
                None,
            )),
        };

        let mut ctx = ctx();
        ctx.plan_parallel = false;
        ctx.project_root = Some(plan_root.path().to_path_buf());

        let out = s
            .spawn(ctx, &step("pwd"))
            .await
            .expect("pwd subprocess must succeed");
        assert_eq!(out.exit_code, Some(0));

        let expected = std::fs::canonicalize(plan_root.path()).unwrap();
        let observed = std::fs::canonicalize(std::path::PathBuf::from(out.stdout.trim())).unwrap();
        assert_eq!(
            observed,
            expected,
            "GitWorktreeSpawner sequential delegation lost ctx.project_root: \
             observed {observed:?}, expected {expected:?}, daemon was {:?}",
            daemon_cwd.path()
        );
    }

    #[tokio::test]
    async fn unconfigured_spawner_reports_actionable_error() {
        let s = UnconfiguredSpawner;
        let err = s.spawn(ctx(), &step("x")).await.unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("FLOWD_AGENT_BIN"));
        assert!(msg.contains("cursor-agent"));
    }

    #[test]
    fn truncate_for_log_passes_through_short_strings() {
        assert_eq!(truncate_for_log("abc", 10), "abc");
    }

    /// `αβγδε` is 10 bytes (5 chars × 2). With `max = 5` we can fit
    /// `αβ` (4 bytes) but not `αβγ` (6 bytes); the cut must land on a
    /// UTF-8 boundary and the elided-byte marker must follow.
    #[test]
    fn truncate_for_log_clips_on_utf8_boundary() {
        let out = truncate_for_log("αβγδε", 5);
        assert_eq!(out, "αβ…[+6 bytes]");
    }

    #[test]
    fn is_claude_like_recognises_canonical_names() {
        assert!(is_claude_like(OsStr::new("claude")));
        assert!(is_claude_like(OsStr::new("/usr/local/bin/claude")));
        assert!(is_claude_like(OsStr::new("cursor-agent")));
        assert!(is_claude_like(OsStr::new("/opt/cursor-agent")));
        // case-insensitive on the basename
        assert!(is_claude_like(OsStr::new("/x/Claude")));
        assert!(is_claude_like(OsStr::new("CLAUDE.exe")));
    }

    #[test]
    fn is_claude_like_rejects_unrelated_bins() {
        assert!(!is_claude_like(OsStr::new("echo")));
        assert!(!is_claude_like(OsStr::new("codex")));
        assert!(!is_claude_like(OsStr::new("aider")));
        assert!(!is_claude_like(OsStr::new("/usr/bin/claude-like-but-not")));
    }

    /// Helper: build a spawner with a chosen bin and `claude_bare`
    /// flag, sidestepping `detect()`'s env reads. Tests should not
    /// depend on the ambient `FLOWD_AGENT_CLAUDE_BARE` value.
    fn spawner_with_bare(bin: &str, claude_bare: bool) -> LocalShellSpawner {
        LocalShellSpawner {
            bin: OsString::from(bin),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: None,
            extra_args: Vec::new(),
            claude_bare,
        }
    }

    /// Non-Claude bins never get injected flags, regardless of the
    /// `claude_bare` toggle.
    #[test]
    fn default_args_empty_for_non_claude_bins() {
        assert!(spawner_with_bare("echo", true).default_args().is_empty());
        assert!(spawner_with_bare("echo", false).default_args().is_empty());
        assert!(spawner_with_bare("aider", true).default_args().is_empty());
        assert!(spawner_with_bare("codex", false).default_args().is_empty());
    }

    /// Default ON path: `--bare` is the first injected arg for every
    /// Claude-like basename, followed by the stream-json pair
    /// (`--output-format=stream-json --verbose`) introduced in PR 2.
    /// This is the default state most operators hit.
    #[test]
    fn default_args_includes_bare_when_enabled_for_claude_family() {
        assert_eq!(
            spawner_with_bare("claude", true).default_args(),
            &[
                "--bare",
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        );
        assert_eq!(
            spawner_with_bare("cursor-agent", true).default_args(),
            &[
                "--bare",
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        );
        // Path-prefixed bins still resolve as claude-like.
        assert_eq!(
            spawner_with_bare("/usr/local/bin/claude", true).default_args(),
            &[
                "--bare",
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        );
    }

    /// Opt-out path: `FLOWD_AGENT_CLAUDE_BARE=0` (modeled here as
    /// `claude_bare: false`) drops `--bare` but retains the
    /// stream-json output pair, since the daemon still relies on the
    /// `result` and `init` events regardless of bare/non-bare mode.
    #[test]
    fn default_args_omits_bare_when_disabled_for_claude_family() {
        assert_eq!(
            spawner_with_bare("claude", false).default_args(),
            &[
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        );
        assert_eq!(
            spawner_with_bare("cursor-agent", false).default_args(),
            &[
                "--dangerously-skip-permissions",
                "--output-format=stream-json",
                "--verbose",
            ]
        );
    }

    /// Resume injection: when `resume_id` is supplied the assembled
    /// argv must lead with `--resume <id>` BEFORE the default flags
    /// and the operator's extras, ending with `prompt_flag prompt`.
    /// Pinning the order keeps a future refactor of `spawn_in_cwd`
    /// from quietly reshuffling positions and breaking last-flag-wins
    /// override semantics that operators rely on.
    #[test]
    fn assemble_args_prepends_resume_before_defaults_and_extras() {
        let injected = ["--bare", "--output-format=stream-json", "--verbose"];
        let extras = vec!["--model".to_owned(), "haiku".to_owned()];
        let argv = assemble_args(Some("sid-abc"), &injected, &extras, "-p", "do thing");
        assert_eq!(
            argv,
            vec![
                "--resume".to_owned(),
                "sid-abc".to_owned(),
                "--bare".to_owned(),
                "--output-format=stream-json".to_owned(),
                "--verbose".to_owned(),
                "--model".to_owned(),
                "haiku".to_owned(),
                "-p".to_owned(),
                "do thing".to_owned(),
            ]
        );
    }

    /// Without a `resume_id` the assembled argv must look exactly like
    /// the pre-PR3 layout: defaults, then extras, then prompt. Anyone
    /// changing this contract should fail this test loudly.
    #[test]
    fn assemble_args_omits_resume_when_id_is_none() {
        let injected = ["--bare", "--output-format=stream-json", "--verbose"];
        let extras = vec!["--model".to_owned(), "haiku".to_owned()];
        let argv = assemble_args(None, &injected, &extras, "-p", "do thing");
        assert_eq!(
            argv,
            vec![
                "--bare".to_owned(),
                "--output-format=stream-json".to_owned(),
                "--verbose".to_owned(),
                "--model".to_owned(),
                "haiku".to_owned(),
                "-p".to_owned(),
                "do thing".to_owned(),
            ]
        );
    }

    /// Empty `injected` and `extras` is the legitimate non-Claude
    /// path: prompt comes through unchanged, no resume regardless of
    /// what was passed (the caller is expected to filter `resume_id`
    /// to claude-like binaries before calling, but we still verify
    /// the minimal output shape).
    #[test]
    fn assemble_args_minimal_just_prompt_when_nothing_injected() {
        let argv = assemble_args(None, &[], &[], "-p", "hello");
        assert_eq!(argv, vec!["-p".to_owned(), "hello".to_owned()]);
    }

    /// `spawn_in_cwd` filters `prior_session_id` through
    /// `claude_like` before constructing the `resume_id` passed to
    /// `assemble_args`, so a foreign bin (codex/aider) gets `None`.
    /// We pin that filtering by reproducing the call shape: a
    /// non-Claude bin's filtered `resume_id` is `None`, and the
    /// assembled argv contains no `--resume`.
    #[test]
    fn assemble_args_no_resume_when_caller_filtered_for_non_claude() {
        // Simulating: prior_session_id was Some("sid-x") but the
        // caller filtered it out because is_claude_like returned
        // false.
        let resume_id: Option<&str> = None;
        let argv = assemble_args(resume_id, &[], &[], "--prompt", "p");
        assert!(!argv.iter().any(|a| a == "--resume"));
    }

    /// Pin the toggle's tri-state parser. Only the literal strings
    /// `"0"` and `"1"` map to a hard `Off` / `On`; everything else
    /// (unset, empty, typos, `"false"`, `"yes"`) resolves to `Auto`
    /// so the credential probe gets the final say. We test the pure
    /// parser rather than `read_claude_bare_toggle` itself so this
    /// test does not race with parallel tests over the process
    /// environment.
    #[test]
    fn parse_claude_bare_toggle_tri_state() {
        assert_eq!(
            parse_claude_bare_toggle(None),
            ClaudeBareToggle::Auto,
            "unset resolves to Auto"
        );
        assert_eq!(
            parse_claude_bare_toggle(Some("0")),
            ClaudeBareToggle::Off,
            "\"0\" forces Off"
        );
        assert_eq!(
            parse_claude_bare_toggle(Some("1")),
            ClaudeBareToggle::On,
            "\"1\" forces On"
        );
        for stray in ["", "false", "true", "yes", "no", "00", "01"] {
            assert_eq!(
                parse_claude_bare_toggle(Some(stray)),
                ClaudeBareToggle::Auto,
                "unrecognised value {stray:?} must resolve to Auto",
            );
        }
    }

    /// Pin the resolver policy. Pure -- no env mutation, no keychain
    /// probe -- so we can exhaustively cover the truth table.
    #[test]
    fn resolve_claude_bare_truth_table() {
        use ClaudeBareToggle::{Auto, Off, On};
        use ClaudeCredentialState::{Available, Missing};

        for creds in [Available, Missing] {
            for env_api_key_set in [true, false] {
                assert!(
                    resolve_claude_bare(On, creds, env_api_key_set),
                    "explicit On always wins (creds={creds:?}, env={env_api_key_set})"
                );
                assert!(
                    !resolve_claude_bare(Off, creds, env_api_key_set),
                    "explicit Off always wins (creds={creds:?}, env={env_api_key_set})"
                );
            }
        }

        assert!(
            resolve_claude_bare(Auto, Missing, true),
            "Auto + ANTHROPIC_API_KEY set must keep bare ON"
        );
        assert!(
            resolve_claude_bare(Auto, Available, true),
            "Auto + env key set wins over local creds (env is sufficient for bare mode)"
        );
        assert!(
            !resolve_claude_bare(Auto, Available, false),
            "Auto + creds + no env key must auto-disable bare for OAuth/keychain auth"
        );
        assert!(
            resolve_claude_bare(Auto, Missing, false),
            "Auto + neither auth source must default to bare ON (legacy safe default; \
             warning fires from log_claude_bare_decision)"
        );
    }

    /// Build a `ClaudeJsonEnvelope` directly from a JSON-blob string,
    /// for tests of the validator. In production the envelope arrives
    /// via [`parse_claude_stream`]; here we skip that hop to keep
    /// validator tests focused on the validator's own contracts
    /// (failure surfacing, metrics propagation).
    fn envelope_from_blob(blob: &str) -> ClaudeJsonEnvelope {
        serde_json::from_str(blob).expect("test fixture must be valid envelope JSON")
    }

    #[test]
    fn validate_envelope_extracts_result_on_success() {
        let env = envelope_from_blob(r#"{"is_error":false,"result":"OK","permission_denials":[]}"#);
        let (out, _) = validate_claude_envelope("step-1", &env).unwrap();
        assert_eq!(out, "OK");
    }

    /// Permission denials persisting past `--dangerously-skip-permissions`
    /// are the canary the spawner is built to catch. The model returns
    /// exit 0 with `is_error: false` but `permission_denials` is non-empty;
    /// this used to land as `step_completed` against an empty working tree.
    #[test]
    fn validate_envelope_fails_on_permission_denials() {
        let env = envelope_from_blob(
            r#"{
                "is_error": false,
                "result": "File creation was denied. Let me know if you'd like to approve.",
                "permission_denials": [{"tool_name": "Write"}]
            }"#,
        );
        let err = validate_claude_envelope("step-1", &env).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("denied despite --dangerously-skip-permissions"),
            "got: {msg}"
        );
        assert!(msg.contains("Write"), "got: {msg}");
    }

    #[test]
    fn validate_envelope_fails_on_is_error_true() {
        let env = envelope_from_blob(r#"{"is_error":true,"result":"upstream API error"}"#);
        let err = validate_claude_envelope("step-1", &env).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("is_error=true"), "got: {msg}");
        assert!(msg.contains("upstream API error"), "got: {msg}");
    }

    /// Schema-drift defence: tolerate unknown keys without choking,
    /// and treat absent keys as their defaults (`is_error=false`,
    /// no denials, empty result).
    #[test]
    fn validate_envelope_tolerates_unknown_fields() {
        let env = envelope_from_blob(
            r#"{"is_error":false,"result":"x","future_field":42,"another":"y"}"#,
        );
        let (out, _) = validate_claude_envelope("step-1", &env).unwrap();
        assert_eq!(out, "x");
    }

    /// Happy-path metric extraction: a full envelope populates every
    /// numeric field the daemon bills against.
    #[test]
    fn validate_envelope_extracts_metrics_on_success() {
        let env = envelope_from_blob(
            r#"{
                "is_error": false,
                "result": "done",
                "permission_denials": [],
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 50,
                    "cache_creation_input_tokens": 10,
                    "cache_read_input_tokens": 20
                },
                "total_cost_usd": 0.0123,
                "duration_ms": 1500,
                "duration_api_ms": 1200,
                "modelUsage": {
                    "claude-sonnet-4-5": {
                        "inputTokens": 100,
                        "outputTokens": 50,
                        "cacheCreationInputTokens": 10,
                        "cacheReadInputTokens": 20,
                        "costUSD": 0.0123
                    }
                }
            }"#,
        );
        let (stdout, metrics) = validate_claude_envelope("step-1", &env).unwrap();
        assert_eq!(stdout, "done");
        let m = metrics.expect("metrics must be present on success");
        assert_eq!(m.input_tokens, 100);
        assert_eq!(m.output_tokens, 50);
        assert_eq!(m.cache_creation_input_tokens, 10);
        assert_eq!(m.cache_read_input_tokens, 20);
        assert!((m.total_cost_usd - 0.0123).abs() < f64::EPSILON);
        assert_eq!(m.duration_ms, 1500);
        assert_eq!(m.duration_api_ms, 1200);
        let model = m
            .model_usage
            .get("claude-sonnet-4-5")
            .expect("per-model entry must be decoded");
        assert_eq!(model.input_tokens, 100);
        assert_eq!(model.output_tokens, 50);
        assert_eq!(model.cache_creation_input_tokens, 10);
        assert_eq!(model.cache_read_input_tokens, 20);
        assert!((model.cost_usd - 0.0123).abs() < f64::EPSILON);
    }

    /// Failed steps still cost money: the `is_error: true` path must
    /// surface the envelope's metrics through the `Err` so accounting
    /// stays honest.
    #[test]
    fn validate_envelope_attaches_metrics_to_is_error_failure() {
        let env = envelope_from_blob(
            r#"{
                "is_error": true,
                "result": "upstream API error",
                "usage": {"input_tokens": 7, "output_tokens": 3},
                "total_cost_usd": 0.004,
                "duration_ms": 250,
                "duration_api_ms": 240
            }"#,
        );
        let err = validate_claude_envelope("step-1", &env).unwrap_err();
        match err {
            FlowdError::PlanExecution { metrics, .. } => {
                let m = metrics.expect("is_error failure must carry metrics");
                assert_eq!(m.input_tokens, 7);
                assert_eq!(m.output_tokens, 3);
                assert!((m.total_cost_usd - 0.004).abs() < f64::EPSILON);
                assert_eq!(m.duration_ms, 250);
                assert_eq!(m.duration_api_ms, 240);
            }
            other => panic!("expected PlanExecution, got {other:?}"),
        }
    }

    /// Same contract as `is_error`: a denied-tool failure must still
    /// report the cost incurred getting to the refusal.
    #[test]
    fn validate_envelope_attaches_metrics_to_permission_denial_failure() {
        let env = envelope_from_blob(
            r#"{
                "is_error": false,
                "result": "refused",
                "permission_denials": [{"tool_name": "Write"}],
                "usage": {"input_tokens": 12, "output_tokens": 4},
                "total_cost_usd": 0.002
            }"#,
        );
        let err = validate_claude_envelope("step-1", &env).unwrap_err();
        match err {
            FlowdError::PlanExecution { metrics, .. } => {
                let m = metrics.expect("permission_denials failure must carry metrics");
                assert_eq!(m.input_tokens, 12);
                assert_eq!(m.output_tokens, 4);
                assert!((m.total_cost_usd - 0.002).abs() < f64::EPSILON);
            }
            other => panic!("expected PlanExecution, got {other:?}"),
        }
    }

    /// Schema-drift defence: an envelope without `usage` (or any other
    /// new field) must decode into zero-valued metrics rather than
    /// erroring. Upstream can drop or rename a field between releases
    /// and the spawner keeps running.
    #[test]
    fn validate_envelope_handles_minimal_envelope() {
        let env = envelope_from_blob(r#"{"is_error":false,"result":"minimal"}"#);
        let (stdout, metrics) = validate_claude_envelope("step-1", &env).unwrap();
        assert_eq!(stdout, "minimal");
        let m = metrics.expect("metrics envelope is constructed unconditionally on parse");
        assert_eq!(m.input_tokens, 0);
        assert_eq!(m.output_tokens, 0);
        assert_eq!(m.cache_creation_input_tokens, 0);
        assert_eq!(m.cache_read_input_tokens, 0);
        assert!(m.total_cost_usd.abs() < f64::EPSILON);
        assert_eq!(m.duration_ms, 0);
        assert_eq!(m.duration_api_ms, 0);
        assert!(m.model_usage.is_empty());
    }

    // ---- parse_claude_stream ------------------------------------------

    /// Helper: feed a literal NDJSON string to `parse_claude_stream`.
    fn parse_stream(raw: &str) -> ClaudeStreamSummary {
        parse_claude_stream("step-test", std::io::Cursor::new(raw.as_bytes()))
            .expect("BufRead over &[u8] cannot fail; line decoding is infallible here")
    }

    /// Happy path: a real-world four-event stream (`init`, `assistant`,
    /// `rate_limit_event`, `result`) yields both a session id and a
    /// validated envelope.
    #[test]
    fn parse_stream_extracts_session_id_and_result() {
        let raw = concat!(
            r#"{"type":"system","subtype":"init","session_id":"abc-123","model":"claude-sonnet-4-5"}"#,
            "\n",
            r#"{"type":"assistant","message":{"content":[{"type":"text","text":"working..."}]}}"#,
            "\n",
            r#"{"type":"rate_limit_event","rate_limit_info":{"status":"ok"}}"#,
            "\n",
            r#"{"type":"result","subtype":"success","session_id":"abc-123","is_error":false,"result":"OK","usage":{"input_tokens":10,"output_tokens":5},"total_cost_usd":0.001}"#,
            "\n",
        );
        let summary = parse_stream(raw);
        assert_eq!(summary.session_id.as_deref(), Some("abc-123"));
        let env = summary
            .final_envelope
            .expect("result event must populate the envelope");
        let (stdout, metrics) = validate_claude_envelope("step-1", &env).unwrap();
        assert_eq!(stdout, "OK");
        let m = metrics.unwrap();
        assert_eq!(m.input_tokens, 10);
        assert_eq!(m.output_tokens, 5);
    }

    /// Schema-drift defence: unknown event types and unparseable
    /// lines (an in-band stderr leak, a partial flush) must not
    /// abort the parse. We still get the result envelope.
    #[test]
    fn parse_stream_tolerates_unknown_event_types_and_garbage() {
        let raw = concat!(
            r#"{"type":"system","subtype":"init","session_id":"sid"}"#,
            "\n",
            r#"{"type":"future_event_type_we_do_not_handle","payload":"whatever"}"#,
            "\n",
            "this line is not JSON at all\n",
            r#"{"type":"stream_event","event":{"type":"content_block_delta"}}"#,
            "\n",
            r#"{"type":"result","subtype":"success","is_error":false,"result":"done"}"#,
            "\n",
        );
        let summary = parse_stream(raw);
        assert_eq!(summary.session_id.as_deref(), Some("sid"));
        let env = summary.final_envelope.expect("result still landed");
        assert_eq!(env.result, "done");
    }

    /// Defensive: spec says exactly one `result` line, but if a buggy
    /// build ships duplicates the LAST one wins (typically the more
    /// complete envelope after retries).
    #[test]
    fn parse_stream_uses_last_result_when_multiple_present() {
        let raw = concat!(
            r#"{"type":"system","subtype":"init","session_id":"sid"}"#,
            "\n",
            r#"{"type":"result","subtype":"success","is_error":false,"result":"first","total_cost_usd":0.001}"#,
            "\n",
            r#"{"type":"result","subtype":"success","is_error":false,"result":"second","total_cost_usd":0.002}"#,
            "\n",
        );
        let summary = parse_stream(raw);
        let env = summary.final_envelope.unwrap();
        assert_eq!(env.result, "second");
        assert!((env.total_cost_usd - 0.002).abs() < f64::EPSILON);
    }

    /// Fallback: when the init event is suppressed (a non-bare CLI
    /// without `--verbose`, or a future schema drift) but the result
    /// event still carries `session_id`, lift it from there.
    #[test]
    fn parse_stream_falls_back_to_session_id_in_result_event() {
        let raw = concat!(
            r#"{"type":"result","subtype":"success","session_id":"sid-from-result","is_error":false,"result":"done"}"#,
            "\n",
        );
        let summary = parse_stream(raw);
        assert_eq!(summary.session_id.as_deref(), Some("sid-from-result"));
    }

    /// Empty NDJSON: no session id captured, no envelope. The caller
    /// (`spawn_claude_streaming`) is responsible for raising the
    /// "no result event" error; the parser itself is content-free.
    #[test]
    fn parse_stream_returns_empty_summary_on_empty_input() {
        let summary = parse_stream("");
        assert!(summary.session_id.is_none());
        assert!(summary.final_envelope.is_none());
    }

    /// A broken stream (init only, no `result`) yields a session id
    /// but no envelope. Caller must error.
    #[test]
    fn parse_stream_yields_no_envelope_when_result_absent() {
        let raw = concat!(
            r#"{"type":"system","subtype":"init","session_id":"sid"}"#,
            "\n",
            r#"{"type":"assistant","message":{"content":[]}}"#,
            "\n",
        );
        let summary = parse_stream(raw);
        assert_eq!(summary.session_id.as_deref(), Some("sid"));
        assert!(summary.final_envelope.is_none());
    }

    #[tokio::test]
    async fn worktree_manager_rejects_dirty_base_and_records_step_branch() {
        let repo = tempfile::tempdir().expect("repo");
        git_sync(repo.path(), &["init"]);
        std::fs::write(repo.path().join("README.md"), "base\n").unwrap();
        git_sync(repo.path(), &["add", "README.md"]);
        git_sync(
            repo.path(),
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "commit",
                "-m",
                "base",
            ],
        );

        let worktrees = tempfile::tempdir().expect("worktrees");
        let manager = GitWorktreeManager::new(
            repo.path().to_path_buf(),
            worktrees.path().join("worktrees"),
            None,
        );
        let plan = flowd_core::orchestration::Plan::new(
            "p",
            "proj",
            vec![
                step_with_id("a", "do a", &[]),
                step_with_id("b", "do b", &[]),
            ],
        );

        std::fs::write(repo.path().join("dirty.txt"), "dirty\n").unwrap();
        let dirty = manager.prepare_plan(&plan).await.unwrap_err();
        assert!(dirty.to_string().contains("requires a clean git base"));
        std::fs::remove_file(repo.path().join("dirty.txt")).unwrap();

        manager.prepare_plan(&plan).await.unwrap();
        let ctx = AgentSpawnContext {
            plan_id: plan.id,
            project: plan.project.clone(),
            plan_parallel: true,
            layer_width: 2,
            project_root: None,
            prior_session_id: None,
        };
        let step = step_with_id("a", "do a", &[]);
        let worktree = manager.prepare_step(&ctx, &step).await.unwrap();
        assert!(worktree.is_dir());
        std::fs::write(worktree.join("a.txt"), "a\n").unwrap();
        manager.finish_step(&ctx, &step, &worktree).await.unwrap();

        let branches = manager.branches.lock().unwrap();
        assert_eq!(
            branches
                .get(&plan.id)
                .and_then(|plan_branches| plan_branches.get("a"))
                .map(String::as_str),
            Some(step_branch(&ctx, &step).as_str()),
        );
    }

    /// End-to-end persistence: `finish_step` must write the durable
    /// record, and a *fresh* `GitWorktreeManager` (modelling a daemon
    /// restart) must rehydrate that state via `ensure_branches_loaded`
    /// so dependency resolution still works -- this is the scenario
    /// the in-process map could never satisfy.
    #[tokio::test]
    async fn worktree_manager_persists_branch_and_rehydrates_after_restart() {
        let repo = tempfile::tempdir().expect("repo");
        git_sync(repo.path(), &["init"]);
        std::fs::write(repo.path().join("README.md"), "base\n").unwrap();
        git_sync(repo.path(), &["add", "README.md"]);
        git_sync(
            repo.path(),
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "commit",
                "-m",
                "base",
            ],
        );

        let worktrees = tempfile::tempdir().expect("worktrees");
        let db_dir = tempfile::tempdir().expect("db dir");
        let db_path = db_dir.path().join("flowd.sqlite");

        // ---- First process: open store, run a step, drop the manager.
        let plan = flowd_core::orchestration::Plan::new(
            "p",
            "proj",
            vec![
                step_with_id("a", "do a", &[]),
                step_with_id("b", "do b", &["a"]),
            ],
        );
        let step_a = step_with_id("a", "do a", &[]);
        let ctx = AgentSpawnContext {
            plan_id: plan.id,
            project: plan.project.clone(),
            plan_parallel: true,
            layer_width: 2,
            project_root: None,
            prior_session_id: None,
        };

        let recorded_branch;
        let recorded_tip_sha;
        {
            let store = Arc::new(
                flowd_storage::step_branch_store::SqliteStepBranchStore::open(&db_path)
                    .expect("open store"),
            );
            let manager = GitWorktreeManager::new(
                repo.path().to_path_buf(),
                worktrees.path().join("worktrees"),
                Some(Arc::clone(&store)),
            );
            manager.prepare_plan(&plan).await.unwrap();
            let worktree_a = manager.prepare_step(&ctx, &step_a).await.unwrap();
            std::fs::write(worktree_a.join("a.txt"), "a\n").unwrap();
            manager
                .finish_step(&ctx, &step_a, &worktree_a)
                .await
                .unwrap();

            // Read what got persisted so the second pass can assert
            // the durable record carried the same branch + tip sha
            // that the in-process map saw.
            let rows = store.list_for_plan(plan.id).await.unwrap();
            assert_eq!(rows.len(), 1, "exactly one row written for step a");
            assert_eq!(rows[0].step_id, "a");
            assert_eq!(rows[0].branch, step_branch(&ctx, &step_a));
            assert!(
                !rows[0].tip_sha.is_empty(),
                "tip_sha must be captured from the worktree HEAD"
            );
            assert_eq!(
                rows[0].worktree_path.as_deref(),
                Some(worktree_a.to_string_lossy().as_ref()),
            );
            recorded_branch = rows[0].branch.clone();
            recorded_tip_sha = rows[0].tip_sha.clone();
        }

        // ---- Second process: brand-new manager, brand-new in-memory
        // map, same on-disk DB. `ensure_branches_loaded` must pull
        // step a's branch back so a step b that depends on it can
        // resolve its parent without the original manager ever having
        // existed in this process. Without MIGRATION_007 + the
        // rehydration hook this would error with
        // "dependency branch for step `a` is missing".
        let store2 = Arc::new(
            flowd_storage::step_branch_store::SqliteStepBranchStore::open(&db_path)
                .expect("reopen store"),
        );
        let manager2 = GitWorktreeManager::new(
            repo.path().to_path_buf(),
            worktrees.path().join("worktrees"),
            Some(Arc::clone(&store2)),
        );

        manager2.ensure_branches_loaded(plan.id).await.unwrap();
        let cached_branch = {
            let guard = manager2.branches.lock().unwrap();
            guard
                .get(&plan.id)
                .and_then(|m| m.get("a"))
                .cloned()
                .expect("branch must be rehydrated from durable store")
        };
        assert_eq!(
            cached_branch, recorded_branch,
            "rehydrated branch name must match the persisted record"
        );

        // Sanity: a step that depends on `a` must now resolve through
        // the rehydrated map -- this is the production read site we
        // care about, not just the cache contents.
        let step_b = step_with_id("b", "do b", &["a"]);
        let dep_branches = manager2.dependency_branches(plan.id, &step_b).unwrap();
        assert_eq!(dep_branches, vec![recorded_branch.clone()]);

        // The tip sha is preserved verbatim across the boundary --
        // future integration uses it to fast-forward instead of
        // walking the worktree.
        let rows2 = store2.list_for_plan(plan.id).await.unwrap();
        assert_eq!(rows2.len(), 1);
        assert_eq!(rows2[0].tip_sha, recorded_tip_sha);
    }

    /// `ensure_branches_loaded` must be safe to call even when the
    /// plan has nothing recorded yet -- this is the common case before
    /// any step finishes (and the only case for sequential plans). It
    /// must not error and must not poison the cache against a
    /// subsequent `finish_step`.
    #[tokio::test]
    async fn ensure_branches_loaded_is_noop_for_legacy_empty_plan_state() {
        let dir = tempfile::tempdir().expect("dir");
        let store = Arc::new(
            flowd_storage::step_branch_store::SqliteStepBranchStore::open(
                &dir.path().join("flowd.sqlite"),
            )
            .expect("open store"),
        );
        let manager = GitWorktreeManager::new(
            dir.path().to_path_buf(),
            dir.path().join("worktrees"),
            Some(Arc::clone(&store)),
        );

        let plan_id = uuid::Uuid::new_v4();
        // No prior writes -- DB has the table but no rows for this
        // plan. `ensure_branches_loaded` should treat that as
        // "nothing to rehydrate" and succeed.
        manager.ensure_branches_loaded(plan_id).await.unwrap();
        assert!(
            manager
                .branches
                .lock()
                .unwrap()
                .get(&plan_id)
                .is_none_or(std::collections::HashMap::is_empty),
            "no rows recorded -> in-memory map stays empty for this plan"
        );

        // Calling again is also a no-op: the loaded set guards
        // against a redundant DB read on every subsequent prepare.
        manager.ensure_branches_loaded(plan_id).await.unwrap();
    }

    /// Bootstrap a real on-disk git repo with one commit so it has a
    /// resolvable HEAD. Returns the canonical repo path so callers can
    /// compare against canonicalised git output without macOS's
    /// `/var` -> `/private/var` symlink biting them.
    fn init_repo(dir: &Path) -> PathBuf {
        git_sync(dir, &["init"]);
        std::fs::write(dir.join("seed.txt"), "seed\n").unwrap();
        git_sync(dir, &["add", "seed.txt"]);
        git_sync(
            dir,
            &[
                "-c",
                "user.name=flowd-test",
                "-c",
                "user.email=flowd@test.local",
                "commit",
                "-m",
                "seed",
            ],
        );
        std::fs::canonicalize(dir).expect("canonicalize repo path")
    }

    /// Happy path of the post-`git worktree add` identity guard: a real
    /// repo plus a worktree it actually owns must be accepted. This
    /// also pins the canonicalisation behaviour — the helper has to
    /// resolve the relative `.git` path returned from the primary
    /// repo and the absolute path returned from the linked worktree
    /// to the same on-disk location.
    #[tokio::test]
    async fn assert_worktree_identity_accepts_legitimate_worktree() {
        let repo_dir = tempfile::tempdir().expect("repo dir");
        let repo = init_repo(repo_dir.path());

        let wt_parent = tempfile::tempdir().expect("worktree parent");
        let wt_path = wt_parent.path().join("wt");
        git_sync(
            &repo,
            &["worktree", "add", wt_path.to_str().unwrap(), "HEAD"],
        );

        assert_worktree_identity(&repo, &wt_path)
            .await
            .expect("a worktree git itself created must satisfy the identity guard");
    }

    /// Foreign-repo mismatch: a worktree that legitimately belongs to
    /// repo B must be rejected when the guard is asked whether it
    /// belongs to repo A. This is the README-promised defence against
    /// `git worktree add` quietly landing in the wrong repository
    /// (stale linked worktree, symlinked path, nested checkout).
    /// The error must name both common dirs so the operator can see
    /// at a glance which tree won and which they expected.
    #[tokio::test]
    async fn assert_worktree_identity_rejects_foreign_repo_worktree() {
        let expected_dir = tempfile::tempdir().expect("expected repo dir");
        let expected_repo = init_repo(expected_dir.path());
        let foreign_dir = tempfile::tempdir().expect("foreign repo dir");
        let foreign_repo = init_repo(foreign_dir.path());

        let wt_parent = tempfile::tempdir().expect("worktree parent");
        let wt_path = wt_parent.path().join("wt-of-foreign");
        git_sync(
            &foreign_repo,
            &["worktree", "add", wt_path.to_str().unwrap(), "HEAD"],
        );

        let err = assert_worktree_identity(&expected_repo, &wt_path)
            .await
            .expect_err("guard must reject a worktree that belongs to a foreign repo");
        let msg = err.to_string();
        assert!(
            msg.contains("refusing to dispatch step against the wrong repository"),
            "guard error must explain why the step was refused; got: {msg}"
        );
        assert!(
            msg.contains("git-common-dir"),
            "guard error must surface the common-dir comparison; got: {msg}"
        );
    }

    /// Cleanup helper: a worktree git registered must be unregistered
    /// from the source repo's `.git/worktrees` admin dir AND removed
    /// from disk, so the next step's `path.exists()` precondition in
    /// `prepare_step` does not trip on an orphaned directory.
    #[tokio::test]
    async fn cleanup_rejected_worktree_removes_real_worktree_and_admin_entry() {
        let repo_dir = tempfile::tempdir().expect("repo dir");
        let repo = init_repo(repo_dir.path());

        let wt_parent = tempfile::tempdir().expect("worktree parent");
        let wt_path = wt_parent.path().join("doomed");
        let wt_arg = wt_path.to_string_lossy().into_owned();
        git_sync(&repo, &["worktree", "add", &wt_arg, "HEAD"]);
        assert!(
            wt_path.exists(),
            "sanity: worktree must exist before cleanup"
        );

        cleanup_rejected_worktree(&repo, &wt_path, &wt_arg).await;

        assert!(
            !wt_path.exists(),
            "cleanup must remove the worktree directory at {}",
            wt_path.display()
        );
        let admin = repo.join(".git/worktrees/doomed");
        assert!(
            !admin.exists(),
            "cleanup must let git remove the admin entry at {}; \
             leaving it behind makes the worktree name unreusable",
            admin.display()
        );
    }

    /// Cleanup must still take the directory off disk when
    /// `git worktree remove --force` cannot — for example because the
    /// directory was never registered as a worktree at all (a stale
    /// leftover from an aborted operation, or a nested checkout the
    /// guard caught before git would have linked it). Otherwise the
    /// next attempt to add a worktree at the same path would fail the
    /// `path.exists()` precondition and the operator would be stuck.
    #[tokio::test]
    async fn cleanup_rejected_worktree_falls_back_to_rmdir_when_git_refuses() {
        let repo_dir = tempfile::tempdir().expect("repo dir");
        let repo = init_repo(repo_dir.path());

        let wt_parent = tempfile::tempdir().expect("worktree parent");
        let wt_path = wt_parent.path().join("not-a-worktree");
        let wt_arg = wt_path.to_string_lossy().into_owned();
        std::fs::create_dir_all(&wt_path).unwrap();
        std::fs::write(wt_path.join("placeholder"), "x\n").unwrap();
        assert!(
            wt_path.exists(),
            "sanity: directory must exist before cleanup"
        );

        cleanup_rejected_worktree(&repo, &wt_path, &wt_arg).await;

        assert!(
            !wt_path.exists(),
            "cleanup fallback must rmdir an unregistered directory at {}",
            wt_path.display()
        );
    }

    /// The `modelUsage` block is the one place upstream swaps to
    /// camelCase. Verify the serde renames actually bite: if any of
    /// them regress to `snake_case`, the inner fields silently zero out.
    #[test]
    #[allow(non_snake_case)]
    fn validate_envelope_parses_modelUsage_camelcase() {
        let env = envelope_from_blob(
            r#"{
                "is_error": false,
                "result": "ok",
                "modelUsage": {
                    "claude-opus-4-7": {
                        "inputTokens": 11,
                        "outputTokens": 22,
                        "cacheCreationInputTokens": 33,
                        "cacheReadInputTokens": 44,
                        "costUSD": 0.55
                    }
                }
            }"#,
        );
        let (_, metrics) = validate_claude_envelope("step-1", &env).unwrap();
        let m = metrics.expect("metrics present on success");
        let entry = m
            .model_usage
            .get("claude-opus-4-7")
            .expect("camelCase model-usage key must decode");
        assert_eq!(entry.input_tokens, 11);
        assert_eq!(entry.output_tokens, 22);
        assert_eq!(entry.cache_creation_input_tokens, 33);
        assert_eq!(entry.cache_read_input_tokens, 44);
        assert!((entry.cost_usd - 0.55).abs() < f64::EPSILON);
    }

    // ----------------------------------------------------------------------
    // Live smoke tests against a real `claude` binary.
    //
    // These are gated with `#[ignore]` so default `cargo test` runs do not
    // make API calls. Run explicitly with:
    //
    //     cargo test -p flowd-cli --bin flowd live_spawner -- --ignored
    //
    // They skip with a printed warning (and return `Ok`) when `claude` is
    // not on `$PATH`, so CI hosts without the binary do not flag spurious
    // failures. Each costs ~$0.05-0.15 of real API spend; keep them tight.
    // ----------------------------------------------------------------------

    /// Returns `Some(path)` if `claude` is callable; `None` if the binary
    /// is missing. Used by every live test as the skip predicate.
    fn claude_path_or_skip(test_name: &str) -> Option<String> {
        which_in_path("claude").or_else(|| {
            eprintln!(
                "skipping {test_name}: `claude` not found on $PATH; \
                 install Claude Code or unset FLOWD_AGENT_BIN to enable"
            );
            None
        })
    }

    /// Live: a no-tool-required prompt should round-trip cleanly --
    /// JSON envelope parsed, `result` extracted, no permission denials.
    /// Validates the happy path of `--output-format=json` injection.
    #[tokio::test]
    #[ignore = "live: hits real claude API, costs money, skipped by default"]
    async fn live_spawner_claude_simple_reply() {
        let Some(claude) = claude_path_or_skip("live_spawner_claude_simple_reply") else {
            return;
        };
        let s = LocalShellSpawner::new(claude);
        let out = s
            .spawn(ctx(), &step("Reply with exactly the two characters: OK"))
            .await
            .expect("live claude smoke: simple reply must succeed");

        assert_eq!(out.exit_code, Some(0), "claude exited non-zero: {out:?}");
        // The interpreter strips the JSON envelope and returns just the
        // model's `result` field. If we still see raw JSON it means the
        // claude-like detection or the parse path silently regressed.
        assert!(
            !out.stdout.contains("\"is_error\""),
            "stdout still wrapped in JSON envelope; the stream-json parser did not run: {out:?}"
        );
        assert!(
            out.stdout.to_uppercase().contains("OK"),
            "expected 'OK' in result, got: {stdout}",
            stdout = out.stdout
        );

        // Live wire-up check: a real claude run must populate metrics
        // and report non-zero spend. If either fails, the JSON-envelope
        // contract with upstream has drifted and `From<&ClaudeJsonEnvelope>`
        // needs a look.
        assert!(
            out.metrics.is_some(),
            "expected metrics on live claude run; got: {out:?}"
        );
        let m = out.metrics.clone().unwrap();
        assert!(
            m.total_cost_usd > 0.0,
            "expected non-zero total_cost_usd on live claude run; got: {m:?}"
        );
    }

    /// Live: THE smoke test for the `--dangerously-skip-permissions`
    /// fix. Asks claude to use the `Write` tool to create a file in a
    /// tempdir. Without the injected flag, headless mode silently
    /// denies the write while exit-code stays 0 -- the very bug this
    /// patch exists to kill. We assert on the post-spawn filesystem
    /// state, not just the exit code, so the test cannot be fooled by
    /// a chatty refusal.
    #[tokio::test]
    #[ignore = "live: hits real claude API, performs disk write, skipped by default"]
    async fn live_spawner_claude_writes_file_in_cwd() {
        let Some(claude) = claude_path_or_skip("live_spawner_claude_writes_file_in_cwd") else {
            return;
        };
        let workdir = tempfile::tempdir().expect("tempdir");
        let target_rel = "flowd-spawner-smoketest.txt";
        let target_abs = workdir.path().join(target_rel);

        // Construct the spawner with cwd pinned to the tempdir. `new()`
        // does not expose cwd; use struct-literal access (legal from a
        // child module of the crate).
        let s = LocalShellSpawner {
            bin: OsString::from(claude),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: Some(workdir.path().to_path_buf()),
            extra_args: Vec::new(),
            claude_bare: live_claude_bare(),
        };

        let prompt = format!(
            "Use the Write tool to create exactly one file at the relative path \
             `{target_rel}` containing the literal text `OK`. \
             Do not list directories. Do not run shell commands. Do not edit any \
             other file. Reply with the single word `done` when complete."
        );
        let out = s
            .spawn(ctx(), &step(&prompt))
            .await
            .expect("live claude smoke: write step must succeed");

        assert_eq!(out.exit_code, Some(0), "claude exited non-zero: {out:?}");

        // The contractual assertion: the file is on disk. If this fails,
        // either the spawner is not injecting --dangerously-skip-permissions,
        // or claude refused for another reason and the JSON-envelope failure
        // detection should already have surfaced an Err above.
        assert!(
            target_abs.exists(),
            "claude reported success but did not create {target_abs:?}; \
             stdout was: {stdout}",
            stdout = out.stdout
        );
        let body = std::fs::read_to_string(&target_abs).expect("read created file");
        assert!(
            body.contains("OK"),
            "file exists but body does not contain 'OK': {body:?}"
        );
    }

    /// Live: when an operator forces a non-stream-json output format
    /// via `extra_args`, the spawner must fail loudly rather than
    /// silently pass plain text up the stack. Under stream-json the
    /// daemon depends on the `result` event for metrics and on the
    /// `init` event for `session_id`; absent both, the run cannot be
    /// recorded as success. Operators who genuinely need plain-text
    /// output must wrap the binary (point `FLOWD_AGENT_BIN` at a
    /// shell script that strips the injected flags) -- there is no
    /// in-band escape hatch via `extra_args` anymore.
    #[tokio::test]
    #[ignore = "live: hits real claude API, skipped by default"]
    async fn live_spawner_claude_text_override_fails_with_actionable_error() {
        let Some(claude) =
            claude_path_or_skip("live_spawner_claude_text_override_fails_with_actionable_error")
        else {
            return;
        };
        // Injected default ends with `--output-format=stream-json --verbose`.
        // Operator-supplied extra_args come AFTER injected args, so a
        // trailing `--output-format=text` wins per last-flag-wins
        // parsing -- and the stream parser sees no `result` event.
        let s = LocalShellSpawner {
            bin: OsString::from(claude),
            prompt_flag: DEFAULT_PROMPT_FLAG.to_owned(),
            cwd: None,
            extra_args: vec!["--output-format=text".to_owned()],
            claude_bare: live_claude_bare(),
        };
        let err = s
            .spawn(ctx(), &step("Reply with exactly the two characters: OK"))
            .await
            .expect_err("forcing text output must fail under stream-json");
        let msg = err.to_string();
        assert!(
            msg.contains("no `result` event"),
            "expected error to mention missing result event; got: {msg}"
        );
    }
}
