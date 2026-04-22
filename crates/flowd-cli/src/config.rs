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
//! #   * "stub"      -- StubPlanCompiler (default; deterministic markdown parser).
//! #   * "rejecting" -- RejectingPlanCompiler (every prose-first call errors;
//! #                    use this when prose-first must be disabled at the
//! #                    deployment level).
//! # The "llm" value is reserved for the upcoming LlmPlanCompiler.
//! compiler = "stub"
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

/// Compiler implementation the daemon should instantiate.
///
/// Kept deliberately small. The future `Llm` variant is reserved so we
/// can extend the `Display` impl and the `try_from_str` parser without
/// breaking the on-disk grammar; today using `compiler = "llm"` errors
/// with a "not yet wired" message that points at the next-PR work.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompilerSelection {
    Stub,
    Rejecting,
}

impl CompilerSelection {
    fn try_from_str(raw: &str) -> Result<Self> {
        match raw.trim().to_ascii_lowercase().as_str() {
            "stub" => Ok(Self::Stub),
            "rejecting" => Ok(Self::Rejecting),
            "llm" => Err(anyhow!(
                "[plan] compiler = \"llm\" is reserved for the upcoming LlmPlanCompiler \
                 and is not yet wired; use \"stub\" until the follow-up PR lands"
            )),
            other => Err(anyhow!(
                "[plan] compiler = \"{other}\" is not a known compiler; \
                 expected one of: \"stub\", \"rejecting\""
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
        }
    }
}

/// `[plan]` section. Splits the parse-side raw struct from the
/// resolved [`PlanConfig`] to keep defaulting explicit.
#[derive(Debug, Default, Clone, Deserialize)]
struct RawPlanConfig {
    #[serde(default)]
    max_questions: Option<usize>,
    #[serde(default)]
    compiler: Option<String>,
}

/// Resolved `[plan]` section, all fields populated.
#[derive(Debug, Clone, Copy)]
pub struct PlanConfig {
    pub max_questions: usize,
    pub compiler: CompilerSelection,
}

impl Default for PlanConfig {
    fn default() -> Self {
        Self {
            max_questions: DEFAULT_MAX_QUESTIONS,
            compiler: CompilerSelection::Stub,
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
        Ok(Self {
            max_questions,
            compiler,
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
#[derive(Debug, Default, Clone, Copy)]
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

    #[test]
    fn missing_file_returns_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let cfg = FlowdConfig::load(&dir.path().join("absent.toml")).unwrap();
        assert_eq!(cfg.plan.max_questions, DEFAULT_MAX_QUESTIONS);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
    }

    #[test]
    fn empty_table_keeps_all_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, "[plan]\n").unwrap();
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.max_questions, DEFAULT_MAX_QUESTIONS);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Stub);
    }

    #[test]
    fn explicit_overrides_are_honoured() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, "[plan]\nmax_questions = 4\ncompiler = \"rejecting\"\n").unwrap();
        let cfg = FlowdConfig::load(&p).unwrap();
        assert_eq!(cfg.plan.max_questions, 4);
        assert_eq!(cfg.plan.compiler, CompilerSelection::Rejecting);
    }

    #[test]
    fn unknown_compiler_value_fails_loudly() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, "[plan]\ncompiler = \"gpt\"\n").unwrap();
        let err = FlowdConfig::load(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not a known compiler"), "{msg}");
    }

    #[test]
    fn llm_compiler_value_errors_with_pointer_to_followup() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, "[plan]\ncompiler = \"llm\"\n").unwrap();
        let err = FlowdConfig::load(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("not yet wired"), "{msg}");
    }

    #[test]
    fn zero_max_questions_is_rejected() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("flowd.toml");
        std::fs::write(&p, "[plan]\nmax_questions = 0\n").unwrap();
        let err = FlowdConfig::load(&p).unwrap_err();
        let msg = format!("{err:#}");
        assert!(msg.contains("max_questions"), "{msg}");
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
    }
}
