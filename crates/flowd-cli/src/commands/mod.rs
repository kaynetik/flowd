//! Subcommand handlers.
//!
//! Each module exposes a single async `run` function that takes the
//! relevant clap args plus the shared [`crate::paths::FlowdPaths`] and
//! [`crate::output::Style`]. The command modules do not touch clap types,
//! so they stay trivially testable.

pub mod export;
pub mod history;
pub mod hook;
pub mod init;
pub mod observe;
pub mod plan;
pub mod rules;
pub mod search;
pub mod start;
pub mod status;
pub mod stop;
