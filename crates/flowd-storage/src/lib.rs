//! `SQLite` + FTS5 storage backend for flowd.
//!
//! Implements `MemoryBackend` from `flowd-core` using rusqlite with
//! bundled `SQLite` and FTS5 for full-text keyword search.

pub mod migrations;
pub mod plan_event_store;
pub mod plan_store;
pub mod sqlite;
pub mod step_branch_store;
