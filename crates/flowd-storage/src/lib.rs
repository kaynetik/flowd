//! `SQLite` + FTS5 storage backend for flowd.
//!
//! Implements `MemoryBackend` from `flowd-core` using rusqlite with
//! bundled `SQLite` and FTS5 for full-text keyword search.

pub mod migrations;
pub mod sqlite;
