//! Schema migrations for the flowd SQLite database.

#![allow(clippy::doc_markdown)]

use rusqlite::Connection;

const MIGRATIONS: &[&str] = &[MIGRATION_001, MIGRATION_002];

const MIGRATION_001: &str = r"
CREATE TABLE IF NOT EXISTS migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    project TEXT NOT NULL,
    summary TEXT,
    started_at TEXT NOT NULL,
    ended_at TEXT
);

CREATE TABLE IF NOT EXISTS observations (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id),
    project TEXT NOT NULL,
    content TEXT NOT NULL,
    tier TEXT NOT NULL DEFAULT 'hot',
    metadata TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS observations_fts USING fts5(
    content,
    content_rowid='rowid',
    tokenize='porter unicode61'
);

CREATE TRIGGER IF NOT EXISTS observations_ai AFTER INSERT ON observations BEGIN
    INSERT INTO observations_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TRIGGER IF NOT EXISTS observations_ad AFTER DELETE ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
END;

CREATE TRIGGER IF NOT EXISTS observations_au AFTER UPDATE OF content ON observations BEGIN
    INSERT INTO observations_fts(observations_fts, rowid, content) VALUES ('delete', old.rowid, old.content);
    INSERT INTO observations_fts(rowid, content) VALUES (new.rowid, new.content);
END;

CREATE TABLE IF NOT EXISTS rules (
    id TEXT PRIMARY KEY,
    scope TEXT NOT NULL,
    level TEXT NOT NULL,
    description TEXT NOT NULL,
    match_pattern TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'file',
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS plans (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT
);

CREATE TABLE IF NOT EXISTS plan_steps (
    id TEXT NOT NULL,
    plan_id TEXT NOT NULL REFERENCES plans(id),
    agent_type TEXT NOT NULL,
    prompt TEXT NOT NULL,
    depends_on TEXT NOT NULL DEFAULT '[]',
    timeout_secs INTEGER,
    retry_count INTEGER NOT NULL DEFAULT 0,
    status TEXT NOT NULL DEFAULT 'pending',
    output TEXT,
    error TEXT,
    started_at TEXT,
    completed_at TEXT,
    PRIMARY KEY (plan_id, id)
);

CREATE INDEX IF NOT EXISTS idx_observations_project ON observations(project);
CREATE INDEX IF NOT EXISTS idx_observations_session ON observations(session_id);
CREATE INDEX IF NOT EXISTS idx_observations_tier ON observations(tier);
CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at);
CREATE INDEX IF NOT EXISTS idx_sessions_project ON sessions(project);
CREATE INDEX IF NOT EXISTS idx_plan_steps_status ON plan_steps(status);
";

const MIGRATION_002: &str = r"
ALTER TABLE plans ADD COLUMN project TEXT;
";

/// Run all pending migrations.
///
/// # Errors
/// Returns `rusqlite::Error` if any migration SQL fails to execute.
#[allow(clippy::cast_possible_wrap)]
pub fn run_migrations(conn: &Connection) -> Result<usize, rusqlite::Error> {
    conn.execute_batch("CREATE TABLE IF NOT EXISTS migrations (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL DEFAULT (datetime('now')))")?;

    let current_version: i64 = conn
        .query_row(
            "SELECT COALESCE(MAX(version), 0) FROM migrations",
            [],
            |row| row.get(0),
        )
        .unwrap_or(0);

    let mut applied = 0;
    for (i, migration) in MIGRATIONS.iter().enumerate() {
        let version = (i + 1) as i64;
        if version > current_version {
            conn.execute_batch(migration)?;
            conn.execute("INSERT INTO migrations (version) VALUES (?1)", [version])?;
            applied += 1;
            tracing::info!(version, "applied migration");
        }
    }

    Ok(applied)
}
