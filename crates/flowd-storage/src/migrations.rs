//! Schema migrations for the flowd SQLite database.

#![allow(clippy::doc_markdown)]

use rusqlite::Connection;

const MIGRATIONS: &[&str] = &[
    MIGRATION_001,
    MIGRATION_002,
    MIGRATION_003,
    MIGRATION_004,
    MIGRATION_005,
    MIGRATION_006,
];

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

// SQLite cannot ALTER a column to NOT NULL or add a NOT NULL constraint
// in place, so we rebuild the table. Pre-existing rows with NULL projects
// are backfilled to '__legacy__' so the new constraint never trips on
// historic data; new code paths now require a real project up-front.
//
// NOTE: this is a destructive schema change for the column shape; the row
// data (incl. `definition` JSON) is preserved verbatim.
const MIGRATION_003: &str = r"
CREATE TABLE plans_new (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'draft',
    definition TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    started_at TEXT,
    completed_at TEXT,
    project TEXT NOT NULL DEFAULT '__legacy__'
);

INSERT INTO plans_new (id, name, status, definition, created_at, started_at, completed_at, project)
SELECT id, name, status, definition, created_at, started_at, completed_at,
       COALESCE(NULLIF(TRIM(project), ''), '__legacy__')
FROM plans;

DROP TABLE plans;
ALTER TABLE plans_new RENAME TO plans;

CREATE INDEX IF NOT EXISTS idx_plans_project ON plans(project);
";

// Dedicated audit log for plan-lifecycle events (HL-39). Replaces the
// previous behaviour of writing plan events into the `observations` table
// via `MemoryService`. Keeping plan telemetry separate avoids polluting
// hybrid-search results and the embedding budget with structured logging.
//
// Idempotent (`IF NOT EXISTS`) so re-running on a partially-migrated DB is
// safe, even though the `migrations` table normally guards re-execution.
const MIGRATION_004: &str = r"
CREATE TABLE IF NOT EXISTS plan_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    plan_id     TEXT NOT NULL,
    project     TEXT NOT NULL,
    kind        TEXT NOT NULL,
    step_id     TEXT,
    agent_type  TEXT,
    payload     TEXT NOT NULL,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_plan_events_plan
    ON plan_events(plan_id, created_at);
CREATE INDEX IF NOT EXISTS idx_plan_events_project
    ON plan_events(project, created_at);
CREATE INDEX IF NOT EXISTS idx_plan_events_kind
    ON plan_events(kind);
";

// First-class storage for the prose-first plan front door (HL-44).
//
// Promotes the four clarification fields out of the catch-all `definition`
// JSON blob so we can:
//   * Surface "draft plans with N unresolved questions" in `plan_list`
//     without parsing every blob.
//   * Diff prose changes across rounds via a single column read.
//   * Set defaults that load cleanly even on rows written by older
//     binaries that never knew about these columns.
//
// Defaults (`NULL`, `'[]'`, `'[]'`, `0`) are chosen to round-trip back into
// the [`flowd_core::orchestration::Plan`] defaults (None, empty Vec, empty
// Vec, false). The plan_store read path reads these columns first and only
// falls back to `definition` JSON when both sources disagree -- the column
// wins because it's the canonical home going forward.
const MIGRATION_005: &str = r"
ALTER TABLE plans ADD COLUMN source_doc TEXT;
ALTER TABLE plans ADD COLUMN open_questions_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE plans ADD COLUMN decisions_json TEXT NOT NULL DEFAULT '[]';
ALTER TABLE plans ADD COLUMN definition_dirty INTEGER NOT NULL DEFAULT 0;
";

// Plan-level execution root. Distinct from `project` (the namespace
// label, which stays NOT NULL with the '__legacy__' default from
// MIGRATION_003): `project_root` holds the canonical filesystem path
// the plan was submitted from, captured at plan creation time so
// resume / list / show paths know where the plan was meant to run
// even when the daemon's CWD has since drifted.
//
// Nullable because pre-MIGRATION_006 rows have no value to recover
// (the field didn't exist before this migration); the load path
// preserves NULL as `Plan::project_root = None` and downstream
// callers fall back to per-call discovery.
const MIGRATION_006: &str = r"
ALTER TABLE plans ADD COLUMN project_root TEXT;
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
