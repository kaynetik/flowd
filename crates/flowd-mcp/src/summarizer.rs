//! Summarizer implementations for the memory compactor.
//!
//! The compactor delegates Hot → Warm summarization to a [`Summarizer`]
//! supplied at composition time. Two flavours live here:
//!
//! * [`NoopSummarizer`] -- deterministic, offline, dependency-free fallback
//!   that concatenates the observation contents and truncates to a budget.
//!   Always available; safe to use when no MCP client is connected (e.g.
//!   when the compactor runs from `flowd compact` rather than during a live
//!   stdio session).
//! * Future: a `ClientSamplingSummarizer` that calls back into the connected
//!   MCP client via the 2024-11-05 `sampling/createMessage` capability. That
//!   path needs server-initiated requests on the same JSON-RPC stream, which
//!   the current `McpServer` does not yet implement.
//!
//! Both impls satisfy the contract that the produced summary must be
//! self-contained (no external IDs needed to interpret it) so that subsequent
//! `memory_context` calls can return only the summary without losing
//! reproducible meaning.
//!
//! `NoopSummarizer` is intentionally not "smart". It does not call an LLM,
//! does not paraphrase, and never invents content -- if its budget forces
//! truncation it appends a `... [+N more]` marker so the loss is visible.
//! That is strictly preferable to a hallucinated summary, especially when
//! the compactor is the system that gets to *delete* the originals.

use std::fmt::Write as _;

use flowd_core::error::Result;
use flowd_core::memory::Summarizer;
use flowd_core::types::{Observation, Session};

/// Maximum byte length of a generated summary before truncation.
///
/// Roughly 16 KiB, comfortably below any reasonable LLM context budget while
/// still preserving enough signal for retrieval.
const DEFAULT_MAX_SUMMARY_BYTES: usize = 16 * 1024;

/// Per-observation prefix budget. Keeps very long observations from
/// dominating the summary; we trim each before concatenation.
const PER_OBS_BYTES: usize = 1024;

/// Deterministic, offline summarizer used when no LLM callback is available.
///
/// The output is a single string with one bullet per source observation,
/// truncated to a configured byte budget. Bullets are prefixed with the
/// observation id so a reader can cross-reference back to storage if the
/// row hasn't been deleted yet (the compactor *does* delete after
/// summarization, so the ids are most useful in audit logs / forensics).
#[derive(Debug, Clone)]
pub struct NoopSummarizer {
    max_bytes: usize,
    per_obs_bytes: usize,
}

impl Default for NoopSummarizer {
    fn default() -> Self {
        Self {
            max_bytes: DEFAULT_MAX_SUMMARY_BYTES,
            per_obs_bytes: PER_OBS_BYTES,
        }
    }
}

impl NoopSummarizer {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override the total byte budget for the produced summary.
    #[must_use]
    pub fn with_max_bytes(mut self, max_bytes: usize) -> Self {
        self.max_bytes = max_bytes.max(256);
        self
    }

    /// Override the per-observation byte budget.
    #[must_use]
    pub fn with_per_obs_bytes(mut self, per_obs_bytes: usize) -> Self {
        self.per_obs_bytes = per_obs_bytes.max(64);
        self
    }
}

impl Summarizer for NoopSummarizer {
    async fn summarize(&self, session: &Session, observations: &[Observation]) -> Result<String> {
        Ok(render_summary(
            session,
            observations,
            self.max_bytes,
            self.per_obs_bytes,
        ))
    }
}

fn render_summary(
    session: &Session,
    observations: &[Observation],
    max_bytes: usize,
    per_obs_bytes: usize,
) -> String {
    let header = format!(
        "session {session_id} ({project}) -- {n} observation(s) summarized\n",
        session_id = session.id,
        project = if session.project.is_empty() {
            "(no project)"
        } else {
            session.project.as_str()
        },
        n = observations.len(),
    );

    let mut out = String::with_capacity(header.len() + observations.len() * 64);
    out.push_str(&header);

    let mut included = 0usize;
    for obs in observations {
        let snippet = truncate_chars(obs.content.trim(), per_obs_bytes);
        let line = format!("- [{id}] {snippet}\n", id = obs.id, snippet = snippet);

        // +marker accounts for the worst-case "... [+N more]" tail we may
        // need to append; guarantee the final string fits within the budget.
        if out.len() + line.len() + 32 > max_bytes {
            break;
        }
        out.push_str(&line);
        included += 1;
    }

    if included < observations.len() {
        // `write!` into a `String` is infallible; ignore the return.
        let _ = writeln!(out, "... [+{} more]", observations.len() - included);
    }

    out
}

/// Truncate `s` to at most `max_bytes`, respecting UTF-8 char boundaries.
/// Appends an `…` ellipsis when truncation occurs.
fn truncate_chars(s: &str, max_bytes: usize) -> String {
    if s.len() <= max_bytes {
        return s.to_owned();
    }
    // Find the largest char boundary <= max_bytes - 3 (room for ellipsis).
    let target = max_bytes.saturating_sub(3);
    let mut cut = 0usize;
    for (i, _) in s.char_indices() {
        if i > target {
            break;
        }
        cut = i;
    }
    let mut out = s[..cut].to_owned();
    out.push('…');
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use flowd_core::types::MemoryTier;
    use serde_json::json;
    use uuid::Uuid;

    fn obs(content: &str) -> Observation {
        Observation {
            id: Uuid::new_v4(),
            session_id: Uuid::nil(),
            project: "p".into(),
            content: content.into(),
            tier: MemoryTier::Hot,
            metadata: json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        }
    }

    fn session() -> Session {
        Session {
            id: Uuid::nil(),
            project: "p".into(),
            summary: None,
            started_at: Utc::now(),
            ended_at: None,
        }
    }

    #[tokio::test]
    async fn summarizes_all_when_under_budget() {
        let s = NoopSummarizer::new();
        let obs = vec![obs("hello"), obs("world")];
        let out = s.summarize(&session(), &obs).await.unwrap();
        assert!(out.contains("hello"));
        assert!(out.contains("world"));
        assert!(!out.contains("more]"));
    }

    #[tokio::test]
    async fn truncates_when_over_budget() {
        let s = NoopSummarizer::new().with_max_bytes(256);
        let obs: Vec<Observation> = (0..20).map(|i| obs(&"x".repeat(40 + i))).collect();
        let out = s.summarize(&session(), &obs).await.unwrap();
        assert!(
            out.len() <= 256,
            "summary exceeded budget: {} bytes",
            out.len()
        );
        assert!(out.contains("more]"));
    }

    #[test]
    fn truncate_chars_respects_utf8_boundaries() {
        // 4 4-byte chars = 16 bytes; budget 6 forces truncation in the middle.
        let s = "🦀🦀🦀🦀";
        let t = truncate_chars(s, 6);
        assert!(t.ends_with('…'));
        assert!(t.is_char_boundary(t.len() - '…'.len_utf8()));
    }
}
