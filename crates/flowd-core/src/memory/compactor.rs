//! Background compaction of tiered memory.
//!
//! A single long-lived tokio task that wakes up periodically. When the daemon
//! has been idle longer than `idle_threshold`, the task:
//!
//! 1. Loads observations past their tier's age threshold.
//! 2. For hot observations: groups them by session, asks the `Summarizer` for
//!    a compact summary, stores the summary as a new warm observation, and
//!    deletes the originals (from SQL + vector index).
//! 3. For warm observations: demotes them to cold in place; they stay
//!    searchable but are excluded from auto-context injection.

use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use serde_json::json;
use tokio::sync::Notify;
use uuid::Uuid;

use crate::error::Result;
use crate::memory::tier::TieringPolicy;
use crate::memory::{EmbeddingProvider, MemoryBackend, Summarizer, VectorIndex};
use crate::types::{Embedding, MemoryTier, Observation, Session};

/// Tracks the wall-clock time of the last external activity.
///
/// Cheap to clone (internally `Arc<RwLock<Instant>>`) and safe to share across
/// threads. `touch()` is called from request handlers; the compactor polls
/// `elapsed()` to detect idle periods.
#[derive(Debug, Clone)]
pub struct ActivityMonitor {
    last: Arc<std::sync::RwLock<Instant>>,
}

impl ActivityMonitor {
    #[must_use]
    pub fn new() -> Self {
        Self {
            last: Arc::new(std::sync::RwLock::new(Instant::now())),
        }
    }

    /// Record that activity has just occurred.
    pub fn touch(&self) {
        if let Ok(mut guard) = self.last.write() {
            *guard = Instant::now();
        }
    }

    /// Duration since the last `touch()` call.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.last.read().map(|g| g.elapsed()).unwrap_or_default()
    }
}

impl Default for ActivityMonitor {
    fn default() -> Self {
        Self::new()
    }
}

/// Compactor configuration.
#[derive(Debug, Clone, Copy)]
pub struct CompactorConfig {
    /// How often the compactor wakes up to check whether to run.
    pub check_interval: Duration,
    /// How long the daemon must be idle before compaction runs.
    pub idle_threshold: Duration,
    /// Maximum observations processed per compaction pass (back-pressure).
    pub batch_limit: usize,
}

impl Default for CompactorConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(60),
            idle_threshold: Duration::from_secs(5 * 60),
            batch_limit: 256,
        }
    }
}

/// Handle returned by `Compactor::spawn`. Dropping it signals the background
/// task to shut down at its next tick.
#[derive(Debug)]
pub struct CompactorHandle {
    shutdown: Arc<Notify>,
    join: Option<tokio::task::JoinHandle<()>>,
}

impl CompactorHandle {
    /// Signal shutdown and await the background task.
    pub async fn stop(mut self) {
        self.shutdown.notify_waiters();
        if let Some(join) = self.join.take() {
            let _ = join.await;
        }
    }
}

impl Drop for CompactorHandle {
    fn drop(&mut self) {
        self.shutdown.notify_waiters();
    }
}

/// Background compaction coordinator.
pub struct Compactor<M, V, E, S> {
    memory: Arc<M>,
    vector: Arc<V>,
    embedder: Arc<E>,
    summarizer: Arc<S>,
    policy: TieringPolicy,
    monitor: ActivityMonitor,
    config: CompactorConfig,
}

impl<M, V, E, S> Compactor<M, V, E, S>
where
    M: MemoryBackend + Send + Sync + 'static,
    V: VectorIndex + Send + Sync + 'static,
    E: EmbeddingProvider + Send + Sync + 'static,
    S: Summarizer + 'static,
{
    #[must_use]
    pub fn new(
        memory: Arc<M>,
        vector: Arc<V>,
        embedder: Arc<E>,
        summarizer: Arc<S>,
        policy: TieringPolicy,
        monitor: ActivityMonitor,
        config: CompactorConfig,
    ) -> Self {
        Self {
            memory,
            vector,
            embedder,
            summarizer,
            policy,
            monitor,
            config,
        }
    }

    /// Spawn the compactor as a long-running tokio task.
    #[must_use]
    pub fn spawn(self) -> CompactorHandle {
        let shutdown = Arc::new(Notify::new());
        let shutdown_task = Arc::clone(&shutdown);

        let join = tokio::spawn(async move {
            self.run(shutdown_task).await;
        });

        CompactorHandle {
            shutdown,
            join: Some(join),
        }
    }

    async fn run(self, shutdown: Arc<Notify>) {
        let mut ticker = tokio::time::interval(self.config.check_interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

        loop {
            tokio::select! {
                _ = ticker.tick() => {
                    if self.monitor.elapsed() < self.config.idle_threshold {
                        continue;
                    }
                    if let Err(e) = self.compact_once().await {
                        tracing::warn!(error = %e, "compaction pass failed");
                    }
                }
                () = shutdown.notified() => {
                    tracing::info!("compactor shutting down");
                    return;
                }
            }
        }
    }

    /// Execute a single compaction pass. Exposed for tests and CLI use.
    ///
    /// # Errors
    /// Returns errors from the underlying backends; individual session
    /// failures are logged and skipped so one bad session does not block
    /// the rest of the batch.
    pub async fn compact_once(&self) -> Result<CompactionStats> {
        let now = Utc::now();
        let mut stats = CompactionStats::default();

        // ---- Hot → Warm: summarize & replace ----
        let hot_cutoff = now - self.policy.hot_max_age();
        let mut hot = self
            .memory
            .list_by_tier_and_age(MemoryTier::Hot, hot_cutoff)
            .await?;
        hot.truncate(self.config.batch_limit);

        if !hot.is_empty() {
            let sessions = self.memory.list_sessions(None).await?;
            let grouped = group_by_session(hot);

            for (session_id, observations) in grouped {
                // Fall back to a synthesized session when the row is
                // missing (legacy data or imports). `summarize_and_replace`
                // upserts before writing the summary so the FK
                // observations.session_id -> sessions.id is satisfied.
                let session = sessions
                    .iter()
                    .find(|s| s.id == session_id)
                    .cloned()
                    .unwrap_or_else(|| placeholder_session(session_id, &observations));

                match self.summarize_and_replace(&session, &observations).await {
                    Ok(()) => {
                        stats.sessions_summarized += 1;
                        stats.hot_replaced += observations.len();
                    }
                    Err(e) => {
                        tracing::warn!(
                            session_id = %session_id,
                            error = %e,
                            "failed to summarize session; leaving observations in hot tier"
                        );
                    }
                }
            }
        }

        // ---- Warm → Cold: in-place tier downgrade ----
        let warm_cutoff = now - self.policy.warm_max_age();
        let mut warm = self
            .memory
            .list_by_tier_and_age(MemoryTier::Warm, warm_cutoff)
            .await?;
        warm.truncate(self.config.batch_limit);

        if !warm.is_empty() {
            let ids: Vec<Uuid> = warm.iter().map(|o| o.id).collect();
            let updated = self.memory.update_tier(&ids, MemoryTier::Cold).await?;
            stats.warm_demoted += usize::try_from(updated).unwrap_or(usize::MAX);
        }

        tracing::info!(
            sessions_summarized = stats.sessions_summarized,
            hot_replaced = stats.hot_replaced,
            warm_demoted = stats.warm_demoted,
            "compaction pass complete"
        );
        Ok(stats)
    }

    async fn summarize_and_replace(
        &self,
        session: &Session,
        observations: &[Observation],
    ) -> Result<()> {
        let summary_text = self.summarizer.summarize(session, observations).await?;

        let now = Utc::now();
        let summary = Observation {
            id: Uuid::new_v4(),
            session_id: session.id,
            project: session.project.clone(),
            content: summary_text.clone(),
            tier: MemoryTier::Warm,
            metadata: json!({
                "kind": "summary",
                "source_count": observations.len(),
                "source_ids": observations.iter().map(|o| o.id.to_string()).collect::<Vec<_>>(),
            }),
            created_at: now,
            updated_at: now,
        };

        // Idempotently ensure the sessions row exists. Without this,
        // the summary insert can violate the FK
        // `observations.session_id REFERENCES sessions(id)` whenever
        // the session row is missing -- e.g. data imported before the
        // MCP layer started calling `ensure_session`, a placeholder
        // synthesized in `compact_once`, or a manually deleted
        // `sessions` row. `upsert_session` is `ON CONFLICT DO UPDATE`
        // so the happy path is a near-no-op.
        self.memory.upsert_session(session).await?;

        self.memory.store(&summary).await?;

        let embedder = Arc::clone(&self.embedder);
        let text = summary_text;
        let vec = tokio::task::spawn_blocking(move || embedder.embed(&text))
            .await
            .map_err(|e| {
                crate::error::FlowdError::Embedding(format!("embed task panicked: {e}"))
            })??;

        self.vector
            .upsert(&Embedding {
                observation_id: summary.id,
                project: summary.project.clone(),
                vector: vec,
            })
            .await?;

        let ids: Vec<Uuid> = observations.iter().map(|o| o.id).collect();
        for id in &ids {
            // Vector-index deletions are best-effort: an orphaned embedding
            // cannot be returned by search once its observation row is gone
            // (`HybridSearch` rehydrates through the memory backend), so we
            // log failures but still proceed with the SQL delete.
            if let Err(e) = self.vector.delete(*id).await {
                tracing::warn!(observation_id = %id, error = %e, "vector delete failed");
            }
        }
        self.memory.delete_observations(&ids).await?;

        Ok(())
    }
}

/// Outcome of a single compaction pass.
#[derive(Debug, Default, Clone, Copy)]
pub struct CompactionStats {
    pub sessions_summarized: usize,
    pub hot_replaced: usize,
    pub warm_demoted: usize,
}

fn group_by_session(observations: Vec<Observation>) -> Vec<(Uuid, Vec<Observation>)> {
    let mut out: Vec<(Uuid, Vec<Observation>)> = Vec::new();
    for obs in observations {
        match out.iter_mut().find(|(sid, _)| *sid == obs.session_id) {
            Some(entry) => entry.1.push(obs),
            None => out.push((obs.session_id, vec![obs])),
        }
    }
    out
}

fn placeholder_session(id: Uuid, observations: &[Observation]) -> Session {
    let project = observations
        .first()
        .map(|o| o.project.clone())
        .unwrap_or_default();
    let started_at = observations
        .iter()
        .map(|o| o.created_at)
        .min()
        .unwrap_or_else(Utc::now);
    Session {
        id,
        project,
        summary: None,
        started_at,
        ended_at: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activity_monitor_tracks_touches() {
        let m = ActivityMonitor::new();
        std::thread::sleep(Duration::from_millis(10));
        let before = m.elapsed();
        m.touch();
        let after = m.elapsed();
        assert!(after < before);
    }

    #[test]
    fn group_by_session_preserves_order() {
        let s1 = Uuid::new_v4();
        let s2 = Uuid::new_v4();
        let mk = |sid| Observation {
            id: Uuid::new_v4(),
            session_id: sid,
            project: "p".into(),
            content: "c".into(),
            tier: MemoryTier::Hot,
            metadata: json!({}),
            created_at: Utc::now(),
            updated_at: Utc::now(),
        };
        let v = vec![mk(s1), mk(s2), mk(s1), mk(s2)];
        let groups = group_by_session(v);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].1.len(), 2);
        assert_eq!(groups[1].1.len(), 2);
    }
}
