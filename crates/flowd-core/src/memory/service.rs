//! High-level memory service facade.
//!
//! `MemoryService` wraps a `MemoryBackend`, `VectorIndex`, and
//! `EmbeddingProvider` and exposes the domain-level use cases used by the
//! CLI and MCP layers: recording observations, running hybrid search, and
//! retrieving auto-context for the current session/file.

use std::sync::Arc;

use chrono::Utc;
use serde_json::Value as JsonValue;
use uuid::Uuid;

use crate::error::Result;
use crate::memory::context::AutoContextQuery;
use crate::memory::{EmbeddingProvider, HybridSearch, MemoryBackend, VectorIndex};
use crate::types::{Embedding, MemoryTier, Observation, SearchQuery, SearchResult, Session};

/// Facade that orchestrates the three memory backends.
///
/// Generic over concrete backends to preserve zero-cost dispatch and let
/// callers compose in-memory test doubles alongside the production
/// SQLite/Qdrant/ONNX stack.
pub struct MemoryService<M, V, E> {
    memory: Arc<M>,
    vector: Arc<V>,
    embedder: Arc<E>,
}

impl<M, V, E> MemoryService<M, V, E>
where
    M: MemoryBackend + 'static,
    V: VectorIndex + 'static,
    E: EmbeddingProvider + 'static,
{
    /// Construct a new service from the three backends.
    pub fn new(memory: M, vector: V, embedder: E) -> Self {
        Self {
            memory: Arc::new(memory),
            vector: Arc::new(vector),
            embedder: Arc::new(embedder),
        }
    }

    /// Construct a service from already-shared backends (e.g. to share the
    /// memory backend with a background compactor).
    #[must_use]
    pub fn from_shared(memory: Arc<M>, vector: Arc<V>, embedder: Arc<E>) -> Self {
        Self {
            memory,
            vector,
            embedder,
        }
    }

    #[must_use]
    pub fn memory(&self) -> &Arc<M> {
        &self.memory
    }

    #[must_use]
    pub fn vector(&self) -> &Arc<V> {
        &self.vector
    }

    #[must_use]
    pub fn embedder(&self) -> &Arc<E> {
        &self.embedder
    }

    /// Begin a new session in storage; returns the generated UUID.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on write failure.
    pub async fn start_session(&self, project: &str) -> Result<Uuid> {
        let now = Utc::now();
        let session = Session {
            id: Uuid::new_v4(),
            project: project.to_owned(),
            summary: None,
            started_at: now,
            ended_at: None,
        };
        self.memory.upsert_session(&session).await?;
        Ok(session.id)
    }

    /// Idempotently register a session with a caller-supplied UUID.
    ///
    /// Used by entry points (MCP `memory_store`, CLI `observe`) where the
    /// client owns the session identifier and we must not fail the
    /// observation FK on a never-seen session. Safe to call on every
    /// store: it's one list plus a conditional upsert.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on read or write failure.
    pub async fn ensure_session(&self, project: &str, session_id: Uuid) -> Result<()> {
        let existing = self.memory.list_sessions(Some(project)).await?;
        if existing.iter().any(|s| s.id == session_id) {
            return Ok(());
        }
        let now = Utc::now();
        self.memory
            .upsert_session(&Session {
                id: session_id,
                project: project.to_owned(),
                summary: None,
                started_at: now,
                ended_at: None,
            })
            .await
    }

    /// Mark a session as ended with an optional summary.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on write failure.
    pub async fn end_session(&self, session_id: Uuid, summary: Option<String>) -> Result<()> {
        // Load the existing session to preserve `started_at` and `project`.
        // If no row exists we create a minimal one so callers can safely
        // `end_session` on a session whose `start_session` was never reached
        // (e.g. crash recovery).
        let existing = self.memory.list_sessions(None).await?;
        let mut session = existing
            .into_iter()
            .find(|s| s.id == session_id)
            .unwrap_or(Session {
                id: session_id,
                project: String::new(),
                summary: None,
                started_at: Utc::now(),
                ended_at: None,
            });
        session.summary = summary.or(session.summary);
        session.ended_at = Some(Utc::now());
        self.memory.upsert_session(&session).await
    }

    /// List sessions, optionally filtered by project.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on query failure.
    pub async fn list_sessions(&self, project: Option<&str>) -> Result<Vec<Session>> {
        self.memory.list_sessions(project).await
    }

    /// Record a new observation: stores the row, embeds the content, and
    /// upserts the vector. Returns the observation ID.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on DB failure, `FlowdError::Embedding`
    /// on inference failure, or `FlowdError::Vector` on vector index failure.
    pub async fn record(
        &self,
        project: &str,
        session_id: Uuid,
        content: impl Into<String> + Send,
        metadata: Option<JsonValue>,
    ) -> Result<Uuid> {
        let now = Utc::now();
        let content: String = content.into();
        let observation = Observation {
            id: Uuid::new_v4(),
            session_id,
            project: project.to_owned(),
            content: content.clone(),
            tier: MemoryTier::Hot,
            metadata: metadata.unwrap_or_else(|| JsonValue::Object(serde_json::Map::new())),
            created_at: now,
            updated_at: now,
        };

        self.memory.store(&observation).await?;

        // Embedding runs on a CPU thread pool; run it via spawn_blocking to
        // keep the async runtime responsive for concurrent requests.
        let embedder = Arc::clone(&self.embedder);
        let content_for_embed = content;
        let vector = tokio::task::spawn_blocking(move || embedder.embed(&content_for_embed))
            .await
            .map_err(|e| {
                crate::error::FlowdError::Embedding(format!("embed task panicked: {e}"))
            })??;

        let embedding = Embedding {
            observation_id: observation.id,
            project: observation.project.clone(),
            vector,
        };
        self.vector.upsert(&embedding).await?;

        Ok(observation.id)
    }

    /// Run a hybrid search (FTS5 + ANN via RRF).
    ///
    /// # Errors
    /// Returns errors from the underlying backends.
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        HybridSearch::new(&*self.memory, &*self.vector, &*self.embedder)
            .search(query)
            .await
    }

    /// Retrieve auto-context: hybrid search with the cold tier filtered out.
    ///
    /// Cold observations are kept in storage (searchable on explicit request)
    /// but never auto-injected, per the issue specification.
    ///
    /// # Errors
    /// Returns errors from the underlying backends.
    pub async fn auto_context(&self, ctx: &AutoContextQuery) -> Result<Vec<SearchResult>> {
        let query = ctx.to_search_query();
        let results = HybridSearch::new(&*self.memory, &*self.vector, &*self.embedder)
            .search(&query)
            .await?;

        // `HybridSearch::search` rehydrates vector-only hits from storage, so
        // every result carries a populated `Observation` with an up-to-date
        // tier. Filter out cold entries and clamp to the caller's limit.
        let mut filtered: Vec<SearchResult> = results
            .into_iter()
            .filter(|r| r.observation.tier != MemoryTier::Cold)
            .collect();
        filtered.truncate(ctx.limit);
        Ok(filtered)
    }
}
