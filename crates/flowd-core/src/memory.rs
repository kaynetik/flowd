//! Memory subsystem traits and composition.
//!
//! Defines the storage, vector index, embedding, and summarization abstractions
//! plus the high-level `MemoryService` that composes them. Storage
//! implementations live in their respective crates (flowd-storage,
//! flowd-vector, flowd-onnx); the `Summarizer` is supplied by the MCP layer.

pub mod compactor;
pub mod context;
pub mod service;
pub mod tier;

use std::collections::HashMap;
use std::future::Future;

use crate::error::Result;
use crate::types::{
    Embedding, MemoryTier, Observation, SearchQuery, SearchResult, SearchSource, Session,
};
use chrono::{DateTime, Utc};
use uuid::Uuid;

/// RRF constant (standard value from the literature).
const RRF_K: f64 = 60.0;

/// Structured storage backend (`SQLite` implementation).
///
/// Handles persistence of observations, sessions, and FTS5 keyword search.
pub trait MemoryBackend: Send + Sync {
    /// Store a new observation.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on database write failure.
    fn store(&self, observation: &Observation) -> impl Future<Output = Result<()>> + Send;

    /// Retrieve an observation by ID.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on database read failure.
    fn get(&self, id: Uuid) -> impl Future<Output = Result<Option<Observation>>> + Send;

    /// Full-text keyword search via FTS5. Returns observations ranked by relevance.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on query failure.
    fn keyword_search(
        &self,
        query: &SearchQuery,
    ) -> impl Future<Output = Result<Vec<(Observation, f64)>>> + Send;

    /// List sessions, optionally filtered by project.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on query failure.
    fn list_sessions(
        &self,
        project: Option<&str>,
    ) -> impl Future<Output = Result<Vec<Session>>> + Send;

    /// Create or update a session.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on database write failure.
    fn upsert_session(&self, session: &Session) -> impl Future<Output = Result<()>> + Send;

    /// Delete all observations stored at the given tier.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on compaction failure.
    fn compact(&self, tier: MemoryTier) -> impl Future<Output = Result<u64>> + Send;

    /// List observations at a given tier older than `cutoff`, ordered by `session_id`.
    ///
    /// Used by the compactor to discover candidates for summarization or archival.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on query failure.
    fn list_by_tier_and_age(
        &self,
        tier: MemoryTier,
        cutoff: DateTime<Utc>,
    ) -> impl Future<Output = Result<Vec<Observation>>> + Send;

    /// Atomically move a set of observations to a new tier.
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on update failure.
    fn update_tier(
        &self,
        ids: &[Uuid],
        new_tier: MemoryTier,
    ) -> impl Future<Output = Result<u64>> + Send;

    /// Delete a set of observations by ID (hard delete from storage and FTS).
    ///
    /// # Errors
    /// Returns `FlowdError::Storage` on delete failure.
    fn delete_observations(&self, ids: &[Uuid]) -> impl Future<Output = Result<u64>> + Send;
}

/// Delegates session summarization to an external LLM.
///
/// The MCP server implements this by calling back into the connected Claude /
/// Cursor client. Keeping the trait in `flowd-core` lets the compactor stay
/// transport-agnostic.
pub trait Summarizer: Send + Sync {
    /// Produce a compressed summary of a session's observations.
    ///
    /// # Errors
    /// Returns `FlowdError::Internal` if the LLM callback fails.
    fn summarize(
        &self,
        session: &Session,
        observations: &[Observation],
    ) -> impl Future<Output = Result<String>> + Send;
}

/// Vector index backend (Qdrant implementation).
///
/// Handles ANN search over embedding vectors with payload filtering.
pub trait VectorIndex: Send + Sync {
    /// Insert an embedding for an observation.
    ///
    /// # Errors
    /// Returns `FlowdError::Vector` on index write failure.
    fn upsert(&self, embedding: &Embedding) -> impl Future<Output = Result<()>> + Send;

    /// Search for similar vectors. Returns observation IDs with similarity scores.
    ///
    /// # Errors
    /// Returns `FlowdError::Vector` on search failure.
    fn search(
        &self,
        query_vector: &[f32],
        limit: usize,
        project_filter: Option<&str>,
    ) -> impl Future<Output = Result<Vec<(Uuid, f64)>>> + Send;

    /// Remove an embedding by observation ID.
    ///
    /// # Errors
    /// Returns `FlowdError::Vector` on deletion failure.
    fn delete(&self, observation_id: Uuid) -> impl Future<Output = Result<()>> + Send;
}

/// Embedding generation provider (ONNX implementation).
///
/// Generates dense vector embeddings from text input.
pub trait EmbeddingProvider: Send + Sync {
    /// Generate an embedding for a single text input.
    ///
    /// # Errors
    /// Returns `FlowdError::Embedding` if inference fails.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Batch embed multiple texts. Implementations should parallelize via rayon.
    ///
    /// # Errors
    /// Returns `FlowdError::Embedding` if any inference call fails.
    fn embed_batch(&self, texts: &[&str]) -> Result<Vec<Vec<f32>>>;

    /// Dimensionality of the output vectors.
    fn dimensions(&self) -> usize;
}

/// Hybrid search combining keyword and vector results via Reciprocal Rank Fusion.
///
/// Borrows the three backends for the duration of the search so that callers
/// can share them (via `Arc`, a service, etc.) across multiple concurrent
/// queries without giving up ownership.
pub struct HybridSearch<'a, M: ?Sized, V: ?Sized, E: ?Sized> {
    pub memory: &'a M,
    pub vector: &'a V,
    pub embedder: &'a E,
}

impl<'a, M, V, E> HybridSearch<'a, M, V, E>
where
    M: MemoryBackend + ?Sized,
    V: VectorIndex + ?Sized,
    E: EmbeddingProvider + ?Sized,
{
    #[must_use]
    pub fn new(memory: &'a M, vector: &'a V, embedder: &'a E) -> Self {
        Self {
            memory,
            vector,
            embedder,
        }
    }

    /// Execute a hybrid search: FTS5 keyword + Qdrant ANN in parallel, merged via RRF.
    ///
    /// # Errors
    /// Returns errors from underlying storage, vector, or embedding providers.
    #[allow(clippy::cast_precision_loss)]
    pub async fn search(&self, query: &SearchQuery) -> Result<Vec<SearchResult>> {
        let query_vector = self.embedder.embed(&query.text)?;

        let (keyword_results, vector_results) = tokio::join!(
            self.memory.keyword_search(query),
            self.vector
                .search(&query_vector, query.limit * 2, query.project.as_deref()),
        );

        let keyword_results = keyword_results?;
        let vector_results = vector_results?;

        let mut scores: HashMap<Uuid, (f64, SearchSource, Option<Observation>)> = HashMap::new();

        for (rank, (obs, _fts_score)) in keyword_results.into_iter().enumerate() {
            let rrf = 1.0 / (RRF_K + rank as f64 + 1.0);
            scores.insert(obs.id, (rrf, SearchSource::Keyword, Some(obs)));
        }

        for (rank, (obs_id, _sim_score)) in vector_results.into_iter().enumerate() {
            let rrf = 1.0 / (RRF_K + rank as f64 + 1.0);
            scores
                .entry(obs_id)
                .and_modify(|(score, source, _)| {
                    *score += rrf;
                    *source = SearchSource::Both;
                })
                .or_insert((rrf, SearchSource::Vector, None));
        }

        let mut results = Vec::with_capacity(scores.len());
        for (obs_id, (rrf_score, source, maybe_obs)) in scores {
            let observation = match maybe_obs {
                Some(obs) => obs,
                None => match self.memory.get(obs_id).await? {
                    Some(obs) => obs,
                    None => continue,
                },
            };
            results.push(SearchResult {
                observation,
                rrf_score,
                source,
            });
        }

        results.sort_by(|a, b| {
            b.rrf_score
                .partial_cmp(&a.rrf_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.truncate(query.limit);

        Ok(results)
    }
}
