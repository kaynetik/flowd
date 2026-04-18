use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Memory tier classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum MemoryTier {
    /// L0: verbatim recent entries.
    Hot,
    /// L1: compressed summaries of older sessions.
    Warm,
    /// L2: archived, searchable but not auto-injected.
    Cold,
}

/// A stored memory observation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    pub id: Uuid,
    pub session_id: Uuid,
    pub project: String,
    pub content: String,
    pub tier: MemoryTier,
    pub metadata: serde_json::Value,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Session metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Session {
    pub id: Uuid,
    pub project: String,
    pub summary: Option<String>,
    pub started_at: DateTime<Utc>,
    pub ended_at: Option<DateTime<Utc>>,
}

/// A search result combining FTS5 and vector scores.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchResult {
    pub observation: Observation,
    /// Reciprocal rank fusion score (higher = more relevant).
    pub rrf_score: f64,
    /// Whether this result came from keyword search, vector search, or both.
    pub source: SearchSource,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SearchSource {
    Keyword,
    Vector,
    Both,
}

/// An embedding vector produced by the ONNX provider.
#[derive(Debug, Clone)]
pub struct Embedding {
    pub observation_id: Uuid,
    pub project: String,
    pub vector: Vec<f32>,
}

/// Search query parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchQuery {
    pub text: String,
    pub project: Option<String>,
    pub since: Option<DateTime<Utc>>,
    pub limit: usize,
}

impl Default for SearchQuery {
    fn default() -> Self {
        Self {
            text: String::new(),
            project: None,
            since: None,
            limit: 10,
        }
    }
}
