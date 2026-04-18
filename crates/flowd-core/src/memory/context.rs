//! Auto-context retrieval.
//!
//! Given a minimal description of what the caller is currently doing
//! (project, optional file, optional session), synthesize a search query
//! and return the most relevant non-cold memories.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::types::SearchQuery;

/// Describes the caller's current working context for auto-injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoContextQuery {
    pub project: String,
    pub file_path: Option<String>,
    pub session_id: Option<Uuid>,
    pub hint: Option<String>,
    pub limit: usize,
}

impl AutoContextQuery {
    /// Build a new auto-context query with a default limit.
    #[must_use]
    pub fn new(project: impl Into<String>) -> Self {
        Self {
            project: project.into(),
            file_path: None,
            session_id: None,
            hint: None,
            limit: 5,
        }
    }

    #[must_use]
    pub fn with_file(mut self, file_path: impl Into<String>) -> Self {
        self.file_path = Some(file_path.into());
        self
    }

    #[must_use]
    pub fn with_session(mut self, session_id: Uuid) -> Self {
        self.session_id = Some(session_id);
        self
    }

    #[must_use]
    pub fn with_hint(mut self, hint: impl Into<String>) -> Self {
        self.hint = Some(hint.into());
        self
    }

    #[must_use]
    pub fn with_limit(mut self, limit: usize) -> Self {
        self.limit = limit;
        self
    }

    /// Fold the context into a single hybrid-search query string.
    ///
    /// Concatenates the hint (if present), the file path (as a keyword), and
    /// the project name. Empty components are dropped. The FTS5 `MATCH`
    /// grammar is permissive enough to accept this directly; the vector side
    /// uses the same text for embedding.
    #[must_use]
    pub fn to_search_query(&self) -> SearchQuery {
        let mut parts: Vec<String> = Vec::new();
        if let Some(hint) = &self.hint {
            parts.push(hint.clone());
        }
        if let Some(file) = &self.file_path {
            parts.push(file.clone());
        }
        parts.push(self.project.clone());
        let text = parts.join(" ");

        SearchQuery {
            text,
            project: Some(self.project.clone()),
            since: None,
            limit: self.limit,
        }
    }
}
