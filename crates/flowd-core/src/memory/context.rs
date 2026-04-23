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
    /// the project name. Empty components are dropped.
    ///
    /// The FTS5 query grammar is **not** permissive: characters outside the
    /// bareword set (`A-Za-z0-9_` plus unicode letters) and the operator
    /// alphabet (`AND OR NOT NEAR + - : ( ) " * ^`) cause `SQLite` to return
    /// `fts5: syntax error near "<char>"`. Caller-supplied fragments routinely
    /// contain `/`, `.`, `:` etc. (file paths, identifiers, punctuation in
    /// hints), so each fragment is passed through [`sanitize_match`] before
    /// being joined. This preserves the original implicit-AND token matching
    /// semantics -- the unicode61 tokenizer applies the same boundary rules
    /// at index time -- without exposing the FTS5 query parser to grammar
    /// errors. The vector side uses the same sanitized text for embedding;
    /// dropping symbol noise is harmless for the embedding model.
    #[must_use]
    pub fn to_search_query(&self) -> SearchQuery {
        let parts = [
            self.hint.as_deref(),
            self.file_path.as_deref(),
            Some(self.project.as_str()),
        ];
        let text = parts
            .iter()
            .filter_map(|p| p.map(sanitize_match))
            .filter(|s| !s.is_empty())
            .collect::<Vec<_>>()
            .join(" ");

        SearchQuery {
            text,
            project: Some(self.project.clone()),
            since: None,
            limit: self.limit,
        }
    }
}

/// Sanitize a single fragment for inclusion in an FTS5 `MATCH` expression.
///
/// Replaces every character outside the FTS5 bareword set (alphanumeric or
/// `_`, including unicode letters/digits) with a single space, then collapses
/// runs of whitespace. The result is always a sequence of bareword tokens
/// separated by single spaces -- safe to concatenate into a MATCH expression
/// where it parses as implicit-AND across the tokens.
///
/// Examples:
/// - `"rnd_docs/notes.md"` -> `"rnd_docs notes md"`
/// - `"src/main.rs:42"` -> `"src main rs 42"`
/// - `"why pick \"JWT\"?"` -> `"why pick JWT"`
fn sanitize_match(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut last_was_space = true;
    for ch in raw.chars() {
        if ch.is_alphanumeric() || ch == '_' {
            out.push(ch);
            last_was_space = false;
        } else if !last_was_space {
            out.push(' ');
            last_was_space = true;
        }
    }
    if out.ends_with(' ') {
        out.pop();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_path_separators() {
        assert_eq!(sanitize_match("rnd_docs/notes.md"), "rnd_docs notes md");
    }

    #[test]
    fn sanitize_strips_colons_and_collapses_spaces() {
        assert_eq!(sanitize_match("src/main.rs:42"), "src main rs 42");
    }

    #[test]
    fn sanitize_handles_quotes_and_punctuation() {
        assert_eq!(sanitize_match(r#"why pick "JWT"?"#), "why pick JWT");
    }

    #[test]
    fn sanitize_keeps_unicode_letters() {
        assert_eq!(sanitize_match("café/ünicode"), "café ünicode");
    }

    #[test]
    fn sanitize_collapses_runs_and_trims() {
        assert_eq!(sanitize_match("///a   b\n\nc///"), "a b c");
    }

    #[test]
    fn sanitize_empty_and_pure_punctuation() {
        assert_eq!(sanitize_match(""), "");
        assert_eq!(sanitize_match("///"), "");
    }

    #[test]
    fn to_search_query_drops_empty_fragments() {
        let q = AutoContextQuery::new("flowd")
            .with_hint("///") // sanitizes to empty
            .to_search_query();
        // Only the project token survives.
        assert_eq!(q.text, "flowd");
    }

    #[test]
    fn to_search_query_joins_sanitized_fragments_in_order() {
        let q = AutoContextQuery::new("flowd")
            .with_file("rnd_docs/notes.md")
            .with_hint("induce karpathy")
            .to_search_query();
        assert_eq!(q.text, "induce karpathy rnd_docs notes md flowd");
    }
}
