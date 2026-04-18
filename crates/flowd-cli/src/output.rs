//! Terminal output helpers.
//!
//! ANSI escape codes are emitted directly rather than pulling in a colour
//! crate; callers that want plain output set `NO_COLOR` (honoured by
//! [`Style::colour_enabled`]).
//!
//! All formatting sits here so commands stay focused on data retrieval.

use std::fmt::Write as _;

use chrono::{DateTime, Utc};
use flowd_core::types::{SearchResult, Session};

/// ANSI SGR codes. Kept minimal on purpose.
mod sgr {
    pub const RESET: &str = "\x1b[0m";
    pub const BOLD: &str = "\x1b[1m";
    pub const DIM: &str = "\x1b[2m";
    pub const GREEN: &str = "\x1b[32m";
    pub const YELLOW: &str = "\x1b[33m";
    pub const CYAN: &str = "\x1b[36m";
    pub const RED: &str = "\x1b[31m";
}

#[derive(Debug, Clone, Copy)]
pub struct Style {
    pub colour: bool,
}

impl Style {
    /// Colour enabled unless `NO_COLOR` is set (see <https://no-color.org>).
    #[must_use]
    pub fn from_env() -> Self {
        Self {
            colour: std::env::var_os("NO_COLOR").is_none(),
        }
    }

    #[must_use]
    pub fn plain() -> Self {
        Self { colour: false }
    }

    fn wrap(self, code: &str, text: &str) -> String {
        if self.colour {
            format!("{code}{text}{reset}", reset = sgr::RESET)
        } else {
            text.to_owned()
        }
    }

    #[must_use]
    pub fn bold(self, text: &str) -> String {
        self.wrap(sgr::BOLD, text)
    }
    #[must_use]
    pub fn dim(self, text: &str) -> String {
        self.wrap(sgr::DIM, text)
    }
    #[must_use]
    pub fn green(self, text: &str) -> String {
        self.wrap(sgr::GREEN, text)
    }
    #[must_use]
    pub fn yellow(self, text: &str) -> String {
        self.wrap(sgr::YELLOW, text)
    }
    #[must_use]
    pub fn cyan(self, text: &str) -> String {
        self.wrap(sgr::CYAN, text)
    }
    #[must_use]
    pub fn red(self, text: &str) -> String {
        self.wrap(sgr::RED, text)
    }
}

/// Truncate `s` to `max` chars with an ellipsis marker. Grapheme-naive on
/// purpose -- we want byte-budget truncation, not visual alignment.
#[must_use]
pub fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_owned();
    }
    let mut out: String = s.chars().take(max.saturating_sub(1)).collect();
    out.push('…');
    out
}

/// Format a UTC timestamp as `YYYY-MM-DD HH:MM:SS` (no timezone).
#[must_use]
pub fn fmt_ts(ts: DateTime<Utc>) -> String {
    ts.format("%Y-%m-%d %H:%M:%S").to_string()
}

/// Render a list of search results as an aligned, colourised block.
#[must_use]
pub fn format_search_results(results: &[SearchResult], style: Style) -> String {
    if results.is_empty() {
        return style.dim("(no results)\n");
    }
    let mut out = String::new();
    for (i, r) in results.iter().enumerate() {
        let header = format!(
            "{rank} {score}  {project}  {when}",
            rank = style.bold(&format!("#{}", i + 1)),
            score = style.dim(&format!("rrf={:.4}", r.rrf_score)),
            project = style.cyan(&r.observation.project),
            when = style.dim(&fmt_ts(r.observation.created_at)),
        );
        let _ = writeln!(out, "{header}");
        let preview = truncate(r.observation.content.trim(), 240);
        for line in preview.lines() {
            let _ = writeln!(out, "  {line}");
        }
        let _ = writeln!(out, "  {}", style.dim(&format!("id: {}", r.observation.id)));
        let _ = writeln!(out);
    }
    out
}

/// Render a session list.
#[must_use]
pub fn format_sessions(sessions: &[Session], style: Style) -> String {
    if sessions.is_empty() {
        return style.dim("(no sessions)\n");
    }
    let mut out = String::new();
    for s in sessions {
        let started = fmt_ts(s.started_at);
        let ended = s.ended_at.map_or_else(|| style.green("running"), fmt_ts);
        let _ = writeln!(
            out,
            "{id}  {project}  {started} → {ended}",
            id = style.dim(&s.id.to_string()),
            project = style.cyan(&s.project),
            started = started,
            ended = ended,
        );
        if let Some(summary) = &s.summary {
            let _ = writeln!(out, "  {}", style.dim(&truncate(summary, 160)));
        }
    }
    out
}

/// Emit a banner like `=== title ===`.
#[must_use]
pub fn banner(title: &str, style: Style) -> String {
    style.bold(&format!("\n=== {title} ===\n"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn truncate_preserves_short_strings() {
        assert_eq!(truncate("abc", 10), "abc");
    }

    #[test]
    fn truncate_adds_ellipsis() {
        let t = truncate("hello world", 5);
        assert_eq!(t.chars().count(), 5);
        assert!(t.ends_with('…'));
    }

    #[test]
    fn plain_style_strips_colour() {
        let plain = Style::plain();
        assert_eq!(plain.bold("x"), "x");
        assert_eq!(plain.red("x"), "x");
    }

    #[test]
    fn coloured_style_wraps_with_escape_codes() {
        let c = Style { colour: true };
        assert!(c.red("x").starts_with("\x1b["));
        assert!(c.red("x").ends_with("\x1b[0m"));
    }

    #[test]
    fn format_search_results_empty_shows_placeholder() {
        let out = format_search_results(&[], Style::plain());
        assert!(out.contains("no results"));
    }

    #[test]
    fn format_sessions_empty_shows_placeholder() {
        let out = format_sessions(&[], Style::plain());
        assert!(out.contains("no sessions"));
    }
}
