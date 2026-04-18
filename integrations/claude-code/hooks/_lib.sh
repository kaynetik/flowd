#!/usr/bin/env bash
# Shared helpers for flowd hook scripts.
# Sourced by session-start.sh, post-tool-use.sh, session-end.sh.
#
# Contract with Claude Code: hooks read a single JSON object on stdin.
# Fields consumed here: .session_id, .cwd, .tool_name, .tool_input, .tool_response.
# Missing fields are tolerated; hooks must never fail the parent process.
#
# Prerequisites on $PATH: bash 4+, jq, uuidgen, flowd.

set -euo pipefail

FLOWD_HOME="${FLOWD_HOME:-$HOME/.flowd}"
HOOK_SESSION_DIR="$FLOWD_HOME/hook-sessions"

# flowd_read_event reads the hook payload from stdin into $HOOK_EVENT.
flowd_read_event() {
    HOOK_EVENT="$(cat || true)"
    if [[ -z "$HOOK_EVENT" ]]; then HOOK_EVENT="{}"; fi
}

# flowd_event_field <jq-path> -> prints the value or empty string.
flowd_event_field() {
    jq -r "${1} // empty" <<<"$HOOK_EVENT"
}

# flowd_project prints the project slug derived from .cwd (falls back to $PWD).
flowd_project() {
    local cwd
    cwd="$(flowd_event_field '.cwd')"
    cwd="${cwd:-$PWD}"
    basename "$cwd"
}

# flowd_session_for <claude_session_id> prints the flowd session UUID.
# Creates the mapping file on first use so all hooks for one Claude session
# write to the same flowd session.
flowd_session_for() {
    local claude_id="$1"
    if [[ -z "$claude_id" ]]; then
        uuidgen | tr '[:upper:]' '[:lower:]'
        return
    fi
    local file="$HOOK_SESSION_DIR/$claude_id"
    if [[ -f "$file" ]]; then
        cat "$file"
    else
        mkdir -p "$HOOK_SESSION_DIR"
        local uuid
        uuid="$(uuidgen | tr '[:upper:]' '[:lower:]')"
        printf '%s' "$uuid" >"$file"
        printf '%s' "$uuid"
    fi
}

# flowd_forget_session <claude_session_id>.
flowd_forget_session() {
    local claude_id="$1"
    [[ -n "$claude_id" ]] || return 0
    rm -f "$HOOK_SESSION_DIR/$claude_id"
}

# flowd_observe <project> <session_uuid> <metadata_json>.
# Content is piped on stdin. Swallows errors so hooks never block Claude Code.
flowd_observe() {
    local project="$1" session="$2" metadata="${3:-{\}}"
    flowd observe --project "$project" --session "$session" --metadata "$metadata" \
        >/dev/null 2>&1 || true
}
