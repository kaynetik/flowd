#!/usr/bin/env bash
# Claude Code PostToolUse hook -> flowd observation.
#
# Records tool name, input summary, and response summary. Large payloads are
# truncated to keep SQLite rows small; the full JSON survives in `.metadata`.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$HERE/_lib.sh"

MAX_CONTENT=2000

flowd_read_event
project="$(flowd_project)"
claude_session="$(flowd_event_field '.session_id')"
flowd_session="$(flowd_session_for "$claude_session")"

tool="$(flowd_event_field '.tool_name')"
tool="${tool:-unknown}"
input_summary="$(jq -c '.tool_input // {}' <<<"$HOOK_EVENT" | head -c "$MAX_CONTENT")"
response_summary="$(jq -c '.tool_response // {}' <<<"$HOOK_EVENT" | head -c "$MAX_CONTENT")"

metadata="$(jq -nc \
    --arg kind 'post_tool_use' \
    --arg tool "$tool" \
    --arg cs "$claude_session" \
    --argjson input "$(jq '.tool_input // {}' <<<"$HOOK_EVENT")" \
    --argjson response "$(jq '.tool_response // {}' <<<"$HOOK_EVENT")" \
    '{kind:$kind, tool:$tool, claude_session:$cs, input:$input, response:$response}')"

printf 'tool=%s\ninput=%s\nresponse=%s\n' "$tool" "$input_summary" "$response_summary" \
    | flowd_observe "$project" "$flowd_session" "$metadata"
