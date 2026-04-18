#!/usr/bin/env bash
# Claude Code SessionStart hook -> flowd observation.
#
# Records a "session started" marker so the session shows up in
# `flowd history` even if the user never invokes a tool.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$HERE/_lib.sh"

flowd_read_event
project="$(flowd_project)"
claude_session="$(flowd_event_field '.session_id')"
flowd_session="$(flowd_session_for "$claude_session")"

printf 'session_start project=%s claude_session=%s\n' "$project" "$claude_session" \
    | flowd_observe "$project" "$flowd_session" \
        "$(jq -nc --arg kind 'session_start' --arg cs "$claude_session" '{kind:$kind, claude_session:$cs}')"
