#!/usr/bin/env bash
# Claude Code SessionEnd hook -> flowd observation.
#
# Records a terminator row and drops the Claude->flowd session mapping so
# a restart starts a clean session.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=_lib.sh
source "$HERE/_lib.sh"

flowd_read_event
project="$(flowd_project)"
claude_session="$(flowd_event_field '.session_id')"
flowd_session="$(flowd_session_for "$claude_session")"

printf 'session_end project=%s claude_session=%s\n' "$project" "$claude_session" \
    | flowd_observe "$project" "$flowd_session" \
        "$(jq -nc --arg kind 'session_end' --arg cs "$claude_session" '{kind:$kind, claude_session:$cs}')"

flowd_forget_session "$claude_session"
