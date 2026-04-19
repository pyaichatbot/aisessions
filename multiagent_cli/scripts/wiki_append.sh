#!/usr/bin/env bash
# Append one event to the CRDT wiki log.
# Usage: wiki_append.sh <actor> <page> <op> <payload_json>
# op: upsert | append | retract
# The log is append-only JSONL → git-merge-conflict-free by construction.
set -euo pipefail

ACTOR="${1:?actor required}"
PAGE="${2:?page required}"
OP="${3:?op required}"
PAYLOAD="${4-}"
[ -z "$PAYLOAD" ] && PAYLOAD='{}'

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
WIKI="$REPO_ROOT/.multiagent_cli/wiki"
mkdir -p "$WIKI"
FILE="$WIKI/events.jsonl"

# uuid v4 (no external deps required on most systems)
UUID="$(cat /proc/sys/kernel/random/uuid 2>/dev/null || \
        python3 -c 'import uuid; print(uuid.uuid4().hex)' 2>/dev/null || \
        date +%s%N)"

TS="$(date +%s.%N 2>/dev/null || date +%s)"

ENTRY="$(jq -cn \
  --arg id "$UUID" \
  --arg ts "$TS" \
  --arg actor "$ACTOR" \
  --arg page "$PAGE" \
  --arg op "$OP" \
  --argjson payload "$PAYLOAD" \
  '{id:$id, ts:($ts|tonumber), actor:$actor, page:$page, op:$op, payload:$payload, parent:null}' )"

# O_APPEND is atomic on POSIX; safe under concurrent writers.
printf '%s\n' "$ENTRY" >> "$FILE"
