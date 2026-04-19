#!/usr/bin/env bash
# Fold events.jsonl into per-page markdown snapshots under wiki/pages/.
# Snapshots are derived views — events.jsonl is the source of truth.
# Uses jq only. Two-pass CRDT: collect retracted ids, then replay.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
WIKI="$REPO_ROOT/.multiagent_cli/wiki"
EVENTS="$WIKI/events.jsonl"
PAGES="$WIKI/pages"
mkdir -p "$PAGES"

[ -f "$EVENTS" ] || { echo "(no events)"; exit 0; }

# Pass 1: collect tombstoned ids.
RETRACTED_FILE="$(mktemp)"
jq -r 'select(.op=="retract") | .payload.target_id // empty' "$EVENTS" \
  | sort -u > "$RETRACTED_FILE"

# Pass 2: slurp non-retracted events, group by page, fold state.
jq -s --slurpfile retracted <(jq -R '.' "$RETRACTED_FILE") '
  ([$retracted[] | .] ) as $skip
  | map(select(.op != "retract" and (.id as $i | $skip | index($i) | not)))
  | group_by(.page)
  | map({
      page: .[0].page,
      state: (reduce .[] as $e ({};
        if $e.op == "upsert" then
          (. * $e.payload)
        elif $e.op == "append" then
          (reduce ($e.payload | to_entries[]) as $kv (.;
            .[$kv.key] = (((.[$kv.key] // []) + [$kv.value]) | unique)
          ))
        else . end
      ))
    })
' "$EVENTS" > "$WIKI/_pages.json"

rm -f "$RETRACTED_FILE"

jq -r '.[] | .page' "$WIKI/_pages.json" | while IFS= read -r P; do
  [ -z "$P" ] && continue
  SAFE="$(echo "$P" | tr '/' '__')"
  OUT="$PAGES/$SAFE.md"
  {
    echo "# $P"
    echo
    jq -r --arg p "$P" '
      .[] | select(.page==$p) | .state | to_entries[] |
      "## \(.key)\n\(.value | tostring)\n"
    ' "$WIKI/_pages.json"
  } > "$OUT"
done

echo "rendered $(find "$PAGES" -type f | wc -l) pages"
