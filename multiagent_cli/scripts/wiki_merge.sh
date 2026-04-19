#!/usr/bin/env bash
# Merge two events.jsonl files into a third. Deduplicate by id.
# Use case: offline reconciliation (git handles online merges natively).
set -euo pipefail

A="${1:?path to events A}"
B="${2:?path to events B}"
OUT="${3:?output path}"

mkdir -p "$(dirname "$OUT")"
cat "$A" "$B" | jq -c 'select(. != null)' \
  | awk '!seen[$0]++' > "$OUT"

wc -l < "$OUT"
