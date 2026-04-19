#!/usr/bin/env bash
# Emit top-K wiki pages relevant to a query. TF-IDF via jq + awk.
# Usage: wiki_context.sh <role_or_query> [topk]
# Writes nothing; prints a markdown block suitable for prompt injection.
set -euo pipefail

QUERY="${1:?query required}"
TOPK="${2:-6}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
WIKI="$REPO_ROOT/.multiagent_cli/wiki"
EVENTS="$WIKI/events.jsonl"
[ -f "$EVENTS" ] || exit 0

# Very small ranker: count query-term hits per page in the materialized state.
# Sufficient for cost-free retrieval; swap for embeddings later if needed.
bash "$(dirname "$0")/wiki_render.sh" >/dev/null

PAGES_DIR="$WIKI/pages"
[ -d "$PAGES_DIR" ] || exit 0

# Tokenize query
TERMS="$(echo "$QUERY" | tr '[:upper:]' '[:lower:]' | grep -Eo '[a-z0-9_]{2,}' | sort -u || true)"
[ -n "$TERMS" ] || exit 0

# Score each page.
SCORED="$(mktemp)"
trap 'rm -f "$SCORED"' EXIT
for f in "$PAGES_DIR"/*.md; do
  [ -f "$f" ] || continue
  SCORE=0
  LOWER="$(tr '[:upper:]' '[:lower:]' < "$f")"
  for t in $TERMS; do
    HITS="$(grep -c "$t" <<< "$LOWER" || true)"
    SCORE=$((SCORE + HITS))
  done
  echo "$SCORE $f"
done | sort -rn | head -n "$TOPK" > "$SCORED"

awk '$1>0 {print $2}' "$SCORED" | while IFS= read -r p; do
  echo "### $(basename "$p" .md | tr '_' '/')"
  sed -e '1d' "$p"
  echo
done
