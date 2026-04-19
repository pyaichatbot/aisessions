#!/usr/bin/env bash
# Idempotent GitLab MR open/update.
# Usage: open_mr.sh <dev|debug> <source_branch> <target_branch>
# Requires env: GITLAB_TOKEN (or CI_JOB_TOKEN) + CI_PROJECT_ID.
# Optional env: CI_API_V4_URL (default https://gitlab.com/api/v4).
set -euo pipefail

MODE="${1:?mode dev|debug required}"
SOURCE="${2:?source branch required}"
TARGET="${3:?target branch required}"

TOKEN="${GITLAB_TOKEN:-${CI_JOB_TOKEN:-}}"
PROJECT="${CI_PROJECT_ID:-${MULTIAGENT_PROJECT_ID:-}}"
API="${CI_API_V4_URL:-https://gitlab.com/api/v4}"

if [ -z "$TOKEN" ] || [ -z "$PROJECT" ]; then
  echo "(skip MR: GITLAB_TOKEN or CI_PROJECT_ID missing)"
  exit 0
fi

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="$REPO_ROOT/.multiagent_cli/run"

LABEL="agent/$MODE"
PLAN="$(cat "$RUN_DIR/plan.json" 2>/dev/null || echo '{}')"
GATE="$(cat "$RUN_DIR/gate.json" 2>/dev/null || echo '{}')"

SUMMARY="$(jq -r '.summary // .root_cause // "agent run"' <<< "$PLAN")"
TITLE="[agent/$MODE] $(echo "$SUMMARY" | cut -c1-60)"

DESC="$(cat <<EOF
## Summary
$SUMMARY

## Plan
\`\`\`json
$PLAN
\`\`\`

## Gate
\`\`\`json
$GATE
\`\`\`
EOF
)"

# Look for existing MR.
EXISTING="$(curl -sS --fail \
  -H "PRIVATE-TOKEN: $TOKEN" \
  "$API/projects/$PROJECT/merge_requests?source_branch=$SOURCE&target_branch=$TARGET&state=opened")"

IID="$(jq -r '.[0].iid // empty' <<< "$EXISTING")"

PAYLOAD="$(jq -n --arg t "$TITLE" --arg d "$DESC" --arg l "$LABEL" \
  '{title:$t, description:$d, labels:$l}')"

if [ -n "$IID" ]; then
  URL="$(curl -sS --fail -X PUT \
    -H "PRIVATE-TOKEN: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$PAYLOAD" \
    "$API/projects/$PROJECT/merge_requests/$IID" | jq -r .web_url)"
else
  FULL="$(jq -n --arg t "$TITLE" --arg d "$DESC" --arg l "$LABEL" \
    --arg s "$SOURCE" --arg g "$TARGET" \
    '{title:$t, description:$d, labels:$l, source_branch:$s, target_branch:$g}')"
  URL="$(curl -sS --fail -X POST \
    -H "PRIVATE-TOKEN: $TOKEN" \
    -H "Content-Type: application/json" \
    -d "$FULL" \
    "$API/projects/$PROJECT/merge_requests" | jq -r .web_url)"
fi

echo "$URL" > "$RUN_DIR/mr_url.txt"
echo "$URL"
