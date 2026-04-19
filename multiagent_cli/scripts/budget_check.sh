#!/usr/bin/env bash
# Cheap cost tracker. Counts subagent invocations (turns) per run.
# Not a true $ budget — Claude Code does not expose per-turn token cost in a
# scriptable way. Turn count is a proxy; combine with tier routing in md.
#
# Usage:
#   budget_check.sh incr <role>      # increments counter, enforces cap
#   budget_check.sh snapshot         # prints JSON snapshot
#
# Env:
#   MAX_AGENT_TURNS (default 60)
set -euo pipefail

CMD="${1:-snapshot}"
REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="$REPO_ROOT/.multiagent_cli/run"
mkdir -p "$RUN_DIR"
FILE="$RUN_DIR/budget.json"

[ -f "$FILE" ] || echo '{"turns":0,"by_role":{}}' > "$FILE"

case "$CMD" in
  incr)
    ROLE="${2:-unknown}"
    MAX="${MAX_AGENT_TURNS:-60}"
    TMP="$(mktemp)"
    jq --arg r "$ROLE" --argjson max "$MAX" '
      .turns = (.turns + 1) |
      .by_role[$r] = ((.by_role[$r] // 0) + 1) |
      if .turns > $max then error("budget exceeded: " + (.turns|tostring)) else . end
    ' "$FILE" > "$TMP" && mv "$TMP" "$FILE"
    ;;
  snapshot)
    cat "$FILE"
    ;;
  *)
    echo "usage: budget_check.sh {incr <role>|snapshot}" >&2
    exit 2
    ;;
esac
