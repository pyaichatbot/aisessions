#!/usr/bin/env bash
# Run the project's test suite. Detect framework. Emit JSON report.
# Output: .multiagent_cli/run/last_test.json with
#   {passed: bool, coverage: float, failing: [...], logs_tail: "..."}.
set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="$REPO_ROOT/.multiagent_cli/run"
mkdir -p "$RUN_DIR"

LOG_FILE="$RUN_DIR/test.log"
: > "$LOG_FILE"

run() {
  echo "+ $*" >> "$LOG_FILE"
  "$@" >> "$LOG_FILE" 2>&1
  echo "$?"
}

if [ -n "${MULTIAGENT_TEST_CMD:-}" ]; then
  bash -c "$MULTIAGENT_TEST_CMD" >> "$LOG_FILE" 2>&1
  RC=$?
elif [ -f package.json ]; then
  npm test --silent -- --coverage >> "$LOG_FILE" 2>&1
  RC=$?
elif [ -f go.mod ]; then
  go test ./... -cover >> "$LOG_FILE" 2>&1
  RC=$?
else
  python -m pytest --maxfail=20 --cov --cov-report=term-missing >> "$LOG_FILE" 2>&1
  RC=$?
fi

# Parse coverage.
COV="$(grep -Eo 'TOTAL[[:space:]]+[0-9]+[[:space:]]+[0-9]+[[:space:]]+[0-9]+%' "$LOG_FILE" \
        | tail -1 | grep -Eo '[0-9]+%' | tr -d '%' || true)"
if [ -z "$COV" ]; then
  COV="$(grep -Eo 'All files[[:space:]]*\|[[:space:]]*[0-9.]+' "$LOG_FILE" \
          | tail -1 | grep -Eo '[0-9.]+$' || true)"
fi
COV="${COV:-0}"

# Parse failing tests.
FAILING="$(grep -E '^FAILED ' "$LOG_FILE" | awk '{print $2}' | head -50 || true)"

PASSED=$([ "$RC" -eq 0 ] && echo "true" || echo "false")
TAIL="$(tail -c 4096 "$LOG_FILE" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))' 2>/dev/null \
        || tail -c 4096 "$LOG_FILE" | sed 's/"/\\"/g' | tr '\n' ' ' | awk '{print "\""$0"\""}')"

FAILING_JSON="$(printf '%s' "$FAILING" | jq -R -s 'split("\n")|map(select(length>0))')"

cat > "$RUN_DIR/last_test.json" <<EOF
{
  "passed": $PASSED,
  "coverage": $COV,
  "failing": $FAILING_JSON,
  "logs_tail": $TAIL
}
EOF

cat "$RUN_DIR/last_test.json"
exit $RC
