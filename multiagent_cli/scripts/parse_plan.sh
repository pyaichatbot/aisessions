#!/usr/bin/env bash
# Read agent output on stdin. Extract the first ```json ... ``` block.
# Write to .multiagent_cli/run/plan.json. Echo the batch schedule.
# Requires: jq.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="$REPO_ROOT/.multiagent_cli/run"
mkdir -p "$RUN_DIR"

INPUT="$(cat)"

JSON_BODY="$(printf '%s' "$INPUT" | awk '
  /```json/ { flag=1; next }
  /```/     { if (flag) { flag=0; exit } }
  flag      { print }
')"

if [ -z "$JSON_BODY" ]; then
  # Fallback: treat entire stdin as JSON.
  JSON_BODY="$INPUT"
fi

echo "$JSON_BODY" | jq '.' > "$RUN_DIR/plan.json" || {
  echo "ERROR: plan JSON parse failed" >&2
  echo "$JSON_BODY" >&2
  exit 1
}

# Emit batch schedule: groups by parallel_group honoring depends_on.
jq -r '
  . as $root |
  ([.subtasks[]? | .id] // []) as $all_ids |
  reduce (.subtasks // [])[] as $t ({batches: [], done: []};
    ($t.parallel_group // $t.id) as $g |
    if ([$t.depends_on[]? | select(. as $d | $d | IN(.done[]?) | not)] | length) > 0
    then . + {batches: .batches, done: .done}
    else
      (.batches | length) as $bl |
      if ($bl == 0) or ((.batches[$bl-1] | map(.parallel_group // .id) | all(. == $g)) | not)
      then . + {batches: (.batches + [[$t]]), done: (.done + [$t.id])}
      else . + {batches: (.batches[:-1] + [ .batches[-1] + [$t] ]), done: (.done + [$t.id])}
      end
    end
  )
  | .batches
' "$RUN_DIR/plan.json" > "$RUN_DIR/batches.json"

wc -l < "$RUN_DIR/plan.json"
