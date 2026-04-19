#!/usr/bin/env bash
# Exit 0 if debug workflow should run, 64 if it should skip.
# Gate by $MULTIAGENT_DEBUG_JOBS (comma list). Empty list => allow all.
set -euo pipefail

JOB="${1:-}"
ALLOWED="${MULTIAGENT_DEBUG_JOBS:-}"

if [ -z "$ALLOWED" ]; then
  exit 0
fi
if [ -z "$JOB" ]; then
  # Job name unknown — be conservative, allow.
  exit 0
fi

IFS=',' read -ra ARR <<< "$ALLOWED"
for j in "${ARR[@]}"; do
  if [ "$(echo "$j" | xargs)" = "$JOB" ]; then
    exit 0
  fi
done

echo "skip: job '$JOB' not in MULTIAGENT_DEBUG_JOBS=$ALLOWED"
exit 64
