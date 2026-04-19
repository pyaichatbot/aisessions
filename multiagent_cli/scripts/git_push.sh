#!/usr/bin/env bash
# Push current branch upstream with retries on network errors.
set -euo pipefail

BRANCH="$(git branch --show-current)"
if [ -z "$BRANCH" ]; then
  echo "ERROR: detached HEAD" >&2
  exit 1
fi

for attempt in 1 2 3 4; do
  if git push -u origin "$BRANCH"; then
    exit 0
  fi
  sleep $((2 ** attempt))
done

echo "ERROR: push failed after 4 attempts" >&2
exit 1
