#!/usr/bin/env bash
# Stage and commit all changes in the current workdir.
# No-op if tree is clean.
set -euo pipefail

MSG="${1:?commit message required}"
git add -A
if [ -z "$(git status --porcelain)" ]; then
  echo "(nothing to commit)"
  exit 0
fi
git commit -m "$MSG"
