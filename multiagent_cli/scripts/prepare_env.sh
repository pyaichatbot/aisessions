#!/usr/bin/env bash
# Usage: prepare_env.sh <branch> <base> [mode]
# mode: dev (default) | debug
# Sets up either a git worktree (local) or a fresh branch (CI).
# Emits .multiagent_cli/run/env.json with {workdir, mode, branch, base}.
set -euo pipefail

BRANCH="${1:?branch required}"
BASE="${2:-main}"
MODE="${3:-dev}"

REPO_ROOT="$(git rev-parse --show-toplevel)"
RUN_DIR="$REPO_ROOT/.multiagent_cli/run"
mkdir -p "$RUN_DIR"

# GitLab CI => branch mode; else worktree.
if [ "${GITLAB_CI:-}" = "true" ]; then
  GIT_MODE="branch"
  WORKDIR="$REPO_ROOT"
  if [ "$MODE" = "dev" ]; then
    if ! git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
      git checkout -b "$BRANCH"
    else
      git checkout "$BRANCH"
    fi
  else
    # debug: stay on current MR branch
    git fetch origin "$BRANCH" || true
    git checkout "$BRANCH"
  fi
else
  GIT_MODE="worktree"
  SLUG="$(echo "$BRANCH" | tr '/' '_')"
  WORKDIR="$REPO_ROOT/.multiagent_cli/worktrees/$SLUG"
  mkdir -p "$(dirname "$WORKDIR")"
  if [ ! -d "$WORKDIR" ]; then
    git fetch origin "$BASE" || true
    if git rev-parse --verify "$BRANCH" >/dev/null 2>&1; then
      git worktree add "$WORKDIR" "$BRANCH"
    else
      git worktree add -b "$BRANCH" "$WORKDIR" "origin/$BASE"
    fi
  fi
fi

cat > "$RUN_DIR/env.json" <<EOF
{
  "workdir": "$WORKDIR",
  "mode": "$GIT_MODE",
  "branch": "$BRANCH",
  "base": "$BASE",
  "workflow": "$MODE"
}
EOF

echo "$WORKDIR"
