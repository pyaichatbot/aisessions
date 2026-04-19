#!/usr/bin/env bash
# Link .claude/ agents + commands into the repo root so Claude Code picks
# them up. Idempotent. Copies scripts to make them discoverable.
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
SRC="$ROOT/multiagent_cli"

mkdir -p "$ROOT/.claude/agents" "$ROOT/.claude/commands"

for f in "$SRC/.claude/agents"/*.md; do
  cp "$f" "$ROOT/.claude/agents/$(basename "$f")"
done

for f in "$SRC/.claude/commands"/*.md; do
  cp "$f" "$ROOT/.claude/commands/$(basename "$f")"
done

# Merge settings.json if the user has none, otherwise instruct.
if [ ! -f "$ROOT/.claude/settings.json" ]; then
  cp "$SRC/.claude/settings.json" "$ROOT/.claude/settings.json"
else
  echo "NOTE: existing .claude/settings.json found; merge hooks manually from multiagent_cli/.claude/settings.json"
fi

echo "installed agents: $(ls "$ROOT/.claude/agents"/*.md | wc -l)"
echo "installed commands: $(ls "$ROOT/.claude/commands"/*.md | wc -l)"
echo "scripts: $SRC/scripts (call by full path or add to PATH)"
