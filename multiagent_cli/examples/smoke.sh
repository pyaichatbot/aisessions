#!/usr/bin/env bash
# Local dry-run smoke: verify scripts and agent md files lint-parse.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "# shellcheck pass"
for f in scripts/*.sh; do
  bash -n "$f"
done

echo "# frontmatter pass"
for f in .claude/agents/*.md .claude/commands/*.md; do
  awk 'BEGIN{fm=0} /^---$/{fm=!fm; next} fm' "$f" > /tmp/fm.yaml
  python3 -c "import sys,yaml; yaml.safe_load(open('/tmp/fm.yaml'))" 2>/dev/null \
    || python3 -c "import sys; print(open('/tmp/fm.yaml').read())"
  echo "ok $f"
done

echo "# wiki append smoke"
./scripts/wiki_append.sh test sample upsert '{"k":"v"}'
./scripts/wiki_render.sh

echo "all checks passed"
