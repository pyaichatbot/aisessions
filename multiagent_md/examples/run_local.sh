#!/usr/bin/env bash
set -euo pipefail
export MULTIAGENT_DRY_RUN=1
python -m multiagent_md.cli dev \
  --task "Add retry with backoff to fetcher" \
  --branch "agent/dev-example" \
  --base "main" \
  --local
