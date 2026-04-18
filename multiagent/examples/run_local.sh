#!/usr/bin/env bash
# Local dry-run end-to-end using the cached dummy responder.
set -euo pipefail
export MULTIAGENT_DRY_RUN=1
python -m multiagent.cli dev \
  --task "Add retry with backoff to fetcher" \
  --branch "agent/dev-example" \
  --base "main" \
  --local
