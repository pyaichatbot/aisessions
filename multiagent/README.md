# multiagent

Multi-agent dev team. Plan, code, review, test, gate.

## layout

- `agents/` subagent roles (planner, coder, reviewer, tester, debugger, gatekeeper)
- `harness/` orchestrator, budget, router, cache, state, git env
- `memory/` Karpathy LLM-wiki, CRDT versioning
- `workflows/` dev + debugger flows
- `gitlab/` pipeline templates
- `prompts/` role prompts
- `tools/` shell, fs, tests, git
- `cli.py` local entrypoint
- `docs/` architecture, gaps, memory design

## run local

```
python -m multiagent.cli dev --task "add retry to fetcher"
python -m multiagent.cli debug --mr 42
```

## gitlab

Include `multiagent/gitlab/.gitlab-ci.yml`. Set `MULTIAGENT_DEBUG_JOBS` env var.

## cost

Router picks cheap model by default. Escalate only on review fail.
See `docs/architecture.md`.
