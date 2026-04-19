# multiagent_md

Same requirements as `multiagent/`. Subagents now live as markdown files,
not Python classes. One `AgentRunner` loads any `.md` and runs the tool loop.

## layout

- `subagents/*.md` role definitions with YAML frontmatter
- `harness/` runner, orchestrator, router, budget, cache, state, git_env, tool_registry
- `memory/` Karpathy LLM-wiki with CRDT versioning
- `workflows/` dev + debug flows
- `tools/` fs, shell, tests, git
- `gitlab/` CI templates
- `cli.py` entrypoint

## add a new subagent

Just drop a `.md` file in `subagents/` with frontmatter:

```md
---
name: linter
description: Runs static checks.
model_tier: cheap
tools: [read_file, grep]
output_format: json
max_tokens: 1024
max_tool_iters: 5
---
<system prompt body>
```

No Python edits required.

## run

```
python -m multiagent_md.cli dev --task "add retry" --branch agent/dev-x --local
python -m multiagent_md.cli debug --branch agent/dev-x --logs-file fail.log --local
```

## frontmatter fields

- `name` (required) — agent id; `<name>.md` must match.
- `description` — one line; shown in listings.
- `model_tier` — `cheap` | `mid` | `strong`.
- `tools` — list of tool names from the registry.
- `output_format` — `text` | `json` | `fenced`.
- `max_tokens`, `max_tool_iters` — per-turn caps.

See `docs/architecture.md`, `docs/gaps.md`, `docs/memory.md`.
