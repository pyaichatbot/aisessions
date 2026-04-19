# multiagent_cli

Third version. **No Python code.** Claude Code CLI is the lead agent.
All subagents are markdown files. Orchestration lives in slash commands.
Helpers are bash scripts.

## layout

- `.claude/agents/*.md` — six subagents (Claude Code convention)
- `.claude/commands/dev.md` — development workflow slash command
- `.claude/commands/debug.md` — debugger workflow slash command
- `.claude/settings.json` — permissions + hooks (budget guard)
- `scripts/*.sh` — git env, tests, MR, wiki CRDT, budget
- `gitlab/` — CI templates
- `docs/` — architecture, memory, gaps

## install (local)

From the repo root:

```
multiagent_cli/scripts/install.sh
```

This links `.claude/agents/*` and `.claude/commands/*` into the repo so
Claude Code picks them up.

## run (local)

Start Claude Code in the repo root, then:

```
/dev add retry with backoff to the HTTP fetcher in src/fetcher.py
```

To trigger the debugger workflow:

```
FAILED_JOB_LOGS_FILE=/tmp/fail.log FAILED_JOB_NAME=test \
  claude -p "/debug $FAILED_JOB_LOGS_FILE"
```

## gitlab

Include `multiagent_cli/gitlab/.gitlab-ci.yml`. Set CI variables:
- `ANTHROPIC_API_KEY`
- `GITLAB_TOKEN` (or rely on `CI_JOB_TOKEN`)
- `MULTIAGENT_DEBUG_JOBS` — comma list of jobs eligible for auto-debug

Hook failing jobs with `.emit_debug_trigger_on_fail` from
`gitlab/trigger_on_failure.yml`.

## add a new subagent

Drop a new `.md` under `.claude/agents/` with frontmatter:

```
---
name: linter
description: Static checks.
tools: Read, Grep
model: haiku
---
<system prompt>
```

No code changes. Claude Code auto-discovers it.

## memory

CRDT-style append-only JSONL at `.multiagent_cli/wiki/events.jsonl`.
- `scripts/wiki_append.sh` — add events (atomic O_APPEND).
- `scripts/wiki_render.sh` — fold into `pages/*.md` snapshots.
- `scripts/wiki_merge.sh` — offline merge two logs.
- `scripts/wiki_context.sh` — top-K retrieval for prompt injection.

Git merges of the JSONL file never conflict — lines concatenate.

See `docs/architecture.md`, `docs/memory.md`, `docs/gaps.md`.
