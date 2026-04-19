# Architecture (CLI-first, no Python)

## Topology

```
┌──────────────────────────────────────────────────────────┐
│               Claude Code CLI (lead agent)                │
│  runs .claude/commands/dev.md | debug.md as prompt        │
└─────┬──────────────────────────────────────────────────┬──┘
      │ Task tool                                        │ Bash
      ▼                                                  ▼
  Subagents (.claude/agents/*.md)              Scripts (scripts/*.sh)
   planner, coder, reviewer, tester,            prepare_env, run_tests,
   debugger, gatekeeper                         open_mr, git_commit,
                                                git_push, parse_plan,
                                                wiki_append, wiki_render,
                                                wiki_merge, wiki_context,
                                                budget_check, debug_guard
                                                      │
                                                      ▼
                                       .multiagent_cli/ (run state)
                                         ├── run/env.json
                                         ├── run/plan.json
                                         ├── run/gate.json
                                         ├── run/last_test.json
                                         ├── run/budget.json
                                         ├── wiki/events.jsonl  (CRDT)
                                         └── wiki/pages/*.md    (derived)
```

## Lead agent

Claude Code CLI is invoked via `claude -p "/dev <task>"` or `claude -p "/debug ..."`. The slash command markdown becomes the prompt. The lead agent never edits code — it orchestrates subagents via the `Task` tool and runs scripts via `Bash`.

## Subagents

Each is a standalone `.md` file with YAML frontmatter that declares:
- `name` — used by the Task tool and by Claude Code's auto-invoke
- `description` — hint for auto-invocation
- `tools` — which built-in Claude Code tools the agent may use
- `model` — haiku | sonnet | opus (tier routing)

Claude Code loads these on startup. Adding a role = drop a new md file.

## Parallelism

When the lead issues multiple `Task` calls in a single assistant turn, they run concurrently. The planner labels disjoint subtasks with `parallel_group`. The dev command instructs the lead to batch them in one turn per group.

## Workflows

### /dev

1. `prepare_env.sh` → worktree (local) or branch (CI).
2. `Task(planner)` → JSON plan.
3. `parse_plan.sh` → validate, schedule batches.
4. `Task(coder)` × N in parallel per batch.
5. `git_commit.sh`.
6. Loop: `Task(reviewer)` → if not approved, `Task(coder)` with issues.
7. `Task(tester)`.
8. `Task(gatekeeper)` → runs tests + fixes + coverage gate.
9. `git_push.sh`, then `open_mr.sh dev …` (idempotent).

### /debug

1. `debug_guard.sh` — respects `MULTIAGENT_DEBUG_JOBS` allow-list.
2. `prepare_env.sh` in debug mode — uses the current MR branch, no new branch.
3. `Task(debugger)` → JSON fix plan from failing logs.
4. Same coder → review → test → gate chain.
5. `open_mr.sh debug …` — updates the existing MR, never creates a new one.

## Cost efficiency

- Tier routing per subagent via `model:` frontmatter. Haiku default; sonnet/opus where needed.
- Claude Code caches system prompts (subagent body) automatically.
- Disjoint `parallel_group` batching avoids redundant re-work.
- `budget_check.sh` counts turns; enforces `MAX_AGENT_TURNS` cap via `SubagentStop` hook.
- No retrieval model calls — wiki retrieval uses term-hit ranking in bash.

## Harness guarantees

- `.multiagent_cli/run/*.json` is the single source of truth for state during a run.
- Every subagent records a wiki event before returning (see agent md bodies).
- A single branch per workflow run; a single MR (create-or-update).
- `git apply --3way` not required — Claude Code's Edit/Write tools operate on files directly; no patch fences to apply.

## Memory (CRDT)

`.multiagent_cli/wiki/events.jsonl` — append-only. Git merges concatenate lines with no conflict markers. `wiki_render.sh` folds the log into `pages/*.md` for humans. Two-pass replay honors `retract` tombstones regardless of order.

See `memory.md`.
