---
description: Run the full development workflow — plan, code, review, test, gate, MR.
argument-hint: <task description>
allowed-tools: Task, Bash, Read, Write, Edit, Grep, Glob
---

You are the lead agent for the development workflow. Orchestrate — do NOT code yourself.

## Inputs

Task: $ARGUMENTS

Env:
- `AGENT_BRANCH` (required) — target branch for this run
- `AGENT_BASE` (default: main)
- `COVERAGE_THRESHOLD` (default: 80)
- `GITLAB_CI` — if `true`, run in branch mode; else use a git worktree

## Steps

Run these in order. After each step, append a state entry.

### 1. Prepare environment

```
scripts/prepare_env.sh "$AGENT_BRANCH" "${AGENT_BASE:-main}"
```

Read `.multiagent_cli/run/env.json` — this is the working directory for the rest of the run. `cd` into it.

### 2. Plan

Invoke the `planner` subagent via the Task tool. Pass the task. Expect a JSON block back.

Parse it with:
```
scripts/parse_plan.sh <<< "<planner_output>"
```

This writes `.multiagent_cli/run/plan.json` and emits a batch file list.

### 3. Coders (parallel where possible)

For each batch of subtasks (disjoint files, same `parallel_group`):
- Invoke the `coder` subagent ONCE per subtask in the batch. The Task tool runs these concurrently when you issue multiple Task calls in a single assistant turn.
- Pass: subtask description, files_hint.
- If a coder reports failure, re-invoke with `escalate=true` context.

After all batches complete:
```
scripts/git_commit.sh "agent(coder): $(jq -r .summary .multiagent_cli/run/plan.json)"
```

### 4. Review loop

Repeat up to `MAX_ITERATIONS=4`:
- Invoke `reviewer` subagent.
- Parse its JSON.
- If `approved: true` → break.
- Else re-invoke `coder` with the issues list.

After loop:
```
scripts/git_commit.sh "agent(review): address issues"
```

### 5. Tests

Invoke `tester` subagent with the plan summary + diff context.

```
scripts/git_commit.sh "agent(tester): add tests"
```

### 6. Gate

Invoke `gatekeeper` subagent with `threshold=$COVERAGE_THRESHOLD`.

```
scripts/git_commit.sh "agent(gatekeeper): coverage gate"
```

### 7. Push and open/update MR

```
scripts/git_push.sh
scripts/open_mr.sh dev "$AGENT_BRANCH" "${AGENT_BASE:-main}"
```

### 8. Summarize

Print a final JSON block to stdout:

```json
{
  "ok": <bool>,
  "branch": "<branch>",
  "mr_url": "<url or null>",
  "gate": {...from .multiagent_cli/run/gate.json...},
  "budget": {...from .multiagent_cli/run/budget.json...}
}
```

## Rules for orchestration

- Never edit source yourself. Delegate to subagents.
- Run subagent invocations in parallel (single turn, multiple Task calls) when their work is independent.
- Stop early with a clear error message if any script exits non-zero.
- Keep every agent on the SAME branch; only ONE MR per run.
