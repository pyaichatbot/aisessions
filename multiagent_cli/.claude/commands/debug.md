---
description: Run the debugger workflow — diagnose failing job, fix, re-test, update current MR.
argument-hint: [path to failing logs file]
allowed-tools: Task, Bash, Read, Write, Edit, Grep, Glob
---

You are the lead agent for the debugger workflow. Orchestrate — do NOT code yourself.

## Inputs

Logs path (optional): $ARGUMENTS

Env:
- `AGENT_BRANCH` (required) — current MR source branch
- `AGENT_BASE` (default: main)
- `FAILED_JOB_NAME` (optional)
- `FAILED_JOB_LOGS_FILE` (required if $ARGUMENTS empty)
- `MULTIAGENT_DEBUG_JOBS` (comma list; empty = all)
- `COVERAGE_THRESHOLD` (default: 80)

## Guard: only run for allowed jobs

```
scripts/debug_guard.sh "$FAILED_JOB_NAME"
```

If that script exits with code 64 (skip), print skip message and stop.

## Steps

### 1. Prepare environment on CURRENT branch

```
scripts/prepare_env.sh "$AGENT_BRANCH" "${AGENT_BASE:-main}" debug
```

Read `.multiagent_cli/run/env.json`. `cd` into that workdir. Do NOT create a new branch.

### 2. Load logs

If $ARGUMENTS is empty, use `$FAILED_JOB_LOGS_FILE`.

Read the last 8K bytes via `tail -c 8192`. Pass to the debugger.

### 3. Diagnose

Invoke the `debugger` subagent with:
- `failing_logs`
- `failed_job`

Expect a JSON fix plan. Parse with:
```
scripts/parse_plan.sh <<< "<debugger_output>"
```

### 4. Coders

Same as dev workflow step 3. Batch by `parallel_group`, invoke `coder` concurrently.

```
scripts/git_commit.sh "agent(debug): $(jq -r .summary .multiagent_cli/run/plan.json)"
```

### 5. Review

Same as dev step 4.

### 6. Tests

Invoke `tester` with `spec="Stabilize around fix: <root_cause>"`.

```
scripts/git_commit.sh "agent(tester): stabilize"
```

### 7. Gate

Same as dev step 6.

### 8. Push and UPDATE existing MR

```
scripts/git_push.sh
scripts/open_mr.sh debug "$AGENT_BRANCH" "${AGENT_BASE:-main}"
```

`open_mr.sh` is idempotent — if an MR already exists for (source,target), it updates instead of creating.

### 9. Summarize

Print the same final JSON block as the dev workflow.

## Rules

- Work on the CURRENT MR branch only. No new branch.
- Do not force push.
- Same single-MR rule as dev.
