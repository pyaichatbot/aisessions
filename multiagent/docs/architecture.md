# Architecture

## Topology

```
            ┌──────────────────────┐
  task ───▶ │     Orchestrator     │ ──▶ SharedState (run/state.json)
            │    (lead agent)      │
            └──┬───┬──┬──┬──┬──┬──┘
               │   │  │  │  │  │
               ▼   ▼  ▼  ▼  ▼  ▼
            Planner Coder(×N) Reviewer Tester Gatekeeper Debugger
                           │
                           ▼
                 ToolLayer: shell, fs, tests, git
                           │
                           ▼
                 GitEnv (worktree | branch)
                           │
                           ▼
                 WikiMemory (JSONL CRDT + TF-IDF)
```

## Runtime modes

| Mode | branch | MR |
|------|--------|-----|
| dev local | new branch in git worktree | push + idempotent MR |
| dev CI    | new branch in pipeline ws  | push + idempotent MR |
| debug local | current branch worktree | updates existing MR |
| debug CI  | current MR branch          | updates existing MR |

## Workflows

### development
1. Planner (mid tier) → JSON plan with dep graph + parallel groups.
2. Orchestrator expands batches. Within a batch, coders run in parallel using
   `ThreadPoolExecutor(max_parallel_coders)` — they never touch each other's files
   (enforced by planner's disjoint `parallel_group`).
3. Reviewer (mid tier). On reject, Coder gets issues back with `escalate=True`
   after first retry.
4. Tester fills tests for the diff.
5. Gatekeeper runs the full suite, patches tests/impl until green and coverage ≥ threshold.
6. Single commit chain, single push, single idempotent MR.

### debugger
1. Trigger: failed CI job matching `MULTIAGENT_DEBUG_JOBS` or local invocation.
2. Debugger (strong tier) → root cause + fix plan.
3. Same coder → review → tester → gate chain, but on the EXISTING MR branch.
4. Updates the same MR; no new branch.

## Cost efficiency

- Tier routing: cheap (haiku) → mid (sonnet) → strong (opus) on escalation only.
- Prompt-cache breakpoint on role system prompts (Anthropic ephemeral cache).
- Disk-backed exact-match cache skips identical calls entirely.
- Parallel batching only where files are disjoint — no redundant rework.
- BudgetTracker aborts with `BudgetExceeded` on cap breach.
- TF-IDF retrieval over the wiki (no embedding API calls).

## Harness guarantees

- All state written to `.multiagent/run/state.json` for post-mortem.
- Every agent turn appended to the wiki log.
- A single branch per workflow run; a single MR (idempotent update).
- Atomic patch application via `git apply --3way` fallback.

## Memory model (Karpathy LLM-Wiki)

- `events.jsonl` is append-only. Two branches appending = union merge.
  Git merges concatenate lines — zero textual conflicts.
- Ordering established by `(ts, id)` at read time. `retract` tombstones supersede.
- Materialized "pages" are derived state; they are never source of truth and
  can be regenerated with `WikiMemory.render_snapshots()`.
- Retrieval uses local TF-IDF (`memory/indexer.py`). Swap with embeddings
  by replacing `TfidfIndex.rank()` only.

See `memory.md` for CRDT invariants.

## Tools layer

- `fs.apply_patch` accepts `diff` fences or `file:path` full-file fences.
- `test_runner` auto-detects pytest, jest, go test. Override with `MULTIAGENT_TEST_CMD`.
- `git_ops.open_mr` is idempotent: reuses existing MR (source,target) or creates one.
- `shell.run_shell` has a deny-list and timeout.
