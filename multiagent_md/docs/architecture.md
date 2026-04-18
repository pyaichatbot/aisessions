# Architecture (md-first)

## Key difference from v1

All subagent logic lives in `subagents/*.md`. The harness is generic:
one `AgentRunner` loads the md, picks the model, runs the tool loop, and
returns text. Roles differ only by prompt + allowed tools + tier.

Adding a role = dropping a new `.md` file. No Python change required.

## Topology

```
                      ┌───────────────────────────┐
task / failing-logs ─▶│        Orchestrator       │─▶ SharedState (run/state.json)
                      │ (lead, parses JSON,       │
                      │  commits, opens MR)       │
                      └──┬──────────────────────┬─┘
                         │                      │
                         ▼                      ▼
                 AgentRunner (generic)     GitEnv (worktree|branch)
                         │
                         ▼
        ┌───── loads ─────┐     ┌───── calls ─────┐
        ▼                 ▼     ▼                 ▼
  subagents/*.md     ModelRouter         ToolRegistry ──▶ read/write/apply_diff/
   planner.md        haiku→sonnet→opus              grep/list/run_tests
   coder.md          PromptCache+BudgetTracker           (sandboxed to workdir)
   reviewer.md
   tester.md                   ▲
   debugger.md                 │
   gatekeeper.md         WikiMemory (JSONL CRDT + TF-IDF)
```

## Lifecycle of one agent run

1. `AgentRunner.run("coder", prompt, context)`.
2. Loader parses `coder.md` frontmatter + body. Cached.
3. System blocks = `[prompt with cache_control, wiki context]`.
4. Tool schema derived from `tools:` list, resolved via `ToolRegistry`.
5. Disk cache lookup on `(model, system, messages, tools)` — skip call on hit.
6. Model call. Parse `tool_use` content blocks.
7. For each tool_use: execute handler, append `tool_result` to messages, loop.
8. Stop on `stop_reason=end_turn`, no tool_use, or `max_tool_iters`.
9. On final iter cap, force a final no-tools turn for a summary.
10. Record turn in wiki; return text + tokens + cost.

## Why not just bake the loop per-role?

Because the loop is identical. Differences are prompt and tool set — both
already declared in frontmatter. Collapsing gives:

- Zero per-role Python code.
- Hot-swap roles by editing `.md`.
- Consistent budget/cache/memory accounting across all roles.

## Workflow orchestration (unchanged from v1 in intent)

- **dev**: planner → coder×N (parallel/sequential by plan batches) → reviewer
  loop → tester → gatekeeper → single push → idempotent MR.
- **debug**: debugger → coder×N → reviewer → tester → gatekeeper → updates
  the existing MR. Triggered by `MULTIAGENT_DEBUG_JOBS` allow-list.

Single branch per run. Single MR (create-or-update by source/target pair).

## Cost efficiency

- Tier routing (`cheap|mid|strong`) per subagent.
- Escalation rule: retry failed step at next tier up.
- Prompt-cache breakpoint on system prompt — survives across turns.
- Disk cache on `(model, system, messages, tools)` skips identical calls.
- `max_tool_iters` cap prevents runaway loops.
- Parallel coder batches only when files are disjoint (planner enforces).
- Token accounting per call; hard `BudgetTracker` cap.

## Subagent contract

Required frontmatter keys: `name`, `model_tier`.
Optional: `description`, `tools`, `output_format`, `max_tokens`, `max_tool_iters`.
Body: system prompt. Keep terse — loaded on every turn.

JSON-producing roles (planner, reviewer, debugger) must emit inside a
```json fence; the orchestrator strips the fence before parsing.

## Memory model

Identical CRDT design as v1. `memory/wiki.py` + `events.jsonl`:
append-only union/LWW, tombstone retract, git-merge-conflict-free,
TF-IDF retrieval swappable with embeddings.

## Harness guarantees

- Single JSON state blob at `.multiagent_md/run/state.json`.
- Every agent turn logged to the wiki.
- Tool calls sandboxed to workdir (path traversal guarded).
- Shell deny-list + timeouts.
- Atomic patch application with `git apply --3way` fallback.
