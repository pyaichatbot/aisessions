# Gaps, edge cases, mitigations

Explicit enumeration of scenarios the implementation handles or flags.

## Concurrency / isolation

- **Two coders same file**: planner forbids via `parallel_group` = disjoint files.
  If planner misgroups, second coder's patch fails to apply — orchestrator
  retries with `escalate=True`.
- **Worktree collision (local)**: branch name hashed into worktree path.
- **Budget race**: `BudgetTracker` is lock-protected; parallel coders can't overspend.

## Git / MR

- **Idempotent MR**: `open_mr` reuses existing (source,target) pair. Re-runs
  update the same MR instead of creating duplicates.
- **Rebase drift**: debug workflow works on current MR branch only — no rebase.
  Fast-forward push fails loudly; operator resolves upstream.
- **Protected branch**: push failure surfaces; no silent force-push.
- **Large diff review**: reviewer receives `git diff origin/base...HEAD`, not
  full repo. Still bounded by model context; very large diffs should be split
  by planner.

## Testing / gate

- **No test framework detected**: falls through to pytest. Set
  `MULTIAGENT_TEST_CMD` to override.
- **Flaky tests**: gatekeeper re-runs and patches. Persistent flakiness surfaces
  as unresolved failing set.
- **Coverage tooling absent**: report parses 0%; gate fails visibly instead of
  falsely passing.
- **Infinite fix loop**: `max_iterations` caps gate attempts; returns `passed=False`.

## Model cost

- **Escalation pattern**: cheap → mid → strong, only on repeated failures.
- **Budget breach**: raises `BudgetExceeded`; orchestrator surfaces partial state.
- **Cache poisoning**: disk cache keyed on full `(model, system, messages)` hash.
  No TTL — prune `.multiagent/cache` manually or via cron.

## Memory

- **Merge conflicts on wiki**: impossible with append-only JSONL.
- **Tombstoned but referenced**: consumers filter retracted ids at read time.
- **Unbounded growth**: `page_bytes_max` + `topk` bound injection. Compact
  offline by running `JsonlLog.materialize()` and rewriting (future work).

## Security

- **Shell injection**: `shell.run_shell` deny-list + shlex split for simple cases.
  Destructive commands blocked. Dedicated CI runner required — do not run on
  shared hosts.
- **Token leakage**: uses `GITLAB_TOKEN`/`CI_JOB_TOKEN` from env only. No logging
  of secrets. Tests run inside job container.
- **Prompt injection via repo**: reviewer operates on diffs, planner on tree
  listing — no arbitrary file execution via model output.

## Human-in-the-loop

- Gate failure + budget breach leave MR open with clear description.
- `docs/architecture.md` labels each commit by agent role for easy audit.
- State file `.multiagent/run/state.json` captures full run timeline.

## Explicit non-goals (call out if you need them)

- Cross-repo changes.
- Migrations or infra-as-code deploys.
- Multi-language monorepos beyond auto-detect of pytest/jest/go.
- Long-running background tasks across sessions.
- Semantic merge of the wiki beyond union CRDT.

## Known weak spots

- Planner quality is the bottleneck; bad decomposition wastes all downstream cost.
- Review verdict parsing assumes JSON; malformed output falls back to heuristic.
- TF-IDF misses semantic matches — upgrade to embeddings is the right next step.
- No token-level streaming budget; cap enforced per-call only.
