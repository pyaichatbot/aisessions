# Gaps, edge cases, mitigations (md-first)

## md-specific edge cases

- **Malformed frontmatter**: loader falls back to defaults; name defaults to
  file stem. Validate by running `python -m multiagent_md.cli dev --help`
  after adding a new subagent.
- **Missing tool in registry**: `schema_for` skips unknown names silently;
  agent proceeds without that tool. Add a console warning in future.
- **Body prompt drift**: md is hot-reloaded on each orchestrator process start;
  `AgentRunner._load` caches per-process. Restart to pick up edits.
- **Tool iteration runaway**: `max_tool_iters` is hard cap; final turn forced
  with `tools=None` to guarantee text output.

## Concurrency / isolation

- **Two coders same file**: planner forbids via `parallel_group` = disjoint files.
- **Worktree collision**: branch name hashed into worktree path.
- **Budget race**: `BudgetTracker` lock-protected across threads.
- **Shared cache races**: atomic `os.replace` during write.

## Git / MR

- **Idempotent MR**: `open_mr` reuses existing (source,target).
- **Rebase drift**: debug flow stays on current branch — no rebase.
- **Protected branch**: push failure surfaces; no silent force push.
- **Large diff**: reviewer gets `git diff origin/base...HEAD` only; planner
  should split oversize tasks.

## Testing / gate

- **Framework absent**: falls through to pytest. Override with
  `MULTIAGENT_TEST_CMD`.
- **Flaky tests**: gatekeeper reruns via its `run_tests` tool. Persistent
  flakiness shows in final report.
- **No coverage tool**: parses 0% → gate fails loudly.
- **Infinite fix loop**: `max_iterations` caps review fix attempts;
  `max_tool_iters` caps gatekeeper tool loops.

## Model cost

- **Escalation**: cheap → mid → strong only on repeated failure.
- **Budget breach**: raises `BudgetExceeded`; partial state preserved.
- **Cache poisoning**: hash keyed on full call arguments including tools schema.
- **Silent tool schemas**: if frontmatter lists a tool the registry lacks, it
  is dropped rather than raising (prevents accidental run failure from typos).

## Security

- **Path traversal**: `_safe()` in `fs.py` blocks escape via `..`.
- **Shell injection**: deny-list + shlex split. Use dedicated runners.
- **Token leakage**: secrets read from env only; never written to state or cache.
- **Prompt injection via repo**: tools return file content verbatim, but
  orchestrator trusts only structured outputs (JSON) from planner/reviewer/debugger.

## Memory

- **Merge conflicts on wiki**: impossible with append-only JSONL.
- **Retracted-but-referenced**: filtered at read time (two-pass).
- **Unbounded growth**: bounded by `page_bytes_max` + `topk`. Offline compaction
  TBD.

## Non-goals

- Cross-repo changes.
- Migrations or infra deploys.
- Multi-language monorepos beyond auto-detect.
- Long-running cross-session tasks.
- Semantic wiki merge beyond union CRDT.

## Known weak spots

- Planner quality bottlenecks all downstream cost.
- JSON parsing assumes fenced code block; fallback is heuristic.
- TF-IDF misses semantic matches — upgrade to embeddings is straightforward.
- No streaming token budget; enforced per-call.

## v1 vs v2 tradeoffs

| | v1 (Python classes) | v2 (md-first) |
|---|---|---|
| new role | write class | add md file |
| tool surface | hand-wired per class | declared in frontmatter |
| harness code | per-role logic | generic runner |
| debug | stack traces rich | md-lint optional |
| migration cost | known-good | requires md discipline |
