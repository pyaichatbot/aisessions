# Gaps, edge cases, mitigations (CLI-first)

## CLI-specific

- **No exposed per-token cost from Claude Code**: `budget_check.sh` counts
  subagent turns as a proxy and enforces `MAX_AGENT_TURNS`. For true $ caps,
  set `$ANTHROPIC_SMALL_FAST_MODEL` and rely on Anthropic-side usage monitoring.
- **Tool allow-list drift**: if you add a new subagent that needs an
  unsanctioned Bash pattern, update `.claude/settings.json` `permissions.allow`.
- **Hooks must be on PATH**: `scripts/` is referenced by relative path in hooks;
  run Claude Code from the repo root. The `install.sh` places a symlink-free copy.
- **stream-json parsing in CI**: use `claude -p "/dev ..." --output-format stream-json`
  so pipelines can grep final status; else use default text output.

## Concurrency / isolation

- **Two coders same file**: planner forbids via `parallel_group` = disjoint
  files. If the planner misgroups, the second coder's Edit will collide with
  the first — either the second Edit fails (unique-match rule) or commit shows
  only the latest. Re-invocation with `escalate=true` is automatic.
- **Worktree collision (local)**: branch name slashified into a unique path.
- **Budget race**: `budget_check.sh` uses `jq` with atomic `mv` after `mktemp`.

## Git / MR

- **Idempotent MR**: `open_mr.sh` reuses existing (source,target) pair — MR is
  updated, not duplicated. Re-runs safe.
- **Rebase drift in debug**: debug flow stays on current MR branch; no rebase
  attempt. Non-fast-forward push surfaces; operator resolves upstream.
- **Protected branch**: `git_push.sh` retries on network error only. Policy
  rejection is not retried.
- **Large diff review**: reviewer reads `git diff origin/base...HEAD`. If it
  hits Claude Code's context ceiling, planner should split the task earlier.

## Testing / gate

- **No test framework detected**: `run_tests.sh` falls through to pytest. Set
  `MULTIAGENT_TEST_CMD` to override.
- **Flaky tests**: gatekeeper loops up to a cap; persistent flakiness reported
  in `run/last_test.json` and wiki.
- **Coverage tooling absent**: `coverage=0` parsed → gate fails loudly rather
  than silently passing.
- **Infinite fix loop**: gatekeeper md declares a max iteration ceiling;
  `MAX_AGENT_TURNS` hook also caps total spend.

## Model cost

- **Tier routing**: haiku default (coder, tester), sonnet for reviewer and
  gatekeeper, opus for debugger — set via `model:` frontmatter.
- **No built-in escalation**: on repeated failure, the lead re-invokes the same
  subagent with an `escalate` context note. To force a higher tier, duplicate
  the md with `-strong` suffix and a higher `model:`.
- **Cache**: Claude Code caches system prompts automatically; keep subagent
  bodies terse.

## Memory

- **Merge conflicts on wiki**: impossible — append-only.
- **Retracted-but-referenced**: filtered at read time (two-pass replay).
- **Unbounded growth**: `pages/*.md` are regenerated; compact offline with
  `wiki_merge.sh` on two snapshots.

## Security

- **Shell injection**: scripts quote all inputs; `.claude/settings.json`
  `permissions.allow` narrows Bash patterns. Keep the allow-list tight.
- **Token leakage**: `GITLAB_TOKEN` / `ANTHROPIC_API_KEY` read from env only.
  Never logged.
- **Prompt injection via repo**: reviewer/planner read files — do not put
  untrusted text into tracked files and expect it ignored. Gate with
  `.gitattributes` and branch protection in sensitive repos.

## Non-goals

- Cross-repo changes.
- Migrations / infra-as-code deploys.
- Multi-language monorepos beyond `run_tests.sh` auto-detect.
- Long-running cross-session tasks.
- Semantic wiki merge beyond union CRDT.

## Known weak spots

- Planner quality bottlenecks the whole run.
- Bash TF-IDF is crude; replace `wiki_context.sh` with embeddings if scale
  warrants.
- `MAX_AGENT_TURNS` is a proxy, not a dollar cap.
- No streaming budget; caps applied per-invocation.

## v1 / v2 / v3 tradeoffs

| | v1 (Python classes) | v2 (md + Python harness) | v3 (CLI only) |
|---|---|---|---|
| new role | write class | add md | add md |
| harness | custom Python | generic AgentRunner | Claude Code |
| tool loop | custom | custom | built-in |
| cost model | precise $ | precise $ | turn-count proxy |
| dependencies | anthropic sdk | anthropic sdk | Claude Code CLI |
| hot-reload | restart | restart | on Claude Code start |
