# Memory: CRDT JSONL + jq retrieval

## Why append-only JSONL

Mutable markdown on multiple branches = merge conflicts. This design:

- `wiki/events.jsonl` is append-only. Writers use shell `>>` under `O_APPEND`
  (POSIX atomic).
- Two branches appending distinct entries → git 3-way merge concatenates with
  no conflict markers.
- Order resolved by `(ts, id)` at read time. `retract` tombstones supersede.

## Event shape

```json
{
  "id": "uuid",
  "ts": 1713480000.0,
  "actor": "coder",
  "page": "turns/coder",
  "op": "upsert|append|retract",
  "payload": {"notes": "..."},
  "parent": null
}
```

## CRDT semantics

- `upsert` — last-writer-wins by `ts` on scalar fields.
- `append` — union set on list fields (dedupe preserves idempotence).
- `retract` — tombstone by `target_id`, honored across replicas.

Union/LWW are conflict-free. No coordination required between agents.

## Retrieval

`wiki_context.sh` calls `wiki_render.sh` to materialize pages, then ranks by
term-hit count using grep + awk. Top-K pages are printed as a markdown block
suitable for prompt injection.

Swap path: if you later want embeddings, replace the ranker in
`wiki_context.sh` — callers don't change.

## Versioning

Check `events.jsonl` into the repo next to code. Then:
- `git log -- .multiagent_cli/wiki/events.jsonl` gives full knowledge history.
- `git diff` on the log shows the delta per commit.
- No merge conflicts even when two agent runs diverge.

`pages/*.md` are derived — regenerate with `wiki_render.sh`; do not commit as
source of truth.
