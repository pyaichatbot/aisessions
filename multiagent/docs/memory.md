# Memory: Karpathy LLM Wiki + CRDT versioning

## Why append-only JSONL

Mutable markdown pages on multiple branches = merge conflicts. Here:

- `events.jsonl` is append-only. Writes are `O_APPEND`-atomic on POSIX.
- Two branches both appending distinct entries → git 3-way merge concatenates,
  no conflict markers, no data loss.
- Pages are a fold over events. `materialize()` produces current state.

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

- `upsert`: last-writer-wins by `ts` on scalar fields.
- `append`: union set on list fields (dedupe preserves idempotence).
- `retract`: tombstone by `target_id`, removes entries across replicas.

Union/LWW are conflict-free. No coordination required.

## Retrieval

Local TF-IDF per page. No network. Swap for embeddings by replacing
`TfidfIndex.rank()` — no call-site churn.

## Versioning

The log is checked into git alongside code. You get:
- Full history via `git log` on the JSONL file.
- Diff of knowledge per commit.
- No merge conflicts even with divergent agent runs.

For snapshots humans read, `render_snapshots()` writes `pages/*.md`.
Regenerate-on-read, never commit as source of truth.

## Cost knobs

- `page_bytes_max` truncates per-page context.
- `topk_retrieval` caps blocks injected into system prompt.
- System prompt carries `cache_control: ephemeral` so retrieved memory benefits
  from Anthropic prompt caching across turns.
