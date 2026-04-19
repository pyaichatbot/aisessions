# Memory: Karpathy LLM Wiki + CRDT versioning

## Append-only JSONL

`events.jsonl` is the source of truth. Writers use `O_APPEND` (POSIX atomic).
Two branches appending distinct entries → git 3-way merge concatenates with no
conflict markers. Event order is by `(ts, id)` at read time.

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
- `append`: union set on list fields (dedupe keeps idempotence).
- `retract`: tombstone by `target_id`; applied before replay.

## Retrieval

Local TF-IDF (dependency-free). Swap in embeddings by replacing
`TfidfIndex.rank()`.

## Versioning

Checked in alongside code. Git history = knowledge history. Pages under
`pages/*.md` are regenerated views, never committed as source.

## Cost knobs

- `page_bytes_max` caps per-page injection.
- `topk_retrieval` caps number of pages injected.
- System prompt carries `cache_control: ephemeral`.
