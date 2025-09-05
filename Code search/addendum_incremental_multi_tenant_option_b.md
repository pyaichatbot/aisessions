
# Multi‑Tenant Indexing (Option B) — Incremental Indexing Additions
**Applies to:** `multi_tenant_indexing_with_option_b.md`  
**Goal:** Keep one branch per repo (e.g., `main`) but support **incremental indexing** so only changed files are re‑embedded and graph updates are minimal.

---

## 0) Branch Policy
- System indexes **one branch per repo** (default `main` or `master`).
- API accepts `ref` but normalizes to the configured branch per repo.
- Postgres tracks `last_indexed_commit` per `(tenant_key, project_id, branch)`.

```sql
ALTER TABLE repos
  ADD COLUMN branch TEXT NOT NULL DEFAULT 'main',
  ADD COLUMN last_indexed_commit TEXT;
CREATE INDEX IF NOT EXISTS repos_tenant_branch_idx ON repos(tenant_id, project_id, branch);
```

---

## 1) Snapshot → MinIO (unchanged)
- Snapshot job still produces `snapshot.tar.zst` and `manifest.jsonl.zst` at
  `s3://code-index/<tenant_key>/<project_id>/<commit>/...`.
- **Manifest** now also includes a content hash per file (`sha256`).
- Keep a small **manifest index** table for quick diffing:

```sql
CREATE TABLE IF NOT EXISTS file_manifest (
  tenant_id UUID NOT NULL,
  project_id BIGINT NOT NULL,
  branch TEXT NOT NULL,
  commit TEXT NOT NULL,
  rel_path TEXT NOT NULL,
  sha256 TEXT NOT NULL,
  size BIGINT NOT NULL,
  lang TEXT,
  PRIMARY KEY (tenant_id, project_id, branch, commit, rel_path)
);
```

On finalize, write current commit’s manifest rows (bulk copy). You may prune manifests older than K commits.

---

## 2) Incremental Diff (what changed since last run)
At the start of a new snapshot job:
1. Read `last_indexed_commit` from `repos` (for `(tenant, project, branch)`).
2. If null → treat as **full index**.
3. Else compute **changed sets** against prior manifest:
   - `ADDED`: in new manifest, not in prior.
   - `MODIFIED`: in both but `sha256` differs.
   - `DELETED`: in prior, not in new manifest.

### Efficient approach
- Download only both manifests (`*.jsonl.zst`); do diff in memory.
- No need to expand `snapshot.tar.zst` now.

---

## 3) Chunk Identity & Upserts
To safely re‑use or replace chunks, make **stable IDs**:

```
chunk_id = sha1(
  tenant_key | project_id | branch | commit | rel_path | start_line | end_line | file_sha256 | model_name | chunking_version
)
```
- Store `file_sha256`, `commit`, `branch`, `rel_path`, `start_line`, `end_line`, `lang` with each vector.
- In Milvus, keep scalars for `tenant_key`, `project_id`, `branch`, `commit`, `rel_path`, `chunk_id`.

**Upsert strategy**
- For `ADDED` + `MODIFIED`: re‑chunk the file and **upsert** vectors with new `commit` and `file_sha256`.
- For `DELETED`: delete vectors/graph for that `rel_path` scoped to `(tenant, project, branch, old_commit)`.

**Pruning old commit**
- Option A: soft-delete via a `is_active` flag (scalar) and filter on queries.
- Option B: after finalize, **hard‑delete** old commit’s vectors/edges for the affected files only.

---

## 4) Graph Incrementals (Neo4j)
On `ADDED/MODIFIED` files:
- Recompute symbols for the file → compute edges (CALLS/IMPORTS/EXTENDS) limited to those symbols.
- **MERGE** nodes with `{tenant, project_id, branch, fqn, rel_path}` props.
- Delete **stale edges** originating from modified symbols (use `commit` or a `version` to match).

On `DELETED` files:
- Detach delete nodes whose `rel_path` matches and `branch` matches:
  ```cypher
  MATCH (n {tenant:$t, project_id:$p, branch:$b, rel_path:$path}) DETACH DELETE n;
  ```

Optional: mark nodes with `commit` and run a cleanup for nodes not present in the new manifest + branch.

---

## 5) Shard Planning for Incrementals
- Build shard lists from **(ADDED ∪ MODIFIED) files** only.
- Optionally include **neighbors** for better graph consistency (e.g., files that directly import/call modified ones) — can be deferred.
- Deleted files do **not** create shards; they’re handled as a fast delete phase.

---

## 6) API Behaviour
### `/index` (default incremental)
- Resolve target branch & commit; compute diff to `last_indexed_commit`.
- If first time → full index; else only changed files.
- After all shards succeed, update `repos.last_indexed_commit = commit`.

### `/reindex` (force full)
- Same as `/index` but pass `force_full=true` to snapshot job.
- Clears vectors/edges for the repo+branch before re‑inserting (or mark inactive then vacuum).

**Input stays simple (repo URL; tokens stateless).**

---

## 7) Mermaid (incremental view)

```mermaid
flowchart TD
    Start[New /index request] --> Snapshot[Snapshot Job (locked)]
    Snapshot -->|Read last_indexed_commit| ReposDB[(Postgres: repos)]
    Snapshot -->|Create manifest.jsonl.zst| MinIO[(MinIO)]
    Snapshot --> Diff[Diff new vs prior manifest]
    Diff -->|ADDED/MODIFIED| Shards[Create shard lists]
    Diff -->|DELETED| Deletes[Plan deletes]

    Shards --> Queue[Shard Jobs]
    Queue --> Shard1[Shard #1 re-embed]
    Queue --> ShardN[Shard #N re-embed]

    Shard1 --> Milvus[(Milvus upserts)]
    Shard1 --> Neo4j[(Neo4j merges)]
    ShardN --> Milvus
    ShardN --> Neo4j

    Deletes --> Neo4jDel[Neo4j delete stale]
    Deletes --> MilvusDel[Milvus delete stale]

    Milvus --> Finalize[Finalize Job]
    Neo4j --> Finalize
    Finalize -->|set last_indexed_commit| ReposDB
```

---

## 8) Pseudocode (core parts)

```python
def run_snapshot_incremental(tenant_key, project_id, branch="main", force_full=False):
    last = repos.get_last_indexed_commit(tenant_key, project_id, branch)
    commit = resolve_commit(branch)

    # Build manifest + upload
    manifest_new = build_manifest(worktree)
    upload_manifest_and_tar(manifest_new, tar_path, tenant_key, project_id, commit)

    if force_full or not last:
        added, modified, deleted = manifest_new.paths, [], []
    else:
        manifest_old = download_manifest(tenant_key, project_id, last)
        added, modified, deleted = diff_manifests(manifest_old, manifest_new)

    enqueue_deletes(deleted, tenant_key, project_id, branch, last)
    enqueue_shards(added + modified, tenant_key, project_id, branch, commit)
```

Delete step (scoped):
```python
def delete_file_artifacts(rel_path, tenant_key, project_id, branch, old_commit):
    milvus.delete(expr=f"tenant_key=={tenant_key} && project_id=={project_id} && branch=='{branch}' && commit=='{old_commit}' && rel_path=='{rel_path}'")
    neo4j.run("MATCH (n {tenant:$t, project_id:$p, branch:$b, rel_path:$path}) DETACH DELETE n", {...})
```

---

## 9) Acceptance
- `/index` updates only changed files for the one configured branch.
- `/reindex` does a clean rebuild (force full) for that branch.
- Queries return only active commit data (or filter by branch=main and latest commit).
- No token storage; tokens in headers only.

---

## 10) Notes
- If your chunker is context‑aware, include `chunking_version` in `chunk_id` to avoid collisions on future algorithm changes.
- For filenames or title vectors, always upsert on `ADDED/MODIFIED` and delete on `DELETED`.
- Keep manifests only for the **latest K commits** to save MinIO space.
