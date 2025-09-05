
# Code Indexing Requirements — Incremental Mode (Additions)
**Applies to:** `code_indexing_requirements_prompt.md`  
**Intent:** Keep single‑branch policy but avoid re‑embedding whole repos on each run.

---

## Key Points
- Index **one branch per repo** (main/master).  
- Track `last_indexed_commit`.  
- Use **manifest diff** to limit work to `ADDED`, `MODIFIED`, `DELETED` files.  
- Upsert vectors/graph nodes for changed files; delete for removed files.  
- Keep API contracts unchanged (repo URL inputs; stateless tokens).

---

## Data needed (new fields)
- `repos.branch` (default `main`), `repos.last_indexed_commit`  
- Manifest row: `rel_path`, `sha256`, `size`, `lang`  
- Vector scalars: `tenant_key`, `project_id`, `branch`, `commit`, `rel_path`, `chunk_id`, `file_sha256`  
- Graph node props: `tenant`, `project_id`, `branch`, `rel_path`, `fqn`, `commit`

---

## Flow
1. Snapshot job (locked) builds new manifest & tar and uploads to MinIO.  
2. Compute diff vs previous manifest (or full if first time).  
3. Enqueue delete tasks for `DELETED`; enqueue shard tasks for `ADDED ∪ MODIFIED`.  
4. Shards re‑embed and MERGE graph for their files only.  
5. Finalize sets `last_indexed_commit` and optionally prunes old commit data.

---

## Delete semantics
- Vectors: delete by `(tenant, project_id, branch, commit=old, rel_path=path)` or mark inactive.  
- Graph: `DETACH DELETE` nodes with matching `(tenant, project_id, branch, rel_path)` from old commit.

---

## Testing
- Create tiny repo with 3 files; run full index.  
- Modify 1 file, add 1 file, delete 1 file; run `/index`.  
- Assert Milvus insert count ≈ chunks from 2 files; Neo4j nodes/edges updated only for those; deleted file artifacts gone.

---

## Ops
- Retain manifests and artifacts for last **K** commits.  
- Include `chunking_version` in IDs to allow future upgrades.  
- Use per‑repo advisory lock only in the snapshot step.

---

## Done Criteria
- Incremental run touches only changed files.  
- Full reindex via `/reindex` works and produces same final state as fresh full index.  
- Search returns correct results after add/modify/delete scenarios.
