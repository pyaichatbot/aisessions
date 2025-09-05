
# Requirements & Implementation Prompt for GPT in VS Code

> Paste this prompt into VS Code (e.g., Copilot Chat) to guide GPT in scaffolding and implementing the **enterprise code indexing and search system** with Milvus, Postgres, Neo4j, and MinIO.  
> It includes requirements, architecture, indexing pipeline, locks/idempotency, and acceptance criteria.

---

## 1. Requirements

- **Tech Stack**
  - Python 3.11 + FastAPI (backend APIs)
  - Milvus (vector DB for semantic search)
  - Neo4j (graph DB for dependency graph)
  - Postgres (metadata, ACLs, control plane)
  - MinIO (object storage for snapshots & manifests)
  - Embedding model: OSS bge-m3 (via TEI) — pluggable
  - GitLab integration for repo mirroring

- **Features**
  - Multi-tenant support with ACL
  - Repo indexing via snapshot + shard pipeline
  - Semantic + graph hybrid search
  - Expansion to methods/classes
  - Confidence thresholds to filter junk results
  - Observability, retries, idempotency

---

## 2. Indexing Pipeline (Production Pattern)

### Step 1 — **Snapshot Job** (short, locked)
- Acquire **per-repo advisory lock**.
- Update mirror (`git fetch --prune`).
- Resolve target commit (SHA).
- Create a **read-only snapshot** (worktree or tar) and a **file manifest** (paths + hashes).
- Upload snapshot + manifest to **MinIO** (`s3://code-index/<tenant>/<repo>/<sha>/...`).
- Release the lock.
- Emit `snapshot_id = (tenant, repo_id, commit)`.

👉 Ensures everyone indexes the **same commit** and only one mirror update runs at a time.

### Step 2 — **Index Shards** (parallel, no lock)
- Split manifest into N shards (e.g., 1k files/shard).
- Each job:
  - Downloads files from snapshot (or mounts worktree).
  - Parse → chunk (method/class + line windows) → add **title-vector per file**.
  - Embed in batches; upsert to **Milvus**.
  - Extract **symbols/edges**; batch MERGE to **Neo4j**.
  - Write per-shard status (counts, errors).

👉 Parallelized, no global locks, safe to retry.

### Step 3 — **Finalize Job** (quick)
- Verify all shards succeed.
- Mark commit **indexed in Postgres**.
- Mark previous commit as **superseded** (or soft-delete old vectors/edges).
- Optionally trigger **Milvus index build/compaction**.

---

## 3. Locks & Idempotency

- **Lock scope**: only the **Snapshot job** holds the per-repo lock.  
  Heavy compute (shards) runs without lock.

- **Idempotency keys**:
  - Snapshot: `sha1(tenant|repo|commit)`
  - Shard: `sha1(tenant|repo|commit|shard_id)`

- **Primary keys for data**:
  - Vectors: `(tenant, repo_id, commit, chunk_id)`
  - Symbols: `(tenant, repo_id, commit, fqn, signature_hash)`
  - Edges: `(tenant, repo_id, commit, edge_type, from_id, to_id)`

👉 Prevents duplicates even if jobs retry or crash mid-way.

---

## 4. Search Flow

1. Parse intent (filename, symbol, semantic).  
2. Recall candidates from Milvus (`top_k = 10x`).  
3. Fetch graph features from Neo4j (callers, callees, centrality).  
4. Re-rank (semantic + lexical + graph fusion).  
5. Apply **confidence floor** (e.g., 0.35).  
6. Expand to **full methods/classes** from snapshot/MinIO.  
7. Return enriched hits (code span + deps).

---

## 5. Deliverables

- FastAPI service with `/index`, `/search`, `/expand`, `/repos/my`
- Background worker for Snapshot → Shards → Finalize jobs
- Milvus + Neo4j + Postgres + MinIO integration
- Docker Compose with persistent volumes for all services
- Docs + tests + metrics

---

## 6. Acceptance Criteria

- Idempotent, retry-safe indexing (no duplicate vectors/edges).
- Repo isolation per tenant.
- Accurate search with hybrid re-ranking.
- Persistence across container restarts (volumes for DBs).

