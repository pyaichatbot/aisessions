
# Multi‑Tenant GitLab Code Indexing & Search  
**Requirements & Implementation Prompt (for GPT in VS Code)**

> Paste this prompt into VS Code AI chat (Copilot Chat, Cursor, Windsurf) to scaffold multi‑tenant support for your **code indexing & search system**.  
> Goal: support multiple GitLab domains and groups (e.g., `emea/teama`, `emea/teamb`, `usa/teamx`, `india/teamy`) with clean tenant isolation.

---

## 0) Why tenants?

- Different business units (BUs) use **different GitLab domains** or **different root groups**.  
- We need **isolation** so Team A cannot see Team B’s repos or results.  
- A tenant = **(domain + root group)**, e.g.:  
  - `gitlab-scm.company.com//emea/teama`  
  - `gitlab-scm.company.com//emea/teamb`  
  - `gitlab-us.company.com//usa/teamx`  
  - `gitlab-in.company.com//india/teamy`

---

## 1) Data Model Changes

### Tenants
```sql
CREATE TABLE tenants (
  id UUID PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,              -- e.g., 'gitlab-scm.company.com//emea/teama'
  display_name TEXT NOT NULL,
  gitlab_domain TEXT NOT NULL,
  root_group_path TEXT NOT NULL,         -- 'emea/teama'
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);
```

### Allowed Git Domains (optional service tokens per tenant)
```sql
CREATE TABLE allowed_git_domains (
  id SERIAL PRIMARY KEY,
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  domain TEXT NOT NULL,
  service_token TEXT,                    -- encrypted or NULL if stateless
  is_active BOOLEAN NOT NULL DEFAULT TRUE,
  UNIQUE (tenant_id, domain)
);
```

### Repos Registry
```sql
ALTER TABLE repos
  ADD COLUMN tenant_id UUID,
  ADD COLUMN group_path TEXT,
  ADD COLUMN repo_id BIGINT,
  ADD COLUMN visibility TEXT;

CREATE INDEX ON repos(tenant_id);
CREATE INDEX ON repos(repo_id);
```

### User Repo Access (cache of GitLab memberships)
```sql
CREATE TABLE user_repo_access (
  user_id TEXT NOT NULL,
  tenant_id UUID NOT NULL REFERENCES tenants(id),
  repo_id BIGINT NOT NULL,
  scope TEXT NOT NULL DEFAULT 'member',
  expires_at TIMESTAMPTZ NOT NULL,
  PRIMARY KEY (user_id, tenant_id, repo_id)
);
```

---

## 2) Milvus (vector DB)

- One collection `code_chunks`.  
- **Partition per tenant**: `partition_names=[tenant_id]`.  
- Scalar fields: `tenant_id`, `repo_id`, `commit`, `rel_path`, `lang`, `chunk_id`, etc.  
- **Search**: always include tenant partition + repo_id filter.

---

## 3) Neo4j (graph DB)

- Add `tenant` and `repo_id` properties on all nodes/edges.  
- Create index:  
  ```cypher
  CREATE INDEX method_by_tenant_fqn IF NOT EXISTS
  FOR (m:Method) ON (m.tenant, m.fqn);
  ```  
- Every Cypher must include `WHERE n.tenant = $tenant AND n.repo_id IN $repo_ids`.

---

## 4) MinIO (object storage)

Prefix with tenant key:  
```
s3://code-index/<tenant_key>/<repo_id>/<commit>/files/<rel_path>.gz
                                       /manifest.json.gz
                                       /symbols.jsonl.gz
                                       /edges.jsonl.gz
```

---

## 5) API Changes (FastAPI)

### Headers
- `X-Tenant-Id: <uuid>` (preferred)  
- Or `X-Tenant-Key: gitlab-scm.company.com//emea/teama` (resolve internally).  
- Optional: `X-Git-Identity: domain=...,token=...` (stateless user tokens).

### Search body
```json
{
  "query": "jwt refresh token rotation",
  "mode": "my" | "repo" | "list",
  "repo_id": 12345,
  "repo_ids": [111,222],
  "top_k": 5
}
```

---

## 6) Indexing Flow (per tenant)

1. Parse repo URL → `(domain, root_group)` → tenant key.  
2. Look up/create tenant row.  
3. Register repo under tenant.  
4. Mirror path: `/data/mirrors/<tenant_key>/<repo_id>.git`  
5. Worktrees: `/data/worktrees/<job_id>`  
6. Artifacts: `/<tenant_key>/<repo_id>/<commit>/...` in MinIO.  
7. Write: Milvus insert → **tenant partition**; Neo4j MERGE with tenant + repo_id.

**Locking:** use per‑repo lock scoped to `(tenant_id, repo_id)`:  
```sql
SELECT pg_advisory_lock(hashtextextended(:tenant_id::text || ':' || :repo_id::text, 42));
```

---

## 7) Search Flow (per tenant)

1. Resolve allowed repo set:  
   - `mode="repo"`: validate `repo_id` belongs to tenant.  
   - `mode="list"`: validate each repo_id belongs to tenant.  
   - `mode="my"`: load from `user_repo_access`; refresh from GitLab if expired.  

2. Apply scope:  
   - Milvus: `partition_names=[tenant_id]`, `expr="repo_id in [...]"`.  
   - Neo4j: `WHERE n.tenant=$tenant AND n.repo_id IN $repo_ids`.  
   - MinIO: read under `<tenant_key>/...`.

3. Audit: log `(tenant_id, repo_count, query_hash)`, never tokens.

---

## 8) Tenant Key Resolution (helper)

```python
from urllib.parse import urlparse

def parse_tenant_key(url: str) -> tuple[str,str,str]:
    u = urlparse(url.replace("git@", "ssh://git@"))
    domain = u.hostname
    parts = [p for p in u.path.strip("/").removesuffix(".git").split("/") if p]
    if len(parts) < 3:
        raise ValueError("Expect at least region/team/project")
    root_group = "/".join(parts[:2])   # e.g., "emea/teama"
    repo_path  = "/".join(parts[:-1])  # e.g., "emea/teama/projects"
    return domain, root_group, repo_path

def tenant_key(domain: str, root_group: str) -> str:
    return f"{domain}//{root_group}"
```

---

## 9) Example Tenants

| Tenant Key                                   | Domain                   | Root Group   | Display Name |
|----------------------------------------------|--------------------------|--------------|--------------|
| `gitlab-scm.company.com//emea/teama`         | `gitlab-scm.company.com` | `emea/teama` | EMEA Team A  |
| `gitlab-scm.company.com//emea/teamb`         | `gitlab-scm.company.com` | `emea/teamb` | EMEA Team B  |
| `gitlab-us.company.com//usa/teamx`           | `gitlab-us.company.com`  | `usa/teamx`  | USA Team X   |
| `gitlab-in.company.com//india/teamy`         | `gitlab-in.company.com`  | `india/teamy`| India Team Y |

---

## 10) Tasks for GPT (do in order)

1. Add Alembic migrations for `tenants`, `allowed_git_domains`, `user_repo_access`, extend `repos`.  
2. Update indexer: parse repo URL → resolve tenant → tag data everywhere.  
3. Update mirror/worktree & MinIO paths to include tenant key.  
4. Update Milvus wrapper: create/search partitions by tenant.  
5. Update Neo4j writes/queries with tenant filter.  
6. Update FastAPI: require `X-Tenant-Id` or `X-Tenant-Key`.  
7. Update search: enforce repo filters per tenant.  
8. Add audit logging.  
9. Update README with examples for multi‑tenant indexing & search.

---

## 11) Example Requests

**Index repo (Team A)**  
```http
POST /index
X-Tenant-Key: gitlab-scm.company.com//emea/teama
Content-Type: application/json

{
  "repo_url": "https://gitlab-scm.company.com/emea/teama/projects/service-a.git",
  "ref": "main",
  "mode": "incremental"
}
```

**Search in Team B repos**  
```http
POST /search
X-Tenant-Key: gitlab-scm.company.com//emea/teamb
Content-Type: application/json

{
  "query": "jwt refresh token",
  "mode": "my",
  "top_k": 5
}
```

---

**Acceptance Criteria:**  
- Each tenant sees only its repos.  
- Indexing writes data partitioned by tenant.  
- Search enforces tenant + repo scope everywhere.  
- Data persists across container restarts (Docker volumes).  

