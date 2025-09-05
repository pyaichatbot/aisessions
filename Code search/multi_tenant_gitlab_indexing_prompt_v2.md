
# Multi‑Tenant GitLab Code Indexing & Search (Simple & Scalable)
**Requirements & Implementation Prompt (for GPT in VS Code)**  
_Target: easy to implement/maintain, but scalable. Uses stateless tokens (never stored)._

Stack: **FastAPI + Postgres (metadata) + Milvus (vectors) + Neo4j (graph) + MinIO (files)**

---

## 0) Tenancy Model (clear & minimal)

**Tenant = `<domain>//<region>/<team>`**  
Examples:
- `gitlab-scm.company.com//emea/teama`
- `gitlab-scm.company.com//emea/teamb`
- `gitlab-us.company.com//usa/teamx`

All requests carry this tenant and are **scoped** to it.

---

## 1) Headers & Tokens (stateless)

- `X-Tenant-Key: <domain>//<region>/<team or *>`
  - `*` lets you search across **all teams** in that domain/region (e.g., `gitlab-scm.company.com//emea/*`).

- `X-Git-Identity: domain=<domain>,token=<PAT_or_OAuth>` (repeatable)
  - Used **only at request time** for GitLab API and/or `git fetch`.  
  - **Never stored.**

---

## 2) Minimal Postgres schema (control plane)

```sql
CREATE TABLE tenants (
  id UUID PRIMARY KEY,
  key TEXT UNIQUE NOT NULL,              -- domain//region/team  (or domain//region/*)
  gitlab_domain TEXT NOT NULL,
  root_group_path TEXT NOT NULL,         -- region/team or region/*
  is_active BOOLEAN NOT NULL DEFAULT TRUE
);

-- Repos you’ve seen (cache of URL→project_id mapping)
CREATE TABLE repos (
  id BIGSERIAL PRIMARY KEY,
  tenant_id UUID NOT NULL,
  repo_url TEXT NOT NULL,
  project_id BIGINT,                     -- GitLab project id (resolved once)
  group_path TEXT,                       -- region/team/subgroups
  last_indexed_commit TEXT,
  is_index_enabled BOOLEAN NOT NULL DEFAULT TRUE,
  UNIQUE (tenant_id, repo_url)
);
CREATE INDEX ON repos (tenant_id);
CREATE INDEX ON repos (project_id);
```

> Keep it small. No tokens stored. If you later need ACL cache, add it then.

---

## 3) URL → Tenant parsing (one helper)

```python
from urllib.parse import urlparse

def parse_repo_url(url_or_ssh: str):
    if url_or_ssh.startswith("git@"):
        url_or_ssh = url_or_ssh.replace("git@", "ssh://git@").replace(":", "/")
    u = urlparse(url_or_ssh)
    domain = (u.hostname or "").lower()
    parts = u.path.lstrip("/").removesuffix(".git").split("/")
    if len(parts) < 3:
        raise ValueError("Expect at least region/team/project in the path")
    region, team = parts[0], parts[1]
    root_group = f"{region}/{team}"
    group_path = "/".join(parts[:-1])  # region/team/... (no project)
    project = parts[-1]
    return domain, root_group, group_path, project
```

**Tenant key** builder:
```python
def tenant_key(domain: str, root_group: str) -> str:
    return f"{domain}//{root_group}"
```

---

## 4) Indexing — from UI, Webhook, or REST (same contract)

**Headers**
```
X-Tenant-Key: <domain>//<region>/<team>
X-Git-Identity: domain=<domain>,token=<PAT_or_OAuth>
Content-Type: application/json
```

**Body**
```json
{
  "repo_url": "https://gitlab-scm.company.com/emea/teama/projects/service-a.git",
  "ref": "main",                 // or commit SHA
  "mode": "incremental"          // or "full"
}
```

**Flow (simple & safe)**  
1) Parse `repo_url` → `(domain, root_group, group_path, project)`; verify it matches `X‑Tenant‑Key`.  
2) Resolve **project_id** via GitLab API (once); upsert into `repos`.  
3) **Index job** (enqueue or run inline if small):  
   - `git fetch` (using header token) into a **bare mirror**; checkout **target commit**.  
   - Build **chunks** (method/class first; fallback to line windows), **embed**, upsert to **Milvus** (partition = tenant key).  
   - Extract **symbols/edges**; upsert to **Neo4j** with properties `{tenant, project_id}`.  
   - Store files/manifest in **MinIO** under `/<tenant_key>/<project_id>/<commit>/...`.  
4) Update `repos.last_indexed_commit`.

> For concurrency later: add a per‑repo advisory lock and a background queue. You don’t need it to start.

**Webhook**  
Handle GitLab `push`/`merge_request` → call the same handler with `repo_url` + `ref` (commit SHA).

---

## 5) Searching — UI & REST (flexible scopes)

**Headers**
```
X-Tenant-Key: <domain>//<region>/<team or *>
X-Git-Identity: domain=<domain>,token=<PAT_or_OAuth>
Content-Type: application/json
```

**Body (use URLs, not numeric IDs)**
```json
{
  "query": "where is jwt validated?",
  "mode": "repo" | "list" | "my" | "region",
  "repo_url": "https://gitlab-scm.company.com/emea/teama/projects/service-a.git",
  "repo_urls": ["https://.../service-a.git","https://.../service-b.git"],
  "top_k": 5
}
```

**Scopes (keep it intuitive)**
- `repo`  → single repo (by URL).  
- `list`  → selected repo URLs.  
- `my`    → all repos the token can access **within the tenant** (domain/region/team).  
- `region`→ all repos under **domain/region/* (all teams)** — allowed if `X‑Tenant‑Key` uses `*` (e.g., `gitlab-scm.company.com//emea/*`).

**Flow**  
1) Build **allowed repo set**:  
   - If `repo|list`: resolve each URL → `project_id`; ensure each belongs to tenant.  
   - If `my`: call GitLab projects API using the token, filtered to the tenant’s **root_group**.  
   - If `region`: list all projects under `domain/region/*` (you define which groups).  
2) **Vector recall** from Milvus: `partition_names=[tenant_key]`, filter by `project_id`.  
3) Optional **lexical recall** (Postgres trigram).  
4) **Graph features** (Neo4j) for candidates (callers/callees, centrality), filtered by `{tenant, project_id IN (...)}`.  
5) **Fuse** scores; apply **confidence floor**; **expand** to method/class using MinIO blob.  
6) Return top‑K with code & GitLab deep links.

---

## 6) Milvus / Neo4j / MinIO (minimal conventions)

- **Milvus**: single collection `code_chunks`, **partition per tenant key**. Scalar fields include `tenant_key (string)`, `project_id (int64)`, `commit`, `rel_path`, `lang`, `chunk_id`, `start_line`, `end_line`, `kind` (`code|title`). Metric: IP on normalized vectors.
- **Neo4j**: nodes (`:Method`, `:Class`, `:File`) with props `{tenant, project_id, fqn, rel_path, start, end, lang}`; edges `:CALLS`, etc., also carry `{tenant, project_id}`.
- **MinIO**: `/<tenant_key>/<project_id>/<commit>/files/<rel_path>.gz` + `manifest.json.gz`.

---

## 7) FastAPI — concise models (URL-based)

```python
from pydantic import BaseModel, HttpUrl
from typing import Literal, Optional, List

class IndexRequest(BaseModel):
    repo_url: HttpUrl | str
    ref: str = "main"
    mode: Literal["incremental","full"] = "incremental"

class SearchRequest(BaseModel):
    query: str
    mode: Literal["repo","list","my","region"] = "my"
    repo_url: Optional[str] = None
    repo_urls: Optional[List[str]] = None
    top_k: int = 5
```

**Tenant dependency (simple)**
```python
async def get_tenant_key(request) -> str:
    t = request.headers.get("X-Tenant-Key")
    if not t: raise HTTPException(400, "Missing X-Tenant-Key")
    return t
```

---

## 8) Example `curl`s

**Index from UI/REST**
```bash
curl -X POST https://api.example.com/index \
  -H "X-Tenant-Key: gitlab-scm.company.com//emea/teama" \
  -H "X-Git-Identity: domain=gitlab-scm.company.com,token=REDACTED" \
  -H "Content-Type: application/json" \
  -d '{"repo_url":"https://gitlab-scm.company.com/emea/teama/projects/service-a.git","ref":"main"}'
```

**Search single repo**
```bash
curl -X POST https://api.example.com/search \
  -H "X-Tenant-Key: gitlab-scm.company.com//emea/teama" \
  -H "X-Git-Identity: domain=gitlab-scm.company.com,token=REDACTED" \
  -H "Content-Type: application/json" \
  -d '{"query":"jwt refresh","mode":"repo","repo_url":"https://gitlab-scm.company.com/emea/teama/projects/service-a.git","top_k":5}'
```

**Search across team list**
```bash
curl -X POST https://api.example.com/search \
  -H "X-Tenant-Key: gitlab-scm.company.com//emea/teama" \
  -H "X-Git-Identity: domain=gitlab-scm.company.com,token=REDACTED" \
  -H "Content-Type: application/json" \
  -d '{"query":"token rotation","mode":"list","repo_urls":["https://.../service-a.git","https://.../service-b.git"]}'
```

**Search region‑wide (all teams)**
```bash
curl -X POST https://api.example.com/search \
  -H "X-Tenant-Key: gitlab-scm.company.com//emea/*" \
  -H "X-Git-Identity: domain=gitlab-scm.company.com,token=REDACTED" \
  -H "Content-Type: application/json" \
  -d '{"query":"payment webhook","mode":"region","top_k":5}'
```

---

## 9) Acceptance Criteria (keep it simple)

- All requests require `X‑Tenant‑Key` and are scoped accordingly.  
- Users provide tokens **per request**; no tokens stored.  
- Indexing works from **UI**, **GitLab webhooks**, and **REST** with the **same API**.  
- Searching supports **repo**, **list**, **my**, **region** modes using **repo URLs** (backend resolves IDs).  
- Milvus, Neo4j, MinIO namespaces include the tenant key; results never cross tenants.  

---

## 10) Tasks for GPT (do in order)

1. Add PG tables `tenants`, `repos`; create FastAPI models above.  
2. Implement URL parsing & tenant validation helpers.  
3. Implement `/index` that validates tenant, resolves project_id (GitLab API with header token), runs indexing pipeline, updates `repos`.  
4. Implement `/search` that builds allowed repo set based on `mode`, resolves URLs → project_ids, and scopes vector/graph/file reads to `(tenant_key, project_ids)`.  
5. Add Milvus partitioning by tenant; Neo4j writes/queries with `{tenant, project_id}`; MinIO prefixes.  
6. Provide README snippets (curl examples above).

---

**Small surface, clear rules, scales later.**  
Paste this into VS Code AI to scaffold the controllers, helpers, and clients.
