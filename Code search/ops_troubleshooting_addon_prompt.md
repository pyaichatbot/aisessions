
# Ops Troubleshooting Add‑On — Implementation Prompt (for GPT in VS Code)

> Paste this into your VS Code AI chat (Copilot Chat/Cursor/Windsurf) to scaffold features that let you pivot from **Splunk logs → exact code → impact graph** using your existing stack (**FastAPI + Milvus + Neo4j + Postgres + MinIO**).  
> The goal: production‑ready, multi‑tenant, secure, performant.

---

## 0) Scope & Outcomes

**You will implement:**
1) **Log‑to‑code reverse index** (extract logger message templates & exceptions at index time).  
2) **Stack trace → symbol mapper** (resolve frames to methods/classes).  
3) **Route/endpoint catalog** (map HTTP routes to handler methods).  
4) **Signals ingestion** (Splunk alert webhook & optional poller).  
5) **Ops search API** that accepts a raw log/stack/route and returns the code bundle (method/class + callers/callees + deep links).

**Non‑goals:** build a Splunk replacement; store full logs. We only persist **templates, hashes, and minimal signals** needed to pivot to code.

---

## 1) Data Model (Postgres + Neo4j + Milvus + MinIO)

### Postgres (control plane & signals)
Create tables (SQLAlchemy/Alembic):
```sql
-- minimal signals store (aggregates, not raw logs)
CREATE TABLE signals (
  id BIGSERIAL PRIMARY KEY,
  tenant_id UUID NOT NULL,
  service TEXT,
  route TEXT,
  level TEXT,
  message TEXT,              -- redacted sample or template
  message_hash TEXT,         -- sha256 of normalized template
  stack TEXT,                -- optional, trimmed
  symbol_fqn TEXT,           -- optional resolved
  first_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
  last_seen TIMESTAMPTZ NOT NULL DEFAULT now(),
  count_5m INT NOT NULL DEFAULT 1
);
CREATE INDEX ON signals(tenant_id, last_seen DESC);
CREATE INDEX ON signals(tenant_id, message_hash);
```

### Neo4j (graph edges)
- Add `(:Method)-[:EMITS_LOG {tenant, message_hash, template}]` to link log templates to emitters.
- Add `(:Route {tenant, method, path})-[:HANDLED_BY]->(:Method)` for handlers.

### Milvus (vectors for log templates)
- Collection: reuse `code_chunks` **or** create `log_titles` with `dim=embed_dim` and fields `{tenant_id, repo_id, commit, rel_path, fqn, message_hash, kind='log'}`.
- Store an embedding of the **normalized message template** (not raw PII).

### MinIO
- No new buckets needed; continue storing file blobs & manifests.

---

## 2) Indexer Add‑Ons (run during shard indexing)

### 2.1 Extract logger messages & exceptions
Implement per‑language extractors (C#, TS/JS, Python, Java). Look for:
- C#: `logger.Log*("...")`, `Log.*("...")`, `throw new X("...")`
- TS/JS: `logger.info("...")`, `console.error("...")`
- Python: `logger.info("...")`, `raise X("...")`
- Java: `log.info("...")`, `throw new X("...")`

**Normalize template**:
- Lowercase, trim whitespace.
- Replace variables `{userId}`, `%s`, `{0}` with `{}`.
- Drop high‑entropy IDs/emails via regex.

**Emit record** (JSONL per file):
```json
{"type":"log_template",
 "tenant":"<uuid>",
 "repo_id":123,
 "commit":"<sha>",
 "rel_path":"src/auth/JwtService.cs",
 "lang":"cs",
 "fqn":"My.Auth.JwtService.Validate",
 "start":66, "end":118,
 "template":"[auth] refresh failed user={}",
 "message_hash":"sha256:..."
}
```

**Actions**:
- Upsert **Neo4j** edge `(:Method {fqn})-[:EMITS_LOG {tenant,message_hash,template}]->(:Log {message_hash})` (you may model `:Log` explicitly or store as edge props).
- Embed the **template** and insert into **Milvus** (`kind='log'`), keyed by `(tenant, repo_id, commit, fqn, message_hash)`.

### 2.2 Extract routes/endpoints
- ASP.NET: `[HttpGet("/api/orders")]`, controller/action → FQN.
- Express: `router.get('/api/orders', handler)` → handler function FQN.
- FastAPI: `@app.post("/api/charge")`.
- Spring: `@RequestMapping(path="/api/charge", method=POST)`.

**Create**: `(:Route {tenant, method, path})-[:HANDLED_BY]->(:Method {fqn})` in **Neo4j**.

---

## 3) Stack Trace Mapping (service)

Add a small module to parse stack traces:
- Detect language by frame shape.
- Patterns:
  - C#: `at Namespace.Class.Method(Type arg) in /path/File.cs:line 123`
  - Java: `at pkg.Class.method(Class.java:123)`
  - Python: `File "x.py", line 10, in func`
  - Node: `at func (file.js:123:45)`

For each frame, resolve to **FQN + span**:
- Prefer Neo4j: `MATCH (m:Method {tenant:$t}) WHERE m.rel_path=$path AND $line BETWEEN m.start AND m.end RETURN m.fqn`
- Else use `symbol_at` fallback (tree-sitter).

Return a ranked list of candidate symbols.

---

## 4) APIs (FastAPI)

### 4.1 Ops search endpoint
`POST /search/log`
```json
{
  "tenant_id": "uuid",
  "message": "Failed to refresh token for user 42",
  "stack": "optional stack trace text",
  "route": "/api/auth/refresh",
  "service": "auth-service",
  "top_k": 5
}
```

**Server steps:**
1) **If stack provided**: parse → map frames → collect candidate FQNs.
2) **If message provided**:
   - Normalize → hash → try **exact message_hash** via Neo4j EMITS_LOG.
   - Else embed template → Milvus ANN on `kind='log'`, `tenant=...`.
3) **If route provided**: Neo4j `(:Route {tenant, path})-[:HANDLED_BY]->(:Method)` and union candidates.
4) **Graph expand** for final candidates: callers/callees; attach centrality.
5) **Bundle**: for each hit → fetch full method/class from MinIO (commit+rel_path+span), include callers/callees.
6) **Re-rank**: combine hash hit > stack overlap > route match > ANN similarity > centrality.
7) Return `{symbol, code, rel_path, start_line, end_line, callers, callees, open_in_gitlab}`.

### 4.2 Splunk webhook
`POST /signals/splunk`
```json
{
  "tenant_id":"uuid",
  "service":"auth-service",
  "route":"/api/auth/refresh",
  "level":"ERROR",
  "message":"Failed to refresh token for user 42",
  "stack":"at ...",
  "ts":"2025-02-18T10:01:00Z"
}
```
- Normalize message → hash; upsert into `signals` (`last_seen`, `count_5m += 1`).

### 4.3 Saved searches poller (optional)
Worker that hits Splunk saved search API every N minutes, ingests results into `signals`.

---

## 5) Reranking (keep it transparent)

Score contributions (example weights):
- `+1.0` if `message_hash` exact match to EMITS_LOG.
- `+0.6` * stack overlap score (`frames_matched / frames_seen`).
- `+0.4` * ANN similarity (log template embedding).
- `+0.2` * route handler match.
- `+0.1` * normalized centrality (Neo4j).

Return top‑K after fusion. Apply a **confidence floor** (e.g., 0.35) to avoid junk.

---

## 6) Redaction & Privacy

- Only embed **normalized templates** (replace IDs, emails, GUIDs with `{}` or `<redacted>`).
- Do not store raw sensitive values; keep aggregates in `signals` only.
- Never log incoming messages/tokens.

---

## 7) Observability

- Log decision reasons: `reason=hash_hit|stack_match|route_match|ann` and scores.
- Metrics (Prometheus): time to resolve, matches per method, webhook ingest rate.

---

## 8) Docker/Compose (persistence)

Ensure volumes are persisted (examples):
```yaml
services:
  neo4j:
    image: neo4j:5.22
    ports: ["7474:7474","7687:7687"]
    volumes:
      - ./neo4j/data:/data
      - ./neo4j/logs:/logs
      - ./neo4j/conf:/conf
    environment: [NEO4J_AUTH=neo4j/secret]
  milvus: { ... volumes ... }
  postgres: { volumes: ["./pg/data:/var/lib/postgresql/data"] }
  minio: { volumes: ["./minio/data:/data"] }
  api: { depends_on: [neo4j, milvus, postgres, minio] }
```

---

## 9) Acceptance Tests (must pass)

- **Log → Code**: Given a known logger template, `/search/log` returns the correct method/class with span.
- **Stack → Code**: Given a sample C#/TS/Java/Python stack, resolver returns top symbol and its callers.
- **Route → Code**: Given `/api/charge`, returns the correct handler and neighbors.
- **Redaction**: PII is removed from embeddings and persisted messages.
- **Persistence**: Restart containers; Neo4j/Milvus/Postgres data remains intact.

---

## 10) Tasks for GPT (execute in this order)

1. Add Alembic migration for `signals` table.  
2. Extend indexer: per‑language logger/throw & route extractors; normalize templates; emit JSONL; write Neo4j EMITS_LOG edges; insert log‑title vectors.  
3. Implement stack trace parser + resolver service.  
4. Implement `POST /search/log` with fusion scoring and code bundle expansion.  
5. Implement `POST /signals/splunk` and (optional) poller.  
6. Add unit tests for normalization, hash, resolver, and rerank.  
7. Wire observability & Docker changes (persist volumes).  
8. Update README with curl examples.

---

## 11) Example Requests

**A) Log to Code**
```http
POST /search/log
X-Tenant-Id: c0a8...-unitA
Content-Type: application/json

{
  "message": "Failed to refresh token for user 42",
  "service": "auth-service",
  "top_k": 3
}
```

**B) Stack to Code**
```http
POST /search/log
X-Tenant-Id: c0a8...-unitA
Content-Type: application/json

{
  "stack": "at My.Auth.JwtService.Validate(String user) in /src/auth/JwtService.cs:line 87\n at ...",
  "top_k": 3
}
```

**C) Route to Handler**
```http
POST /search/log
X-Tenant-Id: c0a8...-unitA
Content-Type: application/json

{
  "route": "/api/auth/refresh",
  "top_k": 3
}
```
