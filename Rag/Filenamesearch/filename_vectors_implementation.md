# Filename vectors: end-to-end integration (indexing + search)

Below are **5 crisp steps** to make “explain `<file>`” style queries rock‑solid **without adding new infra**. This works with your current stack (API + Milvus + etcd + MinIO).

---

## 1) Define the filename “title vector” representation

Create **one extra embedding per file** that encodes file-identifiers and a small content header. This gives ANN a strong lexical anchor for filename/path queries.

**Template (string you embed):**
```
<basename>
<stem>
<rel_path tokens spaced>
lang=<lang>
<first 150–200 lines of the file>
```

- `basename`: e.g., `auth.service.ts`  
- `stem`: e.g., `auth.service`  
- `rel_path tokens spaced`: e.g., `web src app auth auth service ts`  
- The header gives semantic context even if the filename isn’t present in comments.

---

## 2) Extend your vector schema usage (no schema change required)

Reuse your existing Milvus collection. Just ensure you can distinguish this extra vector:

- Use a **special `chunk_id`** for title vectors: `"{repo_slug}:{rel_path}:__title__"`.  
- Keep normal fields as-is (`repo_url`, `rel_path`, `start_line`, `end_line`, `lang`, `content_hash`, `embedding`).  
- Set `start_line=1`, `end_line=min(200, file_lines)` (it’s a “header span”).  
- (Optional) If you have a `kind` field, set `kind="title"`; if not, the `chunk_id` suffix is enough.

---

## 3) Generate & upsert the title vector during indexing

Add this **once per file** right after you embed normal chunks.

```python
# inside your indexer, after reading file `text` and detecting `lang`
from pathlib import Path

basename = Path(rel_path).name
stem = Path(rel_path).stem
rel_tokens = rel_path.replace("/", " ").replace("\\", " ")
header_lines = text.splitlines()
header = "\\n".join(header_lines[: min(200, len(header_lines))])

title_text = f"{basename}\\n{stem}\\n{rel_tokens}\\nlang={lang}\\n{header}"

title_vec = (await emb.embed([title_text]))[0]

batch_entities.append({
    "chunk_id": f"{repo_slug}:{rel_path}:__title__",
    "repo_url": repo.url,
    "rel_path": rel_path,
    "start_line": 1,
    "end_line": min(200, len(header_lines)),
    "lang": lang,
    "content_hash": sha256_text(title_text),
    "embedding": title_vec,
})
```

> This adds **one** extra embedding per file (tiny cost) and dramatically improves filename/path queries.

---

## 4) Detect filename intent & restrict ANN to candidate paths

Add a lightweight intent check in your `/search` handler. If it looks like a filename (or single token likely to be a stem), **filter vector search to those paths**.

```python
# app/services/searcher.py (sketch)
import re, os
from pathlib import Path

FILENAME_RE = r"[\\w\\-.\\/]+\\.((ts|tsx|js|jsx|cs|java|py))"

def looks_like_filename(q: str) -> bool:
    q = q.strip().lower()
    return bool(re.search(FILENAME_RE, q)) or len(q.split()) == 1  # 'auth'/'jwt' stems

def candidate_paths_from_fs(repo_slug: str, query: str, worktree_root: str) -> list[str]:
    # If you delete the worktree, load a small file-list/manifest from MinIO instead.
    q = query.strip().lower()
    paths = []
    for dp, _, files in os.walk(worktree_root):
        for name in files:
            rel = os.path.relpath(os.path.join(dp, name), worktree_root)
            paths.append(rel)

    # rank: exact basename → exact stem → substring
    exact = [p for p in paths if Path(p).name.lower() == q]
    if exact: return exact
    stems = [p for p in paths if Path(p).stem.lower() == q]
    if stems: return stems
    subs  = [p for p in paths if q in Path(p).name.lower()]
    return subs[:100]

# in handle_search(...)
if looks_like_filename(req.query):
    paths = candidate_paths_from_fs(repo_slug, req.query, worktree_root)
    if paths:
        vec_hits = search_vectors(qvec, req.top_k * 3,
                                  filters={"repo_url": req.repo_url} if req.repo_url else None,
                                  path_list=paths)  # <- **restrict to these files**
        # format and return
```

And extend Milvus search to accept `path_list`:

```python
# app/services/vectorstore.py
def search_vectors(query_vec, top_k, filters=None, path_list=None):
    coll = ensure_collection()
    params = {"metric_type": "IP", "params": {"ef": 128}}
    clauses = []
    if filters:
        for k,v in filters.items():
            if k in ("repo_url","lang"):
                clauses.append(f'{k} == "{v}"')
    if path_list:  # cap to 100-200 items
        quoted = ",".join([f'"{p}"' for p in path_list])
        clauses.append(f"rel_path in [{quoted}]")
    expr = " and ".join(clauses) if clauses else None
    res = coll.search([query_vec], "embedding", param=params, limit=top_k, expr=expr,
                      output_fields=["repo_url","rel_path","start_line","end_line","lang","content_hash"])
    ...
```

> If you delete the local worktree after indexing, load a **manifest** (list of `rel_path`s) from MinIO into memory and match against it instead of walking the filesystem.

---

## 5) Rank boosts & graceful fallback

Make sure filename matches reliably float to the top, and avoid random hits when nothing matches well.

```python
# after you get vec_hits:
from pathlib import Path
q = req.query.strip().lower()

results = []
for h in vec_hits:
    bonus = 0.0
    bn = Path(h["rel_path"]).name.lower()
    st = Path(h["rel_path"]).stem.lower()
    if bn == q:       bonus += 0.25
    elif st == q:     bonus += 0.15
    # combine with your existing hybrid score (semantic + graph)
    score = combine_scores(h["score"], signature_match=0.0, name_sim=0.0, graph_radius=1) + bonus
    results.append({**h, "score": score})

results.sort(key=lambda x: x["score"], reverse=True)
results = results[:req.top_k]

# Confidence floor: if nothing good, return candidates instead of a random snippet
if not results or results[0]["score"] < YOUR_CONFIDENCE_THRESH:
    return {
      "total_results": 0,
      "hits": [],
      "file_candidates": [{"rel_path": p} for p in (paths if looks_like_filename(req.query) else [])]
    }
```

- **Bonus** ensures exact basename/stem dominates.  
- **Confidence floor** prevents “random nearby chunk” when there’s no true match.  
- If you also indexed the **title vector**, it will naturally appear as a high-scoring hit for the right file.

---

### That’s it
With these 5 steps you’ll:
- Guarantee reliable **filename/path** queries,
- Keep **zero extra infra**,
- And preserve your current semantic + graph design.
