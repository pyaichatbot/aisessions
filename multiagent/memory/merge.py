from __future__ import annotations

from pathlib import Path

from .versioning import JsonlLog


def merge_wikis(log_a: Path, log_b: Path, out: Path) -> int:
    """Conflict-free merge: concatenate events, dedupe by id.

    Git's default merge on appended JSONL files already works — this helper
    is for manual/offline merges. Returns number of events written.
    """
    seen: set[str] = set()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as w:
        for src in (log_a, log_b):
            src_log = JsonlLog(src)
            for e in src_log.iter_entries():
                if e.id in seen:
                    continue
                seen.add(e.id)
                w.write(_to_line(e) + "\n")
    return len(seen)


def _to_line(e) -> str:
    import json
    from dataclasses import asdict
    return json.dumps(asdict(e), sort_keys=True)
