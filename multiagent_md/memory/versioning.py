from __future__ import annotations

import json
import os
import threading
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterator


@dataclass
class LogEntry:
    """Append-only event. LWW + union CRDT keys."""
    id: str
    ts: float
    actor: str
    page: str
    op: str            # upsert | append | retract
    payload: dict
    parent: str | None = None


class JsonlLog:
    """Append-only JSONL log = zero merge conflicts.

    Two branches both append distinct lines. A git merge concatenates them; order
    is resolved by `ts` + `id` at read time. Retractions are tombstones.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        if not self.path.exists():
            self.path.touch()

    def append(self, actor: str, page: str, op: str, payload: dict,
               parent: str | None = None) -> LogEntry:
        entry = LogEntry(
            id=uuid.uuid4().hex,
            ts=time.time(),
            actor=actor,
            page=page,
            op=op,
            payload=payload,
            parent=parent,
        )
        line = json.dumps(asdict(entry), sort_keys=True)
        with self._lock:
            # Append under O_APPEND — concurrent writers safe on POSIX.
            fd = os.open(self.path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
            try:
                os.write(fd, (line + "\n").encode("utf-8"))
            finally:
                os.close(fd)
        return entry

    def iter_entries(self) -> Iterator[LogEntry]:
        if not self.path.exists():
            return iter(())
        out: list[LogEntry] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    out.append(LogEntry(**d))
                except (json.JSONDecodeError, TypeError):
                    continue
        out.sort(key=lambda e: (e.ts, e.id))
        return iter(out)

    def materialize(self) -> dict[str, dict]:
        """Fold events into per-page state (union CRDT).

        Two passes: first collect tombstones across the whole log so retractions
        apply regardless of order. Then replay non-retracted events.
        """
        entries = list(self.iter_entries())
        retracted: set[str] = set()
        for e in entries:
            if e.op == "retract":
                tid = e.payload.get("target_id", "")
                if tid:
                    retracted.add(tid)
        pages: dict[str, dict] = {}
        for e in entries:
            if e.op == "retract" or e.id in retracted:
                continue
            page = pages.setdefault(e.page, {"facts": [], "notes": []})
            if e.op == "upsert":
                for k, v in e.payload.items():
                    page[k] = v
            elif e.op == "append":
                for k, v in e.payload.items():
                    bucket = page.setdefault(k, [])
                    if isinstance(bucket, list):
                        if v not in bucket:
                            bucket.append(v)
                    else:
                        page[k] = v
        # Drop pages that only contain the default empty buckets.
        return {k: v for k, v in pages.items()
                if any(val not in ([], "", None) for key, val in v.items()
                       if key not in ("facts", "notes") or val)}
