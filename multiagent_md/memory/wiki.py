from __future__ import annotations

from pathlib import Path
from typing import Any

from .indexer import TfidfIndex
from .versioning import JsonlLog


class WikiMemory:
    """Karpathy-style LLM Wiki.

    Layout under `root/`:
      - events.jsonl        append-only CRDT log (merge-conflict-free)
      - pages/<name>.md     rendered snapshots (regenerated, optional)

    Agents read `context_for(role)` to inject short, retrieved memory into
    their system prompt. Agents write with `record_turn` and `upsert`.
    """

    def __init__(self, root: Path, page_bytes_max: int = 8192, topk: int = 6) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.log = JsonlLog(self.root / "events.jsonl")
        self.page_bytes_max = page_bytes_max
        self.topk = topk
        self._index = TfidfIndex()
        self._rebuild_index()

    def _rebuild_index(self) -> None:
        pages = self.log.materialize()
        for name, body in pages.items():
            self._index.add(name, self._render_page(name, body))

    def record_turn(self, role: str, task: str, output: str) -> None:
        snippet = output.strip()[: self.page_bytes_max]
        self.log.append(
            actor=role,
            page=f"turns/{role}",
            op="append",
            payload={"notes": f"TASK: {task[:200]}\nOUT: {snippet}"},
        )
        self._index.add(
            f"turns/{role}",
            self._render_page(f"turns/{role}", self.log.materialize().get(f"turns/{role}", {})),
        )

    def upsert(self, page: str, fields: dict[str, Any], actor: str = "system") -> None:
        self.log.append(actor=actor, page=page, op="upsert", payload=fields)
        body = self.log.materialize().get(page, {})
        self._index.add(page, self._render_page(page, body))

    def retract(self, target_id: str, actor: str = "system") -> None:
        self.log.append(actor=actor, page="*", op="retract",
                        payload={"target_id": target_id})

    def context_for(self, role: str, query: str | None = None) -> str:
        pages = self.log.materialize()
        q = query or f"role:{role}"
        ranked = self._index.rank(q, topk=self.topk)
        blocks = []
        for name, _ in ranked:
            body = pages.get(name, {})
            rendered = self._render_page(name, body)[: self.page_bytes_max]
            blocks.append(f"### {name}\n{rendered}")
        return "\n\n".join(blocks)

    def render_snapshots(self) -> None:
        """Optional: render markdown files for humans. Events remain source of truth."""
        pages_dir = self.root / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        for name, body in self.log.materialize().items():
            path = pages_dir / f"{name.replace('/', '__')}.md"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(self._render_page(name, body), "utf-8")

    @staticmethod
    def _render_page(name: str, body: dict[str, Any]) -> str:
        lines = [f"# {name}"]
        for k, v in body.items():
            if isinstance(v, list):
                lines.append(f"\n## {k}")
                for item in v[-40:]:
                    lines.append(f"- {item}")
            else:
                lines.append(f"\n## {k}\n{v}")
        return "\n".join(lines)
