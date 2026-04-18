from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class CacheHit:
    text: str
    tokens_in: int
    tokens_out: int


class PromptCache:
    """Disk-backed exact-match cache for deterministic calls.

    Complements Anthropic server-side ephemeral cache. Skip cost entirely
    when the same prompt+messages hash was seen before.
    """

    def __init__(self, root: Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    @staticmethod
    def _key(model: str, system: Any, messages: Any) -> str:
        payload = json.dumps(
            {"m": model, "s": system, "u": messages}, sort_keys=True, default=str
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, model: str, system: Any, messages: Any) -> CacheHit | None:
        if not self.enabled:
            return None
        path = self.root / self._key(model, system, messages)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text("utf-8"))
            return CacheHit(data["text"], data["tokens_in"], data["tokens_out"])
        except (OSError, json.JSONDecodeError, KeyError):
            return None

    def put(self, model: str, system: Any, messages: Any, hit: CacheHit) -> None:
        if not self.enabled:
            return
        path = self.root / self._key(model, system, messages)
        with self._lock:
            tmp = path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "text": hit.text,
                        "tokens_in": hit.tokens_in,
                        "tokens_out": hit.tokens_out,
                        "ts": time.time(),
                    }
                ),
                "utf-8",
            )
            os.replace(tmp, path)
