from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SharedState:
    """Cross-agent shared blackboard. JSON-serializable values only."""
    path: Path
    data: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                self.data = json.loads(self.path.read_text("utf-8"))
            except json.JSONDecodeError:
                self.data = {}

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            return self.data.get(key, default)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            self.data[key] = value
            self._flush()

    def append(self, key: str, value: Any) -> None:
        with self._lock:
            self.data.setdefault(key, []).append(value)
            self._flush()

    def _flush(self) -> None:
        self.path.write_text(json.dumps(self.data, indent=2, default=str), "utf-8")
