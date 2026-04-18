from __future__ import annotations

import threading
from dataclasses import dataclass, field


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class BudgetTracker:
    cap_usd: float
    spent_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    calls: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def charge(self, usd: float, tin: int, tout: int) -> None:
        with self._lock:
            self.spent_usd += usd
            self.tokens_in += tin
            self.tokens_out += tout
            self.calls += 1
            if self.spent_usd > self.cap_usd:
                raise BudgetExceeded(
                    f"budget {self.cap_usd:.2f} exceeded: {self.spent_usd:.2f}"
                )

    def snapshot(self) -> dict[str, float | int]:
        with self._lock:
            return {
                "spent_usd": round(self.spent_usd, 4),
                "cap_usd": self.cap_usd,
                "tokens_in": self.tokens_in,
                "tokens_out": self.tokens_out,
                "calls": self.calls,
            }
