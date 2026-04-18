import pytest
from pathlib import Path

from multiagent.harness.budget import BudgetTracker, BudgetExceeded
from multiagent.harness.cache import PromptCache, CacheHit


def test_budget_charges_and_raises(tmp_path: Path) -> None:
    b = BudgetTracker(cap_usd=0.10)
    b.charge(0.03, 10, 20)
    b.charge(0.06, 5, 5)
    with pytest.raises(BudgetExceeded):
        b.charge(0.05, 1, 1)
    snap = b.snapshot()
    assert snap["calls"] == 3


def test_cache_hit_roundtrip(tmp_path: Path) -> None:
    cache = PromptCache(tmp_path)
    system = [{"type": "text", "text": "sys"}]
    msgs = [{"role": "user", "content": "hi"}]
    assert cache.get("m", system, msgs) is None
    cache.put("m", system, msgs, CacheHit("ok", 1, 2))
    hit = cache.get("m", system, msgs)
    assert hit is not None and hit.text == "ok"
