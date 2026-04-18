import os
from pathlib import Path

from multiagent.harness.cache import PromptCache
from multiagent.harness.router import ModelRouter


def test_router_pick_escalation() -> None:
    r = ModelRouter({"cheap": "c", "mid": "m", "strong": "s"})
    assert r.pick("cheap") == "c"
    assert r.pick("cheap", escalate=True) == "m"
    assert r.pick("mid", escalate=True) == "s"
    assert r.pick("strong", escalate=True) == "s"


def test_router_dry_run_populates_cache(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("MULTIAGENT_DRY_RUN", "1")
    r = ModelRouter({"cheap": "c", "mid": "m", "strong": "s"})
    cache = PromptCache(tmp_path)
    resp = r.call("c", [{"type": "text", "text": "sys"}],
                  [{"role": "user", "content": "hi"}], cache)
    assert resp.cost_usd == 0.0
    assert resp.cached is False
    resp2 = r.call("c", [{"type": "text", "text": "sys"}],
                   [{"role": "user", "content": "hi"}], cache)
    assert resp2.cached is True
