from pathlib import Path

from multiagent.memory import JsonlLog, WikiMemory, merge_wikis


def test_jsonl_log_append_and_materialize(tmp_path: Path) -> None:
    log = JsonlLog(tmp_path / "events.jsonl")
    log.append("coder", "notes", "append", {"notes": "a"})
    log.append("coder", "notes", "append", {"notes": "b"})
    log.append("planner", "plan", "upsert", {"summary": "x"})
    pages = log.materialize()
    assert pages["notes"]["notes"] == ["a", "b"]
    assert pages["plan"]["summary"] == "x"


def test_retract_is_honored(tmp_path: Path) -> None:
    log = JsonlLog(tmp_path / "events.jsonl")
    e = log.append("coder", "p", "upsert", {"k": 1})
    log.append("coder", "*", "retract", {"target_id": e.id})
    pages = log.materialize()
    assert "p" not in pages


def test_union_merge_no_conflict(tmp_path: Path) -> None:
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    JsonlLog(a).append("x", "p", "append", {"notes": "A1"})
    JsonlLog(b).append("y", "p", "append", {"notes": "B1"})
    out = tmp_path / "merged.jsonl"
    n = merge_wikis(a, b, out)
    assert n == 2
    pages = JsonlLog(out).materialize()
    assert set(pages["p"]["notes"]) == {"A1", "B1"}


def test_wiki_context_retrieval(tmp_path: Path) -> None:
    wm = WikiMemory(tmp_path / "wiki", topk=3)
    wm.record_turn("coder", "add retry to fetcher", "diff adds retry wrapper")
    wm.upsert("conventions", {"notes": "use requests.Session for retries"})
    ctx = wm.context_for("coder", query="retry fetcher")
    assert "retry" in ctx.lower()
