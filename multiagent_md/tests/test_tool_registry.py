from pathlib import Path

from multiagent_md.harness.tool_registry import ToolRegistry


def test_schema_for_filters_unknown(tmp_path: Path) -> None:
    r = ToolRegistry(tmp_path)
    schema = r.schema_for(["read_file", "write_file", "nonexistent"])
    names = [t["name"] for t in schema]
    assert names == ["read_file", "write_file"]


def test_read_write_roundtrip(tmp_path: Path) -> None:
    r = ToolRegistry(tmp_path)
    out = r.call("write_file", {"path": "a/b.txt", "content": "hi"})
    assert "wrote" in out
    assert (tmp_path / "a" / "b.txt").read_text() == "hi"
    got = r.call("read_file", {"path": "a/b.txt"})
    assert got == "hi"


def test_grep_finds_matches(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def foo():\n    return 1\n")
    r = ToolRegistry(tmp_path)
    out = r.call("grep", {"pattern": r"def \w+"})
    assert "foo" in out


def test_path_escape_blocked(tmp_path: Path) -> None:
    r = ToolRegistry(tmp_path)
    out = r.call("read_file", {"path": "../etc/passwd"})
    assert "error" in out.lower()
