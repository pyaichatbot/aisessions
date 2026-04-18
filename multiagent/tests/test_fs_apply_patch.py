from pathlib import Path

from multiagent.tools.fs import apply_patch


def test_file_block_creates_file(tmp_path: Path) -> None:
    text = """some preamble
```file:pkg/hello.py
print("hi")
```
tail
"""
    ok = apply_patch(tmp_path, text)
    assert ok
    assert (tmp_path / "pkg/hello.py").read_text("utf-8").strip() == 'print("hi")'


def test_no_fences_returns_false(tmp_path: Path) -> None:
    ok = apply_patch(tmp_path, "just prose, no diff")
    assert ok is False
