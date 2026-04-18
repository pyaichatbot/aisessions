from __future__ import annotations

import re
import subprocess
from pathlib import Path


_DIFF_BLOCK = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_FILE_BLOCK = re.compile(
    r"```file:(?P<path>[^\n]+)\n(?P<body>.*?)```", re.DOTALL
)


def read_file(root: Path, rel: str) -> str:
    p = root / rel
    return p.read_text("utf-8")


def write_file(root: Path, rel: str, content: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")


def apply_patch(root: Path, text: str) -> bool:
    """Apply agent output. Supports two encodings:
       - unified diff in ```diff``` fence → `git apply`
       - explicit write blocks: ```file:path/to/file\n<content>\n```
    Returns True if at least one change applied.
    """
    changed = False
    for m in _FILE_BLOCK.finditer(text):
        rel = m.group("path").strip()
        body = m.group("body")
        if body.endswith("\n"):
            body_to_write = body
        else:
            body_to_write = body + "\n"
        write_file(root, rel, body_to_write)
        changed = True
    for m in _DIFF_BLOCK.finditer(text):
        patch = m.group(1)
        tmp = root / ".multiagent" / "tmp.patch"
        tmp.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(patch, "utf-8")
        r = subprocess.run(
            ["git", "apply", "--whitespace=nowarn", str(tmp)],
            cwd=str(root), text=True, capture_output=True,
        )
        if r.returncode == 0:
            changed = True
        else:
            r2 = subprocess.run(
                ["git", "apply", "--3way", "--whitespace=nowarn", str(tmp)],
                cwd=str(root), text=True, capture_output=True,
            )
            if r2.returncode == 0:
                changed = True
    return changed
