from __future__ import annotations

import re
import subprocess
from pathlib import Path


_DIFF_BLOCK = re.compile(r"```(?:diff|patch)\s*\n(.*?)```", re.DOTALL)
_FILE_BLOCK = re.compile(
    r"```file:(?P<path>[^\n]+)\n(?P<body>.*?)```", re.DOTALL
)


def read_file(root: Path, rel: str, max_bytes: int = 200_000) -> str:
    p = _safe(root, rel)
    data = p.read_text("utf-8")
    if len(data) > max_bytes:
        data = data[:max_bytes] + f"\n...[truncated {len(data)-max_bytes} bytes]"
    return data


def write_file(root: Path, rel: str, content: str) -> str:
    p = _safe(root, rel)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, "utf-8")
    return f"wrote {len(content)} bytes to {rel}"


def apply_patch(root: Path, text: str) -> bool:
    changed = False
    for m in _FILE_BLOCK.finditer(text):
        rel = m.group("path").strip()
        body = m.group("body")
        if not body.endswith("\n"):
            body += "\n"
        write_file(root, rel, body)
        changed = True
    for m in _DIFF_BLOCK.finditer(text):
        patch = m.group(1)
        tmp = root / ".multiagent_md" / "tmp.patch"
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


def apply_diff(root: Path, diff_text: str) -> str:
    ok = apply_patch(root, f"```diff\n{diff_text}\n```")
    return "applied" if ok else "failed"


def list_files(root: Path, subdir: str = ".", limit: int = 200) -> str:
    base = _safe(root, subdir)
    out = []
    for p in sorted(base.rglob("*")):
        if ".git" in p.parts or ".multiagent_md" in p.parts:
            continue
        if p.is_file():
            out.append(str(p.relative_to(root)))
            if len(out) >= limit:
                out.append("...")
                break
    return "\n".join(out)


def grep(root: Path, pattern: str, subdir: str = ".", max_hits: int = 60) -> str:
    base = _safe(root, subdir)
    rx = re.compile(pattern)
    hits: list[str] = []
    for p in base.rglob("*"):
        if ".git" in p.parts or ".multiagent_md" in p.parts or not p.is_file():
            continue
        try:
            for i, line in enumerate(p.read_text("utf-8", errors="ignore").splitlines(), 1):
                if rx.search(line):
                    hits.append(f"{p.relative_to(root)}:{i}: {line.strip()[:200]}")
                    if len(hits) >= max_hits:
                        return "\n".join(hits + ["...truncated"])
        except OSError:
            continue
    return "\n".join(hits) or "no matches"


def _safe(root: Path, rel: str) -> Path:
    p = (root / rel).resolve()
    root_r = root.resolve()
    if root_r not in p.parents and p != root_r:
        raise ValueError(f"path escapes root: {rel}")
    return p
