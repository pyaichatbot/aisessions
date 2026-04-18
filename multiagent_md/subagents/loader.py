from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class SubagentDef:
    name: str
    description: str
    model_tier: str
    tools: list[str] = field(default_factory=list)
    output_format: str = "text"
    max_tokens: int = 4096
    max_tool_iters: int = 10
    system_prompt: str = ""
    source_path: Path | None = None


_FM = re.compile(r"^---\n(.*?)\n---\n(.*)$", re.DOTALL)


def parse_md(text: str) -> tuple[dict[str, Any], str]:
    m = _FM.match(text)
    if not m:
        return {}, text
    fm = yaml.safe_load(m.group(1)) or {}
    return fm, m.group(2).lstrip("\n")


def load_subagent(path: Path) -> SubagentDef:
    fm, body = parse_md(path.read_text("utf-8"))
    return SubagentDef(
        name=fm.get("name", path.stem),
        description=fm.get("description", ""),
        model_tier=fm.get("model_tier", "cheap"),
        tools=list(fm.get("tools", [])),
        output_format=fm.get("output_format", "text"),
        max_tokens=int(fm.get("max_tokens", 4096)),
        max_tool_iters=int(fm.get("max_tool_iters", 10)),
        system_prompt=body,
        source_path=path,
    )


def list_subagents(root: Path | None = None) -> dict[str, SubagentDef]:
    root = root or Path(__file__).resolve().parent
    out: dict[str, SubagentDef] = {}
    for p in sorted(root.glob("*.md")):
        sa = load_subagent(p)
        out[sa.name] = sa
    return out
