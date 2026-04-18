from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from ..tools import fs as _fs
from ..tools.test_runner import run_full_suite


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict[str, Any]
    handler: Callable[[dict[str, Any]], str]


class ToolRegistry:
    """Maps tool names (from subagent md) to concrete Python callables.

    Tools operate inside `workdir` (worktree or branch). Paths are sandboxed
    in `fs` helpers via `_safe`.
    """

    def __init__(self, workdir: Path) -> None:
        self.workdir = workdir
        self._tools: dict[str, Tool] = {}
        self._register_defaults()

    def _register_defaults(self) -> None:
        def _read(args):
            return _fs.read_file(self.workdir, args["path"],
                                 max_bytes=args.get("max_bytes", 200_000))

        def _write(args):
            return _fs.write_file(self.workdir, args["path"], args["content"])

        def _apply(args):
            return _fs.apply_diff(self.workdir, args["diff"])

        def _list(args):
            return _fs.list_files(self.workdir, args.get("subdir", "."),
                                  limit=args.get("limit", 200))

        def _grep(args):
            return _fs.grep(self.workdir, args["pattern"],
                            subdir=args.get("subdir", "."),
                            max_hits=args.get("max_hits", 60))

        def _run_tests(_args):
            rep = run_full_suite(self.workdir)
            return json.dumps({
                "passed": rep.passed,
                "coverage": rep.coverage,
                "failing": rep.failing,
                "logs_tail": rep.logs[-4000:],
            })

        self.register(Tool(
            name="read_file",
            description="Read a file relative to workdir.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_bytes": {"type": "integer"},
                },
                "required": ["path"],
            },
            handler=_read,
        ))
        self.register(Tool(
            name="write_file",
            description="Create or overwrite file at path with content.",
            input_schema={
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
            handler=_write,
        ))
        self.register(Tool(
            name="apply_diff",
            description="Apply a unified diff to the working tree.",
            input_schema={
                "type": "object",
                "properties": {"diff": {"type": "string"}},
                "required": ["diff"],
            },
            handler=_apply,
        ))
        self.register(Tool(
            name="list_files",
            description="List files under subdir.",
            input_schema={
                "type": "object",
                "properties": {
                    "subdir": {"type": "string"},
                    "limit": {"type": "integer"},
                },
            },
            handler=_list,
        ))
        self.register(Tool(
            name="grep",
            description="Search regex across files.",
            input_schema={
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "subdir": {"type": "string"},
                    "max_hits": {"type": "integer"},
                },
                "required": ["pattern"],
            },
            handler=_grep,
        ))
        self.register(Tool(
            name="run_tests",
            description="Run full test suite. Returns JSON with coverage + failing.",
            input_schema={"type": "object", "properties": {}},
            handler=_run_tests,
        ))

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def schema_for(self, names: list[str]) -> list[dict[str, Any]]:
        out = []
        for n in names:
            t = self._tools.get(n)
            if not t:
                continue
            out.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.input_schema,
            })
        return out

    def call(self, name: str, args: dict[str, Any]) -> str:
        t = self._tools.get(name)
        if not t:
            return f"error: unknown tool {name}"
        try:
            return str(t.handler(args))
        except Exception as e:  # noqa: BLE001 — tool errors bubble to agent
            return f"error: {type(e).__name__}: {e}"
