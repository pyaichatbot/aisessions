from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base import Agent, AgentResult


@dataclass
class SubTask:
    id: str
    title: str
    description: str
    depends_on: list[str] = field(default_factory=list)
    parallel_group: str | None = None
    files_hint: list[str] = field(default_factory=list)


@dataclass
class Plan:
    summary: str
    complexity: str  # simple | medium | complex
    subtasks: list[SubTask]
    raw: str = ""

    def batches(self) -> list[list[SubTask]]:
        """Return execution batches honoring deps + parallel groups."""
        remaining = {t.id: t for t in self.subtasks}
        done: set[str] = set()
        batches: list[list[SubTask]] = []
        while remaining:
            ready = [t for t in remaining.values() if all(d in done for d in t.depends_on)]
            if not ready:
                batches.append(list(remaining.values()))
                break
            groups: dict[str, list[SubTask]] = {}
            for t in ready:
                key = t.parallel_group or t.id
                groups.setdefault(key, []).append(t)
            chosen = max(groups.values(), key=len)
            batches.append(chosen)
            for t in chosen:
                done.add(t.id)
                remaining.pop(t.id, None)
        return batches


class Planner(Agent):
    role = "planner"
    tier = "mid"
    prompt_file = "planner.md"

    def plan(self, task: str, context: dict[str, Any] | None = None) -> Plan:
        ctx = dict(context or {})
        ctx["output_format"] = "json"
        result = self.run(task, ctx)
        plan = self._parse(result.output)
        plan.raw = result.output
        return plan

    @staticmethod
    def _parse(text: str) -> Plan:
        block = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        payload = block.group(1) if block else text
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return Plan(summary=text[:200], complexity="simple", subtasks=[
                SubTask(id="t1", title="root", description=text)
            ])
        subs = [
            SubTask(
                id=s["id"],
                title=s.get("title", s["id"]),
                description=s.get("description", ""),
                depends_on=s.get("depends_on", []),
                parallel_group=s.get("parallel_group"),
                files_hint=s.get("files_hint", []),
            )
            for s in data.get("subtasks", [])
        ]
        return Plan(
            summary=data.get("summary", ""),
            complexity=data.get("complexity", "simple"),
            subtasks=subs,
        )
