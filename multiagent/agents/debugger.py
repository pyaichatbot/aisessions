from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from .base import Agent, AgentResult
from .planner import Plan, SubTask


@dataclass
class FixPlan:
    root_cause: str
    affected: list[str] = field(default_factory=list)
    subtasks: list[SubTask] = field(default_factory=list)
    complexity: str = "simple"
    raw: str = ""

    def batches(self) -> list[list[SubTask]]:
        return Plan(summary=self.root_cause, complexity=self.complexity,
                    subtasks=self.subtasks).batches()


class Debugger(Agent):
    role = "debugger"
    tier = "strong"
    prompt_file = "debugger.md"

    def diagnose(self, failing_logs: str, context: dict[str, Any]) -> FixPlan:
        ctx = dict(context)
        ctx["failing_logs"] = failing_logs
        ctx["output_format"] = "json"
        result = self.run("Diagnose and produce fix plan.", ctx)
        return self._parse(result.output)

    @staticmethod
    def _parse(text: str) -> FixPlan:
        block = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        payload = block.group(1) if block else text
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            return FixPlan(root_cause=text[:200], raw=text)
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
        return FixPlan(
            root_cause=data.get("root_cause", ""),
            affected=data.get("affected", []),
            subtasks=subs,
            complexity=data.get("complexity", "simple"),
            raw=text,
        )
