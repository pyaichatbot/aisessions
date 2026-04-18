from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .base import Agent, AgentResult


@dataclass
class ReviewVerdict:
    approved: bool
    issues: list[str]
    severity: str  # none | low | medium | high
    raw: str


class Reviewer(Agent):
    role = "reviewer"
    tier = "mid"
    prompt_file = "reviewer.md"

    def review(self, diff: str, context: dict[str, Any]) -> ReviewVerdict:
        ctx = dict(context)
        ctx["diff"] = diff
        ctx["output_format"] = "json"
        result = self.run("Review this diff.", ctx)
        return self._parse(result.output)

    @staticmethod
    def _parse(text: str) -> ReviewVerdict:
        block = re.search(r"```json\s*(.*?)```", text, re.DOTALL)
        payload = block.group(1) if block else text
        try:
            data = json.loads(payload)
            return ReviewVerdict(
                approved=bool(data.get("approved", False)),
                issues=list(data.get("issues", [])),
                severity=data.get("severity", "low"),
                raw=text,
            )
        except json.JSONDecodeError:
            approved = "approved" in text.lower() and "not approved" not in text.lower()
            return ReviewVerdict(approved=approved, issues=[text], severity="low", raw=text)
