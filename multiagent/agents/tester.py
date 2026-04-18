from __future__ import annotations

from typing import Any

from .base import Agent, AgentResult
from ..tools.fs import apply_patch


class Tester(Agent):
    role = "tester"
    tier = "cheap"
    prompt_file = "tester.md"

    def write_tests(self, spec: str, context: dict[str, Any]) -> AgentResult:
        result = self.run(spec, context)
        ok = apply_patch(self.workdir, result.output)
        result.artifacts["patch_applied"] = ok
        result.ok = ok
        return result
