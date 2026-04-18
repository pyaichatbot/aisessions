from __future__ import annotations

from typing import Any

from .base import Agent, AgentResult
from ..tools.fs import apply_patch


class Coder(Agent):
    role = "coder"
    tier = "cheap"
    prompt_file = "coder.md"

    def code(self, subtask_desc: str, context: dict[str, Any]) -> AgentResult:
        result = self.run(subtask_desc, context)
        diff_applied = apply_patch(self.workdir, result.output)
        result.artifacts["patch_applied"] = diff_applied
        result.ok = diff_applied
        return result
