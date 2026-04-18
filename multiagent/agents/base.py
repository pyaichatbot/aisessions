from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from ..harness.router import ModelRouter
from ..harness.budget import BudgetTracker
from ..harness.cache import PromptCache
from ..memory.wiki import WikiMemory


@dataclass
class AgentResult:
    role: str
    ok: bool
    output: str
    artifacts: dict[str, Any] = field(default_factory=dict)
    cost_usd: float = 0.0
    tokens_in: int = 0
    tokens_out: int = 0
    model: str = ""


class Agent:
    """Base agent. Subclasses override role, tier, prompt_file."""

    role: str = "base"
    tier: str = "cheap"
    prompt_file: str = ""

    def __init__(
        self,
        router: ModelRouter,
        budget: BudgetTracker,
        cache: PromptCache,
        memory: WikiMemory,
        workdir: Path,
    ) -> None:
        self.router = router
        self.budget = budget
        self.cache = cache
        self.memory = memory
        self.workdir = workdir

    def _load_prompt(self) -> str:
        root = Path(__file__).resolve().parent.parent / "prompts"
        path = root / self.prompt_file
        return path.read_text(encoding="utf-8")

    def _system(self) -> list[dict[str, Any]]:
        base = self._load_prompt()
        wiki = self.memory.context_for(self.role)
        blocks = [
            {"type": "text", "text": base, "cache_control": {"type": "ephemeral"}},
        ]
        if wiki:
            blocks.append({"type": "text", "text": wiki})
        return blocks

    def run(self, task: str, context: Optional[dict[str, Any]] = None) -> AgentResult:
        context = context or {}
        model = self.router.pick(self.tier, escalate=context.get("escalate", False))
        user = self._format_user(task, context)
        resp = self.router.call(
            model=model,
            system=self._system(),
            messages=[{"role": "user", "content": user}],
            cache=self.cache,
        )
        self.budget.charge(resp.cost_usd, resp.tokens_in, resp.tokens_out)
        result = AgentResult(
            role=self.role,
            ok=True,
            output=resp.text,
            cost_usd=resp.cost_usd,
            tokens_in=resp.tokens_in,
            tokens_out=resp.tokens_out,
            model=model,
        )
        self.memory.record_turn(self.role, task, resp.text)
        return result

    def _format_user(self, task: str, context: dict[str, Any]) -> str:
        lines = [f"TASK: {task}"]
        for k, v in context.items():
            if k == "escalate":
                continue
            lines.append(f"\n## {k.upper()}\n{v}")
        return "\n".join(lines)
