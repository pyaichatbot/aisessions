from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..subagents.loader import SubagentDef, load_subagent
from ..memory.wiki import WikiMemory
from .budget import BudgetTracker
from .cache import CacheHit, PromptCache
from .router import ModelResponse, ModelRouter
from .tool_registry import ToolRegistry


@dataclass
class RunResult:
    name: str
    ok: bool
    text: str
    iters: int
    cost_usd: float
    tokens_in: int
    tokens_out: int
    model: str
    transcript: list[dict[str, Any]] = field(default_factory=list)


class AgentRunner:
    """Runs an md-defined subagent through a tool-use loop.

    Flow:
      1. Load def from .md (cached).
      2. Pick model by tier (with escalation).
      3. Build system = [prompt (cached), wiki_context].
      4. Loop: call model, execute tool_use blocks, feed back results.
      5. Stop on end_turn, iter cap, or budget breach.
    """

    def __init__(
        self,
        subagents_root: Path,
        router: ModelRouter,
        budget: BudgetTracker,
        cache: PromptCache,
        memory: WikiMemory,
        tools: ToolRegistry,
    ) -> None:
        self.root = Path(subagents_root)
        self.router = router
        self.budget = budget
        self.cache = cache
        self.memory = memory
        self.tools = tools
        self._defs: dict[str, SubagentDef] = {}

    def _load(self, name: str) -> SubagentDef:
        if name not in self._defs:
            self._defs[name] = load_subagent(self.root / f"{name}.md")
        return self._defs[name]

    def run(
        self,
        name: str,
        user_prompt: str,
        context: dict[str, Any] | None = None,
        escalate: bool = False,
    ) -> RunResult:
        ctx = context or {}
        sa = self._load(name)
        model = self.router.pick(sa.model_tier, escalate=escalate)
        system = self._system_blocks(sa)
        tools_schema = self.tools.schema_for(sa.tools) if sa.tools else None
        user_msg = self._format_user(user_prompt, ctx, sa.output_format)

        messages: list[dict[str, Any]] = [{"role": "user", "content": user_msg}]
        transcript: list[dict[str, Any]] = []

        cached = self.cache.get(model, system, messages, tools_schema)
        if cached is not None and not cached.tool_calls:
            self.memory.record_turn(sa.name, user_prompt, cached.text)
            return RunResult(
                sa.name, True, cached.text, 0, 0.0,
                cached.tokens_in, cached.tokens_out, model, transcript,
            )

        total_cost = 0.0
        total_in = 0
        total_out = 0
        final_text = ""

        for i in range(sa.max_tool_iters + 1):
            resp: ModelResponse = self.router.call(
                model=model, system=system, messages=messages,
                tools=tools_schema, max_tokens=sa.max_tokens,
            )
            self.budget.charge(resp.cost_usd, resp.tokens_in, resp.tokens_out)
            total_cost += resp.cost_usd
            total_in += resp.tokens_in
            total_out += resp.tokens_out
            final_text = resp.text
            transcript.append({"iter": i, "text": resp.text,
                               "tool_use": resp.tool_use,
                               "stop": resp.stop_reason})

            if not resp.tool_use or resp.stop_reason == "end_turn":
                break

            messages.append({"role": "assistant", "content": resp.raw_content or []})
            tool_results = []
            for call in resp.tool_use:
                out = self.tools.call(call["name"], call.get("input") or {})
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call["id"],
                    "content": out,
                })
            messages.append({"role": "user", "content": tool_results})

            if i + 1 >= sa.max_tool_iters:
                # Force final turn without tools.
                resp_final: ModelResponse = self.router.call(
                    model=model, system=system, messages=messages,
                    tools=None, max_tokens=sa.max_tokens,
                )
                self.budget.charge(resp_final.cost_usd, resp_final.tokens_in,
                                   resp_final.tokens_out)
                total_cost += resp_final.cost_usd
                total_in += resp_final.tokens_in
                total_out += resp_final.tokens_out
                final_text = resp_final.text
                transcript.append({"iter": i + 1, "text": resp_final.text,
                                   "forced_finalize": True})
                break

        self.cache.put(model, system, messages, CacheHit(final_text, total_in, total_out))
        self.memory.record_turn(sa.name, user_prompt, final_text)
        return RunResult(
            name=sa.name, ok=bool(final_text), text=final_text,
            iters=len(transcript), cost_usd=total_cost,
            tokens_in=total_in, tokens_out=total_out, model=model,
            transcript=transcript,
        )

    # --- helpers -------------------------------------------------------------

    def _system_blocks(self, sa: SubagentDef) -> list[dict[str, Any]]:
        blocks = [{
            "type": "text",
            "text": sa.system_prompt,
            "cache_control": {"type": "ephemeral"},
        }]
        wiki = self.memory.context_for(sa.name)
        if wiki:
            blocks.append({"type": "text", "text": f"# Memory\n{wiki}"})
        return blocks

    @staticmethod
    def _format_user(task: str, ctx: dict[str, Any], output_format: str) -> str:
        lines = [f"TASK: {task}"]
        if output_format == "json":
            lines.append("Return JSON only. No prose outside the code fence.")
        for k, v in ctx.items():
            lines.append(f"\n## {k.upper()}\n{v}")
        return "\n".join(lines)
