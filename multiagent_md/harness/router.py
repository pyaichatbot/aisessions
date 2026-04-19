from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any


PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


@dataclass
class ModelResponse:
    text: str
    tool_use: list[dict[str, Any]]
    stop_reason: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    cached: bool = False
    raw_content: list[dict[str, Any]] | None = None


class ModelRouter:
    """Tier-based model selection with escalation + tool-use support."""

    def __init__(self, cfg: dict[str, str], client: Any | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._dry_run = os.environ.get("MULTIAGENT_DRY_RUN") == "1"

    @property
    def client(self) -> Any:
        if self._client is not None:
            return self._client
        if self._dry_run:
            return None
        from anthropic import Anthropic  # type: ignore
        self._client = Anthropic()
        return self._client

    def pick(self, tier: str, escalate: bool = False) -> str:
        if escalate:
            tier = {"cheap": "mid", "mid": "strong", "strong": "strong"}[tier]
        return self.cfg.get(tier, self.cfg["cheap"])

    def call(
        self,
        model: str,
        system: list[dict[str, Any]] | str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        if self._dry_run:
            text = f"[dry-run {model}] ok"
            return ModelResponse(
                text=text, tool_use=[], stop_reason="end_turn",
                tokens_in=0, tokens_out=0, cost_usd=0.0, model=model,
                raw_content=[{"type": "text", "text": text}],
            )
        kwargs: dict[str, Any] = {
            "model": model,
            "system": system,
            "messages": messages,
            "max_tokens": max_tokens,
        }
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)
        text_parts: list[str] = []
        tool_uses: list[dict[str, Any]] = []
        content_raw: list[dict[str, Any]] = []
        for b in resp.content:
            btype = getattr(b, "type", "")
            if btype == "text":
                text_parts.append(b.text)
                content_raw.append({"type": "text", "text": b.text})
            elif btype == "tool_use":
                tool_uses.append({
                    "id": b.id, "name": b.name, "input": b.input,
                })
                content_raw.append({
                    "type": "tool_use", "id": b.id,
                    "name": b.name, "input": b.input,
                })
        usage = getattr(resp, "usage", None)
        tin = getattr(usage, "input_tokens", 0) or 0
        tout = getattr(usage, "output_tokens", 0) or 0
        price_in, price_out = PRICE_TABLE.get(model, (0.0, 0.0))
        cost = (tin * price_in + tout * price_out) / 1_000_000
        return ModelResponse(
            text="".join(text_parts),
            tool_use=tool_uses,
            stop_reason=getattr(resp, "stop_reason", ""),
            tokens_in=tin, tokens_out=tout, cost_usd=cost, model=model,
            raw_content=content_raw,
        )
