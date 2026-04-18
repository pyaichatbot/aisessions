from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from .cache import PromptCache, CacheHit


# USD per 1M tokens. Keep in sync with pricing docs.
PRICE_TABLE: dict[str, tuple[float, float]] = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "claude-sonnet-4-6": (3.0, 15.0),
    "claude-opus-4-7": (15.0, 75.0),
}


@dataclass
class ModelResponse:
    text: str
    tokens_in: int
    tokens_out: int
    cost_usd: float
    model: str
    cached: bool = False


class ModelRouter:
    """Tier-based model selection with escalation."""

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
        cache: PromptCache,
        max_tokens: int = 4096,
    ) -> ModelResponse:
        hit = cache.get(model, system, messages)
        if hit is not None:
            return ModelResponse(hit.text, hit.tokens_in, hit.tokens_out, 0.0, model, True)
        if self._dry_run:
            text = f"[dry-run {model}] ok"
            resp = ModelResponse(text, 0, 0, 0.0, model, False)
            cache.put(model, system, messages, CacheHit(text, 0, 0))
            return resp
        sdk_resp = self.client.messages.create(
            model=model,
            system=system,
            messages=messages,
            max_tokens=max_tokens,
        )
        text = "".join(
            b.text for b in sdk_resp.content if getattr(b, "type", "") == "text"
        )
        usage = getattr(sdk_resp, "usage", None)
        tin = getattr(usage, "input_tokens", 0) or 0
        tout = getattr(usage, "output_tokens", 0) or 0
        price_in, price_out = PRICE_TABLE.get(model, (0.0, 0.0))
        cost = (tin * price_in + tout * price_out) / 1_000_000
        resp = ModelResponse(text, tin, tout, cost, model, False)
        cache.put(model, system, messages, CacheHit(text, tin, tout))
        return resp
