from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

from ..config import settings
from ..persistence.db import insert_ai_call_cost


logger = logging.getLogger(__name__)


@dataclass
class LLMCallResult:
    provider: str
    model: str
    purpose: str
    text: str
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    total_tokens: Optional[int]
    cost_usd: Optional[float]
    cache_input_tokens: Optional[int] = None


class LLMProvider:
    name: str

    def __init__(self, name: str) -> None:
        self.name = name

    async def call(self, *, model: str, prompt: str, purpose: str, request_id: str, session_id: Optional[str]) -> LLMCallResult:  # pragma: no cover - interface
        raise NotImplementedError

_provider_override: Optional[LLMProvider] = None

def set_provider_override(provider: Optional[LLMProvider]) -> None:
    global _provider_override
    _provider_override = provider


class StubProvider(LLMProvider):
    def __init__(self) -> None:  # pragma: no cover - no longer used
        super().__init__(name="stub")

    async def call(self, *, model: str, prompt: str, purpose: str, request_id: str, session_id: Optional[str]) -> LLMCallResult:  # pragma: no cover - no longer used
        raise RuntimeError("StubProvider is disabled: only 'gpt-5' and 'gpt-5-high' are supported")


    # GoogleProvider and OpenAIProvider are defined in pooolify.llm.providers


def estimate_cost_usd(provider: str, model: str, input_tokens: Optional[int], output_tokens: Optional[int]) -> Optional[float]:
    # Placeholder cost table; adjust per provider/model later
    if input_tokens is None and output_tokens is None:
        return None
    it = input_tokens or 0
    ot = output_tokens or 0
    if provider == "openai":
        # Generic default: $0.5 / 1M input, $1.5 / 1M output
        return (it * 0.5 + ot * 1.5) / 1_000_000.0
    return (it + ot) / 1_000_000.0


def get_provider(requested_model: str) -> LLMProvider:
    if _provider_override is not None:
        return _provider_override
    model = (requested_model or "").strip()
    if model in ("gpt-5", "gpt-5-high"):
        if not settings.llm_openai_api_key:
            raise RuntimeError("LLM_OPENAI_API_KEY is required for gpt-5")
        from .providers.openai import OpenAIProvider
        return OpenAIProvider(api_key=settings.llm_openai_api_key)
    raise RuntimeError(f"Unsupported model: {requested_model}. Allowed: 'gpt-5', 'gpt-5-high'")


