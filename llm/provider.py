from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Dict, Any

from pooolify.db.client import DB


@dataclass
class LLMResult:
    text: str
    usage: Dict[str, int]


class LLMProvider:
    name: str = "base"

    async def call(self, *, prompt: str, model: str, purpose: str, request_id: str, session_id: Optional[str] = None) -> LLMResult:
        raise NotImplementedError


class DummyLLM(LLMProvider):
    name: str = "dummy"

    async def call(self, *, prompt: str, model: str, purpose: str, request_id: str, session_id: Optional[str] = None) -> LLMResult:
        # Trivial echo with pretend token counts and small cost
        started = time.time()
        text = f"[dummy:{model}:{purpose}] {prompt[:200]}"
        usage = {"input_tokens": len(prompt) // 4, "output_tokens": len(text) // 4, "total_tokens": (len(prompt) + len(text)) // 4}
        cost_amount = round(usage["total_tokens"] * 0.000001, 6)
        latency_ms = int((time.time() - started) * 1000)
        await DB.log_ai_cost(
            request_id=request_id,
            session_id=session_id,
            provider=self.name,
            model=model,
            purpose=purpose,
            input_tokens=usage["input_tokens"],
            output_tokens=usage["output_tokens"],
            total_tokens=usage["total_tokens"],
            cost_amount=cost_amount,
        )
        return LLMResult(text=text, usage=usage)

