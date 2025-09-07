from __future__ import annotations

from typing import Any, Dict, Optional, Tuple, Type
import asyncio
import logging

from pydantic import BaseModel

from ..config import settings
from ..persistence.db import insert_ai_call_cost
# from .base import llm_call_and_log  # 삭제됨


async def stream_text(
    *,
    model_name: str,
    prompt: str,
    on_text: Optional[Any] = None,
    on_thought: Optional[Any] = None,
    text_format: Optional[Type[BaseModel]] = None,
    request_id: Optional[str] = None,
    session_id: Optional[str] = None,
    purpose: Optional[str] = None,
) -> Tuple[str, str, Dict[str, Any]]:
    """공통 LLM 스트리밍 헬퍼: gpt-5 계열 지원. 실패시 에러.

    반환값: (final_text, provider_name, usage_dict)
    usage_dict keys: input_tokens, output_tokens, total_tokens
    """
    model_clean = (model_name or "").strip()

    # Gemini 경로 제거됨

    if model_clean in ("gpt-5", "gpt-5-high"):
        try:
            from .providers.openai import OpenAIProvider
            loop = asyncio.get_running_loop()
            provider = OpenAIProvider(api_key=settings.llm_openai_api_key)
            def _run_stream():
                final_text_local: str = ""
                for item in provider.stream_generate(
                    model=("gpt-5-high" if model_clean == "gpt-5-high" else model_name), 
                    prompt=prompt, 
                    include_thoughts=True,
                    text_format=text_format
                ):
                    text = item.get("text") or ""
                    kind = item.get("kind", "answer")
                    if kind == "thought" and on_thought:
                        try:
                            loop.call_soon_threadsafe(on_thought, text)
                        except Exception:
                            pass
                    elif kind in ("answer", "structured_output") and text and on_text:
                        try:
                            loop.call_soon_threadsafe(on_text, text)
                        except Exception:
                            pass
                    if kind in ("answer", "structured_output"):
                        final_text_local += text
                usage_local = provider.last_usage() or {}
                return final_text_local, usage_local
            final_text, usage = await asyncio.to_thread(_run_stream)
            try:
                if request_id and purpose:
                    insert_ai_call_cost(
                        request_id=str(request_id),
                        session_id=str(session_id) if session_id is not None else None,
                        provider="openai",
                        model=("gpt-5-high" if model_clean == "gpt-5-high" else model_name),
                        purpose=str(purpose),
                        input_tokens=(usage or {}).get("input_tokens"),
                        output_tokens=(usage or {}).get("output_tokens"),
                        total_tokens=(usage or {}).get("total_tokens"),
                        cache_input_tokens=(usage or {}).get("cache_input_tokens"),
                    )
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.warning("Failed to insert ai_call_costs (openai): %s", e)
            return final_text, "openai", usage
        except Exception as e:
            import logging
            logger = logging.getLogger(__name__)
            logger.warning("OpenAI streaming failed: %s", e)

    # 스트리밍 실패시 에러 발생 (단건 호출 제거됨)
    raise RuntimeError(f"Streaming failed for model: {model_name}")


