from __future__ import annotations

import json
from typing import Dict, Generator, Optional, Any, Type, Union

import asyncio
from openai import OpenAI
from pydantic import BaseModel

from ..base import LLMProvider, LLMCallResult, estimate_cost_usd


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, base_url: Optional[str] = None) -> None:
        super().__init__(name="openai")
        self.api_key = api_key
        self.base_url = base_url  # SDK가 기본 엔드포인트를 사용하므로 저장만
        if base_url:
            self._client = OpenAI(api_key=api_key, base_url=base_url) if api_key else OpenAI(base_url=base_url)
        else:
            self._client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self._last_usage: Optional[Dict[str, int]] = None

    @staticmethod
    def _extract_usage(obj: object) -> Dict[str, Optional[int]]:
        usage = None
        if isinstance(obj, dict):
            usage = obj.get("usage") or (obj.get("response") or {}).get("usage")

        def pick(u: object, *names: str) -> Optional[int]:
            if u is None:
                return None
            for n in names:
                if isinstance(u, dict) and n in u:
                    try:
                        return int(u[n]) if u[n] is not None else None
                    except Exception:  # noqa: BLE001
                        return None
            return None

        itoks = pick(usage, "input_tokens", "prompt_tokens")
        otoks = pick(usage, "output_tokens", "completion_tokens")
        ttoks = pick(usage, "total_tokens")
        # 캐시 관련 토큰 (Responses API 기준: cache_creation_input_tokens, cache_read_input_tokens)
        cache_create = pick(usage, "cache_creation_input_tokens", "cache_creation_tokens")
        cache_read = pick(usage, "cache_read_input_tokens", "cache_read_tokens", "prompt_cache_hit_tokens", "input_tokens_cached")
        cache_total: Optional[int] = None
        if cache_create is not None or cache_read is not None:
            cache_total = (cache_create or 0) + (cache_read or 0)
        if ttoks is None and (itoks is not None or otoks is not None):
            ttoks = (itoks or 0) + (otoks or 0)
        return {"input_tokens": itoks, "output_tokens": otoks, "total_tokens": ttoks, "cache_input_tokens": cache_total}

 
    def stream_generate(
        self, 
        *, 
        model: str, 
        prompt: str, 
        include_thoughts: bool = True,
        text_format: Optional[Type[BaseModel]] = None
    ) -> Generator[Dict[str, Any], None, None]:
        # Responses API 스트리밍 (SSE): 이벤트를 파싱해 토큰 단위로 thought/answer 방출
        # SDK 스트리밍: responses.stream 사용
        self._last_usage = None

        stream_params = {
            "model": "gpt-5",
            "input": prompt,
            "instructions": "You are a helpful assistant.",
            # effort 설정: gpt-5 -> low, gpt-5-high -> high
            "reasoning": ({"effort": ("high" if model == "gpt-5-high" else "low"), "summary": "auto"}
                           if include_thoughts else None),
        }
        
        # text_format이 제공된 경우 구조화된 출력 사용
        if text_format:
            # Pydantic 모델을 OpenAI JSON Schema 형식으로 변환
            schema = text_format.model_json_schema()
            
            # OpenAI의 strict mode를 위해 additionalProperties: false 추가
            def add_additional_properties_false(obj):
                if isinstance(obj, dict):
                    if obj.get("type") == "object":
                        obj["additionalProperties"] = False
                    for value in obj.values():
                        add_additional_properties_false(value)
                elif isinstance(obj, list):
                    for item in obj:
                        add_additional_properties_false(item)
            
            add_additional_properties_false(schema)
            
            # OpenAI strict 스키마 요구사항 보정:
            # - required가 properties의 모든 키를 포함해야 함
            # - optional이었던 필드는 null 허용으로 변경해 존재를 강제하되 값은 null 가능
            def make_nullable(s: Dict[str, Any]) -> None:
                if not isinstance(s, dict):
                    return
                t = s.get("type")
                if isinstance(t, list):
                    if "null" not in t:
                        s["type"] = t + ["null"]
                elif isinstance(t, str):
                    s["type"] = [t, "null"]
                elif isinstance(s.get("anyOf"), list):
                    # anyOf 내부에 null 타입 추가
                    if not any(isinstance(xx, dict) and xx.get("type") == "null" for xx in s["anyOf"]):
                        s["anyOf"].append({"type": "null"})
                elif isinstance(s.get("oneOf"), list):
                    if not any(isinstance(xx, dict) and xx.get("type") == "null" for xx in s["oneOf"]):
                        s["oneOf"].append({"type": "null"})

            def ensure_required_and_nullable(obj: Any) -> None:
                if isinstance(obj, dict):
                    if obj.get("type") == "object" and isinstance(obj.get("properties"), dict):
                        props: Dict[str, Any] = obj["properties"]
                        req_list = obj.get("required")
                        if not isinstance(req_list, list):
                            req_list = []
                        req_set = set([x for x in req_list if isinstance(x, str)])
                        # optional 필드 식별 (required에 없던 키들)
                        optional_keys = [k for k in props.keys() if k not in req_set]
                        # optional 필드는 null 허용으로 변경
                        for k in optional_keys:
                            make_nullable(props.get(k) or {})
                        # OpenAI strict: required는 모든 키를 포함
                        obj["required"] = list(props.keys())
                    # 하위 스키마 재귀 처리
                    for v in obj.values():
                        ensure_required_and_nullable(v)
                elif isinstance(obj, list):
                    for it in obj:
                        ensure_required_and_nullable(it)

            ensure_required_and_nullable(schema)
            
            stream_params["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": text_format.__name__.lower(),
                    "strict": True,
                    "schema": schema
                }
            }
        else:
            stream_params["text"] = {"verbosity": "medium"}
        
        with self._client.responses.stream(**stream_params) as stream:
            for event in stream:
                # 구조화된 출력 거부 델타 (refusal)
                if getattr(event, "type", "") == "response.refusal.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield {"kind": "refusal", "text": str(delta)}
                    continue
                
                # 구조화된 출력 텍스트 델타
                if getattr(event, "type", "") == "response.output_text.delta":
                    delta = getattr(event, "delta", "") or ""
                    if delta:
                        yield {"kind": "structured_output", "text": str(delta)}
                    continue
                
                # 일반 출력 텍스트 델타 (구조화되지 않은 출력)
                if getattr(event, "type", "") in (
                    "output_text.delta",
                ):
                    delta = getattr(event, "delta", None) or getattr(event, "text", "") or ""
                    if delta:
                        yield {"kind": "answer", "text": str(delta)}
                    continue
                # 추론(생각) 델타
                if include_thoughts and getattr(event, "type", "") in (
                    "response.thinking.delta",
                    "response.thought.delta",
                    "thinking.delta",
                    "response.reasoning_summary_text.delta",
                ):
                    delta = getattr(event, "delta", None) or getattr(event, "text", "") or ""
                    if delta:
                        yield {"kind": "thought", "text": str(delta)}
                    continue
                # 포괄적: 타입명에 reasoning 이 포함되면 thought 로 분류 (신규 타입 대비)
                if include_thoughts and "reasoning" in str(getattr(event, "type", "")):
                    delta = getattr(event, "delta", None) or getattr(event, "text", "") or ""
                    if delta:
                        yield {"kind": "thought", "text": str(delta)}
                    continue
                # 출력 텍스트 완료 신호는 무시 (completed 에서 사용량을 집계)
                if getattr(event, "type", "") in ("response.output_text.done",):
                    continue
                # 에러 이벤트 처리
                if getattr(event, "type", "") == "response.error":
                    err_detail = getattr(event, "error", None)
                    yield {"kind": "error", "error": str(err_detail)}
                    continue
                # 완료 이벤트에서 usage 집계 및 최종 응답 처리
                if getattr(event, "type", "") in ("response.completed", "response.completed.successfully", "response.completed.partially"):
                    try:
                        data = event.response.model_dump() if hasattr(event, "response") and hasattr(event.response, "model_dump") else None
                        if not data and hasattr(event, "response") and hasattr(event.response, "json"):
                            data = json.loads(event.response.json())
                        if isinstance(data, dict):
                            usage = data.get("usage") or {}
                            itoks = usage.get("input_tokens") or usage.get("prompt_tokens")
                            otoks = usage.get("output_tokens") or usage.get("completion_tokens")
                            ttoks = usage.get("total_tokens")
                            # 캐시 집계: 생성/읽기 토큰 합산
                            cache_create = usage.get("cache_creation_input_tokens") or usage.get("cache_creation_tokens")
                            cache_read = usage.get("cache_read_input_tokens") or usage.get("cache_read_tokens") or usage.get("prompt_cache_hit_tokens") or usage.get("input_tokens_cached")
                            cache_total = None
                            try:
                                if cache_create is not None or cache_read is not None:
                                    cache_total = int(cache_create or 0) + int(cache_read or 0)
                            except Exception:
                                cache_total = None
                            self._last_usage = {
                                "input_tokens": int(itoks or 0),
                                "output_tokens": int(otoks or 0),
                                "total_tokens": int(ttoks or ((itoks or 0) + (otoks or 0))),
                                "cache_input_tokens": cache_total if cache_total is not None else None,
                            }
                            
                            # 구조화된 출력의 경우 최종 결과 반환
                            if text_format and hasattr(event, "response"):
                                final_response = None
                                if hasattr(event.response, "output"):
                                    final_response = event.response.output
                                elif "output" in data:
                                    final_response = data["output"]
                                
                                if final_response:
                                    yield {"kind": "final_response", "response": final_response}
                    except Exception:  # noqa: BLE001
                        pass

    def last_usage(self) -> Optional[Dict[str, int]]:
        return self._last_usage


