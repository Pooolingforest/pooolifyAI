from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional
import inspect
import uuid

from ..agents.runtime import AgentDefinition
from ..llm.streaming import stream_text as llm_stream_text
from ..tools.base import Tool
from .conversation import ConversationAccumulator
from .schemas import AgentPlanCall, AgentPlanResult, AgentFinalResult, AgentImproveResult


logger = logging.getLogger(__name__)


def _sse_event(event: str, data: Dict[str, Any]) -> bytes:
    """SSE 이벤트 생성 헬퍼"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n".encode("utf-8")


class ToolExecutor:
    """도구 실행을 담당하는 클래스"""
    
    def __init__(self, agent_manager):
        self.agent_manager = agent_manager
    
    async def run_agent_once(
        self,
        *,
        agent: AgentDefinition,
        session_id: str,
        request_id: str,
        conversation: ConversationAccumulator,
        instruction: str,
        context: Optional[str] = None,
        emit_collector: Optional[Any] = None,
    ) -> str:
        """
        한 에이전트를 한 번 실행:
        1) 에이전트가 사용할 도구 호출 계획(JSON)을 생성
        2) 계획에 따라 도구를 호출하고 이벤트/로그 기록
        3) 도구 결과를 바탕으로 에이전트 최종 요약 텍스트 반환
        에이전트에 도구가 없으면 직접 답변 생성
        """
        # 에이전트 메시지 시작 (개별 버블 생성)
        conversation.start_ai_message(agent=agent.name)

        # 구조화된 출력 모델은 schemas.py에서 import
        # If no tools, let LLM answer directly under agent's system (구조화 출력 + 스트리밍)
        if not agent.tools:
            prompt = (
                self.agent_manager._compose_system_prompt(agent)
                + ("\n\nAgent instruction: " + instruction if instruction else "")
                + ("\n\nPrevious agent context:\n" + context if context else "")
                + "\n\nRequirement: Answer concisely and accurately in English. Return JSON in the form {\"text\": string}."
            )

            def _agent_direct_text(delta: str) -> None:
                conversation.accumulate_answer(delta)

            def _agent_direct_thought(delta: str) -> None:
                conversation.accumulate_thought(delta)

            final_text, _prov, _usage = await llm_stream_text(
                model_name=agent.model,
                prompt=prompt,
                on_text=_agent_direct_text,
                on_thought=_agent_direct_thought,
                text_format=AgentFinalResult,
                request_id=request_id,
                session_id=session_id,
                purpose=f"agent.{agent.name}.direct_answer",
            )
            try:
                j = json.loads((final_text or "").strip())
                return str((j or {}).get("text") or "")
            except Exception:
                return (final_text or "")

        # Build tool catalog description (+ dynamic args schema from tool.run signature)
        def _describe_tool_args(tool: Tool) -> str:
            try:
                sig = inspect.signature(tool.run)
            except Exception:
                return f"- name: {tool.name}\n  description: {tool.description}"

            # 특정 툴 가이드라인(usage_notes) 추가
            usage_notes: Optional[str] = None
            if getattr(tool, "name", "") == "internal_api":
                usage_notes = (
                    "Use 'path' (relative or absolute). Do NOT pass 'url'. "
                    "Base URL is assigned automatically by the tool (API_SERVER_BASE), "
                    "so 'base_url' is usually unnecessary unless you need to override. "
                    "Authorization header is also attached automatically."
                )

            arg_blocks: List[str] = []
            for param_name, param in sig.parameters.items():
                if param_name == "self":
                    continue
                if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
                    # 가변 인자는 카탈로그에 노출하지 않음
                    continue

                # 타입 문자열 생성
                if param.annotation is inspect._empty:
                    type_str = "Any"
                else:
                    try:
                        type_str = getattr(param.annotation, "__name__", str(param.annotation))
                    except Exception:
                        type_str = "Any"

                # 필수/기본값 판단
                is_required = param.default is inspect._empty
                block = [
                    f"    - name: {param_name}",
                    f"      type: {type_str}",
                    f"      required: {str(is_required).lower()}",
                ]
                if not is_required:
                    try:
                        default_repr = repr(param.default)
                    except Exception:
                        default_repr = str(param.default)
                    block.append(f"      default: {default_repr}")
                arg_blocks.append("\n".join(block))

            if arg_blocks:
                extra = (f"  usage_notes: {usage_notes}\n" if usage_notes else "")
                return (
                    f"- name: {tool.name}\n"
                    f"  description: {tool.description}\n"
                    + extra +
                    f"  args:\n" + "\n".join(arg_blocks)
                )
            else:
                extra = (f"\n  usage_notes: {usage_notes}" if usage_notes else "")
                return f"- name: {tool.name}\n  description: {tool.description}" + extra

        tool_descs: List[str] = [
            _describe_tool_args(t) for t in (agent.tools or [])
        ]
        tool_catalog = "\n".join(tool_descs)

        # 툴 이름 정규화 및 해석 유틸 (모델이 툴명을 약간 변경해 출력하는 경우 대비)
        def _normalize_tool_name(name: Any) -> str:
            try:
                s = str(name)
            except Exception:
                return ""
            return "".join(ch.lower() for ch in s if ch.isalnum())

        def _resolve_tool_by_name(tools: List[Tool], requested_name: Any) -> Optional[Tool]:
            if not tools:
                return None
            requested_norm = _normalize_tool_name(requested_name)
            # 1) 정규화된 이름 기준 매칭
            for t in tools:
                if _normalize_tool_name(getattr(t, "name", "")) == requested_norm:
                    return t
            # 2) 원본 이름 완전 일치
            for t in tools:
                if getattr(t, "name", None) == requested_name:
                    return t
            return None

        def _parse_args_to_dict(maybe_args: Any) -> Dict[str, Any]:
            # 이미 dict로 온 경우
            if isinstance(maybe_args, dict):
                return maybe_args
            if not isinstance(maybe_args, str):
                return {}
            raw = maybe_args.strip()
            if not raw:
                return {}
            # 코드펜스/설명 섞인 경우 JSON 본문만 추출
            first = raw.find("{")
            last = raw.rfind("}")
            candidate = raw[first:last + 1] if first != -1 and last != -1 and last >= first else raw
            try:
                parsed = json.loads(candidate)
                return parsed if isinstance(parsed, dict) else {}
            except Exception:
                return {}

        # 루프 실행을 위한 상태
        local_instruction = instruction or ""
        improvement_notes = ""
        last_final_answer_raw: str = ""
        max_loops = max(1, int(getattr(agent, "max_loops", 3) or 3))

        for loop_index in range(max_loops):
            # 1) 계획 생성
            plan_prompt = (
                self.agent_manager._compose_system_prompt(agent)
                + "\n\nYou are an expert performing the role above.\n"
                + ("Agent instruction: " + local_instruction + "\n" if local_instruction else "")
                + ("Previous agent context:\n" + context + "\n" if context else "")
                + ("Improvement notes:\n" + improvement_notes + "\n" if improvement_notes else "")
                + "Using the tool catalog below, write as many tool-call plans as needed to achieve the user's goal. You may call the same tool multiple times with clearly different purposes/inputs.\n"
                + "Output must be a JSON object matching the schema {isRunTool:boolean, calls:[{tool,reason}], text:string}.\n"
                + "Important: Do NOT generate args at this step. Arguments for each tool will be generated during the execution phase.\n"
                + "Tool catalog:\n"
                + tool_catalog
            )
            
            def _plan_text_cb(delta: str) -> None:
                conversation.accumulate_answer(delta)
            
            def _plan_thought_cb(delta: str) -> None:
                conversation.accumulate_thought(delta)
            
            plan_text, _plan_provider, _plan_usage = await llm_stream_text(
                model_name=agent.model,
                prompt=plan_prompt,
                on_text=_plan_text_cb,
                on_thought=_plan_thought_cb,
                text_format=AgentPlanResult,
                request_id=request_id,
                session_id=session_id,
                purpose=f"agent.{agent.name}.plan",
            )
            
            plan_result: Optional[AgentPlanResult] = None
            plan_parsed_ok: bool = False
            try:
                raw_plan = (plan_text or "").strip()
                # 모델이 코드펜스/설명 포함 출력 시 JSON 본문만 안전하게 추출
                first_brace = raw_plan.find("{")
                last_brace = raw_plan.rfind("}")
                json_blob = (
                    raw_plan[first_brace:last_brace + 1]
                    if first_brace != -1 and last_brace != -1 and last_brace >= first_brace
                    else raw_plan
                )
                plan_data = json.loads(json_blob)
                if "calls" not in plan_data or plan_data.get("calls") is None:
                    plan_data["calls"] = []
                for call in plan_data.get("calls", []):
                    if isinstance(call, dict):
                        if "args" not in call:
                            call["args"] = "{}"
                        else:
                            # args가 dict/list로 오면 JSON 문자열로 표준화
                            if isinstance(call.get("args"), (dict, list)):
                                try:
                                    call["args"] = json.dumps(call.get("args"), ensure_ascii=False)
                                except Exception:
                                    call["args"] = "{}"
                            elif not isinstance(call.get("args"), str):
                                call["args"] = "{}"
                if "text" not in plan_data or plan_data.get("text") is None:
                    plan_data["text"] = ""
                if "isRunTool" not in plan_data:
                    plan_data["isRunTool"] = bool(plan_data.get("calls"))
                plan_result = AgentPlanResult(**plan_data)
                # calls가 존재하는데 isRunTool이 false로 온 경우 보정
                try:
                    if plan_result.calls and not plan_result.isRunTool:
                        plan_result.isRunTool = True
                except Exception:
                    pass
                plan_parsed_ok = True
            except Exception:
                plan_result = AgentPlanResult(text="계획 파싱 실패", isRunTool=False, calls=[])

            # 2) 도구 실행
            outputs: List[str] = []
            last_tool_outputs: Dict[str, Any] = {}
            tool_failures: List[str] = []
            if plan_result.isRunTool:
                for idx, call in enumerate((plan_result.calls or [])):
                    tool_name = getattr(call, "tool", None)
                    call_reason = getattr(call, "reason", "")
                    if not isinstance(tool_name, str):
                        continue
                    # 도구 해상 후 스펙 설명 생성
                    tool = _resolve_tool_by_name((agent.tools or []), tool_name)
                    if tool is None:
                        tool_failures.append(f"missing_tool:{tool_name}")
                        continue
                    # 실행 시점에 LLM으로 args 생성
                    # 1) 도구 시그니처/usage_notes
                    current_tool_desc = _describe_tool_args(tool)
                    # 2) 실행 컨텍스트 구성
                    exec_context_parts: List[str] = []
                    if local_instruction:
                        exec_context_parts.append("Agent instruction: " + local_instruction)
                    if context:
                        exec_context_parts.append("Previous agent context:\n" + context)
                    if improvement_notes:
                        exec_context_parts.append("Improvement notes:\n" + improvement_notes)
                    if call_reason:
                        exec_context_parts.append("Purpose of tool use (reason): " + call_reason)
                    exec_context = "\n\n".join(exec_context_parts)

                    # 최근 도구 결과를 컨텍스트에 포함(가능 시)
                    recent_context = ""
                    try:
                        if last_tool_outputs:
                            recent_context = "Recent tool results summary:\n" + json.dumps(last_tool_outputs, ensure_ascii=False) + "\n\n"
                    except Exception:
                        recent_context = ""

                    arggen_prompt = (
                        self.agent_manager._compose_system_prompt(agent)
                        + "\n\nGenerate only a valid args JSON to call the tool below right now. No explanations or code fences. Output exactly one JSON object.\n"
                        + "Rules:\n"
                        + "- Include only keys that exist in the tool signature. No unnecessary keys.\n"
                        + "- internal_api: 'path' must be a path from the spec; never an absolute URL. Do not include base_url/headers.\n"
                        + "- Times/dates must be KST literals (e.g., YYYY-MM-DD). No placeholders.\n"
                        + "- Set method to the correct HTTP method when needed.\n\n"
                        + "Tool spec:\n" + current_tool_desc + "\n\n"
                        + recent_context
                        + (exec_context + "\n\n" if exec_context else "")
                        + "Output: one args JSON object"
                    )

                    generated_args_text, _prov_gen, _usage_gen = await llm_stream_text(
                        model_name=agent.model,
                        prompt=arggen_prompt,
                        on_text=None,
                        on_thought=None,
                        text_format=None,
                        request_id=request_id,
                        session_id=session_id,
                        purpose=f"agent.{agent.name}.arggen.{tool.name}",
                    )
                    args = _parse_args_to_dict(generated_args_text)

                    # 첫 시도 실행
                    result = await self.run_tool_with_logging(
                        request_id=request_id,
                        agent=agent,
                        tool=tool,
                        args=args,
                        emit=emit_collector,
                    )
                    out = result.get("output") if isinstance(result, dict) else result
                    outputs.append(str(out))
                    # 최근 결과 키로 보관
                    try:
                        last_tool_outputs[tool.name] = out
                    except Exception:
                        pass
                    # 대화 툴 호출 기록(첫 시도)
                    try:
                        conversation.add_tool_call(
                            tool_call_id=str(uuid.uuid4()),
                            tool_name=tool_name,
                            tool_index=idx,
                            raw_args=json.dumps(args, ensure_ascii=False),
                            result=result if isinstance(result, dict) else {"output": out},
                        )
                    except Exception:
                        pass

                    # 실패 시: LLM에게 수정된 args를 받아 즉시 재시도 (최대 1회)
                    needs_retry = False
                    try:
                        needs_retry = not bool(result.get("success")) if isinstance(result, dict) else False
                    except Exception:
                        needs_retry = False

                    final_result = result
                    if needs_retry:
                        try:
                            # 현재 도구의 상세 설명(시그니처/usage_notes 포함)
                            current_tool_desc = _describe_tool_args(tool)
                            system_text = self.agent_manager._compose_system_prompt(agent)
                            error_text = ""
                            try:
                                error_text = json.dumps(result, ensure_ascii=False)
                            except Exception:
                                error_text = str(result)

                            repair_prompt = (
                                system_text
                                + "\n\nThe tool call failed. Using the information below, output only a corrected args JSON so that the same tool can be retried immediately.\n"
                                + "Rules:\n"
                                + "- Output exactly one JSON object. No explanations/markdown/code fences.\n"
                                + "- Keys must match the tool signature. No unnecessary keys.\n"
                                + "- No placeholders (${...}). Use literal values only.\n"
                                + "- For internal_api, 'path' must be a documented path; never an absolute URL. Do not include base_url/headers.\n"
                                + "- When time/date is needed, use explicit KST literals such as YYYY-MM-DD. If unknown, use a reasonable default.\n"
                                + "- Set method to an appropriate HTTP method.\n\n"
                                + "Tool spec:\n" + current_tool_desc + "\n\n"
                                + "Previous args (JSON):\n" + json.dumps(args, ensure_ascii=False) + "\n\n"
                                + "Failure result:\n" + error_text + "\n"
                            )

                            repaired_text, _prov, _usage = await llm_stream_text(
                                model_name=agent.model,
                                prompt=repair_prompt,
                                on_text=None,
                                on_thought=None,
                                text_format=None,
                                request_id=request_id,
                                session_id=session_id,
                                purpose=f"agent.{agent.name}.tool_repair.{tool.name}",
                            )
                            repaired_args = _parse_args_to_dict(repaired_text)
                            if repaired_args:
                                retry_result = await self.run_tool_with_logging(
                                    request_id=request_id,
                                    agent=agent,
                                    tool=tool,
                                    args=repaired_args,
                                    emit=emit_collector,
                                )
                                retry_out = retry_result.get("output") if isinstance(retry_result, dict) else retry_result
                                outputs.append(str(retry_out))
                                # 대화 툴 호출 기록(재시도)
                                try:
                                    conversation.add_tool_call(
                                        tool_call_id=str(uuid.uuid4()),
                                        tool_name=tool_name,
                                        tool_index=idx,
                                        raw_args=json.dumps(repaired_args, ensure_ascii=False),
                                        result=retry_result if isinstance(retry_result, dict) else {"output": retry_out},
                                    )
                                except Exception:
                                    pass
                                final_result = retry_result
                        except Exception:
                            # 복구 시도 중 오류 발생 시, 원래 결과를 유지
                            final_result = result

                    # 최종 실패 여부 집계
                    try:
                        ok_final = bool(final_result.get("success")) if isinstance(final_result, dict) else True
                        if not ok_final:
                            tool_failures.append(tool_name)
                    except Exception:
                        pass

            # 3) 품질 평가 및 최종 요약(만족 시) - 한 번에 수행
            tool_outputs_text = ("\n\n".join(outputs)).strip()

            # 4) 품질 평가 및 개선 여부 판단
            # 휴리스틱: 도구 의도 사용되었으나 out이 없거나 실패가 있으면 개선 신호 강화
            heuristic_signals: List[str] = []
            if plan_result.isRunTool and not outputs:
                heuristic_signals.append("no_tool_output")
            if tool_failures:
                heuristic_signals.append("tool_failure:" + ",".join(tool_failures))
            if not plan_parsed_ok:
                heuristic_signals.append("plan_parse_failed")
            if not plan_result.isRunTool and (agent.tools or []):
                heuristic_signals.append("plan_skipped_tools")

            improve_prompt = (
                self.agent_manager._compose_system_prompt(agent)
                + "\n\nBased on the following tool results, perform a quality check and, if satisfied, produce the final summary in one step.\n"
                + "Return JSON of the form {tryAgain:boolean, reason:string, improvements:string, revisedInstruction?:string, qualityScore?:number, finalText?:string}.\n"
                + "Policy:\n- If data/queries are needed but tools were not used or failed, set tryAgain=true\n- If the answer is vague/missing/inaccurate, set tryAgain=true and provide improvements\n- If sufficiently satisfied, set tryAgain=false and write finalText as a concise English response for the user\n"
                + ("Heuristic signals: " + ", ".join(heuristic_signals) + "\n" if heuristic_signals else "")
                + "\nSummary of tool results:\n" + (tool_outputs_text or "(none)")
            )

            def _improve_text(delta: str) -> None:
                conversation.accumulate_answer(delta)
            
            def _improve_thought(delta: str) -> None:
                conversation.accumulate_thought(delta)
            
            improve_raw, _iprov, _iusage = await llm_stream_text(
                model_name=agent.model,
                prompt=improve_prompt,
                on_text=_improve_text,
                on_thought=_improve_thought,
                text_format=AgentImproveResult,
                request_id=request_id,
                session_id=session_id,
                purpose=f"agent.{agent.name}.improve",
            )
            try:
                improve_obj = AgentImproveResult(**json.loads((improve_raw or "").strip()))
            except Exception:
                # 평가 파싱 실패 시, 휴리스틱만으로 종료/반복 결정
                if heuristic_signals:
                    improve_obj = AgentImproveResult(tryAgain=True, reason=",".join(heuristic_signals), improvements="도구 실패 원인 진단 후 올바른 파라미터로 재시도", revisedInstruction=None, qualityScore=None)
                else:
                    improve_obj = AgentImproveResult(tryAgain=False, reason="충분", improvements="", revisedInstruction=None, qualityScore=None)

            # 최신 최종 텍스트 캐시
            try:
                if isinstance(getattr(improve_obj, "finalText", None), str) and improve_obj.finalText:
                    last_final_answer_raw = improve_obj.finalText
            except Exception:
                pass

            if not improve_obj.tryAgain:
                return (improve_obj.finalText or tool_outputs_text or "")

            # 루프 계속: 개선 지시 반영
            improvement_notes = improve_obj.improvements or ""
            if isinstance(improve_obj.revisedInstruction, str) and improve_obj.revisedInstruction.strip():
                local_instruction = improve_obj.revisedInstruction.strip()
            else:
                # 기존 지시사항에 개선사항을 병합
                merged = (local_instruction + "\n" + improvement_notes).strip()
                local_instruction = merged

        # 루프 소진: 마지막 답변 반환
        return last_final_answer_raw
    
    async def run_tool_with_logging(
        self,
        *,
        request_id: str,
        agent: AgentDefinition,
        tool: Tool,
        args: Dict[str, Any],
        emit: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """도구 실행 및 로깅"""
        if emit:
            try:
                emit(_sse_event("tool_call", {"agent": agent.name, "tool": tool.name, "args": args}))
            except Exception:
                pass
        result = tool.safe_call(**args)
        # Tool call logging removed - using ConversationAccumulator instead
        if emit:
            try:
                output = result.get("output") if isinstance(result, dict) else result
                summary = str(output)
                if len(summary) > 400:
                    summary = summary[:400] + "…"
                emit(_sse_event("tool_result", {"agent": agent.name, "tool": tool.name, "result_summary": summary}))
            except Exception:
                pass
        return result
    
    def validate_tool_call(self, agent: AgentDefinition, tool_name: str, args: Dict[str, Any]) -> Optional[str]:
        """도구 호출 유효성 검증"""
        if not agent.tools:
            return f"Agent {agent.name} has no tools"
        
        tool = next((t for t in agent.tools if t.name == tool_name), None)
        if tool is None:
            available_tools = [t.name for t in agent.tools]
            return f"Tool '{tool_name}' not found in agent {agent.name}. Available tools: {available_tools}"
        
        # 추가적인 인자 검증 로직을 여기에 추가할 수 있음
        return None
    
    def get_tool_usage_stats(self) -> Dict[str, Any]:
        """도구 사용 통계 (실제로는 DB에서 조회해야 함)"""
        # 실제 구현에서는 tool_calls 테이블에서 통계를 조회
        return {
            "total_calls": 0,
            "success_rate": 0.0,
            "most_used_tools": [],
            "average_latency": 0.0
        }
