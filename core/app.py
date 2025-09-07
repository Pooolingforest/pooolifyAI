from __future__ import annotations

import asyncio
import json
import logging
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import re

from ..agents.runtime import AgentDefinition
from ..config import settings
# from ..llm.base import llm_call_and_log  # 삭제됨
from ..llm.streaming import stream_text as llm_stream_text
from . import schemas
from ..persistence import db as dbops
from ..tools.base import Tool
from .conversation import ConversationAccumulator, MessageType, ConversationMessage, ToolCall
from .conversation_manager import ConversationManager
from .agent_manager import AgentManager
from .tool_executor import ToolExecutor
from .schemas import AnalysisResult


logger = logging.getLogger(__name__)


@dataclass
class ManagerLimits:
    max_steps: int = 20
    max_depth: int = 4
    max_duration_s: int = 120


class PooolifyApp:
    @dataclass
    class CorsOptions:
        allow_origins: Optional[List[str]] = None
        allow_origin_regex: Optional[str] = None
        allow_methods: Optional[List[str]] = None
        allow_headers: Optional[List[str]] = None
        expose_headers: Optional[List[str]] = None
        allow_credentials: bool = False
        max_age: Optional[int] = None

    @dataclass
    class ManagerConfig:
        model: str = "gpt-5"
        # 통합된 계획+선택 템플릿: 에이전트 카탈로그를 보고 계획과 에이전트 선택을 동시에 수행
        plan_prompt_template: str = (
            "당신은 매니저 에이전트입니다. 사용자의 질문을 분석하여 적절한 처리 방식을 결정하세요.\n\n"
            "사용 가능한 에이전트:\n{agents_catalog}\n\n"
            "출력 규칙:\n"
            "- 반드시 JSON 하나만 출력합니다. 서두/말미의 설명, 마크다운, 코드블록 금지.\n"
            "- Fields: text (string), isRunAgent (boolean), agent (array of objects with keys 'name', 'instruction', 'step', 'runMode', 'dependsOn', 'outputsTo', 'order?').\n"
            "- 일관성 제약:\n"
            "  1) isRunAgent가 false이면 agent는 반드시 빈 배열([])이어야 합니다.\n"
            "  2) isRunAgent가 true이면 agent는 1개 이상이어야 하며, name은 상단 카탈로그 중 하나여야 합니다.\n"
            "  3) instruction은 각 에이전트가 바로 실행할 수 있을 정도로 구체적이고 간결한 한국어 문장(최대 200자)이어야 합니다.\n"
            "  4) 실행 스케줄: step별로 작업을 배치합니다. 같은 step에 있는 에이전트 중 runMode가 PARALLEL인 항목들은 서로 병렬 실행할 수 있습니다. runMode가 ISOLATED인 항목은 해당 step에서 단독으로 실행되어야 하며, 같은 step 내 다른 항목과 병행하지 않습니다. step은 1부터 시작하여 증가합니다.\n"
            "  5) 의존성: dependsOn에 나열된 에이전트들은 현재 항목보다 앞선 step에서 완료되어 있어야 합니다. outputsTo는 결과 전달 대상이며 자동으로 컨텍스트 라우팅됩니다.\n"
            "- 추가 필드 금지(additionalProperties: false). 필드는 오직 text, isRunAgent, agent만 사용.\n\n"
            "- 각 에이전트는 여러개의 도구를 사용할 수 있으므로, 동일한 에이전트를 여러번 선택하는건 최대한 지양하고, 하나의 에이전트가 여러개의 툴을 호출할 수 있는 구조로 지시합니다.\n\n"
            "질문: {query}"
        )
        final_prompt_template: str = (
            "에이전트 실행 결과:\n{context}\n\n"
            "에이전트 실행 결과를 바탕으로 사용자의 원래 요청에 대한 최종 답변을 작성하고, 요청이 충분히 완료되었는지 판단해주세요.\n"
            "출력 규칙:\n"
            "- 반드시 JSON 하나만 출력합니다. 서두/말미의 설명, 마크다운, 코드블록 금지.\n"
            "- Fields: text (string), isComplete (boolean), nextAction (string, optional).\n"
            "- text: 사용자에게 보여줄 완전한 최종 답변\n"
            "- isComplete: 사용자의 원래 요청에 충분히 답변했으면 true, 추가 작업이 필요하면 false\n"
            "- nextAction: isComplete가 false인 경우에만, 다음에 수행해야 할 작업 설명\n"
            "- 추가 필드 금지(additionalProperties: false). 필드는 오직 text, isComplete, nextAction만 사용.\n"
            "사용자의 기존 요청 (질의): {query}"
        )
        # @agent/index.py에서 정의되는 시스템 프롬프트(선택) — 템플릿 앞에 덧붙임
        system_instruction: str = ""

    def __init__(
        self,
        *,
        limits: Optional[ManagerLimits] = None,
        manager: Optional["PooolifyApp.ManagerConfig"] = None,
        cors: Optional["PooolifyApp.CorsOptions"] = None,
    ) -> None:
        self.limits = limits or ManagerLimits()
        self.manager: PooolifyApp.ManagerConfig = manager or PooolifyApp.ManagerConfig()
        self.cors: Optional[PooolifyApp.CorsOptions] = cors
        
        # 매니저들 초기화
        self.agent_manager = AgentManager()
        self.conversation_manager = ConversationManager()
        self.tool_executor = ToolExecutor(self.agent_manager)

    def register_agent(self, agent: AgentDefinition) -> None:
        self.agent_manager.register_agent(agent)

    @property
    def agents(self) -> Dict[str, AgentDefinition]:
        return self.agent_manager.agents

    async def add_message(
        self,
        *,
        session_id: str,
        query: str,
        conversation: Optional[ConversationAccumulator] = None,
        request_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        import time

        options = options or {}
        request_id = request_id or str(uuid.uuid4())

        prefix = (self.manager.system_instruction.strip() + "\n\n") if self.manager.system_instruction.strip() else ""
        model = str((options or {}).get("model") or self.manager.model)

        # orchestrate를 백그라운드에서 실행
        async def orchestrate() -> None:
            return await self._orchestrate_internal(session_id, query, loop_count=1, max_loops=3, request_id=request_id)
        
        # orchestrate를 백그라운드에서 실행 (fire-and-forget)
        asyncio.create_task(orchestrate())
        
        # 즉시 응답 반환 (orchestrate 실행 전에 완료)
        return {
            "status": "processing",
            "session_id": session_id,
            "request_id": request_id
        }

    async def _orchestrate_internal(
        self, 
        session_id: str, 
        user_message: str, 
        loop_count: int = 1, 
        max_loops: int = 3,
        request_id: Optional[str] = None,
    ) -> None:
        """내부 orchestrate 구현 - 재귀 호출 가능"""
        conversation = self.conversation_manager.get_or_create_conversation(session_id)
            
        prefix = (self.manager.system_instruction.strip() + "\n\n") if self.manager.system_instruction.strip() else ""
        model = str(self.manager.model)
        
        print(f"orchestrate 실행 시작 (루프 {loop_count}/{max_loops})")
        
        try:
            # AI 메시지 시작 (매니저)
            conversation.start_ai_message(agent="manager")
            
            print(f"1단계 요청 시작")
            # 1단계 : 요청 분석 및 처리 방식 결정
            agents_catalog = self.agent_manager.build_agents_catalog()
            analysis_prompt = prefix + self.manager.plan_prompt_template.format(query=user_message, agents_catalog=agents_catalog)
            
            def _analysis_text(delta: str) -> None:
                conversation.accumulate_answer(delta)  # 분석 결과를 답변으로 누적
            
            def _analysis_thought(delta: str) -> None:
                conversation.accumulate_thought(delta)  # 분석 과정을 추론으로 누적
            
            analysis_response, _analysis_provider, _analysis_usage = await llm_stream_text(
                model_name=model,
                prompt=analysis_prompt,
                on_text=_analysis_text,
                on_thought=_analysis_thought,
                text_format=AnalysisResult,
                request_id=request_id,
                session_id=session_id,
                purpose="manager.analysis",
            )

            print(f"1단계 요청 완료")
            
            # 구조화된 출력 파싱: AnalysisResult 모델 사용
            analysis_result = None
            try:
                analysis_data = json.loads((analysis_response or "").strip())
                # agent 필드가 없으면 빈 리스트로 설정
                if "agent" not in analysis_data or analysis_data.get("agent") is None:
                    analysis_data["agent"] = []
                # 에이전트 항목 정규화: null/잘못된 타입 교정
                if isinstance(analysis_data.get("agent"), list):
                    normalized_agents = []
                    for item in analysis_data.get("agent", []):
                        if not isinstance(item, dict):
                            # 스킵하거나 최소 구조로 교정
                            try:
                                item = {"name": str(item)}
                            except Exception:
                                continue
                        # 필드 교정
                        name = item.get("name")
                        if not isinstance(name, str) or not name.strip():
                            # 이름이 없으면 무시
                            continue
                        if item.get("instruction") is None:
                            item["instruction"] = ""
                        # 리스트 필드 교정
                        deps = item.get("dependsOn")
                        outs = item.get("outputsTo")
                        if not isinstance(deps, list):
                            item["dependsOn"] = []
                        if not isinstance(outs, list):
                            item["outputsTo"] = []
                        # 선택 필드 기본값 보정
                        run_mode = str(item.get("runMode") or "PARALLEL").upper()
                        if run_mode not in ("PARALLEL", "ISOLATED"):
                            run_mode = "PARALLEL"
                        item["runMode"] = run_mode
                        try:
                            step_val = int(item.get("step", 1) or 1)
                        except Exception:
                            step_val = 1
                        item["step"] = step_val
                        normalized_agents.append(item)
                    analysis_data["agent"] = normalized_agents
                analysis_result = AnalysisResult(**analysis_data)
            except Exception as e:
                logger.warning(f"구조화된 출력 파싱 실패: {e}")
                # 파싱 실패시 기본값으로 폴백 (비실행 경로로 종료)
                analysis_result = AnalysisResult(
                    text="분석 결과를 파싱할 수 없습니다.",
                    isRunAgent=False,
                    agent=[]
                )
            
            print("analysis_result.isRunAgent: ", analysis_result.isRunAgent)
            print("analysis_result.agent: ", analysis_result.agent)

            # 에이전트 실행 여부 확인
            if not analysis_result.isRunAgent:
                # 요청 종료
                conversation.end_request()
                return
            
            # 에이전트 호출이 필요한 경우 계속 진행
            selected_agents: List[Tuple[AgentDefinition, str]] = []
            for agent_task in analysis_result.agent:
                agent = self.agent_manager.get_agent(agent_task.name)
                if agent:
                    selected_agents.append((agent, agent_task.instruction or ""))
            
            if not selected_agents:
                conversation.accumulate_error("사용 가능한 에이전트가 없습니다.")
                conversation.end_request()
                return
            
            # 2단계 : 혼합 실행 스케줄러 (step/runMode/dependsOn 지원)
            agent_results: List[str] = []

            # 계획 작업들 수집
            plan_tasks = list(analysis_result.agent or [])
            if not plan_tasks:
                conversation.accumulate_error("선택된 에이전트가 없습니다.")
                conversation.end_request()
                return

            # 이름 → AgentDefinition 매핑
            name_to_agent: Dict[str, AgentDefinition] = {}
            for task in plan_tasks:
                ag = self.agent_manager.get_agent(task.name)
                if ag:
                    name_to_agent[task.name] = ag

            # outputsTo 역인덱스: 수신자 → 발신자 목록
            incoming_from_outputs: Dict[str, List[str]] = {}
            for task in plan_tasks:
                for dst in getattr(task, "outputsTo", []) or []:
                    if not isinstance(dst, str):
                        continue
                    incoming_from_outputs.setdefault(dst, []).append(task.name)

            # step 단위로 그룹화
            step_to_tasks: Dict[int, List[Any]] = {}
            for task in plan_tasks:
                step_value = getattr(task, "step", 1) or 1
                try:
                    step_int = int(step_value)
                except Exception:
                    step_int = 1
                step_to_tasks.setdefault(step_int, []).append(task)

            # 실행 결과 맵: 에이전트 이름 → 결과 텍스트
            result_by_agent: Dict[str, str] = {}

            async def _run_one(task_obj) -> Tuple[str, str]:
                agent = name_to_agent.get(task_obj.name)
                if not agent:
                    return task_obj.name, f"[{task_obj.name}] 에이전트를 찾을 수 없습니다."

                # 컨텍스트 구성: dependsOn + outputsTo 역참조
                deps: List[str] = list(getattr(task_obj, "dependsOn", []) or [])
                from_outputs: List[str] = list(incoming_from_outputs.get(task_obj.name, []) or [])
                context_sources = []
                # 명시 의존 우선
                for dep in deps:
                    if dep in result_by_agent:
                        context_sources.append(result_by_agent[dep])
                # outputsTo 기반 보조 수신
                for src in from_outputs:
                    if src in result_by_agent and src not in deps:
                        context_sources.append(result_by_agent[src])
                # 폴백: 의존성이 전혀 없으면 이전 단계 전체 요약 제공(선택)
                ctx_text = "\n".join(context_sources).strip()

                try:
                    print(f"2단계 : 에이전트 실행 — step={getattr(task_obj,'step',1)} mode={getattr(task_obj,'runMode','PARALLEL')} name={task_obj.name}")
                    agent_result = await self.tool_executor.run_agent_once(
                        agent=agent,
                        session_id=session_id,
                        request_id=f"loop_{loop_count}_{agent.name}",
                        conversation=conversation,
                        instruction=(getattr(task_obj, "instruction", "") or ""),
                        context=ctx_text,
                        emit_collector=None,
                    )
                    if agent_result:
                        formatted = f"[{agent.name}] {agent_result}"
                    else:
                        formatted = f"[{agent.name}] 에이전트 실행 중 오류가 발생했습니다."
                except Exception as e:
                    logger.error(f"에이전트 {agent.name} 실행 실패: {e}")
                    formatted = f"[{agent.name}] 실행 실패: {str(e)}"
                return agent.name, formatted

            # step 순서대로 실행
            for step in sorted(step_to_tasks.keys()):
                tasks_in_step = step_to_tasks[step]
                parallel_tasks = [t for t in tasks_in_step if str(getattr(t, "runMode", "PARALLEL")).upper() == "PARALLEL"]
                isolated_tasks = [t for t in tasks_in_step if str(getattr(t, "runMode", "PARALLEL")).upper() == "ISOLATED"]

                # 병렬 배치 실행
                if parallel_tasks:
                    coros = [_run_one(t) for t in parallel_tasks]
                    results = await asyncio.gather(*coros)
                    for name, formatted in results:
                        result_by_agent[name] = formatted
                        agent_results.append(formatted)

                # 고립 실행(단독) — 같은 step 내에서도 각각 단독 실행 보장
                for t in isolated_tasks:
                    name, formatted = await _run_one(t)
                    result_by_agent[name] = formatted
                    agent_results.append(formatted)

            print("2단계 모든 에이전트 혼합 실행 완료")

            print("agent_results: ", agent_results)
            
            # 3단계 : 매니저 최종 요약 및 완료 여부 판단
            conversation.start_ai_message(agent="manager")
            
            # 에이전트 결과와 원래 요청을 바탕으로 최종 요약 요청
            agent_results_text = "\n".join(agent_results) if agent_results else "에이전트 실행 중 오류가 발생했습니다."
            
            # final_prompt_template을 활용하여 최종 요약 프롬프트 생성
            final_summary_prompt = prefix + self.manager.final_prompt_template.format(
                context=agent_results_text,
                query=user_message
            )

            try:
                def _final_text(delta: str) -> None:
                    conversation.accumulate_answer(delta)
                
                def _final_thought(delta: str) -> None:
                    conversation.accumulate_thought(delta)
                
                final_summary_response, _final_provider, _final_usage = await llm_stream_text(
                    model_name=model,
                    prompt=final_summary_prompt,
                    on_text=_final_text,
                    on_thought=_final_thought,
                    text_format=schemas.FinalSummaryResult,
                    request_id=request_id,
                    session_id=session_id,
                    purpose="manager.final_summary",
                )
                
                # 구조화된 응답 파싱
                final_summary_result = schemas.FinalSummaryResult.model_validate_json(final_summary_response)
                
                # 완료되지 않은 경우 재실행 검토
                if not final_summary_result.isComplete and loop_count < max_loops:
                    print(f"요청이 완료되지 않음. 다음 작업: {final_summary_result.nextAction}")
                    conversation.accumulate_answer(f"\n\n추가 작업이 필요합니다: {final_summary_result.nextAction}")
                    conversation.end_ai_message()
                    
                    # 재실행을 위한 새로운 사용자 메시지로 처리
                    next_user_message = final_summary_result.nextAction or user_message
                    return await self._orchestrate_internal(session_id, next_user_message, loop_count + 1, max_loops)
                else:
                    # 요청 완료 또는 최대 루프 도달
                    if loop_count >= max_loops:
                        conversation.accumulate_answer(f"\n\n최대 실행 횟수({max_loops})에 도달했습니다.")
                    conversation.end_request()
                    try:
                        conversation.add_completion_message(
                            text="REQUEST_COMPLETED",
                            metadata={
                                "loopCount": loop_count,
                                "maxLoops": max_loops,
                                "complete": True
                            }
                        )
                    except Exception:
                        pass
                    
            except Exception as e:
                logger.error(f"최종 요약 생성 실패: {e}")
                conversation.accumulate_answer(f"\n\n최종 요약 생성 중 오류가 발생했습니다: {str(e)}")
                conversation.end_request()
                try:
                    conversation.add_completion_message(
                        text="REQUEST_COMPLETED_WITH_ERROR",
                        metadata={
                            "loopCount": loop_count,
                            "maxLoops": max_loops,
                            "complete": True,
                            "error": str(e)
                        }
                    )
                except Exception:
                    pass
                    
        except Exception as e:
            logger.error(f"orchestrate 실행 중 오류 발생: {e}")
            conversation.accumulate_answer(f"\n\n처리 중 오류가 발생했습니다: {str(e)}")
            conversation.end_request()
            try:
                conversation.add_completion_message(
                    text="REQUEST_COMPLETED_WITH_ERROR",
                    metadata={
                        "loopCount": loop_count,
                        "maxLoops": max_loops,
                        "complete": True,
                        "error": str(e)
                    }
                )
            except Exception:
                pass
