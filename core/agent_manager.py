from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ..agents.runtime import AgentDefinition
from ..tools.base import Tool


logger = logging.getLogger(__name__)


class AgentManager:
    """에이전트 관리를 담당하는 클래스"""
    
    def __init__(self):
        self._agents: Dict[str, AgentDefinition] = {}
    
    def register_agent(self, agent: AgentDefinition) -> None:
        """에이전트 등록"""
        self._agents[agent.name] = agent
        logger.debug(f"Registered agent: {agent.name}")
    
    @property
    def agents(self) -> Dict[str, AgentDefinition]:
        """등록된 모든 에이전트 반환"""
        return self._agents
    
    def get_agent(self, name: str) -> Optional[AgentDefinition]:
        """이름으로 에이전트 조회"""
        return self._agents.get(name)
    
    def has_agent(self, name: str) -> bool:
        """에이전트 존재 여부 확인"""
        return name in self._agents
    
    def get_agent_names(self) -> List[str]:
        """모든 에이전트 이름 목록 반환"""
        return list(self._agents.keys())
    
    def get_agents_by_model(self, model: str) -> List[AgentDefinition]:
        """특정 모델을 사용하는 에이전트들 반환"""
        return [agent for agent in self._agents.values() if agent.model == model]
    
    def build_agents_catalog(self) -> str:
        """에이전트 카탈로그 문자열 생성"""
        parts: List[str] = []
        for ag in self._agents.values():
            sp = self._compose_system_prompt(ag)
            if len(sp) > 280:
                sp = sp[:280] + "…"
            parts.append(
                f"- name: {ag.name}\n  role: {(ag.role or '-')}\n  goal: {(ag.goal or '-')}\n  background: {ag.background}\n  model: {ag.model}\n  system: {sp}"
            )
        return "\n".join(parts)
    
    @staticmethod
    def _compose_system_prompt(agent: AgentDefinition) -> str:
        """에이전트의 시스템 프롬프트 구성"""
        parts: List[str] = []
        role_text = (agent.role or "").strip()
        goal_text = (agent.goal or "").strip()
        background_text = (agent.background or "").strip()
        knowledge_text = (agent.knowledge or "").strip()

        if role_text:
            parts.append(f"You are {role_text}.")
        if background_text:
            parts.append(background_text)
        if goal_text:
            parts.append(f"- Goal: {goal_text}")
        if knowledge_text:
            parts.append("Baseline knowledge:\n" + knowledge_text)

        if not parts:
            return "You are an expert agent. Answer concisely and accurately in English."
        return "\n".join(parts)
    
    def find_tool(self, tool_name: str) -> Optional[Tuple[AgentDefinition, Tool]]:
        """도구 이름으로 에이전트와 도구 찾기"""
        for agent in self._agents.values():
            for tool in (agent.tools or []):
                if tool.name == tool_name:
                    return agent, tool
        return None
    
    def get_agents_with_tools(self) -> List[AgentDefinition]:
        """도구를 가진 에이전트들만 반환"""
        return [agent for agent in self._agents.values() if agent.tools]
    
    def get_tool_names(self) -> List[str]:
        """모든 도구 이름 목록 반환"""
        tool_names = []
        for agent in self._agents.values():
            if agent.tools:
                tool_names.extend(tool.name for tool in agent.tools)
        return tool_names
    
    def validate_agents(self) -> List[str]:
        """에이전트 설정 검증 및 오류 목록 반환"""
        errors = []
        
        for agent_name, agent in self._agents.items():
            # 필수 필드 검증
            if not agent.name:
                errors.append(f"Agent {agent_name}: name is required")
            
            if not agent.model:
                errors.append(f"Agent {agent_name}: model is required")
            
            # 도구 검증
            if agent.tools:
                tool_names = set()
                for tool in agent.tools:
                    if not tool.name:
                        errors.append(f"Agent {agent_name}: tool name is required")
                    elif tool.name in tool_names:
                        errors.append(f"Agent {agent_name}: duplicate tool name '{tool.name}'")
                    else:
                        tool_names.add(tool.name)
        
        return errors
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """에이전트 통계 정보 반환"""
        total_agents = len(self._agents)
        agents_with_tools = len(self.get_agents_with_tools())
        total_tools = len(self.get_tool_names())
        
        model_counts = {}
        for agent in self._agents.values():
            model = agent.model or "unknown"
            model_counts[model] = model_counts.get(model, 0) + 1
        
        return {
            "total_agents": total_agents,
            "agents_with_tools": agents_with_tools,
            "total_tools": total_tools,
            "model_distribution": model_counts,
            "validation_errors": self.validate_agents()
        }
