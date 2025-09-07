from __future__ import annotations

from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field


class AnalysisResult(BaseModel):
    """1단계 분석 결과 구조 - 매니저가 에이전트 선택 및 실행 여부를 결정"""
    model_config = {"extra": "forbid"}  # additionalProperties: false 강제
    
    class AgentTask(BaseModel):
        name: str = Field(description="실행할 에이전트 이름")
        instruction: str = Field(description="해당 에이전트에 대한 구체 지시 내용")
        order: Optional[int] = Field(default=None, description="(선택) 순차 실행 순서, 1부터 시작하는 정수")
        step: int = Field(default=1, description="실행 단계(스테이지). 같은 step은 병렬 배치 가능")
        runMode: Literal["PARALLEL", "ISOLATED"] = Field(
            default="PARALLEL",
            description="실행 방식: PARALLEL은 동시 실행 가능, ISOLATED는 단독 실행"
        )
        dependsOn: List[str] = Field(default_factory=list, description="컨텍스트를 받기 위해 선행되어야 할 에이전트 이름 목록")
        outputsTo: List[str] = Field(default_factory=list, description="(선택) 결과를 전달할 후행 에이전트 이름 목록")

    text: str = Field(
        description="사용자에게 보여줄 분석 결과나 답변. 에이전트를 실행하는 경우 어떤 작업을 수행할지 설명하고, 직접 답변하는 경우 최종 답변 내용을 포함."
    )
    isRunAgent: bool = Field(
        description="에이전트 실행 여부. true: 복잡한 작업이나 도구 사용이 필요한 경우, false: 간단한 질문이나 일반 상식으로 직접 답변 가능한 경우"
    )
    agent: List[AgentTask] = Field(description="실행할 에이전트 목록. 각 항목은 {name, instruction, step, runMode, dependsOn, outputsTo, order?} 객체여야 함.")


class AgentPlanCall(BaseModel):
    """에이전트의 도구 호출 계획 항목"""
    tool: str = Field(description="사용할 도구 이름")
    args: Optional[str] = Field(default=None, description="도구 인자 (JSON 문자열). 계획 단계에서는 생성하지 않음")
    reason: str = Field(description="도구 선택 이유 요약")


class AgentPlanResult(BaseModel):
    """에이전트의 도구 호출 계획 결과"""
    model_config = {"extra": "forbid"}
    text: str = Field(description="계획 요약 텍스트")
    isRunTool: bool = Field(description="툴 실행 여부")
    calls: List[AgentPlanCall] = Field(description="툴 호출 목록")


class AgentFinalResult(BaseModel):
    """에이전트의 최종 결과"""
    model_config = {"extra": "forbid"}
    text: str = Field(description="최종 사용자에게 보여줄 요약 텍스트")


class AgentImproveResult(BaseModel):
    """최종 결과의 품질 점검 및 개선 지시 구조"""
    model_config = {"extra": "forbid"}
    tryAgain: bool = Field(description="재시도 필요 여부")
    reason: str = Field(description="재시도 또는 종료 판단 근거")
    improvements: str = Field(description="다음 반복에서 개선할 구체 지시사항 요약")
    revisedInstruction: Optional[str] = Field(default=None, description="다음 반복에 사용할 보완된 에이전트 지시사항")
    qualityScore: Optional[int] = Field(default=None, description="0~100 품질 점수(선택)")
    finalText: Optional[str] = Field(default=None, description="tryAgain=false 인 경우 사용자에게 보여줄 최종 요약 텍스트")

class FinalSummaryResult(BaseModel):
    """3단계 매니저 최종 요약 결과 구조"""
    model_config = {"extra": "forbid"}
    
    text: str = Field(
        description="사용자에게 보여줄 최종 답변. 에이전트 결과들을 종합하여 사용자의 원래 질문에 대한 완전한 답변을 제공."
    )
    isComplete: bool = Field(
        description="요청 완료 여부. true: 사용자의 원래 요청에 충분히 답변했음, false: 추가 작업이 필요함"
    )
    nextAction: Optional[str] = Field(
        default=None,
        description="isComplete가 false인 경우, 다음에 수행해야 할 작업에 대한 설명"
    )
