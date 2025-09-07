from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field

from ..tools.base import Tool


class AgentDefinition(BaseModel):
    name: str
    # 새로운 선언 필드들: 역할/목표/지식
    role: Optional[str] = Field(default="", description="에이전트의 역할")
    goal: Optional[str] = Field(default="", description="에이전트의 목표")
    background: str
    knowledge: Optional[str] = Field(default="", description="에이전트가 기본적으로 가지고 있어야 하는 필수 지식")
    model: str
    tools: Optional[List[Tool]] = None
    max_loops: int = Field(default=3, ge=1, le=10, description="최대 개선 루프 횟수")

