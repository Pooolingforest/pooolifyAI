from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime


logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """메시지 타입 정의 - 단순화"""
    HUMAN = "MESSAGE_TYPE_HUMAN"
    AI = "MESSAGE_TYPE_AI"
    COMPLETE = "MESSAGE_TYPE_COMPLETE"


@dataclass
class ToolCall:
    """도구 호출 정보"""
    tool_call_id: str
    tool_name: str
    tool_index: int
    raw_args: str
    result: Dict[str, Any]


@dataclass
class MessageContent:
    """메시지 내용 구조화"""
    answer: str = ""           # 최종 답변 텍스트
    thought: str = ""          # 추론/사고 과정 텍스트
    plan: str = ""             # 계획 텍스트
    route: str = ""            # 라우팅 정보
    decision: str = ""         # 의사결정 내용
    error: str = ""            # 에러 메시지
    completion: str = ""       # 완료 신호 (클라이언트 인식용)
    tool_results: Optional[List[ToolCall]] = None  # 도구 호출 결과들
    
    def __post_init__(self):
        if self.tool_results is None:
            self.tool_results = []
    
    def get_combined_text(self) -> str:
        """모든 텍스트 내용을 결합하여 반환"""
        parts = []
        if self.thought:
            parts.append(f"[추론] {self.thought}")
        if self.plan:
            parts.append(f"[계획] {self.plan}")
        if self.route:
            parts.append(f"[라우팅] {self.route}")
        if self.decision:
            parts.append(f"[결정] {self.decision}")
        if self.answer:
            parts.append(self.answer)
        if self.error:
            parts.append(f"[오류] {self.error}")
        return "\n".join(parts)
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 - 빈 값은 제외"""
        result = {}
        
        # 빈 문자열이 아닌 경우만 포함
        if self.answer:
            result["answer"] = self.answer
        if self.thought:
            result["thought"] = self.thought
        if self.plan:
            result["plan"] = self.plan
        if self.route:
            result["route"] = self.route
        if self.decision:
            result["decision"] = self.decision
        if self.error:
            result["error"] = self.error
        if self.completion:
            result["completion"] = self.completion
        
        if self.tool_results:
            result["tool_results"] = [
                {
                    "toolCallId": tc.tool_call_id,
                    "toolName": tc.tool_name,
                    "toolIndex": tc.tool_index,
                    "rawArgs": tc.raw_args,
                    "result": tc.result,
                }
                for tc in self.tool_results
            ]
        
        return result


@dataclass
class ConversationMessage:
    """대화 메시지 단위"""
    type: MessageType = MessageType.AI
    content: Optional[MessageContent] = None
    bubble_id: Optional[str] = None
    server_bubble_id: Optional[str] = None
    agent: Optional[str] = None
    timestamp: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.bubble_id is None:
            self.bubble_id = str(uuid.uuid4())
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.content is None:
            self.content = MessageContent()
        if self.metadata is None:
            self.metadata = {}

    @property
    def text(self) -> str:
        """하위 호환성을 위한 text 속성"""
        return self.content.get_combined_text() if self.content else ""

    def append_answer(self, delta: str) -> None:
        """답변 텍스트 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.answer += delta

    def append_thought(self, delta: str) -> None:
        """추론 텍스트 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.thought += delta

    def append_plan(self, delta: str) -> None:
        """계획 텍스트 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.plan += delta

    def append_route(self, delta: str) -> None:
        """라우팅 정보 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.route += delta

    def append_decision(self, delta: str) -> None:
        """의사결정 내용 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.decision += delta

    def append_error(self, delta: str) -> None:
        """에러 메시지 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.error += delta

    def append_completion(self, delta: str) -> None:
        """완료 신호 누적 추가"""
        if self.content is None:
            self.content = MessageContent()
        self.content.completion += delta

    def append_text(self, delta: str) -> None:
        """하위 호환성을 위한 텍스트 누적 추가 (answer로 처리)"""
        self.append_answer(delta)

    def add_tool_result(self, tool_call: ToolCall) -> None:
        """도구 호출 결과 추가"""
        if self.content is None:
            self.content = MessageContent()
        if self.content.tool_results is None:
            self.content.tool_results = []
        self.content.tool_results.append(tool_call)

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환 - 빈 값은 제외"""
        result = {
            "type": self.type.value,
        }
        
        # bubbleId는 항상 포함 (필수 필드)
        if self.bubble_id:
            result["bubbleId"] = self.bubble_id
        
        # content는 빈 딕셔너리가 아닌 경우만 포함
        content_dict = self.content.to_dict() if self.content else {}
        if content_dict:
            result["content"] = content_dict
        

        
        # 선택적 필드들 - 값이 있는 경우만 포함
        if self.server_bubble_id:
            result["serverBubbleId"] = self.server_bubble_id
            
        if self.agent:
            result["agent"] = self.agent
            
        if self.timestamp:
            result["timestamp"] = self.timestamp.isoformat()
            
        if self.metadata:
            result["metadata"] = self.metadata
            
        return result


@dataclass
class ConversationAccumulator:
    """대화 데이터 누적기 - 세션별로 관리"""
    conversation: List[ConversationMessage] = field(default_factory=list)
    session_id: Optional[str] = None
    current_message: Optional[ConversationMessage] = None
    current_request_id: Optional[str] = None  # 현재 진행 중인 요청 ID (스트리밍용)
    
    # 자동 저장 관련 필드
    _auto_save_enabled: bool = field(default=False, init=False)
    _save_callback: Optional[Callable[[str, str, Dict[str, Any]], None]] = field(default=None, init=False)  # session_id, message_id, message_data
    _last_save_time: float = field(default=0.0, init=False)
    _save_interval: float = field(default=2.0, init=False)  # 2초마다 저장
    _pending_saves: Dict[str, ConversationMessage] = field(default_factory=dict, init=False)  # message_id -> message
    _save_task: Optional[asyncio.Task] = field(default=None, init=False)
    
    # JSON 스트리밍에서 text 필드 추출을 위한 상태
    _json_buffer: str = field(default="", init=False)
    _in_text_field: bool = field(default=False, init=False)
    _text_content: str = field(default="", init=False)
    _in_string: bool = field(default=False, init=False)
    _escape_next: bool = field(default=False, init=False)
    _last_text_emitted: str = field(default="", init=False)

    def __post_init__(self):
        if self.session_id is None:
            self.session_id = str(uuid.uuid4())
    
    def _reset_json_parsing_state(self) -> None:
        """JSON 파싱 상태 초기화"""
        self._json_buffer = ""
        self._in_text_field = False
        self._text_content = ""
        self._in_string = False
        self._escape_next = False
        self._last_text_emitted = ""

    def _flush_pending_text_field(self) -> None:
        """스트리밍 JSON 파서에 누적된 미완 텍스트를 answer에 반영 후 비움"""
        if self.current_message is None:
            return
        try:
            if self._in_text_field and self._text_content:
                # 증가분만 방출하여 중복 누적 방지
                prev = self._last_text_emitted or ""
                full = self._text_content
                delta = full[len(prev):] if full.startswith(prev) else full
                if delta:
                    self.current_message.append_answer(delta)
                    self._mark_dirty()
                self._last_text_emitted = full
        finally:
            # text 필드 관련 상태 초기화 (다음 요청을 위해 정리)
            self._in_text_field = False
            self._text_content = ""
            self._in_string = False
            self._escape_next = False

    def enable_auto_save(self, save_callback: Callable[[str, str, Dict[str, Any]], None], interval: float = 2.0) -> None:
        """자동 저장 기능 활성화 - 메시지 블록 단위로 저장"""
        self._auto_save_enabled = True
        self._save_callback = save_callback
        self._save_interval = interval
        self._last_save_time = time.time()
        logger.debug(f"Auto-save enabled for session {self.session_id} with {interval}s interval")
    
    def disable_auto_save(self) -> None:
        """자동 저장 기능 비활성화"""
        self._auto_save_enabled = False
        self._save_callback = None
        self._pending_saves.clear()
        if self._save_task and not self._save_task.done():
            self._save_task.cancel()
        logger.debug(f"Auto-save disabled for session {self.session_id}")
    
    def _mark_message_dirty(self, message: ConversationMessage) -> None:
        """특정 메시지가 변경되었음을 표시하고 필요시 저장 스케줄링"""
        if not self._auto_save_enabled or not self._save_callback or not message.bubble_id:
            return
        
        self._pending_saves[message.bubble_id] = message
        current_time = time.time()
        
        # 마지막 저장으로부터 충분한 시간이 지났으면 즉시 저장
        if current_time - self._last_save_time >= self._save_interval:
            self._schedule_save()
        # 아니면 지연 저장 스케줄링
        elif not self._save_task or self._save_task.done():
            delay = self._save_interval - (current_time - self._last_save_time)
            self._save_task = asyncio.create_task(self._delayed_save(delay))
    
    def _mark_dirty(self) -> None:
        """현재 메시지가 변경되었음을 표시"""
        if self.current_message:
            self._mark_message_dirty(self.current_message)
    
    async def _delayed_save(self, delay: float) -> None:
        """지연 저장 실행"""
        try:
            await asyncio.sleep(delay)
            if self._pending_saves:  # 여전히 저장이 필요한 경우
                self._schedule_save()
        except asyncio.CancelledError:
            pass
    
    def _schedule_save(self) -> None:
        """즉시 저장 실행 - 변경된 메시지들만 저장"""
        if not self._auto_save_enabled or not self._save_callback or not self._pending_saves:
            return
        
        try:
            # 변경된 메시지들을 개별적으로 저장
            for message_id, message in list(self._pending_saves.items()):
                message_data = self._message_to_dict(message)
                self._save_callback(self.session_id, message_id, message_data)
            
            self._last_save_time = time.time()
            saved_count = len(self._pending_saves)
            self._pending_saves.clear()
            logger.debug(f"Auto-saved {saved_count} messages for session {self.session_id}")
        except Exception as e:
            logger.error(f"Failed to auto-save messages for session {self.session_id}: {e}")
    
    def _message_to_dict(self, message: ConversationMessage) -> Dict[str, Any]:
        """메시지를 딕셔너리로 변환 - 빈 값은 제외"""
        result = {
            "message_type": message.type.value,
        }
        
        # 선택적 필드들 - 값이 있는 경우만 포함
        if message.agent:
            result["agent"] = message.agent
            
        content_dict = message.content.to_dict() if message.content else {}
        if content_dict:  # 구조화된 내용
            result["content"] = content_dict
            
        if message.metadata:
            result["metadata"] = message.metadata
            
        if message.bubble_id:
            result["bubble_id"] = message.bubble_id
            
        if message.server_bubble_id:
            result["server_bubble_id"] = message.server_bubble_id
            
        return result
    
    def force_save(self) -> None:
        """강제 저장 실행 - 모든 메시지를 저장 대상으로 추가"""
        if self._auto_save_enabled and self._save_callback:
            # 모든 메시지를 저장 대상으로 추가
            for message in self.conversation:
                if message.bubble_id:
                    self._pending_saves[message.bubble_id] = message
            self._schedule_save()

    def start_request(self, request_id: str) -> None:
        """새로운 요청 시작"""
        self.current_request_id = request_id
    
    def end_request(self) -> None:
        """현재 요청 종료"""
        # 종료 직전 남은 텍스트 버퍼를 플러시 (현재 메시지에 반영)
        self._flush_pending_text_field()
        self.current_request_id = None
        self.current_message = None
        # JSON 파싱 상태 초기화
        self._reset_json_parsing_state()
        # 요청 종료 시 저장
        self.force_save()
    
    def start_human_message(self, text: str) -> ConversationMessage:
        """사용자 메시지 시작"""
        message = ConversationMessage(
            type=MessageType.HUMAN,
            content=MessageContent(answer=text)
        )
        self.conversation.append(message)
        self.current_message = message
        self._mark_dirty()
        return message

    def start_ai_message(self, agent: Optional[str] = None) -> ConversationMessage:
        """AI 메시지 시작"""
        message = ConversationMessage(
            type=MessageType.AI,
            agent=agent,
            server_bubble_id=str(uuid.uuid4())
        )
        self.conversation.append(message)
        self.current_message = message
        # JSON 파싱 상태 초기화
        self._reset_json_parsing_state()
        self._mark_dirty()
        return message

    def accumulate_text(self, delta: str) -> None:
        """현재 메시지에 텍스트 누적 (하위 호환성, answer로 처리)"""
        if self.current_message is not None:
            self.current_message.append_text(delta)
            self._mark_dirty()

    def accumulate_answer(self, delta: str) -> None:
        """현재 메시지에 답변 텍스트 누적 - JSON 스트리밍에서 text 필드만 추출 (개선된 버퍼 기반)"""
        if self.current_message is None:
            return

        # 청크를 버퍼에 누적
        self._json_buffer += delta

        # 버퍼를 소비하면서 다수의 "text" 값을 추출
        idx = 0
        while True:
            # text 필드 시작을 찾지 못했고, 현재 문자열 내부가 아니라면 패턴 검색
            if not self._in_text_field:
                m = re.search(r'"\s*text\s*"\s*:\s*"', self._json_buffer[idx:])
                if not m:
                    # 패턴이 없으면, 불필요한 앞부분은 잘라 메모리 누수 방지
                    # 단, 경계 인식 위해 마지막 16자 정도는 남겨둠
                    keep_tail = 64
                    if len(self._json_buffer) > keep_tail:
                        self._json_buffer = self._json_buffer[-keep_tail:]
                    break
                # 패턴 발견: 해당 위치까지는 버림
                start = idx + m.end()  # 시작 따옴표 직후 인덱스
                # 앞부분 제거하여 이후 인덱싱 단순화
                self._json_buffer = self._json_buffer[start:]
                idx = 0
                # 상태 진입
                self._in_text_field = True
                self._in_string = True
                self._text_content = ""

            # 여기까지 왔다면, 현재 text 문자열 내부를 처리 중
            i = idx
            while i < len(self._json_buffer):
                ch = self._json_buffer[i]
                if self._escape_next:
                    # 공통 이스케이프 시퀀스 디코딩 (\n, \t, \r, \", \\)
                    if ch == 'n':
                        decoded = '\n'
                    elif ch == 't':
                        decoded = '\t'
                    elif ch == 'r':
                        decoded = '\r'
                    elif ch == '"':
                        decoded = '"'
                    elif ch == '\\':
                        decoded = '\\'
                    else:
                        decoded = ch
                    self._text_content += decoded
                    self._escape_next = False
                    i += 1
                    continue
                if ch == '\\':
                    self._escape_next = True
                    i += 1
                    continue
                if ch == '"':
                    # 문자열 종료
                    if self._in_text_field:
                        if self._text_content:
                            prev = self._last_text_emitted or ""
                            full = self._text_content
                            delta = full[len(prev):] if full.startswith(prev) else full
                            if delta:
                                self.current_message.append_answer(delta)
                                self._mark_dirty()
                            self._last_text_emitted = full
                    # 상태 초기화
                    self._in_text_field = False
                    self._in_string = False
                    self._text_content = ""
                    # 종료 따옴표까지 소비하고 다음 패턴 탐색으로 이동
                    i += 1
                    # 남은 버퍼만 유지
                    self._json_buffer = self._json_buffer[i:]
                    idx = 0
                    break
                # 일반 문자 누적
                self._text_content += ch
                i += 1
            else:
                # 닫는 따옴표를 아직 못 만남: 다음 청크에서 이어서 처리
                # 이미 처리한 내용을 재처리하지 않도록 버퍼는 비움(텍스트는 _text_content에 보관 중)
                self._json_buffer = ""
                break

    def accumulate_thought(self, delta: str) -> None:
        """현재 메시지에 추론 텍스트 누적"""
        if self.current_message is not None:
            self.current_message.append_thought(delta)
            self._mark_dirty()

    def accumulate_plan(self, delta: str) -> None:
        """현재 메시지에 계획 텍스트 누적"""
        if self.current_message is not None:
            self.current_message.append_plan(delta)
            self._mark_dirty()

    def accumulate_route(self, delta: str) -> None:
        """현재 메시지에 라우팅 정보 누적"""
        if self.current_message is not None:
            self.current_message.append_route(delta)
            self._mark_dirty()

    def accumulate_decision(self, delta: str) -> None:
        """현재 메시지에 의사결정 내용 누적"""
        if self.current_message is not None:
            self.current_message.append_decision(delta)
            self._mark_dirty()

    def accumulate_error(self, delta: str) -> None:
        """현재 메시지에 에러 메시지 누적"""
        if self.current_message is not None:
            self.current_message.append_error(delta)
            self._mark_dirty()

    def accumulate_completion(self, delta: str) -> None:
        """현재 메시지에 완료 신호 누적"""
        if self.current_message is not None:
            # 완료 표기 전에 남은 텍스트 버퍼를 우선 플러시하여 조기 완료 이슈 방지
            self._flush_pending_text_field()
            self.current_message.append_completion(delta)
            self._mark_dirty()

    def add_tool_call(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_index: int,
        raw_args: str,
        result: Dict[str, Any]
    ) -> None:
        """현재 메시지에 도구 호출 결과 추가"""
        if self.current_message is not None:
            tool_call = ToolCall(
                tool_call_id=tool_call_id,
                tool_name=tool_name,
                tool_index=tool_index,
                raw_args=raw_args,
                result=result
            )
            self.current_message.add_tool_result(tool_call)
            self._mark_dirty()

    def add_metadata(self, key: str, value: Any) -> None:
        """현재 메시지에 메타데이터 추가"""
        if self.current_message is not None and self.current_message.metadata is not None:
            self.current_message.metadata[key] = value
            self._mark_dirty()

    def get_last_message(self) -> Optional[ConversationMessage]:
        """마지막 메시지 반환"""
        return self.conversation[-1] if self.conversation else None

    def get_messages_by_type(self, message_type: MessageType) -> List[ConversationMessage]:
        """특정 타입의 메시지들 반환"""
        return [msg for msg in self.conversation if msg.type == message_type]

    def get_messages_by_agent(self, agent: str) -> List[ConversationMessage]:
        """특정 에이전트의 메시지들 반환"""
        return [msg for msg in self.conversation if msg.agent == agent]

    def to_dict(self) -> Dict[str, Any]:
        """전체 대화를 딕셔너리로 변환"""
        return {
            "conversation": [msg.to_dict() for msg in self.conversation],
            "session_id": self.session_id,
            "current_request_id": self.current_request_id,
            "message_count": len(self.conversation)
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """JSON 문자열로 변환"""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)

    def clear(self) -> None:
        """대화 내용 초기화"""
        self.conversation.clear()
        self.current_message = None

    def add_completion_message(self, text: str = "REQUEST_COMPLETED", metadata: Optional[Dict[str, Any]] = None) -> ConversationMessage:
        """요청 종료를 알리는 완료 메시지 추가 (새로운 MessageType.COMPLETE)"""
        message = ConversationMessage(
            type=MessageType.COMPLETE,
            content=MessageContent(completion=text),
            agent="manager",
            metadata=metadata or {}
        )
        self.conversation.append(message)
        # 완료 메시지는 스트리밍이 없으므로 현재 메시지로 유지할 필요 없음
        self._mark_message_dirty(message)
        return message

    def merge_consecutive_same_type(self) -> None:
        """연속된 같은 타입의 메시지들을 병합"""
        if len(self.conversation) <= 1:
            return
            
        merged = [self.conversation[0]]
        
        for current_msg in self.conversation[1:]:
            last_msg = merged[-1]
            
            # 같은 타입이고 같은 에이전트인 경우 병합
            if (current_msg.type == last_msg.type and 
                current_msg.agent == last_msg.agent and
                current_msg.type == MessageType.AI):
                
                # 내용 병합
                if current_msg.content and last_msg.content:
                    last_msg.content.answer += current_msg.content.answer
                    last_msg.content.thought += current_msg.content.thought
                    last_msg.content.plan += current_msg.content.plan
                    last_msg.content.route += current_msg.content.route
                    last_msg.content.decision += current_msg.content.decision
                    last_msg.content.error += current_msg.content.error
                    
                    # 도구 결과 병합
                    if current_msg.content.tool_results:
                        if last_msg.content.tool_results is None:
                            last_msg.content.tool_results = []
                        last_msg.content.tool_results.extend(current_msg.content.tool_results)
                
                # 메타데이터 병합
                if current_msg.metadata:
                    if last_msg.metadata is None:
                        last_msg.metadata = {}
                    last_msg.metadata.update(current_msg.metadata)
            else:
                merged.append(current_msg)
        
        self.conversation = merged

    def get_summary(self) -> Dict[str, Any]:
        """대화 요약 정보 반환"""
        type_counts = {}
        agent_counts = {}
        
        for msg in self.conversation:
            # 타입별 카운트
            type_name = msg.type.value
            type_counts[type_name] = type_counts.get(type_name, 0) + 1
            
            # 에이전트별 카운트
            if msg.agent:
                agent_counts[msg.agent] = agent_counts.get(msg.agent, 0) + 1
        
        total_text_length = sum(len(msg.text) for msg in self.conversation)
        total_tool_calls = sum(len(msg.tool_results or []) for msg in self.conversation)
        
        return {
            "total_messages": len(self.conversation),
            "total_text_length": total_text_length,
            "total_tool_calls": total_tool_calls,
            "message_types": type_counts,
            "agent_activity": agent_counts,
            "session_id": self.session_id,
            "current_request_id": self.current_request_id
        }
