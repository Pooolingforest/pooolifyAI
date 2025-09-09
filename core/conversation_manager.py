from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional
from datetime import datetime
import asyncio

from ..persistence import db as dbops
from .conversation import ConversationAccumulator, MessageType, ConversationMessage, ToolCall, MessageContent


logger = logging.getLogger(__name__)


class ConversationManager:
    """대화 관리를 담당하는 클래스"""
    
    def __init__(self):
        # 세션별 대화 저장소
        self._session_conversations: Dict[str, ConversationAccumulator] = {}
    
    def get_or_create_conversation(self, session_id: str) -> ConversationAccumulator:
        """세션별 대화 누적기를 가져오거나 생성"""
        if session_id not in self._session_conversations:
            # DB에서 기존 대화 로드 시도
            conversation = self._load_conversation_from_db(session_id)
            if conversation is None:
                # 새로운 대화 생성
                conversation = ConversationAccumulator(session_id=session_id)
            
            # 자동 저장 기능 활성화 - 블록 단위 저장 (DB I/O 비동기 오프로딩)
            def save_message_block(session_id: str, message_id: str, message_data: Dict[str, Any]) -> None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(asyncio.to_thread(
                        dbops.upsert_conversation_message,
                        session_id,
                        message_id,
                        message_data,
                    ))
                except RuntimeError:
                    # 실행 중인 이벤트 루프가 없으면 동기 호출 (테스트/동기 컨텍스트)
                    dbops.upsert_conversation_message(
                        session_id=session_id,
                        message_id=message_id,
                        message_data=message_data,
                    )
            
            conversation.enable_auto_save(
                save_callback=save_message_block,
                interval=2.0  # 2초마다 저장
            )
            self._session_conversations[session_id] = conversation
        return self._session_conversations[session_id]

    async def async_get_or_create_conversation(self, session_id: str) -> ConversationAccumulator:
        """세션별 대화 누적기를 가져오거나 생성 (DB 접근은 스레드 오프로딩)"""
        if session_id not in self._session_conversations:
            # DB에서 기존 대화 로드 시도 (비동기 오프로딩)
            try:
                conversation = await asyncio.to_thread(self._load_conversation_from_db, session_id)
            except Exception:
                conversation = None
            if conversation is None:
                conversation = ConversationAccumulator(session_id=session_id)

            # 자동 저장 기능 활성화 - 블록 단위 저장 (DB I/O 비동기 오프로딩)
            def save_message_block(session_id: str, message_id: str, message_data: Dict[str, Any]) -> None:
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(asyncio.to_thread(
                        dbops.upsert_conversation_message,
                        session_id,
                        message_id,
                        message_data,
                    ))
                except RuntimeError:
                    dbops.upsert_conversation_message(
                        session_id=session_id,
                        message_id=message_id,
                        message_data=message_data,
                    )

            conversation.enable_auto_save(
                save_callback=save_message_block,
                interval=2.0
            )
            self._session_conversations[session_id] = conversation
        return self._session_conversations[session_id]
    
    def _load_conversation_from_db(self, session_id: str) -> Optional[ConversationAccumulator]:
        """DB에서 대화 데이터를 로드하여 ConversationAccumulator 생성"""
        try:
            # 메시지들을 블록 단위로 로드
            messages_data = dbops.fetch_conversation_messages(session_id)
            if not messages_data:
                return None
                
            # ConversationAccumulator 재구성
            conversation = ConversationAccumulator(session_id=session_id)
            
            # 대화 메시지들 복원
            for msg_data in messages_data:
                # content 복원
                content = None
                if msg_data.get("content"):
                    content_data = msg_data["content"]
                    content = MessageContent(
                        answer=content_data.get("answer", ""),
                        thought=content_data.get("thought", ""),
                        plan=content_data.get("plan", ""),
                        route=content_data.get("route", ""),
                        decision=content_data.get("decision", ""),
                        error=content_data.get("error", ""),
                        completion=content_data.get("completion", "")
                    )
                elif msg_data.get("text_content"):
                    # 하위 호환성: 기존 text_content를 answer로 변환
                    content = MessageContent(answer=msg_data["text_content"])
                
                message = ConversationMessage(
                    type=MessageType(msg_data.get("message_type", MessageType.AI)),
                    content=content,
                    bubble_id=msg_data.get("bubble_id"),
                    server_bubble_id=msg_data.get("server_bubble_id"),
                    agent=msg_data.get("agent"),
                    metadata=msg_data.get("metadata", {})
                )
                # DB 생성 시각을 메시지 타임스탬프로 매핑
                created_ts = msg_data.get("created_at")
                if isinstance(created_ts, datetime):
                    message.timestamp = created_ts
                
                # 도구 호출 결과 복원
                if msg_data.get("tool_results"):
                    for tool_data in msg_data["tool_results"]:
                        tool_call = ToolCall(
                            tool_call_id=tool_data.get("toolCallId", ""),
                            tool_name=tool_data.get("toolName", ""),
                            tool_index=tool_data.get("toolIndex", 0),
                            raw_args=tool_data.get("rawArgs", ""),
                            result=tool_data.get("result", {})
                        )
                        message.add_tool_result(tool_call)
                
                conversation.conversation.append(message)
            
            logger.info(f"Loaded conversation for session {session_id} with {len(conversation.conversation)} messages")
            return conversation
                
        except Exception as e:
            logger.error(f"Failed to load conversation for session {session_id}: {e}")
        
        return None
    
    def get_conversation_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션의 대화 데이터 반환"""
        # 메모리에 있는 세션 먼저 확인
        conversation = self._session_conversations.get(session_id)
        if conversation:
            return conversation.to_dict()
        
        # 메모리에 없으면 DB에서 로드 시도
        conversation = self._load_conversation_from_db(session_id)
        if conversation:
            # 메모리에 캐시하지 않고 바로 반환 (읽기 전용)
            return conversation.to_dict()
        
        return None

    async def async_get_conversation_data(self, session_id: str) -> Optional[Dict[str, Any]]:
        """세션의 대화 데이터 반환 (DB 접근 오프로딩)"""
        conversation = self._session_conversations.get(session_id)
        if conversation:
            return conversation.to_dict()
        # 메모리에 없으면 DB에서 로드 (오프로딩)
        conversation = await asyncio.to_thread(self._load_conversation_from_db, session_id)
        if conversation:
            return conversation.to_dict()
        return None
    
    def clear_session_conversation(self, session_id: str) -> None:
        """세션의 대화 데이터 삭제"""
        if session_id in self._session_conversations:
            conversation = self._session_conversations[session_id]
            # 자동 저장 비활성화
            conversation.disable_auto_save()
            del self._session_conversations[session_id]
            # DB에서도 삭제
            dbops.delete_conversation(session_id)
    
    def get_all_session_ids(self) -> List[str]:
        """모든 활성 세션 ID 반환"""
        return list(self._session_conversations.keys())
    
    def has_session(self, session_id: str) -> bool:
        """세션이 메모리에 로드되어 있는지 확인"""
        return session_id in self._session_conversations
    
    def get_session_count(self) -> int:
        """현재 메모리에 로드된 세션 수 반환"""
        return len(self._session_conversations)
    
    def remove_from_memory(self, session_id: str) -> bool:
        """메모리에서 conversation 삭제 (DB는 유지)"""
        if session_id in self._session_conversations:
            conversation = self._session_conversations[session_id]
            # 마지막 저장 수행
            conversation.force_save()
            # 자동 저장 비활성화
            conversation.disable_auto_save()
            # 메모리에서 삭제
            del self._session_conversations[session_id]
            logger.info(f"Removed conversation for session {session_id} from memory")
            return True
        return False
    
    def cleanup_inactive_sessions(self, max_sessions: int = 100) -> int:
        """비활성 세션들을 정리 (메모리 관리)"""
        if len(self._session_conversations) <= max_sessions:
            return 0
        
        # 가장 오래된 세션들부터 정리
        # 실제로는 last_activity 같은 필드를 추가해서 관리하는 것이 좋음
        sessions_to_remove = list(self._session_conversations.keys())[:-max_sessions]
        
        removed_count = 0
        for session_id in sessions_to_remove:
            if self.remove_from_memory(session_id):
                removed_count += 1
        
        logger.info(f"Cleaned up {removed_count} inactive sessions")
        return removed_count
