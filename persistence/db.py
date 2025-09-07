import logging
from contextlib import contextmanager
from typing import Any, Dict, Generator, Iterable, List, Optional
from fastapi.encoders import jsonable_encoder
from decimal import Decimal

import psycopg
from psycopg.types.json import Json

from ..config import settings
from .schema import ALL_DDL


logger = logging.getLogger(__name__)


def _require_db_url() -> str:
    if not settings.database_url:
        raise RuntimeError("DATABASE_URL not configured")
    return settings.database_url


def _has_db() -> bool:
    return bool(settings.database_url)


def init_db() -> None:
    """Create tables if they do not exist."""
    if not _has_db():
        logger.info("DB not configured; init skipped (dev mode)")
        return
    dsn = _require_db_url()
    with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
        with conn.cursor() as cur:
            for ddl in ALL_DDL:
                cur.execute(ddl)
        conn.commit()
    logger.info("DB initialized (DDL ensured)")


@contextmanager
def get_conn() -> Generator[psycopg.Connection, None, None]:  # type: ignore[name-defined]
    dsn = _require_db_url()
    with psycopg.connect(dsn) as conn:  # type: ignore[arg-type]
        yield conn


def insert_session(session_id: str, metadata: Optional[Dict[str, Any]] = None) -> None:
    if not _has_db():
        logger.debug("insert_session skipped; DB not configured")
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO chat_sessions (id, metadata) VALUES (%s, %s) ON CONFLICT (id) DO NOTHING",
                (session_id, Json(metadata) if metadata is not None else None),
            )
        conn.commit()








def insert_ai_call_cost(
    *,
    request_id: str,
    session_id: Optional[str],
    provider: str,
    model: str,
    purpose: str,
    input_tokens: Optional[int],
    output_tokens: Optional[int],
    total_tokens: Optional[int],
    cache_input_tokens: Optional[int] = None,
) -> None:
    if not _has_db():
        logger.debug("insert_ai_call_cost skipped; DB not configured")
        return
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO ai_call_costs
                (request_id, session_id, provider, model, purpose, input_tokens, output_tokens, total_tokens, cache_input_tokens)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s)
                """,
                (
                    request_id,
                    session_id,
                    provider,
                    model,
                    purpose,
                    input_tokens,
                    output_tokens,
                    total_tokens,
                    cache_input_tokens,
                ),
            )
        conn.commit()


def upsert_conversation_message(
    session_id: str,
    message_id: str,
    message_data: Dict[str, Any],
) -> None:
    """대화 메시지를 DB에 저장 또는 업데이트"""
    if not _has_db():
        logger.debug("upsert_conversation_message skipped; DB not configured")
        return
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            # 메시지 저장/업데이트
            cur.execute(
                """
                INSERT INTO conversation_messages 
                (session_id, message_id, message_data, updated_at)
                VALUES (%s, %s, %s, now())
                ON CONFLICT (session_id, message_id) DO UPDATE SET
                    message_data = EXCLUDED.message_data,
                    updated_at = now()
                """,
                (
                    session_id,
                    message_id,
                    Json(jsonable_encoder(message_data, custom_encoder={Decimal: str})),
                ),
            )
            
            # 세션 정보 업데이트
            cur.execute(
                """
                UPDATE chat_sessions SET
                    message_count = (
                        SELECT COUNT(*) FROM conversation_messages 
                        WHERE session_id = %s
                    ),
                    last_message_at = now(),
                    updated_at = now()
                WHERE id = %s
                """,
                (session_id, session_id),
            )
        conn.commit()


def fetch_conversation_messages(session_id: str) -> List[Dict[str, Any]]:
    """세션의 모든 대화 메시지를 DB에서 조회"""
    if not _has_db():
        return []
    
    messages = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT message_id, message_data, created_at, updated_at
                FROM conversation_messages 
                WHERE session_id = %s 
                ORDER BY created_at ASC
                """,
                (session_id,),
            )
            
            for row in cur.fetchall():
                message_data = row[1]  # JSONB 데이터
                # 메타데이터 추가
                message_data["message_id"] = row[0]
                message_data["created_at"] = row[2]
                message_data["updated_at"] = row[3]
                messages.append(message_data)
    
    return messages


def fetch_conversation_session(session_id: str) -> Optional[Dict[str, Any]]:
    """세션 정보를 DB에서 조회"""
    if not _has_db():
        return None
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, message_count, last_message_at, created_at, updated_at, metadata FROM chat_sessions WHERE id = %s",
                (session_id,),
            )
            row = cur.fetchone()
            if row:
                return {
                    "session_id": row[0],
                    "message_count": row[1],
                    "last_message_at": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "metadata": row[5],
                }
    return None


def delete_conversation(session_id: str) -> bool:
    """세션의 대화 데이터를 DB에서 삭제"""
    if not _has_db():
        return False
    
    with get_conn() as conn:
        with conn.cursor() as cur:
            # 세션 삭제 (CASCADE로 메시지들도 자동 삭제됨)
            cur.execute("DELETE FROM chat_sessions WHERE id = %s", (session_id,))
            deleted_count = cur.rowcount
            
        conn.commit()
    
    return deleted_count > 0


def list_conversation_sessions() -> List[Dict[str, Any]]:
    """모든 대화 세션 목록 조회"""
    if not _has_db():
        return []
    
    sessions = []
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, message_count, last_message_at, created_at, updated_at, metadata
                FROM chat_sessions 
                ORDER BY last_message_at DESC NULLS LAST, updated_at DESC
                """
            )
            for row in cur.fetchall():
                sessions.append({
                    "session_id": row[0],
                    "message_count": row[1],
                    "last_message_at": row[2],
                    "created_at": row[3],
                    "updated_at": row[4],
                    "metadata": row[5],
                })
    
    return sessions

