from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from decimal import Decimal

from ..core.app import PooolifyApp
from ..persistence.db import insert_session
from .security import require_auth


def build_router(app: PooolifyApp) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.get("/healthz")
    def healthz() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/readyz")
    def readyz() -> Dict[str, str]:
        return {"status": "ready"}

    @router.post("/chat")
    async def chat(request: Request, _: None = Depends(require_auth)):
        body = await request.json()
        session_id = body.get("session_id") or "default"
        query = body.get("query") or ""
        if not query:
            raise HTTPException(status_code=400, detail={"error": {"code": "INVALID", "message": "query is required"}})

        # 세션 생성/보장 (동기 I/O는 스레드 오프로딩)
        await asyncio.to_thread(insert_session, session_id)

        # 대화 객체 확보 (비블로킹)
        try:
            conversation = await app.conversation_manager.async_get_or_create_conversation(session_id)
        except AttributeError:
            # 호환성: async 메서드가 없을 경우 동기 접근을 스레드로 오프로딩
            conversation = await asyncio.to_thread(app.conversation_manager.get_or_create_conversation, session_id)

        # 요청 식별자 생성 및 사용자 메시지 시작
        request_id = str(uuid.uuid4())
        conversation.start_request(request_id)
        conversation.start_human_message(query)

        # 비동기로 add_message 실행 (백그라운드에서 처리)
        asyncio.create_task(app.add_message(session_id=session_id, query=query, conversation=conversation, request_id=request_id))

        # 즉시 응답 반환
        return JSONResponse(content={"status": "processing", "session_id": session_id, "request_id": request_id})

    @router.get("/sessions/{session_id}/conversation")
    async def get_session_conversation(session_id: str, _: None = Depends(require_auth)):
        """세션의 대화 데이터 조회 (비블로킹)"""
        # 메모리 히트 시 즉시 반환, 미스 시 DB 접근은 스레드 오프로딩
        conversation_data = app.conversation_manager.get_conversation_data(session_id)
        if conversation_data is None:
            conversation_data = await app.conversation_manager.async_get_conversation_data(session_id)
        if conversation_data is None:
            raise HTTPException(status_code=404, detail={"error": {"code": "NOT_FOUND", "message": "Session not found"}})
        encoded = jsonable_encoder(conversation_data, custom_encoder={Decimal: str})
        return JSONResponse(content=encoded)

 

    return router

