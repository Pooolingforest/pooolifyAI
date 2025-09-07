from __future__ import annotations

import os
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from ..config import settings


def get_bearer_token(req: Request) -> Optional[str]:
    auth = req.headers.get("authorization") or req.headers.get("Authorization")
    if not auth or not auth.lower().startswith("bearer "):
        return None
    return auth.split(" ", 1)[1].strip()


def require_auth(token: Optional[str] = Depends(get_bearer_token)) -> None:
    expected = settings.api_token
    if expected is None:
        # If no token configured, allow in dev only
        if settings.app_env != "prod":
            return
    if not token or token != expected:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail={"error": {"code": "UNAUTHORIZED", "message": "Invalid or missing token"}})

