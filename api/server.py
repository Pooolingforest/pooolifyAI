from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..persistence.db import init_db
from .routes import build_router
from ..core.app import PooolifyApp


logger = logging.getLogger(__name__)


def create_fastapi_app(core: Optional[PooolifyApp] = None) -> FastAPI:
    core = core or PooolifyApp()
    fapp = FastAPI(title="pooolify API", version="0.1.0")

    # CORS
    cors = getattr(core, "cors", None)
    if cors:
        fapp.add_middleware(
            CORSMiddleware,
            allow_origins=cors.allow_origins or ["*"],
            allow_origin_regex=cors.allow_origin_regex,
            allow_methods=cors.allow_methods or ["*"],
            allow_headers=cors.allow_headers or ["*"],
            expose_headers=cors.expose_headers or None,
            allow_credentials=bool(cors.allow_credentials),
            max_age=cors.max_age or 600,
        )

    @fapp.on_event("startup")
    def _startup() -> None:
        try:
            init_db()
        except Exception as e:  # noqa: BLE001
            logger.warning("DB init failed: %s", e)

    fapp.include_router(build_router(core))
    return fapp

