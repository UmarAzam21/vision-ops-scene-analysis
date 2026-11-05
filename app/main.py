"""FastAPI application factory."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.router import api_router
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.models import load_models
from app.services.streams import StreamManager

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger("app")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        models = load_models(settings)
        app.state.models = models
        app.state.streams = StreamManager(models, settings)
        yield
        app.state.streams.shutdown()

    app = FastAPI(
        title=settings.app_name,
        version="2.0.0",
        description=(
            "Real-time scene understanding with OpenCV DNN: objects, people (pose, actions, "
            "perceived gender/age, expression, clothing), scene segmentation, crowd density "
            "and fire/smoke."
        ),
        lifespan=lifespan,
    )
    app.include_router(api_router)
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
