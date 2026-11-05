from fastapi import APIRouter

from app.api.routes import detect, health, live, streams

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(health.router)
api_router.include_router(detect.router)
api_router.include_router(streams.router)
api_router.include_router(live.router)
