from typing import Annotated

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.services.models import ModelRegistry
from app.services.streams import StreamManager


def get_models(request: Request) -> ModelRegistry:
    return request.app.state.models


def get_stream_manager(request: Request) -> StreamManager:
    return request.app.state.streams


SettingsDep = Annotated[Settings, Depends(get_settings)]
ModelsDep = Annotated[ModelRegistry, Depends(get_models)]
StreamsDep = Annotated[StreamManager, Depends(get_stream_manager)]
