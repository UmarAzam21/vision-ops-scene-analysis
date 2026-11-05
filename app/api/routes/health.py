import cv2
from fastapi import APIRouter

from app.api.deps import ModelsDep, SettingsDep, StreamsDep
from app.schemas.health import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health(models: ModelsDep, streams: StreamsDep, settings: SettingsDep) -> HealthResponse:
    if not settings.enable_hazards:
        hazard_mode = "disabled"
    else:
        hazard_mode = "model" if models.hazard else "heuristic"
    return HealthResponse(
        status="ok",
        model=models.detector.name,
        input_size=models.detector.input_size,
        opencv_version=cv2.__version__,
        hazard_mode=hazard_mode,
        active_streams=streams.count,
        features=models.features,
        disabled=models.disabled,
    )
