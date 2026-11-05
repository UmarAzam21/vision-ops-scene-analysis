from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    model: str
    input_size: int
    opencv_version: str
    hazard_mode: str
    active_streams: int
    features: dict[str, bool]
    disabled: dict[str, str]
