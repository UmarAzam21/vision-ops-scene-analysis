from pydantic import BaseModel, Field

from app.schemas.detection import Summary


class SourceRequest(BaseModel):
    source: str = Field(
        ..., min_length=1, max_length=512,
        description="Webcam index (e.g. '0') or rtsp:// / http(s):// URL",
        examples=["0", "rtsp://user:pass@192.168.1.10:554/stream1"],
    )


class StreamCreated(BaseModel):
    stream_id: str
    video_url: str
    stats_url: str


class StreamStatsOut(BaseModel):
    stream_id: str
    running: bool
    frames: int
    fps: float
    summary: Summary | None
    error: str | None = None
