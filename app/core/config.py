"""Application settings (env vars / .env file)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=BASE_DIR / ".env", extra="ignore")

    app_name: str = "Vision Ops — Object Detection"
    log_level: str = "INFO"

    # --- Detector -----------------------------------------------------------
    # Live video detector. yolo11s finds ~7 pts more people than yolo11n (COCO val, our eval)
    # at ~2x the cost; falls back to yolo11n.onnx if it has not been exported.
    model_path: Path = BASE_DIR / "models" / "yolo11s.onnx"
    input_size: int = 640
    conf_threshold: float = Field(0.35, ge=0.01, le=0.99)
    nms_threshold: float = Field(0.45, ge=0.01, le=0.99)
    use_cuda: bool = False

    # --- Accuracy for photos (video keeps the fast MODEL_PATH above) ----------
    image_model: str = "yolo11x.onnx"       # missing file -> photos use MODEL_PATH
    # Measured on COCO val (YOLO11x): tiles=2 finds more small people (+6.5 pts recall) but adds
    # false alarms (-9 pts precision) at 5x the time, so it is opt-in via mode=max_recall.
    image_tiles: int = Field(1, ge=1, le=4)
    max_recall_tiles: int = Field(2, ge=1, le=4)
    max_recall_conf: float = Field(0.25, ge=0.05, le=0.95)
    open_vocab_model: str = "yoloe-11l-seg-pf.onnx"   # 4,585 object types
    open_vocab_labels: str = "yoloe-11l-seg-pf_labels.json"
    enable_open_vocab: bool = True
    open_vocab_conf: float = Field(0.35, ge=0.05, le=0.95)
    open_vocab_in_video: bool = False       # ~3 s/frame on CPU - photos only by default

    # --- Person & scene understanding (each is optional: missing file = feature off)
    models_dir: Path = BASE_DIR / "models"
    pose_model: str = "yolo11n-pose.onnx"
    face_model: str = "face_yunet.onnx"
    genderage_model: str = "face_genderage.onnx"
    expression_model: str = "face_expression.onnx"
    scene_model: str = "segformer_ade.onnx"
    scene_labels: str = "segformer_ade_labels.json"
    enable_pose: bool = True
    enable_faces: bool = True
    enable_demographics: bool = True   # perceived gender + estimated age
    enable_scene: bool = True
    scene_overlay: bool = True         # tint & label every scene region on the video
    scene_interval_s: float = 2.0      # video: re-segment the scene at most this often
    object_every_n: int = 2            # video: run the 80-class detector every N frames
    face_every_n: int = 3              # video: run face models every N frames

    # --- Fire / smoke --------------------------------------------------------
    enable_hazards: bool = False  # optional fire/smoke add-on; the focus is people & objects
    hazard_model_path: Path | None = None  # optional YOLO ONNX trained on fire/smoke
    hazard_labels: list[str] = ["fire", "smoke"]

    # --- Streaming -----------------------------------------------------------
    stream_max_width: int = 1280
    jpeg_quality: int = Field(80, ge=30, le=95)
    max_streams: int = 4
    allow_device_sources: bool = True  # webcams, e.g. source "0"
    allow_url_sources: bool = False    # rtsp:// / http:// cameras – enable deliberately

    # --- Uploads -------------------------------------------------------------
    upload_dir: Path = BASE_DIR / "data" / "uploads"
    max_image_mb: int = 15
    max_video_mb: int = 300


@lru_cache
def get_settings() -> Settings:
    return Settings()
