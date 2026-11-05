"""Load every model once at startup. Optional models degrade gracefully: if a file is
missing the matching feature is switched off and the reason is logged / reported."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from app.core.config import Settings
from app.ml.detector import ModelNotFoundError, YoloDetector
from app.ml.face import FaceAnalyzer
from app.ml.pose import PoseEstimator
from app.ml.scene import SceneSegmenter

log = logging.getLogger(__name__)
EXPORTED = {"accurate photo model", "open vocabulary"}  # not published as ONNX


@dataclass(slots=True)
class ModelRegistry:
    detector: YoloDetector
    image_detector: YoloDetector | None = None   # accurate model for photos
    open_vocab: YoloDetector | None = None       # thousands of object types
    pose: PoseEstimator | None = None
    faces: FaceAnalyzer | None = None
    scene: SceneSegmenter | None = None
    hazard: YoloDetector | None = None
    disabled: dict[str, str] = field(default_factory=dict)  # feature -> reason

    @property
    def features(self) -> dict[str, bool]:
        return {
            "objects": True,
            "accurate_photo_model": self.image_detector is not None,
            "open_vocabulary": self.open_vocab is not None,
            "pose_and_actions": self.pose is not None,
            "faces": self.faces is not None,
            "gender_age": bool(self.faces and self.faces.has_demographics),
            "scene": self.scene is not None,
            "hazard_model": self.hazard is not None,
        }


def load_models(settings: Settings) -> ModelRegistry:
    d = settings.models_dir
    model_path = settings.model_path
    fallback = d / "yolo11n.onnx"
    if not model_path.is_file() and fallback.is_file():
        log.warning("%s not found - using %s for video (run scripts/export_models.py for better accuracy)",
                    model_path.name, fallback.name)
        model_path = fallback
    detector = YoloDetector(
        model_path,
        input_size=settings.input_size,
        conf_threshold=settings.conf_threshold,
        nms_threshold=settings.nms_threshold,
        use_cuda=settings.use_cuda,
    )
    detector.warmup()
    reg = ModelRegistry(detector=detector)
    log.info("Object detector: %s", detector.name)

    def optional(feature: str, enabled: bool, factory):
        if not enabled:
            reg.disabled[feature] = "disabled in settings"
            return None
        try:
            model = factory()
            log.info("Loaded %s", feature)
            return model
        except ModelNotFoundError as exc:
            script = "export_models.py" if feature in EXPORTED else "download_model.py"
            reg.disabled[feature] = f"{exc} - run: python scripts/{script}"
            log.warning("%s disabled: %s", feature, exc)
            return None

    def make_image_detector() -> YoloDetector:
        det = YoloDetector(d / settings.image_model, input_size=settings.input_size,
                           conf_threshold=settings.conf_threshold, nms_threshold=settings.nms_threshold,
                           use_cuda=settings.use_cuda)
        det.warmup()
        return det

    def make_open_vocab() -> YoloDetector:
        labels_path = d / settings.open_vocab_labels
        if not labels_path.is_file():
            raise ModelNotFoundError(f"Labels not found at '{labels_path}'")
        labels = json.loads(labels_path.read_text(encoding="utf-8"))
        return YoloDetector(d / settings.open_vocab_model, input_size=settings.input_size,
                            conf_threshold=settings.open_vocab_conf, nms_threshold=settings.nms_threshold,
                            labels=labels, use_cuda=settings.use_cuda)

    reg.image_detector = optional("accurate photo model", True, make_image_detector)
    reg.open_vocab = optional("open vocabulary", settings.enable_open_vocab, make_open_vocab)
    reg.pose = optional("pose", settings.enable_pose, lambda: PoseEstimator(d / settings.pose_model))

    def make_faces() -> FaceAnalyzer:
        demo = settings.enable_demographics
        genderage = d / settings.genderage_model
        expr = d / settings.expression_model
        return FaceAnalyzer(
            d / settings.face_model,
            genderage_path=genderage if demo and genderage.is_file() else None,
            expression_path=expr if expr.is_file() else None,
        )

    reg.faces = optional("faces", settings.enable_faces, make_faces)
    reg.scene = optional(
        "scene", settings.enable_scene,
        lambda: SceneSegmenter(d / settings.scene_model, d / settings.scene_labels),
    )
    if settings.enable_hazards and settings.hazard_model_path:
        reg.hazard = YoloDetector(
            settings.hazard_model_path,
            input_size=settings.input_size,
            conf_threshold=settings.conf_threshold,
            labels=settings.hazard_labels,
        )
    return reg
