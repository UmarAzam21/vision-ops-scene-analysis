"""Fire & smoke detection.

Two interchangeable strategies share one interface (`detect(frame) -> list[Hazard]`):

* `HeuristicHazardDetector` – classical OpenCV pipeline (colour rules + motion +
  temporal persistence). Works out of the box, no extra weights. Good for demos and
  as a fallback, but it is a heuristic: tune thresholds for your cameras.
* `ModelHazardDetector` – any YOLOv8/11 ONNX model trained on fire/smoke classes
  (set HAZARD_MODEL_PATH). Recommended for production.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import cv2
import numpy as np

from app.ml.detector import Box, YoloDetector

HazardKind = Literal["fire", "smoke"]


@dataclass(frozen=True, slots=True)
class Hazard:
    kind: HazardKind
    confidence: float
    box: Box


class HazardDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Hazard]: ...


class ModelHazardDetector:
    """Adapter that turns a fire/smoke YOLO model into `Hazard` objects."""

    def __init__(self, detector: YoloDetector) -> None:
        self._detector = detector

    def detect(self, frame: np.ndarray) -> list[Hazard]:
        hazards: list[Hazard] = []
        for det in self._detector.detect(frame):
            label = det.label.lower()
            if label in ("fire", "flame"):
                hazards.append(Hazard("fire", det.confidence, det.box))
            elif label == "smoke":
                hazards.append(Hazard("smoke", det.confidence, det.box))
        return hazards


class HeuristicHazardDetector:
    """Colour + motion + persistence fire/smoke detector (stateful, one per stream).

    Fire  : bright, red-dominant pixels (R > G > B) with a strong Cr-Cb chroma gap.
    Smoke : low-saturation grey haze that is *moving* and *texture-poor* (smoke blurs
            what is behind it). Smoke needs video – it is skipped for still images and
            while the camera itself is moving (motion gate would be meaningless).
    Both masks are averaged over time so a region must persist for several frames.
    """

    def __init__(
        self,
        *,
        temporal: bool = True,
        proc_width: int = 480,
        min_area_ratio: float = 0.0015,
        fire_alpha: float = 0.35,
        smoke_alpha: float = 0.15,
        camera_motion_limit: float = 0.35,
    ) -> None:
        self.temporal = temporal
        self.proc_width = proc_width
        self.min_area_ratio = min_area_ratio
        self.fire_alpha = fire_alpha if temporal else 1.0
        self.smoke_alpha = smoke_alpha
        self.camera_motion_limit = camera_motion_limit

        self._bg = (
            cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=24, detectShadows=False)
            if temporal
            else None
        )
        self._fire_acc: np.ndarray | None = None
        self._smoke_acc: np.ndarray | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    # ---------------------------------------------------------------- masks
    @staticmethod
    def fire_mask(bgr: np.ndarray) -> np.ndarray:
        b, g, r = (c.astype(np.int16) for c in cv2.split(bgr))
        ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = (c.astype(np.int16) for c in cv2.split(ycrcb))
        mask = (
            (r > 180)
            & (r > g)
            & (g > b)
            & (y > cb)
            & ((cr - cb) > 40)
            & ((r - b) > 70)
        )
        return mask.astype(np.uint8) * 255

    @staticmethod
    def smoke_mask(bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        s, v = hsv[..., 1], hsv[..., 2]
        grey = (s < 45) & (v > 80) & (v < 225)
        # Texture-poor: smoke softens edges behind it.
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        lap = np.abs(cv2.Laplacian(gray, cv2.CV_16S, ksize=3))
        lap = cv2.blur(lap.astype(np.float32), (9, 9))
        smooth = lap < 18
        return (grey & smooth).astype(np.uint8) * 255

    # ---------------------------------------------------------------- public
    def detect(self, frame: np.ndarray) -> list[Hazard]:
        h, w = frame.shape[:2]
        scale = min(1.0, self.proc_width / w)
        small = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (3, 3), 0)

        fire = self._clean(self.fire_mask(small))
        smoke: np.ndarray | None = None

        if self._bg is not None:
            motion = self._bg.apply(small)
            motion = cv2.dilate(motion, self._kernel, iterations=2)
            motion_ratio = float(np.count_nonzero(motion)) / motion.size
            camera_static = motion_ratio < self.camera_motion_limit
            if camera_static:
                smoke = self._clean(cv2.bitwise_and(self.smoke_mask(small), motion))
            else:
                smoke = np.zeros_like(fire)

        fire_conf = self._accumulate("_fire_acc", fire, self.fire_alpha)
        hazards = self._regions(fire_conf, "fire", 1 / scale)

        if smoke is not None:
            smoke_conf = self._accumulate("_smoke_acc", smoke, self.smoke_alpha)
            hazards += self._regions(smoke_conf, "smoke", 1 / scale, min_area_mult=4.0)
        return hazards

    # ---------------------------------------------------------------- helpers
    def _clean(self, mask: np.ndarray) -> np.ndarray:
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel, iterations=2)

    def _accumulate(self, attr: str, mask: np.ndarray, alpha: float) -> np.ndarray:
        current = mask.astype(np.float32) / 255.0
        acc: np.ndarray | None = getattr(self, attr)
        if acc is None or acc.shape != current.shape or alpha >= 1.0:
            acc = current
        else:
            acc = cv2.addWeighted(acc, 1 - alpha, current, alpha, 0)
        setattr(self, attr, acc)
        return acc

    def _regions(
        self, conf_map: np.ndarray, kind: HazardKind, inv_scale: float, min_area_mult: float = 1.0
    ) -> list[Hazard]:
        binary = (conf_map > 0.5).astype(np.uint8)
        min_area = self.min_area_ratio * min_area_mult * binary.size
        # Bridge small gaps so one fire / one plume yields one region, not fragments.
        k = max(3, int(binary.shape[1] * 0.04)) | 1
        bridged = cv2.dilate(binary, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        n, labels = cv2.connectedComponents(bridged, connectivity=8)

        hazards: list[Hazard] = []
        for i in range(1, n):
            ys, xs = np.nonzero((labels == i) & (binary > 0))
            area = len(xs)
            if area < min_area:
                continue
            x, y = int(xs.min()), int(ys.min())
            bw, bh = int(xs.max()) - x + 1, int(ys.max()) - y + 1
            region = conf_map[y:y + bh, x:x + bw]
            fill = area / float(bw * bh)
            confidence = float(np.clip(region[region > 0.5].mean() * (0.6 + 0.4 * fill), 0, 0.99))
            box = (
                int(x * inv_scale),
                int(y * inv_scale),
                int((x + bw) * inv_scale),
                int((y + bh) * inv_scale),
            )
            hazards.append(Hazard(kind, confidence, box))
        hazards.sort(key=lambda hz: hz.confidence, reverse=True)
        return hazards
