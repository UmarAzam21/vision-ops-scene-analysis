"""Human pose estimation (YOLO11-pose ONNX, 17 COCO keypoints) on OpenCV DNN."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import Box, ModelNotFoundError, letterbox

KEYPOINTS = (
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
)
KP = {name: i for i, name in enumerate(KEYPOINTS)}

SKELETON = (
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),        # arms
    (5, 11), (6, 12), (11, 12),                      # torso
    (11, 13), (13, 15), (12, 14), (14, 16),          # legs
    (0, 1), (0, 2), (1, 3), (2, 4),                  # face
)


@dataclass(frozen=True, slots=True)
class Pose:
    box: Box
    confidence: float
    keypoints: np.ndarray  # (17, 3): x, y, visibility – original image pixels

    def point(self, name: str, min_conf: float = 0.5) -> tuple[float, float] | None:
        x, y, c = self.keypoints[KP[name]]
        return (float(x), float(y)) if c >= min_conf else None

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]


class PoseEstimator:
    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.4,
        nms_threshold: float = 0.45,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(f"Pose model not found at '{model_path}'")
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._lock = threading.Lock()

    def detect(self, image: np.ndarray) -> list[Pose]:
        padded, info = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=True)
        with self._lock:
            self._net.setInput(blob)
            out = self._net.forward()

        preds = np.squeeze(out, 0)
        if preds.shape[0] == 56 and preds.shape[1] != 56:  # (56, N) -> (N, 56)
            preds = preds.T
        preds = preds[preds[:, 4] >= self.conf_threshold]
        if len(preds) == 0:
            return []

        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x = (cx - bw / 2 - info.pad_x) / info.scale
        y = (cy - bh / 2 - info.pad_y) / info.scale
        boxes = np.stack([x, y, bw / info.scale, bh / info.scale], axis=1)
        scores = preds[:, 4]
        kpts = preds[:, 5:].reshape(-1, 17, 3).copy()
        kpts[..., 0] = (kpts[..., 0] - info.pad_x) / info.scale
        kpts[..., 1] = (kpts[..., 1] - info.pad_y) / info.scale

        keep = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf_threshold, self.nms_threshold)
        h, w = image.shape[:2]
        poses: list[Pose] = []
        for i in np.asarray(keep).reshape(-1):
            bx, by, bw_, bh_ = boxes[i]
            box = (int(max(0, bx)), int(max(0, by)), int(min(w - 1, bx + bw_)), int(min(h - 1, by + bh_)))
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            poses.append(Pose(box, float(scores[i]), kpts[i]))
        poses.sort(key=lambda p: p.confidence, reverse=True)
        return poses
