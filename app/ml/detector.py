"""YOLO (v8 / 11) object detector running on the OpenCV DNN module.

The model is a plain ONNX export, so the only runtime dependency is OpenCV —
no PyTorch / Ultralytics required at inference time.
"""

from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ml.labels import COCO_LABELS

Box = tuple[int, int, int, int]  # x1, y1, x2, y2 (pixels, original image space)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the ONNX weights are missing."""


@dataclass(frozen=True, slots=True)
class Detection:
    class_id: int
    label: str
    confidence: float
    box: Box

    @property
    def width(self) -> int:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> int:
        return self.box[3] - self.box[1]

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


@dataclass(frozen=True, slots=True)
class LetterboxInfo:
    scale: float
    pad_x: int
    pad_y: int


def letterbox(image: np.ndarray, size: int, fill: int = 114) -> tuple[np.ndarray, LetterboxInfo]:
    """Resize keeping aspect ratio and pad to a `size` x `size` square (YOLO convention)."""
    h, w = image.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = round(w * scale), round(h * scale)
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)

    canvas = np.full((size, size, 3), fill, dtype=np.uint8)
    pad_x, pad_y = (size - new_w) // 2, (size - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, LetterboxInfo(scale, pad_x, pad_y)


def merge_detections(dets: list[Detection], iou_threshold: float = 0.5) -> list[Detection]:
    """Class-aware greedy NMS over detections from several passes (full image + tiles).
    Deliberately IoU-only: a containment rule would delete real people standing in front
    of others in crowds. Tile fragments are already removed at the tile borders."""
    dets = sorted(dets, key=lambda d: d.confidence, reverse=True)
    kept: list[Detection] = []
    for d in dets:
        a = d.box
        area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
        dup = False
        for k in kept:
            if k.class_id != d.class_id:
                continue
            b = k.box
            inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
            area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
            if inter / (area_a + area_b - inter) > iou_threshold:
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


class YoloDetector:
    """Thread-safe wrapper around a YOLOv8/YOLO11 ONNX model loaded with `cv2.dnn`."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        input_size: int = 640,
        conf_threshold: float = 0.35,
        nms_threshold: float = 0.45,
        labels: Sequence[str] = COCO_LABELS,
        use_cuda: bool = False,
    ) -> None:
        model_path = Path(model_path)
        if not model_path.is_file():
            raise ModelNotFoundError(
                f"Model not found at '{model_path}'. Run: python scripts/download_model.py"
            )

        self.model_path = model_path
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold
        self.labels = tuple(labels)

        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        if use_cuda:
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        # default backend/target = OpenCV on CPU
        # cv2.dnn.Net is not safe for concurrent forward passes.
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return self.model_path.stem

    def warmup(self) -> None:
        self.detect(np.zeros((self.input_size, self.input_size, 3), dtype=np.uint8))

    def detect(
        self, image: np.ndarray, classes: set[int] | None = None, *, tiles: int = 1,
        conf: float | None = None,
    ) -> list[Detection]:
        """Run detection on a BGR image and return boxes in original image coordinates.

        tiles > 1 adds a tiled pass (tiles x tiles overlapping crops, each zoomed to the
        network size) on top of the full-image pass. Much better on small / distant
        objects and people, at ~(1 + tiles^2)x the cost.
        """
        if image is None or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected a BGR image with shape (H, W, 3)")
        if tiles <= 1:
            return self._detect_once(image, classes, conf)
        return self._detect_tiled(image, classes, tiles, conf)

    def _detect_tiled(self, image: np.ndarray, classes: set[int] | None, n: int, conf: float | None = None,
                      overlap: float = 0.25) -> list[Detection]:
        h, w = image.shape[:2]
        dets = self._detect_once(image, classes, conf)
        th, tw = int(np.ceil(h / (n - (n - 1) * overlap))), int(np.ceil(w / (n - (n - 1) * overlap)))
        ys = np.linspace(0, h - th, n).astype(int)
        xs = np.linspace(0, w - tw, n).astype(int)
        edge = 3
        for y0 in ys:
            for x0 in xs:
                tile = image[y0:y0 + th, x0:x0 + tw]
                for d in self._detect_once(tile, classes, conf):
                    bx1, by1, bx2, by2 = d.box
                    # Objects cut by an *inner* tile border are partial; the full-image pass or a
                    # neighbouring (overlapping) tile sees them whole, so drop the fragment.
                    cut = ((bx1 <= edge and x0 > 0) or (by1 <= edge and y0 > 0)
                           or (bx2 >= tile.shape[1] - 1 - edge and x0 + tw < w)
                           or (by2 >= tile.shape[0] - 1 - edge and y0 + th < h))
                    if not cut:
                        dets.append(Detection(d.class_id, d.label, d.confidence,
                                              (bx1 + x0, by1 + y0, bx2 + x0, by2 + y0)))
        return merge_detections(dets, self.nms_threshold)

    def _detect_once(
        self, image: np.ndarray, classes: set[int] | None = None, conf: float | None = None,
    ) -> list[Detection]:
        padded, info = letterbox(image, self.input_size)
        blob = cv2.dnn.blobFromImage(padded, scalefactor=1 / 255.0, swapRB=True, crop=False)
        with self._lock:
            self._net.setInput(blob)
            output = self._net.forward()
        return self.postprocess(output, info, image.shape[:2], classes, conf)

    def postprocess(
        self,
        output: np.ndarray,
        info: LetterboxInfo,
        image_shape: tuple[int, int],
        classes: set[int] | None = None,
        conf: float | None = None,
    ) -> list[Detection]:
        """Decode raw YOLO output of shape (1, 4 + num_classes, N) into detections."""
        preds = np.squeeze(output, axis=0)
        nc = len(self.labels)
        attrs = {4 + nc, 4 + nc + 32}  # detect head, or segment head (+32 mask coefficients)
        if preds.shape[0] in attrs and preds.shape[1] not in attrs:  # (84, N) -> (N, 84)
            preds = preds.T

        class_scores = preds[:, 4:4 + nc]
        class_ids = np.argmax(class_scores, axis=1)
        scores = class_scores[np.arange(len(class_ids)), class_ids]

        threshold = self.conf_threshold if conf is None else conf
        keep = scores >= threshold
        if classes is not None:
            keep &= np.isin(class_ids, list(classes))
        if not np.any(keep):
            return []

        preds, scores, class_ids = preds[keep], scores[keep], class_ids[keep]

        # cx, cy, w, h (letterboxed input space) -> x, y, w, h (original image space)
        cx, cy, bw, bh = preds[:, 0], preds[:, 1], preds[:, 2], preds[:, 3]
        x = (cx - bw / 2 - info.pad_x) / info.scale
        y = (cy - bh / 2 - info.pad_y) / info.scale
        w = bw / info.scale
        h = bh / info.scale
        boxes_xywh = np.stack([x, y, w, h], axis=1)

        indices = cv2.dnn.NMSBoxesBatched(
            boxes_xywh.tolist(),
            scores.tolist(),
            class_ids.tolist(),
            threshold,
            self.nms_threshold,
        )

        img_h, img_w = image_shape
        detections: list[Detection] = []
        for i in np.asarray(indices).reshape(-1):
            bx, by, bw_, bh_ = boxes_xywh[i]
            x1 = int(np.clip(bx, 0, img_w - 1))
            y1 = int(np.clip(by, 0, img_h - 1))
            x2 = int(np.clip(bx + bw_, 0, img_w - 1))
            y2 = int(np.clip(by + bh_, 0, img_h - 1))
            if x2 <= x1 or y2 <= y1:
                continue
            cid = int(class_ids[i])
            label = self.labels[cid] if cid < len(self.labels) else str(cid)
            detections.append(Detection(cid, label, float(scores[i]), (x1, y1, x2, y2)))

        detections.sort(key=lambda d: d.confidence, reverse=True)
        return detections
