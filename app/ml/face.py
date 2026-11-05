"""Face detection (YuNet) + perceived gender, estimated age and facial expression.

Gender and age come from InsightFace's `genderage` model (buffalo_l pack, 1.3 MB).
They are *appearance estimates* from a face crop and can be wrong (profile views,
heavy make-up, children, low light, masks). In video the tracker votes over many
frames before showing a result and reports "uncertain" when the vote is split.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import Box, ModelNotFoundError

GENDERS = ("male", "female")
EXPRESSIONS = ("angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised")

# Reference 5-point landmarks for the 112x112 aligned face used by the expression model.
_ALIGN_REF = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    dtype=np.float32,
)


@dataclass(frozen=True, slots=True)
class FaceInfo:
    box: Box
    score: float
    gender_probs: np.ndarray | None = None      # (2,)  male, female
    age: float | None = None                    # estimated years
    expression_probs: np.ndarray | None = None  # (7,)

    @property
    def center(self) -> tuple[int, int]:
        return (self.box[0] + self.box[2]) // 2, (self.box[1] + self.box[3]) // 2


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64).ravel()
    if np.all(x >= 0) and abs(x.sum() - 1) < 1e-3:
        return x
    e = np.exp(x - x.max())
    return e / e.sum()


def _load(path: Path | None, what: str) -> cv2.dnn.Net | None:
    if path is None:
        return None
    if not Path(path).is_file():
        raise ModelNotFoundError(f"{what} model not found at '{path}'")
    return cv2.dnn.readNetFromONNX(str(path))


class FaceAnalyzer:
    def __init__(
        self,
        detector_path: Path,
        *,
        genderage_path: Path | None = None,
        expression_path: Path | None = None,
        score_threshold: float = 0.75,
        min_attr_size: int = 30,
    ) -> None:
        if not Path(detector_path).is_file():
            raise ModelNotFoundError(f"Face detector not found at '{detector_path}'")
        self._detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), score_threshold)
        self._genderage = _load(genderage_path, "Gender/age")
        self._expression = _load(expression_path, "Expression")
        self.min_attr_size = min_attr_size
        self._lock = threading.Lock()

    @property
    def has_demographics(self) -> bool:
        return self._genderage is not None

    def analyze(self, image: np.ndarray, demographics: bool = True) -> list[FaceInfo]:
        h, w = image.shape[:2]
        with self._lock:
            self._detector.setInputSize((w, h))
            _, faces = self._detector.detect(image)
            if faces is None:
                return []

            results: list[FaceInfo] = []
            for f in faces:
                x, y, fw, fh = (int(v) for v in f[:4])
                box = (max(0, x), max(0, y), min(w - 1, x + fw), min(h - 1, y + fh))
                if box[2] <= box[0] or box[3] <= box[1]:
                    continue
                gender = age = expr = None
                if min(fw, fh) >= self.min_attr_size:
                    if demographics:
                        gender, age = self._demographics(image, box)
                    expr = self._expression_probs(image, f[4:14].reshape(5, 2))
                results.append(FaceInfo(box, float(f[14]), gender, age, expr))
            return results

    # ------------------------------------------------------------------ models
    def _demographics(self, image: np.ndarray, box: Box) -> tuple[np.ndarray | None, float | None]:
        """InsightFace protocol: face-centred square crop, 1.5x the face size, 96x96 RGB,
        raw 0..255 input (the network normalises internally). Output: [female, male, age/100]."""
        if self._genderage is None:
            return None, None
        x1, y1, x2, y2 = box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        scale = 96 / (max(x2 - x1, y2 - y1) * 1.5)
        matrix = np.array([[scale, 0, 48 - cx * scale], [0, scale, 48 - cy * scale]], np.float32)
        crop = cv2.warpAffine(image, matrix, (96, 96))
        self._genderage.setInput(cv2.dnn.blobFromImage(crop, 1.0, (96, 96), (0, 0, 0), swapRB=True))
        out = self._genderage.forward()[0]
        female_male = _softmax(out[:2])
        return np.array([female_male[1], female_male[0]]), float(out[2] * 100)

    def _expression_probs(self, image: np.ndarray, landmarks: np.ndarray) -> np.ndarray | None:
        if self._expression is None:
            return None
        matrix, _ = cv2.estimateAffinePartial2D(landmarks.astype(np.float32), _ALIGN_REF, method=cv2.LMEDS)
        if matrix is None:
            return None
        face = cv2.warpAffine(image, matrix, (112, 112))
        x = (cv2.cvtColor(face, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - 0.5) / 0.5
        self._expression.setInput(x.transpose(2, 0, 1)[None])
        return _softmax(self._expression.forward())
