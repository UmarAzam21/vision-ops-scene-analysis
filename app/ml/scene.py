"""Scene understanding: SegFormer-B0 (ADE20K, 150 classes) semantic segmentation.

Labels every pixel (sky, building, wall, tree, plant, road, floor, car, chair, …),
which gives "what is in the background" and lets the HUD highlight everything,
including things the 80-class box detector does not know.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from app.ml.detector import ModelNotFoundError

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

OUTDOOR = {"sky", "tree", "grass", "road", "sidewalk", "mountain", "plant", "building", "field", "sand",
           "sea", "water", "river", "earth", "path", "fence", "skyscraper", "palm", "hill", "streetlight",
           "car", "dirt track", "land", "bridge", "house", "flower"}
INDOOR = {"ceiling", "floor", "bed", "cabinet", "sofa", "table", "chair", "shelf", "bathtub", "cushion",
          "desk", "wardrobe", "lamp", "curtain", "television receiver", "refrigerator", "kitchen island",
          "stairs", "rug", "bookcase", "pillow", "screen door", "counter", "sink", "toilet", "computer"}


@dataclass(frozen=True, slots=True)
class SceneRegion:
    label: str
    fraction: float
    anchor: tuple[float, float]  # normalised (x, y) of the label position


@dataclass(slots=True)
class SceneResult:
    class_map: np.ndarray                       # (H', W') uint8 class ids
    regions: list[SceneRegion] = field(default_factory=list)  # biggest first, person excluded
    setting: str = "unknown"                    # indoor | outdoor | unknown
    person_class: int = -1


def _short(label: str) -> str:
    """ADE labels look like 'windowpane;window' or 'plant;flora;plant;life' – keep the first name."""
    return label.split(";")[0].strip()


class SceneSegmenter:
    def __init__(self, model_path: Path, labels_path: Path, *, input_size: int = 512) -> None:
        if not Path(model_path).is_file():
            raise ModelNotFoundError(f"Scene model not found at '{model_path}'")
        if not Path(labels_path).is_file():
            raise ModelNotFoundError(f"Scene labels not found at '{labels_path}'")
        cfg = json.loads(Path(labels_path).read_text(encoding="utf-8"))
        id2label = cfg.get("id2label", cfg)
        self.labels = [_short(id2label[str(i)]) for i in range(len(id2label))]
        self.person_class = self.labels.index("person") if "person" in self.labels else -1
        self.input_size = input_size
        self._net = cv2.dnn.readNetFromONNX(str(model_path))
        self._lock = threading.Lock()
        rng = np.random.default_rng(7)
        self.palette = rng.integers(40, 255, size=(len(self.labels), 3), dtype=np.uint8)

    def segment(self, image: np.ndarray, *, min_fraction: float = 0.015, max_regions: int = 8) -> SceneResult:
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (self.input_size, self.input_size), swapRB=True)
        blob = (blob - _MEAN) / _STD
        with self._lock:
            self._net.setInput(blob.astype(np.float32))
            logits = self._net.forward()
        class_map = logits[0].argmax(0).astype(np.uint8)

        ids, counts = np.unique(class_map, return_counts=True)
        order = np.argsort(-counts)
        regions: list[SceneRegion] = []
        outdoor = indoor = 0.0
        for idx in order:
            cid, frac = int(ids[idx]), counts[idx] / class_map.size
            label = self.labels[cid]
            outdoor += frac if label in OUTDOOR else 0
            indoor += frac if label in INDOOR else 0
            if cid == self.person_class or frac < min_fraction or len(regions) >= max_regions:
                continue
            regions.append(SceneRegion(label, float(frac), self._anchor(class_map == cid)))

        if outdoor > indoor and outdoor > 0.05:
            setting = "outdoor"
        elif indoor > outdoor and indoor > 0.05:
            setting = "indoor"
        else:
            setting = "unknown"
        return SceneResult(class_map, regions, setting, self.person_class)

    @staticmethod
    def _anchor(mask: np.ndarray) -> tuple[float, float]:
        """Label position: the point deepest inside the largest blob of that class."""
        m = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=4)
        if n > 1:
            biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            m = (labels == biggest).astype(np.uint8)
        dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        y, x = np.unravel_index(int(dist.argmax()), dist.shape)
        return x / mask.shape[1], y / mask.shape[0]

    def colorize(self, result: SceneResult, size: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
        """Return (BGR colour overlay, person mask) resized to `size` (w, h)."""
        cmap = cv2.resize(result.class_map, size, interpolation=cv2.INTER_NEAREST)
        color = self.palette[cmap]
        person = (cmap == result.person_class).astype(np.uint8) if result.person_class >= 0 else None
        return color, person
