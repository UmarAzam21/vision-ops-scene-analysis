import cv2
import numpy as np

from app.ml.density import CrowdDensityEstimator
from app.ml.hazard import HeuristicHazardDetector


def _scene_with_fire() -> np.ndarray:
    img = np.full((480, 640, 3), (90, 80, 70), dtype=np.uint8)  # dull background
    cv2.ellipse(img, (320, 300), (80, 110), 0, 0, 360, (0, 120, 255), -1)  # orange flame
    cv2.ellipse(img, (320, 320), (35, 55), 0, 0, 360, (60, 210, 255), -1)  # yellow core
    return img


def test_fire_detected_in_still_image():
    hazards = HeuristicHazardDetector(temporal=False).detect(_scene_with_fire())
    fire = [h for h in hazards if h.kind == "fire"]
    assert len(fire) == 1
    x1, y1, x2, y2 = fire[0].box
    assert x1 < 320 < x2 and y1 < 300 < y2
    assert 0.5 < fire[0].confidence < 1.0


def test_no_fire_on_neutral_scene():
    img = np.full((480, 640, 3), (120, 120, 120), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 300), (200, 50, 30), -1)  # blue object
    assert HeuristicHazardDetector(temporal=False).detect(img) == []


def test_fire_requires_persistence_in_video_mode():
    det = HeuristicHazardDetector(temporal=True)
    blank = np.full((480, 640, 3), (90, 80, 70), dtype=np.uint8)
    for _ in range(5):
        det.detect(blank)
    first = det.detect(_scene_with_fire())   # single-frame flash -> not yet confirmed
    for _ in range(4):
        last = det.detect(_scene_with_fire())
    assert not [h for h in first if h.kind == "fire"]
    assert [h for h in last if h.kind == "fire"]


def test_density_levels():
    est = CrowdDensityEstimator(smoothing=0)
    shape = (720, 1280, 3)
    assert est.level(est.update([(100, 100)], shape)) == "LOW"
    crowd = [(640 + dx, 360 + dy) for dx in range(-20, 21, 10) for dy in range(-20, 21, 10)]
    assert est.level(est.update(crowd, shape)) == "HIGH"


def test_density_colorize_shape():
    est = CrowdDensityEstimator()
    grid = est.update([(10, 10)], (100, 200, 3))
    heat = est.colorize(grid, (320, 160))
    assert heat.shape == (160, 320, 3) and heat.dtype == np.uint8
