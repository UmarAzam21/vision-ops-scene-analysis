import numpy as np
import pytest

from app.ml.detector import LetterboxInfo, ModelNotFoundError, YoloDetector, letterbox
from tests.conftest import MODEL_PATH, requires_model


def test_letterbox_keeps_aspect_ratio_and_centres():
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    out, info = letterbox(img, 640)
    assert out.shape == (640, 640, 3)
    assert info.scale == pytest.approx(1.0)
    assert (info.pad_x, info.pad_y) == (0, 140)


def test_missing_model_raises_clear_error(tmp_path):
    with pytest.raises(ModelNotFoundError, match="download_model"):
        YoloDetector(tmp_path / "nope.onnx")


def _fake_output(rows: list[tuple[float, float, float, float, int, float]], n_classes=80, n=50):
    """Build a raw YOLO tensor (1, 4 + C, N) from (cx, cy, w, h, class, score) rows."""
    out = np.zeros((1, 4 + n_classes, n), dtype=np.float32)
    for i, (cx, cy, w, h, cls, score) in enumerate(rows):
        out[0, :4, i] = (cx, cy, w, h)
        out[0, 4 + cls, i] = score
    return out


@requires_model
def test_postprocess_maps_boxes_back_and_applies_nms():
    det = YoloDetector(MODEL_PATH, conf_threshold=0.3, nms_threshold=0.45)
    raw = _fake_output([
        (320, 320, 100, 200, 0, 0.90),   # person
        (322, 318, 102, 198, 0, 0.80),   # duplicate -> suppressed by NMS
        (320, 320, 100, 200, 2, 0.70),   # car, same place, other class -> kept
        (100, 100, 20, 20, 0, 0.10),     # below threshold
    ])
    # Original image 1280x720 letterboxed to 640: scale 0.5, pad_y 140
    info = LetterboxInfo(scale=0.5, pad_x=0, pad_y=140)
    out = det.postprocess(raw, info, (720, 1280))

    assert [d.label for d in out] == ["person", "car"]
    person = out[0]
    assert person.confidence == pytest.approx(0.9)
    assert person.box == (540, 160, 740, 560)


@requires_model
def test_postprocess_class_filter():
    det = YoloDetector(MODEL_PATH)
    raw = _fake_output([(320, 320, 50, 50, 0, 0.9), (100, 100, 50, 50, 2, 0.9)])
    out = det.postprocess(raw, LetterboxInfo(1.0, 0, 0), (640, 640), classes={2})
    assert [d.label for d in out] == ["car"]


@requires_model
def test_detect_on_blank_image_returns_nothing():
    det = YoloDetector(MODEL_PATH)
    assert det.detect(np.zeros((480, 640, 3), dtype=np.uint8)) == []
