import numpy as np

from app.ml.detector import Detection, YoloDetector, merge_detections


def det(box, conf=0.9, cid=0):
    return Detection(cid, "person" if cid == 0 else "car", conf, box)


def test_merge_removes_duplicates_but_keeps_other_classes_and_neighbours():
    out = merge_detections([
        det((100, 100, 200, 300), 0.9),
        det((102, 98, 201, 305), 0.8),           # same person seen by a tile -> merged
        det((100, 100, 200, 300), 0.7, cid=2),   # a car in the same place -> kept
        det((210, 100, 300, 300), 0.6),          # neighbour -> kept
    ])
    assert [(d.class_id, d.confidence) for d in out] == [(0, 0.9), (2, 0.7), (0, 0.6)]


def test_merge_keeps_person_standing_in_front_of_another():
    adult = det((100, 50, 300, 600), 0.9)
    child = det((150, 300, 250, 600), 0.8)       # fully inside the adult's box, IoU ~0.27
    assert len(merge_detections([adult, child])) == 2


class FakeDetector(YoloDetector):
    """Returns a scripted detection per call so tiling/offset logic can be tested."""

    def __init__(self, script):
        self.script = script
        self.calls = []
        self.nms_threshold = 0.5

    def _detect_once(self, image, classes=None, conf=None):
        self.calls.append(image.shape[:2])
        return self.script(image)


def test_tiles_offset_boxes_and_drop_border_fragments():
    img = np.zeros((400, 400, 3), np.uint8)

    def script(tile):
        h, w = tile.shape[:2]
        if (h, w) == (400, 400):            # full image: sees nothing (object too small)
            return []
        return [det((10, 10, 30, 50), 0.8),  # small person well inside the tile
                det((0, 60, 20, 120), 0.7)]  # fragment touching the tile's left edge

    d = FakeDetector(script)
    out = d.detect(img, tiles=2)
    assert len(d.calls) == 1 + 4                 # full pass + 2x2 tiles
    boxes = {o.box for o in out}
    assert (10, 10, 30, 50) in boxes             # top-left tile, no offset
    # every tile's interior detection is kept, shifted into image coordinates
    assert any(b[0] > 150 and b[1] > 150 for b in boxes)
    # left-edge fragments only survive in tiles that start at the image's left border
    assert all(not (b[0] > 0 and b[2] - b[0] == 20 and b[3] - b[1] == 60) for b in boxes)
