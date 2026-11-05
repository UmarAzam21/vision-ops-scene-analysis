import numpy as np

from app.ml.detector import Detection
from app.ml.pose import Pose
from app.services.analyzer import FrameAnalyzer


def person(box, conf=0.9):
    return Detection(0, "person", conf, box)


def pose(box, conf=0.8):
    return Pose(box, conf, np.zeros((17, 3), np.float32))


class _Bare(FrameAnalyzer):
    """FrameAnalyzer without models - only the merging logic is under test."""

    def __init__(self, temporal: bool):
        self.temporal = temporal


def test_photo_uses_detector_people_and_attaches_skeletons():
    a = _Bare(temporal=False)
    detected = [person((0, 0, 100, 300)), person((200, 0, 300, 300)), person((400, 0, 450, 120))]
    poses = [pose((5, 5, 102, 298))]                    # pose model only saw the first one
    persons, pose_of = a._combine_persons(detected, poses, fresh=True)
    assert len(persons) == 3                             # nobody lost: detector is the authority
    assert pose_of[0] is poses[0] and pose_of[1] is None and pose_of[2] is None


def test_photo_keeps_person_only_pose_model_found():
    a = _Bare(temporal=False)
    persons, pose_of = a._combine_persons([person((0, 0, 100, 300))], [pose((500, 0, 600, 300))], fresh=True)
    assert len(persons) == 2 and pose_of[1] is not None


def test_video_adds_detector_only_people_on_fresh_frames():
    a = _Bare(temporal=True)
    poses = [pose((0, 0, 100, 300))]
    detected = [person((2, 0, 101, 300)), person((300, 0, 360, 200))]
    persons, _ = a._combine_persons(detected, poses, fresh=True)
    assert len(persons) == 2                             # duplicate merged, extra person added
    persons, _ = a._combine_persons(detected, poses, fresh=False)
    assert len(persons) == 1                             # stale detector boxes are not reused


def test_open_vocab_adds_new_types_only():
    known = [person((0, 0, 100, 300)), Detection(2, "car", 0.9, (300, 300, 500, 400))]
    ov = [
        Detection(10, "man", 0.9, (0, 0, 100, 300)),       # synonym of person -> dropped
        Detection(11, "sedan", 0.8, (302, 301, 499, 401)), # same object as the car -> dropped
        Detection(12, "car", 0.8, (600, 300, 700, 400)),   # COCO type -> COCO model is the authority
        Detection(13, "glasses", 0.7, (30, 20, 60, 35)),   # new type -> kept
        Detection(14, "lamp", 0.6, (700, 0, 750, 80)),     # new type -> kept
    ]
    out = FrameAnalyzer._new_types(ov, known)
    assert [d.label for d in out] == ["glasses", "lamp"]
    assert all(d.class_id >= 1000 for d in out)          # never collides with COCO ids
