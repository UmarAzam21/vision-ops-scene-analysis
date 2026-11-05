import numpy as np

from app.ml import activity
from app.ml.detector import Detection
from app.ml.pose import KP, Pose
from app.services.results import PersonInsight
from app.services.tracker import PersonTracker, Track


def make_pose(points: dict[str, tuple[float, float]], box=(100, 50, 300, 650)) -> Pose:
    kpts = np.zeros((17, 3), dtype=np.float32)
    for name, (x, y) in points.items():
        kpts[KP[name]] = (x, y, 0.95)
    return Pose(box, 0.9, kpts)


UPPER = {
    "nose": (200, 100), "left_eye": (215, 90), "right_eye": (185, 90),
    "left_shoulder": (250, 180), "right_shoulder": (150, 180),
    "left_hip": (240, 380), "right_hip": (160, 380),
}


# ------------------------------------------------------------------ posture
def test_standing():
    pose = make_pose({**UPPER, "left_knee": (240, 500), "right_knee": (160, 500),
                      "left_ankle": (240, 620), "right_ankle": (160, 620)})
    assert activity.posture(pose) == "standing"


def test_sitting_thighs_horizontal():
    pose = make_pose({**UPPER, "left_knee": (330, 395), "right_knee": (250, 395),
                      "left_ankle": (330, 520), "right_ankle": (250, 520)})
    assert activity.posture(pose) == "sitting"


def test_lying_down():
    pose = make_pose({"left_shoulder": (120, 300), "right_shoulder": (120, 340),
                      "left_hip": (320, 300), "right_hip": (320, 340)}, box=(80, 260, 480, 380))
    assert activity.posture(pose) == "lying down"


def test_legs_out_of_frame_is_reported_not_guessed():
    assert activity.posture(make_pose(UPPER)) == "upper body only"


# ----------------------------------------------------------------- gestures
def test_facing():
    assert activity.facing(make_pose(UPPER)) == "facing camera"
    back = {k: v for k, v in UPPER.items() if k not in ("nose", "left_eye", "right_eye")}
    assert activity.facing(make_pose(back)) == "back to camera"


def test_hands_raised():
    pose = make_pose({**UPPER, "left_wrist": (260, 40), "right_wrist": (140, 40)})
    assert "both hands raised" in activity.gestures(pose)
    pose = make_pose({**UPPER, "left_wrist": (260, 40), "right_wrist": (150, 380)})
    assert "hand raised" in activity.gestures(pose)


def test_phone_call_and_carrying():
    pose = make_pose({**UPPER, "right_wrist": (175, 120)})
    phone = Detection(67, "cell phone", 0.8, (160, 95, 190, 140))
    bag = Detection(24, "backpack", 0.8, (150, 170, 260, 360))
    items, acts = activity.interactions(pose, [phone, bag])
    assert items == ["backpack", "cell phone"]
    assert "on a phone call" in acts


# --------------------------------------------------------------- appearance
def test_color_names():
    def px(bgr):
        return np.tile(np.array(bgr, np.uint8), (500, 1))
    assert activity.color_name(px((200, 80, 30))) == "blue"
    assert activity.color_name(px((30, 30, 220))) == "red"
    assert activity.color_name(px((20, 20, 20))) == "black"
    assert activity.color_name(px((235, 235, 235))) == "white"
    assert activity.color_name(px((150, 195, 225))) == "beige"   # camel / sand coat
    assert activity.color_name(px((0, 140, 255))) == "orange"    # hi-vis vest stays orange


def test_upper_clothing_color_reads_torso():
    frame = np.full((700, 400, 3), (0, 200, 0), np.uint8)  # green background
    frame[180:380, 150:250] = (200, 80, 30)                 # blue shirt
    assert activity.upper_clothing_color(frame, make_pose(UPPER)) == "blue"


# ------------------------------------------------------------------ tracker
def test_tracker_keeps_ids_and_measures_motion():
    tr = PersonTracker()
    ids = []
    for i in range(8):
        x = 100 + i * 60  # moves ~1 body-height/s at 10 fps
        (t,) = tr.update([(x, 100, x + 80, 400)], i * 0.1)
        ids.append(t.id)
    assert len(set(ids)) == 1
    assert t.motion() in ("walking", "running")


def test_tracker_does_not_swap_neighbours():
    tr = PersonTracker()
    a, b = (100, 100, 180, 400), (200, 100, 280, 400)  # two people side by side
    first = [t.id for t in tr.update([a, b], 0.0)]
    for i in range(1, 6):
        ids = [t.id for t in tr.update([a, b], i * 0.1)]
        assert ids == first
    assert len(set(first)) == 2


def test_gender_vote_needs_agreement():
    t = Track(1, (0, 0, 10, 10), 0)
    for _ in range(4):
        t.add_face(np.array([0.95, 0.05]), None, None)
    assert t.gender()[0] == "male"
    for _ in range(4):
        t.add_face(np.array([0.1, 0.9]), None, None)  # contradicting evidence
    assert t.gender()[0] == "uncertain"


def test_tie_is_worn_not_held():
    pose = make_pose({**UPPER, "right_wrist": (190, 260)})
    tie = Detection(27, "tie", 0.6, (185, 190, 215, 330))
    items, _ = activity.interactions(pose, [tie])
    assert "tie" not in items
    assert activity.worn_items(pose, [tie]) == ["tie"]


def test_description_reads_naturally():
    p = PersonInsight(track_id=1, box=(0, 0, 10, 10), confidence=0.9, gender="male", age=29,
                      posture="sitting", motion="still", facing="facing camera",
                      expression="neutral", expression_confidence=0.8, clothing="blue", holding=["cell phone"])
    d = p.description()
    assert d.startswith("Man, ~29 yrs")
    assert "sitting" in d and "neutral expression" in d and "blue top" in d and "cell phone" in d
