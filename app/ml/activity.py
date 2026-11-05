"""Explainable activity & appearance rules built on pose keypoints and nearby objects.

Everything here is geometric reasoning – no black box – so each output can be traced
to the keypoints / boxes that produced it. When the evidence isn't in the frame
(e.g. legs cut off), we say so instead of guessing.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.ml.detector import Box, Detection
from app.ml.pose import Pose

# ------------------------------------------------------------------- posture


def _angle_from_vertical(a: tuple[float, float], b: tuple[float, float]) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    return math.degrees(math.atan2(abs(dx), abs(dy) + 1e-6))


def posture(pose: Pose) -> str:
    p = pose.point
    sho = [q for q in (p("left_shoulder"), p("right_shoulder")) if q]
    hips = [q for q in (p("left_hip"), p("right_hip")) if q]
    bw, bh = pose.box[2] - pose.box[0], pose.box[3] - pose.box[1]

    # Lying: torso close to horizontal and the box wider than tall.
    if sho and hips:
        s = np.mean(sho, axis=0)
        hp = np.mean(hips, axis=0)
        if _angle_from_vertical(tuple(s), tuple(hp)) > 60 and bw > bh:
            return "lying down"

    votes: list[str] = []
    for side in ("left", "right"):
        hip, knee, ankle = p(f"{side}_hip"), p(f"{side}_knee"), p(f"{side}_ankle")
        if not (hip and knee):
            continue
        thigh = _angle_from_vertical(hip, knee)
        if thigh > 55:
            votes.append("sitting")
        elif ankle and _angle_from_vertical(knee, ankle) < 35 and thigh < 35:
            votes.append("standing")
        elif thigh < 35:
            votes.append("standing")
        else:
            votes.append("crouching")
    if votes:
        return max(set(votes), key=votes.count)
    if hips:
        return "upper body only"
    if sho:
        return "head & shoulders only"
    return "unknown"


# ------------------------------------------------------------------ gestures


def facing(pose: Pose) -> str:
    p = pose.point
    nose, le, re = p("nose"), p("left_eye"), p("right_eye")
    if nose and le and re:
        return "facing camera"
    if nose and (le or re):
        return "side view"
    if p("left_shoulder") and p("right_shoulder") and not nose:
        return "back to camera"
    return "unknown"


def gestures(pose: Pose) -> list[str]:
    p = pose.point
    out: list[str] = []
    nose = p("nose")
    ls, rs = p("left_shoulder"), p("right_shoulder")
    lw, rw = p("left_wrist"), p("right_wrist")
    le, re = p("left_elbow"), p("right_elbow")
    shoulder_w = math.dist(ls, rs) if ls and rs else max(20.0, pose.height * 0.2)
    head_y = nose[1] if nose else (min(ls[1], rs[1]) - shoulder_w * 0.6 if ls and rs else None)

    raised = [w for w in (lw, rw) if w and head_y is not None and w[1] < head_y]
    if len(raised) == 2:
        out.append("both hands raised")
    elif raised:
        out.append("hand raised")

    if nose:
        near_face = [w for w in (lw, rw) if w and w not in raised and math.dist(w, nose) < shoulder_w * 0.6]
        if near_face:
            out.append("hand near face")

    if lw and rw and ls and rs:
        mid_x = (ls[0] + rs[0]) / 2
        # When facing the camera the person's left side is on the image right.
        crossed = (lw[0] - mid_x) * (ls[0] - mid_x) < 0 and (rw[0] - mid_x) * (rs[0] - mid_x) < 0
        chest = all(min(ls[1], rs[1]) < w[1] < max(ls[1], rs[1]) + shoulder_w * 1.3 for w in (lw, rw))
        if crossed and chest:
            out.append("arms crossed")

    for sh, el, wr in ((ls, le, lw), (rs, re, rw)):
        if sh and el and wr and math.dist(sh, wr) > shoulder_w * 1.25 and abs(wr[1] - sh[1]) < shoulder_w * 0.5:
            out.append("arm extended / pointing")
            break

    sitting_hands = [w for w in (lw, rw) if w]
    hips = [q for q in (p("left_hip"), p("right_hip")) if q]
    if not out and sitting_hands and hips:
        hip_y = np.mean([h[1] for h in hips])
        if all(abs(w[1] - hip_y) < shoulder_w * 0.6 for w in sitting_hands):
            out.append("hands resting")
    return out


# ------------------------------------------------------- object interactions

CARRIED = {"backpack", "handbag", "suitcase", "umbrella"}
WORN = {"tie"}
HAND_HELD = {"cell phone", "cup", "bottle", "wine glass", "book", "remote", "knife", "scissors",
             "sports ball", "frisbee", "tennis racket", "baseball bat", "umbrella", "laptop",
             "toothbrush", "hair drier", "mouse", "keyboard", "banana", "apple", "sandwich", "donut",
             "pizza", "hot dog", "teddy bear"}
RIDDEN = {"bicycle", "motorcycle", "horse", "skateboard", "surfboard", "skis", "snowboard"}


def _box_dist(pt: tuple[float, float], box: Box) -> float:
    dx = max(box[0] - pt[0], 0, pt[0] - box[2])
    dy = max(box[1] - pt[1], 0, pt[1] - box[3])
    return math.hypot(dx, dy)


def _ioa(inner: Box, outer: Box) -> float:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


def worn_items(pose: Pose, objects: list[Detection]) -> list[str]:
    """Accessories the detector knows that are worn rather than held (e.g. a tie)."""
    return sorted({o.label for o in objects if o.label in WORN and _ioa(o.box, pose.box) > 0.6})


def interactions(pose: Pose, objects: list[Detection]) -> tuple[list[str], list[str]]:
    """Return (holding/carrying items, derived activities) for one person."""
    p = pose.point
    wrists = [w for w in (p("left_wrist", 0.3), p("right_wrist", 0.3)) if w]
    nose = p("nose")
    reach = max(15.0, pose.height * 0.08)
    items: list[str] = []
    acts: list[str] = []
    for obj in objects:
        lbl = obj.label
        near_hand = any(_box_dist(w, obj.box) < reach for w in wrists)
        if lbl in HAND_HELD and near_hand:
            items.append(lbl)
            if nose and _box_dist(nose, obj.box) < pose.height * 0.15:
                if lbl == "cell phone":
                    acts.append("on a phone call")
                elif lbl in {"cup", "bottle", "wine glass"}:
                    acts.append("drinking")
                elif lbl in {"banana", "apple", "sandwich", "donut", "pizza", "hot dog"}:
                    acts.append("eating")
            elif lbl == "cell phone":
                acts.append("using a phone")
            elif lbl == "book":
                acts.append("reading")
        elif lbl in CARRIED and _ioa(obj.box, pose.box) > 0.5:
            items.append(lbl)
        elif lbl in RIDDEN and _ioa(pose.box, obj.box) > 0.3 and obj.box[3] >= pose.box[3] - pose.height * 0.2:
            acts.append(f"riding a {lbl}")
        elif lbl == "laptop" and _ioa(obj.box, pose.box) > 0.3:
            acts.append("using a laptop")
    return sorted(set(items)), sorted(set(acts))


# --------------------------------------------------------------- appearance

_HUES = ((8, "red"), (20, "orange"), (33, "yellow"), (85, "green"), (100, "teal"),
         (130, "blue"), (150, "purple"), (170, "pink"), (181, "red"))


def color_name(pixels_bgr: np.ndarray) -> str | None:
    """Name the dominant colour of an (N, 3) BGR pixel array."""
    if pixels_bgr is None or len(pixels_bgr) < 30:
        return None
    hsv = cv2.cvtColor(pixels_bgr.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    h, s, v = hsv[:, 0], hsv[:, 1].astype(int), hsv[:, 2].astype(int)
    chromatic = (s > 60) & (v > 50)
    if chromatic.mean() < 0.4:
        vm = np.median(v)
        return "black" if vm < 60 else "white" if vm > 190 and np.median(s) < 40 else "grey"
    hist = np.bincount(h[chromatic], minlength=180)
    hist = np.convolve(hist, np.ones(7), mode="same")
    hue = int(hist.argmax())
    name = next(n for limit, n in _HUES if hue < limit)
    vm = np.median(v[chromatic])
    if name in ("orange", "red", "yellow") and vm < 120:
        return "brown"
    if name in ("orange", "yellow") and np.median(s[chromatic]) < 110 and vm > 150:
        return "beige"  # light, weakly saturated warm tones (camel, sand, cream)
    if vm < 90:
        return f"dark {name}"
    return name


def upper_clothing_color(frame: np.ndarray, pose: Pose) -> str | None:
    p = pose.point
    ls, rs = p("left_shoulder"), p("right_shoulder")
    if not (ls and rs):
        return None
    lh, rh = p("left_hip"), p("right_hip")
    sw = math.dist(ls, rs)
    if not lh:
        lh = (ls[0], ls[1] + sw * 1.2)
    if not rh:
        rh = (rs[0], rs[1] + sw * 1.2)
    poly = np.array([ls, rs, rh, lh], dtype=np.float32)
    # Shrink toward the centre so arms / background at the edges don't leak in.
    c = poly.mean(axis=0)
    poly = (c + (poly - c) * 0.7).astype(np.int32)
    h, w = frame.shape[:2]
    poly[:, 0] = poly[:, 0].clip(0, w - 1)
    poly[:, 1] = poly[:, 1].clip(0, h - 1)
    mask = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(mask, poly, 1)
    return color_name(frame[mask.astype(bool)])
