"""Lightweight IoU tracker that gives each person a stable ID and accumulates evidence
(gender / age / expression votes, posture history, speed) across frames."""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np

from app.ml.detector import Box
from app.ml.face import EXPRESSIONS, GENDERS


def iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@dataclass(slots=True)
class Track:
    id: int
    box: Box
    last_seen: int
    hits: int = 1
    centers: deque = field(default_factory=lambda: deque(maxlen=15))  # (t, cx, cy, height)
    gender_votes: np.ndarray = field(default_factory=lambda: np.zeros(len(GENDERS)))
    gender_samples: int = 0
    ages: deque = field(default_factory=lambda: deque(maxlen=30))
    expression: np.ndarray | None = None
    postures: deque = field(default_factory=lambda: deque(maxlen=9))
    clothing_votes: deque = field(default_factory=lambda: deque(maxlen=9))
    face_box: Box | None = None

    def clothing(self) -> str | None:
        return Counter(self.clothing_votes).most_common(1)[0][0] if self.clothing_votes else None

    # -------------------------------------------------------------- evidence
    def add_face(self, gender, age, expression) -> None:
        if gender is not None:
            self.gender_votes += gender
            self.gender_samples += 1
        if age is not None:
            self.ages.append(age)
        if expression is not None:
            self.expression = expression if self.expression is None else 0.7 * self.expression + 0.3 * expression

    def gender(self, min_samples: int = 3, min_share: float = 0.75) -> tuple[str | None, float]:
        """Majority over all frames. 'uncertain' when evidence is thin or split."""
        if self.gender_samples == 0:
            return None, 0.0
        share = self.gender_votes / self.gender_votes.sum()
        i = int(share.argmax())
        if self.gender_samples < min_samples or share[i] < min_share:
            return "uncertain", float(share[i])
        return GENDERS[i], float(share[i])

    def age(self) -> int | None:
        """Median of recent estimates - robust to the odd bad frame."""
        return int(round(float(np.median(self.ages)))) if self.ages else None

    def expression_label(self) -> tuple[str | None, float]:
        if self.expression is None:
            return None, 0.0
        i = int(self.expression.argmax())
        return EXPRESSIONS[i], float(self.expression[i])

    def posture(self) -> str | None:
        return Counter(self.postures).most_common(1)[0][0] if self.postures else None

    def motion(self) -> str | None:
        """Speed in body-heights per second → still / walking / running."""
        if len(self.centers) < 4:
            return None
        t0, x0, y0, _ = self.centers[0]
        t1, x1, y1, _ = self.centers[-1]
        if t1 - t0 < 0.3:
            return None
        height = np.median([c[3] for c in self.centers])
        speed = np.hypot(x1 - x0, y1 - y0) / max(1.0, height) / (t1 - t0)
        if speed < 0.12:
            return "still"
        if speed < 0.9:
            return "walking"
        return "running"


class PersonTracker:
    def __init__(self, iou_threshold: float = 0.3, max_missing: int = 20) -> None:
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.tracks: dict[int, Track] = {}
        self._next_id = 1
        self._frame = 0

    def _affinity(self, box: Box, track: Track) -> float:
        """Overlap first; fall back to centre distance (in body-heights) so fast movers at
        low analysis FPS keep their ID. Uses a constant-velocity prediction. 0 = no match."""
        overlap = iou(box, track.box)
        h = max(1.0, track.box[3] - track.box[1])
        px, py = (track.box[0] + track.box[2]) / 2, (track.box[1] + track.box[3]) / 2
        if len(track.centers) >= 2:
            (_, x0, y0, _), (_, x1, y1, _) = track.centers[-2], track.centers[-1]
            px, py = px + (x1 - x0), py + (y1 - y0)
        cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
        dist = np.hypot(cx - px, cy - py) / h
        size_ratio = (box[3] - box[1]) / h
        near = dist < 0.75 and 0.6 < size_ratio < 1.6
        if overlap < self.iou_threshold and not near:
            return 0.0
        return overlap + max(0.0, 1.0 - dist / 0.75) * 0.5

    def update(self, boxes: list[Box], t: float) -> list[Track]:
        """Match boxes to tracks (greedy IoU). Returns one Track per input box, same order."""
        self._frame += 1
        pairs = sorted(
            (score, bi, tid)
            for bi, b in enumerate(boxes)
            for tid, tr in self.tracks.items()
            if (score := self._affinity(b, tr)) > 0
        )[::-1]
        assigned: dict[int, Track] = {}
        used: set[int] = set()
        for _, bi, tid in pairs:
            if bi in assigned or tid in used:
                continue
            tr = self.tracks[tid]
            tr.box, tr.last_seen, tr.hits = boxes[bi], self._frame, tr.hits + 1
            assigned[bi] = tr
            used.add(tid)

        for bi, b in enumerate(boxes):
            if bi not in assigned:
                tr = Track(self._next_id, b, self._frame)
                self.tracks[tr.id] = tr
                self._next_id += 1
                assigned[bi] = tr

        for bi, tr in assigned.items():
            b = boxes[bi]
            tr.centers.append((t, (b[0] + b[2]) / 2, (b[1] + b[3]) / 2, b[3] - b[1]))

        for tid in [tid for tid, tr in self.tracks.items() if self._frame - tr.last_seen > self.max_missing]:
            del self.tracks[tid]
        return [assigned[i] for i in range(len(boxes))]
