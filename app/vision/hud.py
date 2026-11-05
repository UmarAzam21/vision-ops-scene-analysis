"""Tactical HUD overlay.

Layers (bottom → top): scene tint + labels · object boxes · person skeletons,
boxes and info cards · density inset · hazards · scene panel · metric bar.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cv2
import numpy as np

from app.ml.detector import Box
from app.ml.pose import SKELETON

if TYPE_CHECKING:
    from app.services.results import FrameResult, PersonInsight

FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA

# BGR palette
LIME = (40, 230, 180)
MAGENTA = (230, 70, 230)
ORANGE = (0, 140, 255)
CYAN = (230, 200, 40)
YELLOW = (60, 220, 255)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
BLACK = (0, 0, 0)

CLASS_COLORS: dict[str, tuple[int, int, int]] = {
    "car": CYAN, "truck": (60, 170, 255), "bus": (255, 160, 60),
    "motorcycle": (200, 120, 255), "bicycle": (200, 120, 255),
}
HAZARD_COLORS = {"fire": ORANGE, "smoke": MAGENTA}
# colour per SKELETON edge index: arms cyan, torso lime, legs yellow, face white
LIMB_COLORS = {0: CYAN, 1: CYAN, 2: CYAN, 3: CYAN, 4: CYAN, 5: LIME, 6: LIME, 7: LIME,
               8: YELLOW, 9: YELLOW, 10: YELLOW, 11: YELLOW}


def _overlap(a: Box, b: Box) -> int:
    """Intersection area of two boxes."""
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


@dataclass(slots=True)
class HudData:
    persons: int
    peak_persons: int
    fire: int
    smoke: int
    frame_ms: float
    density_level: str
    density_heatmap: np.ndarray | None     # BGR heatmap, any size
    density_intensity: np.ndarray | None   # float32 0..1, same size as heatmap
    objects: int = 0
    object_types: int = 0
    show_hazards: bool = False
    scene_color: np.ndarray | None = None  # BGR, frame size
    person_mask: np.ndarray | None = None  # uint8, frame size


class HudRenderer:
    def __init__(
        self,
        *,
        label_min_height: int = 70,
        inset_ratio: float = 0.28,
        inset_min_persons: int = 3,
        max_cards: int = 3,
        scene_alpha: float = 0.22,
    ) -> None:
        self.label_min_height = label_min_height
        self.inset_ratio = inset_ratio
        self.inset_min_persons = inset_min_persons
        self.max_cards = max_cards
        self.scene_alpha = scene_alpha

    # ================================================================ render
    def render(self, frame: np.ndarray, result: FrameResult, data: HudData) -> np.ndarray:
        out = frame.copy()
        h, w = out.shape[:2]
        s = max(0.75, max(h, w) / 1280)  # UI scale, works for portrait too
        bar_h = int(54 * s)
        person_boxes = [p.box for p in result.people] or [
            d.box for d in result.detections if d.label == "person"]

        if data.scene_color is not None and result.scene is not None:
            self._draw_scene(out, data, result, person_boxes, s, bar_h)

        for det in result.detections:
            if det.label != "person":
                self._draw_box(out, det.box, f"{det.label.upper()} {det.confidence:.0%}",
                               CLASS_COLORS.get(det.label, CYAN), s, top=bar_h)

        taken: list[Box] = []
        if result.people:
            ranked = sorted(result.people, key=lambda p: p.box[3] - p.box[1], reverse=True)
            carded = {id(p) for p in ranked[: self.max_cards] if (p.box[3] - p.box[1]) >= h * 0.18}
            for p in result.people:
                self._draw_person(out, p, s, taken, card=id(p) in carded, bar_h=bar_h, people=result.people)
        else:  # no pose model – plain markers
            for det in result.detections:
                if det.label == "person":
                    self._draw_marker(out, det.box, det.confidence, s, taken)

        if data.density_heatmap is not None and data.persons >= self.inset_min_persons:
            rect = self._place_inset(out.shape, s, bar_h, result)
            self._draw_density_inset(out, frame, data, s, rect)

        for hz in result.hazards:
            self._draw_box(out, hz.box, f"{hz.kind.upper()} {hz.confidence:.0%}",
                           HAZARD_COLORS[hz.kind], s, thickness=2, top=bar_h)

        if result.scene is not None and result.scene.regions:
            self._draw_scene_panel(out, result, s, taken)
        if result.no_signal:
            self._draw_no_signal(out, s)
        self._draw_top_bar(out, data, s, bar_h)
        return out

    # ================================================================= scene
    def _draw_scene(self, img, data: HudData, result: FrameResult, person_boxes, s, bar_h) -> None:
        h, w = img.shape[:2]
        # 8-bit blend + masked copy: ~10x faster than per-pixel float alpha on large frames.
        keep = np.zeros((h, w), np.uint8)            # 1 = don't tint (people)
        if data.person_mask is not None:
            keep[data.person_mask > 0] = 1
        for x1, y1, x2, y2 in person_boxes:          # scene may be ~2 s old in video
            keep[y1:y2, x1:x2] = 1
        tinted = cv2.addWeighted(img, 1 - self.scene_alpha, data.scene_color, self.scene_alpha, 0)
        np.copyto(img, tinted, where=(keep == 0)[..., None])

        fs = 0.4 * s
        pad = int(4 * s)
        dot_w = int(10 * s)
        for region in result.scene.regions:
            if region.fraction < 0.02:
                continue
            x = min(w - 1, int(region.anchor[0] * w))
            y = min(h - 1, int(region.anchor[1] * h))
            if any(bx1 <= x <= bx2 and by1 <= y <= by2 for bx1, by1, bx2, by2 in person_boxes):
                continue  # never label the background on top of a person (scene may be ~1 s old)
            color = tuple(int(c) for c in data.scene_color[y, x])
            text = f"{region.label.upper()} {region.fraction:.0%}"
            (tw, th), _ = cv2.getTextSize(text, FONT, fs, 1)
            bw, bh = tw + 2 * pad + dot_w, th + 2 * pad
            x0 = max(0, min(w - bw, x - bw // 2))
            y0 = max(bar_h + pad, min(h - bh, y - bh // 2))
            roi = img[y0:y0 + bh, x0:x0 + bw]
            roi[:] = (roi * 0.35).astype(np.uint8)
            cv2.circle(img, (x0 + pad + dot_w // 3, y0 + bh // 2), max(2, int(3 * s)), color, -1, AA)
            cv2.putText(img, text, (x0 + pad + dot_w, y0 + pad + th), FONT, fs, WHITE, 1, AA)

    def _draw_scene_panel(self, img, result: FrameResult, s: float, taken: list[Box]) -> None:
        scene = result.scene
        lines = [f"SCENE: {scene.setting.upper()}"]
        lines += [f"{r.label:<12} {r.fraction:>4.0%}" for r in scene.regions[:5]]
        h, w = img.shape[:2]
        m = int(12 * s)
        # bottom-left unless a person card already sits there
        for anchor in ((m, h - m, "bottom-left"), (w - m, h - m, "bottom-right")):
            rect = self._panel_rect(lines, anchor, s, img.shape)
            if not any(_overlap(rect, t) for t in taken) or anchor[2] == "bottom-right":
                self._panel(img, lines, rect, s, title_color=CYAN)
                return

    # ================================================================ people
    def _draw_person(
        self, img, p: PersonInsight, s: float, taken: list[Box], *, card: bool, bar_h: int,
        people: list[PersonInsight],
    ) -> None:
        if p.pose is not None:
            self._draw_skeleton(img, p.pose.keypoints, s)
        self._corners(img, p.box, LIME, s)
        if p.face_box is not None:
            cv2.rectangle(img, p.face_box[:2], p.face_box[2:], WHITE, 1, AA)

        if card:
            self._draw_card(img, p, s, taken, bar_h, people)
            return
        label = f"#{p.track_id} " if p.track_id else ""
        doing = p.doing()
        label += (doing[0] if doing else "person").upper()
        if p.gender in ("male", "female"):
            label += f" · {p.gender.upper()}"
        self._tag(img, label, (p.box[0], p.box[3] + int(4 * s)), LIME, s, below=True, taken=taken)

    def _draw_card(
        self, img, p: PersonInsight, s: float, taken: list[Box], bar_h: int, people: list[PersonInsight],
    ) -> None:
        head = f"#{p.track_id} " if p.track_id else ""
        lines = [f"{head}HUMAN  {p.confidence:.0%}"]
        if p.gender in ("male", "female"):
            lines.append(f"GENDER  {p.gender.upper()} ({min(p.gender_confidence, 0.99):.0%}, est.)")
        elif p.gender == "uncertain":
            lines.append("GENDER  UNCERTAIN")
        else:
            lines.append("GENDER  - (no clear face)")
        if p.age is not None:
            lines.append(f"AGE     ABOUT {p.age} (est.)")
        doing = p.doing()
        if p.expression and p.expression_confidence >= 0.4:
            doing = [d for d in doing if not d.endswith("expression")]
            lines.append(f"MOOD    {p.expression.upper()}")
        for i, chunk in enumerate(_wrap(", ".join(doing).upper() if doing else "-", 30)):
            lines.append(("DOING   " if i == 0 else "        ") + chunk)
        if p.clothing:
            worn = " + ".join([f"{p.clothing.upper()} TOP"] + [a.upper() for a in p.accessories])
            lines.append(f"WEARING {worn}")
        if p.holding:
            lines.append(f"HOLDING {', '.join(p.holding).upper()}")

        fs = 0.42 * s
        lh = int(18 * s)
        pad = int(8 * s)
        cw = max(cv2.getTextSize(ln, FONT, fs, 1)[0][0] for ln in lines) + 2 * pad
        ch = lh * len(lines) + 2 * pad
        h, w = img.shape[:2]
        x1, y1, x2, y2 = p.box
        gap = int(8 * s)
        top = max(bar_h + gap, y1)
        candidates = [
            (x2 + gap, top), (x1 - cw - gap, top),                       # beside the person
            (x1 + gap, top + gap), (x2 - cw - gap, top + gap),           # inside, top corners
            (x1 + gap, y2 - ch - gap), (x2 - cw - gap, y2 - ch - gap),   # inside, bottom corners
        ]
        # Never hide a face (anyone's), avoid other cards, prefer not to cover other people.
        faces = [q.face_box for q in people if q.face_box is not None]
        others = [q.box for q in people if q is not p]

        def cost(c: tuple[int, int]) -> float:
            cx, cy = c
            rect = (cx, cy, cx + cw, cy + ch)
            off = (max(0, -cx) + max(0, cx + cw - w)) * ch + (max(0, bar_h - cy) + max(0, cy + ch - h)) * cw
            return (off * 50
                    + sum(_overlap(rect, f) for f in faces) * 30
                    + sum(_overlap(rect, t) for t in taken) * 10
                    + sum(_overlap(rect, o) for o in others) * 1.0
                    + _overlap(rect, p.box) * 0.2)

        cx, cy = min(candidates, key=cost)
        cx = int(np.clip(cx, 0, max(0, w - cw)))
        cy = int(np.clip(cy, bar_h + 2, max(bar_h + 2, h - ch)))
        taken.append((cx, cy, cx + cw, cy + ch))

        roi = img[cy:cy + ch, cx:cx + cw]
        roi[:] = (roi * 0.25).astype(np.uint8)
        cv2.rectangle(img, (cx, cy), (cx + cw, cy + ch), LIME, 1, AA)
        cv2.line(img, (cx, cy), (cx + int(30 * s), cy), LIME, max(2, int(3 * s)))
        for i, ln in enumerate(lines):
            color = LIME if i == 0 else WHITE
            cv2.putText(img, ln, (cx + pad, cy + pad + lh * (i + 1) - int(5 * s)), FONT, fs, color, 1, AA)

    @staticmethod
    def _draw_skeleton(img, kpts: np.ndarray, s: float) -> None:
        t = max(1, int(2 * s))
        for i, (a, b) in enumerate(SKELETON):
            if kpts[a, 2] >= 0.5 and kpts[b, 2] >= 0.5:
                pa = (int(kpts[a, 0]), int(kpts[a, 1]))
                pb = (int(kpts[b, 0]), int(kpts[b, 1]))
                cv2.line(img, pa, pb, LIMB_COLORS.get(i, WHITE), t, AA)
        r = max(2, int(3 * s))
        for x, y, c in kpts:
            if c >= 0.5:
                cv2.circle(img, (int(x), int(y)), r, BLACK, -1, AA)
                cv2.circle(img, (int(x), int(y)), r, WHITE, 1, AA)

    def _draw_marker(self, img, box: Box, conf: float, s: float, taken: list[Box]) -> None:
        x1, y1, x2, y2 = box
        h = y2 - y1
        cx = (x1 + x2) // 2
        r = int(np.clip(h * 0.14, 4 * s, 11 * s))
        cy = y1 + int(min(h * 0.3, r * 2.5))  # head / shoulders
        cv2.circle(img, (cx, cy), r + 2, BLACK, 2, AA)
        cv2.circle(img, (cx, cy), r, LIME, max(1, int(2 * s)), AA)
        cv2.circle(img, (cx, cy), max(1, r // 3), LIME, -1, AA)
        if h >= self.label_min_height * s:
            self._corners(img, box, LIME, s)
            self._tag(img, f"PERSON {conf:.0%}", (x1, y2 + int(4 * s)), LIME, s, below=True, taken=taken)

    # ============================================================ primitives
    def _draw_box(self, img, box: Box, text: str, color, s: float, thickness: int = 1, top: int = 0) -> None:
        x1, y1, x2, y2 = box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, int(thickness * s)), AA)
        self._tag(img, text, (x1, y1), color, s, top=top)

    @staticmethod
    def _corners(img, box: Box, color, s: float) -> None:
        x1, y1, x2, y2 = box
        ln = int(max(6, min(x2 - x1, y2 - y1) * 0.22))
        t = max(1, int(2 * s))
        for (px, py), (dx, dy) in (((x1, y1), (1, 1)), ((x2, y1), (-1, 1)),
                                   ((x1, y2), (1, -1)), ((x2, y2), (-1, -1))):
            cv2.line(img, (px, py), (px + dx * ln, py), color, t, AA)
            cv2.line(img, (px, py), (px, py + dy * ln), color, t, AA)

    @staticmethod
    def _tag(
        img, text: str, anchor: tuple[int, int], color, s: float,
        below: bool = False, taken: list[Box] | None = None, top: int = 0,
    ) -> None:
        fs = 0.42 * s
        (tw, th), _ = cv2.getTextSize(text, FONT, fs, 1)
        pad = int(4 * s)
        x, y = anchor
        y0 = y if below else y - th - 2 * pad
        y0 = max(top, min(img.shape[0] - th - 2 * pad, y0))  # never under the HUD bar
        x = max(0, min(img.shape[1] - tw - 2 * pad, x))
        rect = (x, y0, x + tw + 2 * pad, y0 + th + 2 * pad)
        if taken is not None:
            if any(_overlap(rect, r) > 0 for r in taken):
                return  # crowded: the marker / skeleton already identifies this person
            taken.append(rect)
        cv2.rectangle(img, rect[:2], rect[2:], color, -1)
        cv2.putText(img, text, (x + pad, y0 + th + pad - 1), FONT, fs, BLACK, 1, AA)

    @staticmethod
    def _panel_rect(lines: list[str], anchor: tuple[int, int, str], s: float, shape) -> Box:
        fs = 0.4 * s
        lh, pad = int(17 * s), int(8 * s)
        cw = max(cv2.getTextSize(ln, FONT, fs, 1)[0][0] for ln in lines) + 2 * pad
        ch = lh * len(lines) + 2 * pad
        x, y, corner = anchor
        x = x - cw if corner.endswith("right") else x
        y = y - ch
        x = max(0, min(shape[1] - cw, x))
        y = max(0, min(shape[0] - ch, y))
        return x, y, x + cw, y + ch

    @staticmethod
    def _panel(img, lines: list[str], rect: Box, s: float, *, title_color) -> None:
        fs = 0.4 * s
        lh, pad = int(17 * s), int(8 * s)
        x, y, x2, y2 = rect
        roi = img[y:y2, x:x2]
        roi[:] = (roi * 0.25).astype(np.uint8)
        cv2.rectangle(img, (x, y), (x2, y2), title_color, 1, AA)
        for i, ln in enumerate(lines):
            cv2.putText(img, ln, (x + pad, y + pad + lh * (i + 1) - int(5 * s)), FONT, fs,
                        title_color if i == 0 else WHITE, 1, AA)

    # ======================================================== density inset
    def _place_inset(self, shape, s: float, bar_h: int, result: FrameResult) -> Box:
        """Pick the corner where the inset hides the least (hazards weigh most)."""
        h, w = shape[:2]
        iw = int(w * self.inset_ratio)
        ih = int(iw * h / w)
        if ih > h * 0.38:  # portrait frames: cap height, keep aspect
            ih = int(h * 0.38)
            iw = int(ih * w / h)
        m = int(12 * s)
        top, bottom = bar_h + m, h - ih - m
        left, right = m, w - iw - m
        candidates = [(right, top), (left, top), (right, bottom), (left, bottom)]

        def cost(x0: int, y0: int) -> float:
            rect = (x0, y0, x0 + iw, y0 + ih)
            return (sum(_overlap(rect, hz.box) for hz in result.hazards) * 20
                    + sum(_overlap(rect, d.box) for d in result.detections))

        x0, y0 = min(candidates, key=lambda c: cost(*c))  # min() keeps first on ties
        return x0, y0, x0 + iw, y0 + ih

    def _draw_density_inset(self, img, clean, data: HudData, s: float, rect: Box) -> None:
        x0, y0, x1, y1 = rect
        iw, ih = x1 - x0, y1 - y0
        if iw <= 0 or ih <= 0:
            return
        mini = cv2.resize(clean, (iw, ih), interpolation=cv2.INTER_AREA)
        mini = (mini * 0.5).astype(np.float32)
        heat = cv2.resize(data.density_heatmap, (iw, ih)).astype(np.float32)
        # Per-pixel alpha from heat intensity: empty areas show the dimmed scene,
        # crowded areas glow (JET's dark-blue floor would otherwise flood the inset).
        intensity = cv2.resize(data.density_intensity, (iw, ih))[..., None]
        alpha = np.clip(intensity * 1.6, 0, 0.85)
        img[y0:y0 + ih, x0:x0 + iw] = (mini * (1 - alpha) + heat * alpha).astype(np.uint8)
        cv2.rectangle(img, (x0, y0), (x0 + iw, y0 + ih), WHITE, 1, AA)

        cap_h = int(20 * s)
        roi = img[y0 + ih - cap_h:y0 + ih, x0:x0 + iw]
        roi[:] = (roi * 0.35).astype(np.uint8)
        fs = 0.4 * s
        cv2.putText(img, "CROWD DENSITY", (x0 + int(6 * s), y0 + ih - int(6 * s)), FONT, fs, WHITE, 1, AA)
        level_color = {"HIGH": ORANGE, "MEDIUM": (0, 220, 255), "LOW": LIME}.get(data.density_level, GREY)
        (tw, _), _ = cv2.getTextSize(data.density_level, FONT, fs, 1)
        cv2.putText(img, data.density_level, (x0 + iw - tw - int(6 * s), y0 + ih - int(6 * s)),
                    FONT, fs, level_color, 1, AA)

    @staticmethod
    def _draw_no_signal(img, s: float) -> None:
        h, w = img.shape[:2]
        lines = [("NO SIGNAL", 1.1 * s, ORANGE, 2),
                 ("CAMERA PICTURE IS BLANK (OFF OR COVERED)", 0.5 * s, WHITE, 1),
                 ("OPEN THE PRIVACY SHUTTER / PRESS THE CAMERA KEY (F9)", 0.45 * s, GREY, 1)]
        y = h // 2 - int(30 * s)
        for text, fs, color, th in lines:
            (tw, tht), _ = cv2.getTextSize(text, FONT_BOLD, fs, th)
            cv2.putText(img, text, ((w - tw) // 2, y), FONT_BOLD, fs, color, th, AA)
            y += tht + int(16 * s)

    # =============================================================== top bar
    def _draw_top_bar(self, img, data: HudData, s: float, bar_h: int) -> None:
        roi = img[0:bar_h]
        roi[:] = (roi * 0.28).astype(np.uint8)
        cv2.line(img, (0, bar_h), (img.shape[1], bar_h), (60, 60, 60), 1)

        small, big = 0.38 * s, 0.95 * s
        col_w = int(118 * s)
        right_w = int(130 * s)
        metrics = [
            ("PERSONS", data.persons, LIME, data.persons / max(1, data.peak_persons)),
            ("OBJECTS", data.objects, CYAN, min(1.0, data.objects / 10)),
            ("TYPES", data.object_types, YELLOW, min(1.0, data.object_types / 10)),
        ]
        if data.show_hazards:
            metrics += [("SMOKE", data.smoke, MAGENTA, min(1.0, data.smoke / 3)),
                        ("FIRE", data.fire, ORANGE, min(1.0, data.fire / 3))]
        # Narrow (portrait) frames: shrink columns so all four metrics still fit.
        col_w = min(col_w, (img.shape[1] - right_w - int(12 * s)) // len(metrics))
        bar_w = max(0, col_w - int(56 * s))
        for i, (name, value, color, frac) in enumerate(metrics):
            x = int(12 * s) + i * col_w
            cv2.putText(img, name, (x, int(16 * s)), FONT, small, GREY, 1, AA)
            cv2.putText(img, str(value), (x, int(42 * s)), FONT_BOLD, big, color, max(1, int(2 * s)), AA)
            if bar_w >= int(16 * s):
                bx, by = x + int(42 * s), int(36 * s)
                cv2.line(img, (bx, by), (bx + bar_w, by), (70, 70, 70), max(1, int(3 * s)))
                if frac > 0:
                    cv2.line(img, (bx, by), (bx + int(bar_w * frac), by), color, max(1, int(3 * s)))

        for j, text in enumerate((f"PEAK {data.peak_persons}", f"{data.frame_ms:.0f} ms / frame")):
            (tw, _), _ = cv2.getTextSize(text, FONT, small, 1)
            cv2.putText(img, text, (img.shape[1] - tw - int(12 * s), int(20 * s) + j * int(18 * s)),
                        FONT, small, GREY, 1, AA)


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for wd in text.split(" "):
        if cur and len(cur) + 1 + len(wd) > width:
            lines.append(cur)
            cur = wd
        else:
            cur = f"{cur} {wd}".strip()
    if cur:
        lines.append(cur)
    return lines[:4]
