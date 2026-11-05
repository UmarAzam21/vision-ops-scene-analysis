"""Frame analysis pipeline.

    ┌ objects (YOLO, 80 classes) ┐
    ├ pose (YOLO-pose, 17 kpts)  ┤──► persons ─► tracker ─► faces / gender / age / expression
    ├ scene (SegFormer, async)   ┤               └────────► posture / motion / gestures / holding / clothing
    └ fire & smoke               ┘──► crowd density ──► HUD

Heavy models run in parallel threads (OpenCV DNN releases the GIL). In video mode
the slow scene model runs asynchronously and the others run every N frames, so the
stream stays live.
"""

from __future__ import annotations

import logging
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from app.core.config import Settings
from app.ml import activity
from app.ml.density import CrowdDensityEstimator
from app.ml.detector import Box, Detection
from app.ml.face import FaceInfo
from app.ml.hazard import Hazard, HazardDetector, HeuristicHazardDetector, ModelHazardDetector
from app.ml.labels import COCO_LABELS, PERSON_LABEL
from app.ml.pose import Pose
from app.ml.scene import SceneResult
from app.services.models import ModelRegistry
from app.services.results import FrameResult, PersonInsight
from app.services.tracker import PersonTracker, Track
from app.utils.image import is_blank
from app.vision.hud import HudData, HudRenderer

log = logging.getLogger(__name__)
_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="cv")

__all__ = ["FrameAnalyzer", "FrameResult", "PersonInsight"]


COCO_SET = frozenset(COCO_LABELS)
PERSON_SYNONYMS = frozenset({
    "man", "woman", "boy", "girl", "child", "kid", "baby", "people", "pedestrian", "human",
    "human face", "face", "adult", "lady", "gentleman", "player", "crowd", "toddler", "passenger",
    "worker", "student", "soldier", "athlete", "tourist", "rider", "driver", "spectator",
})


def _iou(a: Box, b: Box) -> float:
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def _ioa(inner: Box, outer: Box) -> float:
    ix = max(0, min(inner[2], outer[2]) - max(inner[0], outer[0]))
    iy = max(0, min(inner[3], outer[3]) - max(inner[1], outer[1]))
    return ix * iy / max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))


class FrameAnalyzer:
    """Stateful analyser. One per video stream (temporal=True) or per image request."""

    def __init__(
        self, models: ModelRegistry, settings: Settings, *, temporal: bool, max_recall: bool = False,
    ) -> None:
        self.models = models
        self.settings = settings
        self.temporal = temporal
        self.max_recall = max_recall and not temporal
        self.hazard_detector: HazardDetector | None = None
        if settings.enable_hazards:
            self.hazard_detector = (
                ModelHazardDetector(models.hazard) if models.hazard
                else HeuristicHazardDetector(temporal=temporal)
            )
        self._uses_heuristic = isinstance(self.hazard_detector, HeuristicHazardDetector)
        self.density = CrowdDensityEstimator(smoothing=0.35 if temporal else 0.0)
        self.tracker = PersonTracker()
        self.hud = HudRenderer()
        self.peak_persons = 0

        self._idx = 0
        self._objects: list[Detection] = []
        self._hazards: list[Hazard] = []
        self._scene: SceneResult | None = None
        self._scene_future: Future | None = None
        self._scene_t = 0.0

    # ================================================================ analyse
    def analyze(self, frame: np.ndarray) -> FrameResult:
        t0 = time.perf_counter()
        if is_blank(frame):
            self._scene = None  # segmenting a blank picture yields nonsense like 'sky 100%'
            h, w = frame.shape[:2]
            return FrameResult(
                width=w, height=h, detections=[], hazards=[], counts={}, persons=0,
                peak_persons=self.peak_persons, density_level="NONE",
                total_ms=(time.perf_counter() - t0) * 1000, no_signal=True,
            )
        self._idx += 1
        m, s = self.models, self.settings
        every = lambda n: not self.temporal or n <= 1 or self._idx % n == 1  # noqa: E731

        # Photos: the accurate model, tiled for small/far objects. Video: the fast model.
        detector = m.image_detector if (not self.temporal and m.image_detector) else m.detector
        tiles = 1 if self.temporal else (s.max_recall_tiles if self.max_recall else s.image_tiles)
        conf = s.max_recall_conf if self.max_recall else None
        run_objects = every(s.object_every_n) or not self._objects
        obj_f = _POOL.submit(detector.detect, frame, tiles=tiles, conf=conf) if run_objects else None
        ov_f = (_POOL.submit(m.open_vocab.detect, frame)
                if m.open_vocab and (not self.temporal or s.open_vocab_in_video) and run_objects else None)
        pose_f = _POOL.submit(m.pose.detect, frame) if m.pose else None
        face_f = (_POOL.submit(m.faces.analyze, frame, self._need_demographics())
                  if m.faces and every(s.face_every_n) else None)
        no_signal = False
        self._update_scene(frame)

        if obj_f is not None:
            self._objects = obj_f.result()
            if ov_f is not None:
                self._objects = self._objects + self._new_types(ov_f.result(), self._objects)
        poses: list[Pose] = pose_f.result() if pose_f else []
        faces: list[FaceInfo] | None = face_f.result() if face_f else None
        inference_ms = (time.perf_counter() - t0) * 1000

        objects = [d for d in self._objects if d.label != PERSON_LABEL]
        detector_persons = [d for d in self._objects if d.label == PERSON_LABEL]
        persons, pose_of = self._combine_persons(detector_persons, poses, fresh=obj_f is not None)
        detections = sorted(persons + objects, key=lambda d: d.confidence, reverse=True)

        if self.hazard_detector and every(s.object_every_n):
            self._hazards = self.hazard_detector.detect(frame)
        hazards = self._hazards
        if self._uses_heuristic:
            hazards = self._suppress_clothing(hazards, persons)

        people = self._people(frame, persons, pose_of, objects, faces)

        grid = None
        if persons:
            body = float(np.median([d.height for d in persons]))
            grid = self.density.update((d.center for d in persons), frame.shape, body)
        self.peak_persons = max(self.peak_persons, len(persons))

        h, w = frame.shape[:2]
        return FrameResult(
            width=w, height=h,
            detections=detections,
            hazards=hazards,
            counts=dict(Counter(d.label for d in detections).most_common()),
            persons=len(persons),
            peak_persons=self.peak_persons,
            density_level=self.density.level(grid),
            people=people,
            scene=self._scene,
            density_grid=grid,
            inference_ms=inference_ms,
            total_ms=(time.perf_counter() - t0) * 1000,
            no_signal=no_signal,
        )

    # ------------------------------------------------------------- people
    def _people(
        self,
        frame: np.ndarray,
        persons: list[Detection],
        pose_of: list[Pose | None],
        objects: list[Detection],
        faces: list[FaceInfo] | None,
    ) -> list[PersonInsight]:
        boxes = [p.box for p in persons]
        if self.temporal:
            tracks = self.tracker.update(boxes, time.monotonic())
        else:
            tracks = [Track(i + 1, b, 0) for i, b in enumerate(boxes)]
        min_gender_samples = 3 if self.temporal else 1
        fresh = faces is not None

        insights: list[PersonInsight] = []
        claimed: set[int] = set()
        for det, track, pose in zip(persons, tracks, pose_of):
            face = self._match_face(det.box, faces or [], claimed)
            if face is not None:
                track.add_face(face.gender_probs, face.age, face.expression_probs)
                track.face_box = face.box
            elif fresh:  # faces were analysed this frame and none matched
                track.face_box = None

            ins = PersonInsight(
                track_id=track.id if self.temporal else None,
                box=det.box,
                confidence=det.confidence,
                pose=pose,
                face_box=track.face_box,
            )
            ins.gender, ins.gender_confidence = track.gender(min_samples=min_gender_samples)
            ins.age = track.age()
            ins.expression, ins.expression_confidence = track.expression_label()

            if pose is not None:
                track.postures.append(activity.posture(pose))
                ins.posture = track.posture()
                ins.facing = activity.facing(pose)
                ins.gestures = activity.gestures(pose)
                ins.holding, ins.activities = activity.interactions(pose, objects)
                ins.accessories = activity.worn_items(pose, objects)
                if fresh or not self.temporal:  # colour is stable – no need every frame
                    color = activity.upper_clothing_color(frame, pose)
                    if color:
                        track.clothing_votes.append(color)
                ins.clothing = track.clothing()
            ins.motion = track.motion() if self.temporal else None
            insights.append(ins)
        return insights

    def _combine_persons(
        self, detected: list[Detection], poses: list[Pose], *, fresh: bool,
    ) -> tuple[list[Detection], list[Pose | None]]:
        """One list of people from both models, each with its skeleton if the pose model saw it.

        Photos trust the (larger, tiled) object detector for *who is there* and attach
        skeletons by overlap. Video trusts the per-frame pose model (the detector may be a
        frame old) and adds people only the detector found on frames where it ran.
        """
        if self.temporal:
            primary = [Detection(0, PERSON_LABEL, p.confidence, p.box) for p in poses]
            pose_of: list[Pose | None] = list(poses)
            extra = detected if fresh else []
        else:
            primary, pose_of, extra = list(detected), [None] * len(detected), []
            used: set[int] = set()
            for i, d in enumerate(primary):
                best, best_iou = None, 0.5
                for j, p in enumerate(poses):
                    if j not in used and (v := _iou(d.box, p.box)) > best_iou:
                        best, best_iou = j, v
                if best is not None:
                    used.add(best)
                    pose_of[i] = poses[best]
            # people only the pose model found (rare) are still people
            for j, p in enumerate(poses):
                if j not in used and all(_iou(p.box, d.box) < 0.5 for d in primary):
                    primary.append(Detection(0, PERSON_LABEL, p.confidence, p.box))
                    pose_of.append(p)
        for d in extra:
            if all(_iou(d.box, q.box) < 0.5 for q in primary):
                primary.append(d)
                pose_of.append(None)
        return primary, pose_of

    @staticmethod
    def _new_types(open_vocab: list[Detection], known: list[Detection]) -> list[Detection]:
        """Keep open-vocabulary detections that add something: a type the 80-class model
        doesn't have, and not just another name for an object it already boxed
        (e.g. 'man' for a person, 'sedan' for a car)."""
        known_labels = {d.label for d in known} | set(COCO_SET)
        out = []
        for d in open_vocab:
            if d.label in known_labels or d.label in PERSON_SYNONYMS:
                continue
            if any(_iou(d.box, k.box) > 0.7 for k in known):
                continue
            out.append(Detection(1000 + d.class_id, d.label, d.confidence, d.box))
        return out

    def _need_demographics(self) -> bool:
        """Gender/age nets are the heaviest per-face models. Once every visible person
        has enough votes, only re-check occasionally."""
        if not self.temporal or not self.tracker.tracks:
            return True
        if self._idx % 60 == 1:
            return True
        return any(t.gender_samples < 10 for t in self.tracker.tracks.values())

    @staticmethod
    def _match_face(box: Box, faces: list[FaceInfo], claimed: set[int]) -> FaceInfo | None:
        """The face must sit inside the upper part of the person box."""
        x1, y1, x2, y2 = box
        best, best_i = None, -1
        for i, f in enumerate(faces):
            if i in claimed:
                continue
            cx, cy = f.center
            if x1 <= cx <= x2 and y1 <= cy <= y1 + (y2 - y1) * 0.6 and _ioa(f.box, box) > 0.7:
                if best is None or f.score > best.score:
                    best, best_i = f, i
        if best is not None:
            claimed.add(best_i)
        return best

    # -------------------------------------------------------------- scene
    def _update_scene(self, frame: np.ndarray) -> None:
        scene = self.models.scene
        if scene is None:
            return
        if not self.temporal:
            self._scene = scene.segment(frame)
            return
        fut = self._scene_future
        if fut is not None and fut.done():
            try:
                self._scene = fut.result()
            except Exception:  # noqa: BLE001 – never kill the stream for the scene layer
                log.exception("scene segmentation failed")
            self._scene_future = fut = None
        now = time.monotonic()
        if fut is None and now - self._scene_t >= self.settings.scene_interval_s:
            self._scene_t = now
            self._scene_future = _POOL.submit(scene.segment, frame.copy())

    # ============================================================== render
    def render(self, frame: np.ndarray, result: FrameResult) -> np.ndarray:
        heat = intensity = None
        if result.density_grid is not None:
            gh, gw = result.density_grid.shape
            size = (gw * 2, gh * 2)
            heat = self.density.colorize(result.density_grid, size)
            intensity = self.density.normalize(result.density_grid, size)
        scene_color = person_mask = None
        if result.scene is not None and self.models.scene is not None:
            scene_color, person_mask = self.models.scene.colorize(result.scene, (result.width, result.height))
        data = HudData(
            persons=result.persons,
            objects=result.objects,
            object_types=len(result.counts),
            show_hazards=self.hazard_detector is not None,
            peak_persons=result.peak_persons,
            fire=result.fire,
            smoke=result.smoke,
            frame_ms=result.total_ms,
            density_level=result.density_level,
            density_heatmap=heat,
            density_intensity=intensity,
            scene_color=scene_color if self.settings.scene_overlay else None,
            person_mask=person_mask,
        )
        return self.hud.render(frame, result, data)

    @staticmethod
    def _suppress_clothing(hazards: list[Hazard], persons: list[Detection]) -> list[Hazard]:
        """Colour heuristics fire on orange/red clothing (e.g. hi-vis vests).
        Drop 'fire' blobs that sit almost entirely inside a detected person."""
        if not persons:
            return hazards
        return [hz for hz in hazards if hz.kind != "fire" or max(_ioa(hz.box, p.box) for p in persons) < 0.6]
