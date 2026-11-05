from __future__ import annotations

from pydantic import BaseModel, Field

from app.ml.detector import Box
from app.ml.pose import KEYPOINTS
from app.services.results import FrameResult, PersonInsight


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int

    @classmethod
    def of(cls, b: Box) -> BoundingBox:
        return cls(x1=b[0], y1=b[1], x2=b[2], y2=b[3])


class DetectionOut(BaseModel):
    label: str
    class_id: int
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class HazardOut(BaseModel):
    kind: str
    confidence: float = Field(ge=0, le=1)
    box: BoundingBox


class KeypointOut(BaseModel):
    name: str
    x: float
    y: float
    visibility: float


class PersonOut(BaseModel):
    track_id: int | None = Field(None, description="Stable ID across video frames")
    description: str = Field(..., description="Human-readable summary of everything below")
    confidence: float
    box: BoundingBox
    face_box: BoundingBox | None = None
    gender: str | None = Field(None, description="Perceived gender estimate: male | female | uncertain")
    gender_confidence: float = 0.0
    age: int | None = Field(None, description="Estimated age in years (appearance-based)")
    expression: str | None = None
    expression_confidence: float = 0.0
    posture: str | None = None
    motion: str | None = Field(None, description="still | walking | running (video only)")
    facing: str | None = None
    gestures: list[str] = []
    activities: list[str] = []
    holding: list[str] = []
    clothing_color: str | None = None
    accessories: list[str] = []
    keypoints: list[KeypointOut] = []

    @classmethod
    def of(cls, p: PersonInsight, with_keypoints: bool = False) -> PersonOut:
        kps = []
        if with_keypoints and p.pose is not None:
            kps = [KeypointOut(name=n, x=round(float(x), 1), y=round(float(y), 1), visibility=round(float(c), 2))
                   for n, (x, y, c) in zip(KEYPOINTS, p.pose.keypoints)]
        return cls(
            track_id=p.track_id,
            description=p.description(),
            confidence=round(p.confidence, 3),
            box=BoundingBox.of(p.box),
            face_box=BoundingBox.of(p.face_box) if p.face_box else None,
            gender=p.gender,
            gender_confidence=round(p.gender_confidence, 3),
            age=p.age,
            expression=p.expression,
            expression_confidence=round(p.expression_confidence, 3),
            posture=p.posture,
            motion=p.motion,
            facing=p.facing,
            gestures=p.gestures,
            activities=p.activities,
            holding=p.holding,
            clothing_color=p.clothing,
            accessories=p.accessories,
            keypoints=kps,
        )


class SceneRegionOut(BaseModel):
    label: str
    fraction: float


class SceneOut(BaseModel):
    setting: str = Field(..., description="indoor | outdoor | unknown")
    regions: list[SceneRegionOut]


class Summary(BaseModel):
    persons: int
    peak_persons: int
    objects: int
    fire: int
    smoke: int
    density_level: str
    counts: dict[str, int]
    people: list[PersonOut]
    scene: SceneOut | None
    inference_ms: float
    total_ms: float
    no_signal: bool = False


class DetectionResponse(BaseModel):
    width: int
    height: int
    summary: Summary
    detections: list[DetectionOut]
    hazards: list[HazardOut]
    annotated_image: str | None = Field(
        default=None, description="JPEG data URL, only when include_image=true"
    )

    @classmethod
    def from_result(cls, r: FrameResult, annotated_image: str | None = None) -> DetectionResponse:
        return cls(
            width=r.width,
            height=r.height,
            summary=summary_from_result(r, with_keypoints=True),
            detections=[
                DetectionOut(label=d.label, class_id=d.class_id, confidence=round(d.confidence, 4),
                             box=BoundingBox.of(d.box))
                for d in r.detections
            ],
            hazards=[
                HazardOut(kind=h.kind, confidence=round(h.confidence, 4), box=BoundingBox.of(h.box))
                for h in r.hazards
            ],
            annotated_image=annotated_image,
        )


def summary_from_result(r: FrameResult, with_keypoints: bool = False) -> Summary:
    scene = None
    if r.scene is not None:
        scene = SceneOut(
            setting=r.scene.setting,
            regions=[SceneRegionOut(label=x.label, fraction=round(x.fraction, 3)) for x in r.scene.regions],
        )
    return Summary(
        persons=r.persons,
        peak_persons=r.peak_persons,
        objects=r.objects,
        fire=r.fire,
        smoke=r.smoke,
        density_level=r.density_level,
        counts=r.counts,
        people=[PersonOut.of(p, with_keypoints) for p in r.people],
        scene=scene,
        inference_ms=round(r.inference_ms, 1),
        total_ms=round(r.total_ms, 1),
        no_signal=r.no_signal,
    )
