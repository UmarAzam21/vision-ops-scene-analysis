"""Plain result types shared by the analyzer, the HUD renderer and the API schemas."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from app.ml.detector import Box, Detection
from app.ml.hazard import Hazard
from app.ml.pose import Pose
from app.ml.scene import SceneResult



@dataclass(slots=True)
class PersonInsight:
    track_id: int | None
    box: Box
    confidence: float
    pose: Pose | None = field(default=None, repr=False)
    face_box: Box | None = None
    gender: str | None = None            # male | female | uncertain | None (no face seen)
    gender_confidence: float = 0.0
    age: int | None = None               # estimated years
    expression: str | None = None
    expression_confidence: float = 0.0
    posture: str | None = None
    motion: str | None = None
    facing: str | None = None
    gestures: list[str] = field(default_factory=list)
    activities: list[str] = field(default_factory=list)
    holding: list[str] = field(default_factory=list)
    clothing: str | None = None
    accessories: list[str] = field(default_factory=list)  # worn items, e.g. tie

    @property
    def who(self) -> str:
        noun = {"male": "Man", "female": "Woman"}.get(self.gender or "", "Person")
        kid = None
        if self.age is not None:
            kid = "baby" if self.age < 3 else "child" if self.age < 13 else "teenager" if self.age < 18 else None
        if kid:
            noun = {"male": f"{kid} (boy)", "female": f"{kid} (girl)"}.get(self.gender or "", kid)
            noun = noun[0].upper() + noun[1:]
        return f"{noun}, ~{self.age} yrs" if self.age is not None else noun

    def doing(self) -> list[str]:
        """Most specific facts first."""
        out = list(self.activities)
        if self.posture and self.posture != "unknown":
            out.append(self.posture)
        if self.motion:
            out.append(self.motion)
        out += self.gestures
        if self.facing and self.facing != "unknown":
            out.append(self.facing)
        if self.expression and self.expression_confidence >= 0.4:
            out.append(f"{self.expression} expression")
        return out

    def description(self) -> str:
        text = self.who
        doing = self.doing()
        if doing:
            text += " — " + ", ".join(doing)
        extras = []
        worn = []
        if self.clothing:
            article = "an" if self.clothing[0] in "aeiou" else "a"
            worn.append(f"{article} {self.clothing} top")
        worn += [f"a {item}" for item in self.accessories]
        if worn:
            extras.append("wearing " + " and ".join(worn))
        if self.holding:
            extras.append("with " + ", ".join(self.holding))
        if extras:
            text += ". " + "; ".join(extras).capitalize()
        return text + "."


@dataclass(slots=True)
class FrameResult:
    width: int
    height: int
    detections: list[Detection]          # everything with a box (persons + objects)
    hazards: list[Hazard]
    counts: dict[str, int]
    persons: int
    peak_persons: int
    density_level: str
    people: list[PersonInsight] = field(default_factory=list)
    scene: SceneResult | None = field(default=None, repr=False)
    density_grid: np.ndarray | None = field(default=None, repr=False)
    inference_ms: float = 0.0
    total_ms: float = 0.0
    no_signal: bool = False              # frame is flat black (shutter / covered lens)

    @property
    def fire(self) -> int:
        return sum(1 for h in self.hazards if h.kind == "fire")

    @property
    def smoke(self) -> int:
        return sum(1 for h in self.hazards if h.kind == "smoke")

    @property
    def objects(self) -> int:
        return sum(1 for d in self.detections if d.label != "person")
