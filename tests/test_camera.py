import cv2
import numpy as np
import pytest

from app.services import streams
from app.services.streams import StreamError, open_camera
from app.utils.image import is_blank

BLACK = np.zeros((480, 640, 3), np.uint8)
REAL = np.random.default_rng(0).integers(0, 255, (480, 640, 3), dtype=np.uint8)


class FakeCapture:
    """Simulates one backend: `frames` decides what read() returns."""

    behaviour: dict[int, str] = {}
    released: list[int] = []

    def __init__(self, index, api):
        self.api = api
        self.mode = self.behaviour.get(api, "closed")

    def isOpened(self):
        return self.mode != "closed"

    def set(self, *_):
        return True

    def get(self, *_):
        return 30.0

    def read(self):
        if self.mode == "black":
            return True, BLACK.copy()
        if self.mode == "real":
            return True, REAL.copy()
        return False, None

    def release(self):
        FakeCapture.released.append(self.api)


@pytest.fixture
def fake_cv(monkeypatch):
    FakeCapture.released = []
    monkeypatch.setattr(streams.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(streams, "_camera_backends", lambda: [(1, "A"), (2, "B"), (3, "C")])
    return FakeCapture


def test_is_blank():
    assert is_blank(BLACK)
    assert not is_blank(REAL)
    dark_room = np.full((480, 640, 3), 6, np.uint8)
    dark_room[100:200, 100:300] = 60  # dim but real content
    assert not is_blank(dark_room)
    placeholder = np.full((480, 640, 3), (60, 70, 70), np.uint8)   # Windows "camera off" image
    cv2.line(placeholder, (290, 270), (350, 210), (230, 230, 230), 6)
    cv2.rectangle(placeholder, (290, 215), (345, 265), (230, 230, 230), 3)
    assert is_blank(placeholder)


def test_skips_backend_with_black_frames(fake_cv):
    fake_cv.behaviour = {1: "black", 2: "real"}
    cap = open_camera(0, probe_seconds=0.2)
    assert cap.api == 2
    assert 1 in fake_cv.released  # the black backend was closed, not leaked


def test_all_black_gives_actionable_error(fake_cv):
    fake_cv.behaviour = {1: "black", 2: "black", 3: "closed"}
    with pytest.raises(StreamError, match="privacy shutter"):
        open_camera(0, probe_seconds=0.2)


def test_camera_missing_or_busy(fake_cv):
    fake_cv.behaviour = {}
    with pytest.raises(StreamError, match="not available"):
        open_camera(0, probe_seconds=0.2)
