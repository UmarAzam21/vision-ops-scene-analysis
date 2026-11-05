"""Video stream sessions (uploaded files, webcams, RTSP/HTTP cameras)."""

from __future__ import annotations

import logging
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

from app.core.config import Settings
from app.services.analyzer import FrameAnalyzer, FrameResult
from app.services.models import ModelRegistry
from app.utils.image import encode_jpeg, is_blank, resize_max_width

log = logging.getLogger(__name__)


class StreamError(RuntimeError):
    pass


class StreamBusyError(StreamError):
    pass


def _camera_backends() -> list[tuple[int, str]]:
    if sys.platform == "win32":
        # MSMF first: Intel IPU / MIPI laptop cameras (common on Dell, Lenovo, HP) only deliver
        # real images through Media Foundation - DirectShow opens them but returns black frames.
        return [(cv2.CAP_MSMF, "Media Foundation"), (cv2.CAP_DSHOW, "DirectShow"), (cv2.CAP_ANY, "default")]
    if sys.platform == "linux":
        return [(cv2.CAP_V4L2, "V4L2"), (cv2.CAP_ANY, "default")]
    return [(cv2.CAP_ANY, "default")]


def open_camera(index: int, *, probe_seconds: float = 4.0) -> cv2.VideoCapture:
    """Open a local camera with the first backend that delivers a *real* image.

    Many cameras need a moment for auto-exposure, so each backend gets a few seconds;
    a backend that only produces flat black frames is rejected rather than streamed.
    """
    attempts: list[str] = []
    for api, name in _camera_backends():
        cap = cv2.VideoCapture(index, api)
        if not cap.isOpened():
            attempts.append(f"{name}: could not open")
            cap.release()
            continue
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        deadline = time.monotonic() + probe_seconds
        got_frame = False
        while time.monotonic() < deadline:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue
            got_frame = True
            if not is_blank(frame):
                log.info("Camera %d opened via %s (%dx%d)", index, name, frame.shape[1], frame.shape[0])
                return cap
        cap.release()
        attempts.append(f"{name}: {'only black frames' if got_frame else 'no frames'}")
        log.warning("Camera %d via %s rejected: %s", index, name, attempts[-1])

    detail = "; ".join(attempts)
    if any("black" in a for a in attempts):
        raise StreamError(
            f"Camera {index} opens but only sends black frames ({detail}). Check: the physical "
            "privacy shutter, the camera on/off key (often F9 or Fn+F9 on Dell), and that no other "
            "app (Teams, Zoom, Camera, another browser tab) is using the camera."
        )
    raise StreamError(f"Camera {index} is not available ({detail}). Is it in use by another app?")


@dataclass(slots=True)
class StreamStats:
    frames: int = 0
    fps: float = 0.0
    last: FrameResult | None = None


class StreamSession:
    """One video source + its own stateful analyser. A single viewer at a time."""

    def __init__(
        self,
        source: str | int,
        analyzer: FrameAnalyzer,
        settings: Settings,
        *,
        owned_file: Path | None = None,
    ) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.source = source
        self.analyzer = analyzer
        self.settings = settings
        self.owned_file = owned_file
        self.is_file = owned_file is not None
        self.stats = StreamStats()
        self.closed = False
        self.error: str | None = None

        self._cap: cv2.VideoCapture | None = None
        self._source_fps = 25.0
        self._viewer_lock = threading.Lock()
        self._fps_t = time.perf_counter()
        self._fps_frames = 0
        self._play_t0: float | None = None  # wall-clock start of file playback
        self._pos = 0                       # index of the next frame to read

    # -------------------------------------------------------------- lifecycle
    def open(self) -> None:
        cap = open_camera(self.source) if isinstance(self.source, int) else cv2.VideoCapture(self.source)
        if not cap.isOpened():
            raise StreamError(f"Unable to open video source: {self.source!r}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        self._source_fps = fps if 1 <= fps <= 120 else 25.0
        self._cap = cap

    def release_capture(self) -> None:
        """Free a live device as soon as nobody is watching, so the webcam isn't held
        hostage by an abandoned browser tab. It is reopened lazily on the next view."""
        if not self.is_file and self._cap is not None:
            self._cap.release()
            self._cap = None

    def acquire_viewer(self) -> None:
        if not self._viewer_lock.acquire(blocking=False):
            raise StreamBusyError("Stream already has an active viewer")

    def release_viewer(self) -> None:
        if self._viewer_lock.locked():
            self._viewer_lock.release()

    @property
    def frame_interval(self) -> float:
        return 1.0 / self._source_fps if self.is_file else 0.0

    def close(self) -> None:
        self.closed = True
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.owned_file and self.owned_file.exists():
            try:
                self.owned_file.unlink()
            except OSError:
                log.warning("Could not delete %s", self.owned_file)

    # ------------------------------------------------------------- processing
    def next_frame(self) -> bytes | None:
        """Read → analyse → render → JPEG. Returns None when the source ends."""
        if self.closed:
            return None
        if self._cap is None:
            self.open()
        assert self._cap is not None

        if self.is_file:
            self._skip_to_realtime()
        ok, frame = self._cap.read()
        if not ok and self.is_file:  # loop uploaded videos for continuous monitoring
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self._play_t0, self._pos = time.perf_counter(), 0
            ok, frame = self._cap.read()
        if not ok or frame is None:
            return None
        self._pos += 1

        frame = resize_max_width(frame, self.settings.stream_max_width)
        result = self.analyzer.analyze(frame)
        annotated = self.analyzer.render(frame, result)
        self._update_stats(result)
        return encode_jpeg(annotated, self.settings.jpeg_quality)

    def _skip_to_realtime(self) -> None:
        """Analysis is slower than the video's FPS, so drop frames to stay live:
        what you see always matches the video's real timeline (like a live camera)."""
        assert self._cap is not None
        now = time.perf_counter()
        if self._play_t0 is None:
            self._play_t0 = now
            return
        target = int((now - self._play_t0) * self._source_fps)
        for _ in range(min(target - self._pos, int(self._source_fps * 2))):
            if not self._cap.grab():  # grab() skips decoding cost of the pixels
                break
            self._pos += 1

    def _update_stats(self, result: FrameResult) -> None:
        self.stats.frames += 1
        self.stats.last = result
        self._fps_frames += 1
        now = time.perf_counter()
        if now - self._fps_t >= 1.0:
            self.stats.fps = self._fps_frames / (now - self._fps_t)
            self._fps_t, self._fps_frames = now, 0


class StreamManager:
    def __init__(
        self,
        models: ModelRegistry,
        settings: Settings,
    ) -> None:
        self.models = models
        self.settings = settings
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def create(self, source: str | int, owned_file: Path | None = None) -> StreamSession:
        with self._lock:
            if len(self._sessions) >= self.settings.max_streams:
                # Evict the oldest session so the UI never dead-ends.
                oldest_id = next(iter(self._sessions))
                self._sessions.pop(oldest_id).close()

            analyzer = FrameAnalyzer(self.models, self.settings, temporal=True)
            session = StreamSession(source, analyzer, self.settings, owned_file=owned_file)
            try:
                session.open()
            except StreamError:
                session.close()
                raise
            self._sessions[session.id] = session
            log.info("Stream %s opened (%s)", session.id, source)
            return session

    @property
    def count(self) -> int:
        return len(self._sessions)

    def get(self, stream_id: str) -> StreamSession | None:
        return self._sessions.get(stream_id)

    def remove(self, stream_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(stream_id, None)
        if session is None:
            return False
        session.close()
        log.info("Stream %s closed", stream_id)
        return True

    def shutdown(self) -> None:
        for sid in list(self._sessions):
            self.remove(sid)
