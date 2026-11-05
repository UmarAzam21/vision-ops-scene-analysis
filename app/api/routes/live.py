"""Browser-camera live analysis over a WebSocket.

The browser opens the webcam (getUserMedia - the same path video-call apps use, which
works with laptop IPU/MIPI cameras that OpenCV cannot read), sends JPEG frames, and gets
back the annotated frame plus live stats. One frame in flight at a time, so a slow CPU
never builds a backlog: the stream simply runs at whatever FPS the analysis allows.

Protocol (per frame):
    client -> server : binary JPEG
    server -> client : binary JPEG (annotated), then text JSON {"summary": ..., "fps": ...}
    server -> client : text JSON {"error": "..."} for a bad frame (connection stays open)
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, status
from fastapi.concurrency import run_in_threadpool

from app.core.config import get_settings
from app.schemas.detection import summary_from_result
from app.services.analyzer import FrameAnalyzer
from app.utils.image import decode_image, encode_jpeg, resize_max_width

router = APIRouter(tags=["live"])
log = logging.getLogger(__name__)
MAX_FRAME_BYTES = 4 * 1024 * 1024


@router.websocket("/live")
async def live(ws: WebSocket) -> None:
    settings = get_settings()
    state = ws.app.state
    active = getattr(state, "live_sessions", 0)
    if active >= settings.max_streams:
        await ws.close(code=status.WS_1013_TRY_AGAIN_LATER, reason="Too many live sessions")
        return
    state.live_sessions = active + 1
    await ws.accept()
    analyzer = FrameAnalyzer(state.models, settings, temporal=True)
    frames, t_fps, fps = 0, time.perf_counter(), 0.0

    def process(data: bytes) -> tuple[bytes, dict]:
        frame = resize_max_width(decode_image(data), settings.stream_max_width)
        result = analyzer.analyze(frame)
        jpeg = encode_jpeg(analyzer.render(frame, result), settings.jpeg_quality)
        return jpeg, summary_from_result(result).model_dump()

    try:
        while True:
            data = await ws.receive_bytes()
            if len(data) > MAX_FRAME_BYTES:
                await ws.send_json({"error": "Frame too large"})
                continue
            try:
                jpeg, summary = await run_in_threadpool(process, data)
            except ValueError as exc:  # undecodable frame
                await ws.send_json({"error": str(exc)})
                continue
            frames += 1
            now = time.perf_counter()
            if now - t_fps >= 1.0:
                fps, t_fps = frames / (now - t_fps), now
                frames = 0
            await ws.send_bytes(jpeg)
            await ws.send_json({"summary": summary, "fps": round(fps, 1)})
    except WebSocketDisconnect:
        pass
    finally:
        state.live_sessions = max(0, getattr(state, "live_sessions", 1) - 1)
