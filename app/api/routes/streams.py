"""Live video endpoints: create a stream, watch it as MJPEG, poll its stats."""

import asyncio
import re
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import StreamingResponse

from app.api.deps import SettingsDep, StreamsDep
from app.schemas.detection import summary_from_result
from app.schemas.stream import SourceRequest, StreamCreated, StreamStatsOut
from app.services.streams import StreamBusyError, StreamError, StreamSession

router = APIRouter(prefix="/streams", tags=["streams"])

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
URL_SOURCE = re.compile(r"^(rtsp|rtsps|http|https)://", re.IGNORECASE)
BOUNDARY = "frame"


def _created(request: Request, session: StreamSession) -> StreamCreated:
    return StreamCreated(
        stream_id=session.id,
        video_url=str(request.url_for("stream_video", stream_id=session.id)),
        stats_url=str(request.url_for("stream_stats", stream_id=session.id)),
    )


def _get_session(streams: StreamsDep, stream_id: str) -> StreamSession:
    session = streams.get(stream_id)
    if session is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stream not found")
    return session


@router.post("/upload", response_model=StreamCreated, status_code=status.HTTP_201_CREATED)
async def create_from_upload(
    request: Request, streams: StreamsDep, settings: SettingsDep, file: UploadFile = File(...)
) -> StreamCreated:
    ext = Path(file.filename or "").suffix.lower()
    if ext not in VIDEO_EXTENSIONS:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported video type; allowed: {', '.join(sorted(VIDEO_EXTENSIONS))}")

    settings.upload_dir.mkdir(parents=True, exist_ok=True)
    dest = settings.upload_dir / f"{uuid.uuid4().hex}{ext}"  # never trust client filenames
    limit = settings.max_video_mb * 1024 * 1024

    def _save() -> int:
        written = 0
        with dest.open("wb") as out:
            while chunk := file.file.read(1024 * 1024):
                written += len(chunk)
                if written > limit:
                    break
                out.write(chunk)
        return written

    if await run_in_threadpool(_save) > limit:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                            f"Video exceeds {settings.max_video_mb} MB")
    try:
        session = await run_in_threadpool(streams.create, str(dest), dest)
    except StreamError as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _created(request, session)


@router.post("/source", response_model=StreamCreated, status_code=status.HTTP_201_CREATED)
async def create_from_source(
    request: Request, body: SourceRequest, streams: StreamsDep, settings: SettingsDep
) -> StreamCreated:
    raw = body.source.strip()
    source: str | int
    if raw.isdigit():
        if not settings.allow_device_sources:
            raise HTTPException(status.HTTP_403_FORBIDDEN, "Camera devices are disabled")
        source = int(raw)
    elif URL_SOURCE.match(raw):
        if not settings.allow_url_sources:
            raise HTTPException(status.HTTP_403_FORBIDDEN,
                                "URL sources are disabled (set ALLOW_URL_SOURCES=true)")
        source = raw
    else:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Source must be a camera index or rtsp/http(s) URL")
    try:
        session = await run_in_threadpool(streams.create, source)
    except StreamError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc
    return _created(request, session)


@router.get("/{stream_id}/video", name="stream_video", summary="Annotated MJPEG stream")
async def stream_video(stream_id: str, request: Request, streams: StreamsDep) -> StreamingResponse:
    session = _get_session(streams, stream_id)
    try:
        session.acquire_viewer()
    except StreamBusyError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    async def frames():
        loop = asyncio.get_running_loop()
        try:
            while not session.closed and not await request.is_disconnected():
                started = loop.time()
                try:
                    jpeg = await run_in_threadpool(session.next_frame)
                except StreamError as exc:  # e.g. camera taken by another app on reopen
                    session.error = str(exc)
                    break
                if jpeg is None:
                    break
                yield (b"--" + BOUNDARY.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n" + jpeg + b"\r\n")
                # Play files at their native rate; live sources run as fast as they arrive.
                await asyncio.sleep(max(0.0, session.frame_interval - (loop.time() - started)))
        finally:
            session.release_viewer()
            await run_in_threadpool(session.release_capture)  # free the webcam for other apps

    return StreamingResponse(
        frames(),
        media_type=f"multipart/x-mixed-replace; boundary={BOUNDARY}",
        headers={"Cache-Control": "no-store"},
    )


@router.get("/{stream_id}/stats", name="stream_stats", response_model=StreamStatsOut)
async def stream_stats(stream_id: str, streams: StreamsDep) -> StreamStatsOut:
    session = _get_session(streams, stream_id)
    last = session.stats.last
    return StreamStatsOut(
        stream_id=session.id,
        running=not session.closed,
        frames=session.stats.frames,
        fps=round(session.stats.fps, 1),
        summary=summary_from_result(last) if last else None,
        error=session.error,
    )


@router.delete("/{stream_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_stream(stream_id: str, streams: StreamsDep) -> None:
    if not await run_in_threadpool(streams.remove, stream_id):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Stream not found")
