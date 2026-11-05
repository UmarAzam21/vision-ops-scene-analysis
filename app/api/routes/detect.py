"""Single-image detection endpoints."""

from typing import Literal

from fastapi import APIRouter, File, HTTPException, Query, UploadFile, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import Response

from app.api.deps import ModelsDep, SettingsDep
from app.schemas.detection import DetectionResponse
from app.services.analyzer import FrameAnalyzer, FrameResult
from app.utils.image import decode_image, encode_jpeg, to_data_url

router = APIRouter(prefix="/detect", tags=["detection"])

IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/bmp"}


async def _read_image(file: UploadFile, max_mb: int):
    if file.content_type not in IMAGE_TYPES:
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                            f"Unsupported type {file.content_type!r}; use JPEG, PNG, WEBP or BMP")
    data = await file.read(max_mb * 1024 * 1024 + 1)
    if len(data) > max_mb * 1024 * 1024:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"Image exceeds {max_mb} MB")
    try:
        return decode_image(data)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc


def _run(analyzer: FrameAnalyzer, image, annotate: bool) -> tuple[FrameResult, bytes | None]:
    result = analyzer.analyze(image)
    jpeg = encode_jpeg(analyzer.render(image, result), 88) if annotate else None
    return result, jpeg


@router.post("", response_model=DetectionResponse, summary="Detect objects in an image")
async def detect(
    models: ModelsDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    include_image: bool = Query(False, description="Return the annotated image as a data URL"),
    mode: Literal["balanced", "max_recall"] = Query(
        "balanced",
        description="balanced: best precision/recall trade-off. max_recall: zoomed tiles + lower "
                    "threshold - finds more small/far people and objects, more false alarms, ~5x slower.",
    ),
) -> DetectionResponse:
    image = await _read_image(file, settings.max_image_mb)
    analyzer = FrameAnalyzer(models, settings, temporal=False, max_recall=mode == "max_recall")
    result, jpeg = await run_in_threadpool(_run, analyzer, image, include_image)
    return DetectionResponse.from_result(result, to_data_url(jpeg) if jpeg else None)


@router.post(
    "/annotated",
    summary="Detect objects and return the annotated JPEG",
    responses={200: {"content": {"image/jpeg": {}}}},
    response_class=Response,
)
async def detect_annotated(
    models: ModelsDep,
    settings: SettingsDep,
    file: UploadFile = File(...),
    mode: Literal["balanced", "max_recall"] = Query("balanced"),
) -> Response:
    image = await _read_image(file, settings.max_image_mb)
    analyzer = FrameAnalyzer(models, settings, temporal=False, max_recall=mode == "max_recall")
    result, jpeg = await run_in_threadpool(_run, analyzer, image, True)
    return Response(
        content=jpeg,
        media_type="image/jpeg",
        headers={"X-Persons": str(result.persons), "X-Objects": str(len(result.detections))},
    )
