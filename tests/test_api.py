import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.conftest import requires_model

pytestmark = requires_model


@pytest.fixture(scope="module")
def client():
    from app.main import create_app

    with TestClient(create_app()) as c:
        yield c


def _jpeg(img: np.ndarray) -> bytes:
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_health(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model"] in ("yolo11s", "yolo11n")


def test_index_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "VISION" in r.text


def test_detect_json_and_image(client):
    img = np.full((360, 640, 3), 127, dtype=np.uint8)
    r = client.post(
        "/api/v1/detect?include_image=true",
        files={"file": ("x.jpg", _jpeg(img), "image/jpeg")},
    )
    assert r.status_code == 200
    body = r.json()
    assert (body["width"], body["height"]) == (640, 360)
    assert body["summary"]["persons"] == 0
    assert body["annotated_image"].startswith("data:image/jpeg;base64,")


def test_detect_annotated_returns_jpeg(client):
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    r = client.post("/api/v1/detect/annotated", files={"file": ("x.png", cv2.imencode(".png", img)[1].tobytes(), "image/png")})
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/jpeg"
    assert r.content[:2] == b"\xff\xd8"


def test_rejects_non_image(client):
    r = client.post("/api/v1/detect", files={"file": ("a.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_rejects_corrupt_image(client):
    r = client.post("/api/v1/detect", files={"file": ("a.jpg", b"notajpeg", "image/jpeg")})
    assert r.status_code == 400


def test_url_sources_disabled_by_default(client):
    r = client.post("/api/v1/streams/source", json={"source": "rtsp://10.0.0.1/live"})
    assert r.status_code == 403


def test_video_upload_stream_and_stats(client, tmp_path):
    path = tmp_path / "clip.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240))
    for i in range(10):
        frame = np.full((240, 320, 3), 40 + i * 5, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    with path.open("rb") as fh:
        r = client.post("/api/v1/streams/upload", files={"file": ("clip.avi", fh, "video/x-msvideo")})
    assert r.status_code == 201
    sid = r.json()["stream_id"]

    with client.stream("GET", f"/api/v1/streams/{sid}/video") as resp:
        assert resp.headers["content-type"].startswith("multipart/x-mixed-replace")
        chunk = next(resp.iter_bytes())
        assert b"--frame" in chunk

    stats = client.get(f"/api/v1/streams/{sid}/stats").json()
    assert stats["frames"] >= 1

    assert client.delete(f"/api/v1/streams/{sid}").status_code == 204
    assert client.get(f"/api/v1/streams/{sid}/stats").status_code == 404


def test_live_websocket_round_trip(client):
    frame = np.full((360, 640, 3), 90, dtype=np.uint8)
    frame[100:260, 200:440] = (40, 160, 220)
    with client.websocket_connect("/api/v1/live") as ws:
        for _ in range(2):
            ws.send_bytes(_jpeg(frame))
            annotated = ws.receive_bytes()
            stats = ws.receive_json()
            assert annotated[:2] == b"\xff\xd8"
            assert "summary" in stats and stats["summary"]["persons"] == 0
        ws.send_bytes(b"not a jpeg")             # bad frame -> error, connection stays open
        assert "error" in ws.receive_json()
        ws.send_bytes(_jpeg(frame))
        assert ws.receive_bytes()[:2] == b"\xff\xd8"
        ws.receive_json()
