import base64

import cv2
import numpy as np


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes (jpg/png/webp/bmp) into a BGR array."""
    if not data:
        raise ValueError("Empty image payload")
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Unsupported or corrupted image")
    return image


def encode_jpeg(image: np.ndarray, quality: int = 85) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return buf.tobytes()


def to_data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def resize_max_width(image: np.ndarray, max_width: int) -> np.ndarray:
    h, w = image.shape[:2]
    if w <= max_width:
        return image
    scale = max_width / w
    return cv2.resize(image, (max_width, round(h * scale)), interpolation=cv2.INTER_AREA)


def is_blank(frame: np.ndarray) -> bool:
    """True when the camera isn't delivering a real picture:
    * near-black and flat  - closed shutter, covered lens, broken driver path
    * one flat colour with at most a tiny icon - the "camera off" placeholder Windows /
      Intel drivers send when the camera privacy key (e.g. F9) is engaged.
    A real scene, even a dim one, has far more variation than that."""
    small = cv2.cvtColor(frame[::8, ::8], cv2.COLOR_BGR2GRAY).astype(np.int16)
    if float(small.mean()) < 8 and float(small.std()) < 4:
        return True
    flat_share = float(np.mean(np.abs(small - np.median(small)) < 10))
    return flat_share > 0.97
