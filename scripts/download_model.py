"""Download every model the service uses (~55 MB total). Already-present files are skipped.

Usage:
    python scripts/download_model.py              # everything
    python scripts/download_model.py --only core  # just the object detector
    python scripts/download_model.py --force      # re-download
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"
ULTRA = "https://github.com/ultralytics/assets/releases/download/v8.3.0"
ZOO = "https://github.com/opencv/opencv_zoo/raw/main/models"
INSIGHT = "https://huggingface.co/public-data/insightface/resolve/main/models/buffalo_l"
HF = "https://huggingface.co/Xenova/segformer-b0-finetuned-ade-512-512/resolve/main"

# group, filename, url, minimum plausible size (bytes) - catches HTML error pages / LFS pointers
MODELS: list[tuple[str, str, str, int]] = [
    ("core", "yolo11n.onnx", f"{ULTRA}/yolo11n.onnx", 5_000_000),
    ("pose", "yolo11n-pose.onnx", f"{ULTRA}/yolo11n-pose.onnx", 5_000_000),
    ("face", "face_yunet.onnx", f"{ZOO}/face_detection_yunet/face_detection_yunet_2023mar.onnx", 100_000),
    ("face", "face_expression.onnx",
     f"{ZOO}/facial_expression_recognition/facial_expression_recognition_mobilefacenet_2022july.onnx", 1_000_000),
    ("face", "face_genderage.onnx", f"{INSIGHT}/genderage.onnx", 1_000_000),
    ("scene", "segformer_ade.onnx", f"{HF}/onnx/model.onnx", 10_000_000),
    ("scene", "segformer_ade_labels.json", f"{HF}/config.json", 1_000),
]


def download(name: str, url: str, min_size: int, force: bool = False) -> None:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size >= min_size and not force:
        print(f"[ok] {name:28s} present ({dest.stat().st_size / 1e6:.1f} MB)")
        return
    tmp = dest.with_suffix(dest.suffix + ".part")

    def progress(blocks: int, block_size: int, total: int) -> None:
        if total > 0:
            sys.stdout.write(f"\r[..] {name:28s} {min(100, blocks * block_size * 100 // total):3d}%")
            sys.stdout.flush()

    urllib.request.urlretrieve(url, tmp, reporthook=progress)
    print()
    if tmp.stat().st_size < min_size:
        tmp.unlink()
        raise SystemExit(f"[x] {name}: download looks truncated - please retry.")
    tmp.replace(dest)
    print(f"[ok] {name:28s} saved ({dest.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    groups = sorted({g for g, *_ in MODELS})
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", choices=groups, action="append", help="download only these groups")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    for group, name, url, min_size in MODELS:
        if args.only and group not in args.only:
            continue
        download(name, url, min_size, args.force)
