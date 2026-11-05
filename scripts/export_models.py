"""One-time export of the high-accuracy models to ONNX (they are not published as ONNX).

Needs Ultralytics + PyTorch *only for this step* - the app itself stays PyTorch-free.
Use a separate virtual environment so the app's dependencies stay small:

    py -3.12 -m venv .export-venv
    .export-venv\\Scripts\\pip install ultralytics onnx onnxslim
    .export-venv\\Scripts\\python scripts\\export_models.py     # yolo11s + yolo11x + YOLOE

Outputs (into models/):
    yolo11s.onnx                     live-video detector (better than the downloadable yolo11n)
    yolo11x.onnx                     accurate 80-class detector for photos
    yoloe-11l-seg-pf.onnx            open-vocabulary detector (4,585 object types)
    yoloe-11l-seg-pf_labels.json     its class names
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

MODELS = Path(__file__).resolve().parents[1] / "models"


def export(weights: str, out_name: str, imgsz: int) -> Path:
    from ultralytics import YOLO, YOLOE  # imported lazily: optional dependency

    cls = YOLOE if weights.startswith("yoloe") else YOLO
    model = cls(weights)
    produced = Path(model.export(format="onnx", imgsz=imgsz, opset=17, simplify=True))
    dest = MODELS / out_name
    shutil.move(str(produced), dest)
    if cls is YOLOE:
        names = model.names
        (MODELS / out_name.replace(".onnx", "_labels.json")).write_text(
            json.dumps([names[i] for i in sorted(names)]), encoding="utf-8")
    print(f"[ok] {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--detector", nargs="+", default=["yolo11s", "yolo11x"],
                    choices=["yolo11s", "yolo11m", "yolo11l", "yolo11x"],
                    help="yolo11s = live video default, yolo11x = photo default")
    ap.add_argument("--open-vocab", default="yoloe-11l-seg-pf", choices=["yoloe-11s-seg-pf", "yoloe-11m-seg-pf",
                                                                         "yoloe-11l-seg-pf", "none"])
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    MODELS.mkdir(exist_ok=True)
    for name in args.detector:
        export(f"{name}.pt", f"{name}.onnx", args.imgsz)
    if args.open_vocab != "none":
        export(f"{args.open_vocab}.pt", f"{args.open_vocab}.onnx", args.imgsz)
