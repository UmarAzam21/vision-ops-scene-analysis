"""Measure detector accuracy on labelled images (YOLO-format labels, e.g. COCO val2017).

Reports what matters for "find every person and object":
  * mAP50 and mAP50-95 (COCO-style, 101-point interpolation) over all classes
  * person precision / recall at the app's real confidence threshold, split by size
    (COCO sizes: small < 32x32 px, medium < 96x96 px, large >= 96x96 px)

Usage:
    python scripts/evaluate.py --images path/to/images --labels path/to/labels \
        --model models/yolo11x.onnx --imgsz 640 --limit 500
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.ml.detector import YoloDetector  # noqa: E402
from app.ml.labels import COCO_LABELS  # noqa: E402

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)


def load_labels(path: Path, w: int, h: int) -> list[tuple[int, np.ndarray]]:
    out = []
    if path.exists():
        for line in path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            c, cx, cy, bw, bh = int(parts[0]), *map(float, parts[1:5])
            out.append((c, np.array([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])))
    return out


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def average_precision(tp: np.ndarray, scores: np.ndarray, n_gt: int) -> float:
    if n_gt == 0:
        return float("nan")
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores)
    tp = tp[order]
    ctp = np.cumsum(tp)
    recall = ctp / n_gt
    precision = ctp / np.arange(1, len(tp) + 1)
    precision = np.maximum.accumulate(precision[::-1])[::-1]  # monotone envelope
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, grid, side="left")
    return float(np.mean([precision[i] if i < len(precision) else 0.0 for i in idx]))


def size_bucket(box: np.ndarray) -> str:
    area = (box[2] - box[0]) * (box[3] - box[1])
    return "small" if area < 32 ** 2 else "medium" if area < 96 ** 2 else "large"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--model", type=Path, default=Path("models/yolo11n.onnx"))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--conf", type=float, nargs="+", default=[0.25, 0.35, 0.5],
                    help="confidence thresholds for person precision/recall (app default 0.35)")
    ap.add_argument("--tiles", type=int, default=1, help="tiled inference grid (1 = off)")
    args = ap.parse_args()

    # Very low threshold for AP (standard practice); P/R are computed at --conf afterwards.
    det = YoloDetector(args.model, input_size=args.imgsz, conf_threshold=0.001, nms_threshold=0.6)
    images = sorted(args.images.glob("*.jpg"))[: args.limit]

    # per class, per IoU threshold: list of (score, is_tp)
    records: dict[int, list[tuple[float, np.ndarray]]] = defaultdict(list)
    n_gt: dict[int, int] = defaultdict(int)
    person = {t: {"tp": 0, "fp": 0, "gt": defaultdict(int), "found": defaultdict(int)} for t in args.conf}
    t_total = 0.0

    for k, path in enumerate(images, 1):
        img = cv2.imread(str(path))
        h, w = img.shape[:2]
        gts = load_labels(args.labels / f"{path.stem}.txt", w, h)
        t0 = time.perf_counter()
        dets = det.detect(img, tiles=args.tiles)
        t_total += time.perf_counter() - t0

        for c in {c for c, _ in gts} | {d.class_id for d in dets}:
            g = np.array([b for cc, b in gts if cc == c]).reshape(-1, 4)
            ds = sorted((d for d in dets if d.class_id == c), key=lambda d: -d.confidence)
            dboxes = np.array([d.box for d in ds], dtype=float).reshape(-1, 4)
            n_gt[c] += len(g)
            ious = iou_matrix(dboxes, g)
            hits = np.zeros((len(ds), len(IOU_THRESHOLDS)), bool)
            for ti, thr in enumerate(IOU_THRESHOLDS):
                used: set[int] = set()
                for di in range(len(ds)):  # greedy, highest score first
                    if not len(g):
                        break
                    for j in np.argsort(-ious[di]):
                        if ious[di, j] < thr:
                            break
                        if j not in used:
                            used.add(j)
                            hits[di, ti] = True
                            break
            records[c].extend((d.confidence, hits[di]) for di, d in enumerate(ds))

            if c == 0:  # person P/R at each operating threshold, IoU 0.5
                for t, st in person.items():
                    live = [d for d in ds if d.confidence >= t]
                    lb = np.array([d.box for d in live], dtype=float).reshape(-1, 4)
                    m = iou_matrix(lb, g)
                    used = set()
                    for di in range(len(live)):
                        hit = next((j for j in np.argsort(-m[di]) if m[di, j] >= 0.5 and j not in used),
                                   None) if len(g) else None
                        if hit is not None:
                            used.add(hit)
                            st["tp"] += 1
                        else:
                            st["fp"] += 1
                    for j, box in enumerate(g):
                        b = size_bucket(box)
                        st["gt"][b] += 1
                        st["found"][b] += j in used
        if k % 50 == 0:
            print(f"  {k}/{len(images)} images", file=sys.stderr)

    aps = np.zeros((len(COCO_LABELS), len(IOU_THRESHOLDS))) * np.nan
    for c in range(len(COCO_LABELS)):
        if n_gt[c] == 0:
            continue
        rec = records.get(c, [])
        scores = np.array([r[0] for r in rec])
        tps = np.array([r[1] for r in rec]).reshape(-1, len(IOU_THRESHOLDS))
        for ti in range(len(IOU_THRESHOLDS)):
            aps[c, ti] = average_precision(tps[:, ti] if len(tps) else np.array([]), scores, n_gt[c])

    print(f"\nModel {args.model.name} @ {args.imgsz}px tiles={args.tiles} on {len(images)} images "
          f"({t_total / len(images) * 1000:.0f} ms/image, CPU)")
    print(f"  mAP50-95 (all classes): {np.nanmean(aps) * 100:.1f}")
    print(f"  mAP50    (all classes): {np.nanmean(aps[:, 0]) * 100:.1f}")
    print(f"  person AP50           : {aps[0, 0] * 100:.1f}")
    for t, st in person.items():
        p_gt = sum(st["gt"].values())
        sizes = " | ".join(f"{b} {st['found'][b] / st['gt'][b] * 100:.1f}%"
                           for b in ("large", "medium", "small") if st["gt"][b])
        print(f"  person @ conf {t:.2f}: precision {st['tp'] / max(1, st['tp'] + st['fp']) * 100:.1f}% | "
              f"recall {sum(st['found'].values()) / max(1, p_gt) * 100:.1f}% of {p_gt} | {sizes}")


if __name__ == "__main__":
    main()
