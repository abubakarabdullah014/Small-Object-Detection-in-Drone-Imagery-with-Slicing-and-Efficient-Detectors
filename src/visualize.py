"""
Draw qualitative comparisons: full / SAHI / DAS with slice overlays.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import RESULTS_DIR
from src.das import DASConfig, run_das
from src.zoo import load_sahi_model, predict_full_image, predict_sahi_sliced, sahi_prediction_to_dicts


def draw_dets(img, dets, color=(0, 255, 0), max_n=80):
    out = img.copy()
    for d in sorted(dets, key=lambda x: -x["score"])[:max_n]:
        x, y, w, h = map(int, d["bbox"])
        cv2.rectangle(out, (x, y), (x + w, y + h), color, 1)
    return out


def draw_slices(img, slices, color=(0, 0, 255)):
    out = img.copy()
    for sl in slices:
        cv2.rectangle(out, (sl.x1, sl.y1), (sl.x2, sl.y2), color, 2)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov8n")
    parser.add_argument("--image-idx", type=int, default=0)
    parser.add_argument("--gt", default="data/coco/val.json")
    parser.add_argument("--n-max", type=int, default=6)
    args = parser.parse_args()

    with open(args.gt, encoding="utf-8") as f:
        coco = json.load(f)
    im = coco["images"][args.image_idx]
    bgr = cv2.imread(im["path"])
    model = load_sahi_model(args.model, conf=0.25)

    # Full
    full = predict_full_image(model, im["path"])
    full_d = sahi_prediction_to_dicts(full, im["id"])
    img_full = draw_dets(bgr, full_d, (0, 200, 0))

    # SAHI
    sahi_res = predict_sahi_sliced(
        model, im["path"], 512, 512, 0.2, 0.2, True, 0.25
    )
    sahi_d = sahi_prediction_to_dicts(sahi_res, im["id"])
    img_sahi = draw_dets(bgr, sahi_d, (255, 128, 0))

    # DAS
    with open("configs/default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    dcfg = DASConfig.from_dict(cfg["das"])
    dcfg.n_max = args.n_max
    das = run_das(model, im["path"], im["id"], dcfg)
    img_das = draw_dets(bgr, das.detections, (0, 0, 255))
    img_das = draw_slices(img_das, das.slices, (0, 255, 255))

    # Panel
    h = 360
    def resize(im0):
        scale = h / im0.shape[0]
        return cv2.resize(im0, (int(im0.shape[1] * scale), h))

    panel = np.hstack([resize(img_full), resize(img_sahi), resize(img_das)])
    cv2.putText(panel, "Full", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 0), 2)
    cv2.putText(panel, "SAHI", (resize(img_full).shape[1] + 10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 128, 0), 2)
    cv2.putText(
        panel,
        f"DAS n={args.n_max}",
        (resize(img_full).shape[1] + resize(img_sahi).shape[1] + 10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),
        2,
    )

    out_dir = RESULTS_DIR / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"qual_{args.model}_{im['file_name']}"
    cv2.imwrite(str(out), panel)
    paper = Path("paper/figures")
    paper.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(paper / out.name), panel)
    print(f"wrote {out}")
    print(f"full={len(full_d)} sahi={len(sahi_d)} das={len(das.detections)} slices={len(das.slices)}")


if __name__ == "__main__":
    main()
