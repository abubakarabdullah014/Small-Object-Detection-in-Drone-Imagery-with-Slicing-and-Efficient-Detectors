"""
Create a synthetic TinyPerson-like domain-shift subset from VisDrone val
by cropping high-density pedestrian regions (for generalization smoke tests
when full UAVDT/TinyPerson downloads are unavailable).

Also writes a script hook to evaluate DAS on test-dev.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import DATA_COCO, DATA_RAW, RESULTS_DIR, VISDRONE_CATEGORIES


def build_person_crop_subset(
    gt_json: Path,
    out_dir: Path,
    n_crops: int = 100,
    crop_size: int = 640,
    seed: int = 0,
) -> Path:
    """
    Sample crops centered on pedestrian/people annotations to simulate
    a denser small-person domain (TinyPerson-like proxy).
    """
    rng = np.random.default_rng(seed)
    with open(gt_json, encoding="utf-8") as f:
        coco = json.load(f)
    images = {im["id"]: im for im in coco["images"]}
    person_anns = [a for a in coco["annotations"] if a["category_id"] in (1, 2)]
    rng.shuffle(person_anns)

    out_img = out_dir / "images"
    out_img.mkdir(parents=True, exist_ok=True)
    new_images = []
    new_anns = []
    ann_id = 1
    img_id = 1
    used = 0

    for a in tqdm(person_anns, desc="person-crops"):
        if used >= n_crops:
            break
        im = images[a["image_id"]]
        bgr = cv2.imread(im["path"])
        if bgr is None:
            continue
        H, W = bgr.shape[:2]
        x, y, w, h = a["bbox"]
        cx, cy = x + w / 2, y + h / 2
        x1 = int(np.clip(cx - crop_size / 2, 0, max(0, W - crop_size)))
        y1 = int(np.clip(cy - crop_size / 2, 0, max(0, H - crop_size)))
        x2 = min(W, x1 + crop_size)
        y2 = min(H, y1 + crop_size)
        crop = bgr[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        fname = f"crop_{img_id:04d}.jpg"
        fpath = out_img / fname
        cv2.imwrite(str(fpath), crop)

        # Collect all anns overlapping this crop
        for b in coco["annotations"]:
            if b["image_id"] != a["image_id"]:
                continue
            bx, by, bw, bh = b["bbox"]
            # intersection with crop
            ix1 = max(bx, x1)
            iy1 = max(by, y1)
            ix2 = min(bx + bw, x2)
            iy2 = min(by + bh, y2)
            iw, ih = ix2 - ix1, iy2 - iy1
            if iw <= 1 or ih <= 1:
                continue
            new_anns.append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": b["category_id"],
                    "bbox": [ix1 - x1, iy1 - y1, iw, ih],
                    "area": float(iw * ih),
                    "iscrowd": 0,
                    "segmentation": [],
                }
            )
            ann_id += 1

        new_images.append(
            {
                "id": img_id,
                "file_name": fname,
                "width": int(x2 - x1),
                "height": int(y2 - y1),
                "path": str(fpath.resolve()),
            }
        )
        img_id += 1
        used += 1

    out = {
        "info": {"description": "TinyPerson-like proxy from VisDrone person crops"},
        "images": new_images,
        "annotations": new_anns,
        "categories": VISDRONE_CATEGORIES,
        "licenses": [],
    }
    out_json = out_dir / "person_crops.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[ok] {out_json}: {len(new_images)} images, {len(new_anns)} anns")
    return out_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-crops", type=int, default=100)
    args = parser.parse_args()
    gt = DATA_COCO / "val.json"
    out = Path(__file__).resolve().parents[1] / "data" / "domain_shift"
    build_person_crop_subset(gt, out, n_crops=args.n_crops)


if __name__ == "__main__":
    main()
