"""
Download VisDrone2019-DET and convert annotations to COCO JSON.

Drops ignored regions (class 0) and others (class 11) per standard protocol.
Also samples a fixed 200-image tuning subset from train.
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple
from urllib.request import urlretrieve

from PIL import Image
from tqdm import tqdm

# Allow running as script
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import (
    DATA_COCO,
    DATA_RAW,
    DATA_TUNING,
    TUNING_N_IMAGES,
    TUNING_SEED,
    VISDRONE_CATEGORIES,
    VISDRONE_URLS,
)


def download_split(split: str, force: bool = False) -> Path:
    """Download and extract one VisDrone split. Returns folder path."""
    DATA_RAW.mkdir(parents=True, exist_ok=True)
    folder_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test-dev": "VisDrone2019-DET-test-dev",
    }
    folder_name = folder_map[split]
    out_dir = DATA_RAW / folder_name
    zip_path = DATA_RAW / f"{folder_name}.zip"

    if out_dir.exists() and not force:
        print(f"[skip] {out_dir} already exists")
        return out_dir

    url = VISDRONE_URLS[split]
    print(f"[download] {split} from {url}")
    if not zip_path.exists() or force:
        urlretrieve(url, zip_path)
    print(f"[extract] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(DATA_RAW)
    # Some zips nest an extra folder; normalize
    if not out_dir.exists():
        # Search for images folder
        candidates = list(DATA_RAW.glob("**/images"))
        for c in candidates:
            parent = c.parent
            if folder_name.lower() in parent.name.lower() or split in parent.name.lower():
                if parent != out_dir:
                    if out_dir.exists():
                        shutil.rmtree(out_dir)
                    parent.rename(out_dir)
                break
    print(f"[ok] {out_dir}")
    return out_dir


def _find_images_ann(root: Path) -> Tuple[Path, Path]:
    """Locate images/ and annotations/ under a VisDrone split folder."""
    img = root / "images"
    ann = root / "annotations"
    if img.exists() and ann.exists():
        return img, ann
    # Fallback: search
    imgs = list(root.rglob("images"))
    anns = list(root.rglob("annotations"))
    if imgs and anns:
        return imgs[0], anns[0]
    raise FileNotFoundError(f"Cannot find images/annotations under {root}")


def convert_split_to_coco(split_dir: Path, split_name: str, out_json: Path) -> Dict:
    """Convert VisDrone txt annotations to COCO JSON. Returns the coco dict."""
    images_dir, ann_dir = _find_images_ann(split_dir)
    images: List[Dict] = []
    annotations: List[Dict] = []
    ann_id = 1

    img_files = sorted(images_dir.glob("*.jpg"))
    print(f"[convert] {split_name}: {len(img_files)} images")

    for img_id, img_path in enumerate(tqdm(img_files, desc=split_name), start=1):
        with Image.open(img_path) as im:
            w, h = im.size
        images.append(
            {
                "id": img_id,
                "file_name": img_path.name,
                "width": w,
                "height": h,
                "path": str(img_path.resolve()),
            }
        )
        txt_path = ann_dir / f"{img_path.stem}.txt"
        if not txt_path.exists():
            continue
        with open(txt_path, encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 6:
                    continue
                x, y, bw, bh = map(int, parts[:4])
                score = int(parts[4])  # 0 = ignored region
                cls = int(parts[5])
                if score == 0:
                    continue  # ignored
                if cls == 0 or cls == 11:
                    continue  # ignored regions / others
                if cls < 1 or cls > 10:
                    continue
                if bw <= 0 or bh <= 0:
                    continue
                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": cls,
                        "bbox": [x, y, bw, bh],
                        "area": float(bw * bh),
                        "iscrowd": 0,
                        "segmentation": [],
                    }
                )
                ann_id += 1

    coco = {
        "info": {
            "description": f"VisDrone2019-DET {split_name} (COCO format)",
            "version": "1.0",
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": VISDRONE_CATEGORIES,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(coco, f)
    print(f"[ok] wrote {out_json} ({len(images)} imgs, {len(annotations)} anns)")
    return coco


def sample_tuning_subset(
    train_coco_json: Path,
    n: int = TUNING_N_IMAGES,
    seed: int = TUNING_SEED,
) -> Path:
    """Sample a fixed n-image subset of train for DAS hyperparameter tuning."""
    with open(train_coco_json, encoding="utf-8") as f:
        coco = json.load(f)

    rng = random.Random(seed)
    images = list(coco["images"])
    if len(images) < n:
        raise ValueError(f"Train has only {len(images)} images, need {n}")
    sampled = rng.sample(images, n)
    sampled_ids = {im["id"] for im in sampled}
    anns = [a for a in coco["annotations"] if a["image_id"] in sampled_ids]

    # Remap image ids to contiguous 1..n for cleaner eval
    id_map = {}
    new_images = []
    for new_id, im in enumerate(sampled, start=1):
        id_map[im["id"]] = new_id
        new_im = dict(im)
        new_im["id"] = new_id
        new_images.append(new_im)
    new_anns = []
    for i, a in enumerate(anns, start=1):
        na = dict(a)
        na["id"] = i
        na["image_id"] = id_map[a["image_id"]]
        new_anns.append(na)

    subset = {
        "info": {"description": f"VisDrone train tuning subset n={n} seed={seed}"},
        "licenses": [],
        "images": new_images,
        "annotations": new_anns,
        "categories": VISDRONE_CATEGORIES,
    }
    DATA_TUNING.mkdir(parents=True, exist_ok=True)
    out = DATA_TUNING / f"tune_{n}_seed{seed}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(subset, f)
    # Also write image list
    list_path = DATA_TUNING / f"tune_{n}_seed{seed}_images.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for im in new_images:
            f.write(im["path"] + "\n")
    print(f"[ok] tuning subset: {out} ({len(new_images)} images)")
    return out


def main():
    parser = argparse.ArgumentParser(description="Download & convert VisDrone")
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["val", "test-dev", "train"],
        choices=["train", "val", "test-dev"],
    )
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    DATA_COCO.mkdir(parents=True, exist_ok=True)
    folder_map = {
        "train": "VisDrone2019-DET-train",
        "val": "VisDrone2019-DET-val",
        "test-dev": "VisDrone2019-DET-test-dev",
    }

    for split in args.splits:
        if not args.skip_download:
            download_split(split, force=args.force)
        split_dir = DATA_RAW / folder_map[split]
        out_json = DATA_COCO / f"{split}.json"
        convert_split_to_coco(split_dir, split, out_json)

    train_json = DATA_COCO / "train.json"
    if train_json.exists():
        sample_tuning_subset(train_json)


if __name__ == "__main__":
    main()
