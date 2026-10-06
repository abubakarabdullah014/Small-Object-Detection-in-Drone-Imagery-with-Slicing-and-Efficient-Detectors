"""
COCO-style evaluation for VisDrone detections.
Reports AP, AP50, AP75, AP_s / AP_m / AP_l, AR_s.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def detections_to_coco_results(
    detections: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert list of dicts with keys:
      image_id, category_id, bbox [x,y,w,h], score
    into COCO results format.
    """
    results = []
    for d in detections:
        results.append(
            {
                "image_id": int(d["image_id"]),
                "category_id": int(d["category_id"]),
                "bbox": [float(x) for x in d["bbox"]],
                "score": float(d["score"]),
            }
        )
    return results


def evaluate_coco(
    gt_json: str | Path,
    detections: List[Dict[str, Any]],
    save_results: Optional[str | Path] = None,
) -> Dict[str, float]:
    """
    Run COCOeval and return a flat metrics dict.
    If detections is empty, returns zeros.
    """
    gt_json = Path(gt_json)
    coco_gt = COCO(str(gt_json))

    if not detections:
        return _empty_metrics()

    results = detections_to_coco_results(detections)
    if save_results is not None:
        save_results = Path(save_results)
        save_results.parent.mkdir(parents=True, exist_ok=True)
        with open(save_results, "w", encoding="utf-8") as f:
            json.dump(results, f)

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    # COCOeval.stats indices:
    # 0 AP, 1 AP50, 2 AP75, 3 AP_s, 4 AP_m, 5 AP_l,
    # 6 AR1, 7 AR10, 8 AR100, 9 AR_s, 10 AR_m, 11 AR_l
    s = coco_eval.stats
    metrics = {
        "AP": float(s[0]),
        "AP50": float(s[1]),
        "AP75": float(s[2]),
        "AP_small": float(s[3]),
        "AP_medium": float(s[4]),
        "AP_large": float(s[5]),
        "AR_small": float(s[9]),
        "AR_medium": float(s[10]),
        "AR_large": float(s[11]),
        "AR100": float(s[8]),
    }
    return metrics


def _empty_metrics() -> Dict[str, float]:
    return {
        "AP": 0.0,
        "AP50": 0.0,
        "AP75": 0.0,
        "AP_small": 0.0,
        "AP_medium": 0.0,
        "AP_large": 0.0,
        "AR_small": 0.0,
        "AR_medium": 0.0,
        "AR_large": 0.0,
        "AR100": 0.0,
    }


def load_coco_images(gt_json: str | Path) -> List[Dict]:
    with open(gt_json, encoding="utf-8") as f:
        coco = json.load(f)
    return coco["images"]


def subset_gt_json(
    gt_json: str | Path,
    max_images: Optional[int],
    out_dir: Optional[str | Path] = None,
) -> Path:
    """
    When evaluating a prefix of images, COCOeval must use a matching GT JSON.
    Otherwise images without predictions dominate and AP collapses.
    """
    gt_json = Path(gt_json)
    if not max_images:
        return gt_json
    with open(gt_json, encoding="utf-8") as f:
        coco = json.load(f)
    images = coco["images"][:max_images]
    keep = {im["id"] for im in images}
    anns = [a for a in coco["annotations"] if a["image_id"] in keep]
    subset = {
        "info": coco.get("info", {}),
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": anns,
        "categories": coco["categories"],
    }
    if out_dir is None:
        out_dir = gt_json.parent / "subsets"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{gt_json.stem}_first{max_images}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(subset, f)
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", required=True)
    parser.add_argument("--dt", required=True, help="COCO results JSON")
    args = parser.parse_args()
    with open(args.dt, encoding="utf-8") as f:
        dets = json.load(f)
    m = evaluate_coco(args.gt, dets)
    print(json.dumps(m, indent=2))
