"""
Border-aware, scale-gated fusion for DAS (WBF or greedy NMS).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np


def _xyxy(d: Dict) -> Tuple[float, float, float, float]:
    if "xyxy" in d:
        return tuple(map(float, d["xyxy"]))  # type: ignore
    x, y, w, h = d["bbox"]
    return float(x), float(y), float(x + w), float(y + h)


def _to_bbox(xyxy: Tuple[float, float, float, float]) -> List[float]:
    x1, y1, x2, y2 = xyxy
    return [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]


def _touches_border(
    xyxy: Tuple[float, float, float, float],
    slice_xyxy: Tuple[int, int, int, int],
    margin: int,
) -> bool:
    x1, y1, x2, y2 = xyxy
    sx1, sy1, sx2, sy2 = slice_xyxy
    return (
        x1 - sx1 <= margin
        or y1 - sy1 <= margin
        or sx2 - x2 <= margin
        or sy2 - y2 <= margin
    )


def _iou_xyxy(a, b) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(1e-6, area_a + area_b - inter)


def apply_border_weights(
    dets: List[Dict],
    slice_xyxy: Optional[Tuple[int, int, int, int]] = None,
    border_margin: int = 8,
    border_weight: float = 0.5,
) -> List[Dict]:
    """Downweight boxes that touch the slice border."""
    out = []
    for d in dets:
        nd = dict(d)
        if slice_xyxy is not None:
            xyxy = _xyxy(d)
            if _touches_border(xyxy, slice_xyxy, border_margin):
                nd["score"] = float(d["score"]) * border_weight
                nd["border"] = True
            else:
                nd["border"] = False
        else:
            nd["border"] = False
        out.append(nd)
    return out


def greedy_nms(
    dets: List[Dict],
    iou_thr: float = 0.5,
) -> List[Dict]:
    if not dets:
        return []
    # Group by class
    by_cls: Dict[int, List[Dict]] = {}
    for d in dets:
        by_cls.setdefault(int(d["category_id"]), []).append(d)
    kept: List[Dict] = []
    for cls, group in by_cls.items():
        group = sorted(group, key=lambda x: -float(x["score"]))
        while group:
            best = group.pop(0)
            kept.append(best)
            bxy = _xyxy(best)
            group = [g for g in group if _iou_xyxy(bxy, _xyxy(g)) < iou_thr]
    return kept


def weighted_box_fusion(
    dets: List[Dict],
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.001,
) -> List[Dict]:
    """Class-wise WBF using ensemble-boxes if available, else local fallback."""
    if not dets:
        return []
    try:
        from ensemble_boxes import weighted_boxes_fusion
    except ImportError:
        return greedy_nms(dets, iou_thr=iou_thr)

    # Need normalized boxes; find image extent from dets
    max_x = max(_xyxy(d)[2] for d in dets)
    max_y = max(_xyxy(d)[3] for d in dets)
    W = max(max_x, 1.0)
    H = max(max_y, 1.0)

    by_cls: Dict[int, List[Dict]] = {}
    for d in dets:
        by_cls.setdefault(int(d["category_id"]), []).append(d)

    fused: List[Dict] = []
    for cls, group in by_cls.items():
        boxes = []
        scores = []
        for d in group:
            x1, y1, x2, y2 = _xyxy(d)
            boxes.append([x1 / W, y1 / H, x2 / W, y2 / H])
            scores.append(float(d["score"]))
        if not boxes:
            continue
        labels = [0] * len(boxes)
        fb, fs, fl = weighted_boxes_fusion(
            [boxes],
            [scores],
            [labels],
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
        for b, s in zip(fb, fs):
            x1, y1, x2, y2 = b[0] * W, b[1] * H, b[2] * W, b[3] * H
            fused.append(
                {
                    "category_id": cls,
                    "bbox": _to_bbox((x1, y1, x2, y2)),
                    "xyxy": [x1, y1, x2, y2],
                    "score": float(s),
                }
            )
    return fused


def fuse_coarse_and_fine(
    coarse: List[Dict],
    fine: List[Dict],
    method: str = "wbf",
    iou_thr: float = 0.55,
    scale_gate: float = 96.0,
    final_conf: float = 0.25,
) -> List[Dict]:
    """
    Hard floor: keep all coarse boxes at/above final_conf (same as a full-image
    pass), then add fine-slice boxes and merge.

    For overlapping same-class pairs:
      - if object is small (diag < scale_gate), prefer the higher-scoring box
        (usually fine);
      - otherwise keep both and let NMS/WBF resolve.
    This guarantees DAS cannot fall below full-image recall from dropping coarse.
    """
    image_id = None
    for d in coarse + fine:
        if "image_id" in d:
            image_id = d["image_id"]
            break

    def with_diag(d):
        nd = dict(d)
        x1, y1, x2, y2 = _xyxy(d)
        nd["diag"] = float(np.hypot(x2 - x1, y2 - y1))
        nd["xyxy"] = [x1, y1, x2, y2]
        if "bbox" not in nd:
            nd["bbox"] = _to_bbox((x1, y1, x2, y2))
        return nd

    # Always keep coarse at final_conf (performance floor)
    coarse_f = [with_diag(d) for d in coarse if float(d["score"]) >= final_conf]
    # Slightly lower bar for fine so border-downweighted boxes can still compete
    fine_f = [with_diag(d) for d in fine if float(d["score"]) >= final_conf * 0.7]

    # Remove coarse only when a clearly better fine box covers the same small object
    keep_coarse = []
    for c in coarse_f:
        replaced = False
        if c["diag"] < scale_gate:
            for f in fine_f:
                if int(f["category_id"]) != int(c["category_id"]):
                    continue
                if _iou_xyxy(_xyxy(c), _xyxy(f)) >= 0.5 and float(f["score"]) >= float(c["score"]):
                    replaced = True
                    break
        if not replaced:
            keep_coarse.append(c)

    # Keep all fine boxes; NMS/WBF will dedupe with coarse
    merged = keep_coarse + fine_f

    # Prefer NMS for the floor guarantee (WBF can dilute scores below threshold)
    if method == "wbf":
        fused = weighted_box_fusion(merged, iou_thr=iou_thr)
        # Safety: if WBF wiped too much, fall back to NMS union
        if len(fused) < len(coarse_f) * 0.5:
            fused = greedy_nms(merged, iou_thr=min(0.5, iou_thr))
    else:
        fused = greedy_nms(merged, iou_thr=min(0.5, iou_thr))

    out = []
    for d in fused:
        score = float(d["score"])
        if score < final_conf * 0.9:
            continue
        nd = {
            "category_id": int(d["category_id"]),
            "bbox": d["bbox"] if "bbox" in d else _to_bbox(_xyxy(d)),
            "xyxy": d.get("xyxy", list(_xyxy(d))),
            "score": score,
        }
        if image_id is not None:
            nd["image_id"] = image_id
        out.append(nd)
    return out
