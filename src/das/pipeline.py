"""
DAS end-to-end pipeline: coarse pass -> density map -> schedule -> slice infer -> fuse.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .density import build_density_map
from .fusion import apply_border_weights, fuse_coarse_and_fine
from .scheduler import SliceWindow, schedule_slices


@dataclass
class DASResult:
    detections: List[Dict[str, Any]]
    slices: List[SliceWindow]
    coarse_dets: List[Dict[str, Any]]
    latency_ms: float
    forward_passes: int
    density_summary: Dict[str, float] = field(default_factory=dict)


@dataclass
class DASConfig:
    grid_rows: int = 32
    grid_cols: int = 32
    coarse_conf: float = 0.05
    final_conf: float = 0.25
    tau: float = 0.15
    k: float = 8.0
    slice_min: int = 256
    slice_max: int = 1024
    default_slice: int = 512
    overlap_ratio: float = 0.2
    n_max: int = 6
    use_coarse_prior: bool = True
    use_image_prior: bool = True
    scale_adaptive: bool = True
    fusion: str = "wbf"
    border_margin: int = 8
    border_weight: float = 0.5
    scale_gate: float = 96.0
    wbf_iou_thr: float = 0.55
    image_prior_weight: float = 0.35
    coarse_prior_weight: float = 0.65

    @classmethod
    def from_dict(cls, d: Dict) -> "DASConfig":
        valid = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore
        return cls(**{k: v for k, v in d.items() if k in valid})


def _run_detector_on_image(detection_model, image_bgr: np.ndarray, conf: float):
    """Run SAHI model on a numpy BGR image (full or crop)."""
    from sahi.predict import get_prediction

    old = detection_model.confidence_threshold
    detection_model.confidence_threshold = conf
    # SAHI accepts numpy RGB or path; convert BGR->RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    result = get_prediction(image_rgb, detection_model)
    detection_model.confidence_threshold = old
    return result


def _preds_to_dicts(result, image_id: int, offset_xy=(0, 0), filter_coco=False) -> List[Dict]:
    from src.zoo import sahi_prediction_to_dicts

    dets = sahi_prediction_to_dicts(
        result, image_id=image_id, filter_coco_for_visdrone=filter_coco
    )
    ox, oy = offset_xy
    if ox == 0 and oy == 0:
        return dets
    out = []
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        x1, y1, x2, y2 = x1 + ox, y1 + oy, x2 + ox, y2 + oy
        nd = dict(d)
        nd["xyxy"] = [x1, y1, x2, y2]
        nd["bbox"] = [x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)]
        out.append(nd)
    return out


def run_das(
    detection_model,
    image_path: str,
    image_id: int,
    cfg: DASConfig,
    filter_coco: bool = False,
) -> DASResult:
    """Run Density-Adaptive Slicing on one image."""
    t0 = time.perf_counter()
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(image_path)

    forwards = 0

    # 1) Coarse full-image pass
    coarse_result = _run_detector_on_image(
        detection_model, image_bgr, conf=cfg.coarse_conf
    )
    forwards += 1
    coarse_dets = _preds_to_dicts(
        coarse_result, image_id=image_id, filter_coco=filter_coco
    )

    # 2) Density + scale map
    dmap = build_density_map(
        image_bgr,
        coarse_dets if cfg.use_coarse_prior else [],
        rows=cfg.grid_rows,
        cols=cfg.grid_cols,
        use_coarse_prior=cfg.use_coarse_prior,
        use_image_prior=cfg.use_image_prior,
        coarse_weight=cfg.coarse_prior_weight,
        image_weight=cfg.image_prior_weight,
    )

    # 3) Schedule slices under budget
    slices = schedule_slices(
        dmap,
        n_max=cfg.n_max,
        tau=cfg.tau,
        k=cfg.k,
        slice_min=cfg.slice_min,
        slice_max=cfg.slice_max,
        default_slice=cfg.default_slice,
        overlap_ratio=cfg.overlap_ratio,
        scale_adaptive=cfg.scale_adaptive,
    )

    # 4) Run detector on selected slices
    fine_dets: List[Dict] = []
    for sl in slices:
        crop = image_bgr[sl.y1 : sl.y2, sl.x1 : sl.x2]
        if crop.size == 0:
            continue
        res = _run_detector_on_image(detection_model, crop, conf=cfg.final_conf)
        forwards += 1
        dets = _preds_to_dicts(
            res,
            image_id=image_id,
            offset_xy=(sl.x1, sl.y1),
            filter_coco=filter_coco,
        )
        dets = apply_border_weights(
            dets,
            slice_xyxy=(sl.x1, sl.y1, sl.x2, sl.y2),
            border_margin=cfg.border_margin,
            border_weight=cfg.border_weight,
        )
        fine_dets.extend(dets)

    # Coarse dets for fusion use final_conf filter for large objects
    coarse_for_fuse = [d for d in coarse_dets if float(d["score"]) >= cfg.final_conf]

    # 5) Fuse
    fused = fuse_coarse_and_fine(
        coarse_for_fuse,
        fine_dets,
        method=cfg.fusion,
        iou_thr=cfg.wbf_iou_thr,
        scale_gate=cfg.scale_gate,
        final_conf=cfg.final_conf,
    )
    for d in fused:
        d["image_id"] = image_id

    latency = (time.perf_counter() - t0) * 1000.0
    return DASResult(
        detections=fused,
        slices=slices,
        coarse_dets=coarse_dets,
        latency_ms=latency,
        forward_passes=forwards,
        density_summary={
            "mean_density": float(dmap.density.mean()),
            "max_density": float(dmap.density.max()),
            "n_slices": len(slices),
        },
    )
