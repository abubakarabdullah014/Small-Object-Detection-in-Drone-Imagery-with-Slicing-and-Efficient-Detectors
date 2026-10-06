"""
DAS density + scale map from coarse detections and image high-frequency prior.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class DensityMap:
    density: np.ndarray          # [H_grid, W_grid] float
    scale: np.ndarray            # [H_grid, W_grid] median object size (px diagonal)
    rows: int
    cols: int
    image_h: int
    image_w: int

    def cell_bbox(self, r: int, c: int) -> Tuple[int, int, int, int]:
        """Return (x1,y1,x2,y2) pixel bbox of cell (r,c)."""
        cell_h = self.image_h / self.rows
        cell_w = self.image_w / self.cols
        x1 = int(c * cell_w)
        y1 = int(r * cell_h)
        x2 = int(min(self.image_w, (c + 1) * cell_w))
        y2 = int(min(self.image_h, (r + 1) * cell_h))
        return x1, y1, x2, y2


def image_energy_map(
    image_bgr: np.ndarray,
    rows: int = 32,
    cols: int = 32,
) -> np.ndarray:
    """
    Cheap per-cell high-frequency energy via Laplacian variance.
    Returns [rows, cols] normalized to [0, 1].
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=3)
    lap2 = lap * lap
    h, w = gray.shape
    energy = np.zeros((rows, cols), dtype=np.float32)
    cell_h = h / rows
    cell_w = w / cols
    for r in range(rows):
        y1 = int(r * cell_h)
        y2 = int(min(h, (r + 1) * cell_h))
        for c in range(cols):
            x1 = int(c * cell_w)
            x2 = int(min(w, (c + 1) * cell_w))
            patch = lap2[y1:y2, x1:x2]
            energy[r, c] = float(patch.mean()) if patch.size else 0.0
    mx = float(energy.max())
    if mx > 0:
        energy /= mx
    return energy


def coarse_detection_maps(
    detections: List[Dict],
    image_h: int,
    image_w: int,
    rows: int = 32,
    cols: int = 32,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Accumulate coarse detections into density (score-weighted count)
    and scale (median object diagonal per cell).
    detections: list of {xyxy or bbox, score}
    """
    density = np.zeros((rows, cols), dtype=np.float32)
    scale_lists: List[List[float]] = [[[] for _ in range(cols)] for _ in range(rows)]
    cell_h = image_h / rows
    cell_w = image_w / cols

    for d in detections:
        if "xyxy" in d:
            x1, y1, x2, y2 = d["xyxy"]
        else:
            x, y, w, h = d["bbox"]
            x1, y1, x2, y2 = x, y, x + w, y + h
        cx = 0.5 * (x1 + x2)
        cy = 0.5 * (y1 + y2)
        c = int(np.clip(cx / cell_w, 0, cols - 1))
        r = int(np.clip(cy / cell_h, 0, rows - 1))
        score = float(d.get("score", 1.0))
        density[r, c] += score
        diag = float(np.hypot(x2 - x1, y2 - y1))
        scale_lists[r][c].append(diag)

    scale = np.zeros((rows, cols), dtype=np.float32)
    for r in range(rows):
        for c in range(cols):
            if scale_lists[r][c]:
                scale[r, c] = float(np.median(scale_lists[r][c]))

    mx = float(density.max())
    if mx > 0:
        density /= mx
    return density, scale


def build_density_map(
    image_bgr: np.ndarray,
    coarse_dets: List[Dict],
    rows: int = 32,
    cols: int = 32,
    use_coarse_prior: bool = True,
    use_image_prior: bool = True,
    coarse_weight: float = 0.65,
    image_weight: float = 0.35,
) -> DensityMap:
    """Combine coarse detection prior and image energy into a DensityMap."""
    h, w = image_bgr.shape[:2]
    dens = np.zeros((rows, cols), dtype=np.float32)
    scale = np.zeros((rows, cols), dtype=np.float32)

    if use_coarse_prior and coarse_dets:
        d_c, s_c = coarse_detection_maps(coarse_dets, h, w, rows, cols)
        dens = dens + coarse_weight * d_c
        scale = s_c
    if use_image_prior:
        d_i = image_energy_map(image_bgr, rows, cols)
        dens = dens + image_weight * d_i

    mx = float(dens.max())
    if mx > 0:
        dens /= mx

    # Fill empty scale cells with global median or default
    nonzero = scale[scale > 0]
    default_scale = float(np.median(nonzero)) if nonzero.size else 32.0
    scale = np.where(scale > 0, scale, default_scale)

    return DensityMap(
        density=dens,
        scale=scale,
        rows=rows,
        cols=cols,
        image_h=h,
        image_w=w,
    )
