"""
DAS slice scheduler: select regions by density, set per-region slice size,
greedily fill up to N_max slices.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from .density import DensityMap


@dataclass
class SliceWindow:
    x1: int
    y1: int
    x2: int
    y2: int
    score: float
    slice_size: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _connected_components(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    """4-connected components of True cells. Returns list of [(r,c), ...]."""
    rows, cols = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(rows):
        for c in range(cols):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            cells: List[Tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < rows and 0 <= nc < cols and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def region_bbox(dmap: DensityMap, cells: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    xs1, ys1, xs2, ys2 = [], [], [], []
    for r, c in cells:
        x1, y1, x2, y2 = dmap.cell_bbox(r, c)
        xs1.append(x1)
        ys1.append(y1)
        xs2.append(x2)
        ys2.append(y2)
    return min(xs1), min(ys1), max(xs2), max(ys2)


def schedule_slices(
    dmap: DensityMap,
    n_max: int = 6,
    tau: float = 0.15,
    k: float = 8.0,
    slice_min: int = 256,
    slice_max: int = 1024,
    default_slice: int = 512,
    overlap_ratio: float = 0.2,
    scale_adaptive: bool = True,
) -> List[SliceWindow]:
    """
    1. Threshold density map at tau * max
    2. Connected components -> candidate regions
    3. Per-region slice size from median object scale
    4. Tile each region; greedily keep highest-scoring windows up to n_max
    """
    dens = dmap.density
    mx = float(dens.max()) if dens.size else 0.0
    if mx <= 0 or n_max <= 0:
        return []

    thr = tau * mx
    mask = dens >= thr
    if not mask.any():
        # Fallback: take top-k cells
        flat = dens.ravel()
        top = min(n_max, flat.size)
        idxs = np.argpartition(-flat, top - 1)[:top]
        mask = np.zeros_like(dens, dtype=bool)
        for i in idxs:
            mask[np.unravel_index(int(i), dens.shape)] = True

    comps = _connected_components(mask)
    # Score each component by sum of density
    scored_comps = []
    for cells in comps:
        score = float(sum(dens[r, c] for r, c in cells))
        med_scale = float(np.median([dmap.scale[r, c] for r, c in cells]))
        scored_comps.append((score, cells, med_scale))
    scored_comps.sort(key=lambda x: -x[0])

    candidates: List[SliceWindow] = []
    for score, cells, med_scale in scored_comps:
        if scale_adaptive:
            ss = int(np.clip(k * med_scale, slice_min, slice_max))
        else:
            ss = default_slice
        # Make even
        ss = max(slice_min, (ss // 32) * 32)
        x1, y1, x2, y2 = region_bbox(dmap, cells)
        # Expand region a bit so objects near boundary are covered
        pad = int(ss * 0.1)
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(dmap.image_w, x2 + pad)
        y2 = min(dmap.image_h, y2 + pad)

        # Tile region with overlapping windows of size ss
        step = max(1, int(ss * (1.0 - overlap_ratio)))
        ys = list(range(y1, max(y1 + 1, y2 - ss + 1), step))
        xs = list(range(x1, max(x1 + 1, x2 - ss + 1), step))
        if not ys:
            ys = [y1]
        if not xs:
            xs = [x1]
        # Ensure last window covers edge
        if ys[-1] + ss < y2:
            ys.append(max(y1, y2 - ss))
        if xs[-1] + ss < x2:
            xs.append(max(x1, x2 - ss))

        for yy in ys:
            for xx in xs:
                wx2 = min(dmap.image_w, xx + ss)
                wy2 = min(dmap.image_h, yy + ss)
                wx1 = max(0, wx2 - ss) if wx2 - xx < ss // 2 else xx
                wy1 = max(0, wy2 - ss) if wy2 - yy < ss // 2 else yy
                # Local density score for this window
                # Approximate by sampling cells covered
                cell_h = dmap.image_h / dmap.rows
                cell_w = dmap.image_w / dmap.cols
                r0 = int(wy1 / cell_h)
                r1 = int(min(dmap.rows - 1, (wy2 - 1) / cell_h))
                c0 = int(wx1 / cell_w)
                c1 = int(min(dmap.cols - 1, (wx2 - 1) / cell_w))
                local = float(dens[r0 : r1 + 1, c0 : c1 + 1].mean()) if dens.size else 0.0
                candidates.append(
                    SliceWindow(
                        x1=int(wx1),
                        y1=int(wy1),
                        x2=int(wx2),
                        y2=int(wy2),
                        score=local,
                        slice_size=ss,
                    )
                )

    # Deduplicate near-identical windows, keep highest score
    candidates.sort(key=lambda w: -w.score)
    selected: List[SliceWindow] = []
    for w in candidates:
        if len(selected) >= n_max:
            break
        # Suppress highly overlapping windows (IoU > 0.7)
        keep = True
        for s in selected:
            iou = _iou(w, s)
            if iou > 0.7:
                keep = False
                break
        if keep:
            selected.append(w)
    return selected


def _iou(a: SliceWindow, b: SliceWindow) -> float:
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = a.width * a.height
    area_b = b.width * b.height
    return inter / max(1.0, area_a + area_b - inter)
