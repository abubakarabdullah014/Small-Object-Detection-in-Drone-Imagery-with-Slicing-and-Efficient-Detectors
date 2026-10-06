"""
Unified experiment runner: baselines, SAHI grid, DAS, ablations.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import (
    CROSS_DATASET_MODELS,
    DATA_COCO,
    DEFAULT_CONF,
    IN_DOMAIN_MODELS,
    RESULTS_DIR,
)
from src.bench_latency import aggregate_image_latencies, configure_threads
from src.das import DASConfig, run_das
from src.eval_coco import evaluate_coco, load_coco_images, subset_gt_json
from src.zoo import (
    load_sahi_model,
    predict_full_image,
    predict_sahi_sliced,
    sahi_prediction_to_dicts,
)


def _eval_gt(gt_json: Path, max_images: Optional[int], out_dir: Path) -> Path:
    return subset_gt_json(gt_json, max_images, out_dir=out_dir / "_gt_subsets")


def load_config(path: Optional[str] = None) -> Dict:
    cfg_path = Path(path) if path else Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _is_cross(name: str) -> bool:
    return name in CROSS_DATASET_MODELS


def run_full_baseline(
    model_name: str,
    gt_json: Path,
    out_dir: Path,
    conf: float = DEFAULT_CONF,
    device: str = "cpu",
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """Full-image no-slicing baseline."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_coco_images(gt_json)
    if max_images:
        images = images[:max_images]

    model = load_sahi_model(model_name, conf=conf, device=device)
    cross = _is_cross(model_name)
    all_dets: List[Dict] = []
    latencies: List[float] = []

    for im in tqdm(images, desc=f"full/{model_name}"):
        t0 = time.perf_counter()
        res = predict_full_image(model, im["path"], conf=conf)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        dets = sahi_prediction_to_dicts(
            res, image_id=im["id"], filter_coco_for_visdrone=cross
        )
        all_dets.extend(dets)

    eval_gt = _eval_gt(gt_json, max_images, out_dir)
    metrics = evaluate_coco(eval_gt, all_dets, save_results=out_dir / f"{model_name}_dets.json")
    lat = aggregate_image_latencies(latencies, forwards=[1.0] * len(latencies))
    result = {
        "mode": "full",
        "model": model_name,
        "metrics": metrics,
        "latency": lat.to_dict(),
        "n_images": len(images),
    }
    with open(out_dir / f"{model_name}_full.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_sahi_config(
    model_name: str,
    gt_json: Path,
    out_dir: Path,
    slice_size: int,
    overlap: float,
    with_full: bool,
    conf: float = DEFAULT_CONF,
    device: str = "cpu",
    max_images: Optional[int] = None,
) -> Dict[str, Any]:
    """One SAHI uniform-slicing configuration."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_coco_images(gt_json)
    if max_images:
        images = images[:max_images]

    model = load_sahi_model(model_name, conf=conf, device=device)
    cross = _is_cross(model_name)
    all_dets: List[Dict] = []
    latencies: List[float] = []
    n_slices_list: List[float] = []

    tag = f"{model_name}_s{slice_size}_o{overlap}_full{int(with_full)}"
    for im in tqdm(images, desc=f"sahi/{tag}"):
        t0 = time.perf_counter()
        res = predict_sahi_sliced(
            model,
            im["path"],
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap,
            overlap_width_ratio=overlap,
            perform_standard_pred=with_full,
            conf=conf,
        )
        latencies.append((time.perf_counter() - t0) * 1000.0)
        # Approximate slice count from image size
        h, w = im["height"], im["width"]
        step = int(slice_size * (1 - overlap))
        nx = max(1, int((w - slice_size) / step) + 1) if w > slice_size else 1
        ny = max(1, int((h - slice_size) / step) + 1) if h > slice_size else 1
        n_slices = nx * ny + (1 if with_full else 0)
        n_slices_list.append(float(n_slices))

        dets = sahi_prediction_to_dicts(
            res, image_id=im["id"], filter_coco_for_visdrone=cross
        )
        all_dets.extend(dets)

    eval_gt = _eval_gt(gt_json, max_images, out_dir)
    metrics = evaluate_coco(eval_gt, all_dets, save_results=out_dir / f"{tag}_dets.json")
    lat = aggregate_image_latencies(latencies, slices=n_slices_list, forwards=n_slices_list)
    result = {
        "mode": "sahi",
        "model": model_name,
        "slice_size": slice_size,
        "overlap": overlap,
        "with_full": with_full,
        "metrics": metrics,
        "latency": lat.to_dict(),
        "n_images": len(images),
        "eval_gt": str(eval_gt),
    }
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return result


def run_das_config(
    model_name: str,
    gt_json: Path,
    out_dir: Path,
    das_cfg: DASConfig,
    conf: float = DEFAULT_CONF,
    device: str = "cpu",
    max_images: Optional[int] = None,
    tag_suffix: str = "",
) -> Dict[str, Any]:
    """Run DAS with a given config."""
    out_dir.mkdir(parents=True, exist_ok=True)
    images = load_coco_images(gt_json)
    if max_images:
        images = images[:max_images]

    model = load_sahi_model(model_name, conf=conf, device=device)
    cross = _is_cross(model_name)
    all_dets: List[Dict] = []
    latencies: List[float] = []
    slices_list: List[float] = []
    forwards_list: List[float] = []

    tag = f"{model_name}_das_n{das_cfg.n_max}{tag_suffix}"
    for im in tqdm(images, desc=f"das/{tag}"):
        result = run_das(
            model,
            im["path"],
            image_id=im["id"],
            cfg=das_cfg,
            filter_coco=cross,
        )
        latencies.append(result.latency_ms)
        slices_list.append(float(len(result.slices)))
        forwards_list.append(float(result.forward_passes))
        all_dets.extend(result.detections)

    eval_gt = _eval_gt(gt_json, max_images, out_dir)
    metrics = evaluate_coco(eval_gt, all_dets, save_results=out_dir / f"{tag}_dets.json")
    lat = aggregate_image_latencies(latencies, slices=slices_list, forwards=forwards_list)
    out = {
        "mode": "das",
        "model": model_name,
        "das": das_cfg.__dict__,
        "metrics": metrics,
        "latency": lat.to_dict(),
        "n_images": len(images),
        "tag": tag,
        "eval_gt": str(eval_gt),
    }
    with open(out_dir / f"{tag}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Run VisDrone experiments")
    parser.add_argument("--config", default=None)
    parser.add_argument(
        "--mode",
        required=True,
        choices=["baseline", "sahi", "das", "ablation", "tune", "all"],
    )
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--split", default="val", choices=["val", "test-dev", "tune"])
    parser.add_argument("--max-images", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--n-max", type=int, default=None)
    parser.add_argument("--quick", action="store_true", help="Small SAHI grid for smoke test")
    args = parser.parse_args()

    cfg = load_config(args.config)
    configure_threads(cfg["runtime"].get("num_threads", 4))
    conf = cfg["runtime"].get("conf", 0.25)
    device = args.device or cfg["runtime"].get("device", "cpu")

    if args.split == "tune":
        gt_json = Path(__file__).resolve().parents[1] / "data" / "tuning" / "tune_200_seed42.json"
    else:
        gt_json = DATA_COCO / f"{args.split}.json"
    if not gt_json.exists():
        raise FileNotFoundError(f"Missing {gt_json}. Run convert_visdrone.py first.")

    models = args.models or cfg["models"]["in_domain"]
    # Default to first lightweight model for long grids unless specified
    results_summary: List[Dict] = []

    if args.mode in ("baseline", "all"):
        out = RESULTS_DIR / "baselines"
        for m in models:
            r = run_full_baseline(m, gt_json, out, conf=conf, device=device, max_images=args.max_images)
            results_summary.append(r)
            print(m, r["metrics"], r["latency"]["mean_ms"])

    if args.mode in ("sahi", "all"):
        out = RESULTS_DIR / "sahi"
        sizes = cfg["sahi"]["slice_sizes"]
        overlaps = cfg["sahi"]["overlaps"]
        fulls = cfg["sahi"]["with_full_image"]
        if args.quick:
            sizes = [512]
            overlaps = [0.2]
            fulls = [True]
        for m in models:
            for ss in sizes:
                for ov in overlaps:
                    for wf in fulls:
                        r = run_sahi_config(
                            m, gt_json, out, ss, ov, wf,
                            conf=conf, device=device, max_images=args.max_images,
                        )
                        results_summary.append(r)
                        print(r["model"], ss, ov, wf, r["metrics"]["AP_small"], r["latency"]["mean_ms"])

    if args.mode in ("das", "all"):
        out = RESULTS_DIR / "das"
        das_dict = dict(cfg["das"])
        if args.n_max is not None:
            n_values = [args.n_max]
        else:
            n_values = [2, 4, 6, 8, 12]
        for m in models:
            for n_max in n_values:
                das_dict["n_max"] = n_max
                das_cfg = DASConfig.from_dict(das_dict)
                r = run_das_config(
                    m, gt_json, out, das_cfg,
                    conf=conf, device=device, max_images=args.max_images,
                )
                results_summary.append(r)
                print(r["tag"], r["metrics"]["AP_small"], r["latency"]["mean_ms"])

    if args.mode == "ablation":
        out = RESULTS_DIR / "ablations"
        base = dict(cfg["das"])
        base["n_max"] = args.n_max or 6
        ablations = {
            "coarse_only": {**base, "use_coarse_prior": True, "use_image_prior": False},
            "image_only": {**base, "use_coarse_prior": False, "use_image_prior": True},
            "both_priors": {**base, "use_coarse_prior": True, "use_image_prior": True},
            "fixed_slice": {**base, "scale_adaptive": False},
            "scale_adaptive": {**base, "scale_adaptive": True},
            "fusion_nms": {**base, "fusion": "nms"},
            "fusion_wbf": {**base, "fusion": "wbf"},
        }
        for m in models:
            for name, d in ablations.items():
                das_cfg = DASConfig.from_dict(d)
                r = run_das_config(
                    m, gt_json, out, das_cfg,
                    conf=conf, device=device, max_images=args.max_images,
                    tag_suffix=f"_{name}",
                )
                results_summary.append(r)
                print(name, r["metrics"]["AP_small"], r["latency"]["mean_ms"])

    if args.mode == "tune":
        # Hyperparameter search on tuning subset only
        out = RESULTS_DIR / "das" / "tune"
        out.mkdir(parents=True, exist_ok=True)
        m = models[0]
        best = None
        best_score = -1.0
        grid = []
        for tau in [0.10, 0.15, 0.25]:
            for k in [6.0, 8.0, 10.0]:
                for n_max in [4, 6, 8]:
                    grid.append({"tau": tau, "k": k, "n_max": n_max})
        for g in grid:
            d = dict(cfg["das"])
            d.update(g)
            das_cfg = DASConfig.from_dict(d)
            r = run_das_config(
                m, gt_json, out, das_cfg,
                conf=conf, device=device, max_images=args.max_images,
                tag_suffix=f"_tau{g['tau']}_k{g['k']}",
            )
            # Prefer AP_small with latency penalty
            score = r["metrics"]["AP_small"] - 0.00001 * r["latency"]["mean_ms"]
            r["tune_score"] = score
            results_summary.append(r)
            if score > best_score:
                best_score = score
                best = r
        if best:
            with open(out / "best_hparams.json", "w", encoding="utf-8") as f:
                json.dump(best, f, indent=2)
            print("BEST", best["das"], best["metrics"]["AP_small"])

    summary_path = RESULTS_DIR / f"summary_{args.mode}_{args.split}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results_summary, f, indent=2)
    print(f"Wrote {summary_path}")


if __name__ == "__main__":
    main()
