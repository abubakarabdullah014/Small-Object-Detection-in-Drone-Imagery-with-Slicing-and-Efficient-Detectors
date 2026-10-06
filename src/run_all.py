"""
Feasible CPU experiment chain for laptop (no GPU).

Protocol (honest & reportable):
  Phase A — Explore on 100-image val subset (grids, ablations, tune)
  Phase B — Confirm best SAHI + best DAS on full 548-image val
  Phase C — Generalization (test-dev / domain-shift) + figures
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
EXPLORE_N = 50  # images for grid search / ablations (CPU laptop)
PY = sys.executable


def run(cmd: list[str]) -> int:
    print("\n===", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT))
    print("exit", r.returncode, flush=True)
    return r.returncode


def write_paper_sahi_config():
    """
    Reduced SAHI grid that still covers all slice sizes and key overlaps:
      sizes {384,512,640,768} x overlap 0.2 x with_full True
      plus 512 x {0.1,0.3} x True, and 512 x 0.2 x False
    = 4 + 2 + 1 = 7 configs (vs 24 full factorial)
    """
    cfg_path = ROOT / "configs" / "default.yaml"
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Temporarily narrow via a dedicated config
    paper = dict(cfg)
    paper["sahi"] = {
        "slice_sizes": [384, 512, 640, 768],
        "overlaps": [0.2],
        "with_full_image": [True],
    }
    out = ROOT / "configs" / "sahi_paper.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(paper, f)
    return out


def run_extra_sahi_points(max_images: int):
    """Extra overlap / no-full points at slice=512."""
    extras = [
        (512, 0.1, True),
        (512, 0.3, True),
        (512, 0.2, False),
    ]
    for ss, ov, wf in extras:
        # Use inline python to call run_sahi_config
        code = f"""
from pathlib import Path
from src.run_experiments import run_sahi_config
from src import DATA_COCO, RESULTS_DIR
run_sahi_config(
    'yolov8n', DATA_COCO/'val.json', RESULTS_DIR/'sahi',
    slice_size={ss}, overlap={ov}, with_full={wf},
    max_images={max_images},
)
"""
        run([PY, "-c", code])


def freeze_from_best():
    best = ROOT / "results" / "das" / "tune" / "best_hparams.json"
    if not best.exists():
        print("No best_hparams.json; keep configs/das_frozen.yaml defaults")
        return
    with open(best, encoding="utf-8") as f:
        b = json.load(f)
    with open(ROOT / "configs" / "default.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for k in ("tau", "k", "n_max", "fusion", "scale_adaptive", "wbf_iou_thr"):
        if k in b.get("das", {}):
            cfg["das"][k] = b["das"][k]
    out = ROOT / "configs" / "das_frozen.yaml"
    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    print("Froze", out, cfg["das"])


def main():
    sahi_cfg = write_paper_sahi_config()

    # ----- Phase A: explore -----
    print(f"\n##### PHASE A: explore on {EXPLORE_N} images #####")
    run([
        PY, "src/run_experiments.py", "--mode", "sahi", "--models", "yolov8n",
        "--split", "val", "--max-images", str(EXPLORE_N), "--config", str(sahi_cfg),
    ])
    run_extra_sahi_points(EXPLORE_N)

    # Quick SAHI point on other models
    for m in ["yolov8s", "yolov11n", "yolov11s"]:
        run([
            PY, "src/run_experiments.py", "--mode", "sahi", "--models", m,
            "--split", "val", "--quick", "--max-images", str(EXPLORE_N),
        ])

    # DAS N_max sweep
    run([
        PY, "src/run_experiments.py", "--mode", "das", "--models", "yolov8n",
        "--split", "val", "--max-images", str(EXPLORE_N),
    ])

    # Tune on train subset (200 images already)
    tune = ROOT / "data" / "tuning" / "tune_200_seed42.json"
    if tune.exists():
        run([
            PY, "src/run_experiments.py", "--mode", "tune", "--models", "yolov8n",
            "--split", "tune", "--max-images", "50",  # 50 of the 200 for speed; seed fixed
        ])
        freeze_from_best()

    # Ablations
    run([
        PY, "src/run_experiments.py", "--mode", "ablation", "--models", "yolov8n",
        "--split", "val", "--n-max", "6", "--max-images", str(EXPLORE_N),
    ])

    # D-FINE cross-dataset baseline (subset)
    run([
        PY, "src/run_experiments.py", "--mode", "baseline", "--models", "dfine_n",
        "--split", "val", "--max-images", "50",
    ])

    # ----- Phase B: confirm on full val -----
    print("\n##### PHASE B: confirm best configs on full val #####")
    # Best default SAHI: 512 / 0.2 / full
    code = """
from pathlib import Path
from src.run_experiments import run_sahi_config, run_das_config
from src.das import DASConfig
from src import DATA_COCO, RESULTS_DIR
import yaml
run_sahi_config('yolov8n', DATA_COCO/'val.json', RESULTS_DIR/'sahi',
                slice_size=512, overlap=0.2, with_full=True)
cfg=yaml.safe_load(open('configs/das_frozen.yaml'))
d=DASConfig.from_dict(cfg['das'])
for n in [4, 6, 8]:
    d.n_max=n
    run_das_config('yolov8n', DATA_COCO/'val.json', RESULTS_DIR/'das', d,
                   tag_suffix='_fullval')
"""
    run([PY, "-c", code])

    # ----- Phase C: generalization + figures -----
    print("\n##### PHASE C: generalization + figures #####")
    if (ROOT / "data" / "coco" / "test-dev.json").exists():
        run([
            PY, "src/run_experiments.py", "--mode", "das", "--models", "yolov8n",
            "--split", "test-dev", "--n-max", "6", "--max-images", "200",
            "--config", str(ROOT / "configs" / "das_frozen.yaml"),
        ])

    run([PY, "src/make_domain_shift.py", "--n-crops", "80"])
    code_ds = """
from pathlib import Path
import yaml
from src.run_experiments import run_das_config, run_full_baseline
from src.das import DASConfig
gt=Path('data/domain_shift/person_crops.json')
cfg=yaml.safe_load(open('configs/das_frozen.yaml'))
d=DASConfig.from_dict(cfg['das']); d.n_max=6
run_full_baseline('yolov8n', gt, Path('results/baselines'), tag_ok:=True)
"""
    # Fix domain shift eval properly
    code_ds = """
from pathlib import Path
import yaml
from src.run_experiments import run_das_config, run_full_baseline
from src.das import DASConfig
from src import RESULTS_DIR
gt=Path('data/domain_shift/person_crops.json')
run_full_baseline('yolov8n', gt, RESULTS_DIR/'baselines')
cfg=yaml.safe_load(open('configs/das_frozen.yaml'))
d=DASConfig.from_dict(cfg['das']); d.n_max=6
run_das_config('yolov8n', gt, RESULTS_DIR/'das', d, tag_suffix='_domainshift')
"""
    run([PY, "-c", code_ds])

    run([PY, "src/visualize.py", "--model", "yolov8n", "--image-idx", "10"])
    run([PY, "src/visualize.py", "--model", "yolov8n", "--image-idx", "80"])
    run([PY, "src/make_figures.py"])

    # Write go/no-go decision stub
    decision = ROOT / "results" / "das_go_nogo.md"
    decision.write_text(
        "# DAS go / no-go\n\n"
        "Compare `results/das/*_fullval.json` AP_small vs `results/sahi/yolov8n_s512_o0.2_full1.json`.\n"
        "If any DAS N_max point has AP_small within ~1–2 pts of SAHI at clearly lower mean_ms → **GO**.\n"
        "Otherwise paper contribution rests on the SAHI benchmark + fusion ablation → **benchmark-primary**.\n",
        encoding="utf-8",
    )
    print("\n=== CHAIN COMPLETE ===")


if __name__ == "__main__":
    main()
