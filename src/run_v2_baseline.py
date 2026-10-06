"""Parallel worker: full-image baseline on full val."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_experiments import run_full_baseline
from src import DATA_COCO, RESULTS_DIR

out = RESULTS_DIR / "das_v2"
out.mkdir(exist_ok=True)
r = run_full_baseline("yolov8n", DATA_COCO / "val.json", out)
print("BASELINE", r["metrics"]["AP_small"], r["metrics"]["AP"], r["latency"]["mean_ms"])
