"""Parallel worker: DAS N=6 improved on full val."""
from pathlib import Path
import sys
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_experiments import run_das_config
from src.das import DASConfig
from src import DATA_COCO, RESULTS_DIR

cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "das_frozen.yaml"))
d = DASConfig.from_dict(cfg["das"])
d.n_max = 6
out = RESULTS_DIR / "das_v2"
out.mkdir(exist_ok=True)
r = run_das_config("yolov8n", DATA_COCO / "val.json", out, d, tag_suffix="_v2")
print("DAS6", r["metrics"]["AP_small"], r["metrics"]["AP"], r["latency"]["mean_ms"])
