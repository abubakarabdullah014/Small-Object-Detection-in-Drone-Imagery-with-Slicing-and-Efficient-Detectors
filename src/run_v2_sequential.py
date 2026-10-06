"""Sequential v2 improvement runs — faster on 4-core CPU than parallel."""
from pathlib import Path
import sys
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.run_experiments import run_full_baseline, run_das_config
from src.das import DASConfig
from src import DATA_COCO, RESULTS_DIR

out = RESULTS_DIR / "das_v2"
out.mkdir(exist_ok=True)
cfg = yaml.safe_load(open(Path(__file__).resolve().parents[1] / "configs" / "das_frozen.yaml"))

print("=== 1/3 baseline ===", flush=True)
r0 = run_full_baseline("yolov8n", DATA_COCO / "val.json", out)
print("BASELINE", round(r0["metrics"]["AP_small"], 4), round(r0["metrics"]["AP"], 4), round(r0["latency"]["mean_ms"], 1), flush=True)

for n in (4, 6):
    print(f"=== DAS N={n} ===", flush=True)
    d = DASConfig.from_dict(cfg["das"])
    d.n_max = n
    r = run_das_config("yolov8n", DATA_COCO / "val.json", out, d, tag_suffix="_v2")
    print(f"DAS{n}", round(r["metrics"]["AP_small"], 4), round(r["metrics"]["AP"], 4), round(r["latency"]["mean_ms"], 1), flush=True)

print("=== ALL DONE ===", flush=True)
