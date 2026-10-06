"""
Held-out test-dev evaluation with frozen DAS v2 settings.
Runs sequentially: full-image -> SAHI -> DAS N=4 -> DAS N=6 (YOLOv8n).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import DATA_COCO, RESULTS_DIR
from src.bench_latency import configure_threads
from src.das import DASConfig
from src.run_experiments import run_das_config, run_full_baseline, run_sahi_config
from src.zoo import download_yolo_weights


def main():
    cfg = yaml.safe_load(open(ROOT / "configs" / "das_frozen.yaml", encoding="utf-8"))
    configure_threads(cfg["runtime"].get("num_threads", 4))
    conf = cfg["runtime"].get("conf", 0.25)
    device = "cpu"
    gt = DATA_COCO / "test-dev.json"
    if not gt.exists():
        raise FileNotFoundError(gt)

    out = RESULTS_DIR / "testdev_heldout"
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run_log.txt"

    def log(msg: str):
        print(msg, flush=True)
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg + "\n")

    log("Downloading yolov8n weights if needed...")
    p = download_yolo_weights("yolov8n")
    log(f"weights: {p}")

    summary = []

    log("=== 1/4 full-image baseline ===")
    r = run_full_baseline("yolov8n", gt, out, conf=conf, device=device)
    summary.append(r)
    log(f"FULL AP_s={r['metrics']['AP_small']:.4f} AP={r['metrics']['AP']:.4f} ms={r['latency']['mean_ms']:.1f}")

    log("=== 2/4 SAHI 512/0.2 + full ===")
    r = run_sahi_config(
        "yolov8n", gt, out, 512, 0.2, True, conf=conf, device=device
    )
    summary.append(r)
    log(f"SAHI AP_s={r['metrics']['AP_small']:.4f} AP={r['metrics']['AP']:.4f} ms={r['latency']['mean_ms']:.1f}")

    das = DASConfig.from_dict(cfg["das"])
    for n_max in (4, 6):
        log(f"=== DAS v2 N_max={n_max} ===")
        das.n_max = n_max
        r = run_das_config(
            "yolov8n", gt, out, das, conf=conf, device=device, tag_suffix=f"_v2_n{n_max}"
        )
        summary.append(r)
        log(
            f"DAS{n_max} AP_s={r['metrics']['AP_small']:.4f} "
            f"AP={r['metrics']['AP']:.4f} ms={r['latency']['mean_ms']:.1f}"
        )

    with open(out / "summary_testdev.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    md = ["# Held-out VisDrone test-dev (1610 images), YOLOv8n, frozen DAS v2", "",
          "| Method | AP_small | AP | ms/img |",
          "|--------|----------|------|--------|"]
    for r in summary:
        name = r.get("tag") or r.get("mode")
        if r["mode"] == "full":
            name = "Full image"
        elif r["mode"] == "sahi":
            name = "SAHI 512/0.2 + full"
        md.append(
            f"| {name} | {r['metrics']['AP_small']:.3f} | {r['metrics']['AP']:.3f} | "
            f"{r['latency']['mean_ms']:.0f} |"
        )
    (out / "RESULTS_testdev.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    log("DONE. Wrote " + str(out / "RESULTS_testdev.md"))


if __name__ == "__main__":
    main()
