"""Pareto plot for the paper using only full-val (548-image) YOLOv8n runs."""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"

RUNS = [
    ("Full image", RES / "das_v2" / "yolov8n_full.json", "o", "tab:blue", (8, 6)),
    ("SAHI 512/0.2 + full", RES / "sahi" / "yolov8n_s512_o0.2_full1.json", "s", "tab:orange", (-60, -16)),
    ("DAS v1 $N_{max}$=4", RES / "das" / "yolov8n_das_n4_fullval.json", "v", "tab:gray", (-30, 10)),
    ("DAS v1 $N_{max}$=6", RES / "das" / "yolov8n_das_n6_fullval.json", "v", "tab:gray", (-20, -16)),
    ("DAS v1 $N_{max}$=8", RES / "das" / "yolov8n_das_n8_fullval.json", "v", "tab:gray", (8, 4)),
    ("DAS v2 $N_{max}$=4", RES / "das_v2" / "yolov8n_das_n4_v2.json", "D", "tab:green", (-40, -16)),
    ("DAS v2 $N_{max}$=6", RES / "das_v2" / "yolov8n_das_n6_v2.json", "D", "tab:green", (8, -4)),
]


def main():
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    for label, path, marker, color, offset in RUNS:
        d = json.loads(path.read_text())
        assert d["n_images"] == 548, (path, d["n_images"])
        x = d["latency"]["mean_ms"]
        y = d["metrics"]["AP_small"] * 100
        ax.scatter(x, y, marker=marker, color=color, s=60, zorder=3)
        ax.annotate(label, (x, y), textcoords="offset points", xytext=offset, fontsize=8)
    ax.set_xlabel("Mean latency (ms / image, CPU)")
    ax.set_ylabel("AP$_{small}$ (%)")
    ax.set_title("YOLOv8n on VisDrone2019-DET val (548 images)")
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 6500)
    ax.set_ylim(12, 19.8)
    fig.tight_layout()
    for out in (ROOT / "paper" / "pareto_yolov8n.png", ROOT / "paper" / "figures" / "pareto_yolov8n.png"):
        fig.savefig(out, dpi=200)
    print("saved")


if __name__ == "__main__":
    main()
