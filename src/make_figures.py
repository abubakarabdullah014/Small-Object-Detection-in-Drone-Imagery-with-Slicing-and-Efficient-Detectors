"""
Generate Pareto plots, summary tables, and qualitative figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import RESULTS_DIR


def collect_results(folders: List[Path]) -> pd.DataFrame:
    rows = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in folder.rglob("*.json"):
            if p.name.endswith("_dets.json") or p.name.startswith("summary_"):
                continue
            if p.name == "best_hparams.json":
                continue
            try:
                with open(p, encoding="utf-8") as f:
                    d = json.load(f)
            except Exception:
                continue
            if "metrics" not in d:
                continue
            row = {
                "file": str(p),
                "mode": d.get("mode", ""),
                "model": d.get("model", ""),
                "tag": d.get("tag", p.stem),
                "AP": d["metrics"].get("AP", 0),
                "AP50": d["metrics"].get("AP50", 0),
                "AP_small": d["metrics"].get("AP_small", 0),
                "AR_small": d["metrics"].get("AR_small", 0),
                "mean_ms": d.get("latency", {}).get("mean_ms", 0),
                "slices": d.get("latency", {}).get("slices_per_image", 0),
                "slice_size": d.get("slice_size"),
                "overlap": d.get("overlap"),
                "with_full": d.get("with_full"),
                "n_max": (d.get("das") or {}).get("n_max"),
            }
            rows.append(row)
    return pd.DataFrame(rows)


def plot_pareto(df: pd.DataFrame, out_path: Path, model: str = None):
    fig, ax = plt.subplots(figsize=(8, 5))
    data = df if model is None else df[df["model"] == model]
    for mode, marker in [("full", "o"), ("sahi", "s"), ("das", "D")]:
        sub = data[data["mode"] == mode]
        if sub.empty:
            continue
        ax.scatter(
            sub["mean_ms"],
            sub["AP_small"] * 100,
            label=mode.upper(),
            marker=marker,
            s=60,
            alpha=0.85,
        )
        # Connect DAS points by n_max
        if mode == "das" and "n_max" in sub.columns:
            sub2 = sub.sort_values("n_max")
            ax.plot(sub2["mean_ms"], sub2["AP_small"] * 100, "--", alpha=0.5)
    ax.set_xlabel("Latency (ms / image, CPU)")
    ax.set_ylabel("AP_small (%)")
    title = "AP_small vs Latency"
    if model:
        title += f" — {model}"
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_path}")


def write_tables(df: pd.DataFrame, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "all_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"[table] {csv_path}")

    # Markdown summary
    md = out_dir / "results_table.md"
    cols = ["mode", "model", "tag", "AP", "AP50", "AP_small", "AR_small", "mean_ms", "slices"]
    cols = [c for c in cols if c in df.columns]
    with open(md, "w", encoding="utf-8") as f:
        f.write(df[cols].sort_values(["model", "mode", "mean_ms"]).to_markdown(index=False))
    print(f"[table] {md}")


def plot_qualitative_placeholder(out_path: Path):
    """Simple schematic of DAS vs SAHI slicing (no GT needed)."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    # SAHI uniform
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("SAHI: uniform grid")
    for x in range(0, 10, 3):
        for y in range(0, 6, 3):
            ax.add_patch(plt.Rectangle((x, y), 3.2, 3.2, fill=False, edgecolor="C0", lw=2))
    ax.set_xticks([])
    ax.set_yticks([])
    # DAS selective
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.set_title("DAS: density-budgeted slices")
    for (x, y, s) in [(1, 1, 2.5), (5.5, 0.5, 2.0), (7, 3.5, 2.8)]:
        ax.add_patch(plt.Rectangle((x, y), s, s, fill=False, edgecolor="C3", lw=2))
    ax.scatter([2, 2.5, 6, 8, 8.5], [2, 2.3, 1.2, 4.5, 5], c="C3", s=20, alpha=0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"[fig] {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    folders = [
        RESULTS_DIR / "baselines",
        RESULTS_DIR / "sahi",
        RESULTS_DIR / "das",
        RESULTS_DIR / "das_v2",
        RESULTS_DIR / "ablations",
    ]
    df = collect_results(folders)
    fig_dir = RESULTS_DIR / "figures"
    paper_fig = Path(__file__).resolve().parents[1] / "paper" / "figures"

    if df.empty:
        print("No results yet; writing schematic only.")
        plot_qualitative_placeholder(fig_dir / "das_vs_sahi_schematic.png")
        plot_qualitative_placeholder(paper_fig / "das_vs_sahi_schematic.png")
        return

    write_tables(df, fig_dir)
    models = [args.model] if args.model else sorted(df["model"].dropna().unique())
    for m in models:
        plot_pareto(df, fig_dir / f"pareto_{m}.png", model=m)
        plot_pareto(df, paper_fig / f"pareto_{m}.png", model=m)
    plot_pareto(df, fig_dir / "pareto_all.png", model=None)
    plot_pareto(df, paper_fig / "pareto_all.png", model=None)
    plot_qualitative_placeholder(fig_dir / "das_vs_sahi_schematic.png")
    plot_qualitative_placeholder(paper_fig / "das_vs_sahi_schematic.png")


if __name__ == "__main__":
    main()
