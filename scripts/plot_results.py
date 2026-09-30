#!/usr/bin/env python3
"""Plot measured CSV files. Uses matplotlib defaults, not simulated measurements."""
from __future__ import annotations
import argparse
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator
ROOT = Path(__file__).resolve().parents[1]

def records(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "build/results/experiments")
    parser.add_argument("--out", type=Path, default=ROOT / "build/results/plots")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    speed = records(args.results / "speed_summary.csv")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for estep, threads, label, marker in [("dense", "1", "Full-U reference, 1 thread", "s"),
                                         ("streaming", "1", "Streaming, 1 thread", "o"),
                                         ("streaming", "4", "Streaming, 4 threads", "^")]:
        rows = [r for r in speed if r["estep"] == estep and r["threads"] == threads]
        ax.plot([int(r["n"]) for r in rows], [float(r["median_fit_seconds"]) for r in rows], marker=marker, label=label)
    ax.set(xscale="log", yscale="log", xlabel="Source and target points (each)", ylabel="Fit wall time (seconds)",
           title="Same rank 128, same 12 iterations; median of 3 runs")
    ax.set_xticks([512, 2048, 4096], labels=["512", "2,048", "4,096"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.grid(True, which="major", alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "speed.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for estep, threads, label, marker in [("dense", "1", "Full-U reference, 1 thread", "s"),
                                         ("streaming", "4", "Streaming, 4 threads", "^")]:
        rows = [r for r in speed if r["estep"] == estep and r["threads"] == threads]
        ax.plot([int(r["n"]) for r in rows], [float(r["median_peak_rss_kib"]) / 1024 for r in rows], marker=marker, label=label)
    ax.set(xlabel="Source and target points (each)", ylabel="Whole-process peak RSS (MiB)",
           title="Measured CLI memory; same rank and iterations")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "memory.png", dpi=170); plt.close(fig)

    quality = [r for r in records(args.results / "quality_summary.csv") if r["mode"] == "paper"]
    fig, ax = plt.subplots(figsize=(8.7, 4.8))
    positions = np.arange(len(quality))
    ax.barh(positions - 0.18, [float(r["mean_gt_rmse_prealigned"]) for r in quality], height=0.36, label="After centroid/scale alignment")
    ax.barh(positions + 0.18, [float(r["mean_gt_rmse_after"]) for r in quality], height=0.36, label="After nonrigid registration")
    ax.set_yticks(positions, labels=[r["case"] for r in quality])
    ax.invert_yaxis()
    ax.set(xlabel="True-correspondence RMSE (original coordinates)", title="Mean of 3 seeds; paper mode, default rank ratio 0.3")
    ax.legend(loc="upper right"); ax.grid(True, axis="x", alpha=0.25); fig.tight_layout()
    fig.savefig(args.out / "quality.png", dpi=170); plt.close(fig)

    rows = records(args.results / "rank_ablation.csv")
    ranks = [int(r["rank"]) for r in rows]
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(ranks, [float(r["relative_kernel_frobenius_error"]) for r in rows], marker="o")
    ax.set(xlabel="Landmark rank", ylabel="Relative Frobenius kernel error", yscale="log",
           title="Nystrom kernel approximation, N=1,024")
    ax.grid(True, which="major", alpha=0.25); fig.tight_layout()
    fig.savefig(args.out / "rank_kernel.png", dpi=170); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot(ranks, [float(r["gt_rmse_after"]) for r in rows], marker="o", label="True-correspondence RMSE")
    ax.plot(ranks, [float(r["nn_source_to_target_rmse"]) for r in rows], marker="s", label="Nearest-neighbor RMSE")
    ax.set(xlabel="Landmark rank", ylabel="RMSE (original coordinates)",
           title="Smaller kernel error does not guarantee smaller registration error")
    ax.grid(True, alpha=0.25); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "rank_registration.png", dpi=170); plt.close(fig)

    path = args.results / "examples/curve2d_clean"
    source = np.loadtxt(path / "source.csv", delimiter=",")
    target = np.loadtxt(path / "target.csv", delimiter=",")
    registered = np.loadtxt(path / "registered.csv", delimiter=",")
    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    ax.scatter(source[:, 0], source[:, 1], s=13, marker="x", label="Source")
    ax.scatter(target[:, 0], target[:, 1], s=19, marker="o", label="Target")
    ax.scatter(registered[:, 0], registered[:, 1], s=8, marker=".", label="Registered source")
    ax.set(xlabel="x", ylabel="y", title="2D synthetic example; target order shuffled")
    ax.set_aspect("equal"); ax.legend(); fig.tight_layout()
    fig.savefig(args.out / "registration_2d.png", dpi=170); plt.close(fig)

    history = records(args.results / "examples/surface3d_mild/history.csv")
    fig, ax = plt.subplots(figsize=(7.8, 4.8))
    ax.plot([int(r["iteration"]) for r in history], [float(r["after_variance"]) for r in history], marker="o")
    ax.set(xlabel="Iteration", ylabel="Constrained objective J", title="Paper mode: objective after each complete iteration")
    ax.grid(True, alpha=0.25); fig.tight_layout()
    fig.savefig(args.out / "objective.png", dpi=170); plt.close(fig)
    print("Measured-data plots written to", args.out)

if __name__ == "__main__":
    main()
