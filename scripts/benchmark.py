#!/usr/bin/env python3
"""Reproducible CPU measurements and synthetic accuracy/robustness experiments.

Timings are fit() wall time, including normalization and landmark construction,
excluding input/output and metric computation. RSS is the WHOLE CLI process peak.
The dense-E baseline is this project's own mathematically equivalent reference,
NOT the upstream C++/MKL runtime. All seeds/cases, including failures, are retained.
"""
from __future__ import annotations
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"
import argparse
import csv
import json
from pathlib import Path
import platform
import shutil
import statistics
import subprocess
import tempfile
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]

def table(path: Path, rows: list[dict]) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, keys)
        writer.writeheader()
        writer.writerows(rows)

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exe", type=Path, default=ROOT / "build/pixi/clusterreg_cli")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "build/results/experiments")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-n", type=int, default=4096)
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing output directory under build/")
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("repeats must be positive")
    out = args.out_dir.resolve()
    if out.exists():
        build_root = (ROOT / "build").resolve()
        if not args.overwrite:
            raise FileExistsError(f"output already exists: {out}")
        if build_root not in out.parents:
            raise ValueError("--overwrite is restricted to a subdirectory of build/")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=False)
    exe = args.exe.resolve()
    environment = {"platform": platform.platform(), "python": platform.python_version(),
                   "numpy": np.__version__, "cpu_count_os": os.cpu_count(),
                   "compiler": subprocess.check_output(["c++", "--version"], text=True).splitlines()[0],
                   "timing_scope": "fit wall time incl. normalization and Nystrom construction; no I/O or evaluation",
                   "memory_scope": "whole CLI process maximum resident set from /usr/bin/time, KiB",
                   "baseline": "our dense full-U expectation + same reduced solver, not upstream runtime",
                   "repeats": args.repeats}
    for file, key in [("/proc/cpuinfo", "cpuinfo"), ("/sys/fs/cgroup/cpu.max", "cgroup_cpu_max")]:
        if Path(file).exists():
            data = Path(file).read_text()
            environment[key] = next((line for line in data.splitlines() if line.startswith("model name")), data.strip()) if key == "cpuinfo" else data.strip()
    (out / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    with tempfile.TemporaryDirectory(prefix="clusterreg_benchmark_") as temp:
        base = Path(temp)
        count = 0
        def run(**kwargs) -> tuple[dict, Path]:
            nonlocal count
            directory = base / str(count)
            count += 1
            command = [str(exe), "demo", "--out", str(directory)]
            for key, value in kwargs.items():
                command += ["--" + key.replace("_", "-"), str(value)]
            rss = base / f"rss_{count}.txt"
            if Path("/usr/bin/time").exists():
                command = ["/usr/bin/time", "-f", "%M", "-o", str(rss), *command]
            result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=300)
            metrics = json.loads(result.stdout)
            metrics["peak_rss_kib"] = int(rss.read_text()) if rss.exists() else None
            return metrics, directory
        # Warm-up executable/code paths without including this run in the medians.
        run(n=256, dim=3, rank=64, max_iter=4, fixed="yes", threads=1)
        raw = []
        for n in [512, 2048, 4096]:
            if n > args.max_n:
                continue
            baseline_t = None
            for estep, threads in [("dense", 1), ("streaming", 1), ("streaming", 4)]:
                for repeat in range(args.repeats):
                    m, directory = run(n=n, dim=3, amplitude=0.25, noise=0.015, rank=128,
                                       seed=42, max_iter=12, fixed="yes", estep=estep, threads=threads)
                    t = np.loadtxt(directory / "registered.csv", delimiter=",")
                    if baseline_t is None:
                        baseline_t = t
                    error = float(np.max(np.abs(t - baseline_t)))
                    if error > 1e-7:
                        raise AssertionError((n, estep, threads, "trajectory mismatch", error))
                    raw.append({"n": n, "estep": estep, "threads": threads, "repeat": repeat,
                                "max_abs_T_difference_from_dense": error, **m})
                    print("speed", n, estep, threads, repeat, m["fit_seconds"], flush=True)
        table(out / "speed_raw.csv", raw)
        medians = []
        for n in sorted(set(row["n"] for row in raw)):
            ref = statistics.median(row["fit_seconds"] for row in raw if row["n"] == n and row["estep"] == "dense")
            for estep, threads in [("dense", 1), ("streaming", 1), ("streaming", 4)]:
                group = [row for row in raw if row["n"] == n and row["estep"] == estep and row["threads"] == threads]
                times = [row["fit_seconds"] for row in group]
                med = statistics.median(times)
                medians.append({"n": n, "rank": 128, "iterations": 12, "estep": estep, "threads": threads,
                                "median_fit_seconds": med, "min_fit_seconds": min(times), "max_fit_seconds": max(times),
                                "speedup_vs_our_dense_E": ref / med,
                                "median_peak_rss_kib": statistics.median(row["peak_rss_kib"] for row in group),
                                "max_abs_T_difference_from_dense": max(row["max_abs_T_difference_from_dense"] for row in group)})
        table(out / "speed_summary.csv", medians)
        cases = [
            ("curve2d_clean", 2, 256, 0.4, 0.0, 0.0, 0.0),
            ("surface3d_mild", 3, 512, 0.15, 0.003, 0.0, 0.0),
            ("surface3d_noisy", 3, 512, 0.4, 0.015, 0.0, 0.0),
            ("surface3d_missing20", 3, 512, 0.4, 0.003, 0.2, 0.0),
            ("surface3d_large", 3, 512, 1.2, 0.003, 0.0, 0.0),
            ("surface3d_outliers10", 3, 512, 0.4, 0.003, 0.0, 0.1),
        ]
        quality = []
        for name, dim, n, amplitude, noise, missing, outliers in cases:
            for seed in [7, 42, 123]:
                for mode in ["paper", "official"]:
                    m, directory = run(n=n, dim=dim, amplitude=amplitude, noise=noise, missing=missing,
                                       outliers=outliers, seed=seed, mode=mode, threads=1)
                    quality.append({"case": name, "seed": seed, "amplitude": amplitude,
                                    "noise": noise, "missing": missing, "outliers": outliers, **m})
                    if seed == 42 and mode == "paper":
                        dest = out / "examples" / name
                        shutil.copytree(directory, dest)
                    print("quality", name, seed, mode, m["gt_rmse_after"], flush=True)
        table(out / "quality_raw.csv", quality)
        summary = []
        for name, *_ in cases:
            for mode in ["paper", "official"]:
                group = [row for row in quality if row["case"] == name and row["mode"] == mode]
                row = {"case": name, "mode": mode, "seeds": 3}
                for field in ["gt_rmse_before", "gt_rmse_prealigned", "gt_rmse_after", "nn_source_to_target_rmse", "nn_target_to_source_rmse", "fit_seconds"]:
                    values = [item[field] for item in group]
                    row["mean_" + field] = statistics.mean(values)
                    row["sd_" + field] = statistics.stdev(values)
                summary.append(row)
        table(out / "quality_summary.csv", summary)
        ablation = []
        for rank in [32, 64, 128, 308, 1024]:
            m, directory = run(n=1024, dim=3, amplitude=0.25, noise=0.003, rank=rank, threads=1, dump="yes")
            y = np.loadtxt(directory / "prepared_source.csv", delimiter=",")
            q = np.loadtxt(directory / "Q.csv", delimiter=",")
            k = np.exp(-2 * cdist(y, y, "cityblock"))
            approximation_error = float(np.linalg.norm(k - q @ q.T) / np.linalg.norm(k))
            ablation.append({"rank": rank, "relative_kernel_frobenius_error": approximation_error, **m})
            print("rank", rank, approximation_error, m["gt_rmse_after"], flush=True)
        table(out / "rank_ablation.csv", ablation)
    print("Wrote", out, flush=True)

if __name__ == "__main__":
    main()
