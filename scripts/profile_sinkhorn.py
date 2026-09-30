#!/usr/bin/env python3
"""Correctness-gated CPU/CUDA profiling for UOT-CluReg.

The script reports measurements; it never treats a particular speedup as a
test requirement.  CUDA calls are synchronized by the native backend before
``fit`` returns because diagnostics and sufficient statistics are copied to
the host on every outer iteration.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any

import numpy as np

import clusterreg


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def fit_once(
    source: np.ndarray,
    target: np.ndarray,
    *,
    backend: str,
    noise_model: str,
    rank: int,
    outer_iterations: int,
    sinkhorn_iterations: int,
    threads: int,
) -> tuple[clusterreg.RegistrationResult, float]:
    start = time.perf_counter()
    result = clusterreg.fit(
        source,
        target,
        algorithm="sinkhorn",
        noise_model=noise_model,
        backend=backend,
        rank=rank,
        max_iterations=outer_iterations,
        fixed_iterations=True,
        sinkhorn_iterations=sinkhorn_iterations,
        sinkhorn_tolerance=0.0,
        initial_sigma=0.5,
        sigma_floor=1e-4,
        sigma_ceiling=4.0,
        transport_entropy=1.0,
        source_mass_penalty=3.0,
        target_mass_penalty=3.0,
        regularization=0.08,
        threads=threads,
        seed=42,
    )
    return result, time.perf_counter() - start


def profile_case(
    count: int,
    backend: str,
    noise_model: str,
    args: argparse.Namespace,
    cpu_reference: clusterreg.RegistrationResult | None,
) -> tuple[dict[str, Any], clusterreg.RegistrationResult]:
    data = clusterreg.make_synthetic(
        count=count,
        dimension=args.dimension,
        amplitude=0.35,
        noise=0.005,
        missing_fraction=args.missing,
        outlier_fraction=args.outliers,
        seed=42,
    )
    rank = min(count, args.rank)
    # Warm-up is excluded, including first CUDA context initialization.
    fit_once(
        data.source,
        data.target,
        backend=backend,
        noise_model=noise_model,
        rank=rank,
        outer_iterations=1,
        sinkhorn_iterations=min(10, args.sinkhorn_iterations),
        threads=args.threads,
    )
    wall_times: list[float] = []
    native_times: list[float] = []
    final: clusterreg.RegistrationResult | None = None
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for _ in range(args.repeats):
        final, wall = fit_once(
            data.source,
            data.target,
            backend=backend,
            noise_model=noise_model,
            rank=rank,
            outer_iterations=args.outer_iterations,
            sinkhorn_iterations=args.sinkhorn_iterations,
            threads=args.threads,
        )
        wall_times.append(wall)
        native_times.append(final.total_seconds)
    assert final is not None
    rss_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    estep = sum(item.estep_seconds for item in final.history)
    solve = sum(item.solve_seconds for item in final.history)
    row: dict[str, Any] = {
        "count_source": int(data.source.shape[0]),
        "count_target": int(data.target.shape[0]),
        "dimension": args.dimension,
        "rank": rank,
        "backend": backend,
        "noise_model": noise_model,
        "outer_iterations": args.outer_iterations,
        "sinkhorn_iterations_requested": args.sinkhorn_iterations,
        "sinkhorn_sweeps_total": sum(
            item.transport_iterations for item in final.history
        ),
        "wall_median_seconds": statistics.median(wall_times),
        "wall_p95_seconds": percentile(wall_times, 0.95),
        "native_median_seconds": statistics.median(native_times),
        "estep_last_seconds": estep,
        "solve_last_seconds": solve,
        "final_estep_and_overhead_seconds": max(0.0, final.total_seconds - estep - solve),
        "max_rss_before_kib": rss_before,
        "max_rss_after_kib": rss_after,
        "transport_mass": final.transport_mass,
        "sigma": final.sigma2,
        "max_kkt_residual": max(item.kkt_residual for item in final.history),
        "max_dual_residual": max(item.transport_residual for item in final.history),
    }
    if cpu_reference is not None:
        denominator = max(1.0, float(np.linalg.norm(cpu_reference.transformed)))
        row["cpu_relative_transform_error"] = float(
            np.linalg.norm(final.transformed - cpu_reference.transformed) / denominator
        )
        row["cpu_relative_mass_error"] = abs(
            final.transport_mass - cpu_reference.transport_mass
        ) / max(1.0, abs(cpu_reference.transport_mass))
    return row, final


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512])
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=min(4, clusterreg.available_threads()))
    parser.add_argument("--missing", type=float, default=0.1)
    parser.add_argument("--outliers", type=float, default=0.1)
    parser.add_argument("--student", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("build/results/sinkhorn_profile.json"))
    args = parser.parse_args()
    if args.require_cuda and not clusterreg.cuda_available():
        raise SystemExit("CUDA was required but is not available in this build/runtime")

    backends = ["cpu"] + (["cuda"] if clusterreg.cuda_available() else [])
    noises = ["gaussian"] + (["student-t"] if args.student else [])
    rows: list[dict[str, Any]] = []
    for noise_model in noises:
        for count in args.sizes:
            references: dict[str, clusterreg.RegistrationResult] = {}
            for backend in backends:
                reference = references.get("cpu") if backend == "cuda" else None
                row, result = profile_case(
                    count, backend, noise_model, args, reference
                )
                rows.append(row)
                if backend == "cpu":
                    references["cpu"] = result

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "clusterreg": clusterreg.__version__,
        "openmp_threads_available": clusterreg.available_threads(),
        "cuda_compiled": clusterreg.cuda_compiled(),
        "cuda_available": clusterreg.cuda_available(),
        "cuda_device": clusterreg.cuda_device_name(),
        "note": "All-pairs exact UOT; timing thresholds are not correctness gates.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": rows}, indent=2) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"metadata": metadata, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
