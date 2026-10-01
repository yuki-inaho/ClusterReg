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
import hashlib
import json
import math
from pathlib import Path
import platform
import resource
import statistics
import time
from typing import Any

import numpy as np

import clusterreg

ATOL = 1e-9
RTOL = 1e-8
KKT_LIMIT = 1e-8
FIT_CONFIG = {
    "algorithm": "sinkhorn", "fixed_iterations": True, "tolerance": 0.0,
    "sinkhorn_tolerance": 0.0, "initial_sigma": 0.5, "sigma_floor": 1e-4,
    "sigma_ceiling": 4.0, "transport_entropy": 1.0,
    "source_mass_penalty": 3.0, "target_mass_penalty": 3.0,
    "regularization": 0.08, "student_dof": 4.0, "seed": 42,
}
HISTORY_FIELDS = (
    "index", "sigma_before", "sigma_after", "objective_before", "objective_after",
    "transport_mass", "robust_mass", "transport_residual", "kkt_residual",
    "linear_residual", "step_rms", "scale_relative_change", "estep_seconds",
    "solve_seconds", "transport_iterations",
)


def validate_result(result: Any, *, outer_iterations: int,
                    sinkhorn_iterations: int, source_shape: tuple[int, int],
                    target_shape: tuple[int, int]) -> None:
    """Reject incomplete or invalid measured work before collecting timings."""
    errors = []
    for name, expected_shape in (("transformed", source_shape),
                                 ("source_mass", (source_shape[0],)),
                                 ("target_mass", (target_shape[0],))):
        values = np.asarray(getattr(result, name))
        if values.shape != expected_shape or not np.isfinite(values).all():
            errors.append(f"{name} has wrong shape or nonfinite values")
        if name.endswith("mass") and np.any(values < 0):
            errors.append(f"{name} has negative values")
    for name in ("sigma2", "transport_mass", "total_seconds", "preparation_seconds",
                 "iteration_seconds"):
        value = getattr(result, name)
        if not math.isfinite(value) or value < 0:
            errors.append(f"{name} must be finite and nonnegative")
        elif name in {"sigma2", "transport_mass", "total_seconds"} and value == 0:
            errors.append(f"{name} must be positive")
    for name in ("source_mass", "target_mass"):
        if not np.isclose(np.sum(getattr(result, name)), result.transport_mass,
                          atol=ATOL, rtol=RTOL):
            errors.append(f"sum({name}) differs from transport_mass")
    if len(result.history) != outer_iterations or result.stop_reason != "max_iterations":
        errors.append("fixed outer workload did not complete: " + result.stop_reason)
    previous = None
    for index, item in enumerate(result.history, 1):
        values = {name: getattr(item, name) for name in HISTORY_FIELDS}
        if not all(math.isfinite(value) for value in values.values()):
            errors.append(f"history {index} has nonfinite diagnostics")
        if item.index != index or item.transport_iterations != sinkhorn_iterations:
            errors.append(f"history {index} fixed inner workload did not complete")
        if item.kkt_residual < 0 or item.kkt_residual > KKT_LIMIT:
            errors.append(f"history {index} KKT residual exceeds {KKT_LIMIT}")
        if item.transport_residual < 0 or item.linear_residual < 0:
            errors.append(f"history {index} has negative residuals")
        if item.transport_mass <= 0 or item.robust_mass <= 0 or item.sigma_after <= 0:
            errors.append(f"history {index} collapsed mass/scale")
        if item.estep_seconds < 0 or item.solve_seconds < 0:
            errors.append(f"history {index} has negative timing")
        if item.objective_after - item.objective_before > ATOL + RTOL * abs(item.objective_before):
            errors.append(f"history {index} M-step objective increases")
        if previous is not None and item.objective_before - previous > ATOL + RTOL * abs(previous):
            errors.append(f"history {index} E-step objective increases")
        previous = item.objective_after
    if errors:
        raise ValueError("; ".join(errors))


def validate_parity(actual: Any, reference: Any) -> dict[str, float]:
    """Use the same elementwise tolerance for repeats and CPU/CUDA parity."""
    errors = {}
    for name in ("transformed", "source_mass", "target_mass", "sigma2", "transport_mass"):
        candidate = np.asarray(getattr(actual, name))
        expected = np.asarray(getattr(reference, name))
        if candidate.shape != expected.shape or not np.allclose(candidate, expected, atol=ATOL, rtol=RTOL):
            raise ValueError(f"parity mismatch: {name}")
        errors["max_abs_" + name + "_error"] = float(np.max(np.abs(candidate - expected), initial=0))
    if actual.stop_reason != reference.stop_reason or len(actual.history) != len(reference.history):
        raise ValueError("parity mismatch: stopping/workload")
    if [item.transport_iterations for item in actual.history] != [item.transport_iterations for item in reference.history]:
        raise ValueError("parity mismatch: inner workload")
    for name in ("objective_before", "objective_after", "sigma_after", "transport_mass", "robust_mass"):
        if not np.allclose([getattr(item, name) for item in actual.history],
                           [getattr(item, name) for item in reference.history], atol=ATOL, rtol=RTOL):
            raise ValueError("parity mismatch: history." + name)
    return errors


def input_hash(values: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(values, dtype="<f8").tobytes()).hexdigest()


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
        **FIT_CONFIG,
        noise_model=noise_model,
        backend=backend,
        rank=rank,
        max_iterations=outer_iterations,
        sinkhorn_iterations=sinkhorn_iterations,
        threads=threads,
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
    raw_runs: list[dict[str, Any]] = []
    first = None
    final: clusterreg.RegistrationResult | None = None
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    for repeat in range(args.repeats):
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
        validate_result(final, outer_iterations=args.outer_iterations,
                        sinkhorn_iterations=args.sinkhorn_iterations,
                        source_shape=data.source.shape, target_shape=data.target.shape)
        if not math.isfinite(wall) or wall <= 0:
            raise ValueError("measured wall time must be finite and positive")
        if first is None:
            first = final
        repeat_parity = validate_parity(final, first)
        cpu_parity = validate_parity(final, cpu_reference) if cpu_reference is not None else None
        wall_times.append(wall)
        native_times.append(final.total_seconds)
        estep = sum(item.estep_seconds for item in final.history)
        solve = sum(item.solve_seconds for item in final.history)
        raw_runs.append({
            "repeat": repeat, "wall_seconds": wall, "native_seconds": final.total_seconds,
            "preparation_seconds": final.preparation_seconds,
            "iteration_seconds": final.iteration_seconds,
            "history_estep_seconds": estep, "history_solve_seconds": solve,
            "fixed_cost_and_loop_overhead_seconds": final.iteration_seconds - estep - solve,
            "final_estep_and_return_overhead_seconds": final.total_seconds - final.preparation_seconds - final.iteration_seconds,
            "actual_outer_iterations": len(final.history), "stop_reason": final.stop_reason,
            "history": [{name: getattr(item, name) for name in HISTORY_FIELDS} for item in final.history],
            "transport_mass": final.transport_mass, "sigma2": final.sigma2,
            "repeat_parity": repeat_parity, "cpu_parity": cpu_parity,
        })
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
        "actual_outer_iterations": len(final.history),
        "stop_reason": final.stop_reason,
        "input_sha256": {"source": input_hash(data.source), "target": input_hash(data.target)},
        "input_hash_format": "row-major little-endian float64 bytes; shapes recorded separately",
        "fit_config": {**FIT_CONFIG, "rank": rank, "max_iterations": args.outer_iterations,
                       "sinkhorn_iterations": args.sinkhorn_iterations, "threads": args.threads,
                       "backend": backend, "noise_model": noise_model},
        "wall_raw_seconds": wall_times,
        "native_raw_seconds": native_times,
        "raw_runs": raw_runs,
        "sinkhorn_sweeps_total": sum(
            item.transport_iterations for item in final.history
        ),
        "wall_median_seconds": statistics.median(wall_times),
        "wall_p95_seconds": percentile(wall_times, 0.95),
        "native_median_seconds": statistics.median(native_times),
        "native_min_seconds": min(native_times),
        "native_max_seconds": max(native_times),
        "preparation_median_seconds": statistics.median(run["preparation_seconds"] for run in raw_runs),
        "iteration_median_seconds": statistics.median(run["iteration_seconds"] for run in raw_runs),
        "fixed_cost_and_loop_overhead_median_seconds": statistics.median(run["fixed_cost_and_loop_overhead_seconds"] for run in raw_runs),
        "final_estep_and_return_overhead_median_seconds": statistics.median(run["final_estep_and_return_overhead_seconds"] for run in raw_runs),
        "estep_last_seconds": estep,
        "solve_last_seconds": solve,
        "final_estep_and_overhead_seconds": final.total_seconds - final.preparation_seconds - final.iteration_seconds,
        "max_rss_before_kib": rss_before,
        "max_rss_after_kib": rss_after,
        "transport_mass": final.transport_mass,
        "sigma": final.sigma2,
        "max_kkt_residual": max(item["kkt_residual"] for run in raw_runs for item in run["history"]),
        "max_dual_residual": max(item["transport_residual"] for run in raw_runs for item in run["history"]),
        "validation_passed": True,
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
    if min(*args.sizes, args.dimension, args.rank, args.outer_iterations,
           args.sinkhorn_iterations, args.repeats, args.threads) <= 0:
        parser.error("sizes, dimension, rank, iterations, repeats, and threads must be positive")
    if args.dimension not in {2, 3}:
        parser.error("synthetic profiling supports dimension 2 or 3")
    if not math.isfinite(args.missing) or not math.isfinite(args.outliers) or not 0 <= args.missing < 1 or args.outliers < 0:
        parser.error("missing must be in [0,1) and outliers must be nonnegative")
    if args.require_cuda and not clusterreg.cuda_available():
        raise SystemExit("CUDA was required but is not available in this build/runtime")

    backends = ["cpu"] + (["cuda"] if clusterreg.cuda_available() else [])
    noises = ["gaussian"] + (["student-t"] if args.student else [])
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    for noise_model in noises:
        for count in args.sizes:
            references: dict[str, clusterreg.RegistrationResult] = {}
            for backend in backends:
                reference = references.get("cpu") if backend == "cuda" else None
                try:
                    row, result = profile_case(count, backend, noise_model, args, reference)
                    rows.append(row)
                    if backend == "cpu":
                        references["cpu"] = result
                except (ValueError, RuntimeError) as error:
                    message = f"n={count}/{noise_model}/{backend}: {error}"
                    errors.append(message)
                    rows.append({"count_requested": count, "backend": backend,
                                 "noise_model": noise_model, "validation_passed": False,
                                 "error": message})
                    # A CUDA result without a correct CPU reference cannot pass parity.
                    if backend == "cpu":
                        break

    metadata = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "clusterreg": clusterreg.__version__,
        "openmp_threads_available": clusterreg.available_threads(),
        "cuda_compiled": clusterreg.cuda_compiled(),
        "cuda_available": clusterreg.cuda_available(),
        "cuda_device": clusterreg.cuda_device_name(),
        "note": "All-pairs exact UOT; timing thresholds are not correctness gates.",
        "seed": 42, "threads_requested": args.threads,
        "settings": {key: value for key, value in vars(args).items() if key != "output"},
        "fit_defaults": FIT_CONFIG,
        "tolerances": {"absolute": ATOL, "relative": RTOL, "kkt_max": KKT_LIMIT},
        "stage_scope": "prep/iteration are native timers; history E-step/solve exclude final E-step. iteration minus history stages estimates fixed-cost+loop overhead; total minus prep/iteration estimates final E-step+return overhead, not independent timers.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"metadata": metadata, "results": rows, "errors": errors,
                    "validation_passed": not errors}, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    csv_path = args.output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows({key: json.dumps(value) if isinstance(value, (list, dict)) else value
                          for key, value in row.items()} for row in rows)
    print(f"Saved {args.output}: {len(rows)} cases, validation={'PASS' if not errors else 'FAIL'}")
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
