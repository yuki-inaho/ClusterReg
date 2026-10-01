#!/usr/bin/env python3
"""Compare saved CLI builds on identical, persisted Sinkhorn inputs.

Native timings cover fit(), including preparation and the final E-step; wall
timings also include CLI startup, I/O, and nearest-neighbour evaluation. Numeric
failures give a nonzero exit status. A timing regression is reported, not used
as a test failure. Every run and input remains in an adjacent artifact directory.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from typing import Any

import numpy as np

ATOL = 1e-9
RTOL = 1e-8
KKT_LIMIT = 1e-8
ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def executable_metadata(path: Path) -> dict[str, Any]:
    data: dict[str, Any] = {"path": str(path), "sha256": sha256(path)}
    cache = path.parent / "CMakeCache.txt"
    settings = {}
    if cache.is_file():
        for line in cache.read_text().splitlines():
            if not line or line.startswith(("#", "//")) or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.split(":", 1)[0]
            if key.startswith(("CLUSTERREG_", "CMAKE_CXX_FLAGS")) or key in {
                "CMAKE_BUILD_TYPE", "CMAKE_CXX_COMPILER", "CMAKE_GENERATOR",
                "CMAKE_INTERPROCEDURAL_OPTIMIZATION", "CMAKE_CUDA_ARCHITECTURES",
            }:
                settings[key] = value
        data["cmake_cache"] = str(cache)
        data["cmake_cache_sha256"] = sha256(cache)
    data["build_config"] = settings or None
    compiler = settings.get("CMAKE_CXX_COMPILER")
    data["compiler"] = None
    if compiler:
        try:
            result = subprocess.run([compiler, "--version"], capture_output=True,
                                    text=True, check=True, timeout=10)
            data["compiler"] = result.stdout.splitlines()[0]
        except (OSError, subprocess.SubprocessError, IndexError):
            data["compiler"] = "unavailable: " + compiler
    return data


def invoke(command: list[str], log_prefix: Path, timeout: float) -> tuple[dict, float]:
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout,
                            env={**os.environ, "OMP_DYNAMIC": "FALSE",
                                 "OPENBLAS_NUM_THREADS": "1"})
    wall = time.perf_counter() - started
    log_prefix.with_suffix(".stdout.txt").write_text(result.stdout)
    log_prefix.with_suffix(".stderr.txt").write_text(result.stderr)
    log_prefix.with_suffix(".command.json").write_text(
        json.dumps(command, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"CLI exit {result.returncode}: {command}; "
                           f"stderr={result.stderr.strip()}")
    return {"command": command, "returncode": result.returncode}, wall


def load_history(path: Path) -> list[dict[str, float | int]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return [{key: int(value) if key in {"iteration", "transport_iterations"}
             else float(value) for key, value in row.items()} for row in rows]


def check_payload(run: dict[str, Any]) -> list[str]:
    errors = []
    metrics, history = run["metrics"], run["history"]
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not math.isfinite(value):
            errors.append(f"metrics.{key} is not finite")
    if not np.isfinite(run["coordinates"]).all():
        errors.append("registered coordinates are not finite")
    if not np.isfinite(run["source_mass"]).all():
        errors.append("source masses are not finite")
    if list(run["coordinates"].shape) != run["input_source_shape"]:
        errors.append("registered coordinate shape differs from the saved source input")
    if list(run["source_mass"].shape) != [run["input_source_shape"][0]]:
        errors.append("source mass shape differs from the saved source input")
    if [metrics.get("source_count"), metrics.get("dimension")] != run["input_source_shape"]:
        errors.append("metrics source shape differs from the saved source input")
    if [metrics.get("target_count"), metrics.get("dimension")] != run["input_target_shape"]:
        errors.append("metrics target shape differs from the saved target input")
    mass = metrics.get("transport_mass", 0)
    if abs(float(np.sum(run["source_mass"])) - mass) > ATOL + RTOL * abs(mass):
        errors.append("sum(source_mass) differs from transport_mass")
    if np.any(run["source_mass"] < 0):
        errors.append("source masses must be nonnegative")
    if metrics.get("algorithm") != "sinkhorn" or metrics.get("backend") != "cpu":
        errors.append("unexpected algorithm/backend")
    if metrics.get("noise_model") != run["noise_model"]:
        errors.append("unexpected noise model")
    for key in ("sigma2", "transport_mass", "fit_seconds"):
        if not metrics.get(key, 0) > 0:
            errors.append(f"metrics.{key} must be positive")
    for key in ("preparation_seconds", "iteration_seconds", "estep_seconds", "solve_seconds"):
        if metrics.get(key, 0) < 0:
            errors.append(f"metrics.{key} must be nonnegative")
    return errors


def check_run(run: dict[str, Any], args: argparse.Namespace) -> list[str]:
    errors = check_payload(run)
    metrics, history = run["metrics"], run["history"]
    if metrics.get("iterations") != args.outer_iterations or len(history) != args.outer_iterations:
        errors.append("fixed outer iteration workload did not complete")
    if metrics.get("stop_reason") != "max_iterations":
        errors.append("unexpected early stop: " + str(metrics.get("stop_reason")))
    for index, row in enumerate(history, 1):
        if row.get("iteration") != index:
            errors.append(f"history index {index} is inconsistent")
        if row.get("transport_iterations") != args.sinkhorn_iterations:
            errors.append(f"history {index} fixed inner workload did not complete")
        for key, value in row.items():
            if not math.isfinite(value):
                errors.append(f"history {index}.{key} is not finite")
        before, after = row["objective_before"], row["objective_after"]
        if after - before > ATOL + RTOL * abs(before):
            errors.append(f"history {index} objective increases")
        if row["kkt_residual"] > KKT_LIMIT or row["kkt_residual"] < 0:
            errors.append(f"history {index} KKT residual exceeds {KKT_LIMIT}")
        if row["transport_mass"] <= 0 or row["robust_mass"] <= 0 or row["sigma_before"] <= 0 or row["sigma_after"] <= 0:
            errors.append(f"history {index} collapsed mass/scale")
        for key in ("transport_residual", "linear_residual", "estep_seconds", "solve_seconds",
                    "step_rms", "scale_relative_change"):
            if row[key] < 0:
                errors.append(f"history {index}.{key} must be nonnegative")
        if index > 1:
            previous = history[index - 2]["objective_after"]
            if before - previous > ATOL + RTOL * abs(previous):
                errors.append(f"history {index} E-step objective increases")
    return errors


def check_final_e_probe(run: dict[str, Any], args: argparse.Namespace) -> list[str]:
    """The (K+1)st E-step reproduces the unreported final E-step of fit(K).

    Only the first K M-steps need to descend. The additional M-step may reject
    its candidate after recording the valid E-step that this probe needs.
    """
    errors = check_payload(run)
    metrics, history = run["metrics"], run["history"]
    expected = args.outer_iterations + 1
    if len(history) != expected or metrics.get("iterations") != expected:
        errors.append("final E-step probe is missing")
    if metrics.get("stop_reason") not in {"max_iterations", "non_descent"}:
        errors.append("invalid final E-step probe stop reason")
    for index, row in enumerate(history, 1):
        if row.get("iteration") != index:
            errors.append(f"probe history {index} index is inconsistent")
        if row.get("transport_iterations") != args.sinkhorn_iterations:
            errors.append(f"probe history {index} fixed inner workload did not complete")
        if not all(math.isfinite(value) for value in row.values()):
            errors.append(f"probe history {index} has nonfinite diagnostics")
        if row["kkt_residual"] < 0 or row["kkt_residual"] > KKT_LIMIT:
            errors.append(f"probe history {index} KKT residual exceeds {KKT_LIMIT}")
        if row["transport_mass"] <= 0 or row["robust_mass"] <= 0 or row["sigma_before"] <= 0 or row["sigma_after"] <= 0:
            errors.append(f"probe history {index} collapsed mass/scale")
        for key in ("transport_residual", "linear_residual", "estep_seconds", "solve_seconds",
                    "step_rms", "scale_relative_change"):
            if row[key] < 0:
                errors.append(f"probe history {index}.{key} must be nonnegative")
        if index <= args.outer_iterations:
            before, after = row["objective_before"], row["objective_after"]
            if after - before > ATOL + RTOL * abs(before):
                errors.append(f"probe history {index} prefix M-step objective increases")
        if index > 1:
            previous = history[index - 2]["objective_after"]
            if row["objective_before"] - previous > ATOL + RTOL * abs(previous):
                errors.append(f"probe history {index} E-step objective increases")
    return errors


def close_error(actual: Any, reference: Any) -> dict[str, Any]:
    actual, reference = np.asarray(actual), np.asarray(reference)
    if actual.shape != reference.shape:
        return {"passed": False, "shape_actual": list(actual.shape),
                "shape_reference": list(reference.shape)}
    difference = np.abs(actual - reference)
    limit = ATOL + RTOL * np.abs(reference)
    return {"passed": bool(np.all(difference <= limit)),
            "max_abs_error": float(np.max(difference, initial=0)),
            "max_tolerance_ratio": float(np.max(difference / limit, initial=0))}


def compare_runs(actual: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    checks = {"coordinates": close_error(actual["coordinates"], reference["coordinates"]),
              "source_mass": close_error(actual["source_mass"], reference["source_mass"])}
    for key in ("sigma2", "transport_mass"):
        checks[key] = close_error(actual["metrics"][key], reference["metrics"][key])
    for key in ("iterations", "stop_reason", "retained_rank", "requested_rank",
                "source_count", "target_count", "dimension", "threads_requested"):
        checks[key] = {"passed": actual["metrics"][key] == reference["metrics"][key]}
    if len(actual["history"]) == len(reference["history"]):
        for key in ("iteration", "transport_iterations"):
            checks["history." + key] = {"passed": [row[key] for row in actual["history"]]
                                       == [row[key] for row in reference["history"]]}
        for key in ("sigma_before", "sigma_after", "objective_before", "objective_after",
                    "transport_mass", "robust_mass", "kkt_residual", "transport_residual"):
            checks["history." + key] = close_error(
                [row[key] for row in actual["history"]],
                [row[key] for row in reference["history"]])
    else:
        checks["history.length"] = {"passed": False}
    return {"passed": all(check["passed"] for check in checks.values()), "checks": checks}


def compare_final_e_probe(measured: dict[str, Any], probe: dict[str, Any],
                          reference_probe: dict[str, Any],
                          args: argparse.Namespace) -> dict[str, Any]:
    """Check that excluded warm-up work witnesses the measured final E-step."""
    count = args.outer_iterations
    checks: dict[str, Any] = {
        "probe_valid": {"passed": not probe["validation_errors"]},
        "reference_probe_valid": {"passed": not reference_probe["validation_errors"]},
        "prefix_length": {"passed": len(measured["history"]) == count
                          and len(probe["history"]) == count + 1},
    }
    diagnostics: dict[str, Any] = {
        "warmup_excluded_from_measurements": True,
        "probe_stop_reason": probe["metrics"]["stop_reason"],
        "estep_seconds_scope": "K+1 warm-up E-step reference timing; never added to measured native fit",
    }
    if checks["prefix_length"]["passed"]:
        for key in ("iteration", "transport_iterations"):
            checks["prefix." + key] = {"passed": [row[key] for row in measured["history"]]
                                       == [row[key] for row in probe["history"][:count]]}
        for key in ("sigma_before", "sigma_after", "objective_before", "objective_after",
                    "transport_mass", "robust_mass", "kkt_residual", "transport_residual",
                    "linear_residual", "step_rms", "scale_relative_change"):
            checks["prefix." + key] = close_error(
                [row[key] for row in probe["history"][:count]],
                [row[key] for row in measured["history"]])
        final_e = probe["history"][count]
        checks["final.scale"] = close_error(final_e["sigma_before"], measured["metrics"]["sigma2"])
        checks["final.mass"] = close_error(final_e["transport_mass"], measured["metrics"]["transport_mass"])
        checks["final.sweeps_requested"] = {"passed": final_e["transport_iterations"] == args.sinkhorn_iterations}
        checks["final.sweeps_baseline"] = {
            "passed": len(reference_probe["history"]) == count + 1
            and final_e["transport_iterations"] == reference_probe["history"][count]["transport_iterations"]}
        diagnostics.update({
            "final_e_probe_sweeps": final_e["transport_iterations"],
            "final_e_probe_residual": final_e["transport_residual"],
            "final_e_probe_kkt_residual": final_e["kkt_residual"],
            "final_e_probe_mass": final_e["transport_mass"],
            "reference_estep_seconds": final_e["estep_seconds"],
        })
    return {"passed": all(check["passed"] for check in checks.values()),
            "checks": checks, **diagnostics}


def public_run(run: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in run.items() if key not in {"coordinates", "source_mass"}}


def json_safe(value: Any) -> Any:
    """Keep failed nonfinite runs inspectable while writing strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


def registration_flags(args: argparse.Namespace, model: str) -> dict[str, Any]:
    return {"algorithm": "sinkhorn", "backend": "cpu", "noise-model": model,
            "rank": args.rank, "max-iter": args.outer_iterations,
            "sinkhorn-iter": args.sinkhorn_iterations, "threads": args.threads,
            "fixed": "yes", "tol": 0, "sinkhorn-tol": 0, "initial-sigma": 0.5,
            "sigma-floor": 1e-4, "sigma-ceiling": 4, "transport-entropy": 1,
            "tau-source": 3, "tau-target": 3, "regularization": 0.08,
            "student-dof": 4, "seed": 42, "normalize": "yes",
            "align-centroids": "yes", "solver": "nystrom", "estep": "streaming",
            "gamma": 2, "eigen-cutoff": 1e-6, "kmeans-iter": 5, "ratio": 0.3,
            "mass-floor": 1e-12, "entropy": 0.5, "mode": "paper"}


def fit_run(executable: Path, label: str, model: str, inputs: Path,
            out: Path, args: argparse.Namespace, *, final_e_probe: bool = False) -> dict[str, Any]:
    command = [str(executable), "register", "--source", str(inputs / "source.csv"),
               "--target", str(inputs / "target.csv"), "--out", str(out)]
    flags = registration_flags(args, model)
    if final_e_probe:
        flags["max-iter"] += 1
    for key, value in flags.items():
        command.extend(["--" + key, str(value)])
    process, wall = invoke(command, out.parent / out.name, args.timeout)
    run = {"label": label, "noise_model": model, "output_dir": str(out),
           "wall_seconds": wall, **process,
           "input_source_shape": list(np.loadtxt(inputs / "source.csv", delimiter=",", ndmin=2).shape),
           "input_target_shape": list(np.loadtxt(inputs / "target.csv", delimiter=",", ndmin=2).shape),
           "metrics": json.loads((out / "metrics.json").read_text()),
           "history": load_history(out / "history.csv"),
           "coordinates": np.loadtxt(out / "registered.csv", delimiter=",", ndmin=2),
           "source_mass": np.loadtxt(out / "alpha.csv", delimiter=",", ndmin=1)}
    run["validation_errors"] = check_final_e_probe(run, args) if final_e_probe else check_run(run, args)
    return run


def summarize(runs: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    native = [run["metrics"]["fit_seconds"] for run in runs]
    wall = [run["wall_seconds"] for run in runs]
    reference_median = statistics.median(run["metrics"]["fit_seconds"] for run in baseline)
    median = statistics.median(native)
    timing_valid = all(math.isfinite(value) and value > 0 for value in native) and all(
        math.isfinite(run["metrics"]["fit_seconds"]) and run["metrics"]["fit_seconds"] > 0
        for run in baseline)
    slowdown = median / reference_median - 1 if timing_valid else None
    has_iteration_time = all("iteration_seconds" in run["metrics"] for run in runs)
    return {"label": runs[0]["label"], "native_raw_seconds": native,
            "native_min_seconds": min(native), "native_median_seconds": median,
            "native_max_seconds": max(native), "wall_raw_seconds": wall,
            "wall_median_seconds": statistics.median(wall),
            "speedup_vs_baseline": reference_median / median if timing_valid else None,
            "slowdown_fraction": slowdown,
            "performance_verdict": "invalid_timing" if not timing_valid
            else "regression_confirmed" if slowdown > 0.1 and len(runs) >= 5
            else "needs_5_repeat_confirmation" if slowdown > 0.1 else "no_10_percent_regression",
            "median_preparation_seconds": statistics.median(
                run["metrics"]["preparation_seconds"] for run in runs),
            "median_estep_seconds": statistics.median(run["metrics"]["estep_seconds"] for run in runs),
            "median_solve_seconds": statistics.median(run["metrics"]["solve_seconds"] for run in runs),
            "median_fixed_cost_final_estep_and_overhead_seconds": statistics.median(
                run["metrics"]["fit_seconds"] - run["metrics"]["preparation_seconds"]
                - run["metrics"]["estep_seconds"] - run["metrics"]["solve_seconds"] for run in runs),
            "median_fixed_cost_and_loop_overhead_seconds": statistics.median(
                run["metrics"]["iteration_seconds"] - run["metrics"]["estep_seconds"]
                - run["metrics"]["solve_seconds"] for run in runs) if has_iteration_time else None,
            "median_final_estep_and_return_overhead_seconds": statistics.median(
                run["metrics"]["fit_seconds"] - run["metrics"]["preparation_seconds"]
                - run["metrics"]["iteration_seconds"] for run in runs) if has_iteration_time else None,
            "iterations": runs[0]["metrics"]["iterations"],
            "inner_sweeps_total": sum(row["transport_iterations"] for row in runs[0]["history"]),
            "stop_reason": runs[0]["metrics"]["stop_reason"],
            "max_kkt_residual": max(row["kkt_residual"] for run in runs for row in run["history"]),
            "final_e_probe_sweeps": runs[0].get("final_e_probe", {}).get("final_e_probe_sweeps"),
            "final_e_probe_residual": runs[0].get("final_e_probe", {}).get("final_e_probe_residual"),
            "final_e_probe_kkt_residual": runs[0].get("final_e_probe", {}).get("final_e_probe_kkt_residual"),
            "final_e_probe_stop_reason": runs[0].get("final_e_probe", {}).get("probe_stop_reason"),
            "final_e_probe_reference_estep_seconds": runs[0].get("final_e_probe", {}).get("reference_estep_seconds"),
            "validation_passed": timing_valid and all(not run["validation_errors"] and run["comparison"]["passed"]
                                     and run.get("final_e_probe", {"passed": True})["passed"]
                                     for run in runs)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", nargs="+", required=True, metavar="[LABEL=]EXE",
                        help="one or more candidate executables; labels must be unique")
    parser.add_argument("--sizes", type=int, nargs="+", default=[256, 1024, 2048])
    parser.add_argument("--dimension", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--outer-iterations", type=int, default=3)
    parser.add_argument("--sinkhorn-iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--student", action="store_true", help="also compare Student-t")
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in [*args.sizes, args.dimension, args.rank,
                                    args.outer_iterations, args.sinkhorn_iterations,
                                    args.threads, args.repeats, args.timeout]):
        parser.error("sizes, dimensions, rank, iterations, threads, repeats, and timeout must be positive")
    if args.dimension not in {2, 3}:
        parser.error("synthetic CLI inputs require dimension 2 or 3")
    if len(set(args.sizes)) != len(args.sizes):
        parser.error("sizes must be unique")
    executables = {"baseline": args.baseline.resolve()}
    for index, specification in enumerate(args.candidate, 1):
        label, separator, path = specification.partition("=")
        if not separator:
            label, path = f"candidate{index}", specification
        if not label or label in executables or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for char in label):
            parser.error("candidate labels must be unique ASCII letters/numbers/underscore/hyphen")
        executables[label] = Path(path).resolve()
    for path in executables.values():
        if not path.is_file() or not os.access(path, os.X_OK):
            parser.error("executable is missing or not executable: " + str(path))
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts = Path(tempfile.mkdtemp(prefix=output.stem + "_artifacts_", dir=output.parent))
    report: dict[str, Any] = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor(), "python": platform.python_version(),
        "numpy": np.__version__, "cpu_count_os": os.cpu_count(),
        "artifacts_dir": str(artifacts),
        "timing_scope": "native fit includes preparation/final E-step; CLI wall includes I/O/evaluation",
        "measurement_order": "baseline/candidates in forward order on even repeats, reverse on odd repeats",
        "final_e_workload_gate": "Excluded K+1 warm-up probes history[K], reproducing fit(K)'s final E-step; every measured prefix is checked, and final sweeps must match requested workload and baseline.",
        "tolerances": {"absolute": ATOL, "relative": RTOL, "kkt_max": KKT_LIMIT},
        "performance_policy": "timing alone never fails exit status; >10% slowdown requires 5 repeats",
        "executables": {label: executable_metadata(path) for label, path in executables.items()},
        "settings": {key: value for key, value in vars(args).items() if key not in {"baseline", "candidate", "output"}},
        "cases": [], "errors": [], "csv": str(output.with_suffix(".csv")),
    }
    for path, key in [(Path("/proc/cpuinfo"), "cpu_model"),
                      (Path("/sys/fs/cgroup/cpu.max"), "cgroup_cpu_max")]:
        if path.is_file():
            lines = path.read_text().splitlines()
            report[key] = next((line.split(":", 1)[1].strip() for line in lines
                                if line.startswith("model name")), None) if key == "cpu_model" else "\n".join(lines)
    csv_rows = []
    try:
        for size in args.sizes:
            case_dir = artifacts / f"n{size}"
            case_dir.mkdir()
            inputs = case_dir / "inputs"
            command = [str(executables["baseline"]), "synthetic", "--out", str(inputs),
                       "--n", str(size), "--dim", str(args.dimension), "--amplitude", "0.35",
                       "--noise", "0.005", "--missing", "0.1", "--outliers", "0.1", "--seed", "42"]
            invoke(command, case_dir / "synthetic", args.timeout)
            input_files = {name: {"path": str(inputs / name), "sha256": sha256(inputs / name)}
                           for name in ("source.csv", "target.csv", "truth.csv")}
            for model in ["gaussian", "student-t"] if args.student else ["gaussian"]:
                model_dir = case_dir / model
                model_dir.mkdir()
                case: dict[str, Any] = {"size_requested": size, "noise_model": model,
                                        "inputs": input_files, "registration_flags": registration_flags(args, model),
                                        "runs": [], "summaries": []}
                report["cases"].append(case)
                probes = {}
                for label, executable in executables.items():
                    warmup = fit_run(executable, label, model, inputs, model_dir / f"{label}_warmup", args,
                                     final_e_probe=True)
                    probes[label] = warmup
                    if warmup["validation_errors"]:
                        report["errors"].extend(f"n{size}/{model}/{label}/warmup: {error}" for error in warmup["validation_errors"])
                    case.setdefault("warmups", []).append(public_run(warmup))
                groups: dict[str, list[dict[str, Any]]] = {label: [] for label in executables}
                for repeat in range(args.repeats):
                    labels = list(executables)
                    if repeat % 2:
                        labels.reverse()
                    for label in labels:
                        run = fit_run(executables[label], label, model, inputs,
                                      model_dir / f"{label}_{repeat}", args)
                        run["repeat"] = repeat
                        groups[label].append(run)
                    reference = groups["baseline"][-1]
                    for label in executables:
                        run = groups[label][-1]
                        run["comparison"] = compare_runs(run, reference)
                        # Repeated baseline executions must also be numerically stable.
                        if label == "baseline":
                            run["comparison"] = compare_runs(run, groups["baseline"][0])
                        run["final_e_probe"] = compare_final_e_probe(run, probes[label], probes["baseline"], args)
                        if run["validation_errors"] or not run["comparison"]["passed"] or not run["final_e_probe"]["passed"]:
                            report["errors"].append(f"n{size}/{model}/{label}/repeat{repeat}: numeric/workload validation failed")
                        case["runs"].append(public_run(run))
                for label in executables:
                    summary = summarize(groups[label], groups["baseline"])
                    case["summaries"].append(summary)
                    csv_rows.append({"size_requested": size, "noise_model": model,
                                     **{key: value for key, value in summary.items() if not isinstance(value, list)}})
                    speedup = f"{summary['speedup_vs_baseline']:.3f}" if summary["speedup_vs_baseline"] is not None else "unavailable"
                    print(f"n={size} {model} {label}: {summary['native_median_seconds']:.6f}s "
                          f"speedup={speedup} "
                          f"numeric={'PASS' if summary['validation_passed'] else 'FAIL'} "
                          f"performance={summary['performance_verdict']}", flush=True)
    except (OSError, subprocess.SubprocessError, ValueError, RuntimeError, KeyError, ArithmeticError) as error:
        report["errors"].append(str(error))
    report["validation_passed"] = not report["errors"]
    report["performance_remeasurement_needed"] = any(
        row["performance_verdict"] == "needs_5_repeat_confirmation" for row in csv_rows)
    if report["performance_remeasurement_needed"]:
        report["performance_next_step"] = "Rerun this command with --repeats 5 and a new --output; compare alternating-order medians before accepting a regression."
    output.write_text(json.dumps(json_safe(report), indent=2, allow_nan=False) + "\n")
    with output.with_suffix(".csv").open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in csv_rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"report: {output}; artifacts: {artifacts}", flush=True)
    if report["errors"]:
        for error in report["errors"]:
            print(error, file=sys.stderr)
    return 0 if report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
