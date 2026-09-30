#!/usr/bin/env python3
"""Independent NumPy oracle + execution of the hash-verified official Python code."""
from __future__ import annotations
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import importlib.util
import inspect
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import numpy as np
from scipy.spatial.distance import cdist

ROOT = Path(__file__).resolve().parents[1]

def load(path: Path) -> np.ndarray:
    return np.loadtxt(path, delimiter=",", ndmin=2)

def run_cli(exe: Path, out: Path, *args: str) -> dict:
    proc = subprocess.run([str(exe), "demo", "--out", str(out), *args],
                          text=True, capture_output=True, timeout=120, check=True)
    return json.loads(proc.stdout)

def paper_oracle(y: np.ndarray, x: np.ndarray, steps: int) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Full, dense Eq.(3)/(4)/(9) oracle; intentionally no reduced solve or streaming."""
    n, d = y.shape
    m = x.shape[0]
    k = np.exp(-2 * cdist(y, y, "cityblock"))
    t = y.copy()
    alpha = np.ones(n) / n
    sigma = np.mean(cdist(x, y, "sqeuclidean")) / d
    history = []
    for _ in range(steps):
        log_alpha = np.full_like(alpha, -np.inf)
        np.log(alpha, out=log_alpha, where=alpha > 0)
        logits = log_alpha[None, :] - cdist(x, t, "sqeuclidean") / (0.5 * sigma)
        logits -= logits.max(axis=1, keepdims=True)
        u = np.exp(logits)
        u /= u.sum(axis=1, keepdims=True)
        mass = u.sum(axis=0)
        alpha = mass / m
        g = u.T @ x - mass[:, None] * y
        c = np.linalg.solve(mass[:, None] * k + 0.1 * sigma * np.eye(n), g)
        t = y + k @ c
        # Direct pairwise sum at NEW T; no trace-sign/transposition ambiguity.
        sigma = max(float(np.sum(u * cdist(x, t, "sqeuclidean"))) / (m * d), 1e-8)
        history.append(sigma)
    return t, alpha, history

def official_module():
    file = ROOT / "tests/upstream/fuzzyclusterreg.py"
    blob = file.read_bytes()
    digest = hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest()
    if digest != "0df3a88bb5d855674f3617fb04f2664289f9c2d6":
        raise RuntimeError("Upstream reference was modified: " + digest)
    spec = importlib.util.spec_from_file_location("official_clusterreg", file)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, digest

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exe", type=Path, default=ROOT / "build/pixi/clusterreg_cli")
    p.add_argument("--output", type=Path, default=ROOT / "build/pixi/validation.json")
    args = p.parse_args()
    exe = args.exe.resolve()
    tests = exe.parent / "clusterreg_tests"
    core = subprocess.run([str(tests)], capture_output=True, text=True, check=True, timeout=120)
    passed = sum(line.startswith("PASS ") for line in core.stdout.splitlines())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    (args.output.parent / "core_tests.txt").write_text(core.stdout)
    module, blob_sha = official_module()
    results: dict = {"upstream_commit": "47b6ce98fb53534871da4e4bead739ac7d170091",
                    "upstream_python_git_blob": blob_sha, "core_tests_passed": passed,
                    "official_same_landmarks": [], "paper_dense_oracle": []}
    with tempfile.TemporaryDirectory(prefix="clusterreg_validate_") as tmp:
        base = Path(tmp)
        for dim, n, missing in [(2, 79, 0.0), (3, 83, 0.23), (3, 96, 0.0)]:
            directory = base / f"official_{dim}_{n}"
            metrics = run_cli(exe, directory, "--n", str(n), "--dim", str(dim),
                              "--missing", str(missing), "--noise", "0.015", "--mode", "official",
                              "--max-iter", "16", "--fixed", "yes", "--dump", "yes")
            y, x = load(directory / "prepared_source.csv"), load(directory / "prepared_target.csv")
            centers, q = load(directory / "landmarks.csv"), load(directory / "Q.csv")
            module._kmeans = lambda data, m, iterations, centers=centers: centers.copy()
            q_official = module.INys(module.KernelType.RBF, 0.5, y, centers.shape[0], module.Strategy.KMEANS)
            gram_error = float(np.linalg.norm(q @ q.T - q_official @ q_official.T) / np.linalg.norm(q @ q.T))
            # Observe the actual upstream function without rewriting its loop.
            lines, first = inspect.getsourcelines(module.fuzzy_cluster_reg)
            capture_line = first + next(i for i, line in enumerate(lines) if 'logging.debug(f"{iter=:' in line)
            captured: list[dict] = []
            def trace(frame, event, arg):
                if frame.f_code is module.fuzzy_cluster_reg.__code__ and event == "line" and frame.f_lineno == capture_line:
                    v = frame.f_locals
                    captured.append({"iteration": int(v["iter"]), "sigma": float(v["sigma2"]),
                                     "loss": float(v["Loss"]), "T": v["T"].copy()})
                return trace
            old_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                alpha, t = module.fuzzy_cluster_reg(y.copy(), x.copy(), max_iter=16, tol=-1.0)
            finally:
                sys.settrace(old_trace)
            ch = np.genfromtxt(directory / "history.csv", delimiter=",", names=True)
            assert len(captured) == metrics["iterations"] == 16, (dim, n, len(captured), metrics["iterations"], capture_line, captured[-1] if captured else None)
            t_error = float(np.max(np.abs(t - load(directory / "normalized_registered.csv"))))
            alpha_error = float(np.max(np.abs(alpha.ravel() - load(directory / "alpha.csv").ravel())))
            sigma_error = float(np.max(np.abs(np.array([v["sigma"] for v in captured]) - ch["sigma_after"])))
            loss_error = float(np.max(np.abs(np.array([v["loss"] for v in captured]) - ch["official_loss"])))
            # Replay deterministic C++ prefixes to verify T at EVERY iteration.
            trajectory_error = t_error
            for step in range(1, 16):
                prefix = base / f"official_prefix_{dim}_{n}_{step}"
                run_cli(exe, prefix, "--n", str(n), "--dim", str(dim),
                        "--missing", str(missing), "--noise", "0.015", "--mode", "official",
                        "--max-iter", str(step), "--fixed", "yes", "--dump", "yes")
                trajectory_error = max(trajectory_error, float(np.max(np.abs(
                    captured[step - 1]["T"] - load(prefix / "normalized_registered.csv")))))
            assert trajectory_error < 1e-7
            entry = {"dimension": dim, "source_count": n, "target_count": len(x), "iterations": 16,
                     "kernel_relative_frobenius_error": gram_error, "max_abs_T_error": t_error,
                     "max_abs_T_error_all_iterations": trajectory_error,
                     "max_abs_alpha_error": alpha_error, "max_abs_sigma_error_all_iterations": sigma_error,
                     "max_abs_logged_loss_error_all_iterations": loss_error}
            assert gram_error < 1e-11 and t_error < 1e-7 and alpha_error < 1e-9 and sigma_error < 1e-9 and loss_error < 1e-5, entry
            results["official_same_landmarks"].append(entry)
        for dim, n, missing in [(2, 47, 0), (3, 53, 0.17)]:
            directory = base / f"paper_{dim}"
            run_cli(exe, directory, "--n", str(n), "--dim", str(dim), "--missing", str(missing),
                    "--noise", "0.015", "--solver", "dense", "--max-iter", "10", "--fixed", "yes", "--dump", "yes")
            y, x = load(directory / "prepared_source.csv"), load(directory / "prepared_target.csv")
            t, alpha, sigma = paper_oracle(y, x, 10)
            ch = np.genfromtxt(directory / "history.csv", delimiter=",", names=True)
            entry = {"dimension": dim, "source_count": n, "target_count": len(x),
                     "max_abs_T_error": float(np.max(np.abs(t - load(directory / "normalized_registered.csv")))),
                     "max_abs_alpha_error": float(np.max(np.abs(alpha - load(directory / "alpha.csv").ravel()))),
                     "max_abs_sigma_error_all_iterations": float(np.max(np.abs(np.array(sigma) - ch["sigma_after"]))) }
            assert entry["max_abs_T_error"] < 1e-8 and entry["max_abs_sigma_error_all_iterations"] < 1e-10, entry
            results["paper_dense_oracle"].append(entry)
    results["status"] = "PASS"
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
