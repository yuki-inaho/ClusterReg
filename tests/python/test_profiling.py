"""Reject misleading speed reports before interpreting their timings."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile = load_script("profile_sinkhorn")
compare = load_script("compare_sinkhorn")


def fake_result():
    history = []
    for index in range(1, 4):
        history.append(SimpleNamespace(
            index=index, sigma_before=0.5, sigma_after=0.4,
            objective_before=3.0 - index * 0.3,
            objective_after=2.9 - index * 0.3,
            transport_mass=1.0, robust_mass=1.0, transport_residual=0.0,
            kkt_residual=0.0, linear_residual=0.0, step_rms=0.1,
            scale_relative_change=0.2, estep_seconds=0.01,
            solve_seconds=0.005, transport_iterations=50))
    return SimpleNamespace(
        transformed=np.zeros((4, 3)), source_mass=np.full(4, 0.25),
        target_mass=np.full(5, 0.2), sigma2=0.4, transport_mass=1.0,
        preparation_seconds=0.01, iteration_seconds=0.07, total_seconds=0.1,
        stop_reason="max_iterations", history=history)


def validate(result):
    return profile.validate_result(result, outer_iterations=3,
                                   sinkhorn_iterations=50,
                                   source_shape=(4, 3), target_shape=(5, 3))


def test_profile_accepts_valid_complete_result():
    validate(fake_result())


@pytest.mark.parametrize("mutation", [
    lambda result: setattr(result, "stop_reason", "mass_collapse"),
    lambda result: result.history.pop(),
    lambda result: setattr(result.history[1], "transport_iterations", 49),
    lambda result: setattr(result.history[1], "objective_after", 10.0),
    lambda result: setattr(result.history[1], "kkt_residual", 1e-7),
    lambda result: setattr(result.history[1], "transport_residual", float("nan")),
    lambda result: setattr(result, "sigma2", float("nan")),
    lambda result: result.transformed.__setitem__((0, 0), float("nan")),
    lambda result: result.source_mass.__setitem__(0, float("nan")),
    lambda result: result.target_mass.__setitem__(0, float("nan")),
    lambda result: setattr(result, "total_seconds", 0.0),
])
def test_profile_rejects_invalid_or_incomplete_result(mutation):
    result = fake_result()
    mutation(result)
    with pytest.raises(ValueError):
        validate(result)


def test_profile_parity_rejects_incorrect_cuda_marginals():
    reference = fake_result()
    candidate = deepcopy(reference)
    candidate.target_mass[0] += 1e-3
    with pytest.raises(ValueError, match="target_mass"):
        profile.validate_parity(candidate, reference)


def test_compare_zero_timing_remains_a_serializable_failed_report():
    run = {"label": "candidate", "metrics": {
        "fit_seconds": 0.0, "preparation_seconds": 0.0,
        "estep_seconds": 0.0, "solve_seconds": 0.0,
        "iterations": 1, "stop_reason": "max_iterations"},
        "history": [{"transport_iterations": 50, "kkt_residual": 0.0}],
        "wall_seconds": 0.1, "validation_errors": ["nonpositive fit time"],
        "comparison": {"passed": False}}
    summary = compare.summarize([run], [run])
    assert summary["speedup_vs_baseline"] is None
    assert summary["slowdown_fraction"] is None
    assert not summary["validation_passed"]
    json.dumps(summary, allow_nan=False)


def test_compare_nonfinite_failure_diagnostics_are_strict_json():
    converted = compare.json_safe({"errors": ["nonfinite"],
                                   "history": [{"objective": float("nan")}],
                                   "mass": float("inf")})
    assert converted["mass"] is None
    assert converted["history"][0]["objective"] is None
    json.dumps(converted, allow_nan=False)


def fake_compare_runs():
    result = fake_result()
    measured = {
        "noise_model": "gaussian", "coordinates": result.transformed,
        "source_mass": result.source_mass, "input_source_shape": [4, 3],
        "input_target_shape": [5, 3], "metrics": {
            "source_count": 4, "target_count": 5, "dimension": 3,
            "algorithm": "sinkhorn", "backend": "cpu", "noise_model": "gaussian",
            "iterations": 3, "stop_reason": "max_iterations", "sigma2": 0.4,
            "transport_mass": 1.0, "fit_seconds": 0.1},
        "history": [{("iteration" if key == "index" else key): value
                     for key, value in vars(row).items()} for row in result.history],
    }
    probe = deepcopy(measured)
    extra = deepcopy(probe["history"][-1])
    extra.update(iteration=4, sigma_before=0.4, objective_before=1.9,
                 objective_after=1.8)
    probe["history"].append(extra)
    probe["metrics"]["iterations"] = 4
    args = SimpleNamespace(outer_iterations=3, sinkhorn_iterations=50)
    probe["validation_errors"] = compare.check_final_e_probe(probe, args)
    return measured, probe, args


def test_compare_final_e_probe_accepts_rejected_extra_m_step():
    measured, probe, args = fake_compare_runs()
    probe["metrics"]["stop_reason"] = "non_descent"
    probe["history"][-1]["objective_after"] = 20.0
    probe["coordinates"][:] = 10.0  # Extra M-step coordinates are not compared.
    probe["validation_errors"] = compare.check_final_e_probe(probe, args)
    assert not probe["validation_errors"]
    checks = compare.compare_final_e_probe(measured, probe, probe, args)
    assert checks["passed"]
    assert checks["final_e_probe_sweeps"] == 50
    assert checks["warmup_excluded_from_measurements"]


@pytest.mark.parametrize("mutation", [
    lambda probe: probe["history"].pop(),
    lambda probe: probe["history"][0].__setitem__("objective_before", 2.8),
    lambda probe: probe["history"][-1].__setitem__("transport_iterations", 49),
])
def test_compare_final_e_probe_rejects_missing_mismatched_or_short_work(mutation):
    measured, probe, args = fake_compare_runs()
    baseline_probe = deepcopy(probe)
    mutation(probe)
    probe["validation_errors"] = compare.check_final_e_probe(probe, args)
    assert not compare.compare_final_e_probe(measured, probe, baseline_probe, args)["passed"]


def test_compare_final_e_probe_requires_baseline_sweep_agreement():
    measured, probe, args = fake_compare_runs()
    baseline_probe = deepcopy(probe)
    baseline_probe["history"][-1]["transport_iterations"] = 49
    baseline_probe["validation_errors"] = compare.check_final_e_probe(baseline_probe, args)
    assert not compare.compare_final_e_probe(measured, probe, baseline_probe, args)["passed"]
