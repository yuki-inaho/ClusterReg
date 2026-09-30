from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import pytest

import clusterreg


SOURCE = np.array(
    [
        [-0.80, -0.25],
        [0.05, 0.70],
        [0.95, -0.35],
    ],
    dtype=np.float64,
)
TARGET = np.array(
    [
        [-0.60, -0.10],
        [0.10, 0.85],
        [1.15, -0.20],
        [2.80, 1.90],
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class _Transport:
    gamma: np.ndarray
    omega: np.ndarray
    log_u: np.ndarray
    log_v: np.ndarray
    objective: float
    residual: float
    kkt_residual: float
    iterations: int


def _logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(
        maximum + np.log(np.exp(values - maximum).sum(axis=axis, keepdims=True)),
        axis=axis,
    )


def _pair_cost(
    transformed: np.ndarray,
    target: np.ndarray,
    scale: float,
    noise_model: str,
    student_dof: float,
) -> tuple[np.ndarray, np.ndarray]:
    dimension = transformed.shape[1]
    squared = np.square(transformed[:, None, :] - target[None, :, :]).sum(axis=2)
    if noise_model == "gaussian":
        normalizer = 0.5 * dimension * math.log(2.0 * math.pi * scale)
        return normalizer + squared / (2.0 * scale), np.ones_like(squared)

    normalizer = (
        math.lgamma(student_dof / 2.0)
        - math.lgamma((student_dof + dimension) / 2.0)
        + 0.5 * dimension * math.log(student_dof * math.pi * scale)
    )
    cost = normalizer + 0.5 * (student_dof + dimension) * np.log1p(
        squared / (student_dof * scale)
    )
    robust_weight = (student_dof + dimension) / (
        student_dof + squared / scale
    )
    return cost, robust_weight


def _generalized_kl(value: np.ndarray, reference: float) -> float:
    positive = value > 0.0
    return float(
        np.sum(value[positive] * np.log(value[positive] / reference) - value[positive])
        + reference * value.size
    )


def _transport(
    transformed: np.ndarray,
    target: np.ndarray,
    scale: float,
    *,
    noise_model: str,
    eta: float,
    tau_source: float,
    tau_target: float,
    student_dof: float,
    tolerance: float,
    max_iterations: int,
    initial_log_u: np.ndarray | None = None,
    initial_log_v: np.ndarray | None = None,
) -> _Transport:
    source_count = transformed.shape[0]
    target_count = target.shape[0]
    log_b = -math.log(source_count)
    log_a = -math.log(target_count)
    log_ba = log_b + log_a
    cost, robust_weight = _pair_cost(
        transformed, target, scale, noise_model, student_dof
    )
    log_kernel = log_ba - cost / eta
    theta_source = tau_source / (tau_source + eta)
    theta_target = tau_target / (tau_target + eta)
    contraction = theta_source * theta_target
    log_u = (
        np.zeros(source_count)
        if initial_log_u is None
        else np.array(initial_log_u, copy=True)
    )
    log_v = (
        np.zeros(target_count)
        if initial_log_v is None
        else np.array(initial_log_v, copy=True)
    )
    residual = math.inf
    used_iterations = 0
    for iteration in range(max_iterations):
        next_u = theta_source * (log_b - _logsumexp(log_kernel + log_v, axis=1))
        next_v = theta_target * (
            log_a - _logsumexp(log_kernel + next_u[:, None], axis=0)
        )
        residual = float(np.max(np.abs(next_v - log_v)) / (1.0 - contraction))
        log_u, log_v = next_u, next_v
        used_iterations = iteration + 1
        if residual <= tolerance:
            break

    log_gamma = log_u[:, None] + log_kernel + log_v[None, :]
    gamma = np.exp(log_gamma)
    omega = gamma * robust_weight
    source_mass = gamma.sum(axis=1)
    target_mass = gamma.sum(axis=0)
    plan_kl = float(np.sum(gamma * (log_gamma - log_ba) - gamma) + 1.0)
    objective = (
        float(np.sum(gamma * cost))
        + eta * plan_kl
        + tau_source * _generalized_kl(source_mass, 1.0 / source_count)
        + tau_target * _generalized_kl(target_mass, 1.0 / target_count)
    )
    kkt = (
        eta * (log_u[:, None] + log_v[None, :])
        + tau_source * np.log(source_mass[:, None] * source_count)
        + tau_target * np.log(target_mass[None, :] * target_count)
    )
    return _Transport(
        gamma=gamma,
        omega=omega,
        log_u=log_u,
        log_v=log_v,
        objective=objective,
        residual=residual,
        kkt_residual=float(np.max(np.abs(kkt))),
        iterations=used_iterations,
    )


def _dense_update(
    source: np.ndarray,
    target: np.ndarray,
    transport: _Transport,
    scale: float,
    *,
    kernel_gamma: float,
    regularization: float,
    sigma_floor: float,
    sigma_ceiling: float,
) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    kernel = np.exp(
        -kernel_gamma
        * np.abs(source[:, None, :] - source[None, :, :]).sum(axis=2)
    )
    mass = transport.omega.sum(axis=1)
    px = transport.omega @ target
    system = mass[:, None] * kernel + regularization * scale * np.eye(len(source))
    coefficients = np.linalg.solve(system, px - mass[:, None] * source)
    transformed = source + kernel @ coefficients
    penalty = float(np.sum(coefficients * (kernel @ coefficients)))
    squared = np.square(transformed[:, None, :] - target[None, :, :]).sum(axis=2)
    weighted_sse = float(np.sum(transport.omega * squared))
    next_scale = float(
        np.clip(
            weighted_sse / (source.shape[1] * transport.gamma.sum()),
            sigma_floor,
            sigma_ceiling,
        )
    )
    return transformed, coefficients, penalty, weighted_sse, next_scale


def _options(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "algorithm": "sinkhorn",
        "noise_model": "gaussian",
        "backend": "cpu",
        "solver": "dense",
        "normalize": False,
        "align_centroids": False,
        "fixed_iterations": True,
        "max_iterations": 1,
        "sinkhorn_iterations": 600,
        "sinkhorn_tolerance": 1e-12,
        "initial_sigma": 0.65,
        "sigma_floor": 1e-8,
        "sigma_ceiling": 10.0,
        "transport_entropy": 0.7,
        "source_mass_penalty": 2.5,
        "target_mass_penalty": 3.5,
        "regularization": 0.2,
        "gamma": 1.1,
        "threads": 1,
        "seed": 19,
    }
    values.update(overrides)
    return values


def _oracle_from_options(
    source: np.ndarray, target: np.ndarray, options: dict[str, object]
) -> tuple[_Transport, np.ndarray, np.ndarray, float, float, float]:
    first = _transport(
        source,
        target,
        float(options["initial_sigma"]),
        noise_model=str(options["noise_model"]),
        eta=float(options["transport_entropy"]),
        tau_source=float(options["source_mass_penalty"]),
        tau_target=float(options["target_mass_penalty"]),
        student_dof=float(options.get("student_dof", 4.0)),
        tolerance=float(options["sinkhorn_tolerance"]),
        max_iterations=int(options["sinkhorn_iterations"]),
    )
    transformed, coefficients, penalty, weighted_sse, next_scale = _dense_update(
        source,
        target,
        first,
        float(options["initial_sigma"]),
        kernel_gamma=float(options["gamma"]),
        regularization=float(options["regularization"]),
        sigma_floor=float(options["sigma_floor"]),
        sigma_ceiling=float(options["sigma_ceiling"]),
    )
    return first, transformed, coefficients, penalty, weighted_sse, next_scale


def test_gaussian_matches_dense_log_domain_oracle_and_is_unbalanced() -> None:
    options = _options()
    result = clusterreg.fit(SOURCE, TARGET, **options)
    first, transformed, coefficients, penalty, _, next_scale = _oracle_from_options(
        SOURCE, TARGET, options
    )
    record = result.history[0]

    candidate_cost, _ = _pair_cost(
        transformed, TARGET, next_scale, "gaussian", 4.0
    )
    objective_after = (
        first.objective
        - float(np.sum(first.gamma * _pair_cost(SOURCE, TARGET, 0.65, "gaussian", 4.0)[0]))
        + float(np.sum(first.gamma * candidate_cost))
        + 0.5 * float(options["regularization"]) * penalty
    )

    np.testing.assert_allclose(result.transformed, transformed, rtol=2e-11, atol=2e-12)
    np.testing.assert_allclose(result.coefficients, coefficients, rtol=2e-11, atol=2e-12)
    assert result.sigma2 == pytest.approx(next_scale, rel=2e-11, abs=2e-12)
    assert record.transport_mass == pytest.approx(first.gamma.sum(), rel=2e-12)
    assert record.robust_mass == pytest.approx(first.gamma.sum(), rel=2e-12)
    assert record.objective_before == pytest.approx(first.objective, rel=3e-12)
    assert record.objective_after == pytest.approx(objective_after, rel=3e-11)
    assert record.transport_residual == pytest.approx(first.residual, rel=1e-2, abs=2e-15)
    assert record.kkt_residual == pytest.approx(first.kkt_residual, rel=2e-3, abs=2e-14)
    assert record.transport_iterations == first.iterations
    assert record.objective_after < record.objective_before
    assert record.transport_residual <= float(options["sinkhorn_tolerance"])
    assert record.kkt_residual < 2e-11

    # UOT deliberately does not project either marginal onto a probability simplex.
    assert abs(record.transport_mass - 1.0) > 1e-3
    assert result.alpha.sum() == pytest.approx(result.transport_mass, rel=2e-12)
    assert result.source_mass.sum() == pytest.approx(result.transport_mass, rel=2e-12)
    assert result.target_mass.sum() == pytest.approx(result.transport_mass, rel=2e-12)
    assert np.isfinite(result.transformed).all()
    assert np.isfinite(result.source_mass).all()
    assert np.isfinite(result.target_mass).all()

    final = _transport(
        transformed,
        TARGET,
        next_scale,
        noise_model="gaussian",
        eta=float(options["transport_entropy"]),
        tau_source=float(options["source_mass_penalty"]),
        tau_target=float(options["target_mass_penalty"]),
        student_dof=4.0,
        tolerance=float(options["sinkhorn_tolerance"]),
        max_iterations=int(options["sinkhorn_iterations"]),
        initial_log_u=first.log_u,
        initial_log_v=first.log_v,
    )
    np.testing.assert_allclose(result.source_mass, final.gamma.sum(axis=1), rtol=3e-12)
    np.testing.assert_allclose(result.target_mass, final.gamma.sum(axis=0), rtol=3e-12)


def test_target_permutation_preserves_solution_and_reindexes_target_mass() -> None:
    options = _options(max_iterations=3)
    reference = clusterreg.fit(SOURCE, TARGET, **options)
    permutation = np.array([2, 0, 3, 1])
    permuted = clusterreg.fit(SOURCE, TARGET[permutation], **options)

    np.testing.assert_allclose(permuted.transformed, reference.transformed, atol=3e-12)
    np.testing.assert_allclose(permuted.source_mass, reference.source_mass, atol=3e-12)
    np.testing.assert_allclose(
        permuted.target_mass[np.argsort(permutation)], reference.target_mass, atol=3e-12
    )
    np.testing.assert_allclose(
        [item.objective_after for item in permuted.history],
        [item.objective_after for item in reference.history],
        rtol=2e-12,
        atol=2e-12,
    )


def test_common_normalization_and_transform_contract() -> None:
    source = SOURCE * np.array([7.0, 0.4]) + np.array([12.0, -3.0])
    target = TARGET * np.array([7.0, 0.4]) + np.array([12.0, -3.0])
    result = clusterreg.fit(
        source,
        target,
        **_options(normalize=True, align_centroids=True, max_iterations=2),
    )

    count = len(source) + len(target)
    center = (source.sum(axis=0) + target.sum(axis=0)) / count
    scale = math.sqrt(
        (np.square(source - center).sum() + np.square(target - center).sum()) / count
    )
    prepared_source = (source - center) / scale
    prepared_target = (target - center) / scale
    shift = prepared_target.mean(axis=0) - prepared_source.mean(axis=0)

    np.testing.assert_allclose(result.source_center, center, rtol=0.0, atol=2e-15)
    np.testing.assert_allclose(result.target_center, center, rtol=0.0, atol=2e-15)
    assert result.source_scale == pytest.approx(scale, rel=2e-15)
    assert result.target_scale == pytest.approx(scale, rel=2e-15)
    np.testing.assert_allclose(result.source_shift, shift, atol=2e-15)
    np.testing.assert_allclose(result.prepared_source, prepared_source + shift, atol=2e-15)
    np.testing.assert_allclose(result.prepared_target, prepared_target, atol=2e-15)
    np.testing.assert_allclose(result.transform(source), result.transformed, atol=3e-12)
    np.testing.assert_allclose(
        result.normalized_transformed * scale + center,
        result.transformed,
        atol=3e-12,
    )


def test_student_t_uses_gamma_mass_for_scale_and_descends() -> None:
    options = _options(noise_model="student-t", student_dof=3.0)
    result = clusterreg.fit(SOURCE, TARGET, **options)
    first, _, _, _, weighted_sse, next_scale = _oracle_from_options(
        SOURCE, TARGET, options
    )
    record = result.history[0]

    gamma_mass = float(first.gamma.sum())
    robust_mass = float(first.omega.sum())
    wrong_robust_denominator = weighted_sse / (SOURCE.shape[1] * robust_mass)
    assert abs(gamma_mass - robust_mass) > 1e-3
    assert record.transport_mass == pytest.approx(gamma_mass, rel=3e-12)
    assert record.robust_mass == pytest.approx(robust_mass, rel=3e-12)
    assert record.sigma_after == pytest.approx(next_scale, rel=3e-11)
    assert abs(record.sigma_after - wrong_robust_denominator) > 1e-3
    assert record.objective_after <= record.objective_before + 2e-9 * (
        1.0 + abs(record.objective_before)
    )
    assert np.isfinite(record.objective_before)
    assert np.isfinite(record.objective_after)
    assert np.isfinite(result.transformed).all()


def test_explicit_cuda_request_has_well_defined_availability_behavior() -> None:
    options = _options(sinkhorn_iterations=40, sinkhorn_tolerance=1e-7)
    automatic = clusterreg.fit(SOURCE, TARGET, **(options | {"backend": "auto"}))
    expected_backend = (
        clusterreg.Backend.CUDA
        if clusterreg.cuda_available()
        else clusterreg.Backend.CPU
    )
    assert automatic.backend_used == expected_backend
    if clusterreg.cuda_available():
        result = clusterreg.fit(SOURCE, TARGET, **(options | {"backend": "cuda"}))
        assert result.backend_used == clusterreg.Backend.CUDA
        assert np.isfinite(result.transformed).all()
    else:
        reason = "no CUDA device is available" if clusterreg.cuda_compiled() else "no CUDA support"
        with pytest.raises(RuntimeError, match=reason):
            clusterreg.fit(SOURCE, TARGET, **(options | {"backend": "cuda"}))


@pytest.mark.skipif(not clusterreg.cuda_available(), reason="CUDA backend unavailable")
@pytest.mark.parametrize("noise_model", ["gaussian", "student-t"])
def test_cpu_cuda_parity_when_available(noise_model: str) -> None:
    options = _options(
        noise_model=noise_model,
        max_iterations=2,
        sinkhorn_iterations=300,
        sinkhorn_tolerance=1e-10,
    )
    cpu = clusterreg.fit(SOURCE, TARGET, **options)
    gpu = clusterreg.fit(SOURCE, TARGET, **(options | {"backend": "cuda"}))

    assert cpu.backend_used == clusterreg.Backend.CPU
    assert gpu.backend_used == clusterreg.Backend.CUDA
    np.testing.assert_allclose(gpu.transformed, cpu.transformed, rtol=2e-8, atol=2e-9)
    np.testing.assert_allclose(gpu.source_mass, cpu.source_mass, rtol=3e-8, atol=2e-9)
    np.testing.assert_allclose(gpu.target_mass, cpu.target_mass, rtol=3e-8, atol=2e-9)
    assert gpu.sigma2 == pytest.approx(cpu.sigma2, rel=3e-8, abs=2e-9)
    np.testing.assert_allclose(
        [item.objective_after for item in gpu.history],
        [item.objective_after for item in cpu.history],
        rtol=5e-8,
        atol=5e-9,
    )


@pytest.mark.skipif(not clusterreg.cuda_available(), reason="CUDA backend unavailable")
def test_cpu_cuda_parity_for_generic_dimension_kernel() -> None:
    rng = np.random.default_rng(20260930)
    source = rng.normal(scale=0.4, size=(6, 5))
    target = rng.normal(scale=0.4, size=(8, 5))
    options = _options(
        noise_model="student-t",
        student_dof=3.5,
        initial_sigma=0.8,
        max_iterations=2,
        sinkhorn_iterations=300,
        sinkhorn_tolerance=1e-10,
    )
    cpu = clusterreg.fit(source, target, **options)
    gpu = clusterreg.fit(source, target, **(options | {"backend": "cuda"}))

    np.testing.assert_allclose(gpu.transformed, cpu.transformed, rtol=3e-8, atol=3e-9)
    np.testing.assert_allclose(gpu.source_mass, cpu.source_mass, rtol=4e-8, atol=3e-9)
    np.testing.assert_allclose(gpu.target_mass, cpu.target_mass, rtol=4e-8, atol=3e-9)
    assert gpu.sigma2 == pytest.approx(cpu.sigma2, rel=4e-8, abs=3e-9)
