"""Independent boundary coverage for exact, vectorized UOT kernels."""
import numpy as np
import pytest

import clusterreg
from test_sinkhorn import SOURCE, TARGET, _options, _oracle_from_options


@pytest.mark.parametrize("dimension", [1, 3, 5])
@pytest.mark.parametrize("noise_model", ["gaussian", "student-t"])
def test_worker_reduction_preserves_solution(dimension, noise_model):
    rng = np.random.default_rng(20261001)
    source = rng.normal(scale=0.4, size=(17, dimension))
    target = rng.normal(scale=0.4, size=(23, dimension))
    options = _options(solver="nystrom", rank=7, noise_model=noise_model,
                       max_iterations=3, sinkhorn_iterations=70,
                       sinkhorn_tolerance=0.0, initial_sigma=0.8)
    serial = clusterreg.fit(source, target, **options)
    parallel = clusterreg.fit(source, target, **(options | {"threads": 4}))
    assert serial.stop_reason == parallel.stop_reason == "max_iterations"
    assert [h.transport_iterations for h in serial.history] == [
        h.transport_iterations for h in parallel.history]
    for field in ("transformed", "source_mass", "target_mass"):
        np.testing.assert_allclose(getattr(parallel, field), getattr(serial, field),
                                   rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose([h.objective_after for h in parallel.history],
                               [h.objective_after for h in serial.history],
                               rtol=3e-11, atol=3e-12)


def test_student_small_eta_matches_independent_dense_oracle():
    options = _options(noise_model="student-t", transport_entropy=0.003,
                       source_mass_penalty=0.3, target_mass_penalty=0.4,
                       sinkhorn_iterations=600, sinkhorn_tolerance=0.0)
    result = clusterreg.fit(SOURCE, TARGET, **options)
    first, transformed, coefficients, _, _, scale = _oracle_from_options(
        SOURCE, TARGET, options)
    np.testing.assert_allclose(result.transformed, transformed, rtol=3e-11, atol=3e-12)
    np.testing.assert_allclose(result.coefficients, coefficients, rtol=3e-11, atol=3e-12)
    assert result.sigma2 == pytest.approx(scale, rel=3e-11, abs=3e-12)
    assert result.history[0].transport_mass == pytest.approx(first.gamma.sum(), rel=3e-11)


def test_gaussian_extinguished_tail_has_no_artificial_mass():
    target = np.vstack((TARGET, [1000.0, 1000.0]))
    result = clusterreg.fit(SOURCE, target, **_options(max_iterations=2))
    assert result.stop_reason == "max_iterations"
    assert result.target_mass[-1] == 0.0
    assert np.isfinite(result.transformed).all()
    assert all(np.isfinite(h.kkt_residual) for h in result.history)
    assert result.transport_mass > 0


@pytest.mark.parametrize("eta,tau", [(1.0, 1e300), (1e-300, 1.0), (1e300, 1e-300)])
def test_unrepresentable_contraction_is_explicitly_rejected(eta, tau):
    with pytest.raises(ValueError, match="(?i)(theta|contraction|condition)"):
        clusterreg.fit(SOURCE, TARGET, **_options(transport_entropy=eta,
                      source_mass_penalty=tau, target_mass_penalty=tau))
