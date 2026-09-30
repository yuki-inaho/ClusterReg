from __future__ import annotations

import gc
from pathlib import Path
import subprocess
import sys
import threading

import numpy as np
import pytest

import clusterreg

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "build/pixi/clusterreg_cli"


def _prealigned(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_scale = np.sqrt(np.square(source - source_center).sum() / len(source))
    target_scale = np.sqrt(np.square(target - target_center).sum() / len(target))
    return (source - source_center) / source_scale * target_scale + target_center


def _rmse(actual: np.ndarray, expected: np.ndarray) -> float:
    return float(np.linalg.norm(actual - expected) / np.sqrt(actual.shape[0]))


def test_import_surface_and_synthetic_reproducibility() -> None:
    assert clusterreg.__version__ == "1.1.0"
    assert clusterreg.available_threads() >= 1
    first = clusterreg.make_synthetic(count=32, dimension=3, seed=9)
    second = clusterreg.make_synthetic(count=32, dimension=3, seed=9)
    np.testing.assert_array_equal(first.source, second.source)
    np.testing.assert_array_equal(first.target, second.target)
    np.testing.assert_array_equal(first.truth, second.truth)
    assert first.target_ids == second.target_ids


def test_python_to_cpp_registration_end_to_end() -> None:
    data = clusterreg.make_synthetic(
        count=128, dimension=2, amplitude=0.4, noise=0.0, seed=42
    )
    before = _rmse(_prealigned(data.source, data.target), data.truth)
    result = clusterreg.fit(
        data.source,
        data.target,
        rank=39,
        max_iterations=40,
        threads=min(4, clusterreg.available_threads()),
        seed=42,
    )
    after = _rmse(result.transformed, data.truth)

    assert after < before * 0.7
    assert result.transformed.shape == data.source.shape
    assert result.alpha.shape == (data.source.shape[0],)
    assert abs(result.alpha.sum() - 1.0) < 1e-10
    assert result.history
    assert result.retained_rank <= result.requested_rank == 39
    np.testing.assert_allclose(
        result.transform(data.source), result.transformed, rtol=2e-12, atol=2e-12
    )


def test_layout_dtype_ownership_and_no_input_mutation() -> None:
    data = clusterreg.make_synthetic(count=48, dimension=2, seed=5)
    source_c = np.array(data.source, order="C")
    target_c = np.array(data.target, order="C")
    source_original = source_c.copy()
    target_original = target_c.copy()
    expected = clusterreg.fit(
        source_c,
        target_c,
        rank=12,
        max_iterations=5,
        fixed_iterations=True,
        threads=1,
        seed=5,
    )
    np.testing.assert_array_equal(source_c, source_original)
    np.testing.assert_array_equal(target_c, target_original)

    source_f = np.asfortranarray(source_c)
    target_f = np.asfortranarray(target_c)
    got_f = clusterreg.fit(
        source_f,
        target_f,
        rank=12,
        max_iterations=5,
        fixed_iterations=True,
        threads=1,
        seed=5,
    )
    padded_source = np.zeros((len(source_c), 4))
    padded_target = np.zeros((len(target_c), 4))
    padded_source[:, ::2] = source_c
    padded_target[:, ::2] = target_c
    got_strided = clusterreg.fit(
        padded_source[:, ::2],
        padded_target[:, ::2],
        rank=12,
        max_iterations=5,
        fixed_iterations=True,
        threads=1,
        seed=5,
    )
    np.testing.assert_array_equal(got_f.transformed, expected.transformed)
    np.testing.assert_array_equal(got_strided.transformed, expected.transformed)

    got_float32 = clusterreg.fit(
        source_c.astype(np.float32),
        target_c.astype(np.float32),
        rank=12,
        max_iterations=5,
        fixed_iterations=True,
        threads=1,
        seed=5,
    )
    np.testing.assert_allclose(got_float32.transformed, expected.transformed, atol=2e-6)

    fetched = expected.transformed
    preserved = expected.alpha
    assert fetched.flags.c_contiguous and fetched.dtype == np.float64
    assert preserved.flags.c_contiguous and preserved.dtype == np.float64
    assert not np.shares_memory(fetched, expected.transformed)
    fetched[:] = 123.0
    assert not np.all(expected.transformed == 123.0)
    del expected
    gc.collect()
    assert np.isfinite(preserved).all()
    transformed_f = got_f.transform(np.asfortranarray(source_c).astype(np.float32))
    np.testing.assert_allclose(transformed_f, got_f.transformed, atol=2e-6)


def test_streaming_thread_count_is_numerically_invariant() -> None:
    data = clusterreg.make_synthetic(count=72, dimension=3, seed=17)
    common = dict(
        rank=18,
        max_iterations=6,
        fixed_iterations=True,
        seed=17,
    )
    serial = clusterreg.fit(data.source, data.target, threads=1, **common)
    parallel = clusterreg.fit(
        data.source,
        data.target,
        threads=min(4, clusterreg.available_threads()),
        **common,
    )
    np.testing.assert_allclose(serial.transformed, parallel.transformed, atol=2e-12)
    np.testing.assert_allclose(serial.alpha, parallel.alpha, atol=2e-13)


def test_official_zero_variance_and_validation_errors() -> None:
    points = np.zeros((8, 2))
    result = clusterreg.fit(
        points,
        points,
        semantics="official",
        normalize=False,
        align_centroids=False,
        rank=1,
    )
    assert result.sigma2 == 0.0
    assert not result.history
    assert result.stop_reason == "variance_floor"

    with pytest.raises(ValueError, match="nonempty"):
        clusterreg.fit(np.empty((0, 2)), points)
    with pytest.raises(ValueError, match="NaN"):
        clusterreg.fit(np.array([[np.nan, 0.0]]), points)
    with pytest.raises(ValueError, match="dimensions differ"):
        clusterreg.fit(points, np.zeros((8, 3)))
    with pytest.raises(TypeError, match="unknown ClusterReg option"):
        clusterreg.fit(points, points, made_up=True)
    with pytest.raises(ValueError, match="semantics"):
        clusterreg.fit(points, points, semantics="almost-paper")
    with pytest.raises(ValueError, match="real-valued"):
        clusterreg.fit(points.astype(np.complex128), points)
    with pytest.raises(ValueError, match="2D array"):
        result.transform(np.zeros(2))
    with pytest.raises(ValueError, match="real-valued"):
        result.transform(points.astype(np.complex128))
    with pytest.raises(ValueError, match="nonempty"):
        result.transform(np.empty((0, 2)))
    with pytest.raises(ValueError, match="NaN"):
        result.transform(np.array([[np.nan, 0.0]]))
    with pytest.raises(ValueError, match="dimension mismatch"):
        result.transform(np.zeros((3, 3)))
    options = clusterreg.Options()
    with pytest.raises(TypeError, match="not both"):
        clusterreg.fit(points, points, options, rank=1)
    with pytest.raises(ValueError, match="Invalid synthetic"):
        clusterreg.make_synthetic(count=2)


def test_fit_releases_the_gil() -> None:
    data = clusterreg.make_synthetic(count=220, dimension=3, seed=21)
    ready = threading.Event()
    start = threading.Event()
    progressed = threading.Event()

    def worker() -> None:
        ready.set()
        start.wait()
        progressed.set()

    thread = threading.Thread(target=worker)
    thread.start()
    assert ready.wait(timeout=2)
    old_interval = sys.getswitchinterval()
    try:
        # Prevent an ordinary Python bytecode timeslice from making this pass.
        # The worker can advance only while native code explicitly releases GIL.
        sys.setswitchinterval(1000.0)
        start.set()
        assert not progressed.is_set()
        clusterreg.fit(
            data.source,
            data.target,
            rank=40,
            max_iterations=6,
            fixed_iterations=True,
            threads=1,
            seed=21,
        )
    finally:
        sys.setswitchinterval(old_interval)
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert progressed.is_set()


def test_cli_python_parity_and_truth_non_leak(tmp_path: Path) -> None:
    assert CLI.exists(), "run the test through `pixi run test-python` so the CLI is built"
    demo = tmp_path / "demo"
    common = [
        "--rank", "12",
        "--max-iter", "5",
        "--fixed", "yes",
        "--threads", "1",
        "--seed", "7",
    ]
    subprocess.run(
        [str(CLI), "demo", "--out", str(demo), "--n", "48", "--dim", "2", *common],
        check=True,
        capture_output=True,
        text=True,
    )
    source = np.loadtxt(demo / "source.csv", delimiter=",")
    target = np.loadtxt(demo / "target.csv", delimiter=",")
    native_registered = np.loadtxt(demo / "registered.csv", delimiter=",")
    native_alpha = np.loadtxt(demo / "alpha.csv", delimiter=",")
    bound = clusterreg.fit(
        source,
        target,
        rank=12,
        max_iterations=5,
        fixed_iterations=True,
        threads=1,
        seed=7,
    )
    np.testing.assert_allclose(bound.transformed, native_registered, atol=2e-15)
    np.testing.assert_allclose(bound.alpha, native_alpha, atol=2e-15)

    false_truth = tmp_path / "false_truth.csv"
    np.savetxt(false_truth, np.zeros_like(source), delimiter=",", fmt="%.17g")
    registered = tmp_path / "registered_with_false_truth"
    subprocess.run(
        [
            str(CLI), "register",
            "--source", str(demo / "source.csv"),
            "--target", str(demo / "target.csv"),
            "--truth", str(false_truth),
            "--out", str(registered),
            *common,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with_false_truth = np.loadtxt(registered / "registered.csv", delimiter=",")
    np.testing.assert_array_equal(with_false_truth, native_registered)
