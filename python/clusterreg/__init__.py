"""Fast, correspondence-free non-rigid point-set registration.

The numerical work is performed by the nanobind C++ extension.  This module
only normalizes Python inputs and provides string-valued option shortcuts.
"""
from __future__ import annotations

from typing import Any

import numpy as np

from . import _core
from ._core import (
    Algorithm,
    Backend,
    EStep,
    Iteration,
    NoiseModel,
    Options,
    RegistrationResult,
    Semantics,
    Solver,
    SyntheticData,
    SyntheticOptions,
)

__version__ = _core.__version__
has_openmp: bool = _core.has_openmp
available_threads = _core.available_threads
cuda_compiled = _core.cuda_compiled
cuda_available = _core.cuda_available
cuda_device_name = _core.cuda_device_name

_ENUM_OPTIONS = {
    "semantics": {"paper": Semantics.PAPER, "official": Semantics.OFFICIAL},
    "estep": {"streaming": EStep.STREAMING, "dense": EStep.DENSE},
    "solver": {"nystrom": Solver.NYSTROM, "dense": Solver.DENSE},
    "algorithm": {
        "clustering": Algorithm.CLUSTERING,
        "cluster": Algorithm.CLUSTERING,
        "sinkhorn": Algorithm.SINKHORN,
    },
    "noise_model": {
        "gaussian": NoiseModel.GAUSSIAN,
        "student-t": NoiseModel.STUDENT_T,
        "student_t": NoiseModel.STUDENT_T,
    },
    "backend": {
        "auto": Backend.AUTO,
        "cpu": Backend.CPU,
        "cuda": Backend.CUDA,
    },
}
_OPTION_NAMES = {
    "entropy",
    "regularization",
    "gamma",
    "rank_ratio",
    "eigen_cutoff",
    "sigma_floor",
    "sigma_ceiling",
    "initial_sigma",
    "tolerance",
    "transport_entropy",
    "source_mass_penalty",
    "target_mass_penalty",
    "student_dof",
    "sinkhorn_tolerance",
    "transport_mass_floor",
    "rank",
    "max_iterations",
    "sinkhorn_iterations",
    "kmeans_iterations",
    "threads",
    "seed",
    "normalize",
    "align_centroids",
    "fixed_iterations",
    *_ENUM_OPTIONS,
}


def _points(value: Any, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must contain real-valued coordinates")
    array = np.asarray(value, dtype=np.float64, order="C")
    if array.ndim != 2:
        raise ValueError(f"{name} must be a 2D array shaped (point, dimension)")
    return array


def _set_options(options: Options, values: dict[str, Any]) -> None:
    unknown = values.keys() - _OPTION_NAMES
    if unknown:
        names = ", ".join(sorted(unknown))
        raise TypeError(f"unknown ClusterReg option(s): {names}")
    for name, value in values.items():
        if name in _ENUM_OPTIONS and isinstance(value, str):
            try:
                value = _ENUM_OPTIONS[name][value.lower()]
            except KeyError as error:
                allowed = ", ".join(_ENUM_OPTIONS[name])
                raise ValueError(f"{name} must be one of: {allowed}") from error
        setattr(options, name, value)


def fit(
    source: Any,
    target: Any,
    options: Options | None = None,
    **option_values: Any,
) -> RegistrationResult:
    """Fit a non-rigid map from ``source`` to unordered ``target`` points.

    Inputs are copied to C-contiguous ``float64`` storage and are never
    modified.  The C++ solve releases the Python GIL.  Pass either an
    :class:`Options` object or keyword options, but not both.
    """
    if options is not None and option_values:
        raise TypeError("pass an Options object or keyword options, not both")
    if options is None:
        options = Options()
        _set_options(options, option_values)
    elif not isinstance(options, Options):
        raise TypeError("options must be a clusterreg.Options instance")
    return _core.fit(_points(source, "source"), _points(target, "target"), options)


register = fit


def make_synthetic(**values: Any) -> SyntheticData:
    """Create a deterministic 2D/3D fixture with hidden correspondences."""
    options = SyntheticOptions()
    valid = {
        "count",
        "dimension",
        "amplitude",
        "noise",
        "missing_fraction",
        "outlier_fraction",
        "seed",
    }
    unknown = values.keys() - valid
    if unknown:
        raise TypeError(f"unknown synthetic option(s): {', '.join(sorted(unknown))}")
    for name, value in values.items():
        setattr(options, name, value)
    return _core.make_synthetic(options)


__all__ = [
    "Algorithm",
    "Backend",
    "EStep",
    "Iteration",
    "NoiseModel",
    "Options",
    "RegistrationResult",
    "Semantics",
    "Solver",
    "SyntheticData",
    "SyntheticOptions",
    "available_threads",
    "cuda_available",
    "cuda_compiled",
    "cuda_device_name",
    "fit",
    "has_openmp",
    "make_synthetic",
    "register",
]
