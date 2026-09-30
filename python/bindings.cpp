// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/clusterreg.hpp"
#include "clusterreg/io.hpp"

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <algorithm>
#include <cstddef>
#include <memory>

namespace nb = nanobind;
using namespace nb::literals;

namespace {
using clusterreg::Index;
using clusterreg::Matrix;
using clusterreg::Vector;
using InputArray = nb::ndarray<nb::numpy, const double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using MatrixArray = nb::ndarray<nb::numpy, double, nb::ndim<2>, nb::c_contig, nb::device::cpu>;
using VectorArray = nb::ndarray<nb::numpy, double, nb::ndim<1>, nb::c_contig, nb::device::cpu>;

Matrix matrix_from_numpy(const InputArray& input) {
    const auto rows = static_cast<Index>(input.shape(0));
    const auto cols = static_cast<Index>(input.shape(1));
    Matrix output(rows, cols);
    const double* data = input.data();
    for (Index i = 0; i < rows; ++i) {
        for (Index j = 0; j < cols; ++j) {
            output(i, j) = data[static_cast<std::size_t>(i * cols + j)];
        }
    }
    return output;
}

MatrixArray matrix_to_numpy(const Matrix& input) {
    const std::size_t rows = static_cast<std::size_t>(input.rows());
    const std::size_t cols = static_cast<std::size_t>(input.cols());
    auto data = std::make_unique<double[]>(rows * cols);
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            data[i * cols + j] = input(static_cast<Index>(i), static_cast<Index>(j));
        }
    }
    double* raw = data.get();
    nb::capsule owner(raw, [](void* pointer) noexcept {
        delete[] static_cast<double*>(pointer);
    });
    data.release();
    return MatrixArray(raw, {rows, cols}, owner);
}

VectorArray vector_to_numpy(const Vector& input) {
    const std::size_t size = static_cast<std::size_t>(input.size());
    auto data = std::make_unique<double[]>(size);
    std::copy_n(input.data(), size, data.get());
    double* raw = data.get();
    nb::capsule owner(raw, [](void* pointer) noexcept {
        delete[] static_cast<double*>(pointer);
    });
    data.release();
    return VectorArray(raw, {size}, owner);
}

VectorArray row_vector_to_numpy(const Eigen::RowVectorXd& input) {
    Vector temporary = input.transpose();
    return vector_to_numpy(temporary);
}
} // namespace

NB_MODULE(_core, module) {
    module.doc() = "nanobind interface to the ClusterReg C++17 core";
    module.attr("__version__") = CLUSTERREG_VERSION;
#ifdef _OPENMP
    module.attr("has_openmp") = true;
#else
    module.attr("has_openmp") = false;
#endif

    nb::enum_<clusterreg::Semantics>(module, "Semantics")
        .value("PAPER", clusterreg::Semantics::Paper)
        .value("OFFICIAL", clusterreg::Semantics::Official);
    nb::enum_<clusterreg::EStep>(module, "EStep")
        .value("STREAMING", clusterreg::EStep::Streaming)
        .value("DENSE", clusterreg::EStep::Dense);
    nb::enum_<clusterreg::Solver>(module, "Solver")
        .value("NYSTROM", clusterreg::Solver::Nystrom)
        .value("DENSE", clusterreg::Solver::Dense);
    nb::enum_<clusterreg::Algorithm>(module, "Algorithm")
        .value("CLUSTERING", clusterreg::Algorithm::Clustering)
        .value("SINKHORN", clusterreg::Algorithm::Sinkhorn);
    nb::enum_<clusterreg::NoiseModel>(module, "NoiseModel")
        .value("GAUSSIAN", clusterreg::NoiseModel::Gaussian)
        .value("STUDENT_T", clusterreg::NoiseModel::StudentT);
    nb::enum_<clusterreg::Backend>(module, "Backend")
        .value("AUTO", clusterreg::Backend::Auto)
        .value("CPU", clusterreg::Backend::CPU)
        .value("CUDA", clusterreg::Backend::CUDA);

    nb::class_<clusterreg::Options>(module, "Options")
        .def(nb::init<>())
        .def_rw("entropy", &clusterreg::Options::entropy)
        .def_rw("regularization", &clusterreg::Options::regularization)
        .def_rw("gamma", &clusterreg::Options::gamma)
        .def_rw("rank_ratio", &clusterreg::Options::rank_ratio)
        .def_rw("eigen_cutoff", &clusterreg::Options::eigen_cutoff)
        .def_rw("sigma_floor", &clusterreg::Options::sigma_floor)
        .def_rw("sigma_ceiling", &clusterreg::Options::sigma_ceiling)
        .def_rw("initial_sigma", &clusterreg::Options::initial_sigma)
        .def_rw("tolerance", &clusterreg::Options::tolerance)
        .def_rw("transport_entropy", &clusterreg::Options::transport_entropy)
        .def_rw("source_mass_penalty", &clusterreg::Options::source_mass_penalty)
        .def_rw("target_mass_penalty", &clusterreg::Options::target_mass_penalty)
        .def_rw("student_dof", &clusterreg::Options::student_dof)
        .def_rw("sinkhorn_tolerance", &clusterreg::Options::sinkhorn_tolerance)
        .def_rw("transport_mass_floor", &clusterreg::Options::transport_mass_floor)
        .def_rw("rank", &clusterreg::Options::rank)
        .def_rw("max_iterations", &clusterreg::Options::max_iterations)
        .def_rw("sinkhorn_iterations", &clusterreg::Options::sinkhorn_iterations)
        .def_rw("kmeans_iterations", &clusterreg::Options::kmeans_iterations)
        .def_rw("threads", &clusterreg::Options::threads)
        .def_rw("seed", &clusterreg::Options::seed)
        .def_rw("normalize", &clusterreg::Options::normalize)
        .def_rw("align_centroids", &clusterreg::Options::align_centroids)
        .def_rw("fixed_iterations", &clusterreg::Options::fixed_iterations)
        .def_rw("semantics", &clusterreg::Options::semantics)
        .def_rw("estep", &clusterreg::Options::estep)
        .def_rw("solver", &clusterreg::Options::solver)
        .def_rw("algorithm", &clusterreg::Options::algorithm)
        .def_rw("noise_model", &clusterreg::Options::noise_model)
        .def_rw("backend", &clusterreg::Options::backend);

    nb::class_<clusterreg::Iteration>(module, "Iteration")
        .def_ro("index", &clusterreg::Iteration::index)
        .def_ro("sigma_before", &clusterreg::Iteration::sigma_before)
        .def_ro("sigma_after", &clusterreg::Iteration::sigma_after)
        .def_ro("after_u", &clusterreg::Iteration::after_u)
        .def_ro("after_alpha", &clusterreg::Iteration::after_alpha)
        .def_ro("after_deform", &clusterreg::Iteration::after_deform)
        .def_ro("after_variance", &clusterreg::Iteration::after_variance)
        .def_ro("official_loss", &clusterreg::Iteration::official_loss)
        .def_ro("row_sum_error", &clusterreg::Iteration::row_sum_error)
        .def_ro("alpha_sum", &clusterreg::Iteration::alpha_sum)
        .def_ro("linear_residual", &clusterreg::Iteration::linear_residual)
        .def_ro("step_rms", &clusterreg::Iteration::step_rms)
        .def_ro("estep_seconds", &clusterreg::Iteration::estep_seconds)
        .def_ro("solve_seconds", &clusterreg::Iteration::solve_seconds)
        .def_ro("objective_before", &clusterreg::Iteration::objective_before)
        .def_ro("objective_after", &clusterreg::Iteration::objective_after)
        .def_ro("transport_mass", &clusterreg::Iteration::transport_mass)
        .def_ro("robust_mass", &clusterreg::Iteration::robust_mass)
        .def_ro("transport_residual", &clusterreg::Iteration::transport_residual)
        .def_ro("kkt_residual", &clusterreg::Iteration::kkt_residual)
        .def_ro("scale_relative_change", &clusterreg::Iteration::scale_relative_change)
        .def_ro("transport_iterations", &clusterreg::Iteration::transport_iterations);

    nb::class_<clusterreg::Result>(module, "RegistrationResult")
        .def_prop_ro("transformed", [](const clusterreg::Result& value) { return matrix_to_numpy(value.transformed); }, nb::rv_policy::automatic)
        .def_prop_ro("prepared_source", [](const clusterreg::Result& value) { return matrix_to_numpy(value.prepared_source); }, nb::rv_policy::automatic)
        .def_prop_ro("prepared_target", [](const clusterreg::Result& value) { return matrix_to_numpy(value.prepared_target); }, nb::rv_policy::automatic)
        .def_prop_ro("normalized_transformed", [](const clusterreg::Result& value) { return matrix_to_numpy(value.normalized_transformed); }, nb::rv_policy::automatic)
        .def_prop_ro("coefficients", [](const clusterreg::Result& value) { return matrix_to_numpy(value.coefficients); }, nb::rv_policy::automatic)
        .def_prop_ro("alpha", [](const clusterreg::Result& value) { return vector_to_numpy(value.alpha); }, nb::rv_policy::automatic)
        .def_prop_ro("source_mass", [](const clusterreg::Result& value) { return vector_to_numpy(value.source_mass); }, nb::rv_policy::automatic)
        .def_prop_ro("target_mass", [](const clusterreg::Result& value) { return vector_to_numpy(value.target_mass); }, nb::rv_policy::automatic)
        .def_prop_ro("source_center", [](const clusterreg::Result& value) { return row_vector_to_numpy(value.source_center); }, nb::rv_policy::automatic)
        .def_prop_ro("target_center", [](const clusterreg::Result& value) { return row_vector_to_numpy(value.target_center); }, nb::rv_policy::automatic)
        .def_prop_ro("source_shift", [](const clusterreg::Result& value) { return row_vector_to_numpy(value.source_shift); }, nb::rv_policy::automatic)
        .def_prop_ro("basis_centers", [](const clusterreg::Result& value) { return matrix_to_numpy(value.basis.centers); }, nb::rv_policy::automatic)
        .def_prop_ro("basis_projection", [](const clusterreg::Result& value) { return matrix_to_numpy(value.basis.projection); }, nb::rv_policy::automatic)
        .def_prop_ro("basis_q", [](const clusterreg::Result& value) { return matrix_to_numpy(value.basis.Q); }, nb::rv_policy::automatic)
        .def_ro("sigma2", &clusterreg::Result::sigma2)
        .def_ro("source_scale", &clusterreg::Result::source_scale)
        .def_ro("target_scale", &clusterreg::Result::target_scale)
        .def_ro("history", &clusterreg::Result::history)
        .def_ro("stop_reason", &clusterreg::Result::stop_reason)
        .def_ro("preparation_seconds", &clusterreg::Result::preparation_seconds)
        .def_ro("iteration_seconds", &clusterreg::Result::iteration_seconds)
        .def_ro("total_seconds", &clusterreg::Result::total_seconds)
        .def_ro("backend_used", &clusterreg::Result::backend_used)
        .def_ro("transport_mass", &clusterreg::Result::transport_mass)
        .def_prop_ro("requested_rank", [](const clusterreg::Result& value) { return value.basis.requested_rank; })
        .def_prop_ro("retained_rank", [](const clusterreg::Result& value) { return value.basis.retained_rank; })
        .def_prop_ro("min_retained_eigenvalue", [](const clusterreg::Result& value) { return value.basis.min_retained_eigenvalue; })
        .def("transform", [](const clusterreg::Result& model, nb::object query) {
            nb::object numpy = nb::module_::import_("numpy");
            if(nb::cast<bool>(numpy.attr("iscomplexobj")(query)))
                throw std::invalid_argument("query must contain real-valued coordinates");
            if(nb::cast<int>(numpy.attr("ndim")(query))!=2)
                throw std::invalid_argument("query must be a 2D array shaped (point, dimension)");
            nb::object converted = numpy.attr("ascontiguousarray")(
                query, "dtype"_a=numpy.attr("float64"));
            InputArray array = nb::cast<InputArray>(converted);
            Matrix native_query = matrix_from_numpy(array);
            Matrix output;
            {
                nb::gil_scoped_release release;
                output = clusterreg::transform(model, native_query);
            }
            return matrix_to_numpy(output);
        }, "query"_a, "Apply the learned displacement field to source-space points.");

    nb::class_<clusterreg::SyntheticOptions>(module, "SyntheticOptions")
        .def(nb::init<>())
        .def_rw("count", &clusterreg::SyntheticOptions::count)
        .def_rw("dimension", &clusterreg::SyntheticOptions::dimension)
        .def_rw("amplitude", &clusterreg::SyntheticOptions::amplitude)
        .def_rw("noise", &clusterreg::SyntheticOptions::noise)
        .def_rw("missing_fraction", &clusterreg::SyntheticOptions::missing_fraction)
        .def_rw("outlier_fraction", &clusterreg::SyntheticOptions::outlier_fraction)
        .def_rw("seed", &clusterreg::SyntheticOptions::seed);

    nb::class_<clusterreg::SyntheticData>(module, "SyntheticData")
        .def_prop_ro("source", [](const clusterreg::SyntheticData& value) { return matrix_to_numpy(value.source); }, nb::rv_policy::automatic)
        .def_prop_ro("target", [](const clusterreg::SyntheticData& value) { return matrix_to_numpy(value.target); }, nb::rv_policy::automatic)
        .def_prop_ro("truth", [](const clusterreg::SyntheticData& value) { return matrix_to_numpy(value.truth); }, nb::rv_policy::automatic)
        .def_ro("target_ids", &clusterreg::SyntheticData::target_ids);

    module.def("fit", [](const InputArray& source, const InputArray& target, const clusterreg::Options& options) {
        Matrix native_source = matrix_from_numpy(source);
        Matrix native_target = matrix_from_numpy(target);
        clusterreg::Options native_options = options;
        clusterreg::Result result;
        {
            nb::gil_scoped_release release;
            result = clusterreg::fit(native_source, native_target, native_options);
        }
        return result;
    }, "source"_a, "target"_a, "options"_a = clusterreg::Options{},
       "Fit ClusterReg. Inputs must be C-contiguous float64 arrays.");
    module.def("make_synthetic", [](const clusterreg::SyntheticOptions& options) {
        return clusterreg::make_synthetic(options);
    }, "options"_a = clusterreg::SyntheticOptions{});
    module.def("available_threads", &clusterreg::available_threads);
    module.def("cuda_compiled", &clusterreg::cuda_compiled);
    module.def("cuda_available", &clusterreg::cuda_available);
    module.def("cuda_device_name", &clusterreg::cuda_device_name);
}
