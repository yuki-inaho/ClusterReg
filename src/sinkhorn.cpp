// SPDX-License-Identifier: AGPL-3.0-only
#include "sinkhorn_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace clusterreg::detail {
namespace {
using Clock = std::chrono::steady_clock;

double elapsed(Clock::time_point start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

void require(bool condition, const std::string& message) {
    if (!condition) throw std::invalid_argument(message);
}

int thread_count(int requested) {
#ifdef _OPENMP
    return std::max(1, requested);
#else
    (void)requested;
    return 1;
#endif
}

struct CostModel {
    NoiseModel noise;
    double scale;
    double normalizer;
    double student_dof;
    double student_factor;

    CostModel(Index dimension, double value, const Options& options)
        : noise(options.noise_model), scale(value), student_dof(options.student_dof) {
        if (noise == NoiseModel::Gaussian) {
            normalizer = 0.5 * static_cast<double>(dimension) *
                         std::log(2.0 * std::acos(-1.0) * scale);
            student_factor = 0;
        } else {
            const double nu = student_dof;
            normalizer = std::lgamma(nu / 2.0) -
                         std::lgamma((nu + static_cast<double>(dimension)) / 2.0) +
                         0.5 * static_cast<double>(dimension) *
                             std::log(nu * std::acos(-1.0) * scale);
            student_factor = 0.5 * (nu + static_cast<double>(dimension));
        }
    }

    double operator()(double squared_distance) const {
        if (noise == NoiseModel::Gaussian)
            return normalizer + squared_distance / (2.0 * scale);
        return normalizer +
               student_factor * std::log1p(squared_distance / (student_dof * scale));
    }

    double robust_weight(double squared_distance, Index dimension) const {
        if (noise == NoiseModel::Gaussian) return 1.0;
        return (student_dof + static_cast<double>(dimension)) /
               (student_dof + squared_distance / scale);
    }
};

struct CpuScratch {
    Vector distance, costs, values, weights, auxiliary;
    explicit CpuScratch(Index count)
        : distance(count), costs(count), values(count), weights(count), auxiliary(count) {}
};

struct TransportWorker : CpuScratch {
    Vector target_mass, target_max, target_sum;
    long double gamma_mass = 0, omega_mass = 0, omega_sse = 0, omega_x2 = 0;
    long double cost_term = 0, plan_kl_core = 0;
    TransportWorker(Index count, Index target_count)
        : CpuScratch(count), target_mass(Vector::Zero(target_count)),
          target_max(Vector::Constant(target_count,
              -std::numeric_limits<double>::infinity())),
          target_sum(Vector::Zero(target_count)) {}
};

void distance_vector(const Matrix& points, const Matrix& fixed, Index row,
                     CpuScratch& scratch) {
    const Index count = points.rows();
    scratch.distance.head(count).setZero();
    // Direct differences preserve stability for a large common offset. Each
    // coordinate column is contiguous, allowing Eigen's double packets.
    for (Index axis = 0; axis < points.cols(); ++axis)
        scratch.distance.head(count).array() +=
            (points.col(axis).array() - fixed(row, axis)).square();
}

template<NoiseModel Model>
void cost_vector(const CostModel& cost, Index count, CpuScratch& scratch) {
    if constexpr (Model == NoiseModel::Gaussian) {
        scratch.costs.head(count).array() = cost.normalizer +
            scratch.distance.head(count).array() / (2.0 * cost.scale);
    } else {
        scratch.values.head(count).array() = scratch.distance.head(count).array() /
            (cost.student_dof * cost.scale);
        scratch.auxiliary.head(count).array() = 1.0 + scratch.values.head(count).array();
        // Kahan's log1p formula also used by Eigen's generic_plog1p. A safe
        // denominator prevents 0/0 before the small-argument scalar correction.
        scratch.weights.head(count).array() =
            (scratch.auxiliary.head(count).array() == 1.0).select(
                1.0, scratch.auxiliary.head(count).array() - 1.0);
        scratch.costs.head(count).array() = cost.normalizer + cost.student_factor *
            (scratch.values.head(count).array() *
             (scratch.auxiliary.head(count).array().log() /
              scratch.weights.head(count).array()));
        for (Index j = 0; j < count; ++j) {
            const double x = scratch.values[j];
            if (x < 1e-4 || !std::isfinite(x))
                scratch.costs[j] = cost.normalizer + cost.student_factor * std::log1p(x);
        }
    }
}

// Eigen packet exp clamps its negative tail; restore scalar libm there. The
// plan path retains its existing explicit cutoff, while LSE keeps subnormals.
void vector_exp(const Vector& values, Index count, Vector& output, bool plan) {
    output.head(count).array() = values.head(count).array().max(-700.0).exp();
    for (Index j = 0; j < count; ++j)
        if (values[j] < -700.0)
            output[j] = plan && values[j] < -745.0 ? 0.0 : std::exp(values[j]);
}

double logsumexp(Index count, CpuScratch& scratch, int& invalid) {
    const double maximum = scratch.values.head(count).maxCoeff();
    if (!std::isfinite(maximum)) {
        invalid = 1;
        return 0.0;
    }
    scratch.auxiliary.head(count).array() = scratch.values.head(count).array() - maximum;
    vector_exp(scratch.auxiliary, count, scratch.weights, false);
    const double sum = scratch.weights.head(count).sum();
    if (!(sum > 0.0) || !std::isfinite(sum)) {
        invalid = 1;
        return 0.0;
    }
    return maximum + std::log(sum);
}

void add_log_value(double value, double& maximum, double& shifted_sum) {
    if (value == -std::numeric_limits<double>::infinity()) return;
    if (value <= maximum) shifted_sum += std::exp(value - maximum);
    else {
        shifted_sum = shifted_sum == 0.0
            ? 1.0 : shifted_sum * std::exp(maximum - value) + 1.0;
        maximum = value;
    }
}

double generalized_kl(const Vector& value, double log_reference) {
    long double result = 1.0L; // reference masses sum to one
    for (Index i = 0; i < value.size(); ++i) {
        const double x = value[i];
        if (x > 0) result += static_cast<long double>(x) *
                             (std::log(x) - log_reference) - x;
    }
    return static_cast<double>(result);
}

void common_prepare(Result& result, const Matrix& source, const Matrix& target,
                    const Options& options) {
    result.prepared_source = source;
    result.prepared_target = target;
    const Index dimension = source.cols();
    result.source_center = Eigen::RowVectorXd::Zero(dimension);
    result.target_center = Eigen::RowVectorXd::Zero(dimension);
    result.source_scale = 1.0;
    result.target_scale = 1.0;
    if (options.normalize) {
        const double count = static_cast<double>(source.rows() + target.rows());
        Eigen::RowVectorXd center =
            (source.colwise().sum() + target.colwise().sum()) / count;
        long double squared = 0;
        for (Index i = 0; i < source.rows(); ++i)
            squared += (source.row(i) - center).squaredNorm();
        for (Index i = 0; i < target.rows(); ++i)
            squared += (target.row(i) - center).squaredNorm();
        double scale = std::sqrt(static_cast<double>(squared / count));
        if (!(scale > 0)) scale = 1.0;
        require(std::isfinite(scale), "Common normalization scale overflow; rescale input first");
        result.source_center = center;
        result.target_center = center;
        result.source_scale = scale;
        result.target_scale = scale;
        result.prepared_source.rowwise() -= center;
        result.prepared_target.rowwise() -= center;
        result.prepared_source /= scale;
        result.prepared_target /= scale;
    }
    result.source_shift = Eigen::RowVectorXd::Zero(dimension);
    if (options.align_centroids) {
        result.source_shift = result.prepared_target.colwise().mean() -
                              result.prepared_source.colwise().mean();
        result.prepared_source.rowwise() += result.source_shift;
    }
}

double initial_scale(const Matrix& source, const Matrix& target, const Options& options) {
    if (options.initial_sigma > 0)
        return std::clamp(options.initial_sigma, options.sigma_floor,
                          options.sigma_ceiling);
    const double pairwise_mean =
        source.squaredNorm() / static_cast<double>(source.rows()) +
        target.squaredNorm() / static_cast<double>(target.rows()) -
        2.0 * source.colwise().mean().dot(target.colwise().mean());
    const double raw = pairwise_mean / static_cast<double>(source.cols());
    require(std::isfinite(raw), "Initial Sinkhorn scale overflow; rescale input first");
    return std::clamp(raw, options.sigma_floor, options.sigma_ceiling);
}

TransportOutput run_transport(const Matrix& target, const Matrix& transformed,
                              double scale, const Options& options,
                              const Vector& log_u, const Vector& log_v,
                              Backend backend) {
    if (backend == Backend::CUDA)
        return sinkhorn_transport_cuda(target, transformed, scale, options, log_u, log_v);
    return sinkhorn_transport_cpu(target, transformed, scale, options, log_u, log_v);
}

double fixed_cost(const Matrix& target, const Matrix& e_transformed, double e_scale,
                  const Vector& log_u, const Vector& log_v,
                  const Matrix& candidate, double candidate_scale,
                  const Options& options, Backend backend) {
    if (backend == Backend::CUDA)
        return fixed_plan_cost_cuda(target, e_transformed, e_scale, log_u, log_v,
                                    candidate, candidate_scale, options);
    return fixed_plan_cost_cpu(target, e_transformed, e_scale, log_u, log_v,
                               candidate, candidate_scale, options);
}
} // namespace

template<NoiseModel Model>
TransportOutput transport_cpu_impl(const Matrix& target, const Matrix& transformed,
                                       double scale, const Options& options,
                                       const Vector& initial_log_u,
                                       const Vector& initial_log_v) {
    const Index source_count = transformed.rows();
    const Index target_count = target.rows();
    const Index dimension = transformed.cols();
    const double log_b = -std::log(static_cast<double>(source_count));
    const double log_a = -std::log(static_cast<double>(target_count));
    const double log_ba = log_b + log_a;
    const double theta_y = marginal_exponent(options.source_mass_penalty,
                                             options.transport_entropy);
    const double theta_x = marginal_exponent(options.target_mass_penalty,
                                             options.transport_entropy);
    const double contraction = theta_y * theta_x;
    const CostModel cost(dimension, scale, options);
    const int threads = std::min<int>(thread_count(options.threads),
                                      static_cast<int>(std::max(source_count, target_count)));
    std::vector<TransportWorker> workers;
    workers.reserve(static_cast<std::size_t>(threads));
    for (int id = 0; id < threads; ++id)
        workers.emplace_back(std::max(source_count, target_count), target_count);

    Vector log_u = initial_log_u.size() == source_count
                       ? initial_log_u
                       : Vector::Zero(source_count);
    Vector log_v = initial_log_v.size() == target_count
                       ? initial_log_v
                       : Vector::Zero(target_count);
    Vector next_u(source_count), next_v(target_count);
    double residual = std::numeric_limits<double>::infinity();
    int used_iterations = 0;
    int invalid = 0;
    bool finished = false;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads)
#endif
    {
        int id = 0;
#ifdef _OPENMP
        id = omp_get_thread_num();
#endif
        auto& scratch = workers[static_cast<std::size_t>(id)];
        // Reuse one OpenMP team and its scratch through all dual sweeps.
        for (int iteration = 0; iteration < options.sinkhorn_iterations; ++iteration) {
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:invalid)
#endif
            for (Index i = 0; i < source_count; ++i) {
                distance_vector(target, transformed, i, scratch);
                cost_vector<Model>(cost, target_count, scratch);
                scratch.values.head(target_count) =
                    (log_ba - scratch.costs.head(target_count).array() /
                     options.transport_entropy + log_v.array()).matrix();
                next_u[i] = theta_y * (log_b - logsumexp(target_count, scratch, invalid));
            }
#ifdef _OPENMP
#pragma omp for schedule(static) reduction(|:invalid)
#endif
            for (Index j = 0; j < target_count; ++j) {
                distance_vector(transformed, target, j, scratch);
                cost_vector<Model>(cost, source_count, scratch);
                scratch.values.head(source_count) =
                    (log_ba - scratch.costs.head(source_count).array() /
                     options.transport_entropy + next_u.array()).matrix();
                next_v[j] = theta_x * (log_a - logsumexp(source_count, scratch, invalid));
            }
#ifdef _OPENMP
#pragma omp single
#endif
            {
                const double change = (next_v - log_v).cwiseAbs().maxCoeff();
                residual = change / (1.0 - contraction);
                log_u.swap(next_u);
                log_v.swap(next_v);
                used_iterations = iteration + 1;
                finished = invalid || residual <= options.sinkhorn_tolerance;
            }
            if (finished) break;
        }
    }
    if (invalid) throw std::runtime_error("Nonfinite Sinkhorn log probabilities; rescale input");

    TransportOutput output;
    output.log_u = log_u;
    output.log_v = log_v;
    output.gamma_source_mass = Vector::Zero(source_count);
    output.gamma_target_mass = Vector::Zero(target_count);
    output.omega.mass = Vector::Zero(source_count);
    output.omega.px = Matrix::Zero(source_count, dimension);
    output.dual_residual = residual;
    output.iterations = used_iterations;
    Vector log_p(source_count);
    const Vector target_squared_norm = target.rowwise().squaredNorm();
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) reduction(|:invalid)
#endif
    {
        int id = 0;
#ifdef _OPENMP
        id = omp_get_thread_num();
#endif
        auto& worker = workers[static_cast<std::size_t>(id)];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (Index i = 0; i < source_count; ++i) {
            distance_vector(target, transformed, i, worker);
            cost_vector<Model>(cost, target_count, worker);
            worker.values.head(target_count) =
                (log_u[i] + log_ba - worker.costs.head(target_count).array() /
                 options.transport_entropy + log_v.array()).matrix();
            log_p[i] = logsumexp(target_count, worker, invalid);
            if (worker.values.head(target_count).maxCoeff() > 700.0) {
                invalid = 1;
                continue;
            }
            vector_exp(worker.values, target_count, worker.weights, true);
            if constexpr (Model == NoiseModel::Gaussian)
                worker.auxiliary.head(target_count) = worker.weights.head(target_count);
            else
                worker.auxiliary.head(target_count).array() =
                    worker.weights.head(target_count).array() *
                    ((cost.student_dof + static_cast<double>(dimension)) /
                     (cost.student_dof + worker.distance.head(target_count).array() / scale));
            output.gamma_source_mass[i] = worker.weights.head(target_count).sum();
            output.omega.mass[i] = worker.auxiliary.head(target_count).sum();
            worker.target_mass += worker.weights.head(target_count);
            for (Index axis = 0; axis < dimension; ++axis)
                output.omega.px(i, axis) =
                    worker.auxiliary.head(target_count).dot(target.col(axis));
            for (Index j = 0; j < target_count; ++j) {
                const double gamma = worker.weights[j], omega = worker.auxiliary[j];
                add_log_value(worker.values[j], worker.target_max[j], worker.target_sum[j]);
                worker.gamma_mass += gamma;
                worker.omega_mass += omega;
                worker.omega_sse += static_cast<long double>(omega) * worker.distance[j];
                worker.omega_x2 += static_cast<long double>(omega) * target_squared_norm[j];
                worker.cost_term += static_cast<long double>(gamma) * worker.costs[j];
                if (gamma > 0)
                    worker.plan_kl_core += static_cast<long double>(gamma) *
                                           (worker.values[j] - log_ba);
            }
        }
    }
    if (invalid) throw std::runtime_error("Sinkhorn plan overflow or invalid log probabilities; rescale input or increase eta");
    Vector q_max = Vector::Constant(target_count, -std::numeric_limits<double>::infinity());
    Vector q_sum = Vector::Zero(target_count);
    long double gamma_mass = 0, omega_mass = 0, omega_sse = 0, omega_x2 = 0;
    long double cost_term = 0, plan_kl_core = 0;
    // Ordered worker reduction; target log marginals remain valid when their
    // representable plan mass is zero, so KKT diagnostics do not take log(0).
    for (const auto& worker : workers) {
        output.gamma_target_mass += worker.target_mass;
        gamma_mass += worker.gamma_mass;
        omega_mass += worker.omega_mass;
        omega_sse += worker.omega_sse;
        omega_x2 += worker.omega_x2;
        cost_term += worker.cost_term;
        plan_kl_core += worker.plan_kl_core;
        for (Index j = 0; j < target_count; ++j) {
            if (worker.target_sum[j] == 0.0) continue;
            const double maximum = std::max(q_max[j], worker.target_max[j]);
            q_sum[j] = q_sum[j] * std::exp(q_max[j] - maximum) +
                worker.target_sum[j] * std::exp(worker.target_max[j] - maximum);
            q_max[j] = maximum;
        }
    }
    Vector log_q = q_max + q_sum.array().log().matrix();
    output.gamma_mass = static_cast<double>(gamma_mass);
    output.omega_mass = static_cast<double>(omega_mass);
    output.omega.sse_old = static_cast<double>(omega_sse);
    output.omega.weighted_x2 = static_cast<double>(omega_x2);
    output.cost_term = static_cast<double>(cost_term);
    output.plan_kl = static_cast<double>(plan_kl_core - gamma_mass + 1.0L);
    output.source_kl = generalized_kl(output.gamma_source_mass, log_b);
    output.target_kl = generalized_kl(output.gamma_target_mass, log_a);
    output.transport_objective = output.cost_term +
        options.transport_entropy * output.plan_kl +
        options.source_mass_penalty * output.source_kl +
        options.target_mass_penalty * output.target_kl;

    const Vector row_stationarity = options.transport_entropy * log_u.array() +
        options.source_mass_penalty * (log_p.array() - log_b);
    const Vector column_stationarity = options.transport_entropy * log_v.array() +
        options.target_mass_penalty * (log_q.array() - log_a);
    output.kkt_residual = std::max(
        std::abs(row_stationarity.minCoeff() + column_stationarity.minCoeff()),
        std::abs(row_stationarity.maxCoeff() + column_stationarity.maxCoeff()));
    return output;
}

TransportOutput sinkhorn_transport_cpu(const Matrix& target, const Matrix& transformed,
                                       double scale, const Options& options,
                                       const Vector& initial_log_u,
                                       const Vector& initial_log_v) {
    if (options.noise_model == NoiseModel::Gaussian)
        return transport_cpu_impl<NoiseModel::Gaussian>(target, transformed, scale,
            options, initial_log_u, initial_log_v);
    return transport_cpu_impl<NoiseModel::StudentT>(target, transformed, scale,
        options, initial_log_u, initial_log_v);
}

template<NoiseModel Model>
double fixed_cost_cpu_impl(const Matrix& target, const Matrix& e_transformed,
                           double e_scale, const Vector& log_u,
                           const Vector& log_v, const Matrix& candidate_transformed,
                           double candidate_scale, const Options& options) {
    const Index source_count = e_transformed.rows();
    const Index target_count = target.rows();
    const Index dimension = target.cols();
    const double log_ba = -std::log(static_cast<double>(source_count)) -
                          std::log(static_cast<double>(target_count));
    const CostModel old_cost(dimension, e_scale, options);
    const CostModel candidate_cost(dimension, candidate_scale, options);
    const int threads = std::min<int>(thread_count(options.threads), static_cast<int>(source_count));
    std::vector<CpuScratch> workers;
    std::vector<Vector> plans;
    workers.reserve(static_cast<std::size_t>(threads));
    plans.reserve(static_cast<std::size_t>(threads));
    for (int id = 0; id < threads; ++id) {
        workers.emplace_back(target_count);
        plans.emplace_back(target_count);
    }
    std::vector<long double> partial(static_cast<std::size_t>(threads), 0.0L);
    int invalid = 0;
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) reduction(|:invalid)
#endif
    {
        int id = 0;
#ifdef _OPENMP
        id = omp_get_thread_num();
#endif
        auto& scratch = workers[static_cast<std::size_t>(id)];
        auto& gamma = plans[static_cast<std::size_t>(id)];
        long double& value = partial[static_cast<std::size_t>(id)];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for (Index i = 0; i < source_count; ++i) {
            distance_vector(target, e_transformed, i, scratch);
            cost_vector<Model>(old_cost, target_count, scratch);
            scratch.values.head(target_count) =
                (log_u[i] + log_ba - scratch.costs.head(target_count).array() /
                 options.transport_entropy + log_v.array()).matrix();
            if (scratch.values.head(target_count).maxCoeff() > 700.0) {
                invalid = 1;
                continue;
            }
            vector_exp(scratch.values, target_count, scratch.weights, true);
            if (!scratch.weights.head(target_count).allFinite()) {
                invalid = 1;
                continue;
            }
            // Save old-plan weights before Student-t cost uses scratch.weights.
            gamma = scratch.weights.head(target_count);
            distance_vector(target, candidate_transformed, i, scratch);
            cost_vector<Model>(candidate_cost, target_count, scratch);
            for (Index j = 0; j < target_count; ++j)
                if (gamma[j] > 0)
                    value += static_cast<long double>(gamma[j]) * scratch.costs[j];
        }
    }
    if (invalid) throw std::runtime_error("Sinkhorn plan overflow; rescale input or increase eta");
    long double value = 0;
    for (long double item : partial) value += item;
    return static_cast<double>(value);
}

double fixed_plan_cost_cpu(const Matrix& target, const Matrix& e_transformed,
                           double e_scale, const Vector& log_u, const Vector& log_v,
                           const Matrix& candidate_transformed, double candidate_scale,
                           const Options& options) {
    if (options.noise_model == NoiseModel::Gaussian)
        return fixed_cost_cpu_impl<NoiseModel::Gaussian>(target, e_transformed,
            e_scale, log_u, log_v, candidate_transformed, candidate_scale, options);
    return fixed_cost_cpu_impl<NoiseModel::StudentT>(target, e_transformed,
        e_scale, log_u, log_v, candidate_transformed, candidate_scale, options);
}

Result fit_sinkhorn(const Matrix& source, const Matrix& target, const Options& options) {
    const auto start = Clock::now();
    require(source.rows() > 0 && source.cols() > 0 && target.rows() > 0,
            "Point set must be nonempty");
    require(source.cols() == target.cols(), "Source and target dimensions differ");
    require(source.allFinite() && target.allFinite(), "Point set contains NaN or infinity");
    require(options.transport_entropy > 0 && std::isfinite(options.transport_entropy),
            "transport_entropy must be finite and positive");
    require(options.source_mass_penalty > 0 && std::isfinite(options.source_mass_penalty) &&
                options.target_mass_penalty > 0 && std::isfinite(options.target_mass_penalty),
            "Sinkhorn marginal penalties must be finite and positive");
    const double theta_y = marginal_exponent(options.source_mass_penalty,
                                             options.transport_entropy);
    const double theta_x = marginal_exponent(options.target_mass_penalty,
                                             options.transport_entropy);
    require(theta_y > 0.0 && theta_y < 1.0 && theta_x > 0.0 && theta_x < 1.0,
            "Ill-conditioned Sinkhorn theta/contraction rounds to zero or one");
    require(options.regularization > 0 && std::isfinite(options.regularization),
            "regularization must be finite and positive");
    require(options.gamma > 0 && std::isfinite(options.gamma) &&
                options.rank >= 0 && options.rank_ratio > 0 &&
                options.rank_ratio <= 1,
            "Invalid kernel or rank options");
    require(options.eigen_cutoff >= 0 && std::isfinite(options.eigen_cutoff) &&
                options.kmeans_iterations > 0,
            "Invalid Sinkhorn basis options");
    require(options.sigma_floor > 0 && options.sigma_ceiling >= options.sigma_floor &&
                std::isfinite(options.sigma_ceiling),
            "Invalid Sinkhorn scale interval");
    require(options.initial_sigma >= 0 && std::isfinite(options.initial_sigma),
            "initial_sigma must be finite and nonnegative");
    require(options.sinkhorn_tolerance >= 0 && std::isfinite(options.sinkhorn_tolerance) &&
                options.sinkhorn_iterations > 0 && options.transport_mass_floor >= 0 &&
                std::isfinite(options.transport_mass_floor),
            "Invalid Sinkhorn stopping options");
    require(options.tolerance >= 0 && std::isfinite(options.tolerance) &&
                options.max_iterations > 0 && options.threads > 0,
            "Invalid outer stopping options");
    if (options.noise_model == NoiseModel::StudentT)
        require(options.student_dof > 0 && std::isfinite(options.student_dof),
                "student_dof must be finite and positive");

    Backend backend = options.backend;
    if (backend == Backend::Auto) backend = cuda_available() ? Backend::CUDA : Backend::CPU;
    if (backend == Backend::CUDA && !cuda_available())
        throw std::runtime_error(cuda_compiled()
            ? "CUDA backend requested, but no CUDA device is available"
            : "CUDA backend requested, but this build has no CUDA support");

    Result result;
    result.options = options;
    result.backend_used = backend;
    common_prepare(result, source, target, options);
    Matrix& y = result.prepared_source;
    Matrix& x = result.prepared_target;
    const Index source_count = y.rows();
    const Index dimension = y.cols();
    Matrix kernel;
    if (options.solver == Solver::Nystrom) result.basis = make_basis(y, options);
    else kernel = laplacian_kernel(y, y, options.gamma);
    result.preparation_seconds = elapsed(start);

    double scale = initial_scale(y, x, options);
    Matrix transformed = y;
    result.coefficients = Matrix::Zero(
        options.solver == Solver::Nystrom ? result.basis.Q.cols() : source_count,
        dimension);
    double penalty = 0;
    double previous_objective = std::numeric_limits<double>::infinity();
    Vector log_u, log_v;
    TransportOutput last_transport;
    result.stop_reason = "max_iterations";
    const auto loop_start = Clock::now();
    for (int iteration = 0; iteration < options.max_iterations; ++iteration) {
        Iteration record;
        record.index = iteration + 1;
        record.sigma_before = scale;
        const auto e_start = Clock::now();
        TransportOutput transport = run_transport(x, transformed, scale, options,
                                                  log_u, log_v, backend);
        record.estep_seconds = elapsed(e_start);
        if (!(transport.gamma_mass >= options.transport_mass_floor) ||
            !std::isfinite(transport.gamma_mass)) {
            result.stop_reason = "mass_collapse";
            last_transport = std::move(transport);
            break;
        }
        record.transport_iterations = transport.iterations;
        record.transport_residual = transport.dual_residual;
        record.kkt_residual = transport.kkt_residual;
        record.transport_mass = transport.gamma_mass;
        record.robust_mass = transport.omega_mass;
        record.objective_before = transport.transport_objective +
                                  0.5 * options.regularization * penalty;

        const auto solve_start = Clock::now();
        Update update = options.solver == Solver::Nystrom
            ? reduced_update(y, result.basis.Q, transport.omega,
                             options.regularization * scale)
            : dense_update(y, kernel, transport.omega,
                           options.regularization * scale);
        const double residual_sum = updated_sse(transport.omega, transformed,
                                                update.transformed);
        const double candidate_scale = std::clamp(
            residual_sum / (static_cast<double>(dimension) * transport.gamma_mass),
            options.sigma_floor, options.sigma_ceiling);
        record.solve_seconds = elapsed(solve_start);
        const double candidate_cost = options.noise_model == NoiseModel::Gaussian
            ? transport.gamma_mass * CostModel(dimension, candidate_scale, options).normalizer +
                  residual_sum / (2.0 * candidate_scale)
            : fixed_cost(x, transformed, scale, transport.log_u, transport.log_v,
                         update.transformed, candidate_scale, options, backend);
        record.objective_after = candidate_cost +
            options.transport_entropy * transport.plan_kl +
            options.source_mass_penalty * transport.source_kl +
            options.target_mass_penalty * transport.target_kl +
            0.5 * options.regularization * update.penalty;
        record.sigma_after = candidate_scale;
        record.linear_residual = update.residual;
        record.step_rms = (update.transformed - transformed).norm() /
                          std::sqrt(static_cast<double>(source_count));
        record.scale_relative_change = std::abs(candidate_scale - scale) /
                                       std::max(scale, options.sigma_floor);
        record.alpha_sum = transport.gamma_mass;

        const double acceptance = 2e-9 * (1.0 + std::abs(record.objective_before));
        if (record.objective_after > record.objective_before + acceptance) {
            result.stop_reason = "non_descent";
            result.history.push_back(record);
            last_transport = std::move(transport);
            break;
        }

        const double objective_change = std::isfinite(previous_objective)
            ? std::abs(record.objective_after - previous_objective) /
                  (1.0 + std::abs(previous_objective))
            : std::numeric_limits<double>::infinity();
        transformed.swap(update.transformed);
        result.coefficients.swap(update.coefficients);
        penalty = update.penalty;
        scale = candidate_scale;
        previous_objective = record.objective_after;
        log_u = transport.log_u;
        log_v = transport.log_v;
        last_transport = std::move(transport);
        result.history.push_back(record);
        if (!options.fixed_iterations && objective_change <= options.tolerance &&
            record.step_rms <= options.tolerance &&
            record.scale_relative_change <= options.tolerance &&
            record.transport_residual <= options.sinkhorn_tolerance) {
            result.stop_reason = "tolerance";
            break;
        }
    }
    result.iteration_seconds = elapsed(loop_start);

    // Return marginals that are optimal for the returned transform, not the
    // E-step immediately before the last accepted M-step.
    last_transport = run_transport(x, transformed, scale, options, log_u, log_v, backend);
    result.source_mass = last_transport.gamma_source_mass;
    result.target_mass = last_transport.gamma_target_mass;
    result.alpha = result.source_mass; // compatibility alias; not a probability simplex
    result.transport_mass = last_transport.gamma_mass;
    result.sigma2 = scale;
    result.normalized_transformed = transformed;
    result.transformed = transformed * result.target_scale;
    result.transformed.rowwise() += result.target_center;
    result.total_seconds = elapsed(start);
    return result;
}

} // namespace clusterreg::detail
