// SPDX-License-Identifier: AGPL-3.0-only
#include "sinkhorn_internal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
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

#ifdef _OPENMP
int thread_count(int requested) {
    return std::max(1, requested);
}
#endif

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

struct OnlineLogSumExp {
    double maximum = -std::numeric_limits<double>::infinity();
    double shifted_sum = 0;

    void add(double value) {
        if (value <= maximum) {
            shifted_sum += std::exp(value - maximum);
        } else {
            shifted_sum = maximum == -std::numeric_limits<double>::infinity()
                              ? 1.0
                              : shifted_sum * std::exp(maximum - value) + 1.0;
            maximum = value;
        }
    }

    double value() const { return maximum + std::log(shifted_sum); }
};

double squared_distance(const Matrix& left, Index i, const Matrix& right, Index j) {
    double value = 0;
    for (Index axis = 0; axis < left.cols(); ++axis) {
        const double delta = left(i, axis) - right(j, axis);
        value += delta * delta;
    }
    return value;
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

double safe_exp(double value) {
    if (value > 700.0) throw std::runtime_error("Sinkhorn plan overflow; rescale input or increase eta");
    if (value < -745.0) return 0.0;
    return std::exp(value);
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

TransportOutput sinkhorn_transport_cpu(const Matrix& target, const Matrix& transformed,
                                       double scale, const Options& options,
                                       const Vector& initial_log_u,
                                       const Vector& initial_log_v) {
    const Index source_count = transformed.rows();
    const Index target_count = target.rows();
    const Index dimension = transformed.cols();
    const double log_b = -std::log(static_cast<double>(source_count));
    const double log_a = -std::log(static_cast<double>(target_count));
    const double log_ba = log_b + log_a;
    const double theta_y = options.source_mass_penalty /
                           (options.source_mass_penalty + options.transport_entropy);
    const double theta_x = options.target_mass_penalty /
                           (options.target_mass_penalty + options.transport_entropy);
    const double contraction = theta_y * theta_x;
    const CostModel cost(dimension, scale, options);
#ifdef _OPENMP
    const int threads = std::min<int>(thread_count(options.threads),
                                      static_cast<int>(std::max(source_count, target_count)));
#endif

    Vector log_u = initial_log_u.size() == source_count
                       ? initial_log_u
                       : Vector::Zero(source_count);
    Vector log_v = initial_log_v.size() == target_count
                       ? initial_log_v
                       : Vector::Zero(target_count);
    Vector next_u(source_count), next_v(target_count);
    double residual = std::numeric_limits<double>::infinity();
    int used_iterations = 0;
    for (int iteration = 0; iteration < options.sinkhorn_iterations; ++iteration) {
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (Index i = 0; i < source_count; ++i) {
            OnlineLogSumExp lse;
            for (Index j = 0; j < target_count; ++j) {
                const double value = log_ba -
                    cost(squared_distance(transformed, i, target, j)) /
                        options.transport_entropy + log_v[j];
                lse.add(value);
            }
            next_u[i] = theta_y * (log_b - lse.value());
        }
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(threads)
#endif
        for (Index j = 0; j < target_count; ++j) {
            OnlineLogSumExp lse;
            for (Index i = 0; i < source_count; ++i) {
                const double value = log_ba -
                    cost(squared_distance(transformed, i, target, j)) /
                        options.transport_entropy + next_u[i];
                lse.add(value);
            }
            next_v[j] = theta_x * (log_a - lse.value());
        }
        const double change = (next_v - log_v).cwiseAbs().maxCoeff();
        residual = change / (1.0 - contraction);
        log_u.swap(next_u);
        log_v.swap(next_v);
        used_iterations = iteration + 1;
        if (residual <= options.sinkhorn_tolerance) break;
    }

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
    Vector q_max = Vector::Constant(target_count,
        -std::numeric_limits<double>::infinity());
    Vector q_sum = Vector::Zero(target_count);

    long double gamma_mass = 0, omega_mass = 0, omega_sse = 0, omega_x2 = 0;
    long double cost_term = 0, plan_kl_core = 0;
    for (Index i = 0; i < source_count; ++i) {
        OnlineLogSumExp row_lse;
        for (Index j = 0; j < target_count; ++j) {
            const double distance = squared_distance(transformed, i, target, j);
            const double pair_cost = cost(distance);
            const double log_gamma = log_u[i] + log_ba -
                pair_cost / options.transport_entropy + log_v[j];
            row_lse.add(log_gamma);
            if (log_gamma <= q_max[j]) {
                q_sum[j] += std::exp(log_gamma - q_max[j]);
            } else {
                q_sum[j] = q_max[j] == -std::numeric_limits<double>::infinity()
                               ? 1.0
                               : q_sum[j] * std::exp(q_max[j] - log_gamma) + 1.0;
                q_max[j] = log_gamma;
            }
            const double gamma = safe_exp(log_gamma);
            const double omega = gamma * cost.robust_weight(distance, dimension);
            output.gamma_source_mass[i] += gamma;
            output.gamma_target_mass[j] += gamma;
            output.omega.mass[i] += omega;
            for (Index axis = 0; axis < dimension; ++axis)
                output.omega.px(i, axis) += omega * target(j, axis);
            gamma_mass += gamma;
            omega_mass += omega;
            omega_sse += static_cast<long double>(omega) * distance;
            omega_x2 += static_cast<long double>(omega) * target.row(j).squaredNorm();
            cost_term += static_cast<long double>(gamma) * pair_cost;
            if (gamma > 0)
                plan_kl_core += static_cast<long double>(gamma) * (log_gamma - log_ba);
        }
        log_p[i] = row_lse.value();
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

    double kkt = 0;
    for (Index i = 0; i < source_count; ++i) {
        for (Index j = 0; j < target_count; ++j) {
            const double value = options.transport_entropy * (log_u[i] + log_v[j]) +
                options.source_mass_penalty * (log_p[i] - log_b) +
                options.target_mass_penalty * (log_q[j] - log_a);
            kkt = std::max(kkt, std::abs(value));
        }
    }
    output.kkt_residual = kkt;
    return output;
}

double fixed_plan_cost_cpu(const Matrix& target, const Matrix& e_transformed,
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
    long double value = 0;
    for (Index i = 0; i < source_count; ++i) {
        for (Index j = 0; j < target_count; ++j) {
            const double old_distance = squared_distance(e_transformed, i, target, j);
            const double log_gamma = log_u[i] + log_ba -
                old_cost(old_distance) / options.transport_entropy + log_v[j];
            const double gamma = safe_exp(log_gamma);
            if (gamma > 0) {
                const double new_distance = squared_distance(candidate_transformed, i, target, j);
                value += static_cast<long double>(gamma) * candidate_cost(new_distance);
            }
        }
    }
    return static_cast<double>(value);
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
        const double candidate_cost = fixed_cost(
            x, transformed, scale, transport.log_u, transport.log_v,
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
