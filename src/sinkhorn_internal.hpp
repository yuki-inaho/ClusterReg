#pragma once

#include "clusterreg/clusterreg.hpp"

namespace clusterreg::detail {

// Avoid overflow in tau + eta and preserve the representable contraction.
// Callers validate positive, finite inputs and reject exponents rounded to 0/1.
inline double marginal_exponent(double penalty, double entropy) noexcept {
    if (penalty >= entropy) return 1.0 / (1.0 + entropy / penalty);
    const double ratio = penalty / entropy;
    return ratio / (1.0 + ratio);
}

struct TransportOutput {
    Statistics omega;
    Vector gamma_source_mass;
    Vector gamma_target_mass;
    Vector log_u;
    Vector log_v;
    double gamma_mass = 0;
    double omega_mass = 0;
    double transport_objective = 0;
    double cost_term = 0;
    double plan_kl = 0;
    double source_kl = 0;
    double target_kl = 0;
    double dual_residual = 0;
    double kkt_residual = 0;
    int iterations = 0;
};

TransportOutput sinkhorn_transport_cpu(const Matrix& target,
                                       const Matrix& transformed,
                                       double scale,
                                       const Options& options,
                                       const Vector& initial_log_u,
                                       const Vector& initial_log_v);

double fixed_plan_cost_cpu(const Matrix& target,
                           const Matrix& e_transformed,
                           double e_scale,
                           const Vector& log_u,
                           const Vector& log_v,
                           const Matrix& candidate_transformed,
                           double candidate_scale,
                           const Options& options);

TransportOutput sinkhorn_transport_cuda(const Matrix& target,
                                        const Matrix& transformed,
                                        double scale,
                                        const Options& options,
                                        const Vector& initial_log_u,
                                        const Vector& initial_log_v);

double fixed_plan_cost_cuda(const Matrix& target,
                            const Matrix& e_transformed,
                            double e_scale,
                            const Vector& log_u,
                            const Vector& log_v,
                            const Matrix& candidate_transformed,
                            double candidate_scale,
                            const Options& options);

Result fit_sinkhorn(const Matrix& source,
                    const Matrix& target,
                    const Options& options);

} // namespace clusterreg::detail
