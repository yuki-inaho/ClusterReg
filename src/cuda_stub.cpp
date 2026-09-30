// SPDX-License-Identifier: AGPL-3.0-only
#include "sinkhorn_internal.hpp"

#include <stdexcept>

namespace clusterreg {
bool cuda_compiled() { return false; }
bool cuda_available() { return false; }
std::string cuda_device_name() { return {}; }

namespace detail {
TransportOutput sinkhorn_transport_cuda(const Matrix&, const Matrix&, double,
                                        const Options&, const Vector&, const Vector&) {
    throw std::runtime_error("CUDA backend is not compiled into this build");
}

double fixed_plan_cost_cuda(const Matrix&, const Matrix&, double, const Vector&,
                            const Vector&, const Matrix&, double, const Options&) {
    throw std::runtime_error("CUDA backend is not compiled into this build");
}
} // namespace detail
} // namespace clusterreg
