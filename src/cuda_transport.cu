// SPDX-License-Identifier: AGPL-3.0-only
#include "sinkhorn_internal.hpp"

#include <cuda_runtime.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace clusterreg {
namespace {

constexpr int kBlockSize = 256;
constexpr int kMaximumGridSize = 65535;

[[noreturn]] void throw_cuda_error(cudaError_t error, const char* operation) {
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(error);
    throw std::runtime_error(message.str());
}

void check_cuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) throw_cuda_error(error, operation);
}

void check_launch(const char* kernel) {
    check_cuda(cudaGetLastError(), kernel);
}

std::size_t checked_size(Index count, const char* description) {
    if (count < 0 || static_cast<unsigned long long>(count) >
                         static_cast<unsigned long long>(
                             std::numeric_limits<std::size_t>::max())) {
        throw std::overflow_error(std::string(description) + " is too large");
    }
    return static_cast<std::size_t>(count);
}

std::size_t checked_product(Index left, Index right, const char* description) {
    const std::size_t a = checked_size(left, description);
    const std::size_t b = checked_size(right, description);
    if (b != 0 && a > std::numeric_limits<std::size_t>::max() / b)
        throw std::overflow_error(std::string(description) + " is too large");
    return a * b;
}

template <typename T>
class DeviceBuffer {
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t count) : count_(count) {
        if (count_ > std::numeric_limits<std::size_t>::max() / sizeof(T))
            throw std::overflow_error("CUDA allocation is too large");
        if (count_ != 0)
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), count_ * sizeof(T)),
                       "cudaMalloc");
    }

    ~DeviceBuffer() {
        if (data_ != nullptr) (void)cudaFree(data_);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : data_(other.data_), count_(other.count_) {
        other.data_ = nullptr;
        other.count_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (data_ != nullptr) (void)cudaFree(data_);
            data_ = other.data_;
            count_ = other.count_;
            other.data_ = nullptr;
            other.count_ = 0;
        }
        return *this;
    }

    T* get() { return data_; }
    const T* get() const { return data_; }
    std::size_t size() const { return count_; }

    void copy_from_host(const T* source, std::size_t count) {
        if (count > count_) throw std::logic_error("CUDA host-to-device copy is too large");
        if (count != 0)
            check_cuda(cudaMemcpy(data_, source, count * sizeof(T),
                                  cudaMemcpyHostToDevice),
                       "cudaMemcpy host to device");
    }

    void copy_to_host(T* destination, std::size_t count) const {
        if (count > count_) throw std::logic_error("CUDA device-to-host copy is too large");
        if (count != 0)
            check_cuda(cudaMemcpy(destination, data_, count * sizeof(T),
                                  cudaMemcpyDeviceToHost),
                       "cudaMemcpy device to host");
    }

    void zero() {
        if (count_ != 0)
            check_cuda(cudaMemset(data_, 0, count_ * sizeof(T)), "cudaMemset");
    }

    void swap(DeviceBuffer& other) noexcept {
        using std::swap;
        swap(data_, other.data_);
        swap(count_, other.count_);
    }

private:
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

int grid_size(std::int64_t work_items) {
    return static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(work_items, kMaximumGridSize)));
}

struct DeviceCostModel {
    int student;
    std::int64_t dimension;
    double scale;
    double normalizer;
    double student_dof;
    double student_factor;
};

DeviceCostModel make_cost_model(Index dimension, double scale,
                                const Options& options) {
    DeviceCostModel model{};
    model.student = options.noise_model == NoiseModel::StudentT ? 1 : 0;
    model.dimension = static_cast<std::int64_t>(dimension);
    model.scale = scale;
    model.student_dof = options.student_dof;
    const double d = static_cast<double>(dimension);
    const double pi = std::acos(-1.0);
    if (model.student == 0) {
        model.normalizer = 0.5 * d * std::log(2.0 * pi * scale);
        model.student_factor = 0.0;
    } else {
        const double nu = options.student_dof;
        model.normalizer = std::lgamma(nu / 2.0) -
                           std::lgamma((nu + d) / 2.0) +
                           0.5 * d * std::log(nu * pi * scale);
        model.student_factor = 0.5 * (nu + d);
    }
    return model;
}

double generalized_kl(const Vector& value, double log_reference) {
    long double result = 1.0L;
    for (Index i = 0; i < value.size(); ++i) {
        const double x = value[i];
        if (x > 0.0)
            result += static_cast<long double>(x) *
                          (std::log(x) - log_reference) -
                      x;
    }
    return static_cast<double>(result);
}

__device__ double block_max(double value, double* scratch) {
    const int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset)
            scratch[thread] = fmax(scratch[thread], scratch[thread + offset]);
        __syncthreads();
    }
    const double result = scratch[0];
    __syncthreads();
    return result;
}

__device__ double block_sum(double value, double* scratch) {
    const int thread = threadIdx.x;
    scratch[thread] = value;
    __syncthreads();
    for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
        if (thread < offset) scratch[thread] += scratch[thread + offset];
        __syncthreads();
    }
    const double result = scratch[0];
    __syncthreads();
    return result;
}

__device__ double squared_distance(const double* left, std::int64_t left_count,
                                   std::int64_t left_index,
                                   const double* right,
                                   std::int64_t right_count,
                                   std::int64_t right_index,
                                   std::int64_t dimension) {
    double value = 0.0;
    for (std::int64_t axis = 0; axis < dimension; ++axis) {
        const double delta = left[left_index + left_count * axis] -
                             right[right_index + right_count * axis];
        value += delta * delta;
    }
    return value;
}

__device__ double pair_cost(double squared, const DeviceCostModel& model) {
    if (model.student == 0)
        return model.normalizer + squared / (2.0 * model.scale);
    return model.normalizer +
           model.student_factor *
               log1p(squared / (model.student_dof * model.scale));
}

__device__ double robust_weight(double squared, const DeviceCostModel& model) {
    if (model.student == 0) return 1.0;
    return (model.student_dof + static_cast<double>(model.dimension)) /
           (model.student_dof + squared / model.scale);
}

__device__ double checked_plan_exp(double log_gamma, int* overflow) {
    if (log_gamma > 700.0) {
        atomicExch(overflow, 1);
        return 0.0;
    }
    if (log_gamma < -745.0) return 0.0;
    return exp(log_gamma);
}

__global__ void update_source_dual_kernel(
    const double* target, std::int64_t target_count,
    const double* transformed, std::int64_t source_count,
    DeviceCostModel cost, double log_b, double log_ba, double eta,
    double theta_y, const double* log_v, double* next_u) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t i = blockIdx.x; i < source_count; i += gridDim.x) {
        double local_maximum = -CUDART_INF;
        for (std::int64_t j = threadIdx.x; j < target_count; j += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double value =
                log_ba - pair_cost(distance, cost) / eta + log_v[j];
            local_maximum = fmax(local_maximum, value);
        }
        const double maximum = block_max(local_maximum, scratch);
        double local_sum = 0.0;
        for (std::int64_t j = threadIdx.x; j < target_count; j += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double value =
                log_ba - pair_cost(distance, cost) / eta + log_v[j];
            local_sum += exp(value - maximum);
        }
        const double sum = block_sum(local_sum, scratch);
        if (threadIdx.x == 0)
            next_u[i] = theta_y * (log_b - maximum - log(sum));
        __syncthreads();
    }
}

__global__ void update_target_dual_kernel(
    const double* target, std::int64_t target_count,
    const double* transformed, std::int64_t source_count,
    DeviceCostModel cost, double log_a, double log_ba, double eta,
    double theta_x, const double* next_u, double* next_v) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t j = blockIdx.x; j < target_count; j += gridDim.x) {
        double local_maximum = -CUDART_INF;
        for (std::int64_t i = threadIdx.x; i < source_count; i += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double value =
                log_ba - pair_cost(distance, cost) / eta + next_u[i];
            local_maximum = fmax(local_maximum, value);
        }
        const double maximum = block_max(local_maximum, scratch);
        double local_sum = 0.0;
        for (std::int64_t i = threadIdx.x; i < source_count; i += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double value =
                log_ba - pair_cost(distance, cost) / eta + next_u[i];
            local_sum += exp(value - maximum);
        }
        const double sum = block_sum(local_sum, scratch);
        if (threadIdx.x == 0)
            next_v[j] = theta_x * (log_a - maximum - log(sum));
        __syncthreads();
    }
}

__global__ void maximum_change_kernel(const double* current,
                                      const double* previous,
                                      std::int64_t count, double* result) {
    __shared__ double scratch[kBlockSize];
    double maximum = 0.0;
    for (std::int64_t index = threadIdx.x; index < count;
         index += blockDim.x) {
        maximum = fmax(maximum, fabs(current[index] - previous[index]));
    }
    maximum = block_max(maximum, scratch);
    if (threadIdx.x == 0) result[0] = maximum;
}

template <int Dimension>
__global__ void source_statistics_kernel(
    const double* target, std::int64_t target_count,
    const double* transformed, std::int64_t source_count,
    DeviceCostModel cost, double log_ba, double eta, const double* log_u,
    const double* log_v, double* gamma_source_mass,
    double* omega_source_mass, double* omega_sse, double* omega_x2,
    double* cost_terms, double* plan_kl_cores, double* omega_px,
    int* overflow) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t i = blockIdx.x; i < source_count; i += gridDim.x) {
        double gamma_sum = 0.0;
        double omega_sum = 0.0;
        double sse_sum = 0.0;
        double x2_sum = 0.0;
        double cost_sum = 0.0;
        double kl_sum = 0.0;
        double px_sum[Dimension > 0 ? Dimension : 1] = {};
        for (std::int64_t j = threadIdx.x; j < target_count; j += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double point_cost = pair_cost(distance, cost);
            const double log_gamma =
                log_u[i] + log_ba - point_cost / eta + log_v[j];
            const double gamma = checked_plan_exp(log_gamma, overflow);
            const double omega = gamma * robust_weight(distance, cost);
            double target_x2 = 0.0;
            for (std::int64_t axis = 0; axis < cost.dimension; ++axis) {
                const double coordinate = target[j + target_count * axis];
                target_x2 += coordinate * coordinate;
            }
            gamma_sum += gamma;
            omega_sum += omega;
            sse_sum += omega * distance;
            x2_sum += omega * target_x2;
            cost_sum += gamma * point_cost;
            if (gamma > 0.0) kl_sum += gamma * (log_gamma - log_ba);
            if constexpr (Dimension > 0) {
                for (int axis = 0; axis < Dimension; ++axis)
                    px_sum[axis] +=
                        omega * target[j + target_count * axis];
            }
        }
        const double gamma_total = block_sum(gamma_sum, scratch);
        if (threadIdx.x == 0) gamma_source_mass[i] = gamma_total;
        const double omega_total = block_sum(omega_sum, scratch);
        if (threadIdx.x == 0) omega_source_mass[i] = omega_total;
        const double sse_total = block_sum(sse_sum, scratch);
        if (threadIdx.x == 0) omega_sse[i] = sse_total;
        const double x2_total = block_sum(x2_sum, scratch);
        if (threadIdx.x == 0) omega_x2[i] = x2_total;
        const double cost_total = block_sum(cost_sum, scratch);
        if (threadIdx.x == 0) cost_terms[i] = cost_total;
        const double kl_total = block_sum(kl_sum, scratch);
        if (threadIdx.x == 0) plan_kl_cores[i] = kl_total;
        if constexpr (Dimension > 0) {
            for (int axis = 0; axis < Dimension; ++axis) {
                const double px_total = block_sum(px_sum[axis], scratch);
                if (threadIdx.x == 0)
                    omega_px[i + source_count * axis] = px_total;
            }
        }
        __syncthreads();
    }
}

__global__ void target_mass_kernel(
    const double* target, std::int64_t target_count,
    const double* transformed, std::int64_t source_count,
    DeviceCostModel cost, double log_ba, double eta, const double* log_u,
    const double* log_v, double* gamma_target_mass, int* overflow) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t j = blockIdx.x; j < target_count; j += gridDim.x) {
        double sum = 0.0;
        for (std::int64_t i = threadIdx.x; i < source_count; i += blockDim.x) {
            const double distance = squared_distance(
                transformed, source_count, i, target, target_count, j,
                cost.dimension);
            const double log_gamma =
                log_u[i] + log_ba - pair_cost(distance, cost) / eta + log_v[j];
            sum += checked_plan_exp(log_gamma, overflow);
        }
        sum = block_sum(sum, scratch);
        if (threadIdx.x == 0) gamma_target_mass[j] = sum;
        __syncthreads();
    }
}

__global__ void generic_px_kernel(
    const double* target, std::int64_t target_count,
    const double* transformed, std::int64_t source_count,
    DeviceCostModel cost, double log_ba, double eta, const double* log_u,
    const double* log_v, double* omega_px) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t i = blockIdx.x; i < source_count; i += gridDim.x) {
        for (std::int64_t base = 0; base < target_count;
             base += blockDim.x) {
            const std::int64_t j = base + threadIdx.x;
            double omega = 0.0;
            if (j < target_count) {
                const double distance = squared_distance(
                    transformed, source_count, i, target, target_count, j,
                    cost.dimension);
                const double point_cost = pair_cost(distance, cost);
                const double log_gamma =
                    log_u[i] + log_ba - point_cost / eta + log_v[j];
                double gamma = 0.0;
                if (log_gamma <= 700.0 && log_gamma >= -745.0)
                    gamma = exp(log_gamma);
                omega = gamma * robust_weight(distance, cost);
            }
            for (std::int64_t axis = 0; axis < cost.dimension; ++axis) {
                const double contribution =
                    j < target_count
                        ? omega * target[j + target_count * axis]
                        : 0.0;
                const double subtotal = block_sum(contribution, scratch);
                if (threadIdx.x == 0)
                    omega_px[i + source_count * axis] += subtotal;
            }
        }
        __syncthreads();
    }
}

__global__ void kkt_partial_kernel(
    const double* log_u, const double* log_v, const double* mapped_u,
    std::int64_t source_count,
    std::int64_t target_count, double eta, double tau_y, double tau_x,
    double theta_y, double theta_x, double log_b, double log_a,
    std::int64_t pair_count, double* partial) {
    __shared__ double scratch[kBlockSize];
    double maximum = 0.0;
    const std::int64_t stride =
        static_cast<std::int64_t>(blockDim.x) * gridDim.x;
    for (std::int64_t pair =
             static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         pair < pair_count; pair += stride) {
        const std::int64_t i = pair / target_count;
        const std::int64_t j = pair - i * target_count;
        // mapped_u is F(log_v).  Therefore log_b-mapped_u/theta_y is
        // the exact row log-sum-exp for the returned plan.  The final log_v
        // was produced from the returned log_u, so the analogous target
        // identity is exact as well.  This keeps KKT diagnostics finite even
        // when a marginal is below the range of exp(double).
        const double log_p = log_u[i] + log_b - mapped_u[i] / theta_y;
        const double log_q = log_v[j] + log_a - log_v[j] / theta_x;
        const double value = eta * (log_u[i] + log_v[j]) +
                             tau_y * (log_p - log_b) +
                             tau_x * (log_q - log_a);
        maximum = fmax(maximum, fabs(value));
    }
    maximum = block_max(maximum, scratch);
    if (threadIdx.x == 0) partial[blockIdx.x] = maximum;
}

__global__ void maximum_kernel(const double* values, std::int64_t count,
                               double* result) {
    __shared__ double scratch[kBlockSize];
    double maximum = 0.0;
    for (std::int64_t index = threadIdx.x; index < count;
         index += blockDim.x)
        maximum = fmax(maximum, values[index]);
    maximum = block_max(maximum, scratch);
    if (threadIdx.x == 0) result[0] = maximum;
}

__global__ void fixed_cost_kernel(
    const double* target, std::int64_t target_count,
    const double* old_transformed, const double* candidate_transformed,
    std::int64_t source_count, DeviceCostModel old_cost,
    DeviceCostModel candidate_cost, double log_ba, double eta,
    const double* log_u, const double* log_v, double* source_cost,
    int* overflow) {
    __shared__ double scratch[kBlockSize];
    for (std::int64_t i = blockIdx.x; i < source_count; i += gridDim.x) {
        double sum = 0.0;
        for (std::int64_t j = threadIdx.x; j < target_count; j += blockDim.x) {
            const double old_distance = squared_distance(
                old_transformed, source_count, i, target, target_count, j,
                old_cost.dimension);
            const double log_gamma =
                log_u[i] + log_ba - pair_cost(old_distance, old_cost) / eta +
                log_v[j];
            const double gamma = checked_plan_exp(log_gamma, overflow);
            if (gamma > 0.0) {
                const double candidate_distance = squared_distance(
                    candidate_transformed, source_count, i, target,
                    target_count, j, candidate_cost.dimension);
                sum += gamma * pair_cost(candidate_distance, candidate_cost);
            }
        }
        sum = block_sum(sum, scratch);
        if (threadIdx.x == 0) source_cost[i] = sum;
        __syncthreads();
    }
}

void throw_if_plan_overflow(const DeviceBuffer<int>& overflow) {
    int host_overflow = 0;
    overflow.copy_to_host(&host_overflow, 1);
    if (host_overflow != 0)
        throw std::runtime_error(
            "Sinkhorn plan overflow; rescale input or increase eta");
}

} // namespace

bool cuda_compiled() { return true; }

bool cuda_available() {
    int device_count = 0;
    const cudaError_t status = cudaGetDeviceCount(&device_count);
    if (status != cudaSuccess) {
        (void)cudaGetLastError();
        return false;
    }
    return device_count > 0;
}

std::string cuda_device_name() {
    if (!cuda_available()) return {};
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
    }
    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        (void)cudaGetLastError();
        return {};
    }
    return properties.name;
}

namespace detail {

TransportOutput sinkhorn_transport_cuda(const Matrix& target,
                                        const Matrix& transformed,
                                        double scale,
                                        const Options& options,
                                        const Vector& initial_log_u,
                                        const Vector& initial_log_v) {
    const Index source_count_index = transformed.rows();
    const Index target_count_index = target.rows();
    const Index dimension_index = transformed.cols();
    const std::int64_t source_count =
        static_cast<std::int64_t>(source_count_index);
    const std::int64_t target_count =
        static_cast<std::int64_t>(target_count_index);
    const std::int64_t dimension = static_cast<std::int64_t>(dimension_index);
    if (source_count <= 0 || target_count <= 0 || dimension <= 0)
        throw std::invalid_argument("Point set must be nonempty");
    if (target.cols() != dimension_index)
        throw std::invalid_argument("Source and target dimensions differ");

    const std::size_t source_size = checked_size(source_count_index, "source count");
    const std::size_t target_size = checked_size(target_count_index, "target count");
    const std::size_t transformed_size =
        checked_product(source_count_index, dimension_index, "source matrix");
    const std::size_t target_matrix_size =
        checked_product(target_count_index, dimension_index, "target matrix");
    if (source_count > std::numeric_limits<std::int64_t>::max() / target_count)
        throw std::overflow_error("point-pair count is too large");
    const std::int64_t pair_count = source_count * target_count;

    const double log_b = -std::log(static_cast<double>(source_count));
    const double log_a = -std::log(static_cast<double>(target_count));
    const double log_ba = log_b + log_a;
    const double eta = options.transport_entropy;
    const double theta_y = options.source_mass_penalty /
                           (options.source_mass_penalty + eta);
    const double theta_x = options.target_mass_penalty /
                           (options.target_mass_penalty + eta);
    const double contraction = theta_y * theta_x;
    const DeviceCostModel cost = make_cost_model(dimension_index, scale, options);

    DeviceBuffer<double> device_target(target_matrix_size);
    DeviceBuffer<double> device_transformed(transformed_size);
    DeviceBuffer<double> log_u(source_size);
    DeviceBuffer<double> log_v(target_size);
    DeviceBuffer<double> next_u(source_size);
    DeviceBuffer<double> next_v(target_size);
    DeviceBuffer<double> scalar(1);
    device_target.copy_from_host(target.data(), target_matrix_size);
    device_transformed.copy_from_host(transformed.data(), transformed_size);
    if (initial_log_u.size() == source_count_index)
        log_u.copy_from_host(initial_log_u.data(), source_size);
    else
        log_u.zero();
    if (initial_log_v.size() == target_count_index)
        log_v.copy_from_host(initial_log_v.data(), target_size);
    else
        log_v.zero();

    double residual = std::numeric_limits<double>::infinity();
    int used_iterations = 0;
    for (int iteration = 0; iteration < options.sinkhorn_iterations; ++iteration) {
        update_source_dual_kernel<<<grid_size(source_count), kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_b, log_ba, eta, theta_y, log_v.get(),
            next_u.get());
        check_launch("update_source_dual_kernel");
        update_target_dual_kernel<<<grid_size(target_count), kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_a, log_ba, eta, theta_x, next_u.get(),
            next_v.get());
        check_launch("update_target_dual_kernel");
        maximum_change_kernel<<<1, kBlockSize>>>(
            next_v.get(), log_v.get(), target_count, scalar.get());
        check_launch("maximum_change_kernel");
        double change = 0.0;
        scalar.copy_to_host(&change, 1);
        residual = change / (1.0 - contraction);
        log_u.swap(next_u);
        log_v.swap(next_v);
        used_iterations = iteration + 1;
        if (residual <= options.sinkhorn_tolerance) break;
    }

    DeviceBuffer<double> gamma_source_mass(source_size);
    DeviceBuffer<double> gamma_target_mass(target_size);
    DeviceBuffer<double> omega_source_mass(source_size);
    DeviceBuffer<double> omega_sse(source_size);
    DeviceBuffer<double> omega_x2(source_size);
    DeviceBuffer<double> cost_terms(source_size);
    DeviceBuffer<double> plan_kl_cores(source_size);
    DeviceBuffer<double> omega_px(transformed_size);
    DeviceBuffer<int> overflow(1);
    overflow.zero();

    const int source_grid = grid_size(source_count);
    if (dimension == 1) {
        source_statistics_kernel<1><<<source_grid, kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
            gamma_source_mass.get(), omega_source_mass.get(), omega_sse.get(),
            omega_x2.get(), cost_terms.get(), plan_kl_cores.get(),
            omega_px.get(), overflow.get());
    } else if (dimension == 2) {
        source_statistics_kernel<2><<<source_grid, kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
            gamma_source_mass.get(), omega_source_mass.get(), omega_sse.get(),
            omega_x2.get(), cost_terms.get(), plan_kl_cores.get(),
            omega_px.get(), overflow.get());
    } else if (dimension == 3) {
        source_statistics_kernel<3><<<source_grid, kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
            gamma_source_mass.get(), omega_source_mass.get(), omega_sse.get(),
            omega_x2.get(), cost_terms.get(), plan_kl_cores.get(),
            omega_px.get(), overflow.get());
    } else {
        source_statistics_kernel<0><<<source_grid, kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
            gamma_source_mass.get(), omega_source_mass.get(), omega_sse.get(),
            omega_x2.get(), cost_terms.get(), plan_kl_cores.get(), nullptr,
            overflow.get());
    }
    check_launch("source_statistics_kernel");
    target_mass_kernel<<<grid_size(target_count), kBlockSize>>>(
        device_target.get(), target_count, device_transformed.get(),
        source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
        gamma_target_mass.get(), overflow.get());
    check_launch("target_mass_kernel");
    if (dimension > 3) {
        omega_px.zero();
        generic_px_kernel<<<source_grid, kBlockSize>>>(
            device_target.get(), target_count, device_transformed.get(),
            source_count, cost, log_ba, eta, log_u.get(), log_v.get(),
            omega_px.get());
        check_launch("generic_px_kernel");
    }
    throw_if_plan_overflow(overflow);

    TransportOutput output;
    output.log_u.resize(source_count_index);
    output.log_v.resize(target_count_index);
    output.gamma_source_mass.resize(source_count_index);
    output.gamma_target_mass.resize(target_count_index);
    output.omega.mass.resize(source_count_index);
    output.omega.px.resize(source_count_index, dimension_index);
    output.dual_residual = residual;
    output.iterations = used_iterations;
    log_u.copy_to_host(output.log_u.data(), source_size);
    log_v.copy_to_host(output.log_v.data(), target_size);
    gamma_source_mass.copy_to_host(output.gamma_source_mass.data(), source_size);
    gamma_target_mass.copy_to_host(output.gamma_target_mass.data(), target_size);
    omega_source_mass.copy_to_host(output.omega.mass.data(), source_size);
    omega_px.copy_to_host(output.omega.px.data(), transformed_size);

    std::vector<double> host_omega_sse(source_size);
    std::vector<double> host_omega_x2(source_size);
    std::vector<double> host_cost_terms(source_size);
    std::vector<double> host_plan_kl_cores(source_size);
    omega_sse.copy_to_host(host_omega_sse.data(), source_size);
    omega_x2.copy_to_host(host_omega_x2.data(), source_size);
    cost_terms.copy_to_host(host_cost_terms.data(), source_size);
    plan_kl_cores.copy_to_host(host_plan_kl_cores.data(), source_size);

    long double gamma_mass = 0.0L;
    long double omega_mass = 0.0L;
    long double total_sse = 0.0L;
    long double total_x2 = 0.0L;
    long double cost_term = 0.0L;
    long double plan_kl_core = 0.0L;
    for (std::size_t i = 0; i < source_size; ++i) {
        gamma_mass += output.gamma_source_mass[static_cast<Index>(i)];
        omega_mass += output.omega.mass[static_cast<Index>(i)];
        total_sse += host_omega_sse[i];
        total_x2 += host_omega_x2[i];
        cost_term += host_cost_terms[i];
        plan_kl_core += host_plan_kl_cores[i];
    }
    output.gamma_mass = static_cast<double>(gamma_mass);
    output.omega_mass = static_cast<double>(omega_mass);
    output.omega.sse_old = static_cast<double>(total_sse);
    output.omega.weighted_x2 = static_cast<double>(total_x2);
    output.cost_term = static_cast<double>(cost_term);
    output.plan_kl =
        static_cast<double>(plan_kl_core - gamma_mass + 1.0L);
    output.source_kl = generalized_kl(output.gamma_source_mass, log_b);
    output.target_kl = generalized_kl(output.gamma_target_mass, log_a);
    output.transport_objective =
        output.cost_term + eta * output.plan_kl +
        options.source_mass_penalty * output.source_kl +
        options.target_mass_penalty * output.target_kl;

    const std::int64_t rounded_pair_blocks =
        pair_count / kBlockSize + (pair_count % kBlockSize != 0 ? 1 : 0);
    const int kkt_blocks = static_cast<int>(std::max<std::int64_t>(
        1, std::min<std::int64_t>(1024, rounded_pair_blocks)));
    DeviceBuffer<double> kkt_partial(static_cast<std::size_t>(kkt_blocks));
    update_source_dual_kernel<<<grid_size(source_count), kBlockSize>>>(
        device_target.get(), target_count, device_transformed.get(),
        source_count, cost, log_b, log_ba, eta, theta_y, log_v.get(),
        next_u.get());
    check_launch("update_source_dual_kernel (KKT)");
    kkt_partial_kernel<<<kkt_blocks, kBlockSize>>>(
        log_u.get(), log_v.get(), next_u.get(), source_count, target_count,
        eta, options.source_mass_penalty, options.target_mass_penalty,
        theta_y, theta_x, log_b, log_a, pair_count, kkt_partial.get());
    check_launch("kkt_partial_kernel");
    maximum_kernel<<<1, kBlockSize>>>(kkt_partial.get(), kkt_blocks,
                                      scalar.get());
    check_launch("maximum_kernel");
    scalar.copy_to_host(&output.kkt_residual, 1);
    return output;
}

double fixed_plan_cost_cuda(const Matrix& target,
                            const Matrix& e_transformed,
                            double e_scale,
                            const Vector& log_u,
                            const Vector& log_v,
                            const Matrix& candidate_transformed,
                            double candidate_scale,
                            const Options& options) {
    const Index source_count_index = e_transformed.rows();
    const Index target_count_index = target.rows();
    const Index dimension_index = target.cols();
    const std::int64_t source_count =
        static_cast<std::int64_t>(source_count_index);
    const std::int64_t target_count =
        static_cast<std::int64_t>(target_count_index);
    if (source_count <= 0 || target_count <= 0 || dimension_index <= 0)
        throw std::invalid_argument("Point set must be nonempty");
    if (e_transformed.cols() != dimension_index ||
        candidate_transformed.rows() != source_count_index ||
        candidate_transformed.cols() != dimension_index)
        throw std::invalid_argument("Source and target dimensions differ");
    if (log_u.size() != source_count_index || log_v.size() != target_count_index)
        throw std::invalid_argument("Sinkhorn dual dimensions differ");

    const std::size_t source_size = checked_size(source_count_index, "source count");
    const std::size_t target_size = checked_size(target_count_index, "target count");
    const std::size_t source_matrix_size =
        checked_product(source_count_index, dimension_index, "source matrix");
    const std::size_t target_matrix_size =
        checked_product(target_count_index, dimension_index, "target matrix");
    const double log_ba = -std::log(static_cast<double>(source_count)) -
                          std::log(static_cast<double>(target_count));
    const DeviceCostModel old_cost =
        make_cost_model(dimension_index, e_scale, options);
    const DeviceCostModel candidate_cost =
        make_cost_model(dimension_index, candidate_scale, options);

    DeviceBuffer<double> device_target(target_matrix_size);
    DeviceBuffer<double> device_old(source_matrix_size);
    DeviceBuffer<double> device_candidate(source_matrix_size);
    DeviceBuffer<double> device_log_u(source_size);
    DeviceBuffer<double> device_log_v(target_size);
    DeviceBuffer<double> source_cost(source_size);
    DeviceBuffer<int> overflow(1);
    device_target.copy_from_host(target.data(), target_matrix_size);
    device_old.copy_from_host(e_transformed.data(), source_matrix_size);
    device_candidate.copy_from_host(candidate_transformed.data(),
                                    source_matrix_size);
    device_log_u.copy_from_host(log_u.data(), source_size);
    device_log_v.copy_from_host(log_v.data(), target_size);
    overflow.zero();

    fixed_cost_kernel<<<grid_size(source_count), kBlockSize>>>(
        device_target.get(), target_count, device_old.get(),
        device_candidate.get(), source_count, old_cost, candidate_cost,
        log_ba, options.transport_entropy, device_log_u.get(),
        device_log_v.get(), source_cost.get(), overflow.get());
    check_launch("fixed_cost_kernel");
    throw_if_plan_overflow(overflow);

    std::vector<double> host_cost(source_size);
    source_cost.copy_to_host(host_cost.data(), source_size);
    long double value = 0.0L;
    for (double term : host_cost) value += term;
    return static_cast<double>(value);
}

} // namespace detail
} // namespace clusterreg
