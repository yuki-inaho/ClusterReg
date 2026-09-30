#pragma once
// SPDX-License-Identifier: AGPL-3.0-only
#include <Eigen/Core>
#include <cstdint>
#include <string>
#include <vector>

namespace clusterreg {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Index = Eigen::Index;

enum class Semantics { Paper, Official };
enum class EStep { Streaming, Dense };
enum class Solver { Nystrom, Dense };
enum class Algorithm { Clustering, Sinkhorn };
enum class NoiseModel { Gaussian, StudentT };
enum class Backend { Auto, CPU, CUDA };

struct Options {
    double entropy = 0.5;              // paper lambda, upstream beta
    double regularization = 0.1;       // paper zeta, upstream lambda
    double gamma = 2.0;                // upstream theta = 1 / gamma
    double rank_ratio = 0.3;           // NOT linear memory when rank grows with N
    double eigen_cutoff = 1e-6;        // absolute cutoff, as in upstream INys
    double sigma_floor = 1e-8;
    double sigma_ceiling = 4.0;        // Sinkhorn scale upper bound
    double initial_sigma = 0.0;        // 0: mean all-pairs squared distance / d
    double tolerance = 1e-5;
    double transport_entropy = 1.0;    // eta in the UOT objective
    double source_mass_penalty = 3.0;  // tau_y
    double target_mass_penalty = 3.0;  // tau_x
    double student_dof = 4.0;          // nu; Sinkhorn Student-t only
    double sinkhorn_tolerance = 1e-10; // a-posteriori log-dual bound
    double transport_mass_floor = 1e-12;
    int rank = 0;                       // 0: ceil(rank_ratio * source_count)
    int max_iterations = 50;
    int sinkhorn_iterations = 500;
    int kmeans_iterations = 5;
    int threads = 1;
    std::uint64_t seed = 42;
    bool normalize = true;             // separate for clustering, shared for Sinkhorn
    bool align_centroids = true;       // relevant when normalization is off
    bool fixed_iterations = false;     // ignore tolerance (not upstream variance stop)
    Semantics semantics = Semantics::Paper;
    EStep estep = EStep::Streaming;
    Solver solver = Solver::Nystrom;
    Algorithm algorithm = Algorithm::Clustering;
    NoiseModel noise_model = NoiseModel::Gaussian;
    Backend backend = Backend::CPU;
};

struct KMeansResult {
    Matrix centers;
    std::vector<int> labels;
    double quantization_error = 0;
    int max_cluster_size = 0;
    int iterations = 0;
    std::uint64_t distance_evaluations = 0;
};
struct Basis {
    Matrix centers, projection, Q;
    int requested_rank = 0;
    int retained_rank = 0;
    double min_retained_eigenvalue = 0;
    KMeansResult clustering;
};
struct Statistics {
    Vector mass;                        // U^T 1_M, length N
    Matrix px;                          // U^T X, N x dimension
    double entropy = 0;                // sum U_ij log U_ij
    double sse_old = 0;                // direct weighted distances at current T
    double weighted_x2 = 0;
    double row_sum_error = 0;
};
struct Update {
    Matrix transformed, coefficients;  // reduced B, or full C for dense solver
    double penalty = 0;                // ||B||^2 or tr(C^T K C)
    double residual = 0;               // relative linear system residual
};
struct Iteration {
    int index = 0;
    double sigma_before = 0, sigma_after = 0;
    double after_u = 0, after_alpha = 0, after_deform = 0, after_variance = 0;
    double official_loss = 0;
    double row_sum_error = 0, alpha_sum = 0, linear_residual = 0, step_rms = 0;
    double estep_seconds = 0, solve_seconds = 0;
    double objective_before = 0, objective_after = 0;
    double transport_mass = 0, robust_mass = 0;
    double transport_residual = 0, kkt_residual = 0;
    double scale_relative_change = 0;
    int transport_iterations = 0;
};
struct Result {
    Matrix transformed;               // raw target coordinate system
    Matrix prepared_source, prepared_target, normalized_transformed;
    Basis basis;
    Matrix coefficients;              // B or C
    Vector alpha;
    Vector source_mass, target_mass;
    double sigma2 = 0;
    Eigen::RowVectorXd source_center, target_center, source_shift;
    double source_scale = 1, target_scale = 1;
    Options options;
    std::vector<Iteration> history;
    std::string stop_reason;
    Backend backend_used = Backend::CPU;
    double transport_mass = 0;
    double preparation_seconds = 0, iteration_seconds = 0, total_seconds = 0;
};

// The following functions are public to support independent, small-fixture audits.
Matrix laplacian_kernel(const Matrix& a, const Matrix& b, double gamma);
Matrix kmeans_plus_plus(const Matrix& data, int k, std::uint64_t seed);
KMeansResult kmeans(const Matrix& data, const Matrix& initial_centers, int iterations,
                    bool use_elkan = true, int threads = 1);
Basis make_basis(const Matrix& source, const Options& options);
Statistics expectation(const Matrix& target, const Matrix& transformed,
                       const Vector& alpha, double sigma2, const Options& options);
Update reduced_update(const Matrix& source, const Matrix& Q, const Statistics& s,
                      double kappa);
Update dense_update(const Matrix& source, const Matrix& kernel, const Statistics& s,
                    double kappa);
double updated_sse(const Statistics& s, const Matrix& old_t, const Matrix& new_t);
double objective(double sse, double sigma2, const Statistics& s, const Vector& alpha,
                 double penalty, Index target_count, Index dimension, const Options& options);
Result fit(const Matrix& source, const Matrix& target, const Options& options = {});
Matrix transform(const Result& model, const Matrix& query); // arbitrary new source points
int available_threads();
bool cuda_compiled();
bool cuda_available();
std::string cuda_device_name();
} // namespace clusterreg
