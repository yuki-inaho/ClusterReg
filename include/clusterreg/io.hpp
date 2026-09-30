#pragma once
#include "clusterreg.hpp"
#include <string>
namespace clusterreg {
Matrix read_points(const std::string& filename);
void write_points(const std::string& filename,const Matrix& points);
void write_history(const std::string& filename,const std::vector<Iteration>& history);
double corresponding_rmse(const Matrix& a,const Matrix& b);
double nearest_rmse(const Matrix& from,const Matrix& to);
struct SyntheticOptions {
    int count=512, dimension=2;
    double amplitude=0.4, noise=0.003, missing_fraction=0, outlier_fraction=0;
    std::uint64_t seed=42;
};
struct SyntheticData { Matrix source,target,truth; std::vector<int> target_ids; };
SyntheticData make_synthetic(const SyntheticOptions& options);
void write_synthetic(const std::string& directory,const SyntheticData& data,const SyntheticOptions& options);
} // namespace clusterreg
