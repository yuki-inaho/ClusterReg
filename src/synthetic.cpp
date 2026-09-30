// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/io.hpp"
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <random>
#include <stdexcept>
namespace clusterreg {
SyntheticData make_synthetic(const SyntheticOptions& o) {
    if(o.count<8 || (o.dimension!=2 && o.dimension!=3) || o.noise<0 || o.amplitude<0 ||
       !std::isfinite(o.amplitude) || !std::isfinite(o.noise) || !(o.missing_fraction>=0 && o.missing_fraction<1) ||
       !(o.outlier_fraction>=0 && o.outlier_fraction<1)) throw std::invalid_argument("Invalid synthetic data options");
    std::mt19937_64 rng(o.seed);
    std::normal_distribution<double> normal(0,1);
    std::uniform_real_distribution<double> uniform(0,1);
    SyntheticData a; a.source.resize(o.count,o.dimension); a.truth.resize(o.count,o.dimension);
    const double pi=std::acos(-1.0), golden=pi*(3-std::sqrt(5.0));
    for(int i=0;i<o.count;++i) {
        if(o.dimension==2) {
            double t=2*pi*(i+0.15*uniform(rng))/o.count;
            double r=1+0.22*std::cos(3*t)+0.12*std::sin(2*t)+0.06*std::cos(5*t);
            double x=1.3*r*std::cos(t), y=0.8*r*std::sin(t);
            a.source(i,0)=x; a.source(i,1)=y;
            a.truth(i,0)=x+o.amplitude*(0.22*std::sin(1.7*y)+0.10*x*y)+0.18;
            a.truth(i,1)=y+o.amplitude*(0.25*std::sin(1.2*x)+0.08*(x*x-0.5))-0.12;
        } else {
            double z=1-2*(i+0.5)/o.count, t=i*golden;
            double r=std::sqrt(std::max(0.0,1-z*z))*(1+0.12*std::sin(3*t)*(1-z*z));
            double x=1.2*r*std::cos(t), y=0.75*r*std::sin(t);
            z=0.95*z+0.05*std::sin(2*t)*(1-z*z);
            a.source.row(i)<<x,y,z;
            double angle=o.amplitude*0.4*z;
            a.truth(i,0)=std::cos(angle)*x-std::sin(angle)*y+o.amplitude*0.15*std::sin(2*z)+0.18;
            a.truth(i,1)=std::sin(angle)*x+std::cos(angle)*y+o.amplitude*0.12*x*z-0.12;
            a.truth(i,2)=z+o.amplitude*0.16*std::sin(1.3*x)+0.08;
        }
    }
    std::vector<int> perm(static_cast<std::size_t>(o.count));
    std::iota(perm.begin(),perm.end(),0); std::shuffle(perm.begin(),perm.end(),rng);
    const int retained=std::max(1,static_cast<int>(std::round(o.count*(1-o.missing_fraction))));
    const int outliers=static_cast<int>(std::ceil(retained*o.outlier_fraction/(1-o.outlier_fraction)));
    Matrix target(retained+outliers,o.dimension);
    std::vector<int> ids(static_cast<std::size_t>(target.rows()),-1);
    for(int i=0;i<retained;++i) {
        ids[static_cast<std::size_t>(i)]=perm[static_cast<std::size_t>(i)];
        for(int d=0;d<o.dimension;++d) target(i,d)=a.truth(perm[static_cast<std::size_t>(i)],d)+o.noise*normal(rng);
    }
    for(int i=retained;i<target.rows();++i) for(int d=0;d<o.dimension;++d) target(i,d)=6*uniform(rng)-3;
    std::vector<int> target_perm(static_cast<std::size_t>(target.rows()));
    std::iota(target_perm.begin(),target_perm.end(),0); std::shuffle(target_perm.begin(),target_perm.end(),rng);
    a.target.resize(target.rows(),target.cols()); a.target_ids.resize(target_perm.size());
    for(Index i=0;i<target.rows();++i) { a.target.row(i)=target.row(target_perm[static_cast<std::size_t>(i)]); a.target_ids[static_cast<std::size_t>(i)]=ids[static_cast<std::size_t>(target_perm[static_cast<std::size_t>(i)])]; }
    return a;
}
void write_synthetic(const std::string& directory,const SyntheticData& a,const SyntheticOptions& o) {
    std::filesystem::create_directories(directory);
    write_points(directory+"/source.csv",a.source); write_points(directory+"/target.csv",a.target); write_points(directory+"/truth.csv",a.truth);
    std::ofstream ids(directory+"/target_ids.csv"); if(!ids) throw std::runtime_error("Cannot write target IDs");
    ids<<"# Evaluation only: source index, or -1 for an outlier. Never supplied to fit().\n";
    for(int i:a.target_ids) ids<<i<<'\n';
    std::ofstream f(directory+"/synthetic.json"); if(!f) throw std::runtime_error("Cannot write synthetic metadata");
    f<<std::setprecision(17)<<"{\"source_count\":"<<o.count<<",\"target_count\":"<<a.target.rows()<<",\"dimension\":"<<o.dimension
     <<",\"amplitude\":"<<o.amplitude<<",\"target_noise_std\":"<<o.noise<<",\"missing_fraction\":"<<o.missing_fraction<<",\"outlier_fraction\":"<<o.outlier_fraction<<",\"seed\":"<<o.seed<<"}\n";
}
} // namespace clusterreg
