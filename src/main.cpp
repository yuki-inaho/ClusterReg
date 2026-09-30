// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/clusterreg.hpp"
#include "clusterreg/io.hpp"
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
using namespace clusterreg;
namespace {
const char* usage=R"(ClusterReg C++17: clustering and UOT/Sinkhorn non-rigid registration
Usage:
  clusterreg_cli demo --out DIR [synthetic options] [registration options]
  clusterreg_cli synthetic --out DIR [synthetic options]
  clusterreg_cli register --source FILE --target FILE --out DIR [registration options]
Input: numeric CSV/whitespace (no header) or ASCII PLY. Output: CSV, history, metrics JSON.
Synthetic: --n 512 --dim 2 --amplitude 0.4 --noise 0.003 --missing 0 --outliers 0 --seed 42
Registration:
  --algorithm cluster|sinkhorn  Default cluster
  --backend cpu|cuda|auto       Sinkhorn execution backend; default cpu
  --noise-model gaussian|student-t  Sinkhorn residual model; default gaussian
  --mode paper|official      Default paper; official matches Python/MATLAB update order
  --solver nystrom|dense     Default nystrom; dense is O(N^3) reference
  --estep streaming|dense    Default streaming; dense stores full U for verification
  --rank 128                Fixed rank; omit to use --ratio 0.3
  --gamma 2 --entropy 0.5 --regularization 0.1 --eigen-cutoff 1e-6
  --max-iter 50 --tol 1e-5 --sigma-floor 1e-8 --kmeans-iter 5 --threads 1
  --transport-entropy 1 --tau-source 3 --tau-target 3 --student-dof 4
  --sigma-ceiling 4 --initial-sigma 0 --sinkhorn-iter 500 --sinkhorn-tol 1e-10
  --normalize yes|no --align-centroids yes|no --fixed yes|no
  --truth FILE              Ground truth in SOURCE order; evaluation only
  --dump yes|no             Export prepared inputs, Q, landmarks and coefficients
All options require a value. Unknown options are errors. --out must not exist already.
)";
std::string get(const std::map<std::string,std::string>& a,const std::string& k,const std::string& def) { auto it=a.find(k); return it==a.end()?def:it->second; }
bool yes(const std::string& s) { if(s!="yes" && s!="no") throw std::invalid_argument("Expected yes or no, got "+s); return s=="yes"; }
int integer(const std::string& s) { std::size_t p=0; int v=std::stoi(s,&p); if(p!=s.size()) throw std::invalid_argument("Invalid integer: "+s); return v; }
double number(const std::string& s) { std::size_t p=0; double v=std::stod(s,&p); if(p!=s.size() || !std::isfinite(v)) throw std::invalid_argument("Invalid number: "+s); return v; }
std::string metrics(const Result& r,const Matrix& source,const Matrix* truth) {
    std::ostringstream f; f<<std::setprecision(17);
    f<<"{\n  \"source_count\": "<<source.rows()<<", \"target_count\": "<<r.prepared_target.rows()<<", \"dimension\": "<<source.cols()
     <<",\n  \"algorithm\": \""<<(r.options.algorithm==Algorithm::Clustering?"cluster":"sinkhorn")
     <<"\", \"noise_model\": \""<<(r.options.noise_model==NoiseModel::Gaussian?"gaussian":"student-t")
     <<"\", \"backend\": \""<<(r.backend_used==Backend::CUDA?"cuda":"cpu")
     <<"\", \"mode\": \""<<(r.options.semantics==Semantics::Paper?"paper":"official")<<"\", \"solver\": \""<<(r.options.solver==Solver::Nystrom?"nystrom":"dense")
     <<"\", \"estep\": \""<<(r.options.estep==EStep::Streaming?"streaming":"dense")<<"\",\n  \"threads_requested\": "<<r.options.threads
     <<", \"requested_rank\": "<<r.basis.requested_rank<<", \"retained_rank\": "<<r.basis.retained_rank
     <<",\n  \"iterations\": "<<r.history.size()<<", \"stop_reason\": \""<<r.stop_reason<<"\", \"sigma2\": "<<r.sigma2
     <<", \"transport_mass\": "<<r.transport_mass
     <<",\n  \"preparation_seconds\": "<<r.preparation_seconds<<", \"iteration_seconds\": "<<r.iteration_seconds<<", \"fit_seconds\": "<<r.total_seconds;
    double e=0,m=0; for(const auto& h:r.history) { e+=h.estep_seconds; m+=h.solve_seconds; }
    f<<",\n  \"estep_seconds\": "<<e<<", \"solve_seconds\": "<<m;
    Matrix target=r.prepared_target*r.target_scale; target.rowwise()+=r.target_center;
    f<<",\n  \"nn_source_to_target_rmse\": "<<nearest_rmse(r.transformed,target)<<", \"nn_target_to_source_rmse\": "<<nearest_rmse(target,r.transformed);
    if(truth) {
        Matrix initial=r.prepared_source*r.target_scale; initial.rowwise()+=r.target_center;
        f<<",\n  \"gt_rmse_before\": "<<corresponding_rmse(source,*truth)<<", \"gt_rmse_prealigned\": "<<corresponding_rmse(initial,*truth)<<", \"gt_rmse_after\": "<<corresponding_rmse(r.transformed,*truth);
    }
    f<<"\n}\n"; return f.str();
}
}
int main(int argc,char** argv) {
    try {
        if(argc<2 || std::string(argv[1])=="--help" || std::string(argv[1])=="help") { std::cout<<usage; return 0; }
        std::string command=argv[1]; if(command!="register" && command!="demo" && command!="synthetic") throw std::invalid_argument("Unknown command: "+command);
        std::set<std::string> allowed={"out","source","target","truth","n","dim","amplitude","noise","missing","outliers","seed","algorithm","backend","noise-model","mode","solver","estep","rank","ratio","gamma","entropy","regularization","eigen-cutoff","max-iter","tol","sigma-floor","sigma-ceiling","initial-sigma","transport-entropy","tau-source","tau-target","student-dof","sinkhorn-iter","sinkhorn-tol","mass-floor","kmeans-iter","threads","normalize","align-centroids","fixed","dump"};
        std::map<std::string,std::string> args;
        for(int i=2;i<argc;i+=2) { std::string key=argv[i]; if(key.substr(0,2)!="--" || i+1>=argc) throw std::invalid_argument("Options need --name value"); key=key.substr(2); if(!allowed.count(key) || args.count(key)) throw std::invalid_argument("Unknown or repeated option: "+key); args[key]=argv[i+1]; }
        std::string out=get(args,"out",""); if(out.empty()) throw std::invalid_argument("--out DIR is required");
        if(std::filesystem::exists(out)) throw std::invalid_argument("Output path already exists: "+out+" (choose a new directory)");
        Options o;
        o.rank=integer(get(args,"rank","0")); o.rank_ratio=number(get(args,"ratio","0.3"));
        o.gamma=number(get(args,"gamma","2")); o.entropy=number(get(args,"entropy","0.5")); o.regularization=number(get(args,"regularization","0.1"));
        o.eigen_cutoff=number(get(args,"eigen-cutoff","1e-6")); o.max_iterations=integer(get(args,"max-iter","50")); o.tolerance=number(get(args,"tol","1e-5"));
        o.sigma_floor=number(get(args,"sigma-floor","1e-8")); o.sigma_ceiling=number(get(args,"sigma-ceiling","4")); o.initial_sigma=number(get(args,"initial-sigma","0"));
        o.transport_entropy=number(get(args,"transport-entropy","1")); o.source_mass_penalty=number(get(args,"tau-source","3")); o.target_mass_penalty=number(get(args,"tau-target","3")); o.student_dof=number(get(args,"student-dof","4"));
        o.sinkhorn_iterations=integer(get(args,"sinkhorn-iter","500")); o.sinkhorn_tolerance=number(get(args,"sinkhorn-tol","1e-10")); o.transport_mass_floor=number(get(args,"mass-floor","1e-12"));
        o.kmeans_iterations=integer(get(args,"kmeans-iter","5")); o.threads=integer(get(args,"threads","1"));
        int seed=integer(get(args,"seed","42")); if(seed<0) throw std::invalid_argument("Seed must be nonnegative"); o.seed=static_cast<std::uint64_t>(seed);
        std::string mode=get(args,"mode","paper"); if(mode!="paper" && mode!="official") throw std::invalid_argument("Invalid --mode"); o.semantics=mode=="paper"?Semantics::Paper:Semantics::Official;
        std::string algorithm=get(args,"algorithm","cluster"); if(algorithm!="cluster" && algorithm!="sinkhorn") throw std::invalid_argument("Invalid --algorithm"); o.algorithm=algorithm=="cluster"?Algorithm::Clustering:Algorithm::Sinkhorn;
        std::string backend=get(args,"backend","cpu"); if(backend!="cpu" && backend!="cuda" && backend!="auto") throw std::invalid_argument("Invalid --backend"); o.backend=backend=="cuda"?Backend::CUDA:(backend=="auto"?Backend::Auto:Backend::CPU);
        std::string noise_model=get(args,"noise-model","gaussian"); if(noise_model!="gaussian" && noise_model!="student-t") throw std::invalid_argument("Invalid --noise-model"); o.noise_model=noise_model=="gaussian"?NoiseModel::Gaussian:NoiseModel::StudentT;
        std::string solver=get(args,"solver","nystrom"); if(solver!="nystrom" && solver!="dense") throw std::invalid_argument("Invalid --solver"); o.solver=solver=="nystrom"?Solver::Nystrom:Solver::Dense;
        std::string estep=get(args,"estep","streaming"); if(estep!="streaming" && estep!="dense") throw std::invalid_argument("Invalid --estep"); o.estep=estep=="streaming"?EStep::Streaming:EStep::Dense;
        std::string normalization=get(args,"normalize","yes"); if(normalization!="yes" && normalization!="no" && normalization!="separate" && normalization!="common" && normalization!="none") throw std::invalid_argument("Invalid --normalize"); o.normalize=normalization!="no" && normalization!="none";
        o.align_centroids=yes(get(args,"align-centroids","yes")); o.fixed_iterations=yes(get(args,"fixed","no")); bool dump=yes(get(args,"dump","no"));
        Matrix source,target,truth; bool have_truth=false;
        if(command=="demo" || command=="synthetic") {
            SyntheticOptions s; s.count=integer(get(args,"n","512")); s.dimension=integer(get(args,"dim","2")); s.amplitude=number(get(args,"amplitude","0.4"));
            s.noise=number(get(args,"noise","0.003")); s.missing_fraction=number(get(args,"missing","0")); s.outlier_fraction=number(get(args,"outliers","0")); s.seed=o.seed;
            auto data=make_synthetic(s); source=data.source; target=data.target; truth=data.truth; have_truth=true;
            write_synthetic(out,data,s); if(command=="synthetic") { std::cout<<"Synthetic data written to "<<out<<'\n'; return 0; }
        } else {
            source=read_points(get(args,"source","")); target=read_points(get(args,"target",""));
            if(args.count("truth")) { truth=read_points(args["truth"]); have_truth=true; }
            if(have_truth && (truth.rows()!=source.rows() || truth.cols()!=source.cols())) throw std::invalid_argument("Truth shape must match source");
            std::filesystem::create_directories(out);
        }
        Result result=fit(source,target,o); // No truth, IDs or known transform enter the solver.
        write_points(out+"/registered.csv",result.transformed); write_history(out+"/history.csv",result.history);
        write_points(out+"/alpha.csv",result.alpha);
        if(dump) {
            write_points(out+"/prepared_source.csv",result.prepared_source); write_points(out+"/prepared_target.csv",result.prepared_target);
            write_points(out+"/normalized_registered.csv",result.normalized_transformed); write_points(out+"/coefficients.csv",result.coefficients);
            if(o.solver==Solver::Nystrom) { write_points(out+"/Q.csv",result.basis.Q); write_points(out+"/landmarks.csv",result.basis.centers); }
        }
        std::string report=metrics(result,source,have_truth?&truth:nullptr);
        std::ofstream f(out+"/metrics.json"); if(!f) throw std::runtime_error("Cannot write metrics"); f<<report;
        std::cout<<report;
        return 0;
    } catch(const std::exception& e) { std::cerr<<"Error: "<<e.what()<<'\n'; return 1; }
}
