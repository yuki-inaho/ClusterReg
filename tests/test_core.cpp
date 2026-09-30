// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/clusterreg.hpp"
#include "clusterreg/io.hpp"
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
using namespace clusterreg;
namespace {
int passed=0,failed=0;
void check(bool condition,const std::string& message) { if(!condition) throw std::runtime_error(message); }
void small(double error,double tolerance,const std::string& what) {
    std::cout<<"  "<<what<<"="<<std::setprecision(17)<<error<<" tolerance="<<tolerance<<'\n';
    check(std::isfinite(error) && error<=tolerance,what);
}
void test(const std::string& name,const std::function<void()>& fn) {
    std::cout<<"TEST "<<name<<'\n';
    try { fn(); ++passed; std::cout<<"PASS "<<name<<'\n'; }
    catch(const std::exception& e) { ++failed; std::cout<<"FAIL "<<name<<": "<<e.what()<<'\n'; }
}
Matrix fixture(Index n,Index d) {
    Matrix m(n,d); for(Index i=0;i<n;++i) for(Index j=0;j<d;++j) m(i,j)=std::sin((i+1)*(j+2)*0.731)+0.17*std::cos((i+3)*(j+1)*0.219); return m;
}
void expect_error(const std::function<void()>& fn) { bool error=false; try { fn(); } catch(const std::exception&) { error=true; } check(error,"expected exception"); }
Matrix explicit_u(const Matrix& x,const Matrix& t,const Vector& alpha,double sigma,const Options& o) {
    Matrix u(x.rows(),t.rows());
    for(Index i=0;i<x.rows();++i) {
        for(Index j=0;j<t.rows();++j) u(i,j)=std::exp(-(x.row(i)-t.row(j)).squaredNorm()/(o.entropy*sigma))*alpha[j];
        u.row(i)/=u.row(i).sum();
    }
    return u;
}
}
int main() {
    test("official_zero_variance_is_not_clamped",[] {
        Matrix points=Matrix::Zero(8,2);
        Options o; o.semantics=Semantics::Official; o.normalize=false;
        o.align_centroids=false; o.rank=1;
        auto result=fit(points,points,o);
        small(std::abs(result.sigma2),0,"raw_sigma2");
        check(result.history.empty(),"official loop must not run at zero variance");
        check(result.stop_reason=="variance_floor","official variance stop");
    });
    test("L1_kernel_not_L2_and_PSD",[] {
        Matrix y=fixture(19,3),k=laplacian_kernel(y,y,2);
        small((k-k.transpose()).norm(),1e-14,"symmetry"); small((k.diagonal().array()-1).abs().maxCoeff(),1e-14,"unit_diagonal");
        Eigen::SelfAdjointEigenSolver<Matrix> e(k); check(e.eigenvalues().minCoeff()>-1e-12,"PSD");
        Matrix a(1,2),b(1,2); a<<0,0; b<<1,1; small(std::abs(laplacian_kernel(a,b,2)(0,0)-std::exp(-4)),1e-15,"Manhattan_value");
    });
    for(auto semantics:{Semantics::Paper,Semantics::Official}) test(semantics==Semantics::Paper?"streaming_vs_dense_paper_N_ne_M_d5":"streaming_vs_dense_official_N_ne_M_d5",[=] {
        Matrix x=fixture(23,5),t=fixture(17,5)*0.9;
        Vector a=Vector::LinSpaced(17,0.01,1); a/=a.sum();
        Options o; o.semantics=semantics; auto s=expectation(x,t,a,0.8,o);
        o.estep=EStep::Dense; auto z=expectation(x,t,a,0.8,o);
        small((s.mass-z.mass).norm(),1e-12,"mass_l2"); small((s.px-z.px).norm(),1e-12,"px_frobenius");
        small(std::abs(s.entropy-z.entropy),1e-12,"entropy"); small(std::abs(s.sse_old-z.sse_old),1e-12,"sse");
        o.estep=EStep::Streaming; o.threads=4; auto p=expectation(x,t,a,0.8,o);
        small((s.px-p.px).norm(),1e-12,"parallel_px");
        if(semantics==Semantics::Paper) small(s.row_sum_error,1e-14,"row_normalization");
        else small(std::abs(s.row_sum_error-17e-10),2e-14,"documented_epsilon_mass");
    });
    test("log_sum_exp_underflow_and_extinct_component",[] {
        Matrix t=fixture(9,3),x=fixture(7,3); x.array()+=1000;
        Vector a=Vector::Constant(9,1.0/8); a[0]=0;
        Options o; auto s=expectation(x,t,a,1e-8,o);
        check(s.px.allFinite() && std::isfinite(s.entropy),"finite output"); small(s.row_sum_error,1e-14,"row_error"); small(s.mass[0],0,"extinct_component_mass");
    });
    test("reduced_solve_vs_dense_same_approximate_kernel",[] {
        Matrix y=fixture(27,3),x=fixture(31,3)*0.9,q=fixture(27,8)*0.2;
        Options o; auto s=expectation(x,y,Vector::Constant(27,1.0/27),0.7,o);
        auto a=reduced_update(y,q,s,0.07); auto b=dense_update(y,q*q.transpose(),s,0.07);
        small((a.transformed-b.transformed).norm(),1e-11,"T_frobenius"); small(std::abs(a.penalty-b.penalty),1e-10,"regularizer"); small(a.residual,1e-12,"normal_equation_residual");
    });
    test("SSE_sufficient_statistics_vs_explicit_double_sum",[] {
        Matrix y=fixture(13,2),x=fixture(21,2),q=fixture(13,5)*0.1;
        Options o; Vector a=Vector::Constant(13,1.0/13); auto s=expectation(x,y,a,0.9,o); auto z=reduced_update(y,q,s,0.2);
        Matrix u=explicit_u(x,y,a,0.9,o); long double direct=0;
        for(Index i=0;i<x.rows();++i) for(Index j=0;j<y.rows();++j) direct+=u(i,j)*(x.row(i)-z.transformed.row(j)).squaredNorm();
        small(std::abs(updated_sse(s,y,z.transformed)-double(direct)),1e-12,"SSE_error");
    });
    test("regularized_objective_gradient_finite_difference",[] {
        Matrix y=fixture(11,2),x=fixture(15,2),q=fixture(11,4)*0.2,b=fixture(4,2)*0.03;
        Options o; double sigma=0.8; auto s=expectation(x,y,Vector::Constant(11,1.0/11),sigma,o);
        Matrix dq=q.array().colwise()*s.mass.array(); Matrix g=s.px-(y.array().colwise()*s.mass.array()).matrix();
        Matrix gradient=2/sigma*(q.transpose()*dq*b-q.transpose()*g)+2*o.regularization*b;
        Matrix fd(b.rows(),b.cols()); const double h=1e-6;
        auto loss=[&](const Matrix& v) { Matrix t=y+q*v; return updated_sse(s,y,t)/sigma+o.regularization*v.squaredNorm(); };
        for(Index j=0;j<b.cols();++j) for(Index i=0;i<b.rows();++i) { Matrix plus=b,minus=b; plus(i,j)+=h; minus(i,j)-=h; fd(i,j)=(loss(plus)-loss(minus))/(2*h); }
        small((fd-gradient).norm(),1e-7,"finite_difference_gradient");
    });
    test("Elkan_matches_Lloyd_same_initial_centers",[] {
        Matrix y=fixture(181,3),initial=kmeans_plus_plus(y,11,123);
        auto a=kmeans(y,initial,15,true,4),b=kmeans(y,initial,15,false,1);
        check(a.labels==b.labels,"identical_labels"); small((a.centers-b.centers).norm(),1e-12,"center_error");
        small(std::abs(a.quantization_error-b.quantization_error),1e-12,"quantization_error");
        std::cout<<"  elkan_distances="<<a.distance_evaluations<<" lloyd_distances="<<b.distance_evaluations<<'\n';
    });
    test("empty_cluster_policy_and_duplicate_points",[] {
        Matrix y=fixture(51,2),initial(5,2); initial.topRows(4)=y.topRows(4); initial.row(4)=initial.row(0);
        auto a=kmeans(y,initial,7,true),b=kmeans(y,initial,7,false);
        small((a.centers-b.centers).norm(),1e-12,"empty_cluster_parity"); check(a.centers.allFinite(),"finite_centers");
        Matrix repeated=Matrix::Ones(20,3); auto centers=kmeans_plus_plus(repeated,6,42); check(centers.rows()==1,"distinct_seed_count");
        Options o; o.rank=6; auto basis=make_basis(repeated,o); check(basis.retained_rank==1,"rank_one_duplicate_cloud");
    });
    test("Nystrom_full_landmarks_equals_full_kernel",[] {
        Matrix y=fixture(23,3); Options o; o.rank=23; o.eigen_cutoff=0;
        auto b=make_basis(y,o); auto k=laplacian_kernel(y,y,o.gamma);
        small((k-b.Q*b.Q.transpose()).norm()/k.norm(),1e-12,"relative_kernel_error");
    });
    test("Nystrom_EWinvET_equals_QQt_and_empirical_bound",[] {
        Matrix y=fixture(51,3); Options o; o.rank=9; o.eigen_cutoff=0;
        auto b=make_basis(y,o); auto k=laplacian_kernel(y,y,o.gamma),w=laplacian_kernel(b.centers,b.centers,o.gamma),e=laplacian_kernel(y,b.centers,o.gamma);
        Matrix winv=w.inverse(),approx=e*winv*e.transpose();
        small((approx-b.Q*b.Q.transpose()).norm(),1e-11,"factorization_error");
        double q=b.clustering.quantization_error, t=b.clustering.max_cluster_size, r=b.centers.rows();
        double bound=4*std::sqrt(2.0)*std::pow(t,1.5)*o.gamma*std::sqrt(r*q)+2*r*o.gamma*o.gamma*t*q*winv.norm();
        std::cout<<"  empirical_error="<<(k-approx).norm()<<" printed_bound="<<bound<<'\n';
        check((k-approx).norm()<=bound,"bound on this fixture only, NOT a theorem proof");
    });
    test("supplement_L2_Lipschitz_constant_needs_dimension_factor",[] {
        // a=b=0, c=h*1, d=-h*1: ratio tends to dimension.
        const double h=1e-6,gamma=2,dim=3;
        const double lhs=std::pow(1-std::exp(-gamma*2*dim*h),2);
        const double rhs_printed=2*gamma*gamma*(2*dim*h*h);
        const double ratio=lhs/rhs_printed;
        std::cout<<"  lhs_over_2gamma2_L2_bound="<<ratio<<'\n';
        check(ratio>2.9,"counterexample to dimension-free intermediate constant"); check(lhs<=dim*rhs_printed,"dimension-corrected Lipschitz inequality");
    });
    test("all_four_coordinate_updates_descend_paper_objective",[] {
        SyntheticOptions s; s.count=150; s.noise=0.01; auto data=make_synthetic(s);
        Options o; o.rank=45; o.max_iterations=30; o.fixed_iterations=true; auto r=fit(data.source,data.target,o);
        double worst=0,prev=std::numeric_limits<double>::infinity();
        for(const auto& v:r.history) {
            double eps=1e-9*(1+std::abs(v.after_u));
            check(v.after_alpha<=v.after_u+eps,"alpha descent"); check(v.after_deform<=v.after_alpha+eps,"deformation descent"); check(v.after_variance<=v.after_deform+eps,"variance descent");
            if(std::isfinite(prev)) check(v.after_u<=prev+eps,"membership descent");
            worst=std::max(worst,v.after_variance-prev); prev=v.after_variance;
            check(std::abs(v.alpha_sum-1)<1e-12 && v.row_sum_error<1e-12,"simplex constraints");
        }
        small(worst,1e-8,"worst_objective_increase");
    });
    test("end_to_end_dense_vs_full_rank_Nystrom",[] {
        SyntheticOptions s; s.count=48; s.dimension=3; s.noise=0.015; auto data=make_synthetic(s);
        Options o; o.rank=48; o.max_iterations=12; o.fixed_iterations=true; o.eigen_cutoff=0;
        auto a=fit(data.source,data.target,o); o.solver=Solver::Dense; auto b=fit(data.source,data.target,o);
        small((a.transformed-b.transformed).norm(),1e-8,"end_to_end_T");
    });
    test("target_permutation_invariance_and_unequal_counts",[] {
        SyntheticOptions s; s.count=61; s.dimension=3; s.missing_fraction=0.2; auto data=make_synthetic(s);
        Options o; o.rank=20; o.max_iterations=10; o.fixed_iterations=true;
        auto a=fit(data.source,data.target,o); Matrix reverse=data.target.colwise().reverse(); auto b=fit(data.source,reverse,o);
        small((a.transformed-b.transformed).norm(),1e-8,"shuffled_target_T"); check(data.source.rows()!=data.target.rows(),"unequal counts exercised");
    });
    test("parallel_registration_and_out_of_sample_transform",[] {
        SyntheticOptions s; s.count=120; s.dimension=3; auto data=make_synthetic(s);
        Options o; o.rank=36; o.max_iterations=12; o.fixed_iterations=true;
        auto a=fit(data.source,data.target,o); o.threads=4; auto b=fit(data.source,data.target,o);
        small((a.transformed-b.transformed).norm(),1e-8,"one_vs_four_threads");
        small((transform(a,data.source)-a.transformed).norm(),1e-9,"field_at_training_points");
        check(transform(a,fixture(8,3)).allFinite(),"new_query_points");
    });
    test("generic_dimension_5_and_scale_equivariance",[] {
        Matrix y=fixture(33,5),x=fixture(29,5)*0.9;
        Options o; o.rank=10; o.max_iterations=8; o.fixed_iterations=true;
        auto a=fit(y,x,o);
        Matrix sy=(y.array()*7+13).matrix(),sx=(x.array()*11-9).matrix(); auto b=fit(sy,sx,o);
        small((b.transformed-(a.transformed.array()*11-9).matrix()).norm(),1e-8,"scale_equivariance");
    });
    test("coincident_clouds_and_variance_floor",[] {
        Matrix y=Matrix::Zero(12,3),x=Matrix::Constant(17,3,2);
        Options o; o.rank=4; o.max_iterations=5; auto r=fit(y,x,o);
        check(r.transformed.allFinite(),"finite degenerate solution"); small((r.transformed.array()-2).matrix().norm(),1e-12,"degenerate_translation"); check(r.sigma2>=o.sigma_floor,"variance_floor");
    });
    test("sinkhorn_one_pair_closed_form_and_variable_mass",[] {
        Matrix y=Matrix::Zero(1,2),x=Matrix::Zero(1,2);
        Options o; o.algorithm=Algorithm::Sinkhorn; o.backend=Backend::CPU;
        o.normalize=false; o.align_centroids=false; o.rank=1;
        o.initial_sigma=0.3; o.sigma_floor=0.3; o.sigma_ceiling=0.3;
        o.max_iterations=1; o.fixed_iterations=true;
        o.sinkhorn_iterations=1000; o.sinkhorn_tolerance=1e-13;
        auto r=fit(y,x,o);
        const double cost=std::log(2*std::acos(-1.0)*r.sigma2);
        const double lambda=o.transport_entropy+o.source_mass_penalty+o.target_mass_penalty;
        const double expected=std::exp(-cost/lambda);
        small(std::abs(r.transport_mass-expected),2e-12,"one_pair_mass");
        small(std::abs(r.source_mass.sum()-r.transport_mass),2e-14,"source_mass_sum");
        small(std::abs(r.target_mass.sum()-r.transport_mass),2e-14,"target_mass_sum");
        check(std::abs(r.transport_mass-1)>1e-3,"UOT mass must not be normalized");
        check(r.history.front().objective_after<=r.history.front().objective_before+1e-12,"Gaussian M-step descent");
    });
    test("sinkhorn_common_normalization_student_MM_and_transform",[] {
        SyntheticOptions s; s.count=36; s.dimension=3; s.outlier_fraction=0.2; s.seed=77;
        auto data=make_synthetic(s);
        Options o; o.algorithm=Algorithm::Sinkhorn; o.noise_model=NoiseModel::StudentT;
        o.backend=Backend::CPU; o.rank=10; o.max_iterations=3; o.fixed_iterations=true;
        o.sinkhorn_iterations=150; o.sinkhorn_tolerance=1e-9; o.student_dof=4;
        auto r=fit(data.source,data.target,o);
        small((r.source_center-r.target_center).norm(),0,"shared_center");
        small(std::abs(r.source_scale-r.target_scale),0,"shared_scale");
        small((transform(r,data.source)-r.transformed).norm(),2e-11,"sinkhorn_transform_training_points");
        check(r.backend_used==Backend::CPU,"explicit CPU backend");
        check(r.history.size()==3,"fixed outer iterations");
        bool distinct=false;
        for(const auto& v:r.history) {
            check(v.objective_after<=v.objective_before+2e-8*(1+std::abs(v.objective_before)),"Student MM descent");
            check(std::isfinite(v.kkt_residual) && v.transport_mass>0 && v.robust_mass>0,"finite transport diagnostics");
            distinct=distinct || std::abs(v.robust_mass-v.transport_mass)>1e-6;
        }
        check(distinct,"Student robust mass must remain distinct from Gamma mass");
        small(std::abs(r.source_mass.sum()-r.target_mass.sum()),2e-12,"marginal_total_agreement");
    });
    test("reject_invalid_inputs",[] {
        Matrix y=fixture(12,3),x=fixture(15,3);
        expect_error([&] { fit(Matrix(0,3),x); }); expect_error([&] { fit(y,fixture(15,2)); });
        Matrix nan=y; nan(0,0)=std::numeric_limits<double>::quiet_NaN(); expect_error([&] { fit(nan,x); });
        Options o; o.entropy=0; expect_error([&] { fit(y,x,o); }); o=Options{}; o.rank=-1; expect_error([&] { fit(y,x,o); }); o=Options{}; o.threads=0; expect_error([&] { fit(y,x,o); });
        o=Options{}; o.algorithm=Algorithm::Sinkhorn; o.rank=-1; expect_error([&] { fit(y,x,o); });
    });
    test("CSV_and_ASCII_PLY_roundtrip",[] {
        auto dir=std::filesystem::temp_directory_path()/"clusterreg_roundtrip_test";
        check(!std::filesystem::exists(dir),"temporary test directory already exists"); std::filesystem::create_directory(dir);
        try {
            Matrix y=fixture(17,3); write_points((dir/"points.csv").string(),y); small((read_points((dir/"points.csv").string())-y).norm(),1e-14,"CSV_roundtrip");
            write_points((dir/"points.ply").string(),y); small((read_points((dir/"points.ply").string())-y).norm(),1e-14,"PLY_roundtrip");
        } catch(...) { std::filesystem::remove_all(dir); throw; }
        std::filesystem::remove_all(dir);
    });
    std::cout<<"SUMMARY passed="<<passed<<" failed="<<failed<<'\n'; return failed?1:0;
}
