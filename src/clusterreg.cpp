// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/clusterreg.hpp"
#include "sinkhorn_internal.hpp"
#include <Eigen/Cholesky>
#include <Eigen/Eigenvalues>
#include <Eigen/LU>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace clusterreg {
namespace {
using Clock = std::chrono::steady_clock;
double seconds(Clock::time_point t) { return std::chrono::duration<double>(Clock::now()-t).count(); }
void require(bool ok, const std::string& msg) { if (!ok) throw std::invalid_argument(msg); }
void valid_points(const Matrix& p) {
    require(p.rows()>0 && p.cols()>0, "Point set must be nonempty");
    require(p.allFinite(), "Point set contains NaN or infinity");
}
int threads_used(int requested) {
#ifdef _OPENMP
    return std::max(1, requested);
#else
    (void)requested; return 1;
#endif
}
Vector logs(const Vector& alpha) {
    Vector r(alpha.size());
    for (Index j=0;j<alpha.size();++j) r[j]=alpha[j]>0 ? std::log(alpha[j]) : -std::numeric_limits<double>::infinity();
    return r;
}
// Eigen 3.4 packet exp clamps its negative tail. Use scalar libm there,
// preserving true zeros and subnormals rather than inventing tiny mass.
void exp_nonpositive(Vector& out, const Vector& log_values) {
    out=log_values.array().max(-700.0).exp().matrix();
    for(Index j=0;j<log_values.size();++j)
        if(log_values[j]<-700.0) out[j]=std::exp(log_values[j]);
}
double xlogx_sum(const Vector& v) {
    double total=0; for(Index j=0;j<v.size();++j) if(v[j]>0) total+=v[j]*std::log(v[j]); return total;
}
struct Worker {
    Statistics s;
    Vector distance, logits, weights;
    long double ent = 0, sse = 0, x2 = 0;
    Worker(Index n, Index d) : distance(n), logits(n), weights(n) {
        s.mass=Vector::Zero(n); s.px=Matrix::Zero(n,d);
    }
};
void normalize(Matrix& p, Eigen::RowVectorXd& mean, double& scale, bool enabled) {
    mean=Eigen::RowVectorXd::Zero(p.cols()); scale=1;
    if (!enabled) return;
    mean=p.colwise().mean(); p.rowwise()-=mean;
    scale=std::sqrt(p.squaredNorm()/double(p.rows()));
    // A coincident cloud has no meaningful isotropic scale, but is still usable.
    if (!(scale > 0)) scale=1;
    require(std::isfinite(scale), "Normalization scale overflow; rescale input first");
    p/=scale;
}
double log_alpha_term(const Vector& mass, const Vector& alpha) {
    long double sum=0;
    for(Index j=0;j<mass.size();++j) if(mass[j]>0) {
        if (!(alpha[j]>0)) return -std::numeric_limits<double>::infinity();
        sum+=static_cast<long double>(mass[j])*std::log(alpha[j]);
    }
    return double(sum);
}
} // anonymous
int available_threads() {
#ifdef _OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
}
Matrix laplacian_kernel(const Matrix& a, const Matrix& b, double gamma) {
    valid_points(a); valid_points(b);
    require(a.cols()==b.cols() && gamma>0 && std::isfinite(gamma), "Invalid kernel inputs");
    Matrix k(a.rows(),b.rows());
    for(Index j=0;j<b.rows();++j) {
        Vector dist=Vector::Zero(a.rows());
        for(Index d=0;d<a.cols();++d) dist.array()+=(a.col(d).array()-b(j,d)).abs();
        Vector logits=-gamma*dist, weights(dist.size());
        exp_nonpositive(weights,logits); k.col(j)=weights;
    }
    return k;
}
Matrix kmeans_plus_plus(const Matrix& data,int k,std::uint64_t seed) {
    valid_points(data); require(k>0 && k<=data.rows(), "Invalid landmark count");
    std::mt19937_64 rng(seed);
    Matrix centers(k,data.cols());
    centers.row(0)=data.row(static_cast<Index>(rng()%static_cast<std::uint64_t>(data.rows())));
    Vector nearest=Vector::Constant(data.rows(),std::numeric_limits<double>::infinity());
    int used=1;
    for(int j=1;j<k;++j) {
        for(Index i=0;i<data.rows();++i) nearest[i]=std::min(nearest[i],(data.row(i)-centers.row(j-1)).squaredNorm());
        double total=nearest.sum();
        if (!(total>0) || !std::isfinite(total)) break; // fewer distinct points
        double v=std::generate_canonical<double,53>(rng)*total;
        Index chosen=data.rows()-1;
        for(Index i=0;i<data.rows();++i) { v-=nearest[i]; if(v<0) { chosen=i; break; } }
        // Rounding at the extreme tail must not select an already used point.
        if(nearest[chosen]==0) nearest.maxCoeff(&chosen);
        centers.row(j)=data.row(chosen); ++used;
    }
    centers.conservativeResize(used,Eigen::NoChange);
    return centers;
}
KMeansResult kmeans(const Matrix& data,const Matrix& initial,int iterations,bool elkan,int requested_threads) {
    valid_points(data); valid_points(initial);
    require(data.cols()==initial.cols() && iterations>0, "Invalid k-means inputs");
    const Index n=data.rows(), k=initial.rows(), d=data.cols();
    KMeansResult result; result.centers=initial; result.labels.resize(static_cast<std::size_t>(n));
    Vector upper(n);
    Matrix lower;
    if(elkan) lower.resize(n,k);
    const int nt=threads_used(requested_threads);
    (void)nt;
    // Complete initial assignment: deterministic ties use the smallest center index.
    for(Index i=0;i<n;++i) {
        double best=std::numeric_limits<double>::infinity(); int label=0;
        for(Index j=0;j<k;++j) {
            double v=(data.row(i)-result.centers.row(j)).norm();
            if(elkan) lower(i,j)=v;
            if(v<best) { best=v; label=static_cast<int>(j); }
        }
        result.labels[static_cast<std::size_t>(i)]=label; upper[i]=best;
    }
    result.distance_evaluations=static_cast<std::uint64_t>(n*k);
    for(int it=0;it<iterations;++it) {
        Matrix next=Matrix::Zero(k,d);
        std::vector<Index> counts(static_cast<std::size_t>(k),0);
        // Serial ordered reduction: ND work, no shared OpenMP centroid races.
        for(Index i=0;i<n;++i) { int j=result.labels[static_cast<std::size_t>(i)]; next.row(j)+=data.row(i); ++counts[static_cast<std::size_t>(j)]; }
        Vector movement(k);
        for(Index j=0;j<k;++j) {
            if(counts[static_cast<std::size_t>(j)]>0) next.row(j)/=double(counts[static_cast<std::size_t>(j)]);
            else next.row(j)=result.centers.row(j); // explicit empty-cluster policy
            movement[j]=(next.row(j)-result.centers.row(j)).norm();
        }
        result.centers.swap(next);
        Matrix half;
        Vector separation;
        if(elkan) {
            half=Matrix::Constant(k,k,std::numeric_limits<double>::infinity());
            for(Index a=0;a<k;++a) for(Index b=0;b<a;++b)
                half(a,b)=half(b,a)=0.5*(result.centers.row(a)-result.centers.row(b)).norm();
            separation=half.rowwise().minCoeff();
            for(Index j=0;j<k;++j) lower.col(j)=(lower.col(j).array()-movement[j]).max(0.0).matrix();
            for(Index i=0;i<n;++i) upper[i]+=movement[result.labels[static_cast<std::size_t>(i)]];
        }
        std::uint64_t evaluated=0;
        int changed=0;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(nt) reduction(+:evaluated,changed)
#endif
        for(Index i=0;i<n;++i) {
            int old=result.labels[static_cast<std::size_t>(i)], label=old;
            if(elkan && upper[i]<separation[label]) continue;
            double best=elkan?upper[i]:std::numeric_limits<double>::infinity();
            bool tight=!elkan;
            for(Index j=0;j<k;++j) {
                if(elkan) {
                    if(j==label || best<lower(i,j) || best<half(label,j)) continue;
                    if(!tight) {
                        best=(data.row(i)-result.centers.row(label)).norm(); ++evaluated;
                        lower(i,label)=best; tight=true;
                    }
                    if(best<lower(i,j) || best<half(label,j)) continue;
                }
                double v=(data.row(i)-result.centers.row(j)).norm(); ++evaluated;
                if(elkan) lower(i,j)=v;
                if(v<best || (v==best && j<label)) { best=v; label=static_cast<int>(j); }
            }
            result.labels[static_cast<std::size_t>(i)]=label; upper[i]=best;
            if(label!=old) ++changed;
        }
        result.distance_evaluations+=evaluated;
        result.iterations=it+1;
        if(changed==0) break;
    }
    std::vector<int> counts(static_cast<std::size_t>(k),0);
    for(Index i=0;i<n;++i) {
        int j=result.labels[static_cast<std::size_t>(i)];
        result.quantization_error+=(data.row(i)-result.centers.row(j)).squaredNorm();
        ++counts[static_cast<std::size_t>(j)];
    }
    result.max_cluster_size=*std::max_element(counts.begin(),counts.end());
    return result;
}
Basis make_basis(const Matrix& source,const Options& options) {
    int r=options.rank>0?options.rank:static_cast<int>(std::ceil(options.rank_ratio*double(source.rows())));
    r=std::min<int>(r,static_cast<int>(source.rows()));
    require(r>0, "Rank must be positive");
    Basis b; b.requested_rank=r;
    if(r==source.rows()) {
        b.centers=source;
        b.clustering.centers=source; b.clustering.max_cluster_size=1;
    } else {
        Matrix initial=kmeans_plus_plus(source,r,options.seed);
        b.clustering=kmeans(source,initial,options.kmeans_iterations,true,options.threads);
        b.centers=b.clustering.centers;
    }
    Matrix w=laplacian_kernel(b.centers,b.centers,options.gamma);
    Matrix e=laplacian_kernel(source,b.centers,options.gamma);
    Eigen::SelfAdjointEigenSolver<Matrix> eig(w);
    if(eig.info()!=Eigen::Success) throw std::runtime_error("Landmark eigendecomposition failed");
    auto values=eig.eigenvalues();
    std::vector<Index> keep;
    for(Index j=0;j<values.size();++j) if(values[j]>options.eigen_cutoff) keep.push_back(j);
    require(!keep.empty(), "All landmark eigenvalues discarded; reduce eigen_cutoff");
    b.retained_rank=static_cast<int>(keep.size()); b.min_retained_eigenvalue=values[keep.front()];
    b.projection.resize(w.rows(),b.retained_rank);
    for(Index j=0;j<b.retained_rank;++j) b.projection.col(j)=eig.eigenvectors().col(keep[static_cast<std::size_t>(j)])/std::sqrt(values[keep[static_cast<std::size_t>(j)]]);
    b.Q.noalias()=e*b.projection;
    return b;
}
Statistics expectation(const Matrix& x,const Matrix& t,const Vector& alpha,double sigma2,const Options& o) {
    require(x.rows()>0 && x.cols()==t.cols() && t.rows()==alpha.size(),"Invalid expectation dimensions");
    require(x.allFinite() && t.allFinite() && alpha.allFinite() && (alpha.array()>=0).all() && alpha.sum()>0,"Invalid expectation data");
    require(sigma2>0 && std::isfinite(sigma2) && o.entropy>0 && std::isfinite(o.entropy),"Invalid expectation temperature");
    const Index n=t.rows(), m=x.rows(), d=x.cols();
    const double temperature=sigma2*o.entropy;
    require(temperature>0 && std::isfinite(temperature),"Temperature underflow or overflow");
    const double eps=o.semantics==Semantics::Official?1e-10:0.0;
    Vector la=logs(alpha);
    if(o.estep==EStep::Dense) {
        // Deliberately simple full-U reference, not the production hot path.
        Matrix u(m,n), distance(m,n);
        Statistics s;
        for(Index i=0;i<m;++i) {
            double maxlog=-std::numeric_limits<double>::infinity();
            for(Index j=0;j<n;++j) {
                distance(i,j)=(x.row(i)-t.row(j)).squaredNorm();
                u(i,j)=la[j]-distance(i,j)/temperature; maxlog=std::max(maxlog,u(i,j));
            }
            if(!std::isfinite(maxlog)) throw std::runtime_error("Nonfinite log probabilities; rescale input");
            double z=0;
            for(Index j=0;j<n;++j) { u(i,j)=std::exp(u(i,j)-maxlog); z+=u(i,j); }
            for(Index j=0;j<n;++j) u(i,j)=u(i,j)/z+eps;
        }
        s.mass=u.colwise().sum().transpose(); s.px.noalias()=u.transpose()*x;
        Vector rows=u.rowwise().sum();
        s.row_sum_error=(rows.array()-1).abs().maxCoeff();
        s.weighted_x2=rows.dot(x.rowwise().squaredNorm());
        long double ent=0, ss=0;
        for(Index j=0;j<n;++j) for(Index i=0;i<m;++i) {
            double v=u(i,j); if(v>0) ent+=static_cast<long double>(v)*std::log(v);
            ss+=static_cast<long double>(v)*distance(i,j);
        }
        s.entropy=double(ent); s.sse_old=double(ss); return s;
    }
    const int nt=std::min<int>(threads_used(o.threads),static_cast<int>(m));
    std::vector<Worker> workers; workers.reserve(static_cast<std::size_t>(nt));
    for(int q=0;q<nt;++q) workers.emplace_back(n,d);
    int invalid=0;
#ifdef _OPENMP
#pragma omp parallel num_threads(nt) reduction(|:invalid)
#endif
    {
        int id=0;
#ifdef _OPENMP
        id=omp_get_thread_num();
#endif
        Worker& w=workers[static_cast<std::size_t>(id)];
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
        for(Index i=0;i<m;++i) {
            w.distance.setZero();
            // Column-major points make source coordinates SIMD-contiguous.
            for(Index axis=0;axis<d;++axis) w.distance.array()+=(t.col(axis).array()-x(i,axis)).square();
            w.logits=la-w.distance/temperature;
            const double maxlog=w.logits.maxCoeff();
            if(!std::isfinite(maxlog)) { invalid=1; continue; }
            w.logits.array()-=maxlog;
            exp_nonpositive(w.weights,w.logits);
            const double z=w.weights.sum();
            w.weights/=z;
            if(eps>0) w.weights.array()+=eps;
            w.s.mass+=w.weights;
            for(Index axis=0;axis<d;++axis) w.s.px.col(axis).noalias()+=w.weights*x(i,axis);
            const double rowsum=w.weights.sum();
            w.s.row_sum_error=std::max(w.s.row_sum_error,std::abs(rowsum-1));
            w.x2+=static_cast<long double>(rowsum)*x.row(i).squaredNorm();
            w.sse+=w.weights.dot(w.distance);
            if(eps>0) w.ent+=(w.weights.array()*w.weights.array().log()).sum();
            else {
                // Avoid 0*(-infinity) for an extinct component; no artificial mass.
                double ent=0, lz=std::log(z);
                for(Index j=0;j<n;++j) if(w.weights[j]>0) ent+=w.weights[j]*(w.logits[j]-lz);
                w.ent+=ent;
            }
        }
    }
    if(invalid) throw std::runtime_error("Nonfinite log probabilities; rescale input");
    Statistics s; s.mass=Vector::Zero(n); s.px=Matrix::Zero(n,d);
    long double ent=0,ss=0,x2=0;
    for(const auto& w:workers) {
        s.mass+=w.s.mass; s.px+=w.s.px;
        s.row_sum_error=std::max(s.row_sum_error,w.s.row_sum_error);
        ent+=w.ent; ss+=w.sse; x2+=w.x2;
    }
    s.entropy=double(ent); s.sse_old=double(ss); s.weighted_x2=double(x2);
    return s;
}
Update reduced_update(const Matrix& y,const Matrix& q,const Statistics& s,double kappa) {
    require(kappa>0 && std::isfinite(kappa),"Regularization times variance must be positive");
    Matrix dq=q.array().colwise()*s.mass.array();
    Matrix g=s.px-(y.array().colwise()*s.mass.array()).matrix();
    Matrix a=q.transpose()*dq;
    a.diagonal().array()+=kappa;
    Matrix rhs=q.transpose()*g;
    Eigen::LLT<Matrix> factor(a);
    if(factor.info()!=Eigen::Success) throw std::runtime_error("Reduced Cholesky failed; reduce rank or increase variance floor");
    Update out; out.coefficients=factor.solve(rhs);
    out.transformed=y+q*out.coefficients;
    out.penalty=out.coefficients.squaredNorm();
    out.residual=(a*out.coefficients-rhs).norm()/std::max(1e-30,rhs.norm());
    if(!out.transformed.allFinite()) throw std::runtime_error("Nonfinite reduced solution");
    return out;
}
Update dense_update(const Matrix& y,const Matrix& kernel,const Statistics& s,double kappa) {
    require(kappa>0 && std::isfinite(kappa),"Regularization times variance must be positive");
    // (D K + kappa I) C = U^T X - D Y. No D^-1, so zero-mass rows are safe.
    Matrix a=kernel.array().colwise()*s.mass.array();
    a.diagonal().array()+=kappa;
    Matrix rhs=s.px-(y.array().colwise()*s.mass.array()).matrix();
    Update out; out.coefficients=a.partialPivLu().solve(rhs);
    out.transformed=y+kernel*out.coefficients;
    out.penalty=(out.coefficients.array()*(kernel*out.coefficients).array()).sum();
    out.residual=(a*out.coefficients-rhs).norm()/std::max(1e-30,rhs.norm());
    if(!out.transformed.allFinite()) throw std::runtime_error("Nonfinite dense solution");
    return out;
}
double updated_sse(const Statistics& s,const Matrix& old_t,const Matrix& new_t) {
    // Exact sufficient-statistic identity centered at old_t, avoiding subtraction
    // of large global coordinate moments. Final scalar reduction uses long double.
    long double ss=s.sse_old, magnitude=std::abs(s.sse_old);
    for(Index axis=0;axis<old_t.cols();++axis) for(Index j=0;j<old_t.rows();++j) {
        const long double delta=static_cast<long double>(new_t(j,axis))-old_t(j,axis);
        const long double value=static_cast<long double>(s.mass[j])*delta*delta-
            2*delta*(static_cast<long double>(s.px(j,axis))-static_cast<long double>(s.mass[j])*old_t(j,axis));
        ss+=value; magnitude+=std::abs(value);
    }
    if(ss < -1e-10L*(1+magnitude)) throw std::runtime_error("Negative weighted SSE beyond rounding tolerance");
    return std::max(0.0,double(ss));
}
double objective(double sse,double sigma2,const Statistics& s,const Vector& alpha,double penalty,Index m,Index d,const Options& o) {
    return sse/sigma2+double(m*d)*std::log(sigma2)+
        o.entropy*(s.entropy-log_alpha_term(s.mass,alpha))+o.regularization*penalty;
}
Result fit(const Matrix& source,const Matrix& target,const Options& o) {
    if(o.algorithm==Algorithm::Sinkhorn)
        return detail::fit_sinkhorn(source,target,o);
    auto start=Clock::now(); valid_points(source); valid_points(target);
    require(source.cols()==target.cols(),"Source and target dimensions differ");
    require(o.entropy>0 && std::isfinite(o.entropy) && o.regularization>0 && std::isfinite(o.regularization),"Weights must be finite and positive");
    require(o.gamma>0 && std::isfinite(o.gamma) && o.rank>=0 && o.rank_ratio>0 && o.rank_ratio<=1,"Invalid kernel or rank options");
    require(o.sigma_floor>0 && std::isfinite(o.sigma_floor) && o.tolerance>=0 && std::isfinite(o.tolerance),"Invalid stopping options");
    require(o.eigen_cutoff>=0 && std::isfinite(o.eigen_cutoff),"Invalid eigenvalue cutoff");
    require(o.max_iterations>0 && o.kmeans_iterations>0 && o.threads>0,"Iteration counts and threads must be positive");
    Result r; r.options=o; r.prepared_source=source; r.prepared_target=target;
    normalize(r.prepared_source,r.source_center,r.source_scale,o.normalize);
    normalize(r.prepared_target,r.target_center,r.target_scale,o.normalize);
    auto& y=r.prepared_source; auto& x=r.prepared_target;
    const Index n=y.rows(),m=x.rows(),d=y.cols();
    r.source_shift=Eigen::RowVectorXd::Zero(d);
    if(o.align_centroids) { r.source_shift=x.colwise().mean()-y.colwise().mean(); y.rowwise()+=r.source_shift; }
    Matrix kernel;
    if(o.solver==Solver::Nystrom) r.basis=make_basis(y,o);
    else kernel=laplacian_kernel(y,y,o.gamma);
    // O((N+M)d) rather than building either self-Gram or all-pairs distance matrix.
    double sigma2=(x.squaredNorm()/double(m)+y.squaredNorm()/double(n)-
        2*x.colwise().mean().dot(y.colwise().mean()))/double(d);
    // The constrained paper objective clamps variance.  Upstream's official
    // loop instead keeps the raw value and tests sigma2 > 1e-8 at its head.
    if(o.semantics==Semantics::Paper) sigma2=std::max(sigma2,o.sigma_floor);
    require(std::isfinite(sigma2),"Initial variance overflow; rescale inputs");
    Vector alpha=Vector::Constant(n,o.semantics==Semantics::Official?1.0:1.0/double(n));
    Matrix t=y;
    r.coefficients=Matrix::Zero(o.solver==Solver::Nystrom?r.basis.Q.cols():n,d);
    double penalty=0,previous=std::numeric_limits<double>::infinity(), official_previous=1;
    r.preparation_seconds=seconds(start);
    auto loop_start=Clock::now(); r.stop_reason="max_iterations";
    for(int it=0;it<o.max_iterations;++it) {
        if(o.semantics==Semantics::Official && sigma2<=o.sigma_floor) { r.stop_reason="variance_floor"; break; }
        Iteration log; log.index=it+1; log.sigma_before=sigma2;
        auto e_start=Clock::now(); Statistics stats=expectation(x,t,alpha,sigma2,o); log.estep_seconds=seconds(e_start);
        log.after_u=objective(stats.sse_old,sigma2,stats,alpha,penalty,m,d,o);
        alpha=stats.mass/double(m);
        if(o.semantics==Semantics::Official) alpha.array()+=1e-10;
        log.after_alpha=objective(stats.sse_old,sigma2,stats,alpha,penalty,m,d,o);
        auto solve_start=Clock::now();
        Update u=o.solver==Solver::Nystrom?reduced_update(y,r.basis.Q,stats,o.regularization*sigma2):dense_update(y,kernel,stats,o.regularization*sigma2);
        log.solve_seconds=seconds(solve_start);
        const double sse_new=updated_sse(stats,t,u.transformed);
        log.after_deform=objective(sse_new,sigma2,stats,alpha,u.penalty,m,d,o);
        log.official_loss=stats.sse_old/sigma2+double(m*d)*std::log(sigma2)+0.5*o.regularization*penalty+
            o.entropy*(stats.entropy-double(m)*xlogx_sum(alpha));
        double variance_residual=o.semantics==Semantics::Official?stats.sse_old:sse_new;
        const double raw_next_sigma=variance_residual/double(m*d);
        const double next_sigma=o.semantics==Semantics::Paper?
            std::max(raw_next_sigma,o.sigma_floor):raw_next_sigma;
        log.sigma_after=next_sigma;
        log.after_variance=next_sigma>0?
            objective(sse_new,next_sigma,stats,alpha,u.penalty,m,d,o):
            std::numeric_limits<double>::infinity();
        log.row_sum_error=stats.row_sum_error; log.alpha_sum=alpha.sum(); log.linear_residual=u.residual;
        log.step_rms=(t-u.transformed).norm()/std::sqrt(double(n));
        const double criterion=o.semantics==Semantics::Official?
            std::abs((log.official_loss-official_previous)/log.official_loss):
            std::abs(log.after_variance-previous)/(1+std::abs(previous));
        t.swap(u.transformed); r.coefficients.swap(u.coefficients); penalty=u.penalty;
        sigma2=next_sigma; previous=log.after_variance; official_previous=log.official_loss;
        r.history.push_back(log);
        if(!o.fixed_iterations && criterion<=o.tolerance) { r.stop_reason="tolerance"; break; }
        if(!o.fixed_iterations && o.semantics==Semantics::Paper && sigma2<=o.sigma_floor && log.step_rms<=std::sqrt(o.sigma_floor)*0.1) { r.stop_reason="variance_floor"; break; }
    }
    r.iteration_seconds=seconds(loop_start);
    r.alpha=alpha; r.sigma2=sigma2; r.normalized_transformed=t;
    r.transformed=t*r.target_scale; r.transformed.rowwise()+=r.target_center;
    r.total_seconds=seconds(start);
    return r;
}
Matrix transform(const Result& model,const Matrix& query) {
    valid_points(query); require(query.cols()==model.prepared_source.cols(),"Query dimension mismatch");
    Matrix q=query; q.rowwise()-=model.source_center; q/=model.source_scale; q.rowwise()+=model.source_shift;
    Matrix displacement;
    if(model.options.solver==Solver::Nystrom)
        displacement=laplacian_kernel(q,model.basis.centers,model.options.gamma)*(model.basis.projection*model.coefficients);
    else displacement=laplacian_kernel(q,model.prepared_source,model.options.gamma)*model.coefficients;
    q+=displacement; q*=model.target_scale; q.rowwise()+=model.target_center;
    return q;
}
} // namespace clusterreg
