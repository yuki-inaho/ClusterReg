// SPDX-License-Identifier: AGPL-3.0-only
#include "clusterreg/io.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
namespace clusterreg {
namespace {
std::vector<double> numbers(std::string s) {
    std::replace(s.begin(),s.end(),',',' ');
    std::istringstream in(s); std::string word; std::vector<double> row;
    while(in>>word) {
        if(word[0]=='#') break;
        std::size_t count=0; double v=std::stod(word,&count);
        if(count!=word.size() || !std::isfinite(v)) throw std::runtime_error("Non-numeric or nonfinite coordinate");
        row.push_back(v);
    }
    return row;
}
}
Matrix read_points(const std::string& name) {
    std::ifstream file(name); if(!file) throw std::runtime_error("Cannot open "+name);
    std::string line; std::getline(file,line); if(!line.empty() && line.back()=='\r') line.pop_back();
    std::vector<std::vector<double>> rows;
    if(line=="ply") {
        int vertices=-1,property_count=0; bool ascii=false,in_vertex=false,header_done=false;
        std::vector<int> coordinates(3,-1);
        while(std::getline(file,line)) {
            std::istringstream p(line); std::string tag; p>>tag;
            if(tag=="format") { std::string format; p>>format; ascii=format=="ascii"; }
            if(tag=="element") { std::string kind; int count; p>>kind>>count; in_vertex=kind=="vertex"; if(in_vertex) vertices=count; else if(vertices<0 && count>0) throw std::runtime_error("ASCII PLY requires vertex element first"); }
            if(tag=="property" && in_vertex) {
                std::string type,key; p>>type>>key;
                if(type=="list") throw std::runtime_error("List-valued vertex properties unsupported");
                if(key=="x") coordinates[0]=property_count;
                if(key=="y") coordinates[1]=property_count;
                if(key=="z") coordinates[2]=property_count;
                ++property_count;
            }
            if(tag=="end_header") { header_done=true; break; }
        }
        if(!ascii || !header_done || vertices<=0 || coordinates[0]<0 || coordinates[1]<0)
            throw std::runtime_error("Only nonempty ASCII PLY with x,y[,z] is supported");
        for(int i=0;i<vertices;++i) {
            if(!std::getline(file,line)) throw std::runtime_error("Truncated PLY");
            auto all=numbers(line); if(static_cast<int>(all.size())!=property_count) throw std::runtime_error("PLY property count mismatch");
            std::vector<double> row{all[coordinates[0]],all[coordinates[1]]};
            if(coordinates[2]>=0) row.push_back(all[coordinates[2]]);
            rows.push_back(std::move(row));
        }
    } else {
        do { auto row=numbers(line); if(!row.empty()) rows.push_back(std::move(row)); } while(std::getline(file,line));
    }
    if(rows.empty()) throw std::runtime_error("Empty point file: "+name);
    Matrix result(static_cast<Index>(rows.size()),static_cast<Index>(rows[0].size()));
    for(Index i=0;i<result.rows();++i) {
        if(rows[static_cast<std::size_t>(i)].size()!=rows[0].size()) throw std::runtime_error("Inconsistent dimensions in "+name);
        for(Index d=0;d<result.cols();++d) result(i,d)=rows[static_cast<std::size_t>(i)][static_cast<std::size_t>(d)];
    }
    return result;
}
void write_points(const std::string& name,const Matrix& p) {
    std::ofstream out(name); if(!out) throw std::runtime_error("Cannot write "+name);
    out<<std::setprecision(17);
    bool ply=name.size()>=4 && name.substr(name.size()-4)==".ply";
    if(ply) {
        if(p.cols()!=2 && p.cols()!=3) throw std::runtime_error("PLY output supports 2D or 3D only");
        out<<"ply\nformat ascii 1.0\nelement vertex "<<p.rows()<<"\nproperty double x\nproperty double y\n";
        if(p.cols()==3) out<<"property double z\n";
        out<<"end_header\n";
    }
    for(Index i=0;i<p.rows();++i) { for(Index j=0;j<p.cols();++j) { if(j) out<<(ply?' ':','); out<<p(i,j); } out<<'\n'; }
    if(!out) throw std::runtime_error("Failed while writing "+name);
}
void write_history(const std::string& name,const std::vector<Iteration>& h) {
    std::ofstream f(name); if(!f) throw std::runtime_error("Cannot write history");
    f<<std::setprecision(17)<<"iteration,sigma_before,sigma_after,after_u,after_alpha,after_deform,after_variance,official_loss,row_sum_error,alpha_sum,linear_residual,step_rms,estep_seconds,solve_seconds,objective_before,objective_after,transport_mass,robust_mass,transport_residual,kkt_residual,scale_relative_change,transport_iterations\n";
    for(const auto& v:h) f<<v.index<<','<<v.sigma_before<<','<<v.sigma_after<<','<<v.after_u<<','<<v.after_alpha<<','<<v.after_deform<<','<<v.after_variance<<','<<v.official_loss<<','<<v.row_sum_error<<','<<v.alpha_sum<<','<<v.linear_residual<<','<<v.step_rms<<','<<v.estep_seconds<<','<<v.solve_seconds<<','<<v.objective_before<<','<<v.objective_after<<','<<v.transport_mass<<','<<v.robust_mass<<','<<v.transport_residual<<','<<v.kkt_residual<<','<<v.scale_relative_change<<','<<v.transport_iterations<<'\n';
}
double corresponding_rmse(const Matrix& a,const Matrix& b) {
    if(a.rows()!=b.rows() || a.cols()!=b.cols()) throw std::invalid_argument("Ground truth must have source ordering and shape");
    return (a-b).norm()/std::sqrt(double(a.rows()));
}
double nearest_rmse(const Matrix& from,const Matrix& to) {
    if(from.rows()==0 || to.rows()==0 || from.cols()!=to.cols()) throw std::invalid_argument("Invalid nearest neighbor inputs");
    long double total=0;
    for(Index i=0;i<from.rows();++i) {
        Vector dist=Vector::Zero(to.rows());
        for(Index d=0;d<from.cols();++d) dist.array()+=(to.col(d).array()-from(i,d)).square();
        total+=dist.minCoeff();
    }
    return std::sqrt(double(total/from.rows()));
}
} // namespace clusterreg
