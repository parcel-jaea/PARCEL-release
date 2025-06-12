#ifndef PBICG_HPP_
#define PBICG_HPP_

#include <iostream>
#include "mat.hpp"
#include "vec.hpp"
#include "krylov_precon.hpp"

template<class T>
class pbicg{

  std::vector<T> r_;
  std::vector<T> r0_;
  std::vector<T> p_;
  std::vector<T> q_;
  std::vector<T> pe_;
  std::vector<T> v_;
  std::vector<T> pp_;
  std::vector<T> e_;

  vec<T> vec_;
  krylov_precon<T> precon_;
  PARCEL_INT itr_=0;
  // 相対残差ノルム
  T rnorm_ = 0.0; // ||r|| / ||b||

  PARCEL_INT itrmax_ =100;
  PARCEL_INT itrmax_jacobi_ = 4;
  T residualnorm_ = 1.0e-8;

public:
  //  pcg(crs_mat<T> &crsA){
  pbicg(){}
  ~pbicg(){}

  void set_residualnorm(T residualnorm){
    residualnorm_ = residualnorm;
  }
  void set_itrmax(PARCEL_INT itrmax){
    itrmax_ = itrmax;
  }

  PARCEL_INT get_itrationNumber(){
    return itr_;
  }
  
  T get_residualnorm(){
    return rnorm_;
  }

  void preconditioning(
		     std::vector<T> &x,
		     std::vector<T> &y
		     // std::vector<T> &x,
		     // std::vector<T> &y
		       ){
    // 前処理無し
    vec_.axpy(1.0,0.0,x,y);
    // ヤコビ前処理
    //precon_.jacobi_method(y,x,itrmax_jacobi_);
    // 対称ガウスザイデル
    //precon_.Symmetric_GaussSeidel_method(y,x,itrmax_jacobi_);
  }

  void solve(
	     crs_mat<T> &crsA,
	     std::vector<T> &x,
	     std::vector<T> &b
	     ){
    precon_.init(crsA);
    PARCEL_INT nRow = crsA.getValue_nRow();
    //    precon_.init(nRow);
    r_.resize(nRow,0.0);
    r0_.resize(nRow,0.0);
    p_.resize(nRow,0.0);
    q_.resize(nRow,0.0);
    pe_.resize(nRow,0.0);
    v_.resize(nRow,0.0);
    pp_.resize(nRow,0.0);
    e_.resize(nRow,0.0);
    
    // pbicg algorithm
    T bnorm = vec_.norm(b,b);
    // r = b - Ax
    crsA.SpMV(x,r_);
    vec_.axpy(1.0,-1.0,b,r_);
    rnorm_ = vec_.norm(r_,r_);rnorm_ = rnorm_ / bnorm;
    std::cout<<0<<" : residual norm = "<<rnorm_<<std::endl;

    // ! r0=r
    vec_.axpy(1.0,0.0,r_,r0_);
    // ! p=r
    vec_.axpy(1.0,0.0,r_,p_);

    // ! c1=<r0,r0>
    T c1 = vec_.dot(r0_,r0_);
    for(PARCEL_INT itr = 0;itr < itrmax_;itr++){
      // !   solve Mpp=p
      preconditioning(p_,pp_);
      // !   q=App
      crsA.SpMV(pp_,q_);
      // !   c2=<r0,q>
      T c2 = vec_.dot(r0_,q_);
      // !   alpha = c1/c2
      T alpha = c1/c2;
      // !   e=r-alpha*q
      vec_.axpy(1.0   ,0.0,r_,e_);
      vec_.axpy(-alpha,1.0,q_,e_);
      // !   solve Mpe=e
      preconditioning(e_,pe_);
      // !   v=Ape
      crsA.SpMV(pe_,v_);
      // !   c3=<e,v>/<v,v>
      T ev = vec_.dot(e_,v_);
      T vv = vec_.dot(v_,v_);
      T c3 = ev / vv;
      // !   x=x+alpha*pp+c3*pe
      vec_.axpy(alpha,1.0,pp_,x);
      vec_.axpy(c3,1.0,pe_,x);
      // !   r=e-c3*v
      vec_.axpy(1.0,0.0,e_,r_);
      vec_.axpy(-c3,1.0,v_,r_);

      // !   c1=<r0,r>
      c1 = vec_.dot(r0_,r_);

      rnorm_ = vec_.norm(r_,r_);rnorm_ = rnorm_ / bnorm;
      std::cout<<itr+1<<" : residual norm = "<<rnorm_<<std::endl;
      // !   if(||r||<rmax) exit
      if(rnorm_<residualnorm_)break;

      // !   beta=c1/(c3*c2)
      T beta = c1/(c3*c2);
      // !   p=r+beta*(p-c3*q)
      vec_.axpy(-c3,1.0,q_,p_);
      vec_.axpy(1.0,beta,r_,p_);
    }

  }  

};

#endif
