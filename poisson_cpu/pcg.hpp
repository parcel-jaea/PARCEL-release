#ifndef PCG_HPP_
#define PCG_HPP_

// #include<quadmath.h>

#include <iomanip>
#include <iostream>
#include "mat.hpp"
#include "vec.hpp"
#include "krylov_precon.hpp"

template<class T>
class pcg{

  std::vector<T> r_;
  std::vector<T> q_;
  std::vector<T> p_;
  std::vector<T> s_;

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
  pcg(){}
  ~pcg(){}

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
    q_.resize(nRow,0.0);
    p_.resize(nRow,0.0);
    s_.resize(nRow,0.0);

    // pcg algorithm
#if 1
    T bnorm = vec_.dot(b,b);
    bnorm = std::sqrt((double)bnorm);
#else
    T bnorm = vec_.norm(b,b);
#endif
    // r = b - Ax
    crsA.SpMV(x,r_);
    vec_.axpy(1.0,-1.0,b,r_);
#if 1
    rnorm_ = vec_.dot(r_,r_);
    rnorm_ = std::sqrt((double)rnorm_);

#else
    rnorm_ = vec_.norm(r_,r_);
#endif

    //    std::cout<<std::scientific << std::setprecision(14)<<" rnorm,bnorm = "<<rnorm_<<" , "<<bnorm<<std::endl; 
    std::cout<<"bnorm = "<<std::scientific << std::setprecision(14) << (double) bnorm  <<std::endl;
    std::cout<<"rnorm = "<<std::scientific << std::setprecision(14) << (double) rnorm_ <<std::endl;

    rnorm_ = rnorm_ / bnorm;
    std::cout<<std::setw(8)<<0;
    std::cout<<" : residual norm = "<<(double)rnorm_<<std::endl;

    // p=0.0 
    vec_.axpy(0.0,0.0,p_,p_);
    // solve Ms=r 
    preconditioning(r_,s_);
    // rho0=<s,r> 
    T rho0 = vec_.dot(s_,r_);

    // alpha=0.0 
    // beta=0.0 
    T alpha = 0.0 ;
    T beta  = 0.0 ;
    T gamma = 0.0;
    T rho1  = 0.0;

    // for itr =1,.... 
    for(PARCEL_INT itr = 0;itr < itrmax_;itr++){
    //    std::cout<<"            alpha, beta, rho0 = "<< alpha<< "," << beta << "," << rho0 <<std::endl;
    //    for(PARCEL_INT itr = 0;itr < 2;itr++){
    //    for(PARCEL_INT itr = 0;itr < 10;itr++){
      // write(*,*)"     rho1_FP64  = ",rho1_FP64  
      // 	write(*,*)"     beta_FP64  = ",beta_FP64  
      // 	write(*,*)"     alpha_FP64 = ",alpha_FP64 
      // 	write(*,*)"     gamma_FP64 = ",gamma_FP64 
#if 0
      std::cout<<std::scientific << std::setprecision(14)<<"rho0  = " << (double)rho0  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"rho1  = " << (double)rho1  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"beta  = " << (double)beta  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"alpha = " << (double)alpha << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"gamma = " << (double)gamma << std::endl;  
#endif	
      //      x = x + alpha*p 
      vec_.axpy(alpha,1.0,p_,x);
      //      p = s + beta * p 
      vec_.axpy(1.0,beta,s_,p_);
      //      q = Ap 
      crsA.SpMV(p_,q_);
      //      gamma = <p,q> 
      gamma = vec_.dot(p_,q_);
      //      alpha=rho0/gamma 
      alpha = rho0/gamma;

      //      std::cout<<"            alpha, beta, gamma = "<< alpha<< "," << beta << "," << gamma <<std::endl;

      //      r = r - alpha * q 
      vec_.axpy(-alpha,1.0,q_,r_);
      //      solve Ms=r 
      preconditioning(r_,s_);
      //      rho1 = <s,r> 
      rho1 = vec_.dot(s_,r_);
      //      if(||r||<rmax) exit 
#if 1
      rnorm_ = vec_.dot(r_,r_);
      //      std::cout<<std::scientific << std::setprecision(14)<<itr+1<<"      rdot  = "<<(double)rnorm_<<std::endl;
      rnorm_ = std::sqrt((double)rnorm_);
      //      std::cout<<std::scientific << std::setprecision(14)<<itr+1<<"      rnorm = "<<(double)rnorm_<<std::endl;
      rnorm_ = rnorm_/ bnorm;
#else
      rnorm_ = vec_.norm(r_,r_);rnorm_ = rnorm_ / bnorm;
#endif
      
#if 0
      char str[128];
      T res = rnorm_;
      quadmath_snprintf(str,128,"%.40Qf",res);
      printf(" %d residual norm = %s \n",itr+1,str);
#else
      double res = rnorm_;
      std::cout<<std::setw(8)<<itr+1;
      std::cout<<std::scientific << std::setprecision(14)<<" : residual norm = "<<res<<std::endl;
#endif

      if(rnorm_<residualnorm_)break;
      //     beta = rho1/rho0 
      beta = rho1/rho0 ;
      //      rho0=rho1
      rho0=rho1; 
      itr_ = itr;
    }
    // end for 
    // x = x + alpha*p 
    vec_.axpy(alpha,1.0,p_,x);

    // residual
    // crsA.SpMV(x,r_);
    // vec_.axpy(1.0,-1.0,b,r_);
    // rnorm_ = vec_.norm(r_,r_);
    // rnorm_ = rnorm_ / bnorm;
    // std::cout<<" Final norm = "<<(double)rnorm_<<std::endl;

  }

  void solve(
	     dia_mat<T> &diaA,
	     std::vector<T> &x,
	     std::vector<T> &b
	     ){
    //    precon_.init(diaA);
    PARCEL_INT nRow = diaA.getValue_nRow();
    //    precon_.init(nRow);
    r_.resize(nRow,0.0);
    q_.resize(nRow,0.0);
    p_.resize(nRow,0.0);
    s_.resize(nRow,0.0);

    // pcg algorithm
#if 1
    T bnorm = vec_.dot(b,b);
    bnorm = std::sqrt((double)bnorm);
#else
    T bnorm = vec_.norm(b,b);
#endif
    // r = b - Ax
    diaA.SpMV(x,r_);
    vec_.axpy(1.0,-1.0,b,r_);
#if 1
    rnorm_ = vec_.dot(r_,r_);
    rnorm_ = std::sqrt((double)rnorm_);

#else
    rnorm_ = vec_.norm(r_,r_);
#endif

    //    std::cout<<std::scientific << std::setprecision(14)<<" rnorm,bnorm = "<<rnorm_<<" , "<<bnorm<<std::endl; 
    std::cout<<"bnorm = "<<std::scientific << std::setprecision(14) << (double) bnorm  <<std::endl;
    std::cout<<"rnorm = "<<std::scientific << std::setprecision(14) << (double) rnorm_ <<std::endl;

    rnorm_ = rnorm_ / bnorm;
    std::cout<<std::setw(8)<<0;
    std::cout<<" : residual norm = "<<(double)rnorm_<<std::endl;

    // p=0.0 
    vec_.axpy(0.0,0.0,p_,p_);
    // solve Ms=r 
    //    preconditioning(r_,s_);
    vec_.axpy(1.0,0.0,r_,s_);

    // rho0=<s,r> 
    T rho0 = vec_.dot(s_,r_);

    // alpha=0.0 
    // beta=0.0 
    T alpha = 0.0 ;
    T beta  = 0.0 ;
    T gamma = 0.0;
    T rho1  = 0.0;

    // for itr =1,.... 
    for(PARCEL_INT itr = 0;itr < itrmax_;itr++){
    //    std::cout<<"            alpha, beta, rho0 = "<< alpha<< "," << beta << "," << rho0 <<std::endl;
    //    for(PARCEL_INT itr = 0;itr < 2;itr++){
    //    for(PARCEL_INT itr = 0;itr < 10;itr++){
      // write(*,*)"     rho1_FP64  = ",rho1_FP64  
      // 	write(*,*)"     beta_FP64  = ",beta_FP64  
      // 	write(*,*)"     alpha_FP64 = ",alpha_FP64 
      // 	write(*,*)"     gamma_FP64 = ",gamma_FP64 
#if 0
      std::cout<<std::scientific << std::setprecision(14)<<"rho0  = " << (double)rho0  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"rho1  = " << (double)rho1  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"beta  = " << (double)beta  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"alpha = " << (double)alpha << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"gamma = " << (double)gamma << std::endl;  
#endif	
      //      x = x + alpha*p 
      vec_.axpy(alpha,1.0,p_,x);
      //      p = s + beta * p 
      vec_.axpy(1.0,beta,s_,p_);
      //      q = Ap 
      diaA.SpMV(p_,q_);
      //      gamma = <p,q> 
      gamma = vec_.dot(p_,q_);
      //      alpha=rho0/gamma 
      alpha = rho0/gamma;

      //      std::cout<<"            alpha, beta, gamma = "<< alpha<< "," << beta << "," << gamma <<std::endl;

      //      r = r - alpha * q 
      vec_.axpy(-alpha,1.0,q_,r_);
      //      solve Ms=r 
      //      preconditioning(r_,s_);
      vec_.axpy(1.0,0.0,r_,s_);
      //      rho1 = <s,r> 
      rho1 = vec_.dot(s_,r_);
      //      if(||r||<rmax) exit 
#if 1
      rnorm_ = vec_.dot(r_,r_);
      //      std::cout<<std::scientific << std::setprecision(14)<<itr+1<<"      rdot  = "<<(double)rnorm_<<std::endl;
      rnorm_ = std::sqrt((double)rnorm_);
      //      std::cout<<std::scientific << std::setprecision(14)<<itr+1<<"      rnorm = "<<(double)rnorm_<<std::endl;
      rnorm_ = rnorm_/ bnorm;
#else
      rnorm_ = vec_.norm(r_,r_);rnorm_ = rnorm_ / bnorm;
#endif
      
#if 0
      char str[128];
      T res = rnorm_;
      quadmath_snprintf(str,128,"%.40Qf",res);
      printf(" %d residual norm = %s \n",itr+1,str);
#else
      double res = rnorm_;
      std::cout<<std::setw(8)<<itr+1;
      std::cout<<std::scientific << std::setprecision(14)<<" : residual norm = "<<res<<std::endl;
#endif

      if(rnorm_<residualnorm_)break;
      //     beta = rho1/rho0 
      beta = rho1/rho0 ;
      //      rho0=rho1
      rho0=rho1; 
      itr_ = itr;
    }
    // end for 
    // x = x + alpha*p 
    vec_.axpy(alpha,1.0,p_,x);

    // residual
    // diaA.SpMV(x,r_);
    // vec_.axpy(1.0,-1.0,b,r_);
    // rnorm_ = vec_.norm(r_,r_);
    // rnorm_ = rnorm_ / bnorm;
    // std::cout<<" Final norm = "<<(double)rnorm_<<std::endl;

  }



};

#endif
