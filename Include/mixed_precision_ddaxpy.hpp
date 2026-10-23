// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDAXPY_HPP_
#define MIXED_PRECISON_DDAXPY_HPP_

#if _BLOCKING_LOOP_
#include "mixed_precision_ddaxpy_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_DDAXPY_A64FX_HPP_
void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi,double *beta_lo,
	    double *x_hi,double *x_lo,
	    PARCEL_INT ldx,
	    double *y_hi,double *y_lo,
	    PARCEL_INT ldy){      
  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype x_(x_hi[i],x_lo[i]);
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_    = alpha_*x_;
      ddtype by_    = beta_*y_;
      ddtype y_ax_  = by_+ax_; 
      y_hi[i] = y_ax_.get_hi();
      y_lo[i] = y_ax_.get_lo();
    });
}

template<class TdataA,class TdataB,class TdataX,class TdataY>
void ddaxpy(
	    PARCEL_INT N,
	    TdataA *alpha,
	    TdataB *beta,
	    TdataX *x,
	    PARCEL_INT ldx,
	    TdataY *y,
	    PARCEL_INT ldy){      

  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha[0]);
      ddtype beta_(beta[0]);
      ddtype x_(x[i]);
      ddtype y_(y[i]);
      ddtype ax_    = alpha_ * x_;
      ddtype by_    = beta_  * y_;
      ddtype y_ax_  = by_ + ax_; 
      y[i] = y_ax_.get_T<TdataY>();
    });
}


template<class TdataX,class TdataY>
void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi,double *beta_lo,
	    TdataX *x,
	    PARCEL_INT ldx,
	    TdataY *y,
	    PARCEL_INT ldy){      
  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype x_(x[i],0.0);
      ddtype y_(y[i],0.0);
      ddtype ax_    = alpha_*x_;
      ddtype by_    = beta_*y_;
      ddtype y_ax_  = by_+ax_; 
      y[i] = y_ax_.get_hi();
    });
}

void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi,double *beta_lo,
	    double *x_hi,double *x_lo,
	    PARCEL_INT ldx,
	    double *y_hi,double *y_lo,
	    PARCEL_INT ldy,
	    double *z_hi,double *z_lo,
	    PARCEL_INT ldz
	    ){      
  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype x_(x_hi[i],x_lo[i]);
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_    = alpha_*x_;
      ddtype by_    = beta_*y_;
      ddtype y_ax_  = by_+ax_; 
      z_hi[i] = y_ax_.get_hi();
      z_lo[i] = y_ax_.get_lo();
    });
}

template<class TdataA,class TdataB,class TdataX,class TdataY,class TdataZ>
void ddaxpy(
	    PARCEL_INT N,
	    TdataA *alpha,
	    TdataB *beta,
	    TdataX *x,
	    PARCEL_INT ldx,
	    TdataY *y,
	    PARCEL_INT ldy,
	    TdataZ *z,
	    PARCEL_INT ldz
	    ){      

  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha[0]);
      ddtype beta_(beta[0]);
      ddtype x_(x[i]);
      ddtype y_(y[i]);
      ddtype ax_    = alpha_ * x_;
      ddtype by_    = beta_  * y_;
      ddtype y_ax_  = by_ + ax_; 
      z[i] = y_ax_.get_T<TdataY>();
    });
}


template<class TdataX,class TdataY,class TdataZ>
void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi,double *beta_lo,
	    TdataX *x,
	    PARCEL_INT ldx,
	    TdataY *y,
	    PARCEL_INT ldy,
	    TdataZ *z,
	    PARCEL_INT ldz
	    ){      
  size_t MAX   = N;
  ddmath dd;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype x_(x[i],0.0);
      ddtype y_(y[i],0.0);
      ddtype ax_    = alpha_*x_;
      ddtype by_    = beta_*y_;
      ddtype y_ax_  = by_+ax_; 
      z[i] = y_ax_.get_hi();
    });
}

#endif

#endif
