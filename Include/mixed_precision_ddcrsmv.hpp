// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDCRSMV_HPP_
#define MIXED_PRECISON_DDCRSMV_HPP_

#if _BLOCKING_LOOP_SPMV_
#include "mixed_precision_ddcrsmv_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_DDCRSMV_A64FX_HPP_
// ------------------------
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi ,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddmath dd;
      ddtype Ax(0.0,0.0);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A_hi[j],A_lo[j]);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_  = A_*x_;
	ddtype Ax0  = Ax+Ax_;
	Ax = Ax0;
      }
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_  = alpha_*Ax;
      ddtype by_  = beta_ *y_;
      ddtype yx   = ax_+by_;
      y_hi[i] = yx.get_hi();
      y_lo[i] = yx.get_lo();
    });

  return ;
}

template< class TdataA ,class TdataB >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta[0],0.0);
      ddtype alpha_(alpha[0],0.0);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A_hi[j],A_lo[j]);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;
      y_hi[i] = yx.get_hi();
      y_lo[i] = yx.get_lo();
    });
  return ;
}

template< class TdataCRS >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     TdataCRS *A,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_  = alpha_*Ax;
      ddtype by_  = beta_ *y_;
      ddtype yx   = ax_ + by_;
      y_hi[i] = yx.get_hi();
      y_lo[i] = yx.get_lo();
    });

  return ;
}

template< class TdataA ,class TdataB,  class TdataCRS  >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     TdataCRS *A,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta[0],0.0);
      ddtype alpha_(alpha[0],0.0);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;
      y_hi[i] = yx.get_hi();
      y_lo[i] = yx.get_lo();
    });
  return ;
}

template< class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A_hi[j],A_lo[j]);
	ddtype x_(x[col],0.0);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y[i],0.0);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;
      y[i] = yx.get_T<TdataY>();
    });

  return ;
}

template< 
  class TdataA ,class TdataB,class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     double *A_hi,double *A_lo,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x,
	     PARCEL_INT ldx,
	     TdataB *beta,
	     TdataY *y,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta[0],0.0);
      ddtype alpha_(alpha[0],0.0);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A_hi[j],A_lo[j]);
	ddtype x_(x[col],0.0);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y[i],0.0);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;
      y[i] = yx.get_T<TdataY>();
    });

  return ;
}


template< class TdataCRS, class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     TdataCRS *A,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
  ddmath dd;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x[col],0.0);
	ddtype Ax_  = A_ * x_;
	ddtype Ax0  = Ax + Ax_;
	Ax = Ax0;
      }
      ddtype y_(y[i],0.0);
      ddtype ax_  = alpha_*Ax;
      ddtype by_  = beta_ *y_;
      ddtype yx   = ax_ + by_;
      y[i] = yx.get_T<TdataY>();
    });

  return ;
}



template< class TdataA ,class TdataB , class TdataCRS, class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     TdataCRS *A,
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol,
	     TdataX *x,
	     PARCEL_INT ldx,
	     TdataB *beta,
	     TdataY *y,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddtype Ax(0.0,0.0);
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_LINT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x[col],0.0);
	Ax = Ax + A_ * x_;
      }
      ddtype beta_(beta[0],0.0);
      ddtype alpha_(alpha[0],0.0);

      ddtype y_(y[i],0.0);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;

      y[i] = yx.get_T<TdataY>();
    });
  return ;
}

#endif

#endif


