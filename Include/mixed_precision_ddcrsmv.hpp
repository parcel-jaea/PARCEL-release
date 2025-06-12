#ifndef MIXED_PRECISON_DDCRSMV_HPP_
#define MIXED_PRECISON_DDCRSMV_HPP_

#if _BLOCKING_LOOP_
#include "mixed_precision_ddcrsmv_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_DDCRSMV_A64FX_HPP_
// ------------------------
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi ,double *y_lo,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;

  //            std::cout<<"dd SpMV"<<std::endl;

  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      ddmath dd;
#if 0
      // kahan
      ddtype Ax(0.0,0.0);
      ddtype c(0.0,0.0);
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
	ddtype A_(A_hi[j]  ,A_lo[j]);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_ = A_ * x_;
	ddtype y = Ax_ - c;
	ddtype t = Ax + y;
	c = t-Ax;
	c = c-y;
	Ax = t;
      }
#else
      ddtype Ax(0.0,0.0);
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
	ddtype A_(A_hi[j],A_lo[j]);
	ddtype x_(x_hi[col],x_lo[col]);
	ddtype Ax_  = A_*x_;
	ddtype Ax0  = Ax+Ax_;
	Ax = Ax0;
      }
#endif
      ddtype beta_(beta_hi[0],beta_lo[0]);
      ddtype alpha_(alpha_hi[0],alpha_lo[0]);
      ddtype y_(y_hi[i],y_lo[i]);
      ddtype ax_  = alpha_*Ax;
      ddtype by_  = beta_ *y_;
      ddtype yx   = ax_+by_;
      y_hi[i] = yx.get_hi();
      y_lo[i] = yx.get_lo();
      //   y[i] = yx.get_T<TdataY>();
    });

  return ;
}

template< class TdataA ,class TdataB >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      //   y[i] = yx.get_T<TdataY>();
    });
  return ;
}

template< class TdataCRS >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     TdataCRS *A,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      //   y[i] = yx.get_T<TdataY>();
    });

  return ;
}

template< class TdataA ,class TdataB,  class TdataCRS  >
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     TdataA *alpha,
	     TdataCRS *A,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      //   y[i] = yx.get_T<TdataY>();
    });
  return ;
}

template< class TdataX, class TdataY>
void ddcrsmv(
	     PARCEL_INT N,
	     PARCEL_INT NNZ,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      // y_hi[i] = yx.get_hi();
      // y_lo[i] = yx.get_lo();
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
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      // y_hi[i] = yx.get_hi();
      // y_lo[i] = yx.get_lo();
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
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
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
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
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
      // y_hi[i] = yx.get_hi();
      // y_lo[i] = yx.get_lo();
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
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol,
	     TdataX *x,
	     PARCEL_INT ldx,
	     TdataB *beta,
	     TdataY *y,
	     PARCEL_INT ldy,
	     double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;

  //      std::cout<<"dd SpMV"<<std::endl;
  //      ddmath dd;
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#if 0
      // kahan
      ddtype Ax(0.0,0.0);
      ddtype c(0.0,0.0);
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x[col],0.0);
	ddtype Ax_  = A_ * x_;
	ddtype y = Ax_ - c;
	ddtype t = Ax  + y;
	c = t - Ax;
	c = c - y;
	Ax = t;
	//	    ddtype Ax0  = dd.add(Ax,Ax_);
	//Ax = Ax0;
      }
#else
      ddtype Ax(0.0,0.0);
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	PARCEL_INT col = crsCol[j];
	ddtype A_(A[j],0.0);
	ddtype x_(x[col],0.0);
	Ax = Ax + A_ * x_;
      }
#endif
      ddtype beta_(beta[0],0.0);
      ddtype alpha_(alpha[0],0.0);

      ddtype y_(y[i],0.0);
      ddtype ax_  = alpha_ * Ax;
      ddtype by_  = beta_  * y_;
      ddtype yx   = ax_ + by_;

      // y_hi[i] = yx.get_hi();
      // y_lo[i] = yx.get_lo();
      //	  y[i] = yx.get_hi();
      //	  y[i] = yx.get_hi();
      y[i] = yx.get_T<TdataY>();
      //	  	  y[i] = 9.99;
    });
#if 0
  q_[0].wait();
  q_[0].memcpy(y_,y,sizeof(TdataY)*N);
  q_[0].wait();
  for(int i=0;i<N;i++){
    std::cout<<"output : "<<i<<" : "<<y_[i]<<std::endl;
  }
#endif
  return ;
}

#endif

#endif


