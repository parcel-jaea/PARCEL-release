#ifndef MIXED_PRECISON_AXPY_HPP_
#define MIXED_PRECISON_AXPY_HPP_

#if _BLOCKING_LOOP_
#include "mixed_precision_axpy_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_AXPY_A64FX_HPP_

template<class TdataA,class TdataB,class TdataX,class TdataY,class Tcalc>
void axpy(
	  PARCEL_INT N,
	  TdataA *alpha,
	  TdataB *beta,
	  TdataX *x,
	  PARCEL_INT ldx,
	  TdataY *y,
	  PARCEL_INT ldy){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      Tcalc x_ =  x[i];
      Tcalc y_ =  y[i];
      y_   = beta_*y_ + alpha_*x_;
      y[i] = y_;
    });
}

template<class TdataA,class TdataB,class TdataX,class TdataY,class TdataZ,class Tcalc>
void axpy(
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
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      Tcalc x_ =  x[i];
      Tcalc y_ =  y[i];
      y_   = beta_*y_ + alpha_*x_;
      z[i] = y_;
    });
}

template<class Tdata>
void copy(
	  PARCEL_INT N,
	  Tdata *x,
	  PARCEL_INT ldx,
	  Tdata *y,
	  PARCEL_INT ldy){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      y[i] = x[i];
    });
}

template<class Tdata>
void zero(
	  PARCEL_INT N,
	  Tdata *x,
	  PARCEL_INT ldx
	  ){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      x[i] = 0.0;
    });
}

#endif

#endif
