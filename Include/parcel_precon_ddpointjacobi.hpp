// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include "parcel_nblock.hpp"
#ifndef PARCEL_PRECON_DDPOINTJACOBI_
#define PARCEL_PRECON_DDPOINTJACOBI_

template<class T_vec,class T_mat>
void ddpointjacobi(
		 sycl::queue *q_,
		 PARCEL_INT N,
		 T_vec *x,
		 T_mat *Dinv,
		 T_vec *y
		 ){

#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(T_vec));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_hi =  x[i];
	  double x_lo =  0.0;
	  double y_hi =  0.0;
	  double y_lo =  0.0;
	  double Dinv_hi =  Dinv[i];
	  double Dinv_lo =  0.0;
	  double ddmul_var1;
	  ddmul_calc1(x_hi,x_lo,Dinv_hi,Dinv_lo,y_hi,y_lo);
	  y[i] = y_hi + y_lo;
	}
      }
    });

  return ;

}

template<class T_mat>
void ddpointjacobi(
		 sycl::queue *q_,
		 PARCEL_INT N,
		 double *x_hi_,double *x_lo_,
		 T_mat *Dinv,
		 double *y_hi_,double *y_lo_
		 ){

#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 ;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_hi =  x_hi_[i];
	  double x_lo =  x_lo_[i];
	  double y_hi =  0.0;
	  double y_lo =  0.0;
	  double Dinv_hi =  Dinv[i];
	  double Dinv_lo =  0.0;
	  double ddmul_var1;
	  ddmul_calc1(x_hi,x_lo,Dinv_hi,Dinv_lo,y_hi,y_lo);
	  y_hi_[i] = y_hi;
	  y_lo_[i] = y_lo;
	}
      }
    });

  return ;
}

template<class T_vec>
void ddpointjacobi(
		 sycl::queue *q_,
		 PARCEL_INT N,
		 T_vec *x,
		 double *Dinv_hi_,double *Dinv_lo_,
		 T_vec *y
		 ){

#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(T_vec));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_hi =  x[i];
	  double x_lo =  0.0;
	  double y_hi =  0.0;
	  double y_lo =  0.0;
	  double Dinv_hi =  Dinv_hi_[i];
	  double Dinv_lo =  Dinv_lo_[i];
	  double ddmul_var1;
	  ddmul_calc1(x_hi,x_lo,Dinv_hi,Dinv_lo,y_hi,y_lo);
	  y[i] = y_hi + y_lo;
	}
      }
    });

  return ;
}

void ddpointjacobi(
		 sycl::queue *q_,
		 PARCEL_INT N,
		 double *x_hi_,double *x_lo_,
		 double *Dinv_hi_,double *Dinv_lo_,
		 double *y_hi_,double *y_lo_
		 ){

#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_hi =  x_hi_[i];
	  double x_lo =  x_lo_[i];
	  double y_hi =  0.0;
	  double y_lo =  0.0;
	  double Dinv_hi =  Dinv_hi_[i];
	  double Dinv_lo =  Dinv_lo_[i];
	  double ddmul_var1;
	  ddmul_calc1(x_hi,x_lo,Dinv_hi,Dinv_lo,y_hi,y_lo);
	  y_hi_[i] = y_hi;
	  y_lo_[i] = y_lo;
	}
      }
    });

  return ;
}

#endif
