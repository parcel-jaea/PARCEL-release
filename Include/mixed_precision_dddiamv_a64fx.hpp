// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDDIAMV_A64FX_HPP_
#define MIXED_PRECISON_DDDIAMV_A64FX_HPP_

#include "parcel_nblock.hpp"

void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi ,double *y_lo,
	     PARCEL_INT ldy){      
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i]     ,y_lo[i]     ,y_hi[i]     ,y_lo[i]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i],y_lo[i],by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo[i]);
	  }
	}
      }
    });
	    
  return ;
}


template< class TdataA ,class TdataB >
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     TdataA *alpha_hi,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy 
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double alpha_lo_ = 0.0;
	    double beta_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i]     ,y_lo[i]     ,y_hi[i]     ,y_lo[i]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double alpha_lo_ = 0.0;
	    double beta_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i],y_lo[i],by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo[i]);
	  }
	}
      }
    });
	    
  return ;
}

template< class TdataDIA >
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     double *alpha_hi,double *alpha_lo,
	     TdataDIA *A_hi,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double A_lo_=0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i]     ,y_lo[i]     ,y_hi[i]     ,y_lo[i]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double A_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double A_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i],y_lo[i],by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo[i]);
	  }
	}
      }
    });
	    
  return ;
}

template< class TdataA ,class TdataB,  class TdataDIA  >
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     TdataA *alpha_hi,
	     TdataDIA *A_hi,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     double *x_hi,double *x_lo,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     double *y_hi,double *y_lo,
	     PARCEL_INT ldy 
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double A_lo_=0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double alpha_lo_ = 0.0;
	    double beta_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i]     ,y_lo[i]     ,y_hi[i]     ,y_lo[i]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],y_hi[i],y_lo[i],y_hi[i],y_lo[i]);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double A_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double A_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo[i+offset[j]],Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double alpha_lo_ = 0.0;
	    double beta_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i],y_lo[i],by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo[i]);
	  }
	}
      }
    });
	    
  return ;

}

template< class TdataX, class TdataY>
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     double *alpha_hi,double *alpha_lo,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y_hi,
	     PARCEL_INT ldy
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double x_lo_ = 0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }

	  double by_hi[nlocalblock];
	  double by_lo[nlocalblock];
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i]     ,y_lo_     ,by_hi[i-iii],by_lo[i-iii]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    double y_lo_ = 0.0;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double x_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double x_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double y_lo_ =0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i],y_lo_,by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo_);
	  }
	}
      }
    });
	    
  return ;
}

template<class TdataA ,class TdataB,class TdataX, class TdataY>
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     TdataA *alpha_hi,
	     double *A_hi,double *A_lo,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     TdataY *y_hi,
	     PARCEL_INT ldy
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double x_lo_ = 0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }

	  double by_hi[nlocalblock];
	  double by_lo[nlocalblock];
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double alpha_lo_ = 0.0;
	    double beta_lo_  = 0.0;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i]     ,y_lo_     ,by_hi[i-iii],by_lo[i-iii]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    double y_lo_ = 0.0;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double x_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double x_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo[i+j*N],x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double alpha_lo_ = 0.0;
	    double beta_lo_  = 0.0;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i],y_lo_,by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo_);
	  }
	}
      }
    });
	    
  return ;
}

template< class TdataDIA, class TdataX, class TdataY>
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     double *alpha_hi,double *alpha_lo,
	     TdataDIA *A_hi,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     double *beta_hi,double *beta_lo,
	     TdataY *y_hi,
	     PARCEL_INT ldy
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double A_lo_ = 0.0;
	      double x_lo_ = 0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }
	  double by_hi[nlocalblock];
	  double by_lo[nlocalblock];
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i]     ,y_lo_       ,by_hi[i-iii],by_lo[i-iii]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    double y_lo_ = 0.0;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double A_lo_ = 0.0;
		  double x_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double A_lo_ = 0.0;
		double x_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }
	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo[0],Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo[0] ,y_hi[i],y_lo_,by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo_);
	  }
	}
      }
    });
	    
  return ;
}

template< class TdataA,class TdataB, class TdataDIA, class TdataX, class TdataY>
void dddiamv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     TdataA *alpha_hi,
	     TdataDIA *A_hi,
	     PARCEL_INT *offset,
	     PARCEL_INT max_up_offset,
	     PARCEL_INT max_down_offset,
	     TdataX *x_hi,
	     PARCEL_INT ldx,
	     TdataB *beta_hi,
	     TdataY *y_hi,
	     PARCEL_INT ldy
	     ){
  size_t MAX   = N;

  // y = α*Ax + β*y;
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  double Ax_hi[nlocalblock];
	  double Ax_lo[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax_hi[i] = 0.0;
	    Ax_lo[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      double ddadd_var1;
	      double ddmul_var1;
	      double Ax_hi_,Ax_lo_;
	      double A_lo_ = 0.0;
	      double x_lo_ = 0.0;
	      ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
	      ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    }
	  }

	  double by_hi[nlocalblock];
	  double by_lo[nlocalblock];
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddmul_var1;
	    double ddmul_var2;
	    double alpha_lo_ = 0.0;
	    double beta_lo_  = 0.0;
	    double y_lo_ = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_ ,Ax_hi[i-iii],Ax_lo[i-iii],Ax_hi[i-iii],Ax_lo[i-iii]);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i]     ,y_lo_        ,by_hi[i-iii],by_lo[i-iii]);
	  }
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double ddadd_var1;
	    double y_lo_ = 0.0;
	    ddadd_calc1(Ax_hi[i-iii],Ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo_);
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    double Ax_hi = 0.0;
	    double Ax_lo = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  double ddadd_var1;
		  double ddmul_var1;
		  double Ax_hi_,Ax_lo_;
		  double A_lo_ = 0.0;
		  double x_lo_ = 0.0;
		  ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		  ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		double ddadd_var1;
		double ddmul_var1;
		double Ax_hi_,Ax_lo_;
		double A_lo_ = 0.0;
		double x_lo_ = 0.0;
		ddmul_calc1(A_hi[i+j*N],A_lo_,x_hi[i+offset[j]],x_lo_,Ax_hi_,Ax_lo_);
		ddadd_calc1(Ax_hi_,Ax_lo_,Ax_hi,Ax_lo,Ax_hi,Ax_lo);
	      }
	    }

	    double ddadd_var1;
	    double ddmul_var1;
	    double ddmul_var2;
	    double aAx_hi,aAx_lo;
	    double by_hi,by_lo;
	    double alpha_lo_ = 0.0;
	    double beta_lo_  = 0.0;
	    double y_lo_  = 0.0;
	    ddmul_calc1(alpha_hi[0],alpha_lo_,Ax_hi,Ax_lo, aAx_hi,aAx_lo);
	    ddmul_calc2(beta_hi[0] ,beta_lo_ ,y_hi[i],y_lo_,by_hi,by_lo);
	    ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,y_hi[i],y_lo_);
	  }
	}
      }
    });
	    
  return ;
}

#endif

