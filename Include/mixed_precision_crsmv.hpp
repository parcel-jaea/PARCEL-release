#ifndef MIXED_PRECISON_CRSMV_HPP_
#define MIXED_PRECISON_CRSMV_HPP_

#if _BLOCKING_LOOP_
#include "mixed_precision_crsmv_a64fx.hpp"
#endif

void crs_maxInv(
		PARCEL_INT N,
		double *A_hi,double *A_lo,
		double *AmaxInv_hi,double *AmaxInv_lo,
		PARCEL_INT *Rowptr,
		PARCEL_INT *crsCol
		){
#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 8192;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){	    
	  double Amax_hi = 1.0;
	  double Amax_lo = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    auto val_hi = std::abs(A_hi[j]);
	    auto val_lo = std::abs(A_lo[j]);
	    if( Amax_hi == val_hi ){
	      if( Amax_lo < val_lo ){
		Amax_hi = val_hi;
		Amax_lo = val_lo;
	      }
	    }else{
	      if( Amax_hi < val_hi ){
		Amax_hi = val_hi;
		Amax_lo = val_lo;
	      }
	    }
	  }
	  //		  AmaxInv[i] = 1.0/Amax;
	  double one_hi_ = 1.0;
	  double one_lo_ = 0.0;
	  double dddiv_var1;
	  dddiv_calc1(one_hi_,one_lo_,Amax_hi,Amax_lo,Amax_hi,Amax_lo);
	  AmaxInv_hi[i] = Amax_hi;
	  AmaxInv_lo[i] = Amax_lo;
	}
      }
    });
    
  return ;

}

template<
  class T_A,class T_Ainv
  >
void crs_maxInv(
	     PARCEL_INT N,
	     T_A *A,T_Ainv *AmaxInv,
	     PARCEL_INT *Rowptr,
	     PARCEL_INT *crsCol
	     ){
#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 8192;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  // 1. 各行の最大値を取得
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){	    
	  double Amax = 1.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    double val = std::abs(A[j]);
	    Amax = Amax < val ? val : Amax;
	  }
	  AmaxInv[i] = 1.0/Amax;
	}
      }
    });
  q_[0].wait();

  return ;
}

#ifndef MIXED_PRECISON_CRSMV_A64FX_HPP_
template<
  class TdataA,class TdataCRS,class TdataX,      
  class TdataB,class TdataY,      
  class Tcalc>
void crsmv(
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
  // return ;
  size_t MAX   = N;
  //      std::cout<<"run crsmv"<<std::endl;
#if 0
  TdataY y_[N];
  q_[0].wait();
  q_[0].memcpy(y_,y,sizeof(TdataY)*N);
  q_[0].wait();

  TdataX x_[N];
  q_[0].wait();
  q_[0].memcpy(x_,x,sizeof(TdataX)*N);
  q_[0].wait();
  for(int i=0;i<N;i++){
    std::cout<<"input  : "<<i<<" : "<<x_[i]<<" , "<<y_[i]<<std::endl;
  }
#endif

#if 0
  // std::cout<<"crsmv : NNZ = "<<NNZ<<", N = "<<N<<std::endl;
  // 単位行列をかけているのに相当
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      y[i] = x[i];
    });
  q_[0].wait();
#else

  // y = α*Ax + β*y;

#if 0
  
    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	  size_t i0 = i+0;
	  Tcalc Ax0 = 0.0;
	  for(PARCEL_INT j = Rowptr[i0]; j < Rowptr[i0+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const Tcalc A_ = A[j];
	    const Tcalc x_ = x[col];
	    Ax0 = Ax0 + A_*x_;
	  }
	  const Tcalc alpha_ = alpha[0];
	  const Tcalc beta_  = beta[0];
	  Tcalc y0_ = y[i0];
	  y0_ = alpha_*Ax0 + beta_*y0_;
	  y[i0] = y0_;

	
    });  
#else
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#if 0
      // kahan sumation
      Tcalc Ax = 0.0;
      Tcalc c  = 0.0;
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	const PARCEL_INT col = crsCol[j];
	const Tcalc A_ = A[j];
	const Tcalc x_ = x[col];
	Tcalc y = A_*x_ - c;
	Tcalc t = Ax + y;
	c = (t - Ax) -y;
	Ax = t;
      }
#else
      Tcalc Ax = 0.0;
      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	const PARCEL_INT col = crsCol[j];
	const Tcalc A_ = A[j];
	const Tcalc x_ = x[col];
	Ax = Ax + A_*x_;
      }
#endif
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
    });

#endif
  
#endif
  
  // q_[0].wait();
  // q_[0].memcpy(y_,y,sizeof(TdataY)*N);
  // q_[0].wait();
  // for(int i=0;i<N;i++){
  // 	std::cout<<"output : "<<i<<" : "<<y_[i]<<std::endl;
  //      }


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
