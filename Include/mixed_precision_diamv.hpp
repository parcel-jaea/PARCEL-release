#ifndef MIXED_PRECISON_DIAMV_HPP_
#define MIXED_PRECISON_DIAMV_HPP_

#if _BLOCKING_LOOP_
#include "mixed_precision_diamv_a64fx.hpp"
#endif

PARCEL_INT get_diag_id(
		       PARCEL_INT NND,
		       PARCEL_INT *offset
		       ){

  util_sycl::vector<PARCEL_INT> diag_id_;
  diag_id_.resize(1,q_);
  PARCEL_INT *diag_id = diag_id_.data();
 
  q_[0].wait();      
  size_t MAX = NND;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      PARCEL_INT offset_ = offset[i];
      if(offset_==0){
	diag_id[0] = i;
      }
    });
  q_[0].wait();      
  PARCEL_INT id;
  diag_id_.getValue(&id);

  return id;

}

void dia_maxInv(
		PARCEL_INT N,
		PARCEL_INT NND,
		double *A_hi,double *A_lo,
		double *AmaxInv_hi,double *AmaxInv_lo
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
	  for(PARCEL_INT j=0;j<NND;j++){
	    auto val_hi = std::abs(A_hi[i+j*N]);
	    auto val_lo = std::abs(A_lo[i+j*N]);
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
void dia_maxInv(
	     PARCEL_INT N,
	     PARCEL_INT NND,
	     T_A *A,
	     T_Ainv *AmaxInv
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
	  for(PARCEL_INT j=0;j<NND;j++){
	    double val = std::abs(A[i+j*N]);
	    Amax = Amax < val ? val : Amax;
	  }
	  AmaxInv[i] = 1.0/Amax;
	}
      }
    });
  q_[0].wait();

  return ;
}


#ifndef MIXED_PRECISON_DIAMV_A64FX_HPP_
template<
  class TdataA,class TdataDIA,class TdataX,      
  class TdataB,class TdataY,      
  class Tcalc>
void diamv(
	   PARCEL_INT N,
	   PARCEL_INT NND,
	   TdataA *alpha,
	   TdataDIA *A,
	   PARCEL_INT *offset,
	   PARCEL_INT max_up_offset,
	   PARCEL_INT max_down_offset,
	   TdataX *x,
	   PARCEL_INT ldx,
	   TdataB *beta,
	   TdataY *y,
	   PARCEL_INT ldy){      
  size_t MAX   = N;
  // y = α*Ax + β*y;

#if 1

#if 1
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
      const PARCEL_INT i = i_;
      Tcalc Ax = 0.0;
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	for(PARCEL_INT j = 0;j<NND;j++){
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
	    const Tcalc A_ = A[i+j*N];
	    const Tcalc x_ = x[i+offset[j]];
	    Ax = Ax + A_*x_;
	  }
	}
      }else{
	for(PARCEL_INT j = 0;j<NND;j++){
	  const Tcalc A_ = A[i+j*N];
	  const Tcalc x_ = x[i+offset[j]];
	  Ax = Ax + A_*x_;
	}
      }
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
    });
#endif

#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc Ax = 0.0;
      for(PARCEL_INT j = 0;j<NND;j++){
	if( (0<=((PARCEL_INT)i+offset[j]) ) && (((PARCEL_INT)i+offset[j])<N) ){
	  const Tcalc A_ = A[i+j*N];
	  const Tcalc x_ = x[i+offset[j]];
	  Ax = Ax + A_*x_;
	}
      }
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
    });

#endif

  return ;

}
#endif

#endif
