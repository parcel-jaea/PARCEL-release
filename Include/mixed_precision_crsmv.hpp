// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_CRSMV_HPP_
#define MIXED_PRECISON_CRSMV_HPP_

#include "parcel_nblock.hpp"

// 列 JJ の x の参照 x[crsCol[JJ]]。SpMV の本体はこのマクロで x を読む。
#define PARCEL_SPMV_XAT(JJ)  x[crsCol[JJ]]

#if _BLOCKING_LOOP_SPMV_
#include "mixed_precision_crsmv_a64fx.hpp"
#endif

void crs_maxInv(
		PARCEL_INT N,
		double *A_hi,double *A_lo,
		double *AmaxInv_hi,double *AmaxInv_lo,
		PARCEL_LINT *Rowptr,
		PARCEL_LINT *crsCol
		){
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
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
	  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_LINT col = crsCol[j];
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
	     PARCEL_LINT *Rowptr,
	     PARCEL_LINT *crsCol
	     ){
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
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
	  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_LINT col = crsCol[j];
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

#if PARCEL_SPMV_A64FX_BLOCK_ACTIVE
  {
    // A64FX のブロック化版 (parcel_nblock.hpp の PARCEL_SPMV_A64FX_BLOCK_ACTIVE)。nblock 個の work item が
    //   連続した行の範囲を受け持ち、配列のポインタを restrict 付きのローカル変数に写す。行ごとの式と足す順序は
    //   下の平坦版と同じ。beta が 0 のときは y を読まない (y = alpha*Ax)。有限の y では結果は同じで、
    //   y が NaN / Inf のときと alpha*Ax が -0 のときのゼロの符号だけが違いうる。
    const size_t nblock_ = parcel::nblock_runtime();
    const size_t chunk_  = (MAX + nblock_ - 1) / nblock_;
    q_[0].parallel_for(sycl::range<1>{nblock_},[=](sycl::id<1> ii){
        const TdataCRS    * __restrict A_  = A;
        const PARCEL_LINT * __restrict rp_ = Rowptr;
        const PARCEL_LINT * __restrict ci_ = crsCol;
        const TdataX      * __restrict x_  = x;
        TdataY            * __restrict y_  = y;
        const Tcalc alpha_ = alpha[0];
        const Tcalc beta_  = beta[0];
        const bool b0_ = (beta_ == (Tcalc)0.0);
        const size_t is_ = ii[0]*chunk_;
        const size_t ie_ = (is_ + chunk_ < MAX) ? (is_ + chunk_) : MAX;
        for(size_t i = is_; i < ie_; i++){
          Tcalc Ax = 0.0;
#if PARCEL_SPMV_LADDER
          PARCEL_LINT j        = rp_[i];
          const PARCEL_LINT je = rp_[i+1];
          for(; j + 4 <= je; j += 4){
            Ax = Ax + (Tcalc)A_[j  ] * (Tcalc)x_[ci_[j  ]]
                    + (Tcalc)A_[j+1] * (Tcalc)x_[ci_[j+1]]
                    + (Tcalc)A_[j+2] * (Tcalc)x_[ci_[j+2]]
                    + (Tcalc)A_[j+3] * (Tcalc)x_[ci_[j+3]];
          }
          if(j + 2 <= je){
            Ax = Ax + (Tcalc)A_[j  ] * (Tcalc)x_[ci_[j  ]]
                    + (Tcalc)A_[j+1] * (Tcalc)x_[ci_[j+1]];
            j += 2;
          }
          if(j < je){
            Ax = Ax + (Tcalc)A_[j] * (Tcalc)x_[ci_[j]];
          }
#else
          for(PARCEL_LINT j = rp_[i]; j < rp_[i+1]; j++){
            const Tcalc Aj = A_[j];
            const Tcalc xj = x_[ci_[j]];
            Ax = Ax + Aj*xj;
          }
#endif
          if(b0_){
            y_[i] = alpha_*Ax;
          }else{
            Tcalc yv = y_[i];
            yv = alpha_*Ax + beta_*yv;
            y_[i] = yv;
          }
        }
      });
  }
#else
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
        {
      Tcalc Ax = 0.0;
#if PARCEL_SPMV_LADDER
      // 4 -> 2 -> 1 と段階的に処理する。本体はすべて固定長なので
      // コンパイラが展開・先読みできる。行あたり非零数が一定である必要はない。
      {
	PARCEL_LINT j        = Rowptr[i];
	const PARCEL_LINT je = Rowptr[i+1];
	for(; j + 4 <= je; j += 4){
	  Ax = Ax + (Tcalc)A[j  ] * (Tcalc)PARCEL_SPMV_XAT(j  )
	          + (Tcalc)A[j+1] * (Tcalc)PARCEL_SPMV_XAT(j+1)
	          + (Tcalc)A[j+2] * (Tcalc)PARCEL_SPMV_XAT(j+2)
	          + (Tcalc)A[j+3] * (Tcalc)PARCEL_SPMV_XAT(j+3);
	}
	if(j + 2 <= je){
	  Ax = Ax + (Tcalc)A[j  ] * (Tcalc)PARCEL_SPMV_XAT(j  )
	          + (Tcalc)A[j+1] * (Tcalc)PARCEL_SPMV_XAT(j+1);
	  j += 2;
	}
	if(j < je){
	  Ax = Ax + (Tcalc)A[j] * (Tcalc)PARCEL_SPMV_XAT(j);
	}
      }
#else
      for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	const Tcalc A_ = A[j];
	const Tcalc x_ = PARCEL_SPMV_XAT(j);
	Ax = Ax + A_*x_;
      }
#endif
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
        }
      });

#endif

  return ;
}

#endif

#endif
