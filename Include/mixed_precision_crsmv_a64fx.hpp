// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_CRSMV_A64FX_HPP_
#define MIXED_PRECISON_CRSMV_A64FX_HPP_

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

  // スレッド分割は連続、work[] の粒度はタイル T。両者を分離する
  //   (parcel_crs.hpp の PARCEL_SPMV_TILE のコメント参照)。work[] を L1 に収めようと
  //   スレッド分割まで細かくすると、スレッド間のメモリアクセスが入り組んで遅くなるため。
  const size_t nblock = CRS_SPMV_NBLOCK;
  const size_t per    = (size_t)((N + (PARCEL_INT)nblock - 1)/(PARCEL_INT)nblock);
  const size_t TILE   = PARCEL_SPMV_TILE;

  Tcalc *work_ = (Tcalc *)dwork_;
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc *work = &work_[ii*work_bloncksize];
      const size_t b = ii*per;
      const size_t e = (b+per) > (size_t)N ? (size_t)N : (b+per);
      for(size_t t0=b; t0<e; t0+=TILE){
	const size_t t1 = (t0+TILE) > e ? e : (t0+TILE);
	// 第 1 段: タイル内の全非零について A*x を作る。行をまたぐ依存が無い。
	const PARCEL_LINT jb = Rowptr[t0];
	const PARCEL_LINT je = Rowptr[t1];
	for(PARCEL_LINT j=jb; j<je; j++){
	  work[j-jb] = (Tcalc)A[j] * (Tcalc)PARCEL_SPMV_XAT(j);
	}
	// 第 2 段: 行ごとに work を連続に足す。gather は無い。
	for(size_t i=t0;i<t1;i++){
	  Tcalc Ax = 0.0;
	  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    Ax = Ax + work[j-jb];
	  }
	  Tcalc y_ = y[i];
	  y[i] = alpha_*Ax + beta_*y_;
	}
      }
    });

  return ;

}

#endif

