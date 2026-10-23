// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DOT_A64FX_HPP_
#define MIXED_PRECISON_DOT_A64FX_HPP_

#include <vector>

#include "parcel_nblock.hpp"

template<class TdataX,class TdataY,class Tcalc,class Tret>
Tret dot(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){      
  // デバイス同期なし
  
#if PARCEL_BLOCKED_DOT
  const size_t nblock    = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)) ;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
      
  // ブロックごとの部分和はホストから直接読める領域に置き、使い回す。このブロック版は
  //   CPU (OpenMP バックエンド) でしか使わないので、ホスト領域への書き込みで余分な転送は起きない。
  double *out = dot_work_<double>(dot_work_host_, nblock, true);

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out[ii] = 0.0;
      double sum_local[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  sum_local[i-iii] = sum_local[i-iii] + x[i]*y[i];
	}						    
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] + sum_local[i];
      }
						  
    });
  q_[0].wait();
  // ブロックの番号 i の順に足す
  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret + out[i];
  }
  return ret;

  
}

#if PARCEL_BLOCKED_DOT
// MGS の「前のステップの axpy」と「このステップの内積」を 1 本のカーネルにまとめる。
//   w[k] = beta*w[k] + alpha*xprev[k]   (axpy_value と同じ式・同じ型の流れ)
//   sum += w[k] * vcur[k]               (上の dot と同じブロック分割・同じ sum_local の添字・同じ最後の足し順)
//   各要素の演算と順序は「axpy_value のあと dot」と同じなので、-ffp-contract=off なら結果はビット単位で一致する。
//   CPU (OpenMP バックエンド) ではカーネル 1 本の起動と待ちの固定費が大きいので、MGS の 1 ステップの
//   カーネルを 2 本から 1 本に減らす。戻り値はこのランクの局所値 (MPI 集約は呼び出し側)。
template<class TdataX,class TdataW,class Tcalc,class Tret>
Tret axpy_dot_fused(
		    PARCEL_INT N,
		    Tcalc alpha_,
		    Tcalc beta_,
		    TdataX *xprev,
		    TdataW *w,
		    TdataX *vcur){
  const size_t nblock      = parcel::nblock_runtime();
  // 上の dot は nlocalblock を内積の第 1 引数 (w) の型で決める
  const size_t nlocalblock = 4096 * (8/sizeof(TdataW)) ;
  double *out = dot_work_<double>(dot_work_host_, nblock, true);
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out[ii] = 0.0;
      double sum_local[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tcalc x_ =  xprev[i];
	  Tcalc y_ =  w[i];
	  y_   = beta_*y_ + alpha_*x_;
	  const TdataW wn = y_;
	  w[i] = wn;
	  sum_local[i-iii] = sum_local[i-iii] + wn*vcur[i];
	}
      }
      for(size_t i=0;i<nlocalblock;i++){
	out[ii] = out[ii] + sum_local[i];
      }
    });
  q_[0].wait();
  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret + out[i];
  }
  return ret;
}
#endif

#if PARCEL_BLOCKED_DOT
// 局所内積を count 本まとめて 1 本のカーネルで取る (ks_arnoldi のブロック直交化用)。
//   d 本目は <x0 + d*ldx, y> (長さ N)。各ワークアイテム ii は d = 0..count-1 を順に回し、d ごとに
//   上の dot と同じこと (out を 0 に、sum_local を 0 に、同じブロックを同じ順に積み、sum_local を i の順に足す)
//   を行う。ホスト側も d ごとに out を ii の順に足す。したがって各内積の値は dot を count 回呼んだ場合と
//   ビット単位で一致する。sums[d] に局所値を返す (MPI 集約は呼び出し側)。
template<class TdataX,class TdataY>
void multi_dot(
	       PARCEL_INT N,
	       TdataX *x0,
	       size_t ldx,
	       PARCEL_INT count,
	       TdataY *y,
	       double *sums){
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)) ;   // 上の dot と同じ (内積の第 1 引数の型)
  const size_t nd          = (size_t)count;
  double *out = dot_work_<double>(dot_work_host_, nblock*nd, true);
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      double sum_local[nlocalblock];
      for(size_t d=0; d<nd; d++){
	const TdataX *x = x0 + d*ldx;
	double *o = out + d*nblock;
	o[ii] = 0.0;
	for(size_t i=0;i<nlocalblock;i++){
	  sum_local[i] = 0.0;
	}
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    sum_local[i-iii] = sum_local[i-iii] + x[i]*y[i];
	  }
	}
	for(size_t i=0;i<nlocalblock;i++){
	  o[ii] = o[ii] + sum_local[i];
	}
      }
    });
  q_[0].wait();
  for(size_t d=0; d<nd; d++){
    double ret = 0.0;
    for(size_t i=0;i<nblock;i++){
      ret = ret + out[d*nblock + i];
    }
    sums[d] = ret;
  }
}
#endif


#endif

