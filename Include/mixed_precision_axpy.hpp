// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
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

// y = beta*y + alpha*x の alpha / beta を値で渡す版。
//   係数を 1 要素のデバイス配列へ書き戻す (setValue と待ち) 手間を省くために使う。
//   式の形は上の axpy と同じ (beta も実行時の値で渡す) なので、結果も同じになる。
template<class TdataX,class TdataY,class Tcalc>
void axpy_value(
		PARCEL_INT N,
		Tcalc alpha_,
		Tcalc beta_,
		TdataX *x,
		PARCEL_INT ldx,
		TdataY *y,
		PARCEL_INT ldy){
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc x_ =  x[i];
      Tcalc y_ =  y[i];
      y_   = beta_*y_ + alpha_*x_;
      y[i] = y_;
    });
}

// z = beta*y + alpha*x の alpha / beta を値で渡す版。上の 3 本版 axpy と同じ式の形。
template<class TdataX,class TdataY,class TdataZ,class Tcalc>
void axpy_value(
		PARCEL_INT N,
		Tcalc alpha_,
		Tcalc beta_,
		TdataX *x,
		PARCEL_INT ldx,
		TdataY *y,
		PARCEL_INT ldy,
		TdataZ *z,
		PARCEL_INT ldz){
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc x_ =  x[i];
      Tcalc y_ =  y[i];
      y_   = beta_*y_ + alpha_*x_;
      z[i] = y_;
    });
}

// copy / zero の長さは size_t。基底全体 (局所行数 x 列数) を
// 一括で扱う呼び出しがあり、PARCEL_INT (int32) では 2^31 を超えると桁あふれする。
template<class Tdata>
void copy(
	  size_t N,
	  Tdata *x,
	  size_t ldx,
	  Tdata *y,
	  size_t ldy){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      y[i] = x[i];
    });
}

template<class Tdata>
void zero(
	  size_t N,
	  Tdata *x,
	  size_t ldx
	  ){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      x[i] = 0.0;
    });
}

#endif

#endif
