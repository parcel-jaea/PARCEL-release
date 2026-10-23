// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_GEMV_HPP_
#define MIXED_PRECISON_GEMV_HPP_

#include "parcel_nblock.hpp"
#ifndef PARCEL_A64FX_AXPY_C
#define PARCEL_A64FX_AXPY_C 1
#endif

// gemv を M 本の axpy に分解するか (PARCEL_GEMV_AS_AXPY。A64FX 向けビルドで既定 1、他は 0)。
//
//   gemv は y への読み書きを 1 回にまとめる代わりに M 本の列を同時に流す。A64FX では
//   これが極端に遅く、転送量の多い axpy x M の方が速い。x86・GPU では融合した gemv の
//   方が速い (転送量が半分のため) ので、A64FX だけ分解する。
//   A64FX で遅いのは、ldA = n (局所ベクトル長) が 2 の冪のとき M 本の列が同じキャッシュ
//   セットに落ち、M が way 数を超えると追い出しが起きるため。
//
//   **丸めの順序が変わる。** gemv は Tcalc で累算してから alpha を 1 回掛けるが、
//   分解版は項ごとに alpha を掛けて y に足す。
//
//   判定に _A64FX_HIPSYCL_ ではなく _BLOCKING_LOOP_ を使う。_A64FX_HIPSYCL_ は
//   parcel_crs.hpp で導出されるが、このファイルはそれより先に展開されうるため、
//   未定義のまま #if に書くと 0 と評価されて黙って誤った既定になる。
//   _BLOCKING_LOOP_ はコマンドラインで定義されるのでインクルード順に依存しない。
#ifndef PARCEL_GEMV_AS_AXPY
#if defined(_BLOCKING_LOOP_) && (_BLOCKING_LOOP_)
#define PARCEL_GEMV_AS_AXPY 1
#else
#define PARCEL_GEMV_AS_AXPY 0
#endif
#endif

// 行列転置未対応
// y = alpha * Ax + beta * y 
//   ldA は size_t。i+j*ldA を PARCEL_INT (int32) で計算すると、局所行数 n と列数 M の
//   積が 2^31 を超えたところで桁あふれして別の番地を読む。
template<class TdataA,class TdataB,class TdataMat,class TdataX,class TdataY,class Tcalc>
void gemv(
	  PARCEL_INT N, // 行数
	  PARCEL_INT M, // 列数
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMat *A, // N行M列
	  size_t ldA,
	  TdataX *x, // M行1列
	  PARCEL_INT ldx,
	  TdataY *y, // N行1列
	  PARCEL_INT ldy){      
  size_t MAX   = N;
#if PARCEL_CPU_ROWBLOCK
  // CPU 向けの行ブロック版 (PARCEL_CPU_ROWBLOCK)。
  //   1 ワークアイテムが行のブロックを受け持ち、ブロックの中を「列が外側・行が内側」で回す。
  //   読む列は常に 1 本で、行方向にベクトル化される (1 行で M 本の列を同時に読む形はキャッシュの効きが悪い)。
  //   各要素の計算は下の行ブロックでない版と同じ式・同じ順序:
  //     axpy 分解版 (PARCEL_GEMV_AS_AXPY、A64FX): 列の昇順に y = (最初の列 ? beta*y : y) + alpha*(A*x) を
  //       毎回 y に書き戻す (列ごとのカーネルを 1 本にまとめたもの)。
  //     融合版 (他の CPU): 列の昇順に Ax = Ax + A*x を積み、最後に y = beta*y; Ax = alpha*Ax; y = y + Ax。
  if(N < 1){ return ; }
#if PARCEL_GEMV_AS_AXPY
  {
    constexpr size_t RB = 4096;                 // A64FX の他のカーネルの塊 (nlocalblock) と同じ幅
    const size_t nb = ((size_t)N + RB - 1) / RB;
    q_[0].parallel_for(sycl::range<1>{nb},[=](sycl::id<1> b){
	const size_t i0 = (size_t)b[0] * RB;
	const size_t i1 = (i0 + RB) > (size_t)N ? (size_t)N : (i0 + RB);
	// 係数と x[j0] をループの外で 1 回だけ読み、A と y を restrict 付きのローカル変数に写す。
	//   各要素の式 (fst ? beta*y : y) + alpha*(A*x) と列の順は元と同じ (結果はビット単位で同じ)。
#if PARCEL_A64FX_AXPY_C
	const Tcalc alpha_ = alpha[0];
	const Tcalc beta_  = beta[0];
	const TdataMat * __restrict Ar = A;
	TdataY * __restrict yr = y;
	for(PARCEL_INT j0=0; j0<M; j0++){
	  const bool fst = (j0 == 0);
	  const Tcalc x_ = x[j0];
	  const TdataMat * __restrict Aj = Ar + (size_t)j0*ldA;
	  for(size_t i=i0;i<i1;i++){
	    Tcalc A_ = Aj[i];
	    Tcalc y_ = yr[i];
	    y_ = (fst ? beta_*y_ : y_) + alpha_*(A_*x_);
	    yr[i] = y_;
	  }
	}
#else
	// 書き直す前の形 (PARCEL_A64FX_AXPY_C=0): 係数と x[j0] を内側のループで読む。式と列の順は同じ。
	for(PARCEL_INT j0=0; j0<M; j0++){
	  const bool fst = (j0 == 0);
	  for(size_t i=i0;i<i1;i++){
	    Tcalc alpha_ = alpha[0];
	    Tcalc beta_  = beta[0];
	    Tcalc x_ = x[j0];
	    Tcalc A_ = A[i+j0*ldA];
	    Tcalc y_ = y[i];
	    y_ = (fst ? beta_*y_ : y_) + alpha_*(A_*x_);
	    y[i] = y_;
	  }
	}
#endif
      });
    return;
  }
#else
  {
    constexpr size_t RB = 2048;                 // 局所配列 acc (double で 16 KiB) が L1 に収まる幅
    const size_t nb = ((size_t)N + RB - 1) / RB;
    q_[0].parallel_for(sycl::range<1>{nb},[=](sycl::id<1> b){
	const size_t i0 = (size_t)b[0] * RB;
	const size_t i1 = (i0 + RB) > (size_t)N ? (size_t)N : (i0 + RB);
	Tcalc alpha_ = alpha[0];
	Tcalc beta_  = beta[0];
	Tcalc acc[RB];
	for(size_t i=i0;i<i1;i++){ acc[i-i0] = 0.0; }
	for(PARCEL_INT j=0;j<M;j++){
	  Tcalc x_ = x[j];
	  for(size_t i=i0;i<i1;i++){
	    Tcalc A_ = A[i+j*ldA];
	    acc[i-i0] = acc[i-i0] + A_*x_;
	  }
	}
	for(size_t i=i0;i<i1;i++){
	  Tcalc Ax_ = acc[i-i0];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_;
	  Ax_  = alpha_*Ax_;
	  y_   = y_ + Ax_;
	  y[i] = y_;
	}
      });
    return;
  }
#endif
#endif
#if _BLOCKING_LOOP_

#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

#if PARCEL_GEMV_AS_AXPY
  // 列ごとに 1 本の axpy として起動する。
  //   y への依存があるので 1 本ごとに待つ。parcel_blas の axpy も
  //   blas_timer_ の中で毎回 queue を待つので、回数あたりの条件は同じになる。
  for(PARCEL_INT jj=0; jj<M; jj++){
    const PARCEL_INT j0   = jj;
    const bool       fst  = (jj == 0);
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    Tcalc alpha_ = alpha[0];
	    Tcalc beta_  = beta[0];
	    Tcalc x_ = x[j0];
	    Tcalc A_ = A[i+j0*ldA];
	    Tcalc y_ = y[i];
	    y_ = (fst ? beta_*y_ : y_) + alpha_*(A_*x_);
	    y[i] = y_;
	  }
	}
      });
    q_[0].wait();
  }
  return;
#endif

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else
#if PARCEL_GEMV_AS_AXPY
  for(PARCEL_INT jj=0; jj<M; jj++){
    const PARCEL_INT j0  = jj;
    const bool       fst = (jj == 0);
    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	Tcalc alpha_ = alpha[0];
	Tcalc beta_  = beta[0];
	Tcalc x_ = x[j0];
	Tcalc A_ = A[i+j0*ldA];
	Tcalc y_ = y[i];
	y_ = (fst ? beta_*y_ : y_) + alpha_*(A_*x_);
	y[i] = y_;
      });
    q_[0].wait();
  }
  return;
#endif
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      Tcalc Ax_ = 0.0;
      for(PARCEL_INT j=0;j<M;j++){
	Tcalc x_ = x[j];
	Tcalc A_ = A[i+j*ldA];
	Ax_ = Ax_ + A_*x_;
      }
      Tcalc y_ =  y[i];
      y_   = beta_*y_;
      Ax_  = alpha_*Ax_;
      y_   = y_ + Ax_;
      y[i] = y_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

#endif
