// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_AXPY_A64FX_HPP_
#define MIXED_PRECISON_AXPY_A64FX_HPP_

#include "parcel_nblock.hpp"
#include <cmath>

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
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){					   
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  z[i] = y_;
	}
      }
    });
  
}


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
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  y[i] = y_;
	}
      }
    });
  
}

// axpy_value / axpby_value の A64FX 版の本体。z = beta*y + alpha*x (z == y なら y の上書き)。
//   (1) nblock 個 (= スレッド数) の work item が連続した ceil(N/nblock) 要素を受け持つ (SpMV のブロック版と同じ分割)。
//   (2) ポインタと係数を restrict 付きのローカル変数に写す (y への書き込みで係数を読み直さない。z == y のときは
//       y と z を 1 本のポインタにして別名を作らない)。
//   (3) beta == 1 のときは beta*y を掛けない。FCC (-ffp-contract=fast) は beta*y + alpha*x を
//       fma(beta, y, fl(alpha*x)) にするので、beta == 1 の値は fl(y + fl(alpha*x)) と同じ。これを fp contract(off) の
//       複合文の中で y + alpha*x と書いて再現する (alpha*x を y に融合させない)。式と足す順は元と同じ。
//   (4) beta != 1 の式も、FCC が元の書き方で選ぶ融合 fma(beta, y, fl(alpha*x)) を std::fma で明示する。
//       同じ式を書いても周りのコードによって fma(alpha, x, fl(beta*y)) が選ばれ、丸めが変わることがあるため。
//   A64FX 1 ノード (NX=256) で axpy 類の時間が 0.82 倍になる。結果は書き直す前とビット単位で一致する。
#if defined(__clang__) && !defined(__CUDA_ARCH__)
#define PARCEL_AXPY_NO_CONTRACT_ _Pragma("clang fp contract(off)")
#else
#define PARCEL_AXPY_NO_CONTRACT_
#endif
template<class TdataX,class TdataY,class TdataZ,class Tcalc>
void parcel_axpy_value_rr_(PARCEL_INT N, Tcalc alpha_, Tcalc beta_, TdataX *x, TdataY *y, TdataZ *z){
  const size_t MAX    = (N > 0) ? (size_t)N : 0;
  const size_t nblock = parcel::nblock_runtime();
  const size_t chunk  = (MAX + nblock - 1) / nblock;
  const bool b1 = (beta_ == (Tcalc)1.0);
  const bool inplace = ((const void *)z == (const void *)y);
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const size_t is_ = ii[0]*chunk;
      const size_t ie_ = (is_ + chunk < MAX) ? (is_ + chunk) : MAX;
      const Tcalc a_ = alpha_;
      const Tcalc b_ = beta_;
      const TdataX * __restrict xr = x;
      if(inplace){
        TdataY * __restrict yr = y;
        if(b1){
          for(size_t i = is_; i < ie_; i++){
            PARCEL_AXPY_NO_CONTRACT_
            const Tcalc x_ = xr[i];
            Tcalc y_ = yr[i];
            y_ = y_ + a_*x_;
            yr[i] = y_;
          }
        }else{
          for(size_t i = is_; i < ie_; i++){
            const Tcalc x_ = xr[i];
            Tcalc y_ = yr[i];
            y_ = std::fma(b_, y_, a_*x_);   // FCC が元の式 b*y + a*x に選ぶ形 (a*x を丸めて b*y に融合) を明示する
            yr[i] = y_;
          }
        }
      }else{
        const TdataY * __restrict yr = y;
        TdataZ * __restrict zr = z;
        if(b1){
          for(size_t i = is_; i < ie_; i++){
            PARCEL_AXPY_NO_CONTRACT_
            const Tcalc x_ = xr[i];
            Tcalc y_ = yr[i];
            y_ = y_ + a_*x_;
            zr[i] = y_;
          }
        }else{
          for(size_t i = is_; i < ie_; i++){
            const Tcalc x_ = xr[i];
            Tcalc y_ = yr[i];
            y_ = std::fma(b_, y_, a_*x_);   // FCC が元の式 b*y + a*x に選ぶ形 (a*x を丸めて b*y に融合) を明示する
            zr[i] = y_;
          }
        }
      }
    });
}

// y = beta*y + alpha*x の alpha / beta を値で渡す版。上の axpy と同じ分割・同じ式。
template<class TdataX,class TdataY,class Tcalc>
void axpy_value(
		PARCEL_INT N,
		Tcalc alpha_,
		Tcalc beta_,
		TdataX *x,
		PARCEL_INT ldx,
		TdataY *y,
		PARCEL_INT ldy){
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
#if _A64FX_HIPSYCL_ && PARCEL_A64FX_AXPY_C
  (void)nlocalblock;
  parcel_axpy_value_rr_<TdataX,TdataY,TdataY,Tcalc>(N, alpha_, beta_, x, y, y);
#else
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  y[i] = y_;
	}
      }
    });
#endif
}

// z = beta*y + alpha*x の alpha / beta を値で渡す版。上の 3 本版 axpy と同じ分割・同じ式。
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
#if _A64FX_HIPSYCL_
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
#if _A64FX_HIPSYCL_ && PARCEL_A64FX_AXPY_C
  (void)nlocalblock;
  parcel_axpy_value_rr_<TdataX,TdataY,TdataZ,Tcalc>(N, alpha_, beta_, x, y, z);
#else
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  z[i] = y_;
	}
      }
    });
#endif
}

// CG (pstep = 1) の x と p の更新を 1 本のカーネルで行う (PARCEL_CG_FUSED)。各要素で
//   x = (beta1*x) + alpha1*(p*ah)   (gemv の axpy 分解版 M = 1 と同じ式。beta1・alpha1・ah はデバイスの 1 要素配列から読む)
//   p = bp*p + ap*s                (axpby_value (z = beta*y + alpha*x、z == y) と同じ式。上の parcel_axpy_value_rr_ と同じく
//                                   FCC が選ぶ融合 fma(bp, p, fl(ap*s)) を std::fma で明示する)
//   の順に計算する (x は更新前の p を使う。元の gemv → axpby と同じ順)。分割は連続した範囲。
//   x と p を 1 回ずつ読み書きするだけになり、カーネル 1 本分の起動と待ちの固定費も減る。
template<class TdataA,class TdataB,class TdataH,class TdataP,class TdataXv,class TdataS,class Tcalc>
void cg_xp_fused(PARCEL_INT N, TdataA *g_alpha, TdataB *g_beta, TdataH *ah, TdataP *p, TdataXv *x,
                 Tcalc ap_, Tcalc bp_, TdataS *s){
  const size_t MAX    = (N > 0) ? (size_t)N : 0;
  const size_t nblock = parcel::nblock_runtime();
  const size_t chunk  = (MAX + nblock - 1) / nblock;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const size_t is_ = ii[0]*chunk;
      const size_t ie_ = (is_ + chunk < MAX) ? (is_ + chunk) : MAX;
      const Tcalc alpha_ = g_alpha[0];
      const Tcalc beta_  = g_beta[0];
      const Tcalc x_     = ah[0];
      const Tcalc ap = ap_;
      const Tcalc bp = bp_;
      TdataP  * __restrict pr = p;
      TdataXv * __restrict xr = x;
      const TdataS * __restrict sr = s;
      for(size_t i = is_; i < ie_; i++){
        Tcalc A_ = pr[i];
        Tcalc y_ = xr[i];
        y_ = beta_*y_ + alpha_*(A_*x_);
        xr[i] = y_;
        Tcalc s_ = sr[i];
        Tcalc pv_ = A_;
        pv_ = std::fma(bp, pv_, ap*s_);
        pr[i] = pv_;
      }
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

