// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_GEMM_HPP_
#define MIXED_PRECISON_GEMM_HPP_

#include "parcel_nblock.hpp"

// 列の先頭間隔 ldA/ldB/ldC は size_t。縦長行列 (局所行数 n x 列数 m) の
// i+j*ld を PARCEL_INT (int32) で計算すると n*m が 2^31 を超えたところで桁あふれする。
// dd 版 (mixed_precision_ddgemm*.hpp) も同じ。

#if _A64FX_HIPSYCL_
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void tgemm(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){

  // 私的配列 AB_ は AB_CAP 要素の固定長。m_blockLen * n_blk <= AB_CAP を満たすように
  //   行ブロック幅を縮め、行幅 1 でも足りないときは列も分割して、配列の範囲内に収める。
  constexpr size_t AB_CAP = 512 * 20; // 10240 要素

  if( (m < 1) || (n < 1) || (k < 1) ){ return ; }

  size_t m_blockLen = 512;
  size_t n_blk = (size_t)n;
  if( m_blockLen * n_blk > AB_CAP ){
    m_blockLen = AB_CAP / n_blk;
    if( m_blockLen < 1 ){ m_blockLen = 1; n_blk = AB_CAP; } // 行幅 1 でも足りない場合
  }
  // スレッド数
  const size_t nblock = parcel::nblock_runtime();

  // スレッド並列 : 列方向に配置
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      // C[i_c,j_c]
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_[AB_CAP];
      for(PARCEL_INT jb_ = 0; jb_ < n; jb_ = jb_ + (PARCEL_INT)n_blk){
	const PARCEL_INT j_max = (jb_+(PARCEL_INT)n_blk) > n ? n : (jb_+(PARCEL_INT)n_blk);

	for(PARCEL_INT mm_= ii*m_blockLen; mm_ < m; mm_ = mm_ + m_blockLen*nblock){
	  PARCEL_INT m_max = (mm_+m_blockLen) > m ? m : (mm_+m_blockLen);

	  for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	    for(PARCEL_INT m_= 0; m_ < m_blockLen; m_ = m_ + 1){
	      AB_[m_ + (j_-jb_)*m_blockLen] = 0.0;
	    }
	  }

	  for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	    for(PARCEL_INT l_=0;l_<k;l_++){
	      for(PARCEL_INT m_= mm_; m_ < m_max; m_ = m_ + 1){
		// A * B
		const Tcalc A_ = A[m_ + l_*ldA];
		const Tcalc B_ = B[l_ + j_*ldB];
		AB_[(m_-mm_) + (j_-jb_)*m_blockLen] = AB_[(m_-mm_) + (j_-jb_)*m_blockLen] + A_ * B_;
	      }
	    }
	  }
	  for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	    for(PARCEL_INT m_= mm_; m_ < m_max; m_ = m_ + 1){
	      C[m_ + j_*ldC] = alpha_*AB_[(m_-mm_) + (j_-jb_)*m_blockLen];
	    }
	  }
	}
      }
    });

}
#else
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void tgemm(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){

  // 縦長 (m >> k, n) 専用の経路。
  //   下の経路は出力要素ごとに k ループを回すので、出力の列ごとに A を読み直す (A を n 回読む)。
  //   ここでは 1 行 = 1 ワークアイテムにして、A を 1 回読むごとに n 個の出力へ同時に積む。
  //   転送量は A 読み 1 回 + C 書き 1 回だけになる。B は k*n 個なのでキャッシュに載る。
  //   レジスタに置く累算器を固定長にするため n <= 32 のときだけ使い、外れたら下の経路へ落ちる。
#if PARCEL_CPU_ROWBLOCK
  // CPU 向けの行ブロック版 (PARCEL_CPU_ROWBLOCK)。
  //   1 ワークアイテムが行のブロック (RB 行) を受け持ち、A の列 k_ を 1 本ずつ行方向に読んで
  //   ブロックの出力 n 列ぶんの局所配列 AB_ に積む。「1 行で A の k 本の列を同時に読む」形と違い、
  //   読む列が 1 本で済み、行方向にベクトル化される。
  //   出力 1 要素の計算は下の行ブロックでない版と同じ (k_ = 0, 1, ... の順に AB = AB + A_ * B を積み、C = alpha*AB)。
  if(n <= 32){
    if(m < 1){ return ; }
    constexpr size_t AB_CAP = 4096;             // 局所配列 (double で 32 KiB)
    const size_t RB = (AB_CAP / (size_t)n) > 0 ? (AB_CAP / (size_t)n) : 1;
    const size_t nb = ((size_t)m + RB - 1) / RB;
    q_[0].parallel_for(sycl::range<1>{nb},[=](sycl::id<1> b){
        const size_t r0 = (size_t)b[0] * RB;
        const size_t r1 = (r0 + RB) > (size_t)m ? (size_t)m : (r0 + RB);
        const size_t len = r1 - r0;
        const Tcalc alpha_ = alpha[0];
        Tcalc AB_[AB_CAP];
        for(PARCEL_INT j_=0;j_<n;j_++) for(size_t r=0;r<len;r++) AB_[r + (size_t)j_*RB] = 0.0;
        for(PARCEL_INT k_=0;k_<k;k_++){
          for(PARCEL_INT j_=0;j_<n;j_++){
            const TdataMatB B_ = B[k_ + j_*ldB];
            for(size_t r=0;r<len;r++){
              const Tcalc A_ = A[(r0+r) + k_*ldA];
              AB_[r + (size_t)j_*RB] = AB_[r + (size_t)j_*RB] + A_ * B_;
            }
          }
        }
        for(PARCEL_INT j_=0;j_<n;j_++){
          for(size_t r=0;r<len;r++) C[(r0+r) + j_*ldC] = alpha_*AB_[r + (size_t)j_*RB];
        }
      });
    return ;
  }
#endif
  if(n <= 32){
    const size_t nrow = (size_t)m;
    q_[0].parallel_for(sycl::range<1>{nrow},[=](sycl::id<1> ii){
        const PARCEL_INT i_c = (PARCEL_INT)ii;
        const Tcalc alpha_ = alpha[0];
        Tcalc acc[32];
        for(PARCEL_INT j_=0;j_<n;j_++) acc[j_] = 0.0;
        for(PARCEL_INT k_=0;k_<k;k_++){
          const Tcalc A_ = A[i_c + k_*ldA];      // A はここで 1 回だけ読む
          for(PARCEL_INT j_=0;j_<n;j_++) acc[j_] = acc[j_] + A_ * B[k_ + j_*ldB];
        }
        // 下の経路と同じく beta は使わない (C = alpha*A*B)
        for(PARCEL_INT j_=0;j_<n;j_++) C[i_c + j_*ldC] = alpha_*acc[j_];
      });
    return ;
  }

  // SIMD用最内ループ長 : m /m_block
  size_t m_block = 32;
  size_t n_block = n;  
  size_t MAX = m_block * n_block;

  // スレッド数
  const size_t nblock = n;
  
  // スレッド並列 : 列方向に配置
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      // C[i_c,j_c]
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];      
      for(PARCEL_INT m_= 0; m_ < m; m_ = m_ + 1){
      const PARCEL_INT i_c = m_;
      const PARCEL_INT j_c = ii;
      
      Tcalc AB_ = 0.0;
      for(PARCEL_INT k_=0;k_<k;k_++){
	// A * B
	const PARCEL_INT i_a = i_c;
	const PARCEL_INT j_a = k_;
	const PARCEL_INT i_b = k_;
	const PARCEL_INT j_b = j_c;
	const Tcalc A_ = A[i_a + j_a*ldA];
	const Tcalc B_ = B[i_b + j_b*ldB];
	AB_ = AB_ + A_ * B_;
      }      
      C[i_c + j_c*ldC] = alpha_*AB_;
      }
    });
      
}
#endif
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemm(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){      

  // 並列軸は出力先のC行列要素数
  size_t MAX   = m*n;
#if _BLOCKING_LOOP_

  const size_t nblock      = m*n;
  const size_t nlocalblock = 1;

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){

#endif
      // C[i_c,j_c]
      const PARCEL_INT i_c = i%m;
      const PARCEL_INT j_c = i/m;

      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_ = 0.0;

      for(PARCEL_INT k_=0;k_<k;k_++){
	const PARCEL_INT i_a = i_c;
	const PARCEL_INT j_a = k_;
	const PARCEL_INT i_b = k_;
	const PARCEL_INT j_b = j_c;
	const Tcalc A_ = A[i_a + j_a*ldA];
	const Tcalc B_ = B[i_b + j_b*ldB];
	AB_ = AB_ + A_ * B_;
      }
      Tcalc C_ = C[i_c + j_c*ldC];
      C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

// 縦長 (m >> k, n) で beta を使う版: C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
//   pcagmres の reOrth の更新 V -= Q R (m = 局所行数, n = s, k = s*k+1) 用。
//   既定の gemm は出力 1 要素に 1 ワークアイテムを割り当てるので A の各行を n 回読み直す。
//   ここでは 1 行 = 1 ワークアイテムにして、A の行を 1 回読むごとに n 個の出力へ同時に積む。
//   足し算の順序は既定の gemm と同じ (出力 1 要素ごとに k_ = 0, 1, ... の順に積み、最後に
//   alpha*AB + beta*C)。累算器を固定長にするため n <= 32 のときだけ使い、外れたら既定の gemm に落ちる。
//   A64FX (_A64FX_HIPSYCL_) では下の #if の行ブロック版を使う。
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void tgemm_beta(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){
  if( (m < 1) || (n < 1) ){ return ; }
#if _A64FX_HIPSYCL_
  // A64FX 版: A64FX の tgemm (このファイル先頭) と同じ行ブロック形。1 スレッドが行ブロック
  //   (既定 512 行) を受け持ち、A の列 l_ を行方向に連続して読んで AB_ に積む (行方向に SIMD 化される)。
  //   1 要素 = 1 ワークアイテムの既定の gemm は A64FX (omp.library-only) ではベクトル化されない。
  //   出力 1 要素の足し算の順序は既定の gemm と同じ (l_ = 0, 1, ... の順に積み、最後に alpha*AB + beta*C)。
  constexpr size_t AB_CAP = 512 * 20;
  size_t m_blockLen = 512;
  size_t n_blk = (size_t)n;
  if( m_blockLen * n_blk > AB_CAP ){
    m_blockLen = AB_CAP / n_blk;
    if( m_blockLen < 1 ){ m_blockLen = 1; n_blk = AB_CAP; }
  }
  const size_t nblock = parcel::nblock_runtime();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_[AB_CAP];
      for(PARCEL_INT jb_ = 0; jb_ < n; jb_ = jb_ + (PARCEL_INT)n_blk){
	const PARCEL_INT j_max = (jb_+(PARCEL_INT)n_blk) > n ? n : (jb_+(PARCEL_INT)n_blk);
	for(size_t mm_ = ii*m_blockLen; mm_ < (size_t)m; mm_ = mm_ + m_blockLen*nblock){
	  const size_t m_max = (mm_+m_blockLen) > (size_t)m ? (size_t)m : (mm_+m_blockLen);
	  for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	    for(size_t m_= mm_; m_ < m_max; m_++){ AB_[(m_-mm_) + (j_-jb_)*m_blockLen] = 0.0; }
	  }
	  // l_ を外に置き、A の列の断片 (m_blockLen 要素) を n 列ぶん続けて使う
	  for(PARCEL_INT l_=0;l_<k;l_++){
	    for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	      const Tcalc B_ = B[l_ + (size_t)j_*ldB];
	      for(size_t m_= mm_; m_ < m_max; m_++){
		const Tcalc A_ = A[m_ + (size_t)l_*ldA];
		AB_[(m_-mm_) + (j_-jb_)*m_blockLen] = AB_[(m_-mm_) + (j_-jb_)*m_blockLen] + A_ * B_;
	      }
	    }
	  }
	  for(PARCEL_INT j_=jb_;j_<j_max;j_++){
	    for(size_t m_= mm_; m_ < m_max; m_++){
	      const Tcalc C_ = C[m_ + (size_t)j_*ldC];
	      C[m_ + (size_t)j_*ldC] = alpha_*AB_[(m_-mm_) + (j_-jb_)*m_blockLen] + beta_*C_;
	    }
	  }
	}
      }
    });
  return ;
#endif
  if( n > 32 ){
    gemm<TdataA,TdataB,TdataMatA,TdataMatB,TdataMatC,Tcalc>(m,n,k,alpha,beta,A,ldA,B,ldB,C,ldC);
    return ;
  }
#if PARCEL_CPU_ROWBLOCK
  // CPU 向けの行ブロック版 (PARCEL_CPU_ROWBLOCK)。tgemm の行ブロック版と同じ形。
  //   出力 1 要素の計算は下の行ブロックでない版と同じ (k_ の昇順に AB = AB + A_ * B_ を積み、C = alpha*AB + beta*C)。
  {
    constexpr size_t AB_CAP = 4096;
    const size_t RB = (AB_CAP / (size_t)n) > 0 ? (AB_CAP / (size_t)n) : 1;
    const size_t nb = ((size_t)m + RB - 1) / RB;
    q_[0].parallel_for(sycl::range<1>{nb},[=](sycl::id<1> b){
        const size_t r0 = (size_t)b[0] * RB;
        const size_t r1 = (r0 + RB) > (size_t)m ? (size_t)m : (r0 + RB);
        const size_t len = r1 - r0;
        const Tcalc alpha_ = alpha[0];
        const Tcalc beta_  = beta[0];
        Tcalc AB_[AB_CAP];
        for(PARCEL_INT j_=0;j_<n;j_++) for(size_t r=0;r<len;r++) AB_[r + (size_t)j_*RB] = 0.0;
        for(PARCEL_INT k_=0;k_<k;k_++){
          for(PARCEL_INT j_=0;j_<n;j_++){
            const Tcalc B_ = B[k_ + (size_t)j_*ldB];   // 下の版と同じく Tcalc に直してから積む
            for(size_t r=0;r<len;r++){
              const Tcalc A_ = A[(r0+r) + (size_t)k_*ldA];
              AB_[r + (size_t)j_*RB] = AB_[r + (size_t)j_*RB] + A_ * B_;
            }
          }
        }
        for(PARCEL_INT j_=0;j_<n;j_++){
          for(size_t r=0;r<len;r++){
            const Tcalc C_ = C[(r0+r) + (size_t)j_*ldC];
            C[(r0+r) + (size_t)j_*ldC] = alpha_*AB_[r + (size_t)j_*RB] + beta_*C_;
          }
        }
      });
    return ;
  }
#endif
  const size_t nrow = (size_t)m;
  q_[0].parallel_for(sycl::range<1>{nrow},[=](sycl::id<1> ii){
      const size_t i_c = ii;
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc acc[32];
      for(PARCEL_INT j_=0;j_<n;j_++) acc[j_] = 0.0;
      for(PARCEL_INT k_=0;k_<k;k_++){
	const Tcalc A_ = A[i_c + (size_t)k_*ldA];      // A はここで 1 回だけ読む
	for(PARCEL_INT j_=0;j_<n;j_++){
	  const Tcalc B_ = B[k_ + (size_t)j_*ldB];   // 既定の gemm と同じく Tcalc に直してから積む
	  acc[j_] = acc[j_] + A_ * B_;
	}
      }
      for(PARCEL_INT j_=0;j_<n;j_++){
	const Tcalc C_ = C[i_c + (size_t)j_*ldC];
	C[i_c + (size_t)j_*ldC] = alpha_*acc[j_] + beta_*C_;
      }
    });
}

// C[m,n] = alpha * (A[m,k])^T * B[k,n] + beta * C[m,n]
// B基準ループ : AB[m,n] = A[k,m] * B[k,n]
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTN(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){      

      if( (m == 4) && (n%2==1) ){
	// 並列軸は出力先のC行列要素数
	size_t MAX   = m*n;
	const size_t kloopblock = 480; // 配列サイズnのブロックサイズ
	const size_t work_group_size = 1;
	const size_t num_work_groups = 24;
	const size_t num_work_items = num_work_groups*work_group_size;
	Tcalc outbuf_[ m*n * num_work_items ];
	sycl::buffer<Tcalc> outbuf(outbuf_, m*n * num_work_items );

	q_[0].submit([&] (sycl::handler &h){
		       auto scratch = outbuf.get_access(h);
		       h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
				      [=](sycl::nd_item<1> item) {
					const size_t global_id = item.get_global_id(0);
					const size_t group_id  = item.get_group(0);	     
					const size_t local_id  = item.get_local_id(0);
					
					for(PARCEL_INT i=0;i<n;i++){
					  const Tcalc beta_ = beta[0];
					  Tcalc C1_ = C[0 + i*ldC];
					  Tcalc C2_ = C[1 + i*ldC];
					  Tcalc C3_ = C[2 + i*ldC];
					  Tcalc C4_ = C[3 + i*ldC];
					  scratch[0+i*m +global_id*m*n] = global_id==0 ? beta_*C1_ : (Tcalc) 0.0;
					  scratch[1+i*m +global_id*m*n] = global_id==0 ? beta_*C2_ : (Tcalc) 0.0;
					  scratch[2+i*m +global_id*m*n] = global_id==0 ? beta_*C3_ : (Tcalc) 0.0;
					  scratch[3+i*m +global_id*m*n] = global_id==0 ? beta_*C4_ : (Tcalc) 0.0;
					}

					for(PARCEL_INT kk_=group_id*kloopblock;kk_<k;kk_=kk_+kloopblock*num_work_groups){
					  PARCEL_INT kmax_ = (kk_ + kloopblock) > k ? k : (kk_ + kloopblock);
					  const Tcalc alpha_ = alpha[0];
					  {
					    PARCEL_INT i=0;
					    Tcalc AB11_ = 0.0;
					    Tcalc AB21_ = 0.0;
					    Tcalc AB31_ = 0.0;
					    Tcalc AB41_ = 0.0;
#pragma omp simd
					    for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      const Tcalc A1_ = A[k_ + 0*ldA];
					      const Tcalc A2_ = A[k_ + 1*ldA];
					      const Tcalc A3_ = A[k_ + 2*ldA];
					      const Tcalc A4_ = A[k_ + 3*ldA];
					      const Tcalc B1_ = B[k_ + i*ldB];					      
					      AB11_ = AB11_ + A1_ * B1_;
					      AB21_ = AB21_ + A2_ * B1_;
					      AB31_ = AB31_ + A3_ * B1_;
					      AB41_ = AB41_ + A4_ * B1_;
					    }
					    scratch[0+i*m +global_id*m*n] = scratch[0+i*m +global_id*m*n] + alpha_*AB11_;
					    scratch[1+i*m +global_id*m*n] = scratch[1+i*m +global_id*m*n] + alpha_*AB21_;
					    scratch[2+i*m +global_id*m*n] = scratch[2+i*m +global_id*m*n] + alpha_*AB31_;
					    scratch[3+i*m +global_id*m*n] = scratch[3+i*m +global_id*m*n] + alpha_*AB41_;

					  }
					  for(PARCEL_INT i=1;i<n;i=i+2){
					    // C[i_c,j_c]
					    Tcalc AB11_ = 0.0;
					    Tcalc AB21_ = 0.0;
					    Tcalc AB31_ = 0.0;
					    Tcalc AB41_ = 0.0;
					    Tcalc AB12_ = 0.0;
					    Tcalc AB22_ = 0.0;
					    Tcalc AB32_ = 0.0;
					    Tcalc AB42_ = 0.0;
#pragma omp simd
					    for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					      const Tcalc A1_ = A[k_ + 0*ldA];
					      const Tcalc A2_ = A[k_ + 1*ldA];
					      const Tcalc A3_ = A[k_ + 2*ldA];
					      const Tcalc A4_ = A[k_ + 3*ldA];
					      const Tcalc B1_ = B[k_ + i*ldB];					      
					      const Tcalc B2_ = B[k_ + (i+1)*ldB];					      
					      AB11_ = AB11_ + A1_ * B1_;
					      AB21_ = AB21_ + A2_ * B1_;
					      AB31_ = AB31_ + A3_ * B1_;
					      AB41_ = AB41_ + A4_ * B1_;

					      AB12_ = AB12_ + A1_ * B2_;
					      AB22_ = AB22_ + A2_ * B2_;
					      AB32_ = AB32_ + A3_ * B2_;
					      AB42_ = AB42_ + A4_ * B2_;
					    }
					    scratch[0+i*m +global_id*m*n] = scratch[0+i*m +global_id*m*n] + alpha_*AB11_;
					    scratch[1+i*m +global_id*m*n] = scratch[1+i*m +global_id*m*n] + alpha_*AB21_;
					    scratch[2+i*m +global_id*m*n] = scratch[2+i*m +global_id*m*n] + alpha_*AB31_;
					    scratch[3+i*m +global_id*m*n] = scratch[3+i*m +global_id*m*n] + alpha_*AB41_;

					    scratch[0+(i+1)*m +global_id*m*n] = scratch[0+(i+1)*m +global_id*m*n] + alpha_*AB12_;
					    scratch[1+(i+1)*m +global_id*m*n] = scratch[1+(i+1)*m +global_id*m*n] + alpha_*AB22_;
					    scratch[2+(i+1)*m +global_id*m*n] = scratch[2+(i+1)*m +global_id*m*n] + alpha_*AB32_;
					    scratch[3+(i+1)*m +global_id*m*n] = scratch[3+(i+1)*m +global_id*m*n] + alpha_*AB42_;
					    
					  }
					}
					
				      });
		     });
	q_[0].wait();
	// ワークアイテムごとの部分和をデバイス上で合計して C に書く
	//   (buffer の中身は q.wait() の後もホスト配列 outbuf_ へ書き戻されないことがある (GPU) ので、ホスト側では合計しない)
	q_[0].submit([&] (sycl::handler &h){
		       auto scratch = outbuf.get_access(h);
		       h.parallel_for(sycl::nd_range<1>(1, 1),
				      [=](sycl::nd_item<1> item) {
					for(PARCEL_INT i=0;i<m*n;i++){
					  const PARCEL_INT i_c = i%m;
					  const PARCEL_INT j_c = i/m;
					  Tcalc sum_ = scratch[i+0*m*n];
					  for(size_t j=1;j<num_work_items;j++){ sum_ = sum_ + scratch[i+j*m*n]; }
					  C[i_c + j_c*ldC] = sum_;
					}
				      });	  
		     });	  
	q_[0].wait();
	return ;
      }
   // 並列軸は出力先のC行列要素数
  size_t MAX   = m*n;
#if _BLOCKING_LOOP_

  const size_t nblock      = m*n;
  const size_t nlocalblock = 1;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){

#endif
      // C[i_c,j_c]
      const PARCEL_INT i_c = i%m;
      const PARCEL_INT j_c = i/m;

      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc AB_ = 0.0;

      for(PARCEL_INT k_=0;k_<k;k_++){
	const PARCEL_INT i_a = k_;
	const PARCEL_INT j_a = i_c;
	const PARCEL_INT i_b = k_;
	const PARCEL_INT j_b = j_c;
	const Tcalc A_ = A[i_a + j_a*ldA];
	const Tcalc B_ = B[i_b + j_b*ldB];
	AB_ = AB_ + A_ * B_;
      }
      Tcalc C_ = C[i_c + j_c*ldC];
      C[i_c + j_c*ldC] = alpha_*AB_ + beta_*C_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

// Cは上三角部分のみ計算して下三角部分を複製 (m==n)
// C[m,n] = alpha * (A[m,k])^T * B[k,n] + beta * C[m,n]
// B基準ループ : AB[m,n] = A[k,m] * B[k,n]

#if _A64FX_HIPSYCL_    
    // A64FX 向け実装
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTNsym(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){      

  const size_t kloopblock = 480; // 配列サイズnのブロックサイズ
  const size_t work_group_size = 1;
  const size_t num_work_groups = 32;
  const size_t num_work_items = num_work_groups*work_group_size;
  const size_t mm = m*m; // Cは正方行列(m==n)

  // 正方行列サイズ * 並列数
  Tcalc outbuf_[ mm * num_work_items ];
  sycl::buffer<Tcalc> outbuf(outbuf_, mm * num_work_items );
  q_[0].submit([&] (sycl::handler &h){
		 auto scratch = outbuf.get_access(h);
		 h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
				[=](sycl::nd_item<1> item) {
				  const size_t global_id = item.get_global_id(0);
				  const size_t group_id  = item.get_group(0);	     
				  const size_t local_id  = item.get_local_id(0);

				  for(int irow=0;irow<m;irow++){	     
				    for(int icol=irow;icol<m;icol++){
				      const Tcalc beta_  = beta[0];
				      const PARCEL_INT i_c = irow;
				      const PARCEL_INT j_c = icol;
				      Tcalc C_ = C[i_c + j_c*ldC];					
				      scratch[irow+icol*m+global_id*mm] = global_id==0 ? beta_*C_ : (Tcalc)0.0;
				    }
				  }
				  
				  for(PARCEL_INT kk_=group_id*kloopblock;kk_<k;kk_=kk_+kloopblock*num_work_groups){
				    PARCEL_INT kmax_ = (kk_ + kloopblock) > k ? k : (kk_ + kloopblock);

				    if( m%2 == 0 ){
				      // block loop				      
				      const PARCEL_INT rowblock = 2;
				      const PARCEL_INT colblock = 2;
				      for(PARCEL_INT irow_=0;irow_<m;irow_=irow_+rowblock){
					PARCEL_INT irow_max = (irow_+rowblock) > m ? m : (irow_+rowblock);
					for(PARCEL_INT icol_=irow_;icol_<m;icol_=icol_+colblock){
					  PARCEL_INT icol_max = (icol_+colblock) > m ? m : (icol_+colblock);
				      
					  for(PARCEL_INT irow=irow_;irow<irow_max;irow=irow+2){
					    const PARCEL_INT i1 = irow;
					    const PARCEL_INT i2 = irow+1;				      
					    for(PARCEL_INT icol=icol_;icol<icol_max;icol=icol+2){
					      const PARCEL_INT j1 = icol;
					      const PARCEL_INT j2 = icol+1;					
					      Tcalc AB11_ = 0.0;	
					      Tcalc AB12_ = 0.0;	
					      Tcalc AB21_ = 0.0;	
					      Tcalc AB22_ = 0.0;
#pragma omp simd
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
						const Tcalc A1_ = A[k_ + i1*ldA];
						const Tcalc A2_ = A[k_ + i2*ldA];
						const Tcalc B1_ = B[k_ + j1*ldB];
						const Tcalc B2_ = B[k_ + j2*ldB];

						
						AB11_ = AB11_ + A1_ * B1_;
						AB12_ = AB12_ + A1_ * B2_;
						AB21_ = AB21_ + A2_ * B1_;
						AB22_ = AB22_ + A2_ * B2_;
					      }
					      const Tcalc alpha_ = alpha[0];					
					      scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					      scratch[i1+j2*m+global_id*mm] += alpha_*AB12_;
					      scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					      scratch[i2+j2*m+global_id*mm] += alpha_*AB22_;
					    }
					  }
					}
				      }
				    }else{
				      for(PARCEL_INT irow=0;irow<m-1;irow=irow+2){					
					const PARCEL_INT i1 = irow;
					const PARCEL_INT i2 = irow+1;				      
					for(int icol=irow;icol<m-1;icol=icol+2){	     
					  const PARCEL_INT j1 = icol;
					  const PARCEL_INT j2 = icol+1;					
					  Tcalc AB11_ = 0.0;	
					  Tcalc AB12_ = 0.0;	
					  Tcalc AB21_ = 0.0;	
					  Tcalc AB22_ = 0.0;	
					   #pragma omp simd
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					    const Tcalc A1_ = A[k_ + i1*ldA];
					    const Tcalc A2_ = A[k_ + i2*ldA];
					    const Tcalc B1_ = B[k_ + j1*ldB];
					    const Tcalc B2_ = B[k_ + j2*ldB];
					    AB11_ = AB11_ + A1_ * B1_;
					    AB12_ = AB12_ + A1_ * B2_;
					    AB21_ = AB21_ + A2_ * B1_;
					    AB22_ = AB22_ + A2_ * B2_;
					  }
					  const Tcalc alpha_ = alpha[0];					
					  scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					  scratch[i1+j2*m+global_id*mm] += alpha_*AB12_;
					  scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					  scratch[i2+j2*m+global_id*mm] += alpha_*AB22_;
					}
					{
					  const PARCEL_INT icol = m-1;
					  const PARCEL_INT j1 = icol;
					  Tcalc AB11_ = 0.0;	
					  Tcalc AB21_ = 0.0;	
					   #pragma omp simd
					      for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					    const Tcalc A1_ = A[k_ + i1*ldA];
					    const Tcalc A2_ = A[k_ + i2*ldA];
					    const Tcalc B1_ = B[k_ + j1*ldB];
					    AB11_ = AB11_ + A1_ * B1_;
					    AB21_ = AB21_ + A2_ * B1_;
					  }
					  const Tcalc alpha_ = alpha[0];					
					  scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
					  scratch[i2+j1*m+global_id*mm] += alpha_*AB21_;
					}
				      }
				      {
					const PARCEL_INT irow = m-1;
					const PARCEL_INT icol = m-1;
					const PARCEL_INT i1 = irow;
					const PARCEL_INT j1 = icol;
					Tcalc AB11_ = 0.0;	
					 #pragma omp simd
					for(PARCEL_INT k_=kk_+local_id;k_<kmax_;k_=k_+work_group_size){
					  const Tcalc A1_ = A[k_ + i1*ldA];
					  const Tcalc B1_ = B[k_ + j1*ldB];
					  AB11_ = AB11_ + A1_ * B1_;
					}
					const Tcalc alpha_ = alpha[0];					
					scratch[i1+j1*m+global_id*mm] += alpha_*AB11_;
				      }
				    }
				    
				  }
				
				});
	       });
  q_[0].wait();
  for(int j=1;j<num_work_items;j++){	     
    for(int i=0;i<mm;i++){	     
      outbuf_[i+0*mm] =outbuf_[i+0*mm] + outbuf_[i+j*mm] ;
    }
  }
  q_[0].wait();
  q_[0].submit([&] (sycl::handler &h){
		 auto scratch = outbuf.get_access(h);
		 h.parallel_for(sycl::nd_range<1>(1, 1),
				[=](sycl::nd_item<1> item) {
				  for(int irow=0;irow<m;irow++){	     
				    for(int icol=irow;icol<m;icol++){
					const PARCEL_INT i_c = irow;
					const PARCEL_INT j_c = icol;
					C[i_c + j_c*ldC] = scratch[irow+icol*m+0*mm] ;					
					C[j_c + i_c*ldC] = scratch[irow+icol*m+0*mm] ;					
				      }
				    }
				});
	       });
  q_[0].wait();

}
    
#else
    // CUDA用実装
template<class TdataA,class TdataB,class TdataMatA,class TdataMatB,class TdataMatC,class Tcalc>
void gemmTNsym(
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMatA *A, // m行k列
	  size_t ldA,
	  TdataMatB *B, // k行n列
	  size_t ldB,
	  TdataMatC *C, // m行n列
	  size_t ldC){      

  const size_t mm_num = ((m+1)*m)/2; // Cは正方行列(m==n)
  const size_t nblock = 16384 > k ? k : 16384; // 配列サイズnのブロックサイズ
  const size_t mm_group_size = 1; // 下三角領域を計算するグループ多重度
    
  const size_t ngroup_block = nblock * mm_group_size;
  const size_t work_group_size = 1024; // simd lengh
  const size_t num_work_groups = mm_num * mm_group_size; 
  const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
  
  Tcalc outbuf_[m*m*mm_group_size];
  
  // 部分和の作業域を使い回す。カーネルは使う要素をすべて書いてから読むので、前回の値が残っていても結果は変わらない。
  Tcalc * out = dot_work_<Tcalc>(gemm_work_dev_, m*m*mm_group_size, false);

  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	Tcalc,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
             const size_t global_id = item.get_global_id(0);
	     const size_t batch_id  = global_id/work_group_size/mm_num;
	     const size_t local_id  = item.get_local_id(0);
	     const size_t group_id  = item.get_group(0);	     

	     // 列番号
	     int icol = 0;
	     {
	       int mm_ = group_id;
	       for(int i=0;i<mm_num;i++){
		 mm_ = mm_ - i;
		 if(mm_<0){
		   icol = i-1;	
		   break ;
		 }
	       }
	     }
    
	     // 行番号
	     int irow = 0;
	     {
	       int mm_ = group_id;
	       for(int i=0;i<mm_num;i++){
		 mm_ = mm_ - 1;
		 if(mm_<=0){
		   irow = 0;	
		   for(int j=0;j<=icol;j++){
		     irow = irow + j;
		   } 
		   irow = (group_id - irow)%m;
		   break ;
		 }      
	       }
	     }
	     
      // C[i_c,j_c]
      const PARCEL_INT i_c = irow;
      const PARCEL_INT j_c = icol;
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      // beta*C は最初のバッチの 1 ワークアイテムだけが足す (縮約で全ワークアイテムの値を足すため)。beta = 0 なら C を読まない
      scratch[local_id] = (batch_id == 0 && local_id == 0 && beta_ != (Tcalc)0.0) ? (Tcalc)(beta_ * (Tcalc)C[i_c + j_c*ldC]) : (Tcalc)0.0;
      Tcalc AB_ = 0.0;

      // k (ベクトルの局所長) を走る添字は PARCEL_INT。int では局所行数が 2^31 を超えると桁あふれする。
      const PARCEL_INT kk0 = (PARCEL_INT)(ngroup_block*batch_id);
      for(PARCEL_INT kk = kk0; kk<k; kk = kk + (PARCEL_INT)(ngroup_block*mm_group_size)){      
	const PARCEL_INT ks = kk + (PARCEL_INT)local_id;
	PARCEL_INT ke = (kk + (PARCEL_INT)ngroup_block) > k ? k : (kk + (PARCEL_INT)ngroup_block);
	for(PARCEL_INT k_ = ks; k_<ke; k_ = k_ + (PARCEL_INT)work_group_size){
	  const PARCEL_INT i_a = k_;
	  const PARCEL_INT j_a = i_c;
	  const PARCEL_INT i_b = k_;
	  const PARCEL_INT j_b = j_c;
	  const Tcalc A_ = A[i_a + j_a*ldA];
	  const Tcalc B_ = B[i_b + j_b*ldB];
	  AB_ = AB_ + A_ * B_;
	}
      }
      scratch[local_id] += alpha_*AB_;
      // Do a tree reduction on items in work-group
      for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
	::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
	if (local_id < i){
	  scratch[local_id] += scratch[local_id + i];
	}
      }
      if (local_id == 0) {
	out[ i_c + j_c*m + batch_id*m*m] = scratch[0];
      }
     });
    });

  q_[0].wait();

  const size_t bytes = m*m*mm_group_size * sizeof(Tcalc);
  for(size_t i=0;i<m*m*mm_group_size;i++){ outbuf_[i] = i; }
  
  q_[0].memcpy(outbuf_,out,bytes);
  q_[0].wait();
  for(size_t j=1;j<mm_group_size;j++){
    for(size_t i=0;i<m*m;i++){
      outbuf_[i] = outbuf_[i] + outbuf_[i+j*m*m];
    }
  }

  q_[0].memcpy(out,outbuf_,bytes);
  q_[0].wait();
  
  q_[0].submit([&] (sycl::handler &h){
      h.parallel_for(sycl::nd_range<1>(m*m,1),
		     [=](sycl::nd_item<1> item) {
             const size_t global_id = item.get_global_id(0);
	     const PARCEL_INT i_c = global_id%m;
	     const PARCEL_INT j_c = global_id/m;
	     if( i_c <= j_c ){
	       C[i_c + j_c*ldC] = out[ i_c + j_c*m ] ;
	       C[j_c + i_c*ldC] = out[ i_c + j_c*m ] ;
	     }
		     });
	       });
  q_[0].wait();


  return ;
  
}
    
#endif
    
#endif
