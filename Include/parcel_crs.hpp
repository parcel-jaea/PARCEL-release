// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_CRS_HPP_
#define PARCEL_CRS_HPP_

#include "parcel_nblock.hpp"

// カーネルのループ構造を選ぶスイッチ。
//
//   _BLOCKING_LOOP_       0: 平坦ループ (1 行 = 1 work item)。GPU 向け。
//                         1: ブロック化ループ (*_a64fx.hpp)。A64FX 向け。
//   _BLOCKING_LOOP_SPMV_  SpMV だけを別に切り替える (既定 0 = 平坦)。
//   _A64FX_HIPSYCL_       ブロック化したときの寸法。
//                         1 なら 12 ブロック x 4096 行、0 なら N ブロック x 1 行。
//
// A64FX で使うときは -D_BLOCKING_LOOP_=1 だけを指定すればよい。
//
// ■ なぜ SpMV だけ分けてあるか
//
//   A64FX ではブロック化の価値は dot にあり (平坦版より 2 桁以上速い)、
//   SpMV はブロック化すると約 2 倍遅くなる。ブロック化版 SpMV は A*x の積を一旦作業配列 dwork_ に書き出してから
//   行ごとに足し込む方式で、その作業配列は 32*maxCol*n 要素
//   (7 点ステンシルで 1 点あたり 1792 B) にもなる。速度でもメモリでも
//   割に合わないため、既定では SpMV だけ平坦版を使う。
//
// ■ なぜ _A64FX_HIPSYCL_ を連動させているか
//
//   「ブロック化するのに 1 ブロック 1 行」という組み合わせは意味がないうえ、
//   n に比例する作業領域を要求してしまう。独立に指定できると壊れた設定を
//   作れてしまうため、既定でブロック化の有無に従わせる。

#ifndef _BLOCKING_LOOP_
#define _BLOCKING_LOOP_ 0
#endif

#ifndef _BLOCKING_LOOP_SPMV_
#define _BLOCKING_LOOP_SPMV_ 0
#endif

#ifndef _A64FX_HIPSYCL_
#if _BLOCKING_LOOP_ || _BLOCKING_LOOP_SPMV_
#define _A64FX_HIPSYCL_ 1
#else
#define _A64FX_HIPSYCL_ 0
#endif
#endif

// SpMV のブロック幅。_BLOCKING_LOOP_SPMV_ の二段構え実装では、
//   1 ブロックにつき work[B*maxCol] と y_[B] をカーネル内に置く。
//   倍精度・7 点なら B*8*8 バイト = B=4096 で 256 KB となり、A64FX の L1D 64 KB を
//   大きく超えるため work の書き出し・読み戻しがそのままメモリ転送になる。
//   work を L1 に収める粒度は PARCEL_SPMV_TILE で別に指定する。
#ifndef PARCEL_SPMV_BLOCKSIZE
#define PARCEL_SPMV_BLOCKSIZE 4096
#endif

// 二段構え SpMV の work[] の粒度 (行数)。スレッド分割の粒度とは独立。
//   work[] = T * maxCol * sizeof(Tcalc) バイトで、これが L1D (A64FX 64 KB) に
//   収まると書き出し・読み戻しがメモリまで行かない。
//   T=1024 で work は 7*1024*8 = 56 KB となり、A64FX で axpy と同水準の帯域に
//   達する。分割をラウンドロビンにすると粒度を細かくしたとき大きく崩れるので、
//   連続分割にしてある。
#ifndef PARCEL_SPMV_TILE
#define PARCEL_SPMV_TILE 1024
#endif

// work[] の確保量は parcel_blas_crs.hpp で
//   dwork_bloncksize = maxCol * CRS_SPMV_BLOCKSIZE * 8 * 2
// と決まる。タイル 1 つが使うのは maxCol * PARCEL_SPMV_TILE なので、
// TILE が BLOCKSIZE*16 を超えると書き潰す。両者は別々に -D できるので番人を置く。
#if PARCEL_SPMV_TILE > (PARCEL_SPMV_BLOCKSIZE * 16)
#error "PARCEL_SPMV_TILE が work[] の確保量を超える (parcel_blas_crs.hpp の dwork_bloncksize を参照)"
#endif

#if _A64FX_HIPSYCL_

#define CRS_SPMV_NBLOCK (parcel::nblock_runtime())
#define CRS_SPMV_BLOCKSIZE PARCEL_SPMV_BLOCKSIZE
#else
#define CRS_SPMV_NBLOCK    N
#define CRS_SPMV_BLOCKSIZE 1
#endif

#endif
