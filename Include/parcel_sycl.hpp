// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_SYCL_HPP_
#define PARCEL_SYCL_HPP_

// 版のマクロ (PARCEL_VERSION_*) と parcel::version_string() / parcel::get_version()
#include "parcel_version.hpp"

#include <CL/sycl.hpp>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <string>
#include <unistd.h>
#include <vector>

// MPI を使うかどうか。
//   既定                : 1 (MPI あり。mpi.h を読み、parcel::comm は MPI_Comm を包む)
//   -DPARCEL_MPI=0      : MPI なし。mpi.h を読まないので MPI のないコンパイラ (acpp など) で組める。
//                         1 プロセスで動き、parcel::comm は comm(sycl::queue*) で作る (mpi_rank()=0, mpi_size()=1)。
#ifndef PARCEL_MPI
#define PARCEL_MPI 1
#endif

// 積和を FMA 命令にまとめるかどうか。
//   コンパイラは a*b+c を FMA 1 命令にまとめることがあり (AdaptiveCpp は既定で -ffp-contract=fast)、
//   どこをまとめるかは周りのコードに左右される。そのため計算と無関係な変更でビルドし直しただけで
//   丸めが変わり、単精度の構成では反復数がずれることがある (同じ実行ファイルなら毎回ビット単位で同じ)。
//   ビルドをまたいで結果を揃えたいときは、コンパイラに -ffp-contract=off を渡す (推奨)。
//   -DPARCEL_FP_CONTRACT_OFF は以降のコードで #pragma clang fp contract(off) を有効にするが、
//   AdaptiveCpp が既定で渡す -ffp-contract=fast はこの pragma を無視するので、
//   -ffp-contract=fast-honor-pragmas と組み合わせたときだけ効く。
//   double-double の積は明示的に fma を呼ぶので影響しない。
#if defined(PARCEL_FP_CONTRACT_OFF) && (PARCEL_FP_CONTRACT_OFF + 0) != 0
#  pragma clang fp contract(off)
#endif

// -ffast-math (-ffinite-math-only / -fassociative-math を含む) でのビルドを止める。
//   double-double は丸め誤差をそのまま取り出す計算 (Dekker の分割, TwoSum) で成り立っており、
//   結合則による書き換えを許すと誤差項が 0 に簡約され、dd が黙って double に落ちる。
//   また NaN / Inf の検査 (std::isfinite) が消され、収束状態の DIVERGED_NANORINF が働かなくなる。
//   承知のうえで使う場合は -DPARCEL_ALLOW_FAST_MATH を付ける (dd の精度と NaN 検出は保証しない)。
#if (defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)) \
    && !defined(PARCEL_ALLOW_FAST_MATH)
#error "PARCEL: -ffast-math / -ffinite-math-only breaks double-double arithmetic and NaN detection. Remove it, or define PARCEL_ALLOW_FAST_MATH to build anyway."
#endif

// 行番号・列番号・非零数などの大域の整数型 PARCEL_INT。
//   既定                : int64_t (MPI では MPI_INT64_T)
//   -DPARCEL_INT32      : int32_t (MPI_INT32_T)。
//                         -DPARCEL_INT32=0 は 64 ビット (スクリプトで値を切り替えるため)。
//   -DPARCEL_INT        : 64 ビット指定の旧い書き方。互換のため受け付け、64 ビットにする
//                         (値は見ない。-DPARCEL_INT=int32_t でも 64 ビット)。
//                         コマンドラインで定義された PARCEL_INT は "1" などの値を持つだけで型ではないので、
//                         ここで #undef してから型として定義し直す。
//   -DPARCEL_INT と -DPARCEL_INT32 (非 0) の同時指定は矛盾するので #error で止める。
//   選ばれた幅は PARCEL_INT_BITS (64 / 32) で分かる (利用者のコードやラッパの範囲検査用)。
//   MPI_PARCEL_INT も同じくここで定義し直す (事前の定義は無視する)。
//   C / Fortran API (parcel_sycl_c.h の parcel_int) は PARCEL_INT によらず常に 64 ビット。
#include <cstdint>
#if defined(PARCEL_INT32) && (PARCEL_INT32 + 0) != 0
#  if defined(PARCEL_INT)
#    error "PARCEL: -DPARCEL_INT (64-bit, legacy spelling) and -DPARCEL_INT32 contradict each other. Remove one of them (64-bit is now the default)."
#  endif
#  define PARCEL_INT_BITS 32
#else
#  define PARCEL_INT_BITS 64
#endif
#ifdef PARCEL_INT
#  undef PARCEL_INT
#endif
#ifdef MPI_PARCEL_INT
#  undef MPI_PARCEL_INT
#endif
#if PARCEL_INT_BITS == 32
#  define PARCEL_INT int32_t
#  define MPI_PARCEL_INT MPI_INT32_T
#else
#  define PARCEL_INT int64_t
#  define MPI_PARCEL_INT MPI_INT64_T
#endif

// ランク内 (局所) の添字型 PARCEL_LINT。
//   PARCEL_INT は大域の量 (全体の行数、nstart/nend、set_crsMat に渡す大域列番号、袖通信の表を作る
//   大域列番号、C / Fortran API) に使う。デバイス上の CRS (行ポインタと、局所化した後の列番号:
//   自ランクの行 0..n_local-1、袖 n_local..n_local+nadd-1)、袖の詰め込み用の局所番号リスト、
//   SpMV・前処理のカーネルの添字は PARCEL_LINT を使う。
//   理由: 添字を 64 ビットにすると SpMV のメモリ転送量が増え、GPU では 1.2〜2 倍遅くなる。
//   1 ランクの非零数は 2^31 に届かないことがほとんどなので、ランク内の添字は 32 ビットとし、
//   大域の量だけを 64 ビットに保つ。
//   既定                    : int32_t (MPI_INT32_T)
//   -DPARCEL_LOCAL_INT64    : int64_t (MPI_INT64_T)。1 ランクの非零数 (または局所行数 + 袖の数) が
//                             2^31-1 を超えるとき。-DPARCEL_LOCAL_INT64=0 は 32 ビット。
//   -DPARCEL_INT32          : 大域も 32 ビットなので局所も 32 ビット。
//   -DPARCEL_INT32 と -DPARCEL_LOCAL_INT64 (非 0) の同時指定は、局所の型が大域より広くなるので #error。
//   選ばれた幅は PARCEL_LINT_BITS (32 / 64)。set_crsMat は局所化の後、1 ランクの非零数と
//   局所行数 + 袖の数が PARCEL_LINT に収まるかを全ランクで調べ、収まらなければ全ランクが
//   std::overflow_error を投げる (parcel.hpp の check_lint_range)。
//   試験用に -DPARCEL_LINT_TEST_LIMIT=<値> でこの検査の上限だけを下げられる (既定は PARCEL_LINT の最大値)。
#if defined(PARCEL_LOCAL_INT64) && (PARCEL_LOCAL_INT64 + 0) != 0
#  if PARCEL_INT_BITS == 32
#    error "PARCEL: -DPARCEL_LOCAL_INT64 with -DPARCEL_INT32: the rank-local index type (PARCEL_LINT) cannot be wider than the global one (PARCEL_INT). Remove one of them."
#  endif
#  define PARCEL_LINT_BITS 64
#else
#  define PARCEL_LINT_BITS 32
#endif
#ifdef PARCEL_LINT
#  undef PARCEL_LINT
#endif
#ifdef MPI_PARCEL_LINT
#  undef MPI_PARCEL_LINT
#endif
#if PARCEL_LINT_BITS == 32
#  define PARCEL_LINT int32_t
#  define MPI_PARCEL_LINT MPI_INT32_T
#else
#  define PARCEL_LINT int64_t
#  define MPI_PARCEL_LINT MPI_INT64_T
#endif

#include "ddlib.hpp"
#include "util_sycl.hpp"
#include "parcel.hpp"
#include "parcel_blas.hpp"
#include "parcel_krylov.hpp"

#if PARCEL_MPI
#include <mpi.h>
#include "./parcel_mpi/parcel_mpi.hpp"
#endif

#endif
