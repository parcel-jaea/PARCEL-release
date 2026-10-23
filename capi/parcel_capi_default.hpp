// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// C / Fortran ラッパ (wrapper_parcel_sycl.cpp) のコンパイル時の既定値。
//   ラッパの内部精度は parcel_solver_*_type で決まる (既定はすべて倍精度)。
//   旧 API run_parcel_sycl_c / _f は反復上限と前処理をここのマクロで決める。
//   いずれもラッパのビルド時に -D で上書きする。
#ifndef PARCEL_CAPI_DEFAULT_HPP_
#define PARCEL_CAPI_DEFAULT_HPP_

// 旧 API の反復上限 (ライブラリの set_max_iteration() の既定と同じ)
#ifndef MAX_ITERATION
#define MAX_ITERATION 10000
#endif

// 精度 (データ精度・演算精度・前処理の精度・pcagmres の QR の精度)
#ifndef parcel_solver_data_type
#define parcel_solver_data_type parcel_double
#endif
#ifndef parcel_solver_calc_type
#define parcel_solver_calc_type parcel_double
#endif
#ifndef parcel_solver_calc_precon_type
#define parcel_solver_calc_precon_type parcel_double
#endif
#ifndef parcel_solver_data_precon_vec_type
#define parcel_solver_data_precon_vec_type parcel_double
#endif
#ifndef parcel_solver_data_precon_mat_type
#define parcel_solver_data_precon_mat_type parcel_double
#endif
#ifndef parcel_solver_calc_qr_type
#define parcel_solver_calc_qr_type parcel_double
#endif

// 旧 API の前処理 (0 = なし, 1 = 点ヤコビ, 2 = ブロックヤコビ, 4 = ヤコビ) とその設定
#ifndef parcel_solver_precon_algo
#define parcel_solver_precon_algo 0
#endif
#ifndef JACOBI_ITER
#define JACOBI_ITER 1
#endif
#ifndef BLOCKJACOBI_BLOCK
#define BLOCKJACOBI_BLOCK 1
#endif

#endif
