// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_DEFAULT_HPP_
#define PARCEL_DEFAULT_HPP_

#ifndef MAX_ITERATION
#define MAX_ITERATION 10000   // 最大反復回数の既定 (ライブラリの set_max_iteration() の既定と同じ). 実行時は PARCEL_MAX_ITER
#endif

#ifndef NX
#define NX 240
#endif

#ifndef GMRES_QR_ALGO
#define GMRES_QR_ALGO  parcel::qr::QR_MGS
#endif

#ifndef parcel_solver_calc_qr_type
#define parcel_solver_calc_qr_type parcel_double
#endif

#ifndef CAGMRES_RESTART_S
#define CAGMRES_RESTART_S 8
#endif
#ifndef CAGMRES_RESTART_T
#define CAGMRES_RESTART_T 1
#endif
#ifndef CAGMRES_QR_ALGO
#define CAGMRES_QR_ALGO  parcel::qr::QR_MGS
#endif

// ---- 固有値計算: Arnoldi + Krylov-Schur リスタート (ks_arnoldi) ----
#ifndef KS_M
#define KS_M 30            // Krylov 部分空間の最大次元
#endif
#ifndef KS_K_KEEP
#define KS_K_KEEP 10       // restart 毎に保持する Ritz vector 数
#endif
#ifndef KS_NEV
#define KS_NEV 4           // 求めたい固有値数 (収束判定対象)
#endif
#ifndef KS_MAX_RESTART
#define KS_MAX_RESTART 50  // 最大 restart 回数
#endif

// ---- 入れ子ソルバ (内部反復解法) の収束閾値 ----
//   inverse_power は内部 CG、ks_arnoldi の shift-invert は内部 GMRES を使う。
//   これらの閾値は外側の tolerance より厳しくする必要があるが、低精度では
//   到達できない値を指定すると反復上限まで回り続ける。
//   FP32 で使う場合は 1.0e-5 程度まで緩める。
#ifndef INVPOW_LINSOLVE_TOL
#define INVPOW_LINSOLVE_TOL 1.0e-10
#endif
#ifndef INVPOW_LINSOLVE_MAX
#define INVPOW_LINSOLVE_MAX 5000
#endif
#ifndef KS_SI_LINSOLVE_TOL
#define KS_SI_LINSOLVE_TOL 1.0e-10
#endif
#ifndef KS_SI_LINSOLVE_MAX
#define KS_SI_LINSOLVE_MAX 2000
#endif
#ifndef KS_SI_LINSOLVE_RESTART
#define KS_SI_LINSOLVE_RESTART 30
#endif

#ifndef GMRES_RESTART
#define GMRES_RESTART 8
#endif

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

#ifndef parcel_solver_precon_algo
#define parcel_solver_precon_algo 0
#endif

#ifndef JACOBI_ITER
#define JACOBI_ITER 1
#endif

#ifndef BLOCKJACOBI_BLOCK
#define BLOCKJACOBI_BLOCK 1
#endif

// PARCEL_INT で掛ける (int だと NX >= 1291 で桁あふれする。PARCEL_INT は既定 64 ビット)
#define problem_size ((PARCEL_INT)(NX)*(PARCEL_INT)(NX)*(PARCEL_INT)(NX))
#define problem_type double

// ---- ソルバ選択 ----
// 以下のいずれか 1 つをコンパイル時に -D で指定する (例: -Dpcg_solver).
// 本ファイルのコメントを外して指定してもよい.
//
//   線形方程式:
// #define pcg_solver        1   // 前処理付き共役勾配法
// #define pbicgstab_solver  1   // 前処理付き BiCGStab 法
// #define pgmres_solver     1   // 前処理付き GMRES 法
// #define pcagmres_solver   1   // 前処理付き通信削減型 GMRES 法
//
//   固有値計算:
// #define power_solver         1   // べき乗法          (最大 |λ|)
// #define inverse_power_solver 1   // 逆べき乗法        (最小 |λ|)
// #define ks_arnoldi_solver    1   // Arnoldi + Krylov-Schur (上位 nev 個 / shift-invert)

#endif
