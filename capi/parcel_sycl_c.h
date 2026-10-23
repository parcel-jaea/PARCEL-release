/*
 * parcel_sycl_c.h
 *
 * SPDX-License-Identifier: LGPL-3.0-only
 * Copyright (c) 日本原子力研究開発機構 システム計算科学センター
 *
 * PARCEL_SYCL の C 言語 API (実装: wrapper_parcel_sycl.cpp)。
 * Fortran からは parcel_sycl_mod.f90 の同名関数で呼ぶ。
 *
 * 7 種のソルバすべてを実行時に選べる:
 *   線形方程式 Ax = b : pcg / pbicgstab / pgmres / pcagmres  -> parcel_solve_linear()
 *                       (Q を陽に作らない CA-GMRES は pcagmres の cagmres_q_mode =
 *                        PARCEL_CAGMRES_Q_IMPLICIT)
 *   固有値 Av = λv    : power / inverse_power / ks_arnoldi       -> parcel_solve_eigen()
 *
 * ---- 行列の渡し方 (全ランク共通の約束) ----
 *   各ランクは大域行番号が連続する行ブロック [nstart, nstart + n_local - 1] を CRS で持つ。
 *     n_local   : 担当行数 (>= 1)。ランクごとに違ってよい (不均等分割可)。
 *     nstart    : 担当する最初の行の大域行番号
 *     n_global  : 大域行数。全ランクの n_local の和と一致し、行ブロックは重複・欠落なく
 *                 [0, n_global) を覆うこと (ランク番号の順に並んでいる必要はない)。
 *     nnz_local : 担当行の非零要素数
 *     rowptr    : 長さ n_local + 1。rowptr[0] からの差だけを使う (0/1 始まりどちらでも可)
 *     col       : 長さ nnz_local。**大域**列番号
 *     val       : 長さ nnz_local。行列要素 (double)
 *   nstart と col は options.index_base (既定 0) の番号付け。Fortran の 1 始まりなら 1 にする。
 *   入力配列は書き換えない (内部で複製してから PARCEL に渡す)。
 *
 * ---- 整数の型 ----
 *   n_local / nstart / n_global / nnz_local / rowptr[] / col[] は parcel_int (= int64_t、64 ビット固定)。
 *   ライブラリ内部の PARCEL_INT (既定 64 ビット、-DPARCEL_INT32 で
 *   32 ビット) によらずこの ABI は変わらない。ラッパを -DPARCEL_INT32 でビルドしたときに値が
 *   32 ビットに収まらなければ PARCEL_ERR_ARG を返す (全ランクで一致させる。確保の前に調べる)。
 *   旧 API run_parcel_sycl_c / _f の n / nnz / crsCol / crsRow_ptr も parcel_int。
 *   オプション (parcel_options) と結果 (parcel_result) の項目は反復回数などの小さな数なので int。
 *   固有対の個数 (parcel_eigen_count) も int。
 *   Fortran では integer(c_int64_t) (parcel_sycl_mod.f90 の parcel_int_kind) を使う。
 *
 * ---- ベクトル ----
 *   すべて担当行ぶん (長さ n_local) のローカル部分。データは double で受け渡す。
 *   内部のデータ精度・演算精度は wrapper_parcel_sycl.cpp のコンパイル時マクロで決まる
 *   (parcel_capi_default.hpp の parcel_solver_data_type / parcel_solver_calc_type /
 *    parcel_solver_calc_precon_type / parcel_solver_data_precon_{mat,vec}_type /
 *    parcel_solver_calc_qr_type)。
 *
 * ---- 戻り値 ----
 *   PARCEL_OK (0) または負のエラーコード。引数検査の結果は全ランクで一致させて返す。
 *   反復が収束したかどうかは戻り値には現れない。最後の引数 res (parcel_result、NULL 可) に
 *   収束状態・反復回数・残差が入る (全ランクで同じ値)。ライブラリも rank 0 の標準出力に
 *   " converged_reason = ... , its = ... , residual = ..." の 1 行を書く。
 *
 * 呼び出しごとに SYCL キュー (既定デバイス) の作成と行列の登録を行う。
 * MPI_Init 済みであること。
 *
 * ---- MPI なしのビルド (PARCEL_MPI=0) ----
 *   ラッパ (wrapper_parcel_sycl.cpp) と、このヘッダを include するプログラムの両方を
 *   -DPARCEL_MPI=0 でコンパイルする (食い違うとコミュニケータの引数の型が合わない)。
 *   このとき mpi.h を読まず、MPI_Init も要らない。関数の名前と引数の並びは MPI ありと同じで、
 *   コミュニケータの引数 (下の PARCEL_C_COMM / PARCEL_C_FCOMM) は int になり、値は使わない (0 を渡す)。
 *   1 プロセスが全体を持つので nstart = index_base、n_local = n_global とする (違えば PARCEL_ERR_DISTRIBUTION)。
 *   ライブラリ内の例外は MPI_Abort ではなく PARCEL_ERR_RUNTIME として返る。
 *   Fortran (parcel_sycl_mod.f90) はそのまま使える (コミュニケータは integer で、値は使わない)。
 */
#ifndef PARCEL_SYCL_C_H_
#define PARCEL_SYCL_C_H_

#ifndef PARCEL_MPI
#define PARCEL_MPI 1
#endif

#if PARCEL_MPI
#include <mpi.h>
#endif
#include <stdint.h>

/* コミュニケータの引数の型。
 *   MPI あり : PARCEL_C_COMM = MPI_Comm、PARCEL_C_FCOMM = MPI_Fint (Fortran の整数ハンドル)
 *   MPI なし : どちらも int (値は使わない) */
#if PARCEL_MPI
#define PARCEL_C_COMM  MPI_Comm
#define PARCEL_C_FCOMM MPI_Fint
#else
#define PARCEL_C_COMM  int
#define PARCEL_C_FCOMM int
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 行番号・列番号・行数・非零数の型 (常に 64 ビット) ---- */
typedef int64_t parcel_int;

/* ---- ソルバ ---- */
#define PARCEL_SOLVER_PCG            0
#define PARCEL_SOLVER_PBICGSTAB      1
#define PARCEL_SOLVER_PGMRES         2
#define PARCEL_SOLVER_PCAGMRES       3
#define PARCEL_SOLVER_POWER          4   /* 絶対値最大の固有値 1 個            */
#define PARCEL_SOLVER_INVERSE_POWER  5   /* 絶対値最小の固有値 1 個            */
#define PARCEL_SOLVER_KS_ARNOLDI     6   /* 絶対値最大側 (shift 時は σ 近傍) nev 個 */
/* Q を陽に作らない CA-GMRES は PARCEL_SOLVER_PCAGMRES に cagmres_q_mode = PARCEL_CAGMRES_Q_IMPLICIT を
   指定する (cagmres_qr_algo が既定の MGS のままなら CholeskyQR、モーメント法は
   PARCEL_QR_CAGMRES_CHOLESKYQR)。7 を渡すと PARCEL_ERR_ARG */

/* ---- 前処理 (precon_algo / linsolve_precon_algo) ---- */
#define PARCEL_PRECON_NONE           0
#define PARCEL_PRECON_POINTJACOBI    1
#define PARCEL_PRECON_BLOCKJACOBI    2
#define PARCEL_PRECON_JACOBI         4

/* ---- GMRES / CA-GMRES / ks_arnoldi の直交化 (gmres_qr_algo / cagmres_qr_algo / ks_qr_algo) ---- */
#define PARCEL_QR_CGS                0
#define PARCEL_QR_MGS                1
#define PARCEL_QR_CHOLESKYQR         2
#define PARCEL_QR_CAGMRES_CHOLESKYQR 202 /* pcagmres の cagmres_q_mode = PARCEL_CAGMRES_Q_IMPLICIT 専用:
                                            最初のブロックのグラム行列をモーメント (ハンケル行列) から作る。
                                            A が対称・前処理なし・s が偶数のときだけ使える
                                            (下の cagmres_qr_algo / cagmres_q_mode を参照) */

/* ---- CA-GMRES の直交基底 Q の作り方 (cagmres_q_mode。C++ の parcel::krylov::CAGMRES_Q_* と同じ値) ---- */
#define PARCEL_CAGMRES_Q_AUTO        0   /* CHOLESKYQR かつ t = 1 のときだけ Q を作らない           */
#define PARCEL_CAGMRES_Q_EXPLICIT    1   /* 常に Q を作る                                           */
#define PARCEL_CAGMRES_Q_IMPLICIT    2   /* どの t でも Q を作らない。
                                            cagmres_qr_algo は既定 (MGS) のままなら CHOLESKYQR を使う。
                                            CGS は不可                                              */

/* ---- 固有値ソルバの内部線形ソルバ (linsolve_algo, inverse_power が使う) ---- */
#define PARCEL_LINSOLVE_PGMRES       0
#define PARCEL_LINSOLVE_PCG          1

/* ---- 戻り値 ---- */
#define PARCEL_OK                    0
#define PARCEL_ERR_ARG              -1   /* 引数・オプションが不正            */
#define PARCEL_ERR_DISTRIBUTION     -2   /* 行ブロックが [0,n_global) を覆わない */
#define PARCEL_ERR_NOT_BUILT        -3   /* そのソルバがビルドから外されている */
#define PARCEL_ERR_RUNTIME          -4   /* ライブラリ内で例外 (下の注意)     */
/*
 * ライブラリ内で例外が起きた場合 (デバイスのメモリ不足、行列サイズが PARCEL_INT を
 * 超えるなど)、ラッパはメッセージを標準エラーに出してから MPI_Abort(comm, 4) を呼び、
 * 全ランクを終了させる。例外は 1 ランクだけで起きうるので、PARCEL_ERR_RUNTIME を返して
 * 呼び出し側に任せると、他のランクがソルバ内の集団通信で待ち続けてジョブが止まるため。
 * したがって実際には PARCEL_ERR_RUNTIME は呼び出し側に返らない。
 * MPI なしのビルド (PARCEL_MPI=0) では待つ相手がいないので、メッセージを出して PARCEL_ERR_RUNTIME を返す。
 * 引数の検査 (PARCEL_ERR_ARG / _DISTRIBUTION / _NOT_BUILT) は全ランクで結果を
 * そろえてから返すので、中断しない。
 */

/* ---- 収束状態 (parcel_result.converged_reason。C++ の parcel::krylov::CONVERGED_RTOL 等と同じ値) ---- */
#define PARCEL_CONVERGED_RTOL        1   /* 収束判定 (tolerance) を満たした              */
#define PARCEL_ITERATING             0   /* 未実行 (引数エラーで解かなかったときなど)     */
#define PARCEL_DIVERGED_ITS         -1   /* 最大反復 / 最大リスタートに達した            */
#define PARCEL_DIVERGED_NANORINF    -2   /* 判定値 (残差・固有値の推定) に NaN / Inf      */
#define PARCEL_DIVERGED_BREAKDOWN   -3   /* アルゴリズムの破綻 (ks_arnoldi の h_{j+1,j}=0 など) */

/*
 * 解いた結果の収束状態。parcel_solve_linear / parcel_solve_eigen の最後の引数 (NULL 可)。
 * 並びは Fortran の bind(C) 派生型 parcel_result (parcel_sycl_mod.f90) と一致させてある。
 * 各項目の定義はソルバごとに違う (Include/parcel_krylov.hpp の収束状態の説明を参照):
 *   iterations : 線形ソルバは反復回数 (pgmres は内側反復の総数、pcagmres は
 *                基底ベクトルの総数)、power / inverse_power は外側反復の回数、
 *                ks_arnoldi はリスタートの回数
 *   residual   : tolerance と比べた相対値。線形ソルバは ||r||/||b|| (pcg / pbicgstab は漸化式、
 *                pgmres / pcagmres は真の残差)、power / inverse_power は固有値の相対変化、
 *                ks_arnoldi は判定対象 ks_nev 個の Ritz 残差推定の最大値。未計算なら NaN
 *   true_residual : ||b - Ax|| / ||b|| (線形ソルバの終了時の値)。固有値ソルバは NaN
 * 引数エラーなどで解かなかったときは converged_reason = PARCEL_ITERATING (0)、残差は NaN。
 */
typedef struct parcel_result {
  int    converged_reason;       /* PARCEL_CONVERGED_RTOL / PARCEL_DIVERGED_* (正なら収束)   */
  int    iterations;             /* 反復回数 (上の説明)                                      */
  double residual;               /* 判定に使った相対残差 (上の説明)                          */
  double true_residual;          /* ||b - Ax|| / ||b|| (線形ソルバのみ。それ以外は NaN)       */
  int    ks_nconv;               /* ks_arnoldi: tolerance を満たした Ritz 対の数 (ks_nev 個中) */
  int    linsolve_failures;      /* inverse_power / ks_arnoldi の shift-invert:
                                    内部線形ソルバが収束しなかった回数 (外側の判定には影響しない) */
} parcel_result;

/*
 * オプション。必ず parcel_options_init() で既定値を入れてから必要な項目だけ変える。
 * 並びは Fortran の bind(C) 派生型 (parcel_sycl_mod.f90) と一致させてある。
 * 変更するときは両方を直すこと。
 * 既定値は C++ ライブラリのクラスの初期値から取る (parcel_options_init がオブジェクトを作って
 * 読む。値を二重に持たない)。下の「既定」はその値。
 */
typedef struct parcel_options {
  /* ---- double ---- */
  double tolerance;              /* 収束判定値                               既定 1.0e-8   */
  double linsolve_tolerance;     /* 内部線形ソルバの収束判定値               既定 0 (= ソルバの既定:
                                    inverse_power / ks_arnoldi とも 1.0e-10)。0 以下はソルバの既定 */
  double shift_sigma;            /* shift-invert のシフト σ                  既定 0.0      */

  /* ---- 共通 ---- */
  int max_iteration;             /* 最大反復回数 (線形, power, inverse_power の外側) 既定 10000 */
  int precon_algo;               /* 前処理 PARCEL_PRECON_*                   既定 NONE     */
  int jacobi_iter;               /* ヤコビ前処理の反復回数                   既定 1        */
  int blockjacobi;               /* ブロックヤコビのブロック数 (0 = 自動)    既定 0        */
  int scaling_matrix;            /* 行列スケーリング (-1 自動, 0 無効, 1 有効) 既定 -1     */
  int scaling_vector;            /* ベクトルスケーリング (同上)              既定 -1       */
  int index_base;                /* nstart / col の番号付け (0 または 1)     既定 0        */
  int verbose;                   /* 1 で rank 0 がデバイス名と設定を表示     既定 0        */

  /* ---- pgmres ---- */
  int gmres_restart;             /* リスタート周期 m                         既定 30       */
  int gmres_qr_algo;             /* PARCEL_QR_CGS / PARCEL_QR_MGS            既定 MGS
                                    (CHOLESKYQR は pgmres では使えず PARCEL_ERR_ARG) */
  /* ---- pcagmres ---- */
  int cagmres_s;                 /* s ステップ                               既定 8        */
  int cagmres_t;                 /* t (リスタート周期 = s*t)                 既定 4        */
  int cagmres_qr_algo;           /* PARCEL_QR_*                              既定 MGS
                                    CAGMRES_CHOLESKYQR は cagmres_q_mode = IMPLICIT のときだけ
                                    (それ以外は PARCEL_ERR_ARG)                                */

  /* ---- 固有値ソルバ共通 ---- */
  int precon_spectrum;           /* 1: A K^-1 の固有値 (K は precon_algo の前処理)
                                    0: A の固有値                            既定 0        */
  int linsolve_algo;             /* inverse_power の内部ソルバ PARCEL_LINSOLVE_* 既定 PGMRES
                                    (PCG は precon_spectrum=0 のときだけ。1 と併用すると PARCEL_ERR_ARG) */
  int linsolve_max_iteration;    /* 内部線形ソルバの最大反復                 既定 0 (= ソルバの既定:
                                    inverse_power 5000、ks_arnoldi の shift-invert 2000)。0 以下はソルバの既定 */
  int linsolve_restart;          /* 内部 pgmres のリスタート周期             既定 0 (= ソルバの既定:
                                    どちらも 30)。0 以下はソルバの既定                         */
  int linsolve_precon_algo;      /* 内部線形ソルバの前処理 PARCEL_PRECON_*   既定 JACOBI
                                    (precon_spectrum=1 では使われない: 内部も K を使う) */
  int linsolve_jacobi_iter;      /* 内部ソルバのヤコビ反復回数               既定 1        */
  int linsolve_blockjacobi;      /* 内部ソルバのブロック数 (0 = 自動)        既定 0        */

  /* ---- ks_arnoldi ---- */
  int ks_m;                      /* Krylov 部分空間の最大次元                既定 30       */
  int ks_k_keep;                 /* リスタートで保持する Ritz 対の数         既定 10       */
  int ks_nev;                    /* 求める固有値の数                         既定 4
                                    (制約: ks_nev <= ks_k_keep < ks_m <= n_global。外れると PARCEL_ERR_ARG) */
  int ks_max_restart;            /* 最大リスタート回数                       既定 200      */
  int shift_enable;              /* 1: shift-invert (σ 近傍の固有値)         既定 0
                                    ks_arnoldi のみ。precon_spectrum=0 ではラッパが
                                    (A - σI) を組んで渡し、A の固有値を返す。
                                    precon_spectrum=1 では (A K^-1 - σI) を内部 pgmres の
                                    シフトで表す (シフト行列は作らない)。                 */
  int eigvec_seed;               /* 1: ks_arnoldi の初期ベクトルに eigvec の先頭 n_local
                                    要素を使う。0: ラッパが大域行番号から作る   既定 0     */

  /* ---- pcagmres (Q の作り方) ---- */
  int cagmres_q_mode;            /* 直交基底 Q の作り方 PARCEL_CAGMRES_Q_*    既定 AUTO
                                    AUTO     : CHOLESKYQR かつ t = 1 のときだけ Q を作らない
                                    EXPLICIT : 常に Q を作る
                                    IMPLICIT : どの t でも Q を作らない。
                                      cagmres_qr_algo は次のどちらか (既定の MGS のままなら CHOLESKYQR を使う。
                                      C では MGS の明示と未指定を区別できないため。CGS は PARCEL_ERR_ARG):
                                      PARCEL_QR_CHOLESKYQR:          グラム行列を内積で作る
                                                                     (一般の行列・前処理あり可)
                                      PARCEL_QR_CAGMRES_CHOLESKYQR:  最初のブロックをモーメントで作る。
                                        A が対称であること (検出しない。非対称だと誤った解になる)、
                                        precon_algo = PARCEL_PRECON_NONE、cagmres_s が偶数 (>= 2)
                                        が必要。後の 2 つを満たさないと PARCEL_ERR_ARG
                                    0 / 1 / 2 以外は PARCEL_ERR_ARG */

  /* ---- ks_arnoldi (直交化) ---- */
  int ks_qr_algo;                /* PARCEL_QR_CGS / PARCEL_QR_MGS (pgmres と同じ定数)。ks_arnoldi では
                                    どちらも 2 回適用する: CGS = ブロック版の古典グラムシュミット 2 回 (CGS2)、
                                    MGS = 修正グラムシュミット 2 回 (MGS2)。既定はライブラリの既定
                                    (ビルド時の PARCEL_KS_ORTHO_DEFAULT。未定義なら CGS)。
                                    それ以外は PARCEL_ERR_ARG                                    */
} parcel_options;

/* 既定値を入れる */
void parcel_options_init(parcel_options *opt);

/* ソルバ名 ("pcg" など) -> PARCEL_SOLVER_*。不明なら -1 */
int parcel_solver_from_name(const char *name);

/*
 * 固有値ソルバが返す固有対の個数 k。
 *   power / inverse_power : 1
 *   ks_arnoldi            : opt->ks_k_keep (うち収束保証があるのは先頭 ks_nev 個)
 *   線形ソルバ・不正値    : 0
 * eigval は 2*k 要素 (実部・虚部が交互)、eigvec は n_local*k 要素 (列 j が j 番目の
 * 固有ベクトルの担当行ぶん、列優先) を用意する。
 */
int parcel_eigen_count(int solver, const parcel_options *opt);

/*
 * 線形方程式 Ax = b。
 *   solver : PARCEL_SOLVER_PCG / PBICGSTAB / PGMRES / PCAGMRES
 *   b      : 右辺 (n_local)
 *   x      : 入力 初期値、出力 解 (n_local)
 */
int parcel_solve_linear(int solver, const parcel_options *opt, PARCEL_C_COMM comm,
                        parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                        const parcel_int *rowptr, const parcel_int *col, const double *val,
                        const double *b, double *x, parcel_result *res);

/*
 * 固有値 Av = λv (precon_spectrum=1 なら A K^-1)。
 *   solver : PARCEL_SOLVER_POWER / INVERSE_POWER / KS_ARNOLDI
 *   eigval : 出力 2*k 要素 (Re, Im 交互)。k は parcel_eigen_count()
 *   eigvec : 出力 n_local*k 要素。eigvec_seed=1 の ks_arnoldi では先頭 n_local が入力
 *   res    : 出力 収束状態 (NULL 可)。parcel_solve_linear も同じ
 */
int parcel_solve_eigen(int solver, const parcel_options *opt, PARCEL_C_COMM comm,
                       parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                       const parcel_int *rowptr, const parcel_int *col, const double *val,
                       double *eigval, double *eigvec, parcel_result *res);

/* Fortran 用 (コミュニケータを整数ハンドルで受け、MPI_Comm_f2c で変換する。MPI なしでは使わない)。
 * Fortran からは parcel_sycl_mod の parcel_solve_linear / parcel_solve_eigen で呼ぶ。 */
int parcel_solve_linear_f(int solver, const parcel_options *opt, PARCEL_C_FCOMM comm,
                          parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                          const parcel_int *rowptr, const parcel_int *col, const double *val,
                          const double *b, double *x, parcel_result *res);
int parcel_solve_eigen_f(int solver, const parcel_options *opt, PARCEL_C_FCOMM comm,
                         parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                         const parcel_int *rowptr, const parcel_int *col, const double *val,
                         double *eigval, double *eigvec, parcel_result *res);
/* Fortran から sizeof(parcel_options) を照合するため。自動では比べないので、
   Fortran 側で parcel_options_check() を呼んで確かめる (sample_f.f90 を参照) */
int parcel_options_size(void);
/* 同じく sizeof(parcel_result) (Fortran 側は parcel_result_check()) */
int parcel_result_size(void);

/* ---- 版 ----
 * ライブラリの版 (Include/parcel_version.hpp の PARCEL_VERSION_MAJOR / MINOR / PATCH。ラッパをビルドしたときの値)。
 *   parcel_get_version     : 番号を返す。NULL を渡した引数は書かない。戻り値は PARCEL_OK
 *   parcel_version_string  : "1.2.0" の形の文字列 (静的な領域。解放しない)
 * MPI を初期化する前・MPI なしのビルドでも呼べる。 */
int parcel_get_version(int *major, int *minor, int *patch);
const char *parcel_version_string(void);

/*
 * 旧 API (v1.1 以前との互換)。pcg で Ax=b を解く。
 *   tolerance 1e-9、反復上限 MAX_ITERATION、前処理 parcel_solver_precon_algo
 *   (いずれもラッパのコンパイル時マクロ)。
 *   nstart は各ランクの n の累積和 (MPI_Exscan) で求めるので、全ランクで n が
 *   等しければ nstart = n*myrank になり、不均等も扱える。
 *   crsCol / crsRow_ptr は 0 始まり。入力配列は書き換えない。
 *   n / nnz / crsCol / crsRow_ptr は parcel_int (64 ビット。新 API と同じ型。
 *   呼び出し側の配列も 64 ビットにすること)。
 *   run_parcel_sycl_f の Fortran 側の宣言は integer(c_int64_t), value :: n, nnz と
 *   integer(c_int64_t) :: crsCol(*), crsRow_ptr(*) (sample_f.f90 を参照)。
 */
void run_parcel_sycl_c(PARCEL_C_COMM mpi_comm_c, parcel_int n, parcel_int nnz,
                       double *crsA, parcel_int *crsCol, parcel_int *crsRow_ptr,
                       double *vecx, double *vecb);
void run_parcel_sycl_f(int *mpi_comm_f, parcel_int n, parcel_int nnz,
                       double *crsA, parcel_int *crsCol, parcel_int *crsRow_ptr,
                       double *vecx, double *vecb);

#ifdef __cplusplus
}
#endif

#endif /* PARCEL_SYCL_C_H_ */
