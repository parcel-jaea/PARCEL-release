/* =========================================================================
 * parcel_py.h
 *
 * SPDX-License-Identifier: LGPL-3.0-only
 * Copyright (c) 日本原子力研究開発機構 システム計算科学センター
 *
 * Python パッケージ parcel の内部向けネイティブ層 (libparcel_py.so) の宣言。
 * Python からは ctypes で読み込む (python/parcel/_native.py)。
 *
 * C / Fortran 向けの API (parcel_sycl_c.h, wrapper_parcel_sycl.cpp) とは独立している。
 * 構造体・定数・関数は Python 層のためだけのもので, C API の版と合わせる必要はない。
 *
 *   - 値が「未指定」の項目は, 整数は負, 実数は NaN で表す。未指定ならライブラリの既定値を使う
 *     (ネイティブ層は設定関数を呼ばない)。既定値を Python 側に二重に持たないため。
 *   - 行列は各ランクの担当行の CRS (0 始まり, 列番号は大域番号, 整数は int64)。
 *   - 精度は倍精度だけ (データ・演算とも)。
 *   - -DPARCEL_MPI=0 で MPI なしの 1 プロセス版にできる (関数と引数は同じ。コミュニケータは使わない)。
 * ========================================================================= */
#ifndef PARCEL_PY_H_
#define PARCEL_PY_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 構造体の版。Python 側で読み込み時に照合する。項目を変えたら上げる。 */
#define PARCEL_PY_ABI_VERSION 3

/* ソルバ */
enum {
  PARCEL_PY_PCG = 0, PARCEL_PY_PBICGSTAB = 1, PARCEL_PY_PGMRES = 2, PARCEL_PY_PCAGMRES = 3,
  PARCEL_PY_POWER = 10, PARCEL_PY_INVERSE_POWER = 11, PARCEL_PY_KS_ARNOLDI = 12
};

/* 戻り値 */
enum {
  PARCEL_PY_OK = 0,
  PARCEL_PY_ERR_ARG = -1,        /* 引数・設定の誤り (全ランクで同じ値を返す) */
  PARCEL_PY_ERR_NOT_BUILT = -2,  /* そのソルバを外してビルドした */
  PARCEL_PY_ERR_OVERFLOW = -3,   /* 整数型の範囲を超えた (全ランクで同じ値を返す) */
  PARCEL_PY_ERR_MPI = -4,        /* MPI の初期化・コミュニケータの誤り */
  PARCEL_PY_ERR_RUNTIME = -5     /* MPI なしのビルドで, ライブラリ内の例外 (デバイスのメモリ不足など) */
};

typedef struct {
  int32_t abi_version;     /* PARCEL_PY_ABI_VERSION */
  int32_t solver;          /* PARCEL_PY_* */
  int32_t verbose;         /* 0 / 1: ランク 0 がソルバの経過を表示する */

  /* 収束判定 */
  double  tol;             /* NaN: 既定 */
  int64_t maxiter;         /* <= 0: 既定 */

  /* 前処理 (外側のソルバ) */
  int32_t precond;         /* -1: 既定 (前処理なし), 0 なし, 1 ポイント Jacobi, 2 ブロック Jacobi, 4 Jacobi */
  int64_t jacobi_iter;     /* <= 0: 既定 */
  int64_t blocks;          /* <= 0: 自動 */
  int32_t scale_matrix;    /* -1: 自動, 0, 1 */
  int32_t scale_vector;    /* -1: 自動, 0, 1 */

  /* pgmres */
  int64_t restart;         /* <= 0: 既定 */
  int32_t orthogonalization; /* -1: 既定, 0 CGS, 1 MGS */

  /* pcagmres */
  int64_t s, t;            /* <= 0: 既定 (両方とも指定するか, 両方とも未指定) */
  int32_t qr;              /* -1: 既定, 0 CGS, 1 MGS, 2 CholeskyQR, 202 モーメント法 */
  int32_t q_mode;          /* -1: 既定, 0 auto, 1 explicit, 2 implicit */

  /* 固有値ソルバ */
  int32_t precond_spectrum; /* 0: A の固有値, 1: A K^-1 の固有値 */
  int64_t ks_m, ks_k_keep, ks_nev; /* <= 0: 既定 (3 つとも指定するか, 3 つとも未指定) */
  int64_t ks_max_restart;  /* <= 0: 既定 */
  int32_t use_shift;       /* ks_arnoldi の shift-invert */
  double  sigma;
  int32_t use_v0;          /* ks_arnoldi: evecs の先頭 n_local 要素を初期ベクトルとして使う */

  /* 内部線形ソルバ (inverse_power, ks_arnoldi の shift-invert) */
  int32_t inner_method;    /* -1: 既定, 0 pgmres, 1 pcg */
  double  inner_tol;       /* NaN: 既定 */
  int64_t inner_maxiter;   /* <= 0: 既定 */
  int64_t inner_restart;   /* <= 0: 既定 */
  int32_t inner_precond;   /* -1: 既定 */
  int64_t inner_jacobi_iter; /* <= 0: 既定 */
  int64_t inner_blocks;    /* <= 0: 自動 */

  /* ks_arnoldi の直交化 (set_qr_algo。末尾に追加したので ABI の版を 3 に上げた) */
  int32_t ks_orthogonalization; /* -1: 既定, 0 CGS (CGS2), 1 MGS (MGS2) */
} parcel_py_options;

typedef struct {
  int32_t reason;          /* 収束状態 (CONVERGED_RTOL = 1, ITERATING = 0, DIVERGED_ITS = -1,
                              DIVERGED_NANORINF = -2, DIVERGED_BREAKDOWN = -3) */
  int64_t iterations;
  double  residual;
  double  true_residual;
  int64_t ks_nconv;
  int64_t linsolve_failures;
  int64_t n_eig;           /* 固有値ソルバ: 返した固有値の数 */
} parcel_py_info;

int  parcel_py_abi_version(void);
/* ライブラリの版 (Include/parcel_version.hpp の PARCEL_VERSION_STRING。"1.2.0" の形、静的な領域) */
const char *parcel_py_version_string(void);
/* MPI ありでビルドしたら 1, MPI なし (-DPARCEL_MPI=0) なら 0。
   MPI なしでは mpi_init / mpi_finalize は何もせず, comm_rank は 0, comm_size は 1, allreduce は値をそのまま返す。 */
int  parcel_py_has_mpi(void);
/* MPI を初期化していなければ初期化する。初期化したときは 1, 済んでいたときは 0, 失敗は負。 */
int  parcel_py_mpi_init(void);
/* parcel_py_mpi_init() が初期化した場合だけ MPI_Finalize する。 */
void parcel_py_mpi_finalize(void);
/* comm: MPI_Comm へのポインタ (mpi4py の MPI._addressof(comm)) か NULL (MPI_COMM_WORLD) */
int  parcel_py_comm_rank(const void *comm);
int  parcel_py_comm_size(const void *comm);
/* 全ランクの和 (op = 0), 最大 (1), 最小 (2)。集団操作。 */
double parcel_py_allreduce(const void *comm, double v, int op);
/* 固有値ソルバが返す固有値の数 (power / inverse_power は 1, ks_arnoldi は k_keep)。負は誤り。 */
int64_t parcel_py_eig_count(const parcel_py_options *opt);
/* ビルドの情報を buf に書く */
int  parcel_py_build_info(char *buf, int len);
/* 既定のセレクタで選ばれるデバイスの名前を buf に書く (キューを 1 つ作る) */
int  parcel_py_device_name(char *buf, int len);

int parcel_py_solve(const parcel_py_options *opt, const void *comm,
                    int64_t n_local, int64_t row_start, int64_t n_global, int64_t nnz_local,
                    const int64_t *indptr, const int64_t *indices, const double *data,
                    const double *b, double *x, parcel_py_info *info,
                    char *errbuf, int errlen);

/* evals: 2 * n_eig 要素 (実部と虚部を交互), evecs: n_local * n_eig 要素 (1 本ずつ連続) */
int parcel_py_eigs(const parcel_py_options *opt, const void *comm,
                   int64_t n_local, int64_t row_start, int64_t n_global, int64_t nnz_local,
                   const int64_t *indptr, const int64_t *indices, const double *data,
                   double *evals, double *evecs, parcel_py_info *info,
                   char *errbuf, int errlen);

#ifdef __cplusplus
}
#endif
#endif
