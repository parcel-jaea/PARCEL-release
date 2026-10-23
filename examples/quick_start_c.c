/* quick_start_c.c : C言語の API で 1 次元 Laplace 作用素の三重対角行列の連立一次方程式を pcg で解く最小の例
 *   MPI あり: mpicc で -DPARCEL_MPI=1 (既定), MPI なし: gcc で -DPARCEL_MPI=0 (ラッパも同じ指定でビルドする)
 *   ビルドと実行はマニュアルのクイックスタートを参照 (パッケージのトップで -Icapi を付けてコンパイルする).
 *
 * SPDX-License-Identifier: LGPL-3.0-only
 * Copyright (c) 日本原子力研究開発機構 システム計算科学センター */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "parcel_sycl_c.h"

int main(int argc, char *argv[]) {
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
  int myrank, nproc;
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &nproc);
  PARCEL_C_COMM comm = MPI_COMM_WORLD;
#else
  int myrank = 0, nproc = 1;
  PARCEL_C_COMM comm = 0;                 /* MPI なしでは使われない */
#endif
  /* 行の分割: 各ランクが連続した行を持つ (端数は最終ランク) */
  const parcel_int n_global = 1000;
  const parcel_int nstart = (n_global / nproc) * myrank;
  const parcel_int nend   = (myrank == nproc - 1) ? n_global - 1 : nstart + n_global / nproc - 1;
  const parcel_int n      = nend - nstart + 1;

  /* 担当行の CRS (行ポインタは局所 0 始まり, 列番号は大域 0 始まり) */
  parcel_int *rowptr = malloc((n + 1) * sizeof(parcel_int));
  parcel_int *col    = malloc(3 * n * sizeof(parcel_int));
  double *val = malloc(3 * n * sizeof(double));
  double *b   = malloc(n * sizeof(double));
  double *x   = calloc(n, sizeof(double));
  parcel_int nnz = 0;
  rowptr[0] = 0;
  for (parcel_int i = nstart; i <= nend; i++) {
    if (i > 0)            { col[nnz] = i - 1; val[nnz++] = -1.0; }
    col[nnz] = i;           val[nnz++] =  2.0;
    if (i < n_global - 1) { col[nnz] = i + 1; val[nnz++] = -1.0; }
    rowptr[i - nstart + 1] = nnz;
    b[i - nstart] = 1.0;
  }

  /* 共役勾配法 (前処理なし, 倍精度) */
  parcel_options opt;
  parcel_options_init(&opt);              /* 既定値を入れてから必要な項目だけ変える */
  opt.tolerance     = 1.0e-9;             /* 相対残差 ||r|| / ||b|| */
  opt.max_iteration = 10000;
  parcel_result res;
  int rc = parcel_solve_linear(PARCEL_SOLVER_PCG, &opt, comm, n, nstart, n_global, nnz,
                               rowptr, col, val, b, x, &res);

  /* 厳密解との比較. 行列は境界の外の値を 0 とした (Dirichlet 境界) 差分 -x[i-1] + 2 x[i] - x[i+1] = 1 なので,
   *   厳密解は x[i] = (i + 1) (n_global - i) / 2 (i は大域 0 始まり). 全成分の相対誤差の最大を各ランクの担当行で求め,
   *   ランクの最大を集める. しきい値 1e-6 の根拠は quick_start.cpp を参照 (条件数 約 4.1e5, tolerance 1e-9). */
  const double err_limit = 1.0e-6;
  double err_local = 0.0, err_max;
  for (parcel_int k = 0; k < n; k++) {
    const double i = (double)(nstart + k);
    const double exact = 0.5 * (i + 1.0) * ((double)n_global - i);
    double e = fabs(x[k] - exact) / exact;
    if (!(e == e)) e = HUGE_VAL;          /* NaN は無限大として扱う (不合格にする) */
    if (e > err_local) err_local = e;
  }
  err_max = err_local;
#if PARCEL_MPI
  MPI_Allreduce(&err_local, &err_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
  const int ok = (rc == PARCEL_OK && res.converged_reason > 0 && err_max <= err_limit);
  if (myrank == 0) {
    printf("rc = %d, reason = %d, iterations = %d, residual = %.3e, true residual = %.3e\n",
           rc, res.converged_reason, res.iterations, res.residual, res.true_residual);
    printf("x[0] = %.6f (exact %.6f)\n", x[0], 0.5 * (double)n_global);
    printf("check: max relative error = %.3e (limit %.1e) %s\n", err_max, err_limit, ok ? "PASS" : "FAIL");
  }
  free(rowptr); free(col); free(val); free(b); free(x);
#if PARCEL_MPI
  MPI_Finalize();
#endif
  return ok ? 0 : 1;
}
