/* SPDX-License-Identifier: LGPL-3.0-only */
/* Copyright (c) 日本原子力研究開発機構 システム計算科学センター */
/*
 * sample_c.c : PARCEL_SYCL C API (parcel_sycl_c.h) のサンプル兼検証
 *
 * 3 次元 7 点ポアソン行列 (中心 -6, 隣接 1, ディリクレ境界, NX^3 行) に対し
 * 7 種のソルバすべてを呼び、結果を独立に検証して PASS / FAIL を出す。
 *   線形 (pcg / pbicgstab / pgmres / pcagmres) : 相対残差 ||b - Ax|| / ||b||
 *     pcagmres は Q を陽に作らない版 (cagmres_q_mode = PARCEL_CAGMRES_Q_IMPLICIT) も
 *     2 通り試す: CholeskyQR (ブロックヤコビ前処理) と対称行列用のモーメント法
 *     (PARCEL_QR_CAGMRES_CHOLESKYQR、前処理なし・s 偶数。ポアソン行列は対称なので使える)。
 *   固有値 (power / inverse_power / ks_arnoldi) : 解析固有値
 *       λ(i,j,k) = -2 Σ_d (1 - cos(k_d π / (NX+1))),  k_d = 1..NX  (すべて負)
 *     power         : 絶対値最大 λ = -6 (1 + cos(π/(NX+1)))
 *     inverse_power : 絶対値最小 λ = -6 (1 - cos(π/(NX+1)))
 *     ks_arnoldi    : 絶対値最大側 nev 個 / shift-invert (σ=-0.15) は σ に近い側 nev 個
 * さらに旧 API run_parcel_sycl_c (pcg) も呼んで互換を確かめる。
 * 最初に版 (parcel_get_version / parcel_version_string) も確かめる (計 12 項目)。
 * 各ソルバの PASS には、ライブラリが返す収束状態 (parcel_result) が収束
 * (converged_reason > 0) であることも含める。
 *
 * 行はランク r に重み (r+1) で不均等に配る (np=3 なら 1:2:3、z 面の途中で切れる)。
 * 失敗が 1 つでもあれば終了コード 1。
 *
 *   mpirun -n 3 ./sample_c [NX]      (NX 既定 16)
 *
 * MPI なし (ラッパとこのファイルを -DPARCEL_MPI=0 でコンパイル) でも組める。
 *   1 プロセスで動き (./sample_c [NX])、コミュニケータの引数には 0 を渡す (SAMPLE_COMM)。
 *   検証用の集約 (MPI_Allgatherv / MPI_Allreduce) と時間の計測はこのファイルの中で置き換える。
 *
 * 行数・非零数・行ポインタ・列番号は API の型に合わせて parcel_int (int64_t) で持つ。
 *   MPI の個数 (g_counts / g_displs) は int (このサンプルの規模では収まる)。
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "parcel_sycl_c.h"

/* ライブラリに渡すコミュニケータと時間の計測 (MPI なしでは 0 と clock_gettime) */
#if PARCEL_MPI
#include <mpi.h>
#define SAMPLE_COMM MPI_COMM_WORLD
#define wtime() MPI_Wtime()
#else
#include <time.h>
#define SAMPLE_COMM 0
static double wtime(void){
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return (double)t.tv_sec + 1.0e-9 * (double)t.tv_nsec;
}
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static int g_nx;
static int g_myrank, g_np;

/* ---- 行列生成: 大域行 [rstart, rstart+nrow) を 0 始まり大域列番号の CRS で作る ---- */
static parcel_int poisson_rows(parcel_int nx, parcel_int rstart, parcel_int nrow,
                               parcel_int *rowptr, parcel_int *col, double *val){
  parcel_int nnz = 0;
  rowptr[0] = 0;
  for(parcel_int r = 0; r < nrow; r++){
    const parcel_int g  = rstart + r;
    const parcel_int ix = g % nx, iy = (g / nx) % nx, iz = g / (nx * nx);
#define PUT(c, v) do{ if(col){ col[nnz] = (c); val[nnz] = (v); } nnz++; }while(0)
    if(iz > 0)      PUT(g - nx * nx, 1.0);
    if(iy > 0)      PUT(g - nx, 1.0);
    if(ix > 0)      PUT(g - 1, 1.0);
                    PUT(g, -6.0);
    if(ix < nx - 1) PUT(g + 1, 1.0);
    if(iy < nx - 1) PUT(g + nx, 1.0);
    if(iz < nx - 1) PUT(g + nx * nx, 1.0);
#undef PUT
    rowptr[r + 1] = nnz;
  }
  return nnz;
}

/* ---- 分散ベクトルの演算 (検証用、ライブラリとは独立) ---- */
static int *g_counts, *g_displs;
static double *g_full;

/* y_local = A x (x は分散ベクトル、全体を集めてから自分の行を掛ける) */
static void spmv(parcel_int n_local, const parcel_int *rowptr, const parcel_int *col, const double *val,
                 const double *x_local, double *y_local){
#if PARCEL_MPI
  MPI_Allgatherv(x_local, (int)n_local, MPI_DOUBLE, g_full, g_counts, g_displs, MPI_DOUBLE, MPI_COMM_WORLD);
#else
  memcpy(g_full, x_local, sizeof(double) * (size_t)n_local);   /* 1 プロセス: 全体 = 自分の分 */
#endif
  for(parcel_int i = 0; i < n_local; i++){
    double s = 0.0;
    for(parcel_int k = rowptr[i]; k < rowptr[i + 1]; k++) s += val[k] * g_full[col[k]];
    y_local[i] = s;
  }
}
static double dot(parcel_int n, const double *a, const double *b){
  double s = 0.0, t = 0.0;
  for(parcel_int i = 0; i < n; i++) s += a[i] * b[i];
#if PARCEL_MPI
  MPI_Allreduce(&s, &t, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
#else
  t = s;
#endif
  return t;
}

/* ---- 解析固有値 ---- */
static double g_sigma;       /* 並べ替えの基準: |λ - σ| */
static int g_desc;           /* 1: 降順, 0: 昇順 */
static int cmp_eig(const void *a, const void *b){
  const double da = fabs(*(const double *)a - g_sigma), db = fabs(*(const double *)b - g_sigma);
  if(da == db) return 0;
  return ((da < db) ^ g_desc) ? -1 : 1;
}
/* 全固有値を |λ-σ| で並べて先頭 k 個を ref に入れる */
static void analytic_eigs(int nx, double sigma, int desc, int k, double *ref){
  const int n = nx * nx * nx;
  double *all = (double *)malloc(sizeof(double) * n);
  double *c = (double *)malloc(sizeof(double) * nx);
  for(int i = 0; i < nx; i++) c[i] = 2.0 * (1.0 - cos((i + 1) * M_PI / (nx + 1)));
  int p = 0;
  for(int i = 0; i < nx; i++) for(int j = 0; j < nx; j++) for(int l = 0; l < nx; l++)
    all[p++] = -(c[i] + c[j] + c[l]);
  g_sigma = sigma; g_desc = desc;
  qsort(all, n, sizeof(double), cmp_eig);
  for(int i = 0; i < k; i++) ref[i] = all[i];
  free(all); free(c);
}

static int n_fail = 0;
/* res が NULL でなければ収束状態も判定に含め、次の行に表示する */
static void report(const char *name, int rc, int pass, const char *detail, const parcel_result *res){
  if(rc != PARCEL_OK) pass = 0;
  if(res != NULL && !(res->converged_reason > 0)) pass = 0;
  if(!pass) n_fail++;
  if(g_myrank == 0){
    printf("[sample_c] %-22s %s  rc=%d  %s\n", name, pass ? "PASS" : "FAIL", rc, detail);
    if(res != NULL)
      printf("             converged_reason = %d (> 0 で収束)  its = %d  residual = %.3e  true_residual = %.3e"
             "  ks_nconv = %d  linsolve_failures = %d\n",
             res->converged_reason, res->iterations, res->residual, res->true_residual,
             res->ks_nconv, res->linsolve_failures);
  }
  fflush(stdout);
}

int main(int argc, char *argv[]){
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_myrank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_np);
#else
  g_myrank = 0; g_np = 1;
#endif

  g_nx = (argc > 1) ? atoi(argv[1]) : 16;
  const int nx = g_nx;
  const parcel_int n_global = (parcel_int)nx * nx * nx;

  /* ---- 不均等分割: ランク r の重み r+1 ---- */
  g_counts = (int *)malloc(sizeof(int) * g_np);
  g_displs = (int *)malloc(sizeof(int) * g_np);
  {
    const long long W = (long long)g_np * (g_np + 1) / 2;
    long long acc = 0;
    for(int r = 0; r < g_np; r++){
      const int s = (int)((long long)n_global * acc / W);
      acc += r + 1;
      const int e = (int)((long long)n_global * acc / W);
      g_displs[r] = s; g_counts[r] = e - s;
    }
  }
  const parcel_int nstart  = g_displs[g_myrank];
  const parcel_int n_local = g_counts[g_myrank];
  g_full = (double *)malloc(sizeof(double) * n_global);

  parcel_int *rowptr = (parcel_int *)malloc(sizeof(parcel_int) * (n_local + 1));
  const parcel_int nnz_local = poisson_rows(nx, nstart, n_local, rowptr, NULL, NULL);
  parcel_int *col = (parcel_int *)malloc(sizeof(parcel_int) * nnz_local);
  double *val = (double *)malloc(sizeof(double) * nnz_local);
  poisson_rows(nx, nstart, n_local, rowptr, col, val);

  if(g_myrank == 0){
    printf("[sample_c] NX=%d n_global=%lld nprocs=%d rows/rank:", nx, (long long)n_global, g_np);
    for(int r = 0; r < g_np; r++) printf(" %d", g_counts[r]);
    printf("\n");
  }

  double *b  = (double *)malloc(sizeof(double) * n_local);
  double *x  = (double *)malloc(sizeof(double) * n_local);
  double *ax = (double *)malloc(sizeof(double) * n_local);
  char detail[256];

  /* ================= 版 ================= */
  {
    /* 番号と文字列が同じ版を指すか。PARCEL_EXPECT_VERSION を与えればその値とも比べる (回帰試験の生成器が与える) */
    int vmaj = -1, vmin = -1, vpat = -1;
    const int rc = parcel_get_version(&vmaj, &vmin, &vpat);
    const char *vstr = parcel_version_string();
    char vnum[64];
    snprintf(vnum, sizeof(vnum), "%d.%d.%d", vmaj, vmin, vpat);
    const char *vexp = getenv("PARCEL_EXPECT_VERSION");
    if(vexp != NULL && vexp[0] == '\0') vexp = NULL;
    const int pass = vstr != NULL && strcmp(vstr, vnum) == 0 && (vexp == NULL || strcmp(vstr, vexp) == 0);
    snprintf(detail, sizeof(detail), "parcel_version_string = %s  parcel_get_version = %s  (expect %s)",
             vstr != NULL ? vstr : "(null)", vnum, vexp != NULL ? vexp : "-");
    report("version", rc, pass, detail, NULL);
  }

  /* ================= 線形ソルバ ================= */
  const double lin_tol = 1.0e-9, lin_check = 1.0e-7;
  /* 最後の 2 つはどちらも pcagmres の Q を陽に作らない版 (QR の選び方だけ違う。下の設定を参照) */
  const int lin_solvers[6] = { PARCEL_SOLVER_PCG, PARCEL_SOLVER_PBICGSTAB,
                               PARCEL_SOLVER_PGMRES, PARCEL_SOLVER_PCAGMRES,
                               PARCEL_SOLVER_PCAGMRES, PARCEL_SOLVER_PCAGMRES };
  const char *lin_names[6] = { "pcg", "pbicgstab", "pgmres", "pcagmres",
                               "pcagmres(implicit)", "pcagmres(impl,sym)" };
  for(parcel_int i = 0; i < n_local; i++) b[i] = 1.0 + 0.1 * sin(0.37 * (nstart + i));
  const double bnorm = sqrt(dot(n_local, b, b));

  for(int s = 0; s < 6; s++){
    parcel_options opt;
    parcel_options_init(&opt);
    opt.tolerance     = lin_tol;
    opt.max_iteration = 10000;
    opt.precon_algo   = PARCEL_PRECON_BLOCKJACOBI;
    /* ライブラリの既定値によらず、このサンプルで使う設定を明示する */
    opt.gmres_restart   = 8;
    opt.gmres_qr_algo   = PARCEL_QR_MGS;
    opt.cagmres_s       = 8;
    opt.cagmres_t       = 1;
    opt.cagmres_qr_algo = PARCEL_QR_MGS;
    if(s == 4){
      /* Q を陽に作らない版: CholeskyQR。t = 2 でブロック (k > 0) の直交化も通す */
      opt.cagmres_t       = 2;
      opt.cagmres_q_mode  = PARCEL_CAGMRES_Q_IMPLICIT;
      opt.cagmres_qr_algo = PARCEL_QR_CHOLESKYQR;
    }else if(s == 5){
      /* モーメント法 (IMPLICIT 専用): A が対称であること (ここでは成り立つ)、前処理なし、s が偶数が必要 */
      opt.precon_algo     = PARCEL_PRECON_NONE;
      opt.cagmres_t       = 2;
      opt.cagmres_q_mode  = PARCEL_CAGMRES_Q_IMPLICIT;
      opt.cagmres_qr_algo = PARCEL_QR_CAGMRES_CHOLESKYQR;
    }
    opt.verbose       = (s == 0);
    for(parcel_int i = 0; i < n_local; i++) x[i] = 0.0;
    parcel_result pres;
    const double t0 = wtime();
    const int rc = parcel_solve_linear(lin_solvers[s], &opt, SAMPLE_COMM,
                                       n_local, nstart, n_global, nnz_local,
                                       rowptr, col, val, b, x, &pres);
    const double t1 = wtime();
    spmv(n_local, rowptr, col, val, x, ax);
    for(parcel_int i = 0; i < n_local; i++) ax[i] = b[i] - ax[i];
    const double rel = sqrt(dot(n_local, ax, ax)) / bnorm;
    snprintf(detail, sizeof(detail), "||b-Ax||/||b|| = %.3e (<= %.0e)  %.2f s", rel, lin_check, t1 - t0);
    report(lin_names[s], rc, rel <= lin_check, detail, &pres);
  }

  /* ================= 固有値ソルバ ================= */
  {
    /* ---- power: 絶対値最大 ---- */
    parcel_options opt;
    parcel_options_init(&opt);
    opt.tolerance     = 1.0e-10;          /* 固有値の相対変化 */
    opt.max_iteration = 20000;
    double ev[2], ref;
    double *v = (double *)malloc(sizeof(double) * n_local);
    parcel_result pres;
    const double t0 = wtime();
    const int rc = parcel_solve_eigen(PARCEL_SOLVER_POWER, &opt, SAMPLE_COMM,
                                      n_local, nstart, n_global, nnz_local,
                                      rowptr, col, val, ev, v, &pres);
    const double t1 = wtime();
    analytic_eigs(nx, 0.0, 1, 1, &ref);
    const double err = fabs(ev[0] - ref) / fabs(ref);
    snprintf(detail, sizeof(detail), "lambda = %.12f  exact = %.12f  rel.err = %.2e (<= 1e-6)  %.2f s",
             ev[0], ref, err, t1 - t0);
    report("power", rc, err <= 1.0e-6, detail, &pres);
    free(v);
  }
  {
    /* ---- inverse_power: 絶対値最小 (内部 pgmres + ブロックヤコビ) ---- */
    parcel_options opt;
    parcel_options_init(&opt);
    opt.tolerance            = 1.0e-10;
    opt.max_iteration        = 200;
    opt.linsolve_algo        = PARCEL_LINSOLVE_PGMRES;
    opt.linsolve_tolerance   = 1.0e-11;
    opt.linsolve_max_iteration = 5000;
    opt.linsolve_restart     = 30;
    opt.linsolve_precon_algo = PARCEL_PRECON_BLOCKJACOBI;
    double ev[2], ref;
    double *v = (double *)malloc(sizeof(double) * n_local);
    parcel_result pres;
    const double t0 = wtime();
    const int rc = parcel_solve_eigen(PARCEL_SOLVER_INVERSE_POWER, &opt, SAMPLE_COMM,
                                      n_local, nstart, n_global, nnz_local,
                                      rowptr, col, val, ev, v, &pres);
    const double t1 = wtime();
    analytic_eigs(nx, 0.0, 0, 1, &ref);
    const double err = fabs(ev[0] - ref) / fabs(ref);
    /* 固有対の残差 (参考) */
    spmv(n_local, rowptr, col, val, v, ax);
    for(parcel_int i = 0; i < n_local; i++) ax[i] -= ev[0] * v[i];
    const double res = sqrt(dot(n_local, ax, ax)) / (fabs(ev[0]) * sqrt(dot(n_local, v, v)));
    snprintf(detail, sizeof(detail), "lambda = %.12f  exact = %.12f  rel.err = %.2e (<= 1e-6)  resid = %.1e  %.2f s",
             ev[0], ref, err, res, t1 - t0);
    report("inverse_power", rc, err <= 1.0e-6, detail, &pres);
    free(v);
  }
  /* ---- ks_arnoldi: (a) 絶対値最大側 nev 個, (b) shift-invert σ=-0.15 近傍 nev 個 ----
   *   固有値は縮重している (2 番目の固有値は 3 重) ので、parcel_krylov_sample_verify.cpp と
   *   同じく「各 Ritz 値が最も近い解析固有値と一致するか」で判定し、加えて
   *   最良の 1 個 (|λ| 最大 / σ 最近接) が解析値の端と一致すること、
   *   固有対の残差が小さいことを確かめる。多重度まで再現したかは参考表示。 */
  for(int mode = 0; mode < 2; mode++){
    parcel_options opt;
    parcel_options_init(&opt);
    opt.tolerance      = 1.0e-8;          /* Ritz 残差 */
    opt.ks_m           = 30;
    opt.ks_k_keep      = 10;
    opt.ks_nev         = 4;
    opt.ks_max_restart = 100;
    const double sigma = -0.15;
    if(mode == 1){
      opt.shift_enable         = 1;       /* precon_spectrum=0: ラッパが (A - σI) を組む */
      opt.shift_sigma          = sigma;
      opt.linsolve_tolerance   = 1.0e-11;
      /* 内部 pgmres の上限。C の既定 (0) は ks_arnoldi の既定 2000 になるので、5000 を明示する */
      opt.linsolve_max_iteration = 5000;
      opt.linsolve_restart     = 30;
      opt.linsolve_precon_algo = PARCEL_PRECON_BLOCKJACOBI;
    }
    const int k = parcel_eigen_count(PARCEL_SOLVER_KS_ARNOLDI, &opt);
    const int nev = opt.ks_nev;
    double *ev = (double *)malloc(sizeof(double) * 2 * k);
    double *v  = (double *)malloc(sizeof(double) * (size_t)n_local * k);
    parcel_result pres;
    const double t0 = wtime();
    const int rc = parcel_solve_eigen(PARCEL_SOLVER_KS_ARNOLDI, &opt, SAMPLE_COMM,
                                      n_local, nstart, n_global, nnz_local,
                                      rowptr, col, val, ev, v, &pres);
    const double t1 = wtime();
    const double key = (mode == 0) ? 0.0 : sigma;
    const int desc = (mode == 0) ? 1 : 0;
    double *ref = (double *)malloc(sizeof(double) * n_global);   /* 全解析固有値 (key で整列) */
    double *got = (double *)malloc(sizeof(double) * nev);
    analytic_eigs(nx, key, desc, (int)n_global, ref);
    double maxim = 0.0, maxres = 0.0, maxerr = 0.0;
    for(int j = 0; j < nev; j++){
      got[j] = ev[2 * j];
      if(!(fabs(ev[2 * j + 1]) <= maxim)) maxim = fabs(ev[2 * j + 1]);
      const double *vj = v + (size_t)j * n_local;
      spmv(n_local, rowptr, col, val, vj, ax);
      for(parcel_int i = 0; i < n_local; i++) ax[i] -= got[j] * vj[i];
      const double r = sqrt(dot(n_local, ax, ax)) / (fabs(got[j]) * sqrt(dot(n_local, vj, vj)));
      if(!(r <= maxres)) maxres = r;
      /* 最も近い解析固有値との相対誤差 */
      double best = INFINITY;
      for(parcel_int p = 0; p < n_global; p++){
        const double d = fabs(got[j] - ref[p]);
        if(d < best) best = d;
      }
      const double e = best / fabs(got[j]);
      if(!(e <= maxerr)) maxerr = e;                 /* NaN も拾う */
    }
    g_sigma = key; g_desc = desc;
    qsort(got, nev, sizeof(double), cmp_eig);
    const double err_top = fabs(got[0] - ref[0]) / fabs(ref[0]);
    int n_multi = 0;                                 /* 解析値の先頭 nev 個と多重度込みで一致した数 */
    for(int j = 0; j < nev; j++) if(fabs(got[j] - ref[j]) <= 1.0e-6 * fabs(ref[j])) n_multi++;
    const int pass = (maxerr <= 1.0e-6) && (err_top <= 1.0e-6) && (maxim <= 1.0e-8) && (maxres <= 1.0e-6);
    snprintf(detail, sizeof(detail),
             "%d eigs: nearest rel.err = %.2e, extreme rel.err = %.2e (<= 1e-6)  max|Im| = %.1e  resid = %.1e (<= 1e-6)  %.2f s",
             nev, maxerr, err_top, maxim, maxres, t1 - t0);
    report(mode == 0 ? "ks_arnoldi" : "ks_arnoldi(shift-inv)", rc, pass, detail, &pres);
    if(g_myrank == 0){
      for(int j = 0; j < nev; j++) printf("             %2d  %.12f  exact(sorted) %.12f\n", j, got[j], ref[j]);
      printf("             multiplicity reproduced: %d / %d (reference only)\n", n_multi, nev);
    }
    free(ev); free(v); free(ref); free(got);
  }

  /* ================= 旧 API (互換) ================= */
  {
    /* run_parcel_sycl_c は入力配列を書き換えないので同じ配列をそのまま渡せる。
     * 反復上限・前処理はラッパのコンパイル時マクロ (MAX_ITERATION 等) で決まる。 */
    for(parcel_int i = 0; i < n_local; i++) x[i] = 0.0;
    run_parcel_sycl_c(SAMPLE_COMM, n_local, nnz_local, val, col, rowptr, x, b);
    spmv(n_local, rowptr, col, val, x, ax);
    for(parcel_int i = 0; i < n_local; i++) ax[i] = b[i] - ax[i];
    const double rel = sqrt(dot(n_local, ax, ax)) / bnorm;
    snprintf(detail, sizeof(detail), "||b-Ax||/||b|| = %.3e (<= %.0e)", rel, lin_check);
    report("legacy run_parcel_sycl_c", PARCEL_OK, rel <= lin_check, detail, NULL);
  }

  if(g_myrank == 0){
    printf("[sample_c] %s (%d failed)\n", n_fail == 0 ? "ALL PASS" : "FAILED", n_fail);
  }
  free(b); free(x); free(ax); free(rowptr); free(col); free(val);
  free(g_full); free(g_counts); free(g_displs);
#if PARCEL_MPI
  MPI_Finalize();
#endif
  return n_fail == 0 ? 0 : 1;
}
