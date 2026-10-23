// example_pcg_ks.cpp : 線形ソルバと固有値ソルバの例 (C++)
//
//   1 次元ラプラシアン (三重対角行列, n = 100) について
//     (1) ヤコビ前処理付き共役勾配法 (pcg) で A x = b (b = 1) を解く
//     (2) Krylov-Schur 法 (ks_arnoldi) で絶対値の大きい固有値 4 個を求め,
//         解析値 λ_k = 2 - 2 cos(kπ/(n+1)) と比べる
//   いずれかが収束しない, または固有値の相対誤差が 1e-8 を超えると終了コード 1 を返す.
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include <cmath>
#include <cstdio>
#include <vector>
#include "../Include/parcel_sycl.hpp"

// -DPARCEL_MPI=0 (MPI なし, 1 プロセス) でもビルドできる (#if PARCEL_MPI の箇所だけが違う).
int main(int argc, char *argv[]) {
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
#else
  (void)argc; (void)argv;
#endif
  int fail = 0;
  {
    // in-order キュー (PARCEL はカーネルを event なしで順に投入する)
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
#if PARCEL_MPI
    parcel::comm comm(MPI_COMM_WORLD, &q);
#else
    parcel::comm comm(&q);                          // MPI なし: rank 0, 1 プロセス
#endif
    const int myrank = comm.mpi_rank(), nproc = comm.mpi_size();

    const PARCEL_INT n_global = 100;
    const PARCEL_INT nstart = (n_global / nproc) * myrank;
    const PARCEL_INT nend   = (myrank == nproc - 1) ? n_global - 1 : nstart + n_global / nproc - 1;
    const PARCEL_INT n      = nend - nstart + 1;

    std::vector<PARCEL_INT> rowptr(n + 1, 0), col;
    std::vector<double> val;
    for (PARCEL_INT i = nstart; i <= nend; i++) {
      if (i > 0)            { col.push_back(i - 1); val.push_back(-1.0); }
      col.push_back(i);       val.push_back( 2.0);
      if (i < n_global - 1) { col.push_back(i + 1); val.push_back(-1.0); }
      rowptr[i - nstart + 1] = (PARCEL_INT)col.size();
    }
    parcel::matrix<parcel_double> A(&q);
    // 注意: set_crsMat() は col と val を局所番号に書き換える
    A.set_crsMat<double>(&comm, nstart, nend, n, col.size(), rowptr.data(), col.data(), val.data());

    // ---- (1) 線形ソルバ ----
    std::vector<double> x_h(n, 0.0), b_h(n, 1.0);
    parcel::vector<parcel_double> x(n, &q), b(n, &q);
    x.setValue(x_h.data());
    b.setValue(b_h.data());
    parcel::krylov::pcg<parcel_double, parcel_double, parcel_double, parcel_double> cg;
    cg.set_tolerance(1.0e-9);
    cg.set_max_iteration(10000);
    cg.set_precon_algo(parcel::PRECON_JACOBI);
    cg.set_jacobi(1);
    q.wait();
    cg.solver<parcel_double, parcel_double, parcel_double,
              parcel_double, parcel_double, parcel_double>(A, x, b);
    q.wait();
    if (!cg.is_converged()) fail = 1;
    if (myrank == 0)
      std::printf("[pcg] reason = %s, iterations = %d, residual = %.3e\n",
                  parcel::krylov::converged_reason_name(cg.get_converged_reason()),
                  (int)cg.get_iterations(), cg.get_residual());

    // ---- (2) 固有値ソルバ: 絶対値の大きい nev 個 ----
    const PARCEL_INT m = 30, k_keep = 10, nev = 4;
    parcel::krylov::ks_arnoldi<parcel_double, parcel_double, parcel_double, parcel_double> ks;
    ks.set_ks_subspace(m, k_keep, nev);
    ks.set_ks_max_restart(200);
    ks.set_tolerance(1.0e-10);
    ks.set_precon_spectrum(false);                 // 既定: A の固有値

    std::vector<double> eval_h(2 * k_keep, 0.0);     // 実部・虚部が交互
    std::vector<double> evec_h(n * k_keep, 0.0);     // 先頭 n 要素が初期ベクトル
    for (PARCEL_INT i = 0; i < n; i++) {
      const double g = (double)(nstart + i);
      evec_h[i] = 1.0 + 0.5 * std::sin(0.317 * g + 0.7) + 0.3 * std::cos(0.181 * g - 1.3);
    }
    parcel::vector<parcel_double> eval(2 * k_keep, &q), evec(n * k_keep, &q);
    eval.setValue(eval_h.data());
    evec.setValue(evec_h.data());
    q.wait();
    ks.solver<parcel_double, parcel_double, parcel_double>(A, eval, evec);
    eval.getValue(eval_h.data());
    q.wait();
    if (!ks.is_converged()) fail = 1;
    if (myrank == 0) {
      std::printf("[ks_arnoldi] reason = %s, restarts = %d, converged pairs = %d\n",
                  parcel::krylov::converged_reason_name(ks.get_converged_reason()),
                  (int)ks.get_iterations(), (int)ks.get_ks_nconv());
      const double pi = std::acos(-1.0);
      for (PARCEL_INT j = 0; j < nev; j++) {
        const double exact = 2.0 - 2.0 * std::cos((double)(n_global - j) * pi / (double)(n_global + 1));
        const double err = std::fabs(eval_h[2 * j] - exact) / std::fabs(exact);
        std::printf("  lambda_%d = %.12f + i %.1e  (exact %.12f, rel.err %.1e)\n",
                    (int)j, eval_h[2 * j], eval_h[2 * j + 1], exact, err);
        if (err > 1.0e-8) fail = 1;
      }
      std::printf("%s\n", fail ? "FAIL" : "PASS");
    }
#if PARCEL_MPI
    MPI_Bcast(&fail, 1, MPI_INT, 0, MPI_COMM_WORLD);
#endif
  }
#if PARCEL_MPI
  MPI_Finalize();
#endif
  return fail;
}
