// quick_start.cpp : PARCEL_SYCL の最小例
//
//   1 次元ラプラシアン (三重対角行列, n = 1000) の A x = b (b = 1) を
//   前処理なしの共役勾配法 (pcg) で解き, 収束状態と厳密解との誤差の判定を表示する.
//   ビルドと実行は README.md を参照.
//   -DPARCEL_MPI=0 (MPI なし, 1 プロセス) でもビルドできる. 違いは #if PARCEL_MPI の 4 か所だけ.
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <vector>
#include "../Include/parcel_sycl.hpp"

int main(int argc, char *argv[]) {
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
#else
  (void)argc; (void)argv;
#endif
  {
    // 既定のデバイス (--acpp-targets で決まる)。in-order キューを使う
    //   (PARCEL はカーネルを event なしで順に投入するため、in-order でないとソルバが例外を投げる)
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
#if PARCEL_MPI
    parcel::comm comm(MPI_COMM_WORLD, &q);
#else
    parcel::comm comm(&q);                          // MPI なし: rank 0, 1 プロセス
#endif
    const int myrank = comm.mpi_rank(), nproc = comm.mpi_size();

    // 行の分割: 各ランクが連続した行を持つ (端数は最終ランク)
    const PARCEL_INT n_global = 1000;
    const PARCEL_INT nstart = (n_global / nproc) * myrank;
    const PARCEL_INT nend   = (myrank == nproc - 1) ? n_global - 1 : nstart + n_global / nproc - 1;
    const PARCEL_INT n      = nend - nstart + 1;

    // 担当行の CRS (行ポインタは局所 0 始まり, 列番号は大域 0 始まり)
    std::vector<PARCEL_INT> rowptr(n + 1, 0), col;
    std::vector<double> val;
    for (PARCEL_INT i = nstart; i <= nend; i++) {
      if (i > 0)            { col.push_back(i - 1); val.push_back(-1.0); }
      col.push_back(i);       val.push_back( 2.0);
      if (i < n_global - 1) { col.push_back(i + 1); val.push_back(-1.0); }
      rowptr[i - nstart + 1] = (PARCEL_INT)col.size();
    }
    parcel::matrix<parcel_double> A(&q);
    A.set_crsMat<double>(&comm, nstart, nend, n, col.size(), rowptr.data(), col.data(), val.data());

    std::vector<double> x_h(n, 0.0), b_h(n, 1.0);
    parcel::vector<parcel_double> x(n, &q), b(n, &q);
    x.setValue(x_h.data());
    b.setValue(b_h.data());

    // 共役勾配法 (演算はすべて倍精度)
    parcel::krylov::pcg<parcel_double, parcel_double, parcel_double, parcel_double> cg;
    cg.set_tolerance(1.0e-9);                       // 相対残差 ||r|| / ||b||
    cg.set_max_iteration(10000);
    q.wait();
    cg.solver<parcel_double, parcel_double, parcel_double,
              parcel_double, parcel_double, parcel_double>(A, x, b);
    x.getValue(x_h.data());
    q.wait();

    // 厳密解との比較. 行列は境界の外の値を 0 とした (Dirichlet 境界) 差分 -x[i-1] + 2 x[i] - x[i+1] = 1 なので,
    //   厳密解は x[i] = (i + 1) (n_global - i) / 2 (i は大域 0 始まり. x[0] = n_global / 2, 中央で最大 約 n_global^2 / 8).
    //   全成分の相対誤差 |x[i] - exact| / exact の最大を各ランクの担当行で求め, ランクの最大を集める.
    // しきい値 1e-6: 条件数は cot^2(pi / (2 (n_global + 1))) = 約 4.1e5 で, tolerance 1e-9 からの最悪の見積もり
    //   (相対誤差 <= 条件数 x 相対残差) は 4e-4. 実際はそれよりずっと小さい: b = 1 は中央に対して対称なので Krylov 部分空間は
    //   500 次元で尽き, pcg は 500 反復で残差がほぼ 0 になって, 誤差は丸め誤差の程度 (s86 の CPU・GPU, 1・2 ランクで 0〜1.3e-15).
    //   解き方の誤り (行列・分割・通信の誤り) は桁違いに大きい誤差になるので, 1e-6 で区別できる.
    const double err_limit = 1.0e-6;
    double err_local = 0.0;
    for (PARCEL_INT k = 0; k < n; k++) {
      const double i = (double)(nstart + k);
      const double exact = 0.5 * (i + 1.0) * ((double)n_global - i);
      double e = std::fabs(x_h[k] - exact) / exact;
      if (!(e == e)) e = HUGE_VAL;                  // NaN は無限大として扱う (不合格にする)
      err_local = std::max(err_local, e);
    }
    double err_max = err_local;
#if PARCEL_MPI
    MPI_Allreduce(&err_local, &err_max, 1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD);
#endif
    const bool ok = cg.is_converged() && err_max <= err_limit;

    // 収束状態と判定 (全ランクで同じ値)
    if (myrank == 0) {
      std::printf("reason = %d (%s), iterations = %d, residual = %.3e, true residual = %.3e\n",
                  (int)cg.get_converged_reason(),
                  parcel::krylov::converged_reason_name(cg.get_converged_reason()),
                  (int)cg.get_iterations(), cg.get_residual(), cg.get_true_residual());
      std::printf("x[0] = %.6f (exact %.6f)\n", x_h[0], 0.5 * 1.0 * (double)n_global);
      std::printf("check: max relative error = %.3e (limit %.1e) %s\n", err_max, err_limit, ok ? "PASS" : "FAIL");
      std::fflush(stdout);                          // MPI_Abort の前に出力を出し切る
    }
#if PARCEL_MPI
    if (!ok) { MPI_Abort(MPI_COMM_WORLD, 1); }
  }
  MPI_Finalize();
#else
    if (!ok) { return 1; }
  }
#endif
  return 0;
}
