// =========================================================================
// wrapper_parcel_sycl.cpp
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// PARCEL_SYCL の C / Fortran 向けラッパ。API は parcel_sycl_c.h を参照。
//
//   parcel_solve_linear()  : pcg / pbicgstab / pgmres / pcagmres
//                            (Q を陽に作らない CA-GMRES は cagmres_q_mode = IMPLICIT)
//   parcel_solve_eigen()   : power / inverse_power / ks_arnoldi
//   run_parcel_sycl_c/_f() : 旧 API (pcg のみ)。上の新 API で実装している。
//
// 内部のデータ精度・演算精度はこのファイルのコンパイル時マクロで決まる
// (parcel_capi_default.hpp の parcel_solver_*_type。既定はすべて倍精度)。
// ビルドするソルバは parcel_solver_select.hpp の PARCEL_ENABLE_* で絞れる。
// 外したソルバを指定すると PARCEL_ERR_NOT_BUILT を返す。
// API の整数 (行数・非零数・行ポインタ・列番号) は parcel_int (int64_t) 固定。
//   内部の PARCEL_INT は既定 64 ビット。-DPARCEL_INT32 でビルドすると 32 ビットになり、
//   収まらない値は配列を確保する前に PARCEL_ERR_ARG で返す (fits_internal を参照)。
// -DPARCEL_MPI=0 で MPI なしのビルド (1 プロセス) ができる。コミュニケータの引数は
//   int (PARCEL_C_COMM / PARCEL_C_FCOMM) になり使わない。parcel_sycl_c.h の説明を参照。
// =========================================================================
#if !defined(PARCEL_MPI) || PARCEL_MPI
#include <mpi.h>
#endif

#include "../Include/parcel_sycl.hpp"
#include "./parcel_capi_default.hpp"
#include "./parcel_sycl_c.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <exception>
#include <iostream>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace {

  using DT  = parcel_solver_data_type;             // 行列・ベクトルのデータ精度
  using CT  = parcel_solver_calc_type;             // 演算精度
  using CPT = parcel_solver_calc_precon_type;      // 前処理の演算精度
  using PMT = parcel_solver_data_precon_mat_type;  // 前処理行列のデータ精度
  using PVT = parcel_solver_data_precon_vec_type;  // 前処理ベクトルのデータ精度
  using QT  = parcel_solver_calc_qr_type;          // pcagmres の QR 分解の精度

  // C の定数と C++ ライブラリの定数が一致することを確かめる (値を写し間違えたらビルドで止まる)
  static_assert(PARCEL_PRECON_NONE        == parcel::PRECON_NONE,        "PRECON_NONE");
  static_assert(PARCEL_PRECON_POINTJACOBI == parcel::PRECON_POINTJACOBI, "PRECON_POINTJACOBI");
  static_assert(PARCEL_PRECON_BLOCKJACOBI == parcel::PRECON_BLOCKJACOBI, "PRECON_BLOCKJACOBI");
  static_assert(PARCEL_PRECON_JACOBI      == parcel::PRECON_JACOBI,      "PRECON_JACOBI");
#if PARCEL_ENABLE_PGMRES || PARCEL_ENABLE_PCAGMRES
  // parcel::qr は gmres / cagmres のヘッダが読み込む
  static_assert(PARCEL_QR_CGS             == parcel::qr::QR_CGS,         "QR_CGS");
  static_assert(PARCEL_QR_MGS             == parcel::qr::QR_MGS,         "QR_MGS");
  static_assert(PARCEL_QR_CHOLESKYQR      == parcel::qr::QR_CHOLESKYQR,  "QR_CHOLESKYQR");
  static_assert(PARCEL_QR_CAGMRES_CHOLESKYQR == parcel::qr::QR_CAGMRES_CHOLESKYQR, "QR_CAGMRES_CHOLESKYQR");
#endif
#if PARCEL_ENABLE_PCAGMRES
  static_assert(PARCEL_CAGMRES_Q_AUTO     == parcel::krylov::CAGMRES_Q_AUTO,     "CAGMRES_Q_AUTO");
  static_assert(PARCEL_CAGMRES_Q_EXPLICIT == parcel::krylov::CAGMRES_Q_EXPLICIT, "CAGMRES_Q_EXPLICIT");
  static_assert(PARCEL_CAGMRES_Q_IMPLICIT == parcel::krylov::CAGMRES_Q_IMPLICIT, "CAGMRES_Q_IMPLICIT");
#endif
  static_assert(PARCEL_LINSOLVE_PGMRES    == parcel::krylov::LINSOLVE_PGMRES, "LINSOLVE_PGMRES");
  static_assert(PARCEL_LINSOLVE_PCG       == parcel::krylov::LINSOLVE_PCG,    "LINSOLVE_PCG");
  static_assert(PARCEL_CONVERGED_RTOL     == parcel::krylov::CONVERGED_RTOL,     "CONVERGED_RTOL");
  static_assert(PARCEL_ITERATING          == parcel::krylov::ITERATING,          "ITERATING");
  static_assert(PARCEL_DIVERGED_ITS       == parcel::krylov::DIVERGED_ITS,       "DIVERGED_ITS");
  static_assert(PARCEL_DIVERGED_NANORINF  == parcel::krylov::DIVERGED_NANORINF,  "DIVERGED_NANORINF");
  static_assert(PARCEL_DIVERGED_BREAKDOWN == parcel::krylov::DIVERGED_BREAKDOWN, "DIVERGED_BREAKDOWN");

  // C / Fortran API の整数は parcel_int (int64_t) 固定。内部の PARCEL_INT は既定 64 ビット、
  //   -DPARCEL_INT32 で 32 ビット。32 ビットのときは受け取った値が収まるかを確保の前に調べ、
  //   収まらなければ PARCEL_ERR_ARG を返す (make_local_crs / check_shift_range)。
  static_assert(sizeof(parcel_int) == 8, "parcel_int must be 64-bit");
  static_assert(sizeof(PARCEL_INT) * 8 == PARCEL_INT_BITS, "PARCEL_INT_BITS mismatch");
  // v が内部の PARCEL_INT に収まるか (64 ビット版では常に真)
  inline bool fits_internal(parcel_int v){
#if PARCEL_INT_BITS == 32
    return v >= (parcel_int)std::numeric_limits<PARCEL_INT>::min() &&
           v <= (parcel_int)std::numeric_limits<PARCEL_INT>::max();
#else
    (void)v;
    return true;
#endif
  }

  // ---------------------------------------------------------------------
  // 収束状態を parcel_result に写す (res が NULL なら何もしない)
  void clear_result(parcel_result *res){
    if(res == nullptr) return;
    res->converged_reason  = PARCEL_ITERATING;
    res->iterations        = 0;
    res->residual          = std::numeric_limits<double>::quiet_NaN();
    res->true_residual     = std::numeric_limits<double>::quiet_NaN();
    res->ks_nconv          = 0;
    res->linsolve_failures = 0;
  }
  void store_result(const parcel::krylov::krylov_param &kp, parcel_result *res){
    if(res == nullptr) return;
    res->converged_reason = (int)kp.get_converged_reason();
    res->iterations       = (int)kp.get_iterations();
    res->residual         = kp.get_residual();
    res->true_residual    = kp.get_true_residual();
  }

  // ---------------------------------------------------------------------
  // 全ランクで同じ戻り値にする (どこかのランクが失敗したら全員が失敗を返す)。
  //   エラーコードは負なので最小値をとれば最も「悪い」値になる。
  //   MPI なしでは 1 プロセスなので rc のまま。
  int agree(PARCEL_C_COMM comm, int rc){
    int g = rc;
#if PARCEL_MPI
    MPI_Allreduce(&rc, &g, 1, MPI_INT, MPI_MIN, comm);
#else
    (void)comm;
#endif
    return g;
  }

  // ---------------------------------------------------------------------
  // 入力 CRS を PARCEL に渡せる形 (0 始まり、rowptr[0] = 0、PARCEL_INT) に複製する。
  //   PARCEL の set_crsMat() は列番号と値を書き換えるので、利用者の配列は直接渡さない。
  struct local_crs {
    PARCEL_INT nstart = 0, nend = -1, n = 0, nnz = 0;
    std::vector<PARCEL_INT> rowptr, col;
    std::vector<double>     val;
  };

  // 引数検査 + 複製。戻り値はこのランクだけの結果 (呼び出し側で agree() する)。
  //   整数引数は parcel_int (int64_t)。-DPARCEL_INT32 のビルドでは、行数・非零数・
  //   先頭行が 32 ビットに収まるかを配列を確保する前に調べる (収まらなければ PARCEL_ERR_ARG)。
  //   rowptr の差は 0 以上 nnz_local 以下、列番号は n_global 未満であることを下で調べるので、
  //   nnz_local と n_global が収まれば個々の値も収まる。
  int make_local_crs(const parcel_options *opt,
                     parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                     const parcel_int *rowptr, const parcel_int *col, const double *val,
                     local_crs &A){
    if(opt == nullptr) return PARCEL_ERR_ARG;
    const int base = opt->index_base;
    if(base != 0 && base != 1) return PARCEL_ERR_ARG;
    if(n_local < 1 || n_global < 1 || nnz_local < 0) return PARCEL_ERR_ARG;
    if(rowptr == nullptr || (nnz_local > 0 && (col == nullptr || val == nullptr))) return PARCEL_ERR_ARG;
    const parcel_int ns = nstart - base;
    // ns + n_local > n_global を桁あふれしない形で調べる (n_local >= 1, n_global >= 1 は確認済み)
    if(ns < 0 || n_local > n_global || ns > n_global - n_local) return PARCEL_ERR_ARG;
    // 32 ビット版: 内部の PARCEL_INT に収まるか (確保の前に。64 ビット版では常に通る)
    if(!fits_internal(n_global) || !fits_internal(nnz_local)) return PARCEL_ERR_ARG;
    if(rowptr[n_local] - rowptr[0] != nnz_local) return PARCEL_ERR_ARG;
    // rowptr の単調性を確保の前に 64 ビットのまま調べる (32 ビット版で変換後に比べると桁あふれを見逃す)
    for(parcel_int i = 1; i <= n_local; i++){
      if(rowptr[i] < rowptr[i-1]) return PARCEL_ERR_ARG;
    }

    A.nstart = (PARCEL_INT)ns;
    A.n      = (PARCEL_INT)n_local;
    A.nend   = (PARCEL_INT)(ns + n_local - 1);
    A.nnz    = (PARCEL_INT)nnz_local;
    A.rowptr.resize((size_t)n_local + 1);
    for(parcel_int i = 0; i <= n_local; i++){
      A.rowptr[i] = (PARCEL_INT)(rowptr[i] - rowptr[0]);
    }
    A.col.resize((size_t)nnz_local);
    A.val.assign(val, val + nnz_local);
    for(parcel_int k = 0; k < nnz_local; k++){
      const parcel_int c = col[k] - base;
      if(c < 0 || c >= n_global) return PARCEL_ERR_ARG;
      A.col[k] = (PARCEL_INT)c;
    }
    return PARCEL_OK;
  }

  // 行ブロックの分割が [0, n_global) を重複・欠落なく覆うかを確かめる (集団操作)。
  //   PARCEL 本体は各ランクの [nstart, nend] を交換して袖通信表を作るだけなので、
  //   ここが崩れていると通信表が壊れて黙って誤った結果になる。
  //   n_global は parcel_int (64 ビット) で受け、和も 64 ビットで取る。
  //   MPI なしでは自分の行ブロックだけで同じ判定をする (nstart = 0 かつ n_local = n_global のときだけ通る)。
  int check_distribution(PARCEL_C_COMM comm, const local_crs &A, parcel_int n_global){
    int np = 1;
#if PARCEL_MPI
    MPI_Comm_size(comm, &np);
#else
    (void)comm;
#endif
    int64_t mine[2] = { (int64_t)A.nstart, (int64_t)A.n };
    std::vector<int64_t> all(2 * (size_t)np);
#if PARCEL_MPI
    MPI_Allgather(mine, 2, MPI_INT64_T, all.data(), 2, MPI_INT64_T, comm);
#else
    all[0] = mine[0]; all[1] = mine[1];
#endif
    std::vector<std::pair<int64_t,int64_t> > blk(np);
    for(int r = 0; r < np; r++) blk[r] = std::make_pair(all[2*r], all[2*r+1]);
    std::sort(blk.begin(), blk.end());
    int64_t next = 0;
    for(int r = 0; r < np; r++){
      if(blk[r].first != next) return PARCEL_ERR_DISTRIBUTION;
      next += blk[r].second;   // 各項は n_global 以下 (make_local_crs で確認済み) なので 64 ビットで溢れない
    }
    return (next == (int64_t)n_global) ? PARCEL_OK : PARCEL_ERR_DISTRIBUTION;
  }

  // (A - σI) を作る。対角要素が無い行には -σ を足す。
  void make_shifted(const local_crs &A, double sigma, local_crs &S){
    S.nstart = A.nstart; S.nend = A.nend; S.n = A.n;
    S.rowptr.assign((size_t)A.n + 1, 0);
    S.col.clear(); S.val.clear();
    S.col.reserve(A.col.size() + A.n); S.val.reserve(A.val.size() + A.n);
    for(PARCEL_INT i = 0; i < A.n; i++){
      const PARCEL_INT grow = A.nstart + i;
      bool has_diag = false;
      for(PARCEL_INT k = A.rowptr[i]; k < A.rowptr[i+1]; k++){
        double v = A.val[k];
        if(A.col[k] == grow){ v -= sigma; has_diag = true; }
        S.col.push_back(A.col[k]); S.val.push_back(v);
      }
      if(!has_diag){ S.col.push_back(grow); S.val.push_back(-sigma); }
      S.rowptr[i+1] = (PARCEL_INT)S.col.size();
    }
    S.nnz = (PARCEL_INT)S.col.size();
  }

  // ---------------------------------------------------------------------
  // 前処理の設定 (外側ソルバ / 内部ソルバ共通の書き方)
  template<class Tset_algo, class Tset_jacobi, class Tset_bj>
  void apply_precon_common(int algo, int jacobi_iter, int bj,
                           Tset_algo set_algo, Tset_jacobi set_jacobi, Tset_bj set_bj){
    set_algo((PARCEL_INT)algo);
    if(algo == parcel::PRECON_JACOBI)                 set_jacobi((PARCEL_INT)jacobi_iter);
    if(algo == parcel::PRECON_BLOCKJACOBI && bj > 0)  set_bj((PARCEL_INT)bj);
  }

  void apply_precon(parcel::krylov::krylov_param &kp, const parcel_options &o){
    apply_precon_common(o.precon_algo, o.jacobi_iter, o.blockjacobi,
                        [&](PARCEL_INT a){ kp.set_precon_algo(a); },
                        [&](PARCEL_INT j){ kp.set_jacobi(j); },
                        [&](PARCEL_INT b){ kp.set_blockjacobi(b); });
    // -1 は自動 (前処理の種類と精度から PARCEL が決める)
    if(o.scaling_matrix >= 0) kp.set_scaling_matrix(o.scaling_matrix != 0);
    if(o.scaling_vector >= 0) kp.set_scaling_vector(o.scaling_vector != 0);
    kp.set_tolerance(o.tolerance);
    kp.set_max_iteration((PARCEL_INT)o.max_iteration);
  }

  void apply_eigen(parcel::krylov::eigen_param &ep, const parcel_options &o){
    apply_precon(ep, o);
    ep.set_precon_spectrum(o.precon_spectrum != 0);
    ep.set_linsolve_algo(o.linsolve_algo == PARCEL_LINSOLVE_PCG ? parcel::krylov::LINSOLVE_PCG
                                                                : parcel::krylov::LINSOLVE_PGMRES);
    // 0 以下はソルバの既定 (inverse_power と ks_arnoldi で既定が違うので、ここでは上書きしない)
    if(o.linsolve_tolerance > 0.0)    ep.set_linsolve_tolerance(o.linsolve_tolerance);
    if(o.linsolve_max_iteration > 0)  ep.set_linsolve_max_iteration((PARCEL_INT)o.linsolve_max_iteration);
    if(o.linsolve_restart > 0)        ep.set_linsolve_restart((PARCEL_INT)o.linsolve_restart);
    apply_precon_common(o.linsolve_precon_algo, o.linsolve_jacobi_iter, o.linsolve_blockjacobi,
                        [&](PARCEL_INT a){ ep.set_linsolve_precon_algo(a); },
                        [&](PARCEL_INT j){ ep.set_linsolve_jacobi(j); },
                        [&](PARCEL_INT b){ ep.set_linsolve_blockjacobi(b); });
  }

  bool valid_precon(int a){
    return a == PARCEL_PRECON_NONE || a == PARCEL_PRECON_POINTJACOBI ||
           a == PARCEL_PRECON_BLOCKJACOBI || a == PARCEL_PRECON_JACOBI;
  }
  bool valid_qr(int a){
    return a == PARCEL_QR_CGS || a == PARCEL_QR_MGS || a == PARCEL_QR_CHOLESKYQR ||
           a == PARCEL_QR_CAGMRES_CHOLESKYQR;
  }
  // pgmres の Arnoldi 直交化は CGS / MGS だけ。CholeskyQR を渡すと直交化を何もしないまま進むので受け付けない。
  //   CAGMRES_CHOLESKYQR (202) も pgmres には無い (PARCEL_ERR_ARG)。
  bool valid_gmres_qr(int a){ return a == PARCEL_QR_CGS || a == PARCEL_QR_MGS; }
  // pcagmres の設定の検査。
  //   ライブラリは不正な組合せを solver() の入口で std::invalid_argument にするが、例外は MPI_Abort に
  //   なるので、ここで先に PARCEL_ERR_ARG で弾く (parcel::krylov::pcagmres::check_q_combination と同じ規則)。
  //     cagmres_q_mode は AUTO / EXPLICIT / IMPLICIT
  //     IMPLICIT は CHOLESKYQR か CAGMRES_CHOLESKYQR (CGS / MGS は不可)
  //     CAGMRES_CHOLESKYQR (モーメント法) は IMPLICIT のときだけ。さらに前処理なし・s が偶数 (>= 2)
  //       (ライブラリも solver() の入口で調べる)。A が対称かどうかはライブラリも調べない (利用者の責任)。
  int check_cagmres_options(const parcel_options &o){
    if(o.cagmres_s < 1 || o.cagmres_t < 1 || !valid_qr(o.cagmres_qr_algo)) return PARCEL_ERR_ARG;
    const int qm = o.cagmres_q_mode;
    if(qm != PARCEL_CAGMRES_Q_AUTO && qm != PARCEL_CAGMRES_Q_EXPLICIT && qm != PARCEL_CAGMRES_Q_IMPLICIT)
      return PARCEL_ERR_ARG;
    // IMPLICIT で cagmres_qr_algo が既定値 (MGS) のままなら「未指定」とみなし、ライブラリの既定
    //   (IMPLICIT では QR_CHOLESKYQR) に任せる。C の構造体では未指定と MGS の明示を区別できないため。
    //   CGS は明示なので不可。
    if(qm == PARCEL_CAGMRES_Q_IMPLICIT && o.cagmres_qr_algo == PARCEL_QR_CGS) return PARCEL_ERR_ARG;
    if(o.cagmres_qr_algo == PARCEL_QR_CAGMRES_CHOLESKYQR){
      if(qm != PARCEL_CAGMRES_Q_IMPLICIT) return PARCEL_ERR_ARG;
      if(o.precon_algo != PARCEL_PRECON_NONE) return PARCEL_ERR_ARG;
      if(o.cagmres_s < 2 || (o.cagmres_s % 2) != 0) return PARCEL_ERR_ARG;
    }
    return PARCEL_OK;
  }

  int check_common_options(const parcel_options &o){
    if(!(o.tolerance > 0.0) || o.max_iteration < 1) return PARCEL_ERR_ARG;
    if(!valid_precon(o.precon_algo) || o.jacobi_iter < 1 || o.blockjacobi < 0) return PARCEL_ERR_ARG;
    if(o.scaling_matrix < -1 || o.scaling_matrix > 1) return PARCEL_ERR_ARG;
    if(o.scaling_vector < -1 || o.scaling_vector > 1) return PARCEL_ERR_ARG;
    return PARCEL_OK;
  }

  // SYCL キュー。既定デバイス (ACPP_VISIBILITY_MASK 等で選ぶ)。
  //   parcel_krylov_sample_verify.cpp と同じ作り方にする。
  //   in-order キューを使う (PARCEL はカーネルを event なしで順に投入するため。
  //   Include/parcel.hpp の require_in_order を参照)。
  struct queue_holder {
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  };

  const char *solver_name(int s){
    static const char *names[] = { "pcg", "pbicgstab", "pgmres", "pcagmres",
                                   "power", "inverse_power", "ks_arnoldi" };
    return (s >= 0 && s <= 6) ? names[s] : "?";
  }

  void print_header(parcel::comm &comm, sycl::queue &q, int solver, const parcel_options &o,
                    parcel_int n_global){
    if(o.verbose == 0 || comm.mpi_rank() != 0) return;
    std::cout << "[parcel_c] device = " << q.get_device().get_info<sycl::info::device::name>()
              << " , solver = " << solver_name(solver)
              << " , n_global = " << n_global << " , nprocs = " << comm.mpi_size()
              << " , precon = " << o.precon_algo
              << " , tol = " << o.tolerance << std::endl;
  }

  // ---------------------------------------------------------------------
  int solve_linear_impl(int solver, const parcel_options *opt, PARCEL_C_COMM mpi_comm,
                        parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                        const parcel_int *rowptr, const parcel_int *col, const double *val,
                        const double *b, double *x, parcel_result *res){
    clear_result(res);
    local_crs A;
    int rc = make_local_crs(opt, n_local, nstart, n_global, nnz_local, rowptr, col, val, A);
    if(rc == PARCEL_OK && (b == nullptr || x == nullptr)) rc = PARCEL_ERR_ARG;
    if(rc == PARCEL_OK) rc = check_common_options(*opt);
    if(rc == PARCEL_OK){
      switch(solver){
      case PARCEL_SOLVER_PCG:       if(!PARCEL_ENABLE_PCG)       rc = PARCEL_ERR_NOT_BUILT; break;
      case PARCEL_SOLVER_PBICGSTAB: if(!PARCEL_ENABLE_PBICGSTAB) rc = PARCEL_ERR_NOT_BUILT; break;
      case PARCEL_SOLVER_PGMRES:
        if(!PARCEL_ENABLE_PGMRES) rc = PARCEL_ERR_NOT_BUILT;
        else if(opt->gmres_restart < 1 || !valid_gmres_qr(opt->gmres_qr_algo)) rc = PARCEL_ERR_ARG;
        break;
      case PARCEL_SOLVER_PCAGMRES:
        if(!PARCEL_ENABLE_PCAGMRES) rc = PARCEL_ERR_NOT_BUILT;
        else rc = check_cagmres_options(*opt);
        break;
      default: rc = PARCEL_ERR_ARG; break;
      }
    }
    rc = agree(mpi_comm, rc);
    if(rc != PARCEL_OK) return rc;
    rc = agree(mpi_comm, check_distribution(mpi_comm, A, n_global));
    if(rc != PARCEL_OK) return rc;

    try{
      queue_holder qh;
      sycl::queue &q = qh.q;
#if PARCEL_MPI
      parcel::comm comm(mpi_comm, &q);
#else
      parcel::comm comm(&q);
#endif
      print_header(comm, q, solver, *opt, n_global);

      parcel::matrix<DT> Amat(&q);
      q.wait();
      Amat.set_crsMat<double>(&comm, A.nstart, A.nend, (size_t)A.n, (size_t)A.nnz,
                              A.rowptr.data(), A.col.data(), A.val.data());
      q.wait();

      std::vector<double> xb(x, x + n_local), bb(b, b + n_local);
      parcel::vector<DT> vx(n_local, &q);
      parcel::vector<DT> vb(n_local, &q);
      vx.setValue(xb.data());
      vb.setValue(bb.data());
      q.wait();

      switch(solver){
#if PARCEL_ENABLE_PCG
      case PARCEL_SOLVER_PCG: {
        parcel::krylov::pcg<CT,CPT,CT,CT> s;
        apply_precon(s, *opt);
        s.solver<DT,DT,DT,DT,PMT,PVT>(Amat, vx, vb);
        store_result(s, res);
      } break;
#endif
#if PARCEL_ENABLE_PBICGSTAB
      case PARCEL_SOLVER_PBICGSTAB: {
        parcel::krylov::pbicgstab<CT,CPT,CT,CT> s;
        apply_precon(s, *opt);
        s.solver<DT,DT,DT,DT,PMT,PVT>(Amat, vx, vb);
        store_result(s, res);
      } break;
#endif
#if PARCEL_ENABLE_PGMRES
      case PARCEL_SOLVER_PGMRES: {
        parcel::krylov::pgmres<CT,CPT,CT,CT> s;
        apply_precon(s, *opt);
        s.set_restart((PARCEL_INT)opt->gmres_restart);
        s.set_qr_algo((PARCEL_INT)opt->gmres_qr_algo);
        s.solver<DT,DT,DT,DT,PMT,PVT,DT,DT>(Amat, vx, vb);
        store_result(s, res);
      } break;
#endif
#if PARCEL_ENABLE_PCAGMRES
      case PARCEL_SOLVER_PCAGMRES: {
        parcel::krylov::pcagmres<CT,CPT,CT,CT> s;
        apply_precon(s, *opt);
        s.set_restart((PARCEL_INT)opt->cagmres_s, (PARCEL_INT)opt->cagmres_t);
        // IMPLICIT で MGS (既定値) のときは set_qr_algo を呼ばず、ライブラリが CholeskyQR を選ぶ
        if(!(opt->cagmres_q_mode == PARCEL_CAGMRES_Q_IMPLICIT && opt->cagmres_qr_algo == PARCEL_QR_MGS))
          s.set_qr_algo((PARCEL_INT)opt->cagmres_qr_algo);
        s.set_explicit_q((PARCEL_INT)opt->cagmres_q_mode);
        // 型引数は parcel_krylov_sample_verify.cpp と同じ (IMPLICIT の基底の偶数・奇数番目の精度は既定 = Tbasis)
        s.solver<DT,DT,DT,DT,PMT,PVT,DT,QT,QT>(Amat, vx, vb);
        store_result(s, res);
      } break;
#endif
      default: break;
      }
      q.wait();
      vx.getValue(x);
      q.wait();
    }catch(const std::exception &e){
      // 例外は 1 ランクだけで起きうる (デバイスのメモリ不足など)。
      //   PARCEL_ERR_RUNTIME を返すだけだと、他のランクはソルバ内の集団通信で待ち続けて
      //   ジョブが止まる。メッセージを出してから MPI_Abort で全ランクを終わらせる
      //   (parcel_sycl_c.h の PARCEL_ERR_RUNTIME の説明を参照)。
      std::cerr << "[parcel_c] exception: " << e.what() << std::endl;
#if PARCEL_MPI
      MPI_Abort(mpi_comm, -PARCEL_ERR_RUNTIME);
#endif
      return PARCEL_ERR_RUNTIME;   // MPI_Abort は戻らない (念のため)。MPI なしではここで返す
    }
    return PARCEL_OK;
  }

  // ---------------------------------------------------------------------
  int solve_eigen_impl(int solver, const parcel_options *opt, PARCEL_C_COMM mpi_comm,
                       parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                       const parcel_int *rowptr, const parcel_int *col, const double *val,
                       double *eigval, double *eigvec, parcel_result *res){
    clear_result(res);
    local_crs A;
    int rc = make_local_crs(opt, n_local, nstart, n_global, nnz_local, rowptr, col, val, A);
    if(rc == PARCEL_OK && (eigval == nullptr || eigvec == nullptr)) rc = PARCEL_ERR_ARG;
    if(rc == PARCEL_OK) rc = check_common_options(*opt);
    if(rc == PARCEL_OK){
      const parcel_options &o = *opt;
      // linsolve_tolerance / _max_iteration / _restart は 0 以下ならソルバの既定 (NaN だけ不正)
      if(!valid_precon(o.linsolve_precon_algo) || o.linsolve_jacobi_iter < 1 ||
         o.linsolve_blockjacobi < 0 || std::isnan(o.linsolve_tolerance) ||
         (o.linsolve_algo != PARCEL_LINSOLVE_PGMRES && o.linsolve_algo != PARCEL_LINSOLVE_PCG))
        rc = PARCEL_ERR_ARG;
      switch(solver){
      case PARCEL_SOLVER_POWER:
        if(!PARCEL_ENABLE_POWER) rc = PARCEL_ERR_NOT_BUILT;
        else if(o.shift_enable) rc = PARCEL_ERR_ARG;          // shift は ks_arnoldi のみ
        break;
      case PARCEL_SOLVER_INVERSE_POWER:
        if(!PARCEL_ENABLE_INVERSE_POWER) rc = PARCEL_ERR_NOT_BUILT;
        else if(o.shift_enable) rc = PARCEL_ERR_ARG;
        // A K^-1 モードの内部ソルバは pgmres 固定 (pcg は A K^-1 に使えない)
        else if(o.precon_spectrum && o.linsolve_algo == PARCEL_LINSOLVE_PCG) rc = PARCEL_ERR_ARG;
        break;
      case PARCEL_SOLVER_KS_ARNOLDI:
        if(!PARCEL_ENABLE_KS_ARNOLDI) rc = PARCEL_ERR_NOT_BUILT;
        else if(o.ks_nev < 1 || o.ks_k_keep < o.ks_nev || o.ks_m <= o.ks_k_keep ||
                (parcel_int)o.ks_m > n_global || o.ks_max_restart < 1 ||
                !valid_gmres_qr(o.ks_qr_algo)) rc = PARCEL_ERR_ARG;
        // (A - σI) は対角の無い行に要素を足すので非零数が最大 n_local 増える。
        //   32 ビット版ではそれも PARCEL_INT に収まる必要がある (確保の前に調べる)
        else if(o.shift_enable && !o.precon_spectrum && !fits_internal(nnz_local + n_local)) rc = PARCEL_ERR_ARG;
        break;
      default: rc = PARCEL_ERR_ARG; break;
      }
    }
    rc = agree(mpi_comm, rc);
    if(rc != PARCEL_OK) return rc;
    rc = agree(mpi_comm, check_distribution(mpi_comm, A, n_global));
    if(rc != PARCEL_OK) return rc;

    const parcel_options &o = *opt;
    const bool is_ks = (solver == PARCEL_SOLVER_KS_ARNOLDI);
    const int  k     = parcel_eigen_count(solver, opt);

    try{
      queue_holder qh;
      sycl::queue &q = qh.q;
#if PARCEL_MPI
      parcel::comm comm(mpi_comm, &q);
#else
      parcel::comm comm(&q);
#endif
      print_header(comm, q, solver, o, n_global);

      // A の shift-invert (precon_spectrum=0) は (A - σI) を別の行列として渡す。
      //   set_crsMat() が A の列番号・値を書き換える前に作っておく。
      const bool need_shiftmat = is_ks && o.shift_enable && !o.precon_spectrum;
      local_crs S;
      if(need_shiftmat) make_shifted(A, o.shift_sigma, S);

      parcel::matrix<DT> Amat(&q);
      q.wait();
      Amat.set_crsMat<double>(&comm, A.nstart, A.nend, (size_t)A.n, (size_t)A.nnz,
                              A.rowptr.data(), A.col.data(), A.val.data());
      q.wait();
      parcel::matrix<DT> Smat(&q);
      if(need_shiftmat){
        Smat.set_crsMat<double>(&comm, S.nstart, S.nend, (size_t)S.n, (size_t)S.nnz,
                                S.rowptr.data(), S.col.data(), S.val.data());
        q.wait();
      }

      // 固有値: power / inverse_power は 1 要素 (実数)、ks_arnoldi は 2*k_keep (Re,Im 交互)
      // 固有ベクトル: n_local * k
      parcel::vector<DT> vval(is_ks ? 2 * k : 1, &q);
      parcel::vector<DT> vvec((size_t)n_local * k, &q);
      {
        std::vector<double> tmp(is_ks ? 2 * k : 1, 0.0);
        vval.setValue(tmp.data());
        std::vector<double> v0((size_t)n_local * k, 0.0);
        if(is_ks){
          if(o.eigvec_seed){
            for(parcel_int i = 0; i < n_local; i++) v0[i] = eigvec[i];
          }else{
            // 大域行番号で決まる初期ベクトル (分割のしかたに依らない)。
            //   power / inverse_power の内部 seed と同じ式 (parcel::krylov::default_seed_value)。
            //   内部 seed も先頭行を MPI_Exscan で求めるので、不均等分割でも同じ値になる。
            //   全要素 1 は特定の固有ベクトルと直交しうるので避ける。
            for(parcel_int i = 0; i < n_local; i++){
              const double g = (double)(A.nstart + i);
              v0[i] = 1.0 + 0.5*std::sin(0.317*g + 0.7) + 0.3*std::cos(0.181*g - 1.3);
            }
          }
        }
        vvec.setValue(v0.data());
      }
      q.wait();

      switch(solver){
#if PARCEL_ENABLE_POWER
      case PARCEL_SOLVER_POWER: {
        parcel::krylov::power<CT,CPT,CT,CT> s;
        apply_eigen(s, o);
        s.solver<DT,DT,DT,DT,PMT,PVT>(Amat, vval, vvec);
        store_result(s, res);
      } break;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
      case PARCEL_SOLVER_INVERSE_POWER: {
        parcel::krylov::inverse_power<CT,CPT,CT,CT> s;
        apply_eigen(s, o);
        s.solver<DT,DT,DT,DT,PMT,PVT>(Amat, vval, vvec);
        store_result(s, res);
        if(res) res->linsolve_failures = (int)s.get_linsolve_failures();
      } break;
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
      case PARCEL_SOLVER_KS_ARNOLDI: {
        parcel::krylov::ks_arnoldi<CT,CPT,CT,CT> s;
        apply_eigen(s, o);
        s.set_ks_subspace((PARCEL_INT)o.ks_m, (PARCEL_INT)o.ks_k_keep, (PARCEL_INT)o.ks_nev);
        s.set_ks_max_restart((PARCEL_INT)o.ks_max_restart);
        s.set_qr_algo((PARCEL_INT)o.ks_qr_algo);   // CGS (CGS2) / MGS (MGS2)。値は上で検査済み
        if(o.shift_enable){
          s.set_shift(o.shift_sigma);
          // precon_spectrum=1 では ShiftMat は不要 (内部 pgmres のシフトで表す)
          s.solver<DT,DT,DT,PMT,PVT>(Amat, vval, vvec, need_shiftmat ? &Smat : nullptr);
        }else{
          s.disable_shift();
          s.solver<DT,DT,DT,PMT,PVT>(Amat, vval, vvec);
        }
        store_result(s, res);
        if(res){
          res->ks_nconv          = (int)s.get_ks_nconv();
          res->linsolve_failures = (int)s.get_linsolve_failures();
        }
      } break;
#endif
      default: break;
      }
      q.wait();

      vvec.getValue(eigvec);
      if(is_ks){
        vval.getValue(eigval);                 // 2*k_keep (Re, Im 交互)
      }else{
        double lam = 0.0;
        vval.getValue(&lam);
        eigval[0] = lam; eigval[1] = 0.0;
      }
      q.wait();
    }catch(const std::exception &e){
      // 例外は 1 ランクだけで起きうる (デバイスのメモリ不足など)。
      //   PARCEL_ERR_RUNTIME を返すだけだと、他のランクはソルバ内の集団通信で待ち続けて
      //   ジョブが止まる。メッセージを出してから MPI_Abort で全ランクを終わらせる
      //   (parcel_sycl_c.h の PARCEL_ERR_RUNTIME の説明を参照)。
      std::cerr << "[parcel_c] exception: " << e.what() << std::endl;
#if PARCEL_MPI
      MPI_Abort(mpi_comm, -PARCEL_ERR_RUNTIME);
#endif
      return PARCEL_ERR_RUNTIME;   // MPI_Abort は戻らない (念のため)。MPI なしではここで返す
    }
    return PARCEL_OK;
  }

} // namespace

// =========================================================================
// C API
// =========================================================================
extern "C" {

  // 既定値は C++ ライブラリのオブジェクトを作って読む (値をここに二重に持たない)。
  //   ビルドから外したソルバの項目は使われないので 0 のまま。
  //   linsolve_tolerance / _max_iteration / _restart は 0 (= ソルバの既定) にする。
  //   inverse_power (1e-10 / 5000 / 30) と ks_arnoldi (1e-10 / 2000 / 30) で既定が違い、
  //   1 つの項目では両方を表せないため (apply_eigen が 0 以下なら上書きしない)。
  void parcel_options_init(parcel_options *o){
    if(o == nullptr) return;
    std::memset(o, 0, sizeof(*o));
    parcel::krylov::krylov_param kp;             // tolerance / max_iteration / 前処理の既定
    o->tolerance              = kp.get_tolerance();
    o->linsolve_tolerance     = 0.0;
    o->shift_sigma            = 0.0;
    o->max_iteration          = (int)kp.get_max_iteration();
    o->precon_algo            = (int)kp.get_precon_algo();
    o->jacobi_iter            = (int)kp.get_jacobi();
    o->blockjacobi            = 0;               // 0 = 自動 (デバイスの並列度から決める)
    o->scaling_matrix         = -1;              // -1 = 自動 (前処理の種類と精度から決める)
    o->scaling_vector         = -1;
    o->index_base             = 0;
    o->verbose                = 0;
#if PARCEL_ENABLE_PGMRES
    {
      parcel::krylov::pgmres<CT,CPT,CT,CT> g;
      o->gmres_restart        = (int)g.get_restart();
      o->gmres_qr_algo        = (int)g.get_qr_algo();
    }
#endif
#if PARCEL_ENABLE_PCAGMRES
    {
      parcel::krylov::pcagmres<CT,CPT,CT,CT> c;
      o->cagmres_s            = (int)c.get_restart_s();
      o->cagmres_t            = (int)c.get_restart_t();
      o->cagmres_qr_algo      = (int)c.get_qr_algo();
      o->cagmres_q_mode       = (int)c.get_explicit_q();
    }
#endif
#if PARCEL_ENABLE_INVERSE_POWER
    {
      parcel::krylov::inverse_power<CT,CPT,CT,CT> ip;
      o->precon_spectrum      = ip.get_precon_spectrum() ? 1 : 0;
      o->linsolve_algo        = (int)ip.get_linsolve_algo();
      o->linsolve_precon_algo = (int)ip.get_linsolve_precon_param()->get_precon_algo();
      o->linsolve_jacobi_iter = (int)ip.get_linsolve_precon_param()->get_jacobi();
    }
#else
    o->precon_spectrum        = 0;
    o->linsolve_algo          = PARCEL_LINSOLVE_PGMRES;
    {
      // 内部ソルバの前処理の既定は eigen_param が持つ (ヤコビ)。
      //   外側の前処理 (kp) の既定とは別なので、eigen_param から読む。
      parcel::krylov::eigen_param ep(1.0e-9, 1, 1);
      o->linsolve_precon_algo = (int)ep.get_linsolve_precon_param()->get_precon_algo();
      o->linsolve_jacobi_iter = (int)ep.get_linsolve_precon_param()->get_jacobi();
    }
#endif
    o->linsolve_max_iteration = 0;
    o->linsolve_restart       = 0;
    o->linsolve_blockjacobi   = 0;
#if PARCEL_ENABLE_KS_ARNOLDI
    {
      parcel::krylov::ks_arnoldi<CT,CPT,CT,CT> ks;
      o->ks_m                 = (int)ks.get_ks_m();
      o->ks_k_keep            = (int)ks.get_ks_k_keep();
      o->ks_nev               = (int)ks.get_ks_nev();
      o->ks_max_restart       = (int)ks.get_ks_max_restart();
      o->ks_qr_algo           = (int)ks.get_qr_algo();   // ライブラリの既定 (ビルド時の PARCEL_KS_ORTHO_DEFAULT)
    }
#else
    o->ks_qr_algo             = PARCEL_QR_CGS;
#endif
    o->shift_enable           = 0;
    o->eigvec_seed            = 0;
  }

  int parcel_options_size(void){ return (int)sizeof(parcel_options); }
  int parcel_result_size(void){ return (int)sizeof(parcel_result); }

  // 版 (Include/parcel_version.hpp の値をそのまま返す)
  int parcel_get_version(int *major, int *minor, int *patch){
    parcel::get_version(major, minor, patch);
    return PARCEL_OK;
  }
  const char *parcel_version_string(void){ return parcel::version_string(); }

  int parcel_solver_from_name(const char *name){
    if(name == nullptr) return -1;
    for(int s = 0; s <= 6; s++) if(std::strcmp(name, solver_name(s)) == 0) return s;
    return -1;
  }

  int parcel_eigen_count(int solver, const parcel_options *opt){
    switch(solver){
    case PARCEL_SOLVER_POWER:
    case PARCEL_SOLVER_INVERSE_POWER: return 1;
    case PARCEL_SOLVER_KS_ARNOLDI:    return opt ? opt->ks_k_keep : 0;
    default:                          return 0;
    }
  }

  int parcel_solve_linear(int solver, const parcel_options *opt, PARCEL_C_COMM comm,
                          parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                          const parcel_int *rowptr, const parcel_int *col, const double *val,
                          const double *b, double *x, parcel_result *res){
    return solve_linear_impl(solver, opt, comm, n_local, nstart, n_global, nnz_local,
                             rowptr, col, val, b, x, res);
  }

  int parcel_solve_eigen(int solver, const parcel_options *opt, PARCEL_C_COMM comm,
                         parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                         const parcel_int *rowptr, const parcel_int *col, const double *val,
                         double *eigval, double *eigvec, parcel_result *res){
    return solve_eigen_impl(solver, opt, comm, n_local, nstart, n_global, nnz_local,
                            rowptr, col, val, eigval, eigvec, res);
  }

  int parcel_solve_linear_f(int solver, const parcel_options *opt, PARCEL_C_FCOMM comm,
                            parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                            const parcel_int *rowptr, const parcel_int *col, const double *val,
                            const double *b, double *x, parcel_result *res){
#if PARCEL_MPI
    return solve_linear_impl(solver, opt, MPI_Comm_f2c(comm), n_local, nstart, n_global,
                             nnz_local, rowptr, col, val, b, x, res);
#else
    return solve_linear_impl(solver, opt, comm, n_local, nstart, n_global,
                             nnz_local, rowptr, col, val, b, x, res);
#endif
  }

  int parcel_solve_eigen_f(int solver, const parcel_options *opt, PARCEL_C_FCOMM comm,
                           parcel_int n_local, parcel_int nstart, parcel_int n_global, parcel_int nnz_local,
                           const parcel_int *rowptr, const parcel_int *col, const double *val,
                           double *eigval, double *eigvec, parcel_result *res){
#if PARCEL_MPI
    return solve_eigen_impl(solver, opt, MPI_Comm_f2c(comm), n_local, nstart, n_global,
                            nnz_local, rowptr, col, val, eigval, eigvec, res);
#else
    return solve_eigen_impl(solver, opt, comm, n_local, nstart, n_global,
                            nnz_local, rowptr, col, val, eigval, eigvec, res);
#endif
  }

} // extern "C"

// =========================================================================
// 旧 API (互換)。pcg、設定はコンパイル時マクロで決まる。
// =========================================================================
//   n / nnz / crsCol / crsRow_ptr は parcel_int (int64_t。新 API と同じ型)。
static void run_parcel_sycl_main(PARCEL_C_COMM mpi_comm_c, parcel_int n, parcel_int nnz,
                                 double *crsA, parcel_int *crsCol, parcel_int *crsRow_ptr,
                                 double *vecx, double *vecb){
  // 担当範囲は n の累積和で求める。全ランクで n が等しければ nstart = n*myrank になり、不均等も扱える。
  int myrank = 0;
#if PARCEL_MPI
  MPI_Comm_rank(mpi_comm_c, &myrank);
  int64_t n64 = n, nstart = 0, n_global = 0;
  MPI_Exscan(&n64, &nstart, 1, MPI_INT64_T, MPI_SUM, mpi_comm_c);
  if(myrank == 0) nstart = 0;                    // Exscan は rank 0 の値を定めない
  MPI_Allreduce(&n64, &n_global, 1, MPI_INT64_T, MPI_SUM, mpi_comm_c);
#else
  int64_t nstart = 0, n_global = n;              // MPI なし: 1 プロセスが全体を持つ
#endif

  parcel_options o;
  parcel_options_init(&o);
  o.verbose       = 1;                           // rank 0 がデバイス名を出す
  o.tolerance     = 1.0e-9;
  o.max_iteration = MAX_ITERATION;
  o.precon_algo   = parcel_solver_precon_algo;
  switch(o.precon_algo){
  case PARCEL_PRECON_BLOCKJACOBI:
    o.scaling_matrix = 1; o.scaling_vector = 1; o.blockjacobi = BLOCKJACOBI_BLOCK; break;
  case PARCEL_PRECON_JACOBI:
    o.scaling_matrix = 1; o.scaling_vector = 1; o.jacobi_iter = JACOBI_ITER; break;
  default: break;
  }
  const int rc = parcel_solve_linear(PARCEL_SOLVER_PCG, &o, mpi_comm_c, n, nstart, n_global, nnz,
                                     crsRow_ptr, crsCol, crsA, vecb, vecx, nullptr);
  if(rc != PARCEL_OK && myrank == 0){
    std::cerr << "[run_parcel_sycl] parcel_solve_linear が失敗 (rc = " << rc << ")" << std::endl;
  }
}

extern "C" {
  void run_parcel_sycl_f(int *mpi_comm_f, parcel_int n, parcel_int nnz,
                         double *crsA, parcel_int *crsCol, parcel_int *crsRow_ptr,
                         double *vecx, double *vecb){
#if PARCEL_MPI
    run_parcel_sycl_main(MPI_Comm_f2c(mpi_comm_f[0]), n, nnz, crsA, crsCol, crsRow_ptr, vecx, vecb);
#else
    (void)mpi_comm_f;
    run_parcel_sycl_main(0, n, nnz, crsA, crsCol, crsRow_ptr, vecx, vecb);
#endif
  }
  void run_parcel_sycl_c(PARCEL_C_COMM mpi_comm_c, parcel_int n, parcel_int nnz,
                         double *crsA, parcel_int *crsCol, parcel_int *crsRow_ptr,
                         double *vecx, double *vecb){
    run_parcel_sycl_main(mpi_comm_c, n, nnz, crsA, crsCol, crsRow_ptr, vecx, vecb);
  }
}
