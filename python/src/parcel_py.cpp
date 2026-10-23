// =========================================================================
// parcel_py.cpp
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// Python パッケージ parcel のネイティブ層 (libparcel_py.so)。宣言は parcel_py.h。
// PARCEL の C++ ライブラリ (Include/) を直接呼ぶ。C / Fortran 向けの API
// (wrapper_parcel_sycl.cpp) には依存しない。
//
// 方針
//   - 1 回の呼び出しで, キューの作成・行列の登録・解法・結果の取り出しまでを行う。
//   - 精度は倍精度だけ。
//   - 引数の誤りは, 全ランクで判定をそろえて同じ戻り値にする (Python 側で全ランクが同じ例外を投げられる)。
//     PARCEL が全ランクで同じく投げる std::invalid_argument / std::overflow_error も戻り値に変える。
//   - それ以外の例外 (デバイスのメモリ不足など) は一部のランクだけで起きうる。戻ると他のランクが
//     集団通信で待ち続けるので, メッセージを出して MPI_Abort する。
//   - PARCEL のソルバは rank 0 に経過を表示する。verbose = 0 のときは標準出力を一時的に捨てる。
//   - -DPARCEL_MPI=0 で MPI なしの 1 プロセス版になる (mpi.h を読まず, MPI のライブラリもリンクしない)。
//     関数と引数は同じで, コミュニケータの引数は使わない (ランク 0, プロセス数 1)。
//     例外は他のランクを待たせないので, MPI_Abort せずに PARCEL_PY_ERR_RUNTIME を返す。
// =========================================================================
#include "parcel_sycl.hpp"     // -I<PARCEL の Include/> で探す (PARCEL_MPI の既定 1 もここで決まる)
#if PARCEL_MPI
#include <mpi.h>
#endif
#include "parcel_py.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

  using T = parcel_double;

#if PARCEL_MPI
  bool g_we_initialized_mpi = false;

  using comm_t = MPI_Comm;
  MPI_Comm to_comm(const void *p){
    return p ? *static_cast<const MPI_Comm *>(p) : MPI_COMM_WORLD;
  }
#else
  using comm_t = int;                       // MPI なし: コミュニケータは使わない
  int to_comm(const void *){ return 0; }
#endif

  void set_err(char *buf, int len, const std::string &msg){
    if(buf == nullptr || len <= 0) return;
    std::snprintf(buf, (size_t)len, "%s", msg.c_str());
  }

  // 全ランクの戻り値をそろえる (最も悪い値 = 最小値)
  int agree(comm_t comm, int rc){
#if PARCEL_MPI
    int g = rc;
    MPI_Allreduce(&rc, &g, 1, MPI_INT, MPI_MIN, comm);
    return g;
#else
    (void)comm;
    return rc;
#endif
  }

  // 標準出力を一時的に捨てる (verbose = 0 のとき)
  struct stdout_silencer {
    int saved = -1;
    explicit stdout_silencer(bool on){
      if(!on) return;
      std::cout.flush(); std::fflush(stdout);
      saved = ::dup(1);
      const int devnull = ::open("/dev/null", O_WRONLY);
      if(saved >= 0 && devnull >= 0){ ::dup2(devnull, 1); }
      if(devnull >= 0) ::close(devnull);
    }
    ~stdout_silencer(){
      if(saved < 0) return;
      std::cout.flush(); std::fflush(stdout);
      ::dup2(saved, 1); ::close(saved);
    }
  };

  // ---------------------------------------------------------------------
  // 担当行の CRS を PARCEL に渡せる形に複製する (set_crsMat() は列番号と値を書き換えるので)
  struct local_crs {
    PARCEL_INT nstart = 0, nend = -1, n = 0, nnz = 0;
    std::vector<PARCEL_INT> rowptr, col;
    std::vector<double>     val;
  };

  bool fits(int64_t v){
    return v >= (int64_t)std::numeric_limits<PARCEL_INT>::min() &&
           v <= (int64_t)std::numeric_limits<PARCEL_INT>::max();
  }

  // このランクだけの検査 (呼び出し側で agree する)
  int make_local_crs(int64_t n_local, int64_t row_start, int64_t n_global, int64_t nnz_local,
                     const int64_t *indptr, const int64_t *indices, const double *data,
                     local_crs &A, std::string &msg){
    if(n_local < 1){ msg = "each rank must own at least one row (n_local >= 1)"; return PARCEL_PY_ERR_ARG; }
    if(n_global < 1 || nnz_local < 0){ msg = "n_global must be >= 1 and nnz must be >= 0"; return PARCEL_PY_ERR_ARG; }
    if(indptr == nullptr || (nnz_local > 0 && (indices == nullptr || data == nullptr))){
      msg = "null array"; return PARCEL_PY_ERR_ARG;
    }
    if(row_start < 0 || n_local > n_global || row_start > n_global - n_local){
      msg = "row block [row_start, row_start + n_local) is outside [0, n_global)"; return PARCEL_PY_ERR_ARG;
    }
    if(!fits(n_global) || !fits(nnz_local)){
      msg = "matrix too large for PARCEL_INT of this build"; return PARCEL_PY_ERR_OVERFLOW;
    }
    if(indptr[0] != 0 || indptr[n_local] != nnz_local){
      msg = "indptr[0] must be 0 and indptr[n_local] must equal nnz"; return PARCEL_PY_ERR_ARG;
    }
    for(int64_t i = 1; i <= n_local; i++){
      if(indptr[i] < indptr[i-1]){ msg = "indptr must be non-decreasing"; return PARCEL_PY_ERR_ARG; }
    }
    A.nstart = (PARCEL_INT)row_start;
    A.n      = (PARCEL_INT)n_local;
    A.nend   = (PARCEL_INT)(row_start + n_local - 1);
    A.nnz    = (PARCEL_INT)nnz_local;
    A.rowptr.assign(indptr, indptr + n_local + 1);
    A.col.resize((size_t)nnz_local);
    A.val.assign(data, data + nnz_local);
    for(int64_t k = 0; k < nnz_local; k++){
      if(indices[k] < 0 || indices[k] >= n_global){
        msg = "column index out of [0, n_global)"; return PARCEL_PY_ERR_ARG;
      }
      A.col[k] = (PARCEL_INT)indices[k];
    }
    return PARCEL_PY_OK;
  }

  // (A - σI): 対角要素の無い行には -σ を足す
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
  // 設定: 未指定 (負 / NaN) の項目は設定関数を呼ばない
  void apply_common(parcel::krylov::krylov_param &kp, const parcel_py_options &o){
    if(!std::isnan(o.tol))     kp.set_tolerance(o.tol);
    if(o.maxiter > 0)          kp.set_max_iteration((PARCEL_INT)o.maxiter);
    if(o.precond >= 0)         kp.set_precon_algo((PARCEL_INT)o.precond);
    if(o.jacobi_iter > 0)      kp.set_jacobi((PARCEL_INT)o.jacobi_iter);
    if(o.blocks > 0)           kp.set_blockjacobi((PARCEL_INT)o.blocks);
    if(o.scale_matrix >= 0)    kp.set_scaling_matrix(o.scale_matrix != 0);
    if(o.scale_vector >= 0)    kp.set_scaling_vector(o.scale_vector != 0);
  }

  void apply_eigen(parcel::krylov::eigen_param &ep, const parcel_py_options &o){
    apply_common(ep, o);
    ep.set_precon_spectrum(o.precond_spectrum != 0);
    if(o.inner_method >= 0)
      ep.set_linsolve_algo(o.inner_method == 1 ? parcel::krylov::LINSOLVE_PCG : parcel::krylov::LINSOLVE_PGMRES);
    if(!std::isnan(o.inner_tol)) ep.set_linsolve_tolerance(o.inner_tol);
    if(o.inner_maxiter > 0)      ep.set_linsolve_max_iteration((PARCEL_INT)o.inner_maxiter);
    if(o.inner_restart > 0)      ep.set_linsolve_restart((PARCEL_INT)o.inner_restart);
    if(o.inner_precond >= 0)     ep.set_linsolve_precon_algo((PARCEL_INT)o.inner_precond);
    if(o.inner_jacobi_iter > 0)  ep.set_linsolve_jacobi((PARCEL_INT)o.inner_jacobi_iter);
    if(o.inner_blocks > 0)       ep.set_linsolve_blockjacobi((PARCEL_INT)o.inner_blocks);
  }

  void store(const parcel::krylov::krylov_param &kp, parcel_py_info *info){
    info->reason        = (int32_t)kp.get_converged_reason();
    info->iterations    = (int64_t)kp.get_iterations();
    info->residual      = kp.get_residual();
    info->true_residual = kp.get_true_residual();
  }

  void clear_info(parcel_py_info *info){
    if(!info) return;
    info->reason = 0; info->iterations = 0;
    info->residual = info->true_residual = std::numeric_limits<double>::quiet_NaN();
    info->ks_nconv = 0; info->linsolve_failures = 0; info->n_eig = 0;
  }

  bool is_linear(int s){ return s == PARCEL_PY_PCG || s == PARCEL_PY_PBICGSTAB ||
                                s == PARCEL_PY_PGMRES || s == PARCEL_PY_PCAGMRES; }
  bool is_eigen(int s){ return s == PARCEL_PY_POWER || s == PARCEL_PY_INVERSE_POWER ||
                               s == PARCEL_PY_KS_ARNOLDI; }

  bool built(int s){
    switch(s){
    case PARCEL_PY_PCG:            return PARCEL_ENABLE_PCG;
    case PARCEL_PY_PBICGSTAB:      return PARCEL_ENABLE_PBICGSTAB;
    case PARCEL_PY_PGMRES:         return PARCEL_ENABLE_PGMRES;
    case PARCEL_PY_PCAGMRES:       return PARCEL_ENABLE_PCAGMRES;
    case PARCEL_PY_POWER:          return PARCEL_ENABLE_POWER;
    case PARCEL_PY_INVERSE_POWER:  return PARCEL_ENABLE_INVERSE_POWER;
    case PARCEL_PY_KS_ARNOLDI:     return PARCEL_ENABLE_KS_ARNOLDI;
    default:                       return false;
    }
  }

  // 前処理の段階の検査 (このランクだけ。行列に依らないので全ランクで同じ結果になる)
  int check_options(const parcel_py_options *o, std::string &msg){
    if(o == nullptr){ msg = "options is null"; return PARCEL_PY_ERR_ARG; }
    if(o->abi_version != PARCEL_PY_ABI_VERSION){
      msg = "option structure version mismatch (rebuild the native library)"; return PARCEL_PY_ERR_ARG;
    }
    if(!is_linear(o->solver) && !is_eigen(o->solver)){ msg = "unknown solver"; return PARCEL_PY_ERR_ARG; }
    if(!built(o->solver)){ msg = "this solver was excluded from the build"; return PARCEL_PY_ERR_NOT_BUILT; }
    if((o->s > 0) != (o->t > 0)){ msg = "s and t must be given together"; return PARCEL_PY_ERR_ARG; }
    const bool any_ks = o->ks_m > 0 || o->ks_k_keep > 0 || o->ks_nev > 0;
    const bool all_ks = o->ks_m > 0 && o->ks_k_keep > 0 && o->ks_nev > 0;
    if(any_ks && !all_ks){ msg = "m, k_keep and nev must be given together"; return PARCEL_PY_ERR_ARG; }
    if(o->use_shift && o->solver != PARCEL_PY_KS_ARNOLDI){
      msg = "sigma (shift-invert) is available only for ks_arnoldi"; return PARCEL_PY_ERR_ARG;
    }
    return PARCEL_PY_OK;
  }

  // 例外の扱い: 全ランクで同じく投げるもの (引数の検査) は戻り値に, それ以外は MPI_Abort
  template<class F>
  int guarded(comm_t comm, char *errbuf, int errlen, F body){
    try{
      body();
    }catch(const std::invalid_argument &e){
      set_err(errbuf, errlen, e.what());
      return PARCEL_PY_ERR_ARG;
    }catch(const std::overflow_error &e){
      set_err(errbuf, errlen, e.what());
      return PARCEL_PY_ERR_OVERFLOW;
    }catch(const std::exception &e){
#if PARCEL_MPI
      std::cerr << "[parcel] fatal error on one rank: " << e.what()
                << " ; aborting all ranks (other ranks would wait forever)" << std::endl;
      MPI_Abort(comm, 1);
#else
      (void)comm;
      set_err(errbuf, errlen, e.what());
      return PARCEL_PY_ERR_RUNTIME;
#endif
    }
    return PARCEL_PY_OK;
  }

} // namespace

// =========================================================================
extern "C" {

int parcel_py_abi_version(void){ return PARCEL_PY_ABI_VERSION; }

const char *parcel_py_version_string(void){ return parcel::version_string(); }

int parcel_py_has_mpi(void){ return PARCEL_MPI ? 1 : 0; }

#if PARCEL_MPI
int parcel_py_mpi_init(void){
  int flag = 0;
  MPI_Initialized(&flag);
  if(flag) return 0;
  int provided = 0;
  if(MPI_Init_thread(nullptr, nullptr, MPI_THREAD_FUNNELED, &provided) != MPI_SUCCESS) return PARCEL_PY_ERR_MPI;
  g_we_initialized_mpi = true;
  return 1;
}

void parcel_py_mpi_finalize(void){
  int init = 0, fin = 0;
  MPI_Initialized(&init); MPI_Finalized(&fin);
  if(g_we_initialized_mpi && init && !fin) MPI_Finalize();
}

int parcel_py_comm_rank(const void *comm){
  int r = -1; if(MPI_Comm_rank(to_comm(comm), &r) != MPI_SUCCESS) return PARCEL_PY_ERR_MPI; return r;
}
int parcel_py_comm_size(const void *comm){
  int s = -1; if(MPI_Comm_size(to_comm(comm), &s) != MPI_SUCCESS) return PARCEL_PY_ERR_MPI; return s;
}

double parcel_py_allreduce(const void *comm, double v, int op){
  double g = v;
  MPI_Op mop = (op == 1) ? MPI_MAX : (op == 2) ? MPI_MIN : MPI_SUM;
  MPI_Allreduce(&v, &g, 1, MPI_DOUBLE, mop, to_comm(comm));
  return g;
}
#else
// MPI なし: 1 プロセス
int parcel_py_mpi_init(void){ return 0; }
void parcel_py_mpi_finalize(void){}
int parcel_py_comm_rank(const void *){ return 0; }
int parcel_py_comm_size(const void *){ return 1; }
double parcel_py_allreduce(const void *, double v, int){ return v; }
#endif

int64_t parcel_py_eig_count(const parcel_py_options *opt){
  if(opt == nullptr || !is_eigen(opt->solver)) return PARCEL_PY_ERR_ARG;
  if(opt->solver != PARCEL_PY_KS_ARNOLDI) return 1;
  if(opt->ks_k_keep > 0) return opt->ks_k_keep;
#if PARCEL_ENABLE_KS_ARNOLDI
  parcel::krylov::ks_arnoldi<T,T,T,T> ks;
  return (int64_t)ks.get_ks_k_keep();
#else
  return PARCEL_PY_ERR_NOT_BUILT;
#endif
}

int parcel_py_build_info(char *buf, int len){
  std::string s = std::string("PARCEL_INT_BITS=") + std::to_string(PARCEL_INT_BITS) +
                  " PARCEL_LINT_BITS=" + std::to_string(PARCEL_LINT_BITS) +
                  " precision=double mpi=" + std::to_string(PARCEL_MPI ? 1 : 0) +
                  " abi=" + std::to_string(PARCEL_PY_ABI_VERSION);
  set_err(buf, len, s);
  return PARCEL_PY_OK;
}

int parcel_py_device_name(char *buf, int len){
  try{
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
    set_err(buf, len, q.get_device().get_info<sycl::info::device::name>());
  }catch(const std::exception &e){
    set_err(buf, len, std::string("unknown (") + e.what() + ")");
    return PARCEL_PY_ERR_ARG;
  }
  return PARCEL_PY_OK;
}

// -------------------------------------------------------------------------
int parcel_py_solve(const parcel_py_options *opt, const void *commp,
                    int64_t n_local, int64_t row_start, int64_t n_global, int64_t nnz_local,
                    const int64_t *indptr, const int64_t *indices, const double *data,
                    const double *b, double *x, parcel_py_info *info,
                    char *errbuf, int errlen){
  const comm_t mcomm = to_comm(commp);
  clear_info(info);
  std::string msg;
  int rc = check_options(opt, msg);
  if(rc == PARCEL_PY_OK && !is_linear(opt->solver)){ msg = "not a linear solver"; rc = PARCEL_PY_ERR_ARG; }
  local_crs A;
  if(rc == PARCEL_PY_OK) rc = make_local_crs(n_local, row_start, n_global, nnz_local, indptr, indices, data, A, msg);
  if(rc == PARCEL_PY_OK && (b == nullptr || x == nullptr)){ msg = "null vector"; rc = PARCEL_PY_ERR_ARG; }
  const int g = agree(mcomm, rc);
  if(g != PARCEL_PY_OK){
    set_err(errbuf, errlen, rc != PARCEL_PY_OK ? msg : std::string("invalid input on another rank"));
    return g;
  }

  const parcel_py_options &o = *opt;
  rc = guarded(mcomm, errbuf, errlen, [&](){
    stdout_silencer quiet(o.verbose == 0);
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
#if PARCEL_MPI
    parcel::comm comm(mcomm, &q);
#else
    parcel::comm comm(&q);
#endif
    parcel::matrix<T> Amat(&q);
    Amat.set_crsMat<double>(&comm, A.nstart, A.nend, (size_t)A.n, (size_t)A.nnz,
                            A.rowptr.data(), A.col.data(), A.val.data());
    q.wait();
    parcel::vector<T> vx((size_t)n_local, &q), vb((size_t)n_local, &q);
    vx.setValue(x); vb.setValue(b);
    q.wait();
    switch(o.solver){
#if PARCEL_ENABLE_PCG
    case PARCEL_PY_PCG: {
      parcel::krylov::pcg<T,T,T,T> s; apply_common(s, o);
      s.solver(Amat, vx, vb); store(s, info);
    } break;
#endif
#if PARCEL_ENABLE_PBICGSTAB
    case PARCEL_PY_PBICGSTAB: {
      parcel::krylov::pbicgstab<T,T,T,T> s; apply_common(s, o);
      s.solver(Amat, vx, vb); store(s, info);
    } break;
#endif
#if PARCEL_ENABLE_PGMRES
    case PARCEL_PY_PGMRES: {
      parcel::krylov::pgmres<T,T,T,T> s; apply_common(s, o);
      if(o.restart > 0)            s.set_restart((PARCEL_INT)o.restart);
      if(o.orthogonalization >= 0) s.set_qr_algo((PARCEL_INT)o.orthogonalization);
      s.solver(Amat, vx, vb); store(s, info);
    } break;
#endif
#if PARCEL_ENABLE_PCAGMRES
    case PARCEL_PY_PCAGMRES: {
      parcel::krylov::pcagmres<T,T,T,T> s; apply_common(s, o);
      if(o.s > 0)       s.set_restart((PARCEL_INT)o.s, (PARCEL_INT)o.t);
      if(o.q_mode >= 0) s.set_explicit_q((PARCEL_INT)o.q_mode);
      if(o.qr >= 0)     s.set_qr_algo((PARCEL_INT)o.qr);
      s.solver(Amat, vx, vb); store(s, info);
    } break;
#endif
    default: break;
    }
    q.wait();
    vx.getValue(x);
    q.wait();
  });
  return agree(mcomm, rc);
}

// -------------------------------------------------------------------------
int parcel_py_eigs(const parcel_py_options *opt, const void *commp,
                   int64_t n_local, int64_t row_start, int64_t n_global, int64_t nnz_local,
                   const int64_t *indptr, const int64_t *indices, const double *data,
                   double *evals, double *evecs, parcel_py_info *info,
                   char *errbuf, int errlen){
  const comm_t mcomm = to_comm(commp);
  clear_info(info);
  std::string msg;
  int rc = check_options(opt, msg);
  if(rc == PARCEL_PY_OK && !is_eigen(opt->solver)){ msg = "not an eigenvalue solver"; rc = PARCEL_PY_ERR_ARG; }
  local_crs A;
  if(rc == PARCEL_PY_OK) rc = make_local_crs(n_local, row_start, n_global, nnz_local, indptr, indices, data, A, msg);
  if(rc == PARCEL_PY_OK && (evals == nullptr || evecs == nullptr)){ msg = "null output array"; rc = PARCEL_PY_ERR_ARG; }
  const int g = agree(mcomm, rc);
  if(g != PARCEL_PY_OK){
    set_err(errbuf, errlen, rc != PARCEL_PY_OK ? msg : std::string("invalid input on another rank"));
    return g;
  }

  const parcel_py_options &o = *opt;
  const bool is_ks = (o.solver == PARCEL_PY_KS_ARNOLDI);
  const int64_t k = parcel_py_eig_count(opt);
  if(info) info->n_eig = k;

  rc = guarded(mcomm, errbuf, errlen, [&](){
    stdout_silencer quiet(o.verbose == 0);
    sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
#if PARCEL_MPI
    parcel::comm comm(mcomm, &q);
#else
    parcel::comm comm(&q);
#endif
    // (A - σI) は set_crsMat() が A を書き換える前に作る (A の shift-invert のとき)
    const bool need_shiftmat = is_ks && o.use_shift && o.precond_spectrum == 0;
    local_crs S;
    if(need_shiftmat) make_shifted(A, o.sigma, S);

    parcel::matrix<T> Amat(&q);
    Amat.set_crsMat<double>(&comm, A.nstart, A.nend, (size_t)A.n, (size_t)A.nnz,
                            A.rowptr.data(), A.col.data(), A.val.data());
    parcel::matrix<T> Smat(&q);
    if(need_shiftmat){
      Smat.set_crsMat<double>(&comm, S.nstart, S.nend, (size_t)S.n, (size_t)S.nnz,
                              S.rowptr.data(), S.col.data(), S.val.data());
    }
    q.wait();

    parcel::vector<T> vval(is_ks ? 2 * (size_t)k : 1, &q);
    parcel::vector<T> vvec((size_t)n_local * (size_t)k, &q);
    {
      std::vector<double> z(is_ks ? 2 * (size_t)k : 1, 0.0);
      vval.setValue(z.data());
      std::vector<double> v0((size_t)n_local * (size_t)k, 0.0);
      if(is_ks && o.use_v0){
        for(int64_t i = 0; i < n_local; i++) v0[i] = evecs[i];
      }else if(is_ks){
        // 大域行番号で決まる初期ベクトル (分割のしかたに依らない)
        for(int64_t i = 0; i < n_local; i++){
          const double gr = (double)(row_start + i);
          v0[i] = 1.0 + 0.5*std::sin(0.317*gr + 0.7) + 0.3*std::cos(0.181*gr - 1.3);
        }
      }
      vvec.setValue(v0.data());
    }
    q.wait();

    switch(o.solver){
#if PARCEL_ENABLE_POWER
    case PARCEL_PY_POWER: {
      parcel::krylov::power<T,T,T,T> s; apply_eigen(s, o);
      s.solver(Amat, vval, vvec); store(s, info);
    } break;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
    case PARCEL_PY_INVERSE_POWER: {
      parcel::krylov::inverse_power<T,T,T,T> s; apply_eigen(s, o);
      s.solver(Amat, vval, vvec); store(s, info);
      if(info) info->linsolve_failures = (int64_t)s.get_linsolve_failures();
    } break;
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
    case PARCEL_PY_KS_ARNOLDI: {
      parcel::krylov::ks_arnoldi<T,T,T,T> s; apply_eigen(s, o);
      if(o.ks_m > 0) s.set_ks_subspace((PARCEL_INT)o.ks_m, (PARCEL_INT)o.ks_k_keep, (PARCEL_INT)o.ks_nev);
      if(o.ks_max_restart > 0) s.set_ks_max_restart((PARCEL_INT)o.ks_max_restart);
      if(o.ks_orthogonalization >= 0) s.set_qr_algo((PARCEL_INT)o.ks_orthogonalization);
      if(o.use_shift){
        s.set_shift(o.sigma);
        s.solver(Amat, vval, vvec, need_shiftmat ? &Smat : nullptr);
      }else{
        s.disable_shift();
        s.solver(Amat, vval, vvec);
      }
      store(s, info);
      if(info){
        info->ks_nconv          = (int64_t)s.get_ks_nconv();
        info->linsolve_failures = (int64_t)s.get_linsolve_failures();
      }
    } break;
#endif
    default: break;
    }
    q.wait();
    vvec.getValue(evecs);
    if(is_ks){
      vval.getValue(evals);
    }else{
      double lam = 0.0;
      vval.getValue(&lam);
      evals[0] = lam; evals[1] = 0.0;
    }
    q.wait();
  });
  return agree(mcomm, rc);
}

} // extern "C"
