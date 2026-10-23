// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include "../../Include/parcel_sycl.hpp"
#include "./diag_mem.hpp"
// 直接法参照解 (dgesv) と固有値検証 (dgeev) でこのサンプル自身が LAPACKE を使う。
//   ks_arnoldi をビルド対象から外しても lapack_int が見つかるよう、ここで include する。
#include "lapacke.h"
#include "./parcel_default.hpp"
#include "./poisson_cpu.hpp"
#include "./convdiff_cpu/convdiff.hpp"
#include "./unstruct_cpu/unstruct.hpp"
#include <sstream>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <iomanip>
#include <cmath>
#include <vector>
#include <iostream>
#include <streambuf>

// ks_arnoldi の固有ベクトルは、検証に使う nev 本 + 1 本 (上位 nev 個の最後が共役対の片割れのとき、
//   対の相手が要る) だけ受け取る (min(k_keep, nev+1) 本)。ライブラリは受け取った本数だけ書く
//   (parcel_krylov_ks.hpp)。装置側 vecx とホスト側 x_cpu_local が k_keep 本より少なくて済む。

// ===== rank 0 の標準出力をファイルにも書く =====
//   MPI 実装によってはランクの標準出力がジョブログへ出てリダイレクト先が空になるため、
//   MPI 実装に依存しないよう、rank 0 が std::cout の内容を自分でファイルにも書く。
//
//     PARCEL_OUTPUT=<パス>  出力先 (既定 parcel_result.txt)
//     PARCEL_OUTPUT=none    ファイルに書かない
//   標準出力にも同じ内容を出す。
class parcel_tee_buf : public std::streambuf {
public:
  parcel_tee_buf(std::streambuf *a, std::streambuf *b) : a_(a), b_(b) {}
protected:
  int overflow(int c) override {
    if(traits_type::eq_int_type(c, traits_type::eof())) return traits_type::not_eof(c);
    a_->sputc(traits_type::to_char_type(c));
    b_->sputc(traits_type::to_char_type(c));
    return c;
  }
  std::streamsize xsputn(const char *s, std::streamsize n) override {
    // 標準出力側が書けなくてもファイル側は書き続けるため、戻り値は n とする
    a_->sputn(s, n);
    b_->sputn(s, n);
    return n;
  }
  int sync() override {
    int ra = a_->pubsync();
    int rb = b_->pubsync();
    return (ra == 0 && rb == 0) ? 0 : -1;
  }
private:
  std::streambuf *a_, *b_;
};

// ===== 直接法による参照解 =====
//   線形ソルバの解を、密行列に組み直して LAPACK dgesv で解いた解と比較する。
//   ポアソンには製造解があるが移流拡散には無いため、こちらで補う。
//   n^2 のメモリが要るので problem_size <= VERIFY_DENSE_REF_MAX_N のときのみ。
#ifndef VERIFY_DENSE_REF_MAX_N
#define VERIFY_DENSE_REF_MAX_N 1024
#endif
// shift-invert の参照に全解析固有値を作る上限 (rank 0 で n 個の固有値を並べ替える)
#ifndef VERIFY_SHIFT_FULL_MAX_N
#define VERIFY_SHIFT_FULL_MAX_N (1<<25)
#endif
template<class Tcrs, class Tvec>
void verify_direct_solution(Tcrs &crsMatrix_cpu,
                            const std::vector<Tvec> &b_global,
                            const std::vector<Tvec> &x_num,
                            PARCEL_INT n_global){
  if(n_global > (VERIFY_DENSE_REF_MAX_N)){
    std::cout << "\n=== VERIFY: 直接法参照解 ===\n";
    std::cout << "  n = " << n_global << " > " << (VERIFY_DENSE_REF_MAX_N)
              << " のため直接法との比較を行わない\n";
    std::cout << "  DIRECT_SKIPPED\n";
    return;
  }
  const int n = (int)n_global;
  PARCEL_INT *rp = crsMatrix_cpu.crsRowptr_data();
  PARCEL_INT *cl = crsMatrix_cpu.crsCol_data();
  auto *aa = crsMatrix_cpu.A_data();
  // 列優先の密行列に展開
  std::vector<double> Adense((size_t)n*(size_t)n, 0.0);
  for(PARCEL_INT i = 0; i < n; i++)
    for(PARCEL_INT p = rp[i]; p < rp[i+1]; p++)
      Adense[(size_t)i + (size_t)cl[p]*(size_t)n] = (double)aa[p];
  std::vector<double> rhs(n);
  for(int i = 0; i < n; i++) rhs[i] = (double)b_global[i];
  std::vector<lapack_int> ipiv(n);
  int info = LAPACKE_dgesv(LAPACK_COL_MAJOR, n, 1,
                           Adense.data(), n, ipiv.data(), rhs.data(), n);
  std::cout << "\n=== VERIFY: 直接法参照解 (密行列 + LAPACK dgesv, n = " << n << ") ===\n";
  if(info != 0){
    std::cout << "  dgesv info = " << info << " -> 参照解を作成できない\n";
    std::cout << "  DIRECT_SKIPPED\n";
    return;
  }
  // NaN/Inf は if(e > max_err) を必ず false にするため、素朴な最大値計算では
  // 「誤差 0」として素通りする。非有限値は別に数えて明示的に落とす。
  double max_err = 0.0, max_ref = 0.0;
  int n_nonfinite = 0;
  for(int i = 0; i < n; i++){
    if(!std::isfinite((double)x_num[i])){ n_nonfinite++; continue; }
    double e = std::fabs((double)x_num[i] - rhs[i]);
    if(e > max_err) max_err = e;
    if(std::fabs(rhs[i]) > max_ref) max_ref = std::fabs(rhs[i]);
  }
  std::cout << std::scientific << std::setprecision(10);
  if(n_nonfinite > 0){
    std::cout << "  VERIFY_NONFINITE : x_numerical に NaN/Inf が "
              << n_nonfinite << " / " << n << " 要素\n";
    return;
  }
  std::cout << "  max|x_numerical - x_direct|                   = " << max_err << "\n";
  std::cout << "  max|x_numerical - x_direct| / max|x_direct|   = "
            << (max_ref > 0.0 ? max_err/max_ref : 0.0) << "\n";
}



// 3D テスト行列を実行時に切り替える薄いディスパッチャ。
//
//   poisson_equation と convdiff_equation はインタフェースが一致しているので、
//   実行時フラグで振り分けるだけでよい。両方を構築してもコストは無い
//   (保持するのは convdiff の c_ (double 1 個) だけ)。
//
//   環境変数:
//     PARCEL_MATRIX     poisson | convdiff   (未指定ならコンパイル時の既定)
//     PARCEL_CONVDIFF_C 移流の強さ c         (未指定なら CONVDIFF_C)
//
//   これで (ソルバ x 前処理) ごとに 1 回ビルドすれば 3 種の行列を試せる。
template<class T>
struct matrix_selector {
  bool convdiff_ = false;
  bool unstruct_ = false;
  poisson_equation<T>  pois_;
  convdiff_equation<T> conv_;
  unstruct_equation<T> unst_;

  void select(bool cd, double c){ convdiff_ = cd; unstruct_ = false; conv_.set_c(c); }
  //   非構造 (行あたり非零数 4〜10) を選ぶ。SpMV の実装最適化が行長一定に依存していないかを
  //   ソルバ単位で確かめるための行列 (unstruct_cpu/unstruct.hpp 参照)。
  void select_unstruct(){ unstruct_ = true; convdiff_ = false; }
  bool   is_convdiff() const { return convdiff_; }
  bool   is_unstruct() const { return unstruct_; }
  double get_c()       const { return conv_.get_c(); }
  //   解析固有値を持つか (非構造は持たないので固有値ソルバの照合を飛ばす)
  bool   has_analytic_eigen() const { return !unstruct_; }

  PARCEL_INT get_nnz(PARCEL_INT n){
    if(unstruct_) return unst_.get_nnz(n);
    return convdiff_ ? conv_.get_nnz(n) : pois_.get_nnz(n);
  }
  void make_matrix(PARCEL_INT n, crs_mat<T> &A){
    if(unstruct_)      unst_.make_matrix(n, A);
    else if(convdiff_) conv_.make_matrix(n, A);
    else               pois_.make_matrix(n, A);
  }
  void make_right_vector(PARCEL_INT n, std::vector<T> &b){
    if(unstruct_)      unst_.make_right_vector(n, b);
    else if(convdiff_) conv_.make_right_vector(n, b);
    else               pois_.make_right_vector(n, b);
  }
  // 大域の行 [rstart, rend] だけを CRS で生成する (行・列番号は大域のまま)。
  //   make_matrix() は n^3 行すべてを作るので、各ランクが呼ぶとメモリがランク数に
  //   比例して増え、大きな NX ではメモリが足りなくなる。
  //   係数と列の並び (-z, -y, -x, 中心, +x, +y, +z) は poisson_cpu/poisson.hpp と
  //   convdiff_cpu/convdiff.hpp の make_matrix() と同じにする。検証時は大域行列の
  //   切り出しと 1 要素ずつ照合する (run_verify 参照)。
  void make_matrix_rows(PARCEL_INT n, PARCEL_INT rstart, PARCEL_INT rend,
                        std::vector<PARCEL_INT> &rowptr,
                        std::vector<PARCEL_INT> &col,
                        std::vector<T> &val){
    //   非構造は行ごとに非零数が変わるので、大域生成と同じ build_row() を 1 行ずつ呼ぶ。
    //   同じ行には必ず同じ列・同じ値が入るので、検証時の 1 要素ずつの照合が通る。
    if(unstruct_){
      const PARCEL_INT nrow_u = rend - rstart + 1;
      rowptr.assign((size_t)nrow_u + 1, 0);
      col.clear(); val.clear();
      col.reserve((size_t)nrow_u * 10); val.reserve((size_t)nrow_u * 10);
      std::vector<PARCEL_INT> rc; std::vector<T> rv;
      for(PARCEL_INT icc = rstart; icc <= rend; icc++){
        unstruct_equation<T>::build_row(n, icc, rc, rv);
        for(size_t j = 0; j < rc.size(); j++){ col.push_back(rc[j]); val.push_back(rv[j]); }
        rowptr[(size_t)(icc - rstart) + 1] = (PARCEL_INT)col.size();
      }
      return;
    }
    const PARCEL_INT nx = n, ny = n, nz = n;
    const double c = convdiff_ ? conv_.get_c() : 0.0;
    // poisson.hpp: 隣接はすべて 1.0、中心 -6.0
    // convdiff.hpp: -x/-y/-z は 1.0 + c、+x/+y/+z は 1.0 - c、中心 -6.0
    const T c_sub = convdiff_ ? T(1.0 + c) : T(1.0);
    const T c_sup = convdiff_ ? T(1.0 - c) : T(1.0);
    const T c_ctr = T(-6.0);
    const PARCEL_INT nrow = rend - rstart + 1;
    rowptr.assign((size_t)nrow + 1, 0);
    col.clear(); val.clear();
    col.reserve((size_t)nrow * 7); val.reserve((size_t)nrow * 7);
    for(PARCEL_INT icc = rstart; icc <= rend; icc++){
      const PARCEL_INT ix = icc % nx;
      const PARCEL_INT iy = (icc / nx) % ny;
      const PARCEL_INT iz = icc / (nx * ny);
      if(iz > 0)      { col.push_back(icc - nx*ny); val.push_back(c_sub); }
      if(iy > 0)      { col.push_back(icc - nx);    val.push_back(c_sub); }
      if(ix > 0)      { col.push_back(icc - 1);     val.push_back(c_sub); }
                        col.push_back(icc);         val.push_back(c_ctr);
      if(ix < nx - 1) { col.push_back(icc + 1);     val.push_back(c_sup); }
      if(iy < ny - 1) { col.push_back(icc + nx);    val.push_back(c_sup); }
      if(iz < nz - 1) { col.push_back(icc + nx*ny); val.push_back(c_sup); }
      rowptr[(size_t)(icc - rstart) + 1] = (PARCEL_INT)col.size();
    }
  }
  void check_solution(std::vector<T> &x, PARCEL_INT n){
    if(unstruct_)      unst_.check_solution(x, n);
    else if(convdiff_) conv_.check_solution(x, n);
    else               pois_.check_solution(x, n);
  }
  void bottom_eigenvalues_complex(PARCEL_INT n, int k,
                                  std::vector<double> &re, std::vector<double> &im){
    if(unstruct_)      unst_.bottom_eigenvalues_complex(n, k, re, im);
    else if(convdiff_) conv_.bottom_eigenvalues_complex(n, k, re, im);
    else               pois_.bottom_eigenvalues_complex(n, k, re, im);
  }
  void top_eigenvalues_complex(PARCEL_INT n, int k,
                               std::vector<double> &re, std::vector<double> &im){
    if(unstruct_)      unst_.top_eigenvalues_complex(n, k, re, im);
    else if(convdiff_) conv_.top_eigenvalues_complex(n, k, re, im);
    else               pois_.top_eigenvalues_complex(n, k, re, im);
  }
};

// ===== 精度の組合せを実行時に選ぶ =====
//
//   データ精度・計算精度はテンプレート引数なので、本体をテンプレート関数に
//   切り出し、main から組合せごとに呼び分ける。1 回のビルドで 9 構成すべてを
//   実体化するためビルド自体は重くなるが、ビルド回数は減る。
//
//   環境変数 PARCEL_PRECISION で指定 (未指定ならコンパイル時の既定)
//     double | calcF16 | dataF16 | calcF | dataF | calcDD | dataDD
//     precon_calcF16 | precon_calcF   (前処理の演算精度のみ)
//   ※ precon_dataF / precon_allF 系はデータ型をビルド時マクロで与えるため、
//     実行時切り替え (PARCEL_RUNTIME_PRECISION) では選べない (精度ごとに別ビルドする)。
template<class T_DATA, class T_CALC, class T_CALC_PRECON,
         class T_PRECON_MAT, class T_PRECON_VEC, class T_QR>
int run_verify(int argc, char *argv[]) {

  // 問題サイズと検証の有無も実行時に指定できる。
  //   PARCEL_NX      格子点数 (未指定ならコンパイル時の NX)
  //   PARCEL_VERIFY  0 で検証処理を丸ごと飛ばす (性能測定用)
  //
  //   検証処理は PARCEL の外側なので測定値は汚さないが、大域配列を余分に持つ
  //   (大域行列 約 84 B/点、右辺 8 B/点、rank 0 の解 8 B/点、固有値ソルバなら
  //   解析固有値の列挙で n 個の double を作ってソート) ため、大きい問題では
  //   無視できない。PARCEL_VERIFY=0 では大域行列・大域右辺を作らず、各ランクは
  //   担当行だけを生成する。
  int nx = NX;
  if(const char *e = std::getenv("PARCEL_NX")){ nx = std::atoi(e); }
  // NX^3 を 64 ビットで求めて PARCEL_INT に収まるかを先に調べる
  //   (-DPARCEL_INT32 のビルドでは NX >= 1291 で桁あふれする)
  if((long long)nx <= 0 ||
     (long long)nx * (long long)nx * (long long)nx > (long long)std::numeric_limits<PARCEL_INT>::max()){
    std::cout << "[VERIFY] NX = " << nx << " : NX^3 が PARCEL_INT (" << (int)PARCEL_INT_BITS
              << " ビット) に収まらない (32 ビット版なら -DPARCEL_INT32 を外してビルドし直す)" << std::endl;
    return 1;
  }
  const PARCEL_INT n_global = (PARCEL_INT)nx * (PARCEL_INT)nx * (PARCEL_INT)nx;
  bool do_verify = true;
  if(const char *e = std::getenv("PARCEL_VERIFY")){ do_verify = (std::atoi(e) != 0); }
  // ライブラリの既定値モード (PARCEL_LIB_DEFAULTS=1)。
  //   ソルバのパラメータの設定関数 (set_tolerance / set_max_iteration / set_restart / set_qr_algo /
  //   set_explicit_q / set_ks_subspace / set_ks_max_restart / 内部ソルバの set_linsolve_* /
  //   set_jacobi / set_linsolve_jacobi) を、このサンプルのコンパイル時の値 (parcel_default.hpp) では
  //   呼ばず、ライブラリの既定を使う。環境変数で明示された値 (PARCEL_TOL、PARCEL_MAX_ITER、
  //   PARCEL_GMRES_RESTART、PARCEL_CAGMRES_S / T / Q、PARCEL_KS_M / K_KEEP / NEV / RESTART / QR、
  //   PARCEL_INNER_TOL / MAX、PARCEL_LINSOLVE_SOLVER / PRECON、PARCEL_BJ_BLOCK) はそのまま適用する。
  //   前処理の種類 (PARCEL_PRECON) と固有値ソルバのモード (PARCEL_PRECON_SPECTRUM、PARCEL_KS_SHIFT) は
  //   解く問題の指定なので常に設定する。実際に使った値は "[LIB_DEFAULTS]" の行に 1 回だけ出す。
  //   未指定 (0) のときは、設定関数をこのサンプルの値ですべて呼ぶ。
  bool lib_defaults = false;
  if(const char *e = std::getenv("PARCEL_LIB_DEFAULTS")){ lib_defaults = (std::atoi(e) != 0); }
  //   表示用 (明示されたか)。設定はそれぞれのブロックの中で行う
  bool q_env_outer = false, moment_outer = false, lin_algo_env_outer = false, lin_precon_env_outer = false;
  bool ks_qr_env_outer = false;   // PARCEL_KS_QR で ks_arnoldi の直交化を明示したか
  // 検証は rank 0 に解を MPI_Gatherv で集める (個数・変位は int)。n_global が int に
  //   収まらなければ変位が桁あふれするので、検証を止めて解くだけにする (全ランクで同じ判定)。
  if(do_verify && (long long)n_global > (long long)std::numeric_limits<int>::max()){
    std::cout << "[VERIFY] n_global = " << (long long)n_global
              << " は MPI の個数 (int) を超えるので検証を行わない (PARCEL_VERIFY=0 と同じ)" << std::endl;
    do_verify = false;
  }

  // in-order キューを使う (PARCEL はカーネルを event なしで順に投入するため。
  //   Include/parcel.hpp の require_in_order を参照)。
  //   デバイスの選び方は既定構築と同じ (--acpp-targets で決まる)。
  sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}};
  std::cout << q.get_device().get_info<sycl::info::device::name>() << std::endl;

#if PARCEL_MPI
  // MPI_Init は main 側で実施
  parcel::comm comm(MPI_COMM_WORLD,&q);
  int myrank   = comm.mpi_rank();
  int nprocess = comm.mpi_size();
  PARCEL_INT nlocal = n_global / nprocess;
  PARCEL_INT nstart = nlocal*myrank;
  PARCEL_INT nend   = nlocal*(myrank+1)-1;
  if( myrank == (nprocess-1) ){ nend = n_global-1; }
#else
  int myrank   = 0;
  int nprocess = 1;
  PARCEL_INT nlocal = n_global;
  PARCEL_INT nstart = 0;
  PARCEL_INT nend   = n_global-1;
#endif

#ifndef VERIFY_TOLERANCE
#define VERIFY_TOLERANCE 1.0e-9
#endif
  double tolerance = VERIFY_TOLERANCE;
  int itrmax = MAX_ITERATION;

  // 3D テスト行列: poisson (実対称, default) または convection-diffusion (非対称, 複素固有値)
  //   -DMATRIX_CONVDIFF で切り替え
  matrix_selector<problem_type> matrix_obj;
  {
    bool   use_cd = false;
    double cval   = CONVDIFF_C;
#ifdef MATRIX_CONVDIFF
    use_cd = true;
#endif
    bool use_unstruct = false;
    if(const char *e = std::getenv("PARCEL_MATRIX")){
      const std::string mv(e);
      use_cd       = (mv == "convdiff" || mv == "convdiff2");
      //   unstruct: 行あたり非零数 4〜10 の非対称行列 (unstruct_cpu/unstruct.hpp)。
      //   SpMV の実装最適化が行長一定に依存していないかを調べるために使う。
      use_unstruct = (mv == "unstruct");
    }
    if(const char *e = std::getenv("PARCEL_CONVDIFF_C")){ cval = std::atof(e); }
    matrix_obj.select(use_cd, cval);
    if(use_unstruct){ matrix_obj.select_unstruct(); }
  }
  auto nnz = matrix_obj.get_nnz(nx);
  // 大域行列は検証するときだけ作る (直接法参照解、前処理付き固有値の参照解、
  //   固有対の残差計算に使う。いずれも do_verify の中)。性能測定では 1x1 の空。
  crs_mat<problem_type> crsMatrix_cpu(do_verify ? n_global : 1, do_verify ? nnz : 1);
  if(do_verify){ matrix_obj.make_matrix(nx,crsMatrix_cpu); }

  // [VERIFY] 既知の厳密解に対応する b を生成 (eigvalue test 時は未使用)
  //   製造解の右辺。線形ソルバの検証にしか使わないので、検証しないなら作らない
  //   (性能測定の右辺は担当行ぶんだけ 1 にする)。
  std::vector<problem_type> b_global_for_verify;
  if(do_verify){
    b_global_for_verify.assign(n_global, problem_type(0));
    matrix_obj.make_right_vector(nx, b_global_for_verify);
  }

#if PARCEL_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#endif

  parcel::matrix<T_DATA> parcel_matrix(&q);
  PARCEL_INT nblock = nend - nstart  + 1;
  // 担当行 [nstart, nend] だけを生成する (大域行列から切り出さない)
  std::vector< PARCEL_INT >   crsRowptr_local;
  std::vector< PARCEL_INT >   crsCol_local;
  std::vector< problem_type > crsA_local;
  matrix_obj.make_matrix_rows(nx, nstart, nend, crsRowptr_local, crsCol_local, crsA_local);
  PARCEL_INT nnz_local = crsRowptr_local[nblock];

  // 検証時は、生成した担当行が大域行列の切り出しと一致することを確かめる。
  //   make_matrix_rows() は make_matrix() と係数・列順を別に書いているので、
  //   食い違えば数値検証の意味がなくなる。一致しなければ止める。
  if(do_verify){
    PARCEL_INT *rp = crsMatrix_cpu.crsRowptr_data();
    PARCEL_INT *cl = crsMatrix_cpu.crsCol_data();
    problem_type *aa = crsMatrix_cpu.A_data();
    PARCEL_INT n_mismatch = 0;
    if(nnz_local != rp[nend+1] - rp[nstart]){ n_mismatch = 1; }
    for(PARCEL_INT irow = nstart; irow <= nend && n_mismatch == 0; irow++){
      const PARCEL_INT l0 = crsRowptr_local[irow-nstart];
      if(crsRowptr_local[irow-nstart+1] - l0 != rp[irow+1] - rp[irow]){ n_mismatch++; break; }
      for(PARCEL_INT k = 0; k < rp[irow+1] - rp[irow]; k++){
        if(crsCol_local[l0+k] != cl[rp[irow]+k] || !(crsA_local[l0+k] == aa[rp[irow]+k])){ n_mismatch++; break; }
      }
    }
    if(n_mismatch != 0){
      std::cout << "[VERIFY] VERIFY_MATRIX_MISMATCH: rank " << myrank
                << " の担当行の生成が大域行列と一致しない" << std::endl;
#if PARCEL_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      return 1;
    }
    if(myrank == 0){ std::cout << "[VERIFY] 担当行の生成は大域行列の切り出しと一致" << std::endl; }
  }

  // set_crsMat() は引数の crsCol / A を破壊的に書き換える
  //   ・列番号: 大域番号 -> 局所番号 (自ランク分は icol-nstart, 袖は nblock+igcol)
  //   ・値    : 自ランク列が先, 袖が後になるよう行ごとに並べ替え
  // 同じ配列から 2 つ目の行列 (A - σI) を作ると二重に変換され、袖通信テーブルが壊れる。
  // shift-invert 用に原本を退避しておく。
#ifdef KS_SHIFT_INVERT
  std::vector< PARCEL_INT >   crsCol_orig = crsCol_local;
  std::vector< problem_type > crsA_orig   = crsA_local;
#endif

  parcel_matrix.template set_crsMat<problem_type>(
#if PARCEL_MPI
					 &comm,nstart,nend,
					 nblock,nnz_local,
					 crsRowptr_local.data(),
					 crsCol_local.data(),
					 crsA_local.data()
#else
					 n_global,nnz,
					 crsRowptr_local.data(),
					 crsCol_local.data(),
					 crsA_local.data()
#endif
					    );

  // ===== ソルバを実行時に選ぶ =====
  //   環境変数 PARCEL_SOLVER で指定 (未指定ならコンパイル時の既定)
  //     pcg | pbicgstab | pgmres | pcagmres | power | inverse_power | ks_arnoldi
  //   Q を陽に作らない CA-GMRES は PARCEL_SOLVER=pcagmres PARCEL_CAGMRES_Q=implicit で解く
  //     (下の pcagmres の設定を参照)。cholqr_pcagmres という名前は受け付けずに止める。
  //   ソルバ本体は構築しても軽い (パラメタを保持するだけで、作業ベクトルは
  //   solver() の中で確保される)。7 種すべて構築しても実害はない。
  enum solver_kind { SK_PCG = 0, SK_PBICGSTAB, SK_PGMRES, SK_PCAGMRES,
                     SK_POWER, SK_INVERSE_POWER, SK_KS_ARNOLDI };
  static const char *solver_names[] = { "pcg", "pbicgstab", "pgmres", "pcagmres",
                                        "power", "inverse_power", "ks_arnoldi" };
  const int n_solver_kinds = 7;
  // そのソルバがこのバイナリに入っているか (parcel_solver_select.hpp)
  auto parcel_solver_built = [](solver_kind k) -> bool {
    switch(k){
    case SK_PCG:           return PARCEL_ENABLE_PCG           != 0;
    case SK_PBICGSTAB:     return PARCEL_ENABLE_PBICGSTAB     != 0;
    case SK_PGMRES:        return PARCEL_ENABLE_PGMRES        != 0;
    case SK_PCAGMRES:      return PARCEL_ENABLE_PCAGMRES      != 0;
    case SK_POWER:         return PARCEL_ENABLE_POWER         != 0;
    case SK_INVERSE_POWER: return PARCEL_ENABLE_INVERSE_POWER != 0;
    case SK_KS_ARNOLDI:    return PARCEL_ENABLE_KS_ARNOLDI    != 0;
    }
    return false;
  };
  //   ビルド対象から外したソルバは既定にも選ばない (parcel_solver_select.hpp)。
  solver_kind kind =
#if ks_arnoldi_solver && PARCEL_ENABLE_KS_ARNOLDI
    SK_KS_ARNOLDI;
#elif power_solver && PARCEL_ENABLE_POWER
    SK_POWER;
#elif inverse_power_solver && PARCEL_ENABLE_INVERSE_POWER
    SK_INVERSE_POWER;
#elif pbicgstab_solver && PARCEL_ENABLE_PBICGSTAB
    SK_PBICGSTAB;
#elif pgmres_solver && PARCEL_ENABLE_PGMRES
    SK_PGMRES;
#elif pcagmres_solver && PARCEL_ENABLE_PCAGMRES
    SK_PCAGMRES;
#elif PARCEL_ENABLE_PCG
    SK_PCG;
#elif PARCEL_ENABLE_PBICGSTAB
    SK_PBICGSTAB;
#elif PARCEL_ENABLE_PGMRES
    SK_PGMRES;
#elif PARCEL_ENABLE_PCAGMRES
    SK_PCAGMRES;
#elif PARCEL_ENABLE_POWER
    SK_POWER;
#elif PARCEL_ENABLE_INVERSE_POWER
    SK_INVERSE_POWER;
#else
    SK_KS_ARNOLDI;
#endif
  if(const char *e = std::getenv("PARCEL_SOLVER")){
    const std::string v(e);
    bool ok = true;
    if     (v == "pcg")           kind = SK_PCG;
    else if(v == "pbicgstab")     kind = SK_PBICGSTAB;
    else if(v == "pgmres")        kind = SK_PGMRES;
    else if(v == "pcagmres")      kind = SK_PCAGMRES;
    else if(v == "power")         kind = SK_POWER;
    else if(v == "inverse_power") kind = SK_INVERSE_POWER;
    else if(v == "ks_arnoldi")    kind = SK_KS_ARNOLDI;
    else if(v == "cholqr_pcagmres"){
      // この名前は既定のソルバに落とさず止める (別のものを測らないように)
      if(myrank == 0){
        std::cout << " PARCEL_SOLVER=cholqr_pcagmres は廃止した (pcagmres に統合)。"
                  << "PARCEL_SOLVER=pcagmres PARCEL_CAGMRES_Q=implicit を使う" << std::endl;
      }
#if PARCEL_MPI
      MPI_Abort(MPI_COMM_WORLD, 2);
#else
      return 2;
#endif
    }
    else ok = false;
    if(!ok && myrank == 0){
      std::cout << " 未知の PARCEL_SOLVER=" << v << " -> コンパイル時の既定を使う" << std::endl;
    }
    // 外したソルバを指定されたら、黙って別のものを解かずに止める。
    //   測ったつもりのものと違う結果を持ち帰るのを防ぐ。
    if(ok && !parcel_solver_built(kind)){
      if(myrank == 0){
        std::cout << " PARCEL_SOLVER=" << v
                  << " はこのバイナリに入っていない (parcel_solver_select.hpp で外されている)"
                  << std::endl;
        std::cout << " 入っているもの:";
        for(int i = 0; i < n_solver_kinds; i++)
          if(parcel_solver_built((solver_kind)i)) std::cout << " " << solver_names[i];
        std::cout << std::endl;
      }
#if PARCEL_MPI
      MPI_Abort(MPI_COMM_WORLD, 2);
#else
      return 2;
#endif
    }
  }
  const bool is_eigen = (kind == SK_POWER || kind == SK_INVERSE_POWER || kind == SK_KS_ARNOLDI);
  const bool is_ks    = (kind == SK_KS_ARNOLDI);
  //   KS の部分空間パラメタも実行時に振れるようにする。vecx/vecb のサイズ決定にも
  //   使うので、ソルバ構築より前に確定させておく。
  int ks_m = KS_M, ks_keep = KS_K_KEEP, ks_nev = KS_NEV, ks_restart = KS_MAX_RESTART;
  bool ks_m_env = false, ks_keep_env = false, ks_nev_env = false;   // 既定値モードで使う (明示されたか)
  if(const char *e = std::getenv("PARCEL_KS_M"))       { ks_m       = std::atoi(e); ks_m_env    = true; }
  if(const char *e = std::getenv("PARCEL_KS_K_KEEP"))  { ks_keep    = std::atoi(e); ks_keep_env = true; }
  if(const char *e = std::getenv("PARCEL_KS_NEV"))     { ks_nev     = std::atoi(e); ks_nev_env  = true; }
  //   PARCEL_KS_RESTART=default: set_ks_max_restart を呼ばずライブラリの既定を使う (回帰試験はこちら)
  //   既定値モードでは未指定も default と同じ。
  bool ks_restart_default = false;
  if(const char *e = std::getenv("PARCEL_KS_RESTART")) {
    if(std::string(e) == "default") ks_restart_default = true; else ks_restart = std::atoi(e);
  } else if(lib_defaults) {
    ks_restart_default = true;
  }
  if(myrank == 0){ std::cout << " solver = " << solver_names[(int)kind] << std::endl; }

  if(!is_eigen){
  //   線形ソルバーは matrix を GPU に渡せば CPU 側不要だが、
  //   直接法参照解 (密行列 + dgesv) を作る規模なら保持しておく。
  //   (固有値ソルバーは verify で SpMV 残差計算するため常に keep)
  if(n_global > (VERIFY_DENSE_REF_MAX_N)){
    crsMatrix_cpu.reset();
  }
  }

#ifdef KS_SHIFT_INVERT
  // (A - σI) shifted matrix を別途構築 → ks_arnoldi.solver() に渡す
  //   σ は -DKS_SIGMA=<val> で指定 (デフォルト 0.0)
#ifndef KS_SIGMA
#define KS_SIGMA 0.0
#endif
  double ks_sigma = KS_SIGMA;
  //   σ は実行時にも上書きできる (1 バイナリで複数の σ を試すため)
  if(const char *e = std::getenv("PARCEL_KS_SIGMA")){ ks_sigma = std::atof(e); }
  //   退避しておいた原本 (大域列番号) から作る。crsCol_local / crsA_local は
  //   1 つ目の set_crsMat で既に局所番号に書き換わっているので使えない。
  //   crsRowptr_local は書き換えられないためそのまま使う。
  std::vector<PARCEL_INT>   shiftCol_local = crsCol_orig;
  std::vector<problem_type> shiftA_local   = crsA_orig;
  //   行・非零の番号は PARCEL_INT で回す (既定 64 ビット。int では大きな行列で桁あふれする)
  for(PARCEL_INT irow = nstart; irow <= nend; irow++){
    PARCEL_INT loc = irow - nstart;
    for(PARCEL_INT j = crsRowptr_local[loc]; j < crsRowptr_local[loc+1]; j++){
      if(shiftCol_local[j] == irow){ shiftA_local[j] -= ks_sigma; break; }
    }
  }
  parcel::matrix<T_DATA> parcel_matrix_shifted(&q);
  parcel_matrix_shifted.template set_crsMat<problem_type>(
#if PARCEL_MPI
                          &comm,nstart,nend,
                          nblock,nnz_local,
                          crsRowptr_local.data(),
                          shiftCol_local.data(),
                          shiftA_local.data()
#else
                          n_global,nnz,
                          crsRowptr_local.data(),
                          shiftCol_local.data(),
                          shiftA_local.data()
#endif
                          );
  // 原本はもう不要
  std::vector<PARCEL_INT>().swap(crsCol_orig);
  std::vector<problem_type>().swap(crsA_orig);
#endif

  // 担当行の CRS は set_crsMat() で PARCEL 側に写したので、ここで解放する。
  //   残しておくと 1 ランクあたり約 12 B/非零を解き終わるまで持ち続ける。
  //   これ以降は参照しない。
  std::vector<PARCEL_INT>().swap(crsRowptr_local);
  std::vector<PARCEL_INT>().swap(crsCol_local);
  std::vector<problem_type>().swap(crsA_local);

  parcel::vector<T_DATA> vecx(nblock,&q);
  parcel::vector<T_DATA> vecy(nblock,&q);
  parcel::vector<T_DATA> vecb(nblock,&q);
  std::vector<problem_type> x_cpu_local(nblock);
  std::vector<problem_type> b_cpu_local(nblock);
  // [VERIFY] x0=0, b=make_right_vector(nx,...) のローカル切り出し
  for(PARCEL_INT irow=nstart;irow<=nend;irow++){
    x_cpu_local[irow-nstart] = 0.0;
    b_cpu_local[irow-nstart] = do_verify ? b_global_for_verify[irow] : problem_type(1);   // 性能測定は b = 1
  }
  vecx.setValue(x_cpu_local.data());
  vecb.setValue(b_cpu_local.data());

  using CT  = T_CALC;
  using CPT = T_CALC_PRECON;
#if PARCEL_ENABLE_PCG
  parcel::krylov::pcg          <CT,CPT,CT,CT> s_pcg;
#endif
#if PARCEL_ENABLE_PBICGSTAB
  parcel::krylov::pbicgstab    <CT,CPT,CT,CT> s_bicg;
#endif
#if PARCEL_ENABLE_PGMRES
  parcel::krylov::pgmres       <CT,CPT,CT,CT> s_gmres;
#endif
#if PARCEL_ENABLE_PCAGMRES
  parcel::krylov::pcagmres     <CT,CPT,CT,CT> s_cagmres;
#endif
#if PARCEL_ENABLE_POWER
  parcel::krylov::power        <CT,CPT,CT,CT> s_power;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
  parcel::krylov::inverse_power<CT,CPT,CT,CT> s_invpow;
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
  parcel::krylov::ks_arnoldi   <CT,CPT,CT,CT> s_ks;
#endif

  // 共通設定は krylov_param の基底ポインタから行う。
  // krylov_param は precon_param を継承しているので前処理のセッタもここから呼べる。
  parcel::krylov::krylov_param *kp = nullptr;
  switch(kind){
#if PARCEL_ENABLE_PCG
  case SK_PCG:           kp = &s_pcg;     break;
#endif
#if PARCEL_ENABLE_PBICGSTAB
  case SK_PBICGSTAB:     kp = &s_bicg;    break;
#endif
#if PARCEL_ENABLE_PGMRES
  case SK_PGMRES:        kp = &s_gmres;   break;
#endif
#if PARCEL_ENABLE_PCAGMRES
  case SK_PCAGMRES:      kp = &s_cagmres; break;
#endif
#if PARCEL_ENABLE_POWER
  case SK_POWER:         kp = &s_power;   break;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
  case SK_INVERSE_POWER: kp = &s_invpow;  break;
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
  case SK_KS_ARNOLDI:    kp = &s_ks;      break;
#endif
  default: break;
  }

  // ソルバ固有の設定。選ばれていないものに設定しても副作用は無いので全部入れる。
  //   リスタート周期は実行時に上書きできる (未指定ならコンパイル時の値)。
  //     PARCEL_GMRES_RESTART=<m>             : pgmres の restart
  //     PARCEL_CAGMRES_S=<s> PARCEL_CAGMRES_T=<t> : pcagmres の (s, t)
  //   値に default を与えると set_restart を呼ばず、ライブラリの既定を使う (回帰試験はこちら)。
  //   PARCEL_CAGMRES_S=default なら (s, t) とも既定 (PARCEL_CAGMRES_T は見ない)。
  PARCEL_INT gmres_restart = GMRES_RESTART, cagmres_s = CAGMRES_RESTART_S, cagmres_t = CAGMRES_RESTART_T;
  //   既定値モード (PARCEL_LIB_DEFAULTS=1) では、未指定は default と同じ。PARCEL_CAGMRES_S / T の
  //   片方だけを数値で与えたときは、もう片方をライブラリの既定にする。
  bool gmres_restart_default = false, cagmres_restart_default = false;
  bool cagmres_s_env = false, cagmres_t_env = false;
  if(const char *e = std::getenv("PARCEL_GMRES_RESTART")){
    if(std::string(e) == "default") gmres_restart_default = true; else gmres_restart = (PARCEL_INT)std::atol(e);
  } else if(lib_defaults) {
    gmres_restart_default = true;
  }
  if(const char *e = std::getenv("PARCEL_CAGMRES_S")){
    if(std::string(e) == "default") cagmres_restart_default = true; else { cagmres_s = (PARCEL_INT)std::atol(e); cagmres_s_env = true; }
  }
  if(const char *e = std::getenv("PARCEL_CAGMRES_T"))    { cagmres_t     = (PARCEL_INT)std::atol(e); cagmres_t_env = true; }
#if PARCEL_ENABLE_PGMRES
  if(!gmres_restart_default) s_gmres.set_restart(gmres_restart);
  if(!lib_defaults) s_gmres.set_qr_algo(GMRES_QR_ALGO);
  if(myrank == 0 && kind == SK_PGMRES){ std::cout << " gmres_restart = " << (long long)s_gmres.get_restart() << std::endl; }
#endif
#if PARCEL_ENABLE_PCAGMRES
  if(lib_defaults && !cagmres_restart_default){
    if(!cagmres_s_env && !cagmres_t_env){
      cagmres_restart_default = true;
    }else{
      if(!cagmres_s_env) cagmres_s = s_cagmres.get_restart_s();
      if(!cagmres_t_env) cagmres_t = s_cagmres.get_restart_t();
    }
  }
  if(!cagmres_restart_default) s_cagmres.set_restart(cagmres_s, cagmres_t);
  if(!lib_defaults) s_cagmres.set_qr_algo(CAGMRES_QR_ALGO);
  // Q を陽に作るか。
  //   PARCEL_CAGMRES_Q=auto (既定) | explicit | implicit  -> set_explicit_q(CAGMRES_Q_*)
  //     implicit の QR は CholeskyQR 系だけなので、CAGMRES_QR_ALGO によらず QR_CHOLESKYQR にする。
  //   PARCEL_CHOLQR_MOMENT=1 : QR_CAGMRES_CHOLESKYQR (対称行列・前処理なし・s 偶数専用のモーメント法)。
  //     implicit のときだけ使える。
  //   未知の値・implicit 以外でのモーメント法 (pcagmres を解くとき) は止める。
  {
    PARCEL_INT q_mode = parcel::krylov::CAGMRES_Q_AUTO;
    bool q_ok = true;
    std::string q_str = "auto";
    bool q_env = false;
    if(const char *e = std::getenv("PARCEL_CAGMRES_Q")){
      q_str = e;
      q_env = true;
      if     (q_str == "auto")     q_mode = parcel::krylov::CAGMRES_Q_AUTO;
      else if(q_str == "explicit") q_mode = parcel::krylov::CAGMRES_Q_EXPLICIT;
      else if(q_str == "implicit") q_mode = parcel::krylov::CAGMRES_Q_IMPLICIT;
      else q_ok = false;
    }
    bool moment = false;
    if(const char *e = std::getenv("PARCEL_CHOLQR_MOMENT")){ moment = (std::atoi(e) != 0); }
    q_env_outer = q_env; moment_outer = moment;
    const bool moment_bad = moment && (kind == SK_PCAGMRES) && (q_mode != parcel::krylov::CAGMRES_Q_IMPLICIT);
    if(!q_ok || moment_bad){
      if(myrank == 0){
        if(!q_ok){
          std::cout << "[PARCEL] PARCEL_CAGMRES_Q は auto / explicit / implicit のどれか: " << q_str << std::endl;
        }else{
          std::cout << "[PARCEL] PARCEL_CHOLQR_MOMENT=1 は PARCEL_CAGMRES_Q=implicit のときだけ使える (指定: "
                    << q_str << ")" << std::endl;
        }
      }
#if PARCEL_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      return 1;
    }
    //   既定値モードでは、Q の作り方は指定されたときだけ設定する。implicit の QR は、モーメント法を
    //   指定しなければ set_qr_algo を呼ばない (ライブラリが QR_CHOLESKYQR を使う: effective_qr_algo)。
    if(!lib_defaults || q_env) s_cagmres.set_explicit_q(q_mode);
    if(q_mode == parcel::krylov::CAGMRES_Q_IMPLICIT && (!lib_defaults || moment)){
      s_cagmres.set_qr_algo(moment ? parcel::qr::QR_CAGMRES_CHOLESKYQR : parcel::qr::QR_CHOLESKYQR);
    }
    if(myrank == 0 && kind == SK_PCAGMRES){
      std::cout << " cagmres_q = " << parcel::krylov::cagmres_q_name(s_cagmres.get_explicit_q())
                << " , cagmres_qr_algo = " << (int)s_cagmres.get_qr_algo()
                << " , cagmres_s = " << (int)s_cagmres.get_restart_s()
                << " , cagmres_t = " << (int)s_cagmres.get_restart_t() << std::endl;
    }
  }
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
  if(!lib_defaults){
    s_ks.set_ks_subspace(ks_m, ks_keep, ks_nev);
  }else{
    //   既定値モード: 明示された値だけ上書きし、残りはライブラリの既定。
    //   ks_keep は vecx / vecb の大きさに使うので、実際の値を読み直す。
    if(ks_m_env || ks_keep_env || ks_nev_env){
      s_ks.set_ks_subspace(ks_m_env    ? (PARCEL_INT)ks_m    : s_ks.get_ks_m(),
                           ks_keep_env ? (PARCEL_INT)ks_keep : s_ks.get_ks_k_keep(),
                           ks_nev_env  ? (PARCEL_INT)ks_nev  : s_ks.get_ks_nev());
    }
    ks_m    = (int)s_ks.get_ks_m();
    ks_keep = (int)s_ks.get_ks_k_keep();
    ks_nev  = (int)s_ks.get_ks_nev();
  }
  if(!ks_restart_default) s_ks.set_ks_max_restart(ks_restart);
  if(myrank == 0 && kind == SK_KS_ARNOLDI){ std::cout << " ks_max_restart = " << (long long)s_ks.get_ks_max_restart() << (ks_restart_default ? " (default)" : "") << std::endl; }
  //   直交化 (set_qr_algo): PARCEL_KS_QR=cgs | mgs (cgs2 / mgs2 も可。数値はそのまま set_qr_algo に渡す)。
  //   指定しなければ set_qr_algo を呼ばない (ライブラリの既定。ビルド時の PARCEL_KS_ORTHO_DEFAULT)。既定値モードでも同じ
  if(const char *e = std::getenv("PARCEL_KS_QR")){
    const std::string v(e);
    ks_qr_env_outer = true;
    if     (v == "cgs" || v == "cgs2" || v == "CGS" || v == "CGS2") s_ks.set_qr_algo(parcel::qr::QR_CGS);
    else if(v == "mgs" || v == "mgs2" || v == "MGS" || v == "MGS2") s_ks.set_qr_algo(parcel::qr::QR_MGS);
    else if(!v.empty() && v.find_first_not_of("-0123456789") == std::string::npos) s_ks.set_qr_algo((PARCEL_INT)std::atol(e));
    else{
      if(myrank == 0){ std::cout << "[PARCEL] PARCEL_KS_QR は cgs / mgs のどちらか: " << v << std::endl; }
#if PARCEL_MPI
      MPI_Abort(MPI_COMM_WORLD, 1);
#endif
      return 1;
    }
  }
  if(myrank == 0 && kind == SK_KS_ARNOLDI){
    std::cout << " ks_qr_algo = " << (s_ks.get_qr_algo() == parcel::qr::QR_CGS ? "CGS (CGS2)" : "MGS (MGS2)")
              << (ks_qr_env_outer ? " (env)" : " (default)") << std::endl;
  }
#endif
  //   入れ子ソルバの閾値。外側より厳しくするが、低精度では到達できない値を
  //   指定すると反復上限まで回り続けるため -D で調整できるようにしてある。
  //   内部ソルバの閾値・反復上限も実行時に指定できる。
  //   FP16 前処理は相対残差 ~1e-6 が床で、下回るまで追い込むと停滞ではなく
  //   NaN で破綻する。精度構成ごとに
  //   到達可能な値を渡す必要があるため、外側と独立に指定できるようにする。
  double inner_tol = INVPOW_LINSOLVE_TOL;
  int    inner_max = INVPOW_LINSOLVE_MAX;
  bool inner_tol_env = false, inner_max_env = false;   // 既定値モードで使う (明示されたか)
  if(const char *e = std::getenv("PARCEL_INNER_TOL")){ inner_tol = std::atof(e); inner_tol_env = true; }
  if(const char *e = std::getenv("PARCEL_INNER_MAX")){ inner_max = std::atoi(e); inner_max_env = true; }
#if PARCEL_ENABLE_INVERSE_POWER
  if(!lib_defaults || inner_tol_env) s_invpow.set_linsolve_tolerance(inner_tol);
  if(!lib_defaults || inner_max_env) s_invpow.set_linsolve_max_iteration(inner_max);
#endif
  //   PARCEL_PRECON_SPECTRUM=1: 固有値ソルバ (power / inverse_power / ks_arnoldi) が
  //   前処理後の行列 A M^-1 の固有値を求める (固有値ソルバ共通)。
  //   未指定 (0) では A の固有値を求め、PARCEL_PRECON の前処理は外側の反復に使わない。
  //   前処理なしでは M = I なので A の固有値のまま。
  bool precon_spectrum = false;
  if(const char *e = std::getenv("PARCEL_PRECON_SPECTRUM")){ precon_spectrum = (std::atoi(e) != 0); }
#if PARCEL_ENABLE_POWER
  s_power.set_precon_spectrum(precon_spectrum);
#endif
#if PARCEL_ENABLE_INVERSE_POWER
  s_invpow.set_precon_spectrum(precon_spectrum);
  if(!lib_defaults) s_invpow.set_linsolve_restart(KS_SI_LINSOLVE_RESTART);
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
  s_ks.set_precon_spectrum(precon_spectrum);
#endif
  if(myrank == 0 && precon_spectrum){
    std::cout << " precon_spectrum = 1 (固有値ソルバは A M^-1 の固有値)" << std::endl;
  }
  //   固有値ソルバの内部線形ソルバ (inverse_power、ks_arnoldi の shift-invert)。
  //     PARCEL_LINSOLVE_PRECON=<0|1|2|4> : 内部ソルバの前処理。未指定ならライブラリの既定
  //                                        (4 = ヤコビ)。
  //                                        A M^-1 モードでは内部ソルバの前処理は M なので使わない。
  //     PARCEL_LINSOLVE_SOLVER=pgmres|pcg: inverse_power の内部ソルバ (既定 pgmres)。
  //   ブロックヤコビの分割数は PARCEL_BJ_BLOCK (未指定なら自動)、ヤコビの反復回数は JACOBI_ITER。
  {
    PARCEL_INT lin_precon = -1;   // -1 = 未指定 (ライブラリの既定のまま)
    if(const char *e = std::getenv("PARCEL_LINSOLVE_PRECON")){ lin_precon = (PARCEL_INT)std::atoi(e); }
    PARCEL_INT lin_bj = 0;
    if(const char *e = std::getenv("PARCEL_BJ_BLOCK")){ const long v = std::atol(e); if(v > 0){ lin_bj = (PARCEL_INT)v; } }
    PARCEL_INT lin_algo = parcel::krylov::LINSOLVE_PGMRES;
    bool lin_algo_env = false;   // 既定値モードで使う (明示されたか)
    if(const char *e = std::getenv("PARCEL_LINSOLVE_SOLVER")){
      const std::string v(e);
      lin_algo_env = true;
      if(v == "pcg"){ lin_algo = parcel::krylov::LINSOLVE_PCG; }
      else if(v != "pgmres"){
        if(myrank == 0){ std::cout << "[VERIFY] PARCEL_LINSOLVE_SOLVER は pgmres か pcg: " << v << std::endl; }
#if PARCEL_MPI
        MPI_Abort(MPI_COMM_WORLD, 1);
#endif
        return 1;
      }
    }
    lin_algo_env_outer = lin_algo_env; lin_precon_env_outer = (lin_precon >= 0);
    auto set_lin = [&](parcel::krylov::eigen_param &ep){
      if(lin_precon >= 0){ ep.set_linsolve_precon_algo(lin_precon); }
      if(!lib_defaults){ ep.set_linsolve_jacobi(JACOBI_ITER); }
      if(lin_bj > 0){ ep.set_linsolve_blockjacobi(lin_bj); }
    };
#if PARCEL_ENABLE_INVERSE_POWER
    set_lin(s_invpow);
    if(!lib_defaults || lin_algo_env) s_invpow.set_linsolve_algo(lin_algo);
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
    set_lin(s_ks);
#endif
    // 表示は実際に使う値 (未指定ならライブラリの既定)
    PARCEL_INT lin_precon_eff = lin_precon;
#if PARCEL_ENABLE_INVERSE_POWER
    if(kind == SK_INVERSE_POWER){ lin_precon_eff = s_invpow.get_linsolve_precon_param()->get_precon_algo(); }
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
    if(kind == SK_KS_ARNOLDI){ lin_precon_eff = s_ks.get_linsolve_precon_param()->get_precon_algo(); }
#endif
    if(myrank == 0 && (kind == SK_INVERSE_POWER || kind == SK_KS_ARNOLDI)){
      std::cout << " linsolve_precon = " << (int)lin_precon_eff << (lin_precon < 0 ? " (default)" : "")
                << " , linsolve_solver = " << (lin_algo == parcel::krylov::LINSOLVE_PCG ? "pcg" : "pgmres")
                << std::endl;
    }
  }
  //   PARCEL_KS_SHIFT=<σ>: ks_arnoldi の shift-invert を実行時に有効にする。
  //   A M^-1 モード (PARCEL_PRECON_SPECTRUM=1) はシフト行列 (A - σI) を使わないので
  //   -DKS_SHIFT_INVERT の別ビルドが要らない。
  //   PARCEL_PRECON_SPECTRUM=1 なしでも使える。ライブラリは ShiftMat なしの set_shift(σ) を
  //   K = I の A の shift-invert として扱い、A の σ 近傍の固有値を返す。
  //   内部 pgmres の前処理は K = I 固定 (PARCEL_LINSOLVE_PRECON は効かない)。
  //   シフト行列 (A - σI) を渡す経路は -DKS_SHIFT_INVERT でビルドする。
  bool   ks_runtime_shift = false;
  double ks_runtime_sigma = 0.0;
#ifndef KS_SHIFT_INVERT
  if(const char *e = std::getenv("PARCEL_KS_SHIFT")){ ks_runtime_shift = true; ks_runtime_sigma = std::atof(e); }
  if(ks_runtime_shift && !precon_spectrum && myrank == 0){
    std::cout << " ks shift-invert (実行時) : ShiftMat なし, K = I で A の shift-invert" << std::endl;
  }
#if PARCEL_ENABLE_KS_ARNOLDI
  if(ks_runtime_shift){
    s_ks.set_shift(ks_runtime_sigma);
    if(!lib_defaults || inner_tol_env) s_ks.set_shift_linsolve_tolerance(inner_tol);
    if(!lib_defaults || inner_max_env) s_ks.set_shift_linsolve_max_iteration(inner_max);
    if(!lib_defaults) s_ks.set_shift_linsolve_restart(KS_SI_LINSOLVE_RESTART);
    if(myrank == 0){ std::cout << " ks shift-invert (実行時) sigma = " << ks_runtime_sigma << std::endl; }
  }
#endif
#endif
#if defined(KS_SHIFT_INVERT) && PARCEL_ENABLE_KS_ARNOLDI
  s_ks.set_shift(ks_sigma);
  if(!lib_defaults || inner_tol_env) s_ks.set_shift_linsolve_tolerance(inner_tol);
  if(!lib_defaults || inner_max_env) s_ks.set_shift_linsolve_max_iteration(inner_max);
  if(!lib_defaults) s_ks.set_shift_linsolve_restart(KS_SI_LINSOLVE_RESTART);
#endif

  // ks_arnoldi だけ vecx / vecb の意味とサイズが違う
  //   vecb: 固有値 (Re,Im 交互) サイズ 2*k_keep
  //   vecx: 固有ベクトル サイズ n*ks_ev_cols (入力時は先頭 n が seed)
  //     ks_ev_cols = min(k_keep, nev+1) (先頭の説明)
  const int ks_ev_cols =
    std::max(1, std::min(ks_keep, std::max(ks_nev, 1) + 1));
  if(is_ks){
    vecb.resize(ks_keep * 2, &q);
    b_cpu_local.resize(vecb.size());
    for(int i = 0; i < (int)vecb.size(); i++) b_cpu_local[i] = 0.0;
    vecb.setValue(b_cpu_local.data());

    //   長さは size_t で掛ける (n*k_keep が 2^31 を超えうる)
    vecx.resize((size_t)ks_ev_cols * (size_t)nblock, &q);
    x_cpu_local.resize(vecx.size());
    for(size_t i = 0; i < vecx.size(); i++) x_cpu_local[i] = 0.0;
    {
      const long long base = (long long)nstart;
      for(PARCEL_INT irow = 0; irow < nblock; irow++){
        double g = (double)(base + irow);
        x_cpu_local[irow] = 1.0 + 0.5*std::sin(0.317*g + 0.7)
                                + 0.3*std::cos(0.181*g - 1.3);
      }
    }
    vecx.setValue(x_cpu_local.data());
  }

  // 反復上限と収束閾値も実行時に上書きできる。どちらも PARCEL 側は実行時セッタ。
  //   ソルバごとに適切な値が違うため (線形 10000 / power 2000 / inverse_power 200 /
  //   ks_arnoldi はリスタート数 10)、1 バイナリで回すにはここが可変である必要がある。
  //   既定値モードでは、環境変数で明示されたときだけ設定する。
  bool itrmax_env = false, tol_env = false;
  if(const char *e = std::getenv("PARCEL_MAX_ITER")){ itrmax    = std::atoi(e); itrmax_env = true; }
  if(const char *e = std::getenv("PARCEL_TOL"))     { tolerance = std::atof(e); tol_env    = true; }
  if(!lib_defaults || tol_env)    kp->set_tolerance(tolerance);
  if(!lib_defaults || itrmax_env) kp->set_max_iteration(itrmax);

  // 前処理は実行時に切り替えられる。PARCEL 側は set_precon_algo() という
  // セッタを持っており、コンパイル時に決める必要はない。
  // 環境変数 PARCEL_PRECON で上書きし、未指定ならコンパイル時の既定値を使う。
  //   0 = なし / 1 = 点ヤコビ / 2 = ブロックヤコビ / 4 = ヤコビ
  // これにより (ソルバ x 行列) ごとに 1 回ビルドすれば 4 種の前処理を試せる。
  PARCEL_INT precon_algo = parcel_solver_precon_algo;
  if(const char *e = std::getenv("PARCEL_PRECON")){ precon_algo = (PARCEL_INT)std::atoi(e); }
  if(myrank == 0){ std::cout << " precon_algo = " << (int)precon_algo << std::endl; }
  // 未知の値は止める (下の switch に該当が無く、黙って前処理なしで解くのを防ぐ)
  if(!parcel::precon_param::valid_precon_algo(precon_algo)){
    if(myrank == 0){
      std::cout << "[VERIFY] PARCEL_PRECON は 0 / 1 / 2 / 4 のどれか: " << (int)precon_algo << std::endl;
    }
#if PARCEL_MPI
    MPI_Abort(MPI_COMM_WORLD, 1);
#endif
    return 1;
  }
  // 行列・ベクトルのスケーリング。
  //   何も指定しなければ PARCEL が前処理の精度と種類から自動で決める
  //   (半精度なら有効、倍精度・単精度なら無効、ブロックヤコビは常に有効)。
  //   set_scaling_* を呼ぶとその指定が優先される。
  //   ここでは調査用に、環境変数が与えられたときだけ明示指定する。
  auto apply_scaling = [&](){
    const char *em = std::getenv("PARCEL_SCALING_MATRIX");
    const char *ev = std::getenv("PARCEL_SCALING_VECTOR");
    if(em){ kp->set_scaling_matrix(std::atoi(em) != 0); }
    if(ev){ kp->set_scaling_vector(std::atoi(ev) != 0); }
    if(myrank == 0){
      std::cout << " scaling_matrix = " << (em ? em : "auto")
                << " , scaling_vector = " << (ev ? ev : "auto") << std::endl;
    }
  };

  switch (precon_algo) {
  case parcel::PRECON_NONE:
    kp->set_precon_algo(parcel::PRECON_NONE);
    break;
  case parcel::PRECON_POINTJACOBI:
    kp->set_precon_algo(parcel::PRECON_POINTJACOBI);
    apply_scaling();
    break;
  case parcel::PRECON_BLOCKJACOBI:
    kp->set_precon_algo(parcel::PRECON_BLOCKJACOBI);
    apply_scaling();
    {
      // ブロック分割数。指定がなければ PARCEL がデバイスの並列度から決める
      // (CPU はランクあたりスレッド数、GPU は 1 ブロック 2 行)。
      //   カーネルの並列度がこの値に等しいので、合っていないとデバイスが
      //   埋まらない。細かくするほど前処理は弱くなるので、最適値は
      //   「1 反復の時間 x 反復回数」で決まる。
      //   調査用に -DBLOCKJACOBI_BLOCK と PARCEL_BJ_BLOCK で上書きできる。
      PARCEL_INT bjblock = 0;
      if(const char *e = std::getenv("PARCEL_BJ_BLOCK")){
        const long v = std::atol(e);
        if(v > 0){ bjblock = (PARCEL_INT)v; }
      }
#ifdef BLOCKJACOBI_BLOCK_EXPLICIT
      if(bjblock == 0 && !lib_defaults){ bjblock = BLOCKJACOBI_BLOCK; }
#endif
      if(bjblock > 0){ kp->set_blockjacobi(bjblock); }
    }
    break;
  case parcel::PRECON_JACOBI:
    kp->set_precon_algo(parcel::PRECON_JACOBI);
    apply_scaling();
    if(!lib_defaults) kp->set_jacobi(JACOBI_ITER);
    break;
  default:
    // 上の検査で弾いているので来ない
    break;
  }

  // 既定値モードで実際に使う値を 1 回だけ表示する (設定の後、ソルバの前)。
  //   値はすべてソルバの取得関数から読む (このサンプルの変数ではなく、ライブラリが持っている値)。
  //   (env) は環境変数で明示された値、(default) はライブラリの既定。
  if(lib_defaults && myrank == 0){
    auto src = [](bool env) -> const char * { return env ? "(env)" : "(default)"; };
    auto qr_name = [](PARCEL_INT a) -> const char * {
      if(a == parcel::qr::QR_CGS) return "CGS";
      if(a == parcel::qr::QR_MGS) return "MGS";
      if(a == parcel::qr::QR_CHOLESKYQR) return "CholeskyQR";
      if(a == parcel::qr::QR_CAGMRES_CHOLESKYQR) return "CAGMRES_CholeskyQR";
      return "?";
    };
    std::ostringstream ls;
    ls << std::scientific << std::setprecision(3);
    ls << "[LIB_DEFAULTS] solver = " << solver_names[(int)kind]
       << " , tol = " << kp->get_tolerance() << " " << src(tol_env)
       << " , max_iteration = " << (long long)kp->get_max_iteration() << " " << src(itrmax_env)
       << " , precon = " << (int)kp->get_precon_algo();
    if(kp->get_precon_algo() == parcel::PRECON_JACOBI){
      ls << " , jacobi_iter = " << (long long)kp->get_jacobi() << " (default)";
    }
    switch(kind){
#if PARCEL_ENABLE_PGMRES
    case SK_PGMRES:
      ls << " , restart = " << (long long)s_gmres.get_restart() << " " << src(!gmres_restart_default)
         << " , qr = " << qr_name(s_gmres.get_qr_algo()) << " (default)";
      break;
#endif
#if PARCEL_ENABLE_PCAGMRES
    case SK_PCAGMRES:
      ls << " , s = " << (long long)s_cagmres.get_restart_s() << " " << src(!cagmres_restart_default && cagmres_s_env)
         << " , t = " << (long long)s_cagmres.get_restart_t() << " " << src(!cagmres_restart_default && cagmres_t_env)
         << " , q = " << parcel::krylov::cagmres_q_name(s_cagmres.get_explicit_q()) << " " << src(q_env_outer)
         << " , qr = " << qr_name(s_cagmres.get_qr_algo()) << " " << src(moment_outer);
      break;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
    case SK_INVERSE_POWER:
      ls << " , linsolve = " << (s_invpow.get_linsolve_algo() == parcel::krylov::LINSOLVE_PCG ? "pcg" : "pgmres")
         << " " << src(lin_algo_env_outer)
         << " , linsolve_tol = " << s_invpow.get_linsolve_tolerance() << " " << src(inner_tol_env)
         << " , linsolve_max = " << (long long)s_invpow.get_linsolve_max_iteration() << " " << src(inner_max_env)
         << " , linsolve_restart = " << (long long)s_invpow.get_linsolve_restart() << " (default)"
         << " , linsolve_precon = " << (int)s_invpow.get_linsolve_precon_param()->get_precon_algo()
         << " " << src(lin_precon_env_outer)
         << " , linsolve_jacobi_iter = " << (long long)s_invpow.get_linsolve_precon_param()->get_jacobi() << " (default)";
      break;
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
    case SK_KS_ARNOLDI:
      ls << " , ks_m = " << (long long)s_ks.get_ks_m() << " " << src(ks_m_env)
         << " , ks_k_keep = " << (long long)s_ks.get_ks_k_keep() << " " << src(ks_keep_env)
         << " , ks_nev = " << (long long)s_ks.get_ks_nev() << " " << src(ks_nev_env)
         << " , ks_max_restart = " << (long long)s_ks.get_ks_max_restart() << " " << src(!ks_restart_default)
         << " , qr = " << qr_name(s_ks.get_qr_algo()) << " " << src(ks_qr_env_outer);
      //   内部ソルバは shift-invert のときだけ使う
      if(ks_runtime_shift
#ifdef KS_SHIFT_INVERT
         || true
#endif
         ){
        ls << " , linsolve_tol = " << s_ks.get_linsolve_tolerance() << " " << src(inner_tol_env)
           << " , linsolve_max = " << (long long)s_ks.get_linsolve_max_iteration() << " " << src(inner_max_env)
           << " , linsolve_restart = " << (long long)s_ks.get_linsolve_restart() << " (default)";
      }
      break;
#endif
    default: break;
    }
    std::cout << ls.str() << std::endl;
  }

  // 前処理の対称性検査。
  //
  //   pcg は前処理 M が対称であることを前提にしている。M が非対称だと
  //   共役性が崩れて停滞・発散する。半精度で行列を保存すると、ILU 分解の
  //   L と U を別々の演算列で作るため丸めで対称性が崩れうる。
  //
  //   検査は随伴の定義そのもの:  <M^-1 u, v> == <u, M^-1 v>
  //   乱数ベクトル u, v に前処理を適用し、両辺の差を ||u|| ||v|| で正規化する。
  //   対称なら丸め誤差程度 (倍精度で 1e-16、半精度なら 1e-3) に収まる。
  //   これを超えるなら M は対称でない。
  if(const char *e = std::getenv("PARCEL_PRECON_SYMCHECK")){
    if(std::atoi(e) != 0){
      parcel::precon_matrix<
	T_DATA, T_DATA, T_PRECON_MAT, T_PRECON_VEC, T_CALC_PRECON
	> precon_Mat(parcel_matrix, *kp);   // krylov_param は precon_param を継承

      parcel::vector<T_DATA> u_v, v_v, Mu_v, Mv_v;
      u_v.resize(nblock, &q); v_v.resize(nblock, &q);
      Mu_v.resize(nblock, &q); Mv_v.resize(nblock, &q);

      std::vector<double> u_h(nblock), v_h(nblock);
      std::srand(12345);
      for(PARCEL_INT i = 0; i < nblock; i++){
	u_h[i] = 2.0*((double)std::rand()/RAND_MAX) - 1.0;
	v_h[i] = 2.0*((double)std::rand()/RAND_MAX) - 1.0;
      }
      u_v.setValue(u_h.data()); v_v.setValue(v_h.data());
      q.wait();
      precon_Mat.solver(u_v, Mu_v);
      precon_Mat.solver(v_v, Mv_v);
      q.wait();

      std::vector<double> Mu_h(nblock), Mv_h(nblock);
      Mu_v.getValue(Mu_h.data()); Mv_v.getValue(Mv_h.data());
      q.wait();

      double a = 0.0, b = 0.0, nu = 0.0, nv = 0.0;
      for(PARCEL_INT i = 0; i < nblock; i++){
	a  += Mu_h[i]*v_h[i];
	b  += u_h[i]*Mv_h[i];
	nu += u_h[i]*u_h[i];
	nv += v_h[i]*v_h[i];
      }
#if PARCEL_MPI
      double tmp[4] = {a,b,nu,nv}, sum[4];
      MPI_Allreduce(tmp, sum, 4, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
      a = sum[0]; b = sum[1]; nu = sum[2]; nv = sum[3];
#endif
      const double den = std::sqrt(nu)*std::sqrt(nv);
      if(myrank == 0){
	std::cout<<"=== PRECON SYMCHECK ==="<<std::endl;
	std::cout<<"  <M^-1 u, v> = "<<std::scientific<<std::setprecision(14)<<a<<std::endl;
	std::cout<<"  <u, M^-1 v> = "<<std::scientific<<std::setprecision(14)<<b<<std::endl;
	std::cout<<"  非対称度 |差| / (||u|| ||v||) = "
		 <<std::scientific<<std::setprecision(6)<<std::abs(a-b)/den<<std::endl;
      }
    }
  }

  for(int i = 0;i<1;i++){
    vecx.setValue(x_cpu_local.data());
    q.wait();
    // ソルバごとに solver() のテンプレート引数と引数順が違うので振り分ける。
    //   固有値ソルバ : (行列, 固有値, 固有ベクトル)
    //   線形ソルバ   : (行列, 解, 右辺)
    using DT  = T_DATA;
    using PMT = T_PRECON_MAT;
    using PVT = T_PRECON_VEC;
    switch(kind){
#if PARCEL_ENABLE_KS_ARNOLDI
    case SK_KS_ARNOLDI:
#ifdef KS_SHIFT_INVERT
      s_ks.template solver<DT,DT,DT>(parcel_matrix, vecb, vecx, &parcel_matrix_shifted);
#else
      s_ks.template solver<DT,DT,DT>(parcel_matrix, vecb, vecx);
#endif
      break;
#endif
#if PARCEL_ENABLE_POWER
    case SK_POWER:
      s_power.template solver<DT,DT,DT,DT,PMT,PVT>(parcel_matrix, vecb, vecx);   break;
#endif
#if PARCEL_ENABLE_INVERSE_POWER
    case SK_INVERSE_POWER:
      s_invpow.template solver<DT,DT,DT,DT,PMT,PVT>(parcel_matrix, vecb, vecx);  break;
#endif
#if PARCEL_ENABLE_PCG
    case SK_PCG:
      s_pcg.template solver<DT,DT,DT,DT,PMT,PVT>(parcel_matrix, vecx, vecb);     break;
#endif
#if PARCEL_ENABLE_PBICGSTAB
    case SK_PBICGSTAB:
      s_bicg.template solver<DT,DT,DT,DT,PMT,PVT>(parcel_matrix, vecx, vecb);    break;
#endif
#if PARCEL_ENABLE_PGMRES
    case SK_PGMRES:
      s_gmres.template solver<DT,DT,DT,DT,PMT,PVT,DT,DT>(parcel_matrix, vecx, vecb); break;
#endif
#if PARCEL_ENABLE_PCAGMRES
    case SK_PCAGMRES:
      s_cagmres.template solver<DT,DT,DT,DT,PMT,PVT,DT,
                       T_QR,
                       T_QR>(parcel_matrix, vecx, vecb); break;
#endif
    default: break;
    }
    q.wait();
    // 収束状態はライブラリが 1 行書く (converged_reason = ...)。未収束のときだけ印を足す。
    //   "VERIFY" を含めないこと: 回帰試験は VERIFY の有無で NO_VERIFY_OUTPUT を判定する。
    //   終了コードと回帰試験の判定 (status.txt) はこの印では変わらない。
    if(kp != nullptr && kp->get_converged_reason() <= 0 && myrank == 0){
      std::cout << "[PARCEL] not converged: converged_reason = " << kp->get_converged_reason() << std::endl;
    }
  }

  //   ホスト側のピーク常駐メモリ。CPU バックエンドでは PARCEL の「デバイス」
  //   メモリもホストメモリなので、これが実質的な総使用量になる。
  //   PARCEL_MEM_REPORT=1 のときだけ出力。
  diag::report_host_peak("solver 後", myrank);
  //   GPU のデバイス使用量も同じ場所で 1 回だけ読む (nvidia-smi 経由、PARCEL_MEM_REPORT のときだけ)。
  //   計測区間の外なので時間には影響しない。CPU 環境では何も出ない。
  diag::report_device_mem("solver 後", myrank);
  //   PARCEL が確保したデバイスメモリのピーク (util_sycl の記録)。ソルバの作業域を含む。
  //   nvidia-smi の値はソルバの後 (作業域の解放後) で、装置によっては取れないため、装置間の比較にはこちらを使う。
  diag::report_parcel_device_peak("solver 後", myrank);
#if PARCEL_MPI
  //   全ランクの値を rank 0 に集めて 1 行で書く (mem_gather.txt に追記)。
  //   Open MPI の mpirun は -x を付けない限り環境変数を他ノードのランクへ転送しないので、
  //   2 ノード目以降では PARCEL_MEM_REPORT が未設定になり、ランク別ファイルが欠ける。
  //   有効かどうかは rank 0 の環境で決めて全ランクに配る。ランクごとに getenv で判定すると、
  //   rank 0 だけが MPI_Gather に入って他ランクを待ち続ける。
  //   計測区間の外。
  int mem_report_on = std::getenv("PARCEL_MEM_REPORT") ? 1 : 0;
  MPI_Bcast(&mem_report_on, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(mem_report_on){
    int np_ = 1; MPI_Comm_size(MPI_COMM_WORLD, &np_);
    //   host_hugetlb_mib: 大きいページ (HugeTLB) の量 (diag::host_hugetlb_kb)。A64FX ではヒープがここに載り host_mib (VmHWM) に
    //   入らないので、ホストのメモリは host_mib + host_hugetlb_mib で見る (こちらはピークではなく、この時点の値)。他の環境では 0。
    long mine[4] = { diag::host_peak_kb() / 1024, diag::device_mem_used_mib_now(),
                     (long)(util_sycl::device_mem_peak_bytes() >> 20), diag::host_hugetlb_kb() / 1024 };
    std::vector<long> all(4 * (size_t)np_, 0);
    MPI_Gather(mine, 4, MPI_LONG, all.data(), 4, MPI_LONG, 0, MPI_COMM_WORLD);
    if(myrank == 0){
      std::string h = "[MEMG] nranks=" + std::to_string(np_) + " host_mib=";
      std::string d = " dev_mib=";
      std::string k = " parcel_devpeak_mib=";
      std::string g = " host_hugetlb_mib=";
      for(int r = 0; r < np_; r++){
        h += (r ? "," : "") + std::to_string(all[4*r]);
        d += (r ? "," : "") + std::to_string(all[4*r+1]);
        k += (r ? "," : "") + std::to_string(all[4*r+2]);
        g += (r ? "," : "") + std::to_string(all[4*r+3]);
      }
      FILE *f = std::fopen("mem_gather.txt", "a");
      if(f){ std::fprintf(f, "%s%s%s%s\n", h.c_str(), d.c_str(), k.c_str(), g.c_str()); std::fclose(f); }
      std::cout << h << d << k << g << std::endl;
    }
  }
#endif
  vecx.getValue(x_cpu_local.data());  q.wait();
  if(is_eigen){
  b_cpu_local.resize(vecb.size());
  vecb.getValue(b_cpu_local.data());  q.wait();
  }
  q.wait();

  //   非構造行列は解析固有値を持たないので、固有対の照合は行わない (時間と残差は通常どおり出る)
  if(do_verify && is_eigen && matrix_obj.has_analytic_eigen()){
  // ===== Eigenvalue verify (power / ks_arnoldi 共通) =====
  //   全 Ritz pair を rank 0 に集約 → CPU CRS で SpMV → 残差 + 解析値比較
  //   複素共役対 (非対称行列で発生) にも対応:
  //     v = vr + i*vi, λ = lr + i*li
  //     A v - λ v = (A vr - lr*vr + li*vi) + i*(A vi - lr*vi - li*vr)
  //     ||v||² = ||vr||² + ||vi||²
  //   dgeev は共役対を隣接位置 p, p+1 に格納 (ks_arnoldi.hpp で stable_sort 済)
  //   なので EigenVec[:, j] が Re(v), EigenVec[:, j+1] が Im(v) となる
  //   ks_arnoldi が収束判定しているのは上位 nev 個だけ。残りの k_keep - nev 個は
  //   リスタート用に保持されているだけで収束保証がないため、誤差の集計対象にしない。
  //   受け取った固有ベクトルの本数 (ks_ev_cols) までを扱う
  const int n_eigenpairs = is_ks ? ks_ev_cols : 1;
  const int n_verify     = is_ks ? ((ks_nev < ks_keep) ? ks_nev : ks_keep) : 1;

  // gather setup
  std::vector<int> ev_recv(nprocess, (int)nblock), ev_disp(nprocess, 0);
#if PARCEL_MPI
  int ev_my_count = (int)nblock;
  MPI_Gather(&ev_my_count, 1, MPI_INT, ev_recv.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(myrank == 0){
    ev_disp[0] = 0;
    for(int i = 1; i < nprocess; i++) ev_disp[i] = ev_disp[i-1] + ev_recv[i-1];
  }
  MPI_Datatype ev_mpi_type = (sizeof(problem_type) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
#endif

  // 各固有ベクトルを rank 0 に集約 (n_eigenpairs 個の長さ n_global ベクトル)
  std::vector<std::vector<problem_type> > v_globals(n_eigenpairs);
  for(int j = 0; j < n_eigenpairs; j++){
    if(myrank == 0) v_globals[j].assign(n_global, 0.0);
#if PARCEL_MPI
    MPI_Gatherv(x_cpu_local.data() + (long long)j*nblock, (int)nblock, ev_mpi_type,
                myrank==0 ? v_globals[j].data() : nullptr,
                ev_recv.data(), ev_disp.data(),
                ev_mpi_type, 0, MPI_COMM_WORLD);
#else
    for(PARCEL_INT i = 0; i < nblock; i++) v_globals[j][i] = x_cpu_local[(size_t)j*nblock + i];
#endif
  }

  // 固有値配列 (Re, Im) を抽出
  //   ks_arnoldi: b_cpu_local が (Re,Im) 交互
  //   power     : b_cpu_local[0] が実固有値 (Im は 0)
  std::vector<double> lam_re(n_eigenpairs, 0.0), lam_im(n_eigenpairs, 0.0);
    if(is_ks){
  for(int j = 0; j < n_eigenpairs; j++){
    lam_re[j] = b_cpu_local[j*2];
    lam_im[j] = b_cpu_local[j*2+1];
  }
    }else{
  lam_re[0] = b_cpu_local[0];
    }

  // 解析固有値 (複素対応)
  //   Poisson: 全て実数 → ana_im[*] = 0
  //   convdiff: c=2 で純虚数 sqrt → ana_im 非ゼロ (複素共役対)
  // ===== 前処理付き固有値ソルバの検証基準 =====
  //   PARCEL_PRECON_SPECTRUM=1 のとき, 固有値ソルバは A ではなく A M^-1 の固有値を返す.
  //   点ヤコビ (M = d*I, d は一定の対角値) のときに限り λ(A M^-1) = λ(A)/d と
  //   厳密に対応するので, 解析値を 1/d 倍して比較する.
  //   ブロックヤコビ / ヤコビでは解析的な基準が無いため比較・残差評価を行わない.
  //   未指定 (0) のときは A の固有値を返すので調整は不要
  //   (inverse_power / shift-invert の内部ソルバの前処理は PARCEL_LINSOLVE_PRECON で別に指定する).
  double precon_scale = 1.0;   // λ(A M^-1) = precon_scale * λ(A)
  bool   ana_compare  = true;  // 解析値との比較と残差評価を行うか
// 固有値ソルバのときだけ、前処理による固有値の変化を扱う。
//
//   ソルバはコンパイル時マクロ (power_solver / ks_arnoldi_solver) ではなく
//   実行時の PARCEL_SOLVER で選ぶので、#if で判定してはいけない。
//   1 バイナリで 7 ソルバを回すときはこれらのマクロは未定義であり、
//   #if にすると中身が丸ごと消えて、前処理付きの固有値を A の解析値と比較してしまう。
//
//   KS_SHIFT_INVERT は本物のコンパイル時オプション (別の演算子を解く) なので
//   こちらは #ifndef のまま残す。
  //   A M^-1 の固有値を返すのは PARCEL_PRECON_SPECTRUM=1 のときの固有値ソルバすべて。
#ifdef KS_SHIFT_INVERT
  const bool   ks_is_shift_invert = true;
  const double ks_sigma_used      = ks_sigma;
#else
  const bool   ks_is_shift_invert = ks_runtime_shift;
  const double ks_sigma_used      = ks_runtime_sigma;
#endif
  //   A M^-1 の固有値を返すかは PARCEL_PRECON_SPECTRUM だけで決まる (全固有値ソルバ共通)。
  (void)ks_is_shift_invert;
  const bool returns_precon_spectrum = precon_spectrum &&
      (kind == SK_POWER || kind == SK_KS_ARNOLDI || kind == SK_INVERSE_POWER);
  if(returns_precon_spectrum && precon_algo != 0){
    bool   diag_is_const = true;
    double diag_const    = 0.0;
    PARCEL_INT *rp0   = crsMatrix_cpu.crsRowptr_data();
    PARCEL_INT *cl0   = crsMatrix_cpu.crsCol_data();
    problem_type *aa0 = crsMatrix_cpu.A_data();
    for(PARCEL_INT i = 0; i < n_global; i++){
      double d_i = 0.0; bool found = false;
      for(PARCEL_INT q = rp0[i]; q < rp0[i+1]; q++){
        if(cl0[q] == i){ d_i = (double)aa0[q]; found = true; break; }
      }
      if(!found){ diag_is_const = false; break; }
      if(i == 0){ diag_const = d_i; }
      else if(std::fabs(d_i - diag_const) > 1.0e-12*std::fabs(diag_const)){ diag_is_const = false; break; }
    }
    if(precon_algo == 1){
      // 点ヤコビは M^-1 = D^-1。対角が一定値 d なら λ(A M^-1) = λ(A)/d
      if(diag_is_const && std::fabs(diag_const) > 0.0){ precon_scale = 1.0 / diag_const; }
      else                                            { ana_compare  = false;            }
    }else{
      ana_compare = false;
    }
  }

  // ===== 前処理付き作用素 A M^-1 の参照解 (密行列 + LAPACK dgeev) =====
  //   ブロックヤコビ / ヤコビでは λ(A M^-1) の解析式が無い。スケーリングを含む
  //   前処理そのものを単位ベクトルに適用して密行列 B = A M^-1 を組み、dgeev で
  //   参照固有値を求める。PARCEL の前処理実装をそのまま使うので、CPU 側で
  //   前処理を書き直した場合のような実装差が検証をすり抜けることがない。
  //   n^2 のメモリと n 回の前処理適用が要るため、小規模時のみ有効にする。
#ifndef VERIFY_DENSE_REF_MAX_N
#define VERIFY_DENSE_REF_MAX_N 1024
#endif
  std::vector<double> dense_B;    // rank 0 のみ: n x n 列優先
  std::vector<double> ref_re_dense, ref_im_dense;
  bool use_dense_ref = false;
  if(returns_precon_spectrum
     && precon_algo != 0 && !ana_compare && n_global <= (VERIFY_DENSE_REF_MAX_N)){
    if(myrank == 0){
      std::cout << "\n[VERIFY] A M^-1 の参照解を密行列 + dgeev で作成 (n = "
                << n_global << ")" << std::endl;
      dense_B.assign((size_t)n_global*(size_t)n_global, 0.0);
    }
    parcel::precon_matrix<
      T_DATA,
      T_DATA,
      T_PRECON_MAT,
      T_PRECON_VEC,
      T_CALC_PRECON
      > ref_precon(parcel_matrix, *(kp->get_precon_param()));
#if PARCEL_MPI
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> ref_blas(&q, comm.mpi_comm());
    PARCEL_INT ref_nadd = parcel_matrix.get_recv_size();
#else
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> ref_blas(&q);
    PARCEL_INT ref_nadd = 0;
#endif
    parcel::parallel_vector<T_DATA> ref_r(nblock, ref_nadd, 1, &q);
    parcel::parallel_vector<T_DATA> ref_z(nblock, ref_nadd, 1, &q);
    parcel::vector<T_DATA> ref_y(nblock, &q);
    parcel::vector<T_CALC> ref_one(1, &q);
    parcel::vector<T_CALC> ref_zero(1, &q);
    { double v = 1.0; ref_one.setValue(&v); }
    { double v = 0.0; ref_zero.setValue(&v); }

    long long   ref_bad_precon = 0;
    PARCEL_INT  ref_bad_precon_first_col = -1, ref_bad_precon_first_row = -1;
    std::vector<problem_type> ref_zero_buf((size_t)(nblock + ref_nadd), 0.0);
    std::vector<problem_type> ref_col_local(nblock, 0.0);
    std::vector<problem_type> ref_col_global(n_global, 0.0);
#if PARCEL_MPI
    std::vector<int> ref_cnt(nprocess, 0), ref_disp(nprocess, 0);
    int ref_my = (int)nblock;
    MPI_Allgather(&ref_my, 1, MPI_INT, ref_cnt.data(), 1, MPI_INT, MPI_COMM_WORLD);
    for(int i = 1; i < nprocess; i++) ref_disp[i] = ref_disp[i-1] + ref_cnt[i-1];
    MPI_Datatype ref_type = (sizeof(problem_type) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
#endif

    for(PARCEL_INT jcol = 0; jcol < n_global; jcol++){
      // r = e_j (袖も含めてゼロ初期化してから該当成分だけ 1)
      ref_r.vec_add(0).setValue(ref_zero_buf.data());
      for(PARCEL_INT i = 0; i < nblock; i++) ref_col_local[i] = 0.0;
      if(nstart <= jcol && jcol <= nend) ref_col_local[jcol - nstart] = 1.0;
      ref_r.vec(0).setValue(ref_col_local.data());
      ref_z.vec_add(0).setValue(ref_zero_buf.data());
      q.wait();
      // z = M^-1 r
      ref_precon.solver(ref_r.vec_add(0), ref_z.vec_add(0));
      q.wait();
      // 前処理の出力が有限か検査する (壊れているのが前処理か SpMV かを切り分ける)
      ref_z.vec(0).getValue(ref_col_local.data());
      q.wait();
      for(PARCEL_INT i = 0; i < nblock; i++){
        if(!std::isfinite((double)ref_col_local[i])){
          ref_bad_precon++;
          if(ref_bad_precon_first_col < 0){
            ref_bad_precon_first_col = jcol;
            ref_bad_precon_first_row = nstart + i;
          }
        }
      }
      // y = A z
      //   SpMV は y = alpha*A*x + beta*y を計算する。beta=0 でも y が未初期化のままだと
      //   0 * NaN = NaN となり結果が壊れるため、毎回ゼロで初期化する。
      ref_y.setValue(ref_zero_buf.data());
      q.wait();
      ref_blas.SpMV<
        T_CALC, T_CALC,
        T_DATA, T_DATA,
        T_DATA, T_CALC
        >(ref_one, ref_zero, parcel_matrix, ref_z.vec_add(0), ref_y);
      q.wait();
      ref_y.getValue(ref_col_local.data());
      q.wait();
#if PARCEL_MPI
      MPI_Gatherv(ref_col_local.data(), (int)nblock, ref_type,
                  ref_col_global.data(), ref_cnt.data(), ref_disp.data(), ref_type,
                  0, MPI_COMM_WORLD);
#else
      for(PARCEL_INT i = 0; i < nblock; i++) ref_col_global[i] = ref_col_local[i];
#endif
      if(myrank == 0){
        for(PARCEL_INT i = 0; i < n_global; i++){
          dense_B[(size_t)i + (size_t)jcol*(size_t)n_global] = (double)ref_col_global[i];
        }
      }
    }

#if PARCEL_MPI
    {
      long long bad_sum = 0;
      MPI_Reduce(&ref_bad_precon, &bad_sum, 1, MPI_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
      if(myrank == 0) ref_bad_precon = bad_sum;
    }
#endif
    if(myrank == 0 && ref_bad_precon > 0){
      std::cout << "[VERIFY] 前処理 M^-1 の出力に非有限値が " << ref_bad_precon
                << " 個 (全 " << (long long)n_global*(long long)n_global
                << " 成分中)" << std::endl;
    }
    if(ref_bad_precon_first_col >= 0){
      std::cout << "[VERIFY] rank " << myrank << ": 最初の非有限は列 "
                << ref_bad_precon_first_col << " 行 " << ref_bad_precon_first_row << std::endl;
    }
    if(myrank == 0){
      // dgeev に渡す前に非有限値を検査する。
      //   LAPACKE は入力の NaN 検査で info = -5 を返すだけなので、
      //   どこが壊れているか分かるようにしておく。
      size_t bad = 0; PARCEL_INT bad_i = -1, bad_j = -1;
      for(PARCEL_INT jj = 0; jj < n_global; jj++){
        for(PARCEL_INT ii = 0; ii < n_global; ii++){
          const double v = dense_B[(size_t)ii + (size_t)jj*(size_t)n_global];
          if(!std::isfinite(v)){
            if(bad == 0){ bad_i = ii; bad_j = jj; }
            bad++;
          }
        }
      }
      if(bad > 0){
        std::cout << "[VERIFY] 密行列 A M^-1 に非有限値が " << bad << " 個 (全 "
                  << (size_t)n_global*(size_t)n_global << " 要素中), 最初は ("
                  << bad_i << "," << bad_j << ")" << std::endl;
        std::cout << "[VERIFY] 前処理の適用結果が有限でないため参照解を作成できない" << std::endl;
      }
      // dgeev は入力を破壊するのでコピーを渡す
      std::vector<double> Bcopy(dense_B);
      std::vector<double> wr(n_global), wi(n_global);
      int info = (bad > 0) ? -1 :
                 LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'N',
                               (int)n_global, Bcopy.data(), (int)n_global,
                               wr.data(), wi.data(),
                               NULL, 1, NULL, 1);
      if(info != 0){
        std::cout << "[VERIFY] dgeev info = " << info
                  << " -> 参照解を作成できないため比較を行わない" << std::endl;
      }else{
        ref_re_dense.assign(wr.begin(), wr.end());
        ref_im_dense.assign(wi.begin(), wi.end());
        use_dense_ref = true;

        // 参照スペクトルをファイルに残す。
        //   ブロックヤコビは MPI 分割ごとにブロックが変わるため A M^-1 の
        //   固有値そのものが並列数に依存する。並列数ごとに保存しておく。
        {
          std::ostringstream fn;
          fn << "ref_spectrum_precon" << (int)(precon_algo)
             << "_np" << nprocess << ".txt";
          std::ofstream ofs(fn.str().c_str());
          if(ofs){
            ofs << "# reference spectrum of A M^-1 (dense, LAPACK dgeev)\n";
            ofs << "# NX = " << nx << "  n = " << n_global << "\n";
            ofs << "# parcel_solver_precon_algo = " << (int)(precon_algo) << "\n";
            ofs << "# BLOCKJACOBI_BLOCK = " << (int)(BLOCKJACOBI_BLOCK)
                << "  JACOBI_ITER = " << (int)(JACOBI_ITER) << "\n";
            ofs << "# nprocess = " << nprocess << "\n";
            ofs << "# index  Re(lambda)  Im(lambda)\n";
            ofs << std::scientific << std::setprecision(15);
            for(PARCEL_INT k = 0; k < n_global; k++){
              ofs << k << " " << wr[k] << " " << wi[k] << "\n";
            }
            std::cout << "[VERIFY] 参照スペクトルを " << fn.str() << " に出力した" << std::endl;
          }
        }
      }
    }
#if PARCEL_MPI
    { int flag = use_dense_ref ? 1 : 0;
      MPI_Bcast(&flag, 1, MPI_INT, 0, MPI_COMM_WORLD);
      use_dense_ref = (flag != 0); }
#endif
    if(use_dense_ref){ ana_compare = true; }
  }

  //   power / ks_arnoldi は |λ| 最大側, inverse_power は |λ| 最小側と突き合わせる
  std::vector<double> ana_re, ana_im;
  if(myrank == 0){
      if(kind == SK_INVERSE_POWER){
    matrix_obj.bottom_eigenvalues_complex(nx, std::max(50, n_eigenpairs*5),
                                          ana_re, ana_im);
      }else if(is_ks && ks_is_shift_invert && n_global <= (VERIFY_SHIFT_FULL_MAX_N)){
    // shift-invert は σ 近傍の固有値を返すので、|λ| 上位だけのリストでは
    //   σ が内部や 0 付近のとき照合先が無い。全固有値を作り、下で σ に近い順に絞る。
    matrix_obj.top_eigenvalues_complex(nx, (int)n_global, ana_re, ana_im);
      }else{
    matrix_obj.top_eigenvalues_complex(nx, std::max(50, n_eigenpairs*5),
                                       ana_re, ana_im);
      }
    if(use_dense_ref){
      // 密行列から求めた A M^-1 の固有値を参照解にする
      ana_re = ref_re_dense;
      ana_im = ref_im_dense;
    }
    // 前処理付き (点ヤコビ) では λ(A M^-1) = λ(A)/d に合わせる.
    // 一様スケールなので |λ| の順序は変わらない.
    if(ana_compare && !use_dense_ref && precon_scale != 1.0){
      for(size_t k = 0; k < ana_re.size(); k++){
        ana_re[k] *= precon_scale;
        ana_im[k] *= precon_scale;
      }
    }
    // shift-invert の参照は σ に近い順に並べ、先頭 max(50, 5*nev) 個に絞る
    //   (スケール後の値で測る。点ヤコビの A M^-1 モードでは σ はスケール後の固有値に対する値)。
    if(is_ks && ks_is_shift_invert && !ana_re.empty()){
      std::vector<size_t> ord(ana_re.size());
      for(size_t k = 0; k < ord.size(); k++) ord[k] = k;
      std::stable_sort(ord.begin(), ord.end(), [&](size_t a, size_t b){
        return std::hypot(ana_re[a] - ks_sigma_used, ana_im[a])
             < std::hypot(ana_re[b] - ks_sigma_used, ana_im[b]);
      });
      const size_t keep = std::min(ord.size(), (size_t)std::max(50, n_eigenpairs*5));
      std::vector<double> nr(keep), ni(keep);
      for(size_t k = 0; k < keep; k++){ nr[k] = ana_re[ord[k]]; ni[k] = ana_im[ord[k]]; }
      ana_re.swap(nr);
      ana_im.swap(ni);
      std::cout << "[VERIFY] shift-invert: sigma = " << ks_sigma_used
                << " に近い解析固有値 " << keep << " 個と照合" << std::endl;
    }
  }

  if(myrank == 0){
    std::cout << "\n=== VERIFY: "
              << (is_ks ? "ks_arnoldi eigenpairs"
                  : (kind == SK_INVERSE_POWER
                       ? "inverse_power eigenpair (|lambda| 最小)"
                       : "power eigenpair (|lambda| 最大)"))
              << " ===\n";
    if(matrix_obj.is_convdiff()){
      std::cout << "  matrix: 3D 7-pt convection-diffusion (非対称, c=" << matrix_obj.get_c() << ")\n";
      std::cout << "  analytical lam_{ijk} = -6 + 2*sqrt(1-c^2)*[cos(i*pi/(N+1)) + cos(j*pi/(N+1)) + cos(k*pi/(N+1))]\n";
    }else{
      std::cout << "  matrix: 3D 7-pt Poisson (実対称)\n";
      std::cout << "  analytical lam_{ijk} = -6 + 2cos(i*pi/(N+1)) + 2cos(j*pi/(N+1)) + 2cos(k*pi/(N+1))\n";
    }
    if(!ana_compare){
      // 前処理付き作用素 A M^-1 の固有値であり、A の解析固有値と比較しても意味がない
      std::cout << "  precon: parcel_solver_precon_algo=" << (int)(precon_algo)
                << " -> 返る固有値は A ではなく A M^-1 のもの\n";
      std::cout << "  この前処理には解析的な基準が無いため、解析値との比較と残差評価は行わない\n";
      std::cout << "  ------------------------------------------------\n";
      std::cout << std::right
                << std::setw(4)  << "j"
                << std::setw(18) << "Re(lam)"
                << std::setw(18) << "Im(lam)"
                << std::endl;
    }else{
      if(use_dense_ref){
        std::cout << "  precon: parcel_solver_precon_algo=" << (int)(precon_algo)
                  << " -> 返る固有値は A M^-1 のもの\n";
        std::cout << "  参照解: PARCEL の前処理を単位ベクトルに適用して密行列 A M^-1 を組み、\n";
        std::cout << "          LAPACK dgeev で求めた " << ana_re.size() << " 個の固有値と比較する\n";
      }else if(precon_scale != 1.0){
        std::cout << "  precon: 点ヤコビ (M = d*I) のため lam(A M^-1) = lam(A)/d\n";
        std::cout << "          解析値を " << std::scientific << std::setprecision(6) << precon_scale
                  << " 倍して比較する\n";
      }
        if(kind == SK_INVERSE_POWER){
      std::cout << "  matching: |lambda| 昇順の解析値リストから最近接 (距離 = sqrt((dRe)^2 + (dIm)^2))\n";
        }else{
      std::cout << "  matching: |lambda| 降順の解析値リストから最近接 (距離 = sqrt((dRe)^2 + (dIm)^2))\n";
        }
      std::cout << "  eig_err  = |lam_hat - lam_ana| / |lam_ana|   (複素距離)\n";
      if(use_dense_ref || precon_scale != 1.0){
        std::cout << "  residual = ||A M^-1 v - lam v|| / (|lam| * ||v||)   [complex case: vr,vi 同時]\n";
      }else{
        std::cout << "  residual = ||A v - lam v|| / (|lam| * ||v||)   [complex case: vr,vi 同時]\n";
      }
      std::cout << "  ----------------------------------------------------------------------------------------------------------------\n";
      std::cout << std::right
                << std::setw(4)  << "j"
                << std::setw(18) << "Re(lam)"
                << std::setw(18) << "Im(lam)"
                << std::setw(18) << "Re(lam_ana)"
                << std::setw(18) << "Im(lam_ana)"
                << std::setw(18) << "eig_err"
                << std::setw(18) << "rel_residual"
                << std::endl;
    }
  }

  //   max_* : 収束判定対象 (上位 n_verify 個) の最大値 = 回帰試験が見る指標
  //   tail_*: 収束判定対象外 (j >= n_verify) の最大値。参考値
  double max_eig_err = 0.0, max_res = 0.0;
  // NaN/Inf は if(a > b) を必ず false にするため、素朴な最大値計算では
  // 「誤差 0」として素通りする。非有限値は別に数える。
  int n_eig_nonfinite = 0;
  double tail_eig_err = 0.0, tail_res = 0.0;
  bool   has_tail = false;
  if(myrank == 0){
    PARCEL_INT *rp = crsMatrix_cpu.crsRowptr_data();
    PARCEL_INT *cl = crsMatrix_cpu.crsCol_data();
    problem_type *aa = crsMatrix_cpu.A_data();
    int j = 0;
    while(j < n_eigenpairs){
      double lr = lam_re[j];
      double li = lam_im[j];
      double lmag = std::sqrt(lr*lr + li*li);

      // 共役対判定: Im(λ_j) != 0 かつ λ_{j+1} ≈ conjugate
      bool is_complex_pair = false;
      if(std::fabs(li) > 0.0 && j+1 < n_eigenpairs){
        double sum_im = lam_im[j] + lam_im[j+1];
        double diff_re = std::fabs(lam_re[j] - lam_re[j+1]);
        if(std::fabs(sum_im) < 1e-12*std::max(1.0,lmag) && diff_re < 1e-12*std::max(1.0,lmag)){
          is_complex_pair = true;
        }
      }

      if(!ana_compare){
        // 解析基準が無い前処理では計算された固有値のみ出力する
        std::cout << std::right
                  << std::setw(4) << j
                  << std::scientific << std::setprecision(6)
                  << std::setw(18) << lr
                  << std::setw(18) << li
                  << std::endl;
        if(is_complex_pair){
          std::cout << std::right
                    << std::setw(4) << (j+1)
                    << std::scientific << std::setprecision(6)
                    << std::setw(18) << lam_re[j+1]
                    << std::setw(18) << lam_im[j+1]
                    << std::endl;
          j += 2;
        }else{
          j += 1;
        }
        continue;
      }

      // 最近接解析値 (複素距離)
      //   computed (lr, li) に対し ana_top の各 (ana_re[k], ana_im[k]) から
      //   sqrt((dRe)^2 + (dIm)^2) を最小化する分点を選ぶ
      auto nearest_complex = [&](double xr, double xi, double &out_re, double &out_im){
        double best_re = ana_re[0], best_im = ana_im[0];
        double best_d  = std::hypot(xr - best_re, xi - best_im);
        for(size_t k = 1; k < ana_re.size(); k++){
          double d = std::hypot(xr - ana_re[k], xi - ana_im[k]);
          if(d < best_d){ best_d = d; best_re = ana_re[k]; best_im = ana_im[k]; }
        }
        out_re = best_re;
        out_im = best_im;
      };

      // 残差計算
      double res = 0.0;
      if(!is_complex_pair){
        auto &vr = v_globals[j];
        double r2 = 0.0, v2 = 0.0;
        for(PARCEL_INT i = 0; i < n_global; i++){
          double Av_i = 0.0;
          if(use_dense_ref){
            // 密行列 B = A M^-1 を直接使う
            for(PARCEL_INT jj = 0; jj < n_global; jj++)
              Av_i += dense_B[(size_t)i + (size_t)jj*(size_t)n_global] * vr[jj];
          }else{
            for(PARCEL_INT p = rp[i]; p < rp[i+1]; p++) Av_i += aa[p] * vr[cl[p]];
            Av_i *= precon_scale;
          }
          double ri = Av_i - lr * vr[i];
          r2 += ri * ri;
          v2 += vr[i] * vr[i];
        }
        res = (lmag > 0.0 && v2 > 0.0) ? std::sqrt(r2) / (lmag * std::sqrt(v2)) : 0.0;
      } else {
        auto &vr = v_globals[j];
        auto &vi = v_globals[j+1];
        double r2re = 0.0, r2im = 0.0, vr2 = 0.0, vi2 = 0.0;
        for(PARCEL_INT i = 0; i < n_global; i++){
          double Avr_i = 0.0, Avi_i = 0.0;
          if(use_dense_ref){
            for(PARCEL_INT jj = 0; jj < n_global; jj++){
              const double bij = dense_B[(size_t)i + (size_t)jj*(size_t)n_global];
              Avr_i += bij * vr[jj];
              Avi_i += bij * vi[jj];
            }
          }else{
            for(PARCEL_INT p = rp[i]; p < rp[i+1]; p++){
              Avr_i += aa[p] * vr[cl[p]];
              Avi_i += aa[p] * vi[cl[p]];
            }
            Avr_i *= precon_scale;
            Avi_i *= precon_scale;
          }
          double rre = Avr_i - lr * vr[i] + li * vi[i];
          double rim = Avi_i - lr * vi[i] - li * vr[i];
          r2re += rre*rre;
          r2im += rim*rim;
          vr2  += vr[i]*vr[i];
          vi2  += vi[i]*vi[i];
        }
        double rnorm = std::sqrt(r2re + r2im);
        double vnorm = std::sqrt(vr2 + vi2);
        res = (lmag > 0.0 && vnorm > 0.0) ? rnorm / (lmag * vnorm) : 0.0;
      }

      // 行出力: j (λ)
      double lam_ana_re, lam_ana_im;
      nearest_complex(lr, li, lam_ana_re, lam_ana_im);
      double lam_ana_mag = std::hypot(lam_ana_re, lam_ana_im);
      double eig_err = std::hypot(lr - lam_ana_re, li - lam_ana_im)
                       / std::max(1e-300, lam_ana_mag);
      std::cout << std::right
                << std::setw(4) << j
                << std::scientific << std::setprecision(6)
                << std::setw(18) << lr
                << std::setw(18) << li
                << std::setw(18) << lam_ana_re
                << std::setw(18) << lam_ana_im
                << std::setw(18) << eig_err
                << std::setw(18) << res
                << std::endl;
      if(!std::isfinite(lr) || !std::isfinite(li) ||
         !std::isfinite(eig_err) || !std::isfinite(res)) n_eig_nonfinite++;
      if(j < n_verify){
        if(eig_err > max_eig_err) max_eig_err = eig_err;
        if(res     > max_res    ) max_res     = res;
      }else{
        has_tail = true;
        if(eig_err > tail_eig_err) tail_eig_err = eig_err;
        if(res     > tail_res    ) tail_res     = res;
      }

      // 共役 λ̄ 行 (残差は同じ; 解析値も別途 nearest)
      if(is_complex_pair){
        double lr2 = lam_re[j+1];
        double li2 = lam_im[j+1];
        double lam_ana_re2, lam_ana_im2;
        nearest_complex(lr2, li2, lam_ana_re2, lam_ana_im2);
        double lam_ana_mag2 = std::hypot(lam_ana_re2, lam_ana_im2);
        double eig_err2 = std::hypot(lr2 - lam_ana_re2, li2 - lam_ana_im2)
                          / std::max(1e-300, lam_ana_mag2);
        std::cout << std::right
                  << std::setw(4) << (j+1)
                  << std::scientific << std::setprecision(6)
                  << std::setw(18) << lr2
                  << std::setw(18) << li2
                  << std::setw(18) << lam_ana_re2
                  << std::setw(18) << lam_ana_im2
                  << std::setw(18) << eig_err2
                  << std::setw(18) << res
                  << std::endl;
        if(j < n_verify){
          if(eig_err2 > max_eig_err) max_eig_err = eig_err2;
        }else{
          has_tail = true;
          if(eig_err2 > tail_eig_err) tail_eig_err = eig_err2;
        }
        j += 2;
      } else {
        j += 1;
      }
    }
    if(ana_compare){
      std::cout << "  -----------------------------------------------------------------------------------------------\n";
        if(is_ks){
      std::cout << "  収束判定対象 = 上位 " << n_verify << " 個 (KS_NEV)"
                << " / 出力 " << n_eigenpairs << " 個 (KS_K_KEEP=" << ks_keep << ")\n";
        }else{
      std::cout << "  収束判定対象 = " << n_verify << " 個 (本ソルバは固有値 1 個を返す)\n";
        }
      if(n_eig_nonfinite > 0 || !std::isfinite(max_eig_err) || !std::isfinite(max_res)){
        std::cout << "  VERIFY_NONFINITE : 固有値・固有ベクトル残差に NaN/Inf が "
                  << n_eig_nonfinite << " 件\n";
      }
      std::cout << "  max relative eigenvalue err = " << std::scientific << max_eig_err
                << "   (j < " << n_verify << ")\n";
      std::cout << "  max relative residual       = " << std::scientific << max_res
                << "   (j < " << n_verify << ")\n";
      if(has_tail){
        std::cout << "  [参考] 収束判定対象外 (j >= " << n_verify << ") の最大: eig_err = "
                  << std::scientific << tail_eig_err << ", residual = "
                  << std::scientific << tail_res << "\n";
        std::cout << "         これらはリスタート用に保持している Ritz 値で収束保証がない\n";
      }
    }else{
      std::cout << "  ------------------------------------------------\n";
      std::cout << "  VERIFY_SKIPPED : preconditioned operator (A M^-1) のため解析比較なし\n";
    }
  }
  }else if(do_verify){
  // [VERIFY] rank 0 に解を集約して厳密解との誤差を出力 (線形ソルバー)
#if PARCEL_MPI
  std::vector<int> recvcounts(nprocess), displs(nprocess);
  int my_count = (int)nblock;
  MPI_Gather(&my_count, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, MPI_COMM_WORLD);
  std::vector<problem_type> x_global;
  if(myrank == 0){
    displs[0] = 0;
    for(int i = 1; i < nprocess; i++){
      displs[i] = displs[i-1] + recvcounts[i-1];
    }
    x_global.resize(n_global, 0.0);
  }
  MPI_Datatype mpi_type = (sizeof(problem_type) == sizeof(double)) ? MPI_DOUBLE : MPI_FLOAT;
  MPI_Gatherv(x_cpu_local.data(), my_count, mpi_type,
              myrank==0 ? x_global.data() : nullptr,
              recvcounts.data(), displs.data(),
              mpi_type, 0, MPI_COMM_WORLD);
  if(myrank == 0){
    std::cout << "\n=== VERIFY: solution error vs manufactured exact solution ===\n";
    std::cout << "  exact solution: x_exact(ix,iy,iz) = (hx*hy*hz)^2 * (ix+1)^2 * (iy+1)^2 * (iz+1)^2\n";
    std::cout << "                  with hx=hy=hz=1/(NX+1)\n";
    std::cout << "  norm: max_i |x_numerical[i] - x_exact[i]|  (max-norm of solution error vector)\n";
    std::cout << "  NOTE: This is the SOLUTION error (x_num - x_exact), NOT the equation residual (b - A*x_num).\n";
    std::cout << "        They are related by  x_num - x_exact = A^{-1} (b - A*x_num).\n";
    matrix_obj.check_solution(x_global, nx);
    verify_direct_solution(crsMatrix_cpu, b_global_for_verify, x_global, n_global);
  }
#else
  std::cout << "\n=== VERIFY: solution error vs manufactured exact solution ===\n";
  std::cout << "  exact solution: x_exact(ix,iy,iz) = (hx*hy*hz)^2 * (ix+1)^2 * (iy+1)^2 * (iz+1)^2\n";
  std::cout << "                  with hx=hy=hz=1/(NX+1)\n";
  std::cout << "  norm: max_i |x_numerical[i] - x_exact[i]|  (max-norm of solution error vector)\n";
  std::cout << "  NOTE: This is the SOLUTION error (x_num - x_exact), NOT the equation residual (b - A*x_num).\n";
  std::cout << "        They are related by  x_num - x_exact = A^{-1} (b - A*x_num).\n";
  matrix_obj.check_solution(x_cpu_local, nx);
  verify_direct_solution(crsMatrix_cpu, b_global_for_verify, x_cpu_local, n_global);
#endif

  if( myrank == 0){
    for(int i = 0;i<(5>nblock?nblock:5);i++){
      std::cout<<"x["<<i<<"] = "<<x_cpu_local[i]<<std::endl;
    }
  }
  }

#if PARCEL_MPI
  // MPI_Finalize は main 側で実施
#endif

  return 0;
}

// ===== 精度の組合せごとに run_verify を呼び分ける =====
//
//   9 構成。data / calc / calc_precon の 3 つを振る。
//   残り (precon_mat, precon_vec, qr) は全構成で倍精度。
int main(int argc, char *argv[]) {
#if PARCEL_MPI
  MPI_Init(&argc, &argv);
#endif

  // コマンドライン引数の KEY=VALUE を環境変数として取り込む。
  //
  //   実行時オプションは環境変数で受けるが、**MPI 実装によっては複数ノードに
  //   またがると環境変数が転送されない**。その場合リモートのランクだけが既定値で動き、
  //   ランクごとに格子が食い違ってハロ交換が破綻する。
  //
  //   argv は MPI 実装によらず必ず全ランクに届くので、こちらからも指定できる
  //   ようにする。既に設定済みの環境変数は上書きしない (第 3 引数 0) ので、
  //   環境変数と併用しても環境変数側が優先される。
  //
  //     mpirun -n 4 ./run PARCEL_NX=256 PARCEL_SOLVER=pcg
  for(int i = 1; i < argc; i++){
    const char *eq = std::strchr(argv[i], '=');
    if(eq != nullptr && eq != argv[i]){
      const std::string key(argv[i], static_cast<size_t>(eq - argv[i]));
      setenv(key.c_str(), eq + 1, 0);
    }
  }

  // rank 0 の std::cout をファイルにも書く (parcel_tee_buf の説明を参照)。
  //   引数の KEY=VALUE を取り込んだ後に行うので PARCEL_OUTPUT は引数でも指定できる。
  int out_rank = 0;
#if PARCEL_MPI
  MPI_Comm_rank(MPI_COMM_WORLD, &out_rank);
#endif
  std::ofstream out_file;
  parcel_tee_buf *out_tee = nullptr;
  std::streambuf *cout_orig = std::cout.rdbuf();
  if(out_rank == 0){
    std::string path = "parcel_result.txt";
    if(const char *e = std::getenv("PARCEL_OUTPUT")){ path = e; }
    if(!path.empty() && path != "none"){
      out_file.open(path.c_str(), std::ios::out | std::ios::trunc);
      if(out_file){
        out_tee = new parcel_tee_buf(cout_orig, out_file.rdbuf());
        std::cout.rdbuf(out_tee);
        // 出力のたびに書き出す。途中で落ちてもそこまでの出力がファイルに残る。
        std::cout << std::unitbuf;
      }else{
        std::cout << " PARCEL_OUTPUT=" << path << " を開けないので標準出力のみに出す" << std::endl;
      }
    }
  }

  // 版 (Include/parcel_version.hpp)。PARCEL_EXPECT_VERSION (回帰試験の生成器が与える) があれば照合し、
  //   違えば試験をせずに終了コード 3 で終わる (別の版のヘッダで組んだ実行ファイルを試験しないため)
  {
    const std::string ver = parcel::version_string();
    const char *ver_expect = std::getenv("PARCEL_EXPECT_VERSION");
    const bool ver_ok = ver == PARCEL_VERSION_STRING
      && (ver_expect == nullptr || ver_expect[0] == '\0' || ver == ver_expect);
    if(out_rank == 0){
      std::cout << "[VERSION] PARCEL " << ver;
      if(ver_expect != nullptr && ver_expect[0] != '\0') std::cout << " (expect " << ver_expect << ")";
      std::cout << (ver_ok ? "" : " MISMATCH") << std::endl;
    }
    if(!ver_ok){
      std::cout.rdbuf(cout_orig);
      delete out_tee;
#if PARCEL_MPI
      MPI_Finalize();
#endif
      return 3;
    }
  }

  std::string prec = "default";
  if(const char *e = std::getenv("PARCEL_PRECISION")){ prec = e; }

  int rc = 0;
// 4・5 番目 (前処理行列・前処理ベクトルのデータ精度) はマクロ
//   parcel_solver_data_precon_mat_type / parcel_solver_data_precon_vec_type で与える (既定 parcel_double)。
//   前処理クラスもソルバも、これらを本体のデータ精度と独立に受け取る
//   (parcel_precon.hpp の Tprecon_mat / Tprecon_vec、solver<...,Tprecon_mat,Tprecon_vec>)。
//   これで「前処理の演算精度」(parcel_solver_calc_precon_type) と
//   「前処理のデータ精度」を分けて測れる。
#define PARCEL_RUN(D,C,P) rc = run_verify<D,C,P,parcel_solver_data_precon_mat_type,parcel_solver_data_precon_vec_type,parcel_double>(argc, argv)

  // 既定では コンパイル時に指定された 1 構成だけを実体化する。
  //
  //   9 構成すべてを 1 バイナリに入れると 7 ソルバ x 9 精度 = 63 通りの
  //   テンプレート実体化が 1 翻訳単位に入り、1 回のビルドが大幅に遅くなる。
  //   ビルド回数が減っても総時間ではかえって遅い。
  //
  //   -DPARCEL_RUNTIME_PRECISION を付けたときだけ 9 構成を実体化して
  //   PARCEL_PRECISION で選べるようにする。ビルド回数を最小にしたい場合
  //   (ビルドが極端に遅い環境など) に使う。
#ifndef PARCEL_RUNTIME_PRECISION
  (void)prec;
  PARCEL_RUN(parcel_solver_data_type, parcel_solver_calc_type, parcel_solver_calc_precon_type);
#else

  if     (prec == "double")     PARCEL_RUN(parcel_double,       parcel_double,       parcel_double);
  else if(prec == "calcF16")    PARCEL_RUN(parcel_double,       parcel_fp16,         parcel_double);
  else if(prec == "dataF16")    PARCEL_RUN(parcel_fp16,         parcel_fp16,         parcel_double);
  else if(prec == "precon_calcF16") PARCEL_RUN(parcel_double,    parcel_double,       parcel_fp16);
  else if(prec == "calcF")      PARCEL_RUN(parcel_double,       parcel_float,        parcel_double);
  else if(prec == "dataF")      PARCEL_RUN(parcel_float,        parcel_float,        parcel_double);
  else if(prec == "precon_calcF")   PARCEL_RUN(parcel_double,    parcel_double,       parcel_float);
  else if(prec == "calcDD")     PARCEL_RUN(parcel_double,       parcel_doubledouble, parcel_double);
  else if(prec == "dataDD")     PARCEL_RUN(parcel_doubledouble, parcel_doubledouble, parcel_double);
  else {
    // 未指定・未知の場合はコンパイル時の既定値を使う
    PARCEL_RUN(parcel_solver_data_type, parcel_solver_calc_type, parcel_solver_calc_precon_type);
  }
#endif
#undef PARCEL_RUN

  // std::cout を元に戻してからファイルを閉じる (out_file より std::cout の方が長生きするため)
  if(out_tee != nullptr){
    std::cout.flush();
    std::cout.rdbuf(cout_orig);
    delete out_tee;
    out_file.close();
  }

#if PARCEL_MPI
  MPI_Finalize();
#endif
  return rc;
}
