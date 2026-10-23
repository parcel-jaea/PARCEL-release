// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_HPP_
#define PARCEL_KRYLOV_HPP_

#ifdef RESIDUAL_REPLACE
#define TRUE_RESIDUAL 1
#endif

#include <vector>
#include <type_traits>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include "ddlib.hpp"
#include "util_sycl.hpp"
#include "parcel.hpp"
#include "parcel_precon.hpp"
#include "parcel_type.hpp"
#include "parcel_blas.hpp"
#include "parcel_lapack.hpp"
#include "parcel_util.hpp"

#include "parcel_time.hpp"
#include "parcel_solver_select.hpp"

namespace parcel{
  namespace krylov{

    // ---------------------------------------------------------------------
    // 収束状態。PETSc の KSPConvergedReason に倣い、正は収束、負は失敗。
    //   各ソルバの solver() は入口で ITERATING に戻し、出口で以下のどれかを設定する。
    //   値は全ランクで同じ (残差・固有値はどれも allreduce 済みの内積から作るため)。
    //
    //   CONVERGED_RTOL     ( 1) 収束判定 (tol) を満たした
    //   ITERATING          ( 0) 未実行 (solver() を呼ぶ前) またはリセット直後
    //   DIVERGED_ITS       (-1) 最大反復回数 / 最大リスタート回数に達したが tol を満たさない
    //   DIVERGED_NANORINF  (-2) 判定に使う残差 (固有値ソルバは固有値の推定値) に NaN / Inf が出た
    //   DIVERGED_BREAKDOWN (-3) アルゴリズムの破綻。
    //                           ks_arnoldi: LAPACK の失敗、不変部分空間の次元が nev 未満
    //                           pbicgstab : <r0,q> = 0 / 非有限、omega = 0、rho = <r0,r> = 0
    //                           pgmres / pcagmres: Arnoldi の最初の列で H が特異
    //   線形ソルバは判定に使う残差が非有限になった時点で抜ける。
    //   ||b|| = 0 なら x = 0、CONVERGED_RTOL、its = 0、residual = 0 で即座に返す (全線形ソルバ)。
    //   初期残差 ||b - A x0|| / ||b|| < tol なら反復せずに x = x0 のまま CONVERGED_RTOL、its = 0、
    //   residual = true_residual = 初期の相対残差で返す (全線形ソルバ)。
    //   pgmres / pcagmres の「幸運な破綻」(新しい基底のノルム ~ 0) はその列で Arnoldi を止め、
    //   それまでの列で x を更新して、次のリスタート先頭の真の残差で判定する。
    //
    // 取得関数 (krylov_param。全ソルバ共通):
    //   get_converged_reason() : 上の値
    //   is_converged()         : get_converged_reason() > 0
    //   get_iterations()       : 反復回数 (下の表)
    //   get_residual()         : tol と比べた相対値 (下の表)。計算していなければ NaN
    //   get_true_residual()    : ||b - Ax|| / ||b|| (終了時に計算しているソルバだけ)。無ければ NaN
    //
    //   ソルバ           iterations                          residual
    //   pcg              CG 反復回数 (表示の "N : residual")   漸化式の残差 ||r||/||b|| (判定に使う値)。
    //                                                        PARCEL_PSTEP の間引き更新は x だけなので r には影響しない
    //   pbicgstab        反復回数 (同上)                       漸化式の残差 ||r||/||b||
    //   pgmres           内側反復の総数 = リスタート回数 * m    リスタート先頭で計算する真の残差 ||b-Ax||/||b||
    //                    (表示のカウンタと同じ数え方)          反復上限で抜けたときは最後の更新後の値 (= true_residual)
    //                                                        で判定する (ループ内では最後の更新を判定しないため)
    //   pcagmres         基底ベクトルの総数 = 周回数 * s*t      pgmres と同じ
    //                    (表示のカウンタと同じ数え方。CAGMRES_Q_IMPLICIT も同じ)
    //   pgmres / pcagmres の max_iteration は restart (m、s*t) の倍数でなくてよい。
    //     最後の周回だけ端数 (max_iteration - それまでの反復数) 列で最小二乗を解き、上限に達したときの
    //     iterations は max_iteration ちょうど。
    //     pcagmres は最後のブロックの基底を s 本単位で作るので、使わない基底の SpMV・前処理が最大 s-1 回余分に走る。
    //   power            外側反復の回数                        固有値の相対変化 |λ_i - λ_{i-1}| / |λ_i|
    //   inverse_power    外側反復の回数                        同上
    //   ks_arnoldi       実行したリスタートの回数               判定対象 nev 個の Ritz 残差推定 |h_{m+1,m} q|/|λ| の最大値
    //
    //   true_residual: pcg / pbicgstab / pgmres / pcagmres は終了時に "finish : residual true norm"
    //   として表示している値 (||b|| で割った相対値) をそのまま返す。追加の SpMV はしない。
    //   pgmres の前処理空間出力モード (固有値ソルバの内部) では ||b - A K^-1 z + σ z|| / ||b||。
    //   固有値ソルバは NaN。
    //
    //   固有値ソルバ (eigen_param) の追加分:
    //     get_linsolve_failures() : 内部線形ソルバ (inverse_power、ks_arnoldi の shift-invert) が
    //                               収束しなかった回数。外側の converged_reason には反映しない。
    //     ks_arnoldi::get_ks_nconv() : 最後の判定で tol を満たした Ritz 対の数 (nev 個のうち)
    //
    //   標準出力には rank 0 が solver() の最後に 1 行だけ追加で書く:
    //     " converged_reason = 1 (CONVERGED_RTOL) , its = 27 , residual = 4.200000e-10 , true_residual = ..."
    //   回帰試験・性能集計のスクリプトが拾う文字列 ("converged = ", "iterations = ", "restarts = ",
    //   "N : residual norm", 単独の nan / inf) と重ならないよう、反復回数は "its" と書き、
    //   非有限値は数値でなく "n/a" と書く。
    const PARCEL_INT CONVERGED_RTOL     =  1;
    const PARCEL_INT ITERATING          =  0;
    const PARCEL_INT DIVERGED_ITS       = -1;
    const PARCEL_INT DIVERGED_NANORINF  = -2;
    const PARCEL_INT DIVERGED_BREAKDOWN = -3;

    inline const char *converged_reason_name(PARCEL_INT r){
      switch(r){
      case CONVERGED_RTOL:     return "CONVERGED_RTOL";
      case ITERATING:          return "ITERATING";
      case DIVERGED_ITS:       return "DIVERGED_ITS";
      case DIVERGED_NANORINF:  return "DIVERGED_NANORINF";
      case DIVERGED_BREAKDOWN: return "DIVERGED_BREAKDOWN";
      default:                 return "UNKNOWN";
      }
    }

    class krylov_param : public precon_param{
      // 既定値。精度を指定しなくても中規模の問題で DIVERGED_ITS にならないよう厳しめ・多めに取る。
      double tol_ = 1.0e-8;
      PARCEL_INT itrmax_ = 10000;
      // 収束状態 (solver() の出口で設定。上の説明を参照)
      PARCEL_INT converged_reason_ = ITERATING;
      PARCEL_INT iterations_       = 0;
      double     residual_         = std::numeric_limits<double>::quiet_NaN();
      double     true_residual_    = std::numeric_limits<double>::quiet_NaN();
    public :
      krylov_param(){}
      ~krylov_param(){}
      // 範囲外の値 (tol<=0 や非有限、itrmax<1) は std::invalid_argument。
      void set_tolerance(double tol){
	check_tolerance(tol, "set_tolerance");
	tol_ = tol;
      }
      void set_max_iteration(PARCEL_INT itrmax){
	check_positive(itrmax, "set_max_iteration");
	itrmax_ = itrmax;
      }
      // 検査の共通部分 (派生クラスの設定関数も使う)
      static void check_tolerance(double tol, const char *who){
	if(!(tol > 0.0) || !std::isfinite(tol)){
	  throw std::invalid_argument(std::string("parcel::krylov::") + who
				      + ": tolerance must be a finite positive number (got "
				      + std::to_string(tol) + ")");
	}
      }
      static void check_positive(PARCEL_INT v, const char *who){
	if(v < 1){
	  throw std::invalid_argument(std::string("parcel::krylov::") + who
				      + ": value must be >= 1 (got " + std::to_string((long long)v) + ")");
	}
      }
      double get_tolerance(){
	return tol_;
      }
      PARCEL_INT get_max_iteration(){
	return itrmax_;
      }
      PARCEL_INT get_converged_reason() const { return converged_reason_; }
      bool       is_converged()         const { return converged_reason_ > 0; }
      PARCEL_INT get_iterations()       const { return iterations_; }
      double     get_residual()         const { return residual_; }
      double     get_true_residual()    const { return true_residual_; }
    protected :
      // solver() の入口で呼ぶ
      void reset_status(){
	converged_reason_ = ITERATING;
	iterations_       = 0;
	residual_         = std::numeric_limits<double>::quiet_NaN();
	true_residual_    = std::numeric_limits<double>::quiet_NaN();
      }
      // solver() の出口で呼ぶ
      void set_status(PARCEL_INT reason, PARCEL_INT its, double res,
		      double true_res = std::numeric_limits<double>::quiet_NaN()){
	converged_reason_ = reason;
	iterations_       = its;
	residual_         = res;
	true_residual_    = true_res;
      }
      // 反復で抜けた理由を決める (線形ソルバ・power 系の共通部分)
      //   converged : tol を満たして抜けた / nonfinite : 判定値に NaN / Inf が出た
      static PARCEL_INT decide_reason(bool converged, bool nonfinite){
	if(converged) return CONVERGED_RTOL;
	if(nonfinite) return DIVERGED_NANORINF;
	return DIVERGED_ITS;
      }
      // rank 0 が 1 行書く。extra は固有値ソルバの追加項目 (" , ks_nconv = 4/4" など)。
      //   書式 (scientific / precision) は呼び出し前の状態に戻す (後に続く出力を変えないため)。
      void print_status(int myrank, const std::string &extra = std::string()) const {
	if(myrank != 0) return;
	const std::ios_base::fmtflags f = std::cout.flags();
	const std::streamsize p = std::cout.precision();
	std::cout << " converged_reason = " << converged_reason_
		  << " (" << converged_reason_name(converged_reason_) << ")"
		  << " , its = " << iterations_
		  << " , residual = ";
	put_value(residual_);
	if(!std::isnan(true_residual_)){
	  std::cout << " , true_residual = ";
	  put_value(true_residual_);
	}
	std::cout << extra << std::endl;
	std::cout.flags(f);
	std::cout.precision(p);
      }
    private :
      // 非有限値は "n/a" と書く (回帰試験の nan / inf 検出に引っかけないため。値は取得関数で見る)
      static void put_value(double v){
	if(std::isfinite(v)) std::cout << std::scientific << std::setprecision(6) << v;
	else                 std::cout << "n/a";
      }
    };

    // 固有値ソルバの内部線形ソルバの種類 (inverse_power の set_linsolve_algo)
    const PARCEL_INT LINSOLVE_PGMRES = 0;
    const PARCEL_INT LINSOLVE_PCG    = 1;

    // 固有値ソルバ (power / inverse_power / ks_arnoldi) 共通の設定。
    //
    //   set_precon_spectrum(true) : 前処理後の行列 A K^-1 の固有値を求める (全ソルバ共通)。
    //                               K は set_precon_algo() などの前処理設定で A から作る。
    //   set_precon_spectrum(false): A の固有値を求める (既定)。外側の固有値反復は K を使わない。
    //
    //   内部線形ソルバ (inverse_power、ks_arnoldi の shift-invert) の設定は別に持つ。
    //   内部ソルバの前処理 (set_linsolve_precon_algo) の既定はヤコビ前処理 (PRECON_JACOBI、
    //   反復 1 回)。前処理なしにするには set_linsolve_precon_algo(PRECON_NONE)。
    //   A K^-1 モードでは内部 pgmres の前処理は K そのものなので、この指定は使わない。
    class eigen_param : public krylov_param{
      bool       precon_spectrum_  = false;
      PARCEL_INT linsolve_algo_    = LINSOLVE_PGMRES;
      double     linsolve_tol_;
      PARCEL_INT linsolve_max_;
      PARCEL_INT linsolve_restart_;
      precon_param linsolve_precon_;          // 既定: PRECON_JACOBI (コンストラクタで設定)
      precon_param none_precon_;              // 常に PRECON_NONE
      bool       linsolve_precon_set_ = false;   // set_linsolve_precon_algo を呼んだか
      PARCEL_INT linsolve_failures_ = 0;      // 内部線形ソルバが収束しなかった回数 (solver() ごと)
    public :
      eigen_param(double tol, PARCEL_INT itrmax, PARCEL_INT restart)
        : linsolve_tol_(tol), linsolve_max_(itrmax), linsolve_restart_(restart){
        // 既定値なので linsolve_precon_set_ は立てない (ks_arnoldi の K = I 経路の注意は
        //   利用者が明示したときだけ出す)。
        linsolve_precon_.set_precon_algo(PRECON_JACOBI);
      }
      ~eigen_param(){}
      void set_precon_spectrum(bool f)                   { precon_spectrum_ = f; }
      bool get_precon_spectrum() const                   { return precon_spectrum_; }
      // 内部ソルバの設定も範囲外は std::invalid_argument (C++ API では 0 も不可。
      //   C ラッパーの 0 = 「ライブラリ既定」はラッパー側で呼び出しを省いて実現している)。
      void set_linsolve_algo(PARCEL_INT algo){
        if(algo != LINSOLVE_PGMRES && algo != LINSOLVE_PCG){
          throw std::invalid_argument("parcel::krylov::set_linsolve_algo: invalid value "
                                      + std::to_string((long long)algo)
                                      + " (expected LINSOLVE_PGMRES=0 or LINSOLVE_PCG=1)");
        }
        linsolve_algo_ = algo;
      }
      PARCEL_INT get_linsolve_algo() const               { return linsolve_algo_; }
      void set_linsolve_tolerance(double tol){
        check_tolerance(tol, "set_linsolve_tolerance");
        linsolve_tol_ = tol;
      }
      double get_linsolve_tolerance() const              { return linsolve_tol_; }
      void set_linsolve_max_iteration(PARCEL_INT itr){
        check_positive(itr, "set_linsolve_max_iteration");
        linsolve_max_ = itr;
      }
      PARCEL_INT get_linsolve_max_iteration() const      { return linsolve_max_; }
      void set_linsolve_restart(PARCEL_INT m){
        check_positive(m, "set_linsolve_restart");
        linsolve_restart_ = m;
      }
      PARCEL_INT get_linsolve_restart() const            { return linsolve_restart_; }
      // 内部ソルバの前処理を明示したか (ks_arnoldi の K=I 経路で「使えない」と知らせるため)
      bool linsolve_precon_is_set() const                { return linsolve_precon_set_; }
      void set_linsolve_precon_algo(PARCEL_INT algo)     { linsolve_precon_.set_precon_algo(algo); linsolve_precon_set_ = true; }
      void set_linsolve_blockjacobi(PARCEL_INT nblock)   { linsolve_precon_.set_blockjacobi(nblock); }
      void set_linsolve_jacobi(PARCEL_INT iter)          { linsolve_precon_.set_jacobi(iter); }
      void set_linsolve_scaling_matrix(bool f)           { linsolve_precon_.set_scaling_matrix(f); }
      void set_linsolve_scaling_vector(bool f)           { linsolve_precon_.set_scaling_vector(f); }
      precon_param *get_linsolve_precon_param()          { return &linsolve_precon_; }
      // 外側の固有値反復が使う前処理設定。A の固有値を求めるときは K を使わないので前処理なし。
      precon_param *get_outer_precon_param(){
        return precon_spectrum_ ? get_precon_param() : &none_precon_;
      }
      // 内部線形ソルバ (inverse_power、ks_arnoldi の shift-invert) が収束しなかった回数。
      //   外側の converged_reason には反映しない (内部が tol 未達でも外側は収束しうるため)。
      PARCEL_INT get_linsolve_failures() const           { return linsolve_failures_; }
    protected :
      void reset_linsolve_failures()                     { linsolve_failures_ = 0; }
      // 内部ソルバ 1 回ぶんの結果を数える
      void count_linsolve(const krylov_param &inner){
        if(!inner.is_converged()) linsolve_failures_++;
      }
    };

    // キューが in-order か確かめる require_in_order は parcel.hpp (namespace parcel) にある。
    //   parcel::matrix::set_crsMat からも呼ぶため、parcel_krylov.hpp より先に
    //   読まれる parcel.hpp に置く必要がある。parcel::krylov の中からは修飾なしで呼べる。
    //   既定で有効 (PARCEL_REQUIRE_IN_ORDER=1)。外すときは -DPARCEL_REQUIRE_IN_ORDER=0。
    //   in-order が要る理由は parcel.hpp の同関数の注記を参照 (要旨: PARCEL のカーネルは
    //   event / depends_on を張らずに順に投入し、前のカーネルの結果を次が読むため)。

    // 局所ベクトルの先頭のグローバル行番号。
    //   各ランクの局所行数の排他的 prefix sum (rank 0 は 0)。初期ベクトルの種を
    //   プロセス数・不均等分割によらず同じにするために使う。
    //   MPI なし (PARCEL_MPI=0) では呼ばない (呼び出し側が 0 とする)。
#if PARCEL_MPI
    inline long long global_row_start(MPI_Comm mpi_comm, long long nlocal){
      long long start = 0;
      MPI_Exscan(&nlocal, &start, 1, MPI_LONG_LONG, MPI_SUM, mpi_comm);
      int rank = 0;
      MPI_Comm_rank(mpi_comm, &rank);
      if(rank == 0) start = 0;   // MPI_Exscan は rank 0 の受信バッファを定義しない
      return start;
    }
#endif

    // 初期ベクトルの既定の種 (power / inverse_power / ks_arnoldi 共通)。g はグローバル行番号。
    //   全要素 1.0 は特定の固有ベクトルと厳密に直交しうるので sin/cos の組合せにする。
    inline double default_seed_value(long long g_){
      const double g = (double)g_;
      return 1.0 + 0.5 * std::sin(0.317 * g + 0.7) + 0.3 * std::cos(0.181 * g - 1.3);
    }

    // 前処理設定を内部ソルバへ写す。利用者が明示していない項目 (ブロック数・スケーリング) は
    //   写さず、内部ソルバ側の自動設定 (デバイスの並列度・精度から決める) に任せる。
    inline void copy_precon_param(precon_param &src, precon_param &dst){
      dst.set_precon_algo(src.get_precon_algo());
      dst.set_jacobi(src.get_jacobi());
      if(src.blockjacobi_is_set())   { dst.set_blockjacobi(src.get_blockjacobi()); }
      if(src.scaling_matrix_is_set()){ dst.set_scaling_matrix(src.get_scaling_matrix()); }
      if(src.scaling_vector_is_set()){ dst.set_scaling_vector(src.get_scaling_vector()); }
    }

  }

}

// 線形方程式反復解法
// ビルドするソルバの取捨は parcel_solver_select.hpp を参照。
#if PARCEL_ENABLE_PCG
#include "parcel_krylov_cg.hpp"
#endif
#if PARCEL_ENABLE_PBICGSTAB
#include "parcel_krylov_bicgstab.hpp"
#endif
#if PARCEL_ENABLE_PGMRES
#include "parcel_krylov_gmres.hpp"
#endif
#if PARCEL_ENABLE_PCAGMRES
// Q を陽に作らない版 (pcagmres の CAGMRES_Q_IMPLICIT) の本体
//   parcel_krylov_choleskyqr_cagmres.hpp はこのヘッダの末尾で読み込む
#include "parcel_krylov_cagmres.hpp"
#endif

// 固有値計算
#if PARCEL_ENABLE_POWER
#include "parcel_krylov_power.hpp"          // べき乗法            : parcel::krylov::power
#endif
#if PARCEL_ENABLE_INVERSE_POWER
#include "parcel_krylov_inverse_power.hpp"  // 逆べき乗法          : parcel::krylov::inverse_power
#endif
#if PARCEL_ENABLE_KS_ARNOLDI
#include "parcel_krylov_ks.hpp"             // Arnoldi + Krylov-Schur : parcel::krylov::ks_arnoldi
#endif
// v1.1 の通信削減型アーノルディ法 (parcel::krylov::pcaarnoldi) は v1.2 で削除した.

#endif

