// =========================================================================
// parcel_krylov_ks.hpp
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// 標準 Arnoldi + Krylov-Schur リスタート 固有値ソルバ : parcel::krylov::ks_arnoldi
//
// アルゴリズム:
//   Arnoldi で m 次 Krylov 部分空間を構築 → ヘッセンベルグ行列 H_m を dgeev で固有分解
//   → 上位 k_keep 個の Ritz 対を選別 → V_kept = V_m · Y_k で部分空間を縮約
//   → リスタートして同じ手順を繰返し、Ritz 残差 ||Av-λv||/(|λ|·||v||) で収束判定
//
// パラメータ (既定値は m = 30, k_keep = 10, nev = 4, 最大リスタート 200):
//   ks_m_           Krylov 部分空間の最大次元          set_ks_subspace(m, k_keep, nev)
//   ks_k_keep_      リスタート毎に保持する Ritz 対の数   同上
//   ks_nev_         求めたい固有値数 (収束判定の対象)    同上
//   ks_max_restart_ 最大リスタート回数                 set_ks_max_restart(R)
//   tolerance       Ritz 残差の収束判定閾値 (krylov_param 基底クラス)
//
// shift-invert:
//   set_shift(σ) を呼ぶと、各 Arnoldi ステップで内部 pgmres が (A-σI)z = v を解き、
//   固有値を λ_A = σ + 1/μ で逆変換する。呼出側は (A-σI) を組んで solver() に渡す。
//   σ 近傍の内部固有値を抽出する場合に用いる。
//   この既定の shift-invert では内部 pgmres の前処理は set_linsolve_precon_algo() 等で
//   指定し (既定はヤコビ前処理)、返る固有値は A の固有値である。
//   ShiftMat を渡さずに set_shift(σ) だけを呼んだ場合も A の shift-invert を行う
//   (K = I の A K^-1 経路。内部 pgmres が (A - σI) z = v を A とシフト σ から解く)。
//   この経路では内部 pgmres の前処理は K = I 固定で、set_linsolve_precon_algo() は効かない
//   (rank 0 が 1 行知らせる)。
//
// 前処理後の行列 A K^-1 の固有値 (set_precon_spectrum(true)、固有値ソルバ共通):
//   シフトなしでは A K^-1 の絶対値最大側の固有値を求める (既定の false では A の固有値)。
//   shift-invert では同じ行列の σ 近傍 (σ=0 なら絶対値最小) の固有値を求める。
//   各 Arnoldi ステップで内部 pgmres が (A K^-1 - σI) z = v を解いて z を返す
//   (右前処理 GMRES の内部変数 z をそのまま返す。K は陽に作らない)。
//   K は利用者の前処理設定のまま A から作り、シフト行列 ShiftMat は不要。
//   固有値は同じく λ = σ + 1/μ で逆変換する。前処理なし (K = I) なら
//   ShiftMat を組まずに A の shift-invert を行うのと同じになる。
//
// 不変部分空間 (h_{j+1,j} ~ 0) の扱い:
//   低精度の基底では直交化後の w がアンダーフローで 0 になり、偽の不変部分空間になりうる。
//   (A) 不変部分空間に頼る収束は真の残差 ||Op x - θx|| / (|θ| ||x||) <= tol で確かめる。
//   (B) 確かめられなければ乱数ベクトル (グローバル行番号から作る) を直交化して h_{j+1,j} = 0 の
//       まま m まで広げる (ARPACK dgetv0 と同じ。1 回の Arnoldi 過程で 3 回まで)。
//   乱数方向を入れた後の収束も真の残差で確かめる。詳細は solver() 内の注記を参照。
//
// 直交化 (set_qr_algo() で選ぶ。pgmres と同じ定数で、ks_arnoldi ではどちらも 2 回適用する。
//   set_qr_algo() を呼ばないときの既定はビルド時の PARCEL_KS_ORTHO_DEFAULT (cgs2 | mgs)、それも無ければ QR_CGS。
//   PS-A 向けのビルド (spack の ks_ortho=mgs、試験の ps_a) は -DPARCEL_KS_ORTHO_DEFAULT=mgs):
//   QR_CGS: ブロック版の古典グラムシュミットを 2 回 (CGS2)。1 パスは局所内積 j+1 本 + allreduce 1 回 + gemv 1 回。
//   QR_MGS: 修正グラムシュミットを 2 回 (MGS2)。列ごとに内積 -> axpy。
//   どちらも 2 回目で直交性を回復するので、縮重スペクトルでの ghost eigenvalue は抑制でき、
//   locking/deflation は不要。MGS は 1 ランクの局所長が小さい CPU の構成 (axpy が読む列がキャッシュに
//   残る) で速いことがあり、局所長が大きい CPU と GPU では基底をまとめて読む CGS2 が速い。
//
// API:
//   solver(Mat, EigenValue, EigenVec)
//     EigenValue: 2 * k_keep 要素 (実部・虚部が交互)
//     EigenVec  : n * k_keep 要素 (k_keep 個の固有ベクトル; 入力時は先頭 n 要素が初期ベクトル)
//                 n * nev 以上なら受け付け、min(k_keep, EigenVec.size()/n) 本だけ書く
//                 (必要な固有ベクトルが nev 本なら、呼び出し側は n*k_keep 要素を確保しなくてよい)。
//     n は Mat.n_local() (局所行数)。入口で長さと (1 <= nev <= k_keep < m <= 全体行数)
//     を検査し、違反は全ランクそろって std::invalid_argument を投げる。
//     初期ベクトルのノルムが 0 か非有限なら、既定の sin/cos 種に置き換える (rank 0 が知らせる)。
// =========================================================================
#ifndef PARCEL_KRYLOV_KS_HPP_
#define PARCEL_KRYLOV_KS_HPP_

// リスタートの縮約 V(:,0:k) = V(:,0:m) Z は V の中で行う (省メモリ)。
//   縦長 gemm (tgemm) は 1 行 (CPU は行ブロック) ごとに、その行の m 列を全部読んで局所の累算器に積んでから
//   同じ行の k 列へ書くので、出力を入力と同じ配列に置いても読み書きが重ならない。1 要素の計算
//   (i = 0,1,... の順に積み、最後に alpha を掛ける) は 2 本目の基底に書く場合と同じ。
//   使える条件: double-double を含まない型、保持数+1 <= 32 (tgemm の累算器の上限。超えると列ごとの
//   経路に落ち、行の途中で書き換えてしまう)。条件を満たさないときは 2 本目の基底 V_new (m+1 列) を確保する。
//   V の中で縮約するときの V_new は、真の残差の確認に使う作業域 3 列だけ。

#include <algorithm>
#include <cmath>
#include <cstdio>    // ks_sci の snprintf
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "lapacke.h"

namespace parcel {
  namespace krylov {

    template<
      class Tspmv   = parcel_double,
      class Tprecon = parcel_double,
      class Taxpy   = parcel_double,
      class Tdot    = parcel_double
    >
    class ks_arnoldi : public eigen_param {

      parcel::vector_util vec_util;
      parcel::scalar_util host_util;

      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_main;
      parcel::timer timer_solver_SpMV;
      parcel::timer timer_solver_axpy;
      // ブロック直交化の更新段 (v <- v - V h) は gemv なので axpy とは別に計る。
      //   gemv の M は j+1 で、KS_M=40 なら最大 40。
      parcel::timer timer_solver_gemv;
      parcel::timer timer_solver_dot;
      parcel::timer timer_solver_precon;
      //   「その他」(= main loop から上のカーネルを引いた残り) の内訳を切り分けるための計測。
      //   dense_eig   : 収束判定用の固有分解 (dgeev)
      //   dense_schur : リスタート用の実 Schur 分解と並べ替え (dgees + dtrsen)
      //   restart_vec : 基底の入れ替えに伴うホスト側の詰め替え (Z_new / Z_col の作成など)
      parcel::timer timer_dense_eig;
      parcel::timer timer_dense_schur;
      parcel::timer timer_restart_vec;

      // KS パラメータ
      PARCEL_INT ks_m_           = 30;
      PARCEL_INT ks_k_keep_      = 10;
      PARCEL_INT ks_nev_         =  4;
      // 固有値が密集していると数十回のリスタートでは届かないことがあるので大きめに取る
      PARCEL_INT ks_max_restart_ = 200;

      // shift-invert モード設定 (内部 pgmres で (A-σI)⁻¹ を Arnoldi step に使用)
      //   σ 近傍の "内部" 固有値抽出に有効
      bool       shift_invert_mode_ = false;
      double     shift_sigma_       = 0.0;
      // 最後の収束判定で tol を満たした Ritz 対の数 (nev 個のうち。solver() ごと)
      PARCEL_INT ks_nconv_          = 0;
      // 直交化 (parcel::qr::QR_CGS = CGS2, QR_MGS = MGS2)。既定はビルド時の PARCEL_KS_ORTHO_DEFAULT
      PARCEL_INT ks_qr_algo_        = ks_default_qr_algo();
      //   内部 pgmres の設定と A K^-1 モードは eigen_param (固有値ソルバ共通) が持つ。
      //   既定: 収束判定値 1e-10、最大反復 2000、リスタート周期 30、ヤコビ前処理。

      // ---- 補助関数 ----
      //   直交化の既定 (set_qr_algo() を呼ばないとき): ビルド時の -DPARCEL_KS_ORTHO_DEFAULT=cgs2 | mgs、
      //   未定義なら QR_CGS (CGS2)
      static PARCEL_INT ks_default_qr_algo() {
#ifdef PARCEL_KS_ORTHO_DEFAULT
#define PARCEL_KS_ORTHO_STR2_(x) #x
#define PARCEL_KS_ORTHO_STR_(x) PARCEL_KS_ORTHO_STR2_(x)
        const std::string v(PARCEL_KS_ORTHO_STR_(PARCEL_KS_ORTHO_DEFAULT));
#undef PARCEL_KS_ORTHO_STR_
#undef PARCEL_KS_ORTHO_STR2_
        if (v == "cgs2" || v == "CGS2") return parcel::qr::QR_CGS;
        if (v == "mgs"  || v == "MGS")  return parcel::qr::QR_MGS;
        throw std::invalid_argument("PARCEL: PARCEL_KS_ORTHO_DEFAULT must be cgs2 or mgs (got '" + v + "')");
#else
        return parcel::qr::QR_CGS;
#endif
      }
      //   |λ| の並べ替えキー。NaN は -inf 扱いにして降順の最後へ回す
      static double ks_mag_key(double re, double im) {
        const double v = std::sqrt(re*re + im*im);
        return std::isnan(v) ? -std::numeric_limits<double>::infinity() : v;
      }
      //   rank 0 の値を配る (MPI なしでは何もしない)
      //   MPI なしでは MPI_Comm の型がないので、コミュニケータの代わりに int (0) を渡す。
#if PARCEL_MPI
      template<class Tmat>
      static MPI_Comm comm_of(parcel::matrix<Tmat> &Mat) {
        return Mat.comm()->mpi_comm();
      }
      static void ks_bcast(double *p, int cnt, MPI_Comm c) {
        if (cnt > 0) MPI_Bcast(p, cnt, MPI_DOUBLE, 0, c);
      }
#else
      template<class Tmat>
      static int comm_of(parcel::matrix<Tmat> &) { return 0; }
      static void ks_bcast(double *, int, int) {}
#endif
      //   整数 1 個 (int / PARCEL_INT のどちらでも。PARCEL_INT が int のとき多重定義が衝突しないよう
      //   テンプレートで受けて long long で送る)
#if PARCEL_MPI
      template<class TI>
      static void ks_bcast_int(TI *p, MPI_Comm c) {
        long long v = (long long)*p;
        MPI_Bcast(&v, 1, MPI_LONG_LONG, 0, c);
        *p = (TI)v;
      }
#else
      template<class TI>
      static void ks_bcast_int(TI *, int) {}
#endif

      //   不変部分空間の後で Arnoldi を続けるための乱数方向。
      //   g はグローバル行番号 (global_row_start + 局所行)、c はリスタート番号・列番号・試行回数
      //   から作る通し番号。splitmix64 で [-1, 1) の一様乱数にする。行番号だけから決まるので
      //   プロセス数・分割に依らず同じ列になり、どのランク数でも同じ H が作られる。
      static double ks_random_value(long long g, long long c) {
        unsigned long long z = (unsigned long long)g * 0x9E3779B97F4A7C15ULL
                             ^ ((unsigned long long)c * 0xD1B54A32D192ED03ULL + 0x632BE59BD9B4E019ULL);
        z += 0x9E3779B97F4A7C15ULL;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= (z >> 31);
        return (double)(z >> 11) * (2.0 / 9007199254740992.0) - 1.0;
      }
      //   表示用 (%.3e)。std::cout の precision を変えないように文字列で作る
      static std::string ks_sci(double v) {
        char b[48]; std::snprintf(b, sizeof(b), "%.3e", v); return std::string(b);
      }
      //   1 回の Arnoldi 過程 (リスタート 1 回ぶんの拡張) で乱数方向に差し替える上限
      static constexpr int ks_max_random_dir = 3;

    public:
      ks_arnoldi() : eigen_param(1.0e-10, 2000, 30) {}
      ~ks_arnoldi(){}

      // 1 <= nev <= k_keep < m を満たさなければ std::invalid_argument
      //   (m <= 全体行数は行列が要るので solver() の入口で検査する)
      void set_ks_subspace(PARCEL_INT m, PARCEL_INT k_keep, PARCEL_INT nev) {
        if (!(1 <= nev && nev <= k_keep && k_keep < m)) {
          throw std::invalid_argument("parcel::krylov::ks_arnoldi::set_ks_subspace: need 1 <= nev <= k_keep < m (got m="
                                      + std::to_string((long long)m) + ", k_keep=" + std::to_string((long long)k_keep)
                                      + ", nev=" + std::to_string((long long)nev) + ")");
        }
        ks_m_ = m;
        ks_k_keep_ = k_keep;
        ks_nev_ = nev;
      }
      void set_ks_max_restart(PARCEL_INT R) {
        check_positive(R, "ks_arnoldi::set_ks_max_restart");
        ks_max_restart_ = R;
      }
      PARCEL_INT get_ks_k_keep() const { return ks_k_keep_; }
      PARCEL_INT get_ks_nev()    const { return ks_nev_; }
      PARCEL_INT get_ks_m()      const { return ks_m_; }
      PARCEL_INT get_ks_max_restart() const { return ks_max_restart_; }
      // 直交化。pgmres と同じ定数で、ks_arnoldi ではどちらも 2 回適用する:
      //   QR_CGS = ブロック版の古典グラムシュミット 2 回 (CGS2)、QR_MGS = 修正グラムシュミット 2 回 (MGS2)。
      //   それ以外は std::invalid_argument。既定はビルド時の PARCEL_KS_ORTHO_DEFAULT (未定義なら QR_CGS)
      void set_qr_algo(PARCEL_INT qrdecomposition_algo) {
        if (qrdecomposition_algo != parcel::qr::QR_CGS && qrdecomposition_algo != parcel::qr::QR_MGS) {
          throw std::invalid_argument("parcel::krylov::ks_arnoldi::set_qr_algo: invalid value "
                                      + std::to_string((long long)qrdecomposition_algo)
                                      + " (expected QR_CGS or QR_MGS)");
        }
        ks_qr_algo_ = qrdecomposition_algo;
      }
      PARCEL_INT get_qr_algo() const { return ks_qr_algo_; }
      // 収束状態 (krylov_param::get_converged_reason 等) と合わせて使う。
      //   iterations = 実行したリスタートの回数、residual = 判定対象 nev 個の Ritz 残差推定の最大値
      PARCEL_INT get_ks_nconv()  const { return ks_nconv_; }

      // Shift-invert: σ を設定すると solver() に渡される ShiftMat ((A-σI)) を使い、
      //   各 Arnoldi step で pgmres で (A-σI) z = v を解く。
      //   ShiftMat を渡さない場合は A とシフト σ から (A - σI) z = v を解く (K = I)。
      //   出力時に固有値を λ_A = σ + 1/μ で逆変換 (μ は (A-σI)⁻¹ の eigval)。
      void set_shift(double sigma) { shift_sigma_ = sigma; shift_invert_mode_ = true; }
      void disable_shift() { shift_invert_mode_ = false; }
      // 旧名 (互換のため残す)。固有値ソルバ共通の set_linsolve_* と同じ。
      void set_shift_linsolve_tolerance(double t)        { set_linsolve_tolerance(t); }
      void set_shift_linsolve_max_iteration(PARCEL_INT n){ set_linsolve_max_iteration(n); }
      void set_shift_linsolve_restart(PARCEL_INT r)      { set_linsolve_restart(r); }
      // set_precon_spectrum(true) (eigen_param) で A K^-1 の固有値を求める。
      //   シフトなし: A K^-1 の絶対値最大側。shift-invert: A K^-1 の σ 近傍 (ShiftMat 不要)。
      //   既定 (false) は A の固有値で、外側の Arnoldi 過程は K を使わない。

      template<
        class Tmat, class Teigen_val, class Teigen_vec,
        class Tprecon_mat = Teigen_vec,  // 前処理行列データ精度
        class Tprecon_vec = Teigen_vec   // 前処ベクトルデータ精度
      >
      void solver(
          parcel::matrix<Tmat>      &Mat,
          parcel::vector<Teigen_val> &EigenValue,   // size = 2*k_keep (Re,Im 交互)
          parcel::vector<Teigen_vec> &EigenVec,     // size = n*k_keep; 入力時最初の n が seed
          parcel::matrix<Tmat>      *ShiftMat = nullptr  // optional: (A - σ·I) for shift-invert
      ) {
        //   A K^-1 モードでは (A K^-1 - σI) を内部 pgmres のシフトで表すので ShiftMat は要らない。
        //   ShiftMat が無く A K^-1 モードでもない場合は K = I の同じ経路を使う
        //   (外側の前処理は get_outer_precon_param() = 前処理なし)。
        const bool shift_pspec  = (shift_invert_mode_ && (get_precon_spectrum() || ShiftMat == nullptr));
        const bool shift_invert = shift_invert_mode_;
        //   K = I で A の shift-invert をする経路 (表示と注意書き用)
        const bool shift_identity = shift_pspec && !get_precon_spectrum();
        sycl::queue *que = Mat.queue();
        require_in_order(que, "krylov::ks_arnoldi");
        //   n は行列の局所行数から取る
        const PARCEL_INT n = (PARCEL_INT)Mat.n_local();
        const PARCEL_INT m = ks_m_;
        const PARCEL_INT k_keep = ks_k_keep_;
        const PARCEL_INT nev = ks_nev_;
        const PARCEL_INT max_restart = ks_max_restart_;
        const double tol = get_tolerance();
        reset_status();
        reset_linsolve_failures();
        ks_nconv_ = 0;

#if PARCEL_MPI
        parcel::comm *comm = Mat.comm();
        int myrank   = comm->rank();
        PARCEL_INT nadd = Mat.get_recv_size();
        parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que, comm->mpi_comm());
#else
        int myrank   = 0;
        PARCEL_INT nadd = 0;
        parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
#endif
        // 廃止した環境変数 PARCEL_KS_ORTHO が設定されていれば, rank 0 が 1 回だけ警告する (値は使わない)。
        //   直交化は set_qr_algo() で選ぶ。
        {
          static bool ks_ortho_env_warned = false;
          const char *e = std::getenv("PARCEL_KS_ORTHO");
          if (!ks_ortho_env_warned && e != nullptr && e[0] != '\0') {
            ks_ortho_env_warned = true;
            if (myrank == 0)
              std::cerr << "PARCEL warning: PARCEL_KS_ORTHO is no longer used and is ignored; set the orthogonalization with "
                           "ks_arnoldi::set_qr_algo() (C/Fortran: parcel_options ks_qr_algo, Python: eigs(orthogonalization=))" << std::endl;
          }
        }
        //   各列の格納幅 (local + halo + 詰め物)。stride*(m+1) などの積は大規模問題で 2^31 を
        //   超えうるので size_t で持つ。
        //   基底の列の後ろに詰め物 PARCEL_BASIS_PAD 要素を入れる (キャッシュセットの衝突を避ける)。
        //   確保 (V_parallel / V_new_parallel) にも同じ詰め物を渡し、V_parallel.ld() と一致することを下で確かめる。
        //   列 j の局所部分・halo 込み部分は stride*j から始まり、詰め物は halo の後ろに入るだけ。
        const size_t stride = (size_t)n + (size_t)nadd + (size_t)PARCEL_BASIS_PAD;

        // ========== 入口の検査 ==========
        //   違反はランクごとに違いうる (EigenVec の長さ) ので、フラグを MPI_Allreduce(MAX) で
        //   集めてから全ランクそろって投げる (1 ランクだけ投げると他が集団通信で止まる)。
        {
          int bad = 0;
          std::string why;
          if (!(1 <= nev && nev <= k_keep && k_keep < m)) {
            bad = 1; why = "need 1 <= nev <= k_keep < m";
          } else if (EigenVec.size() < (size_t)n * (size_t)nev) {
            // 下限は n*nev (書く本数は出力の箇所で決める)
            bad = 1; why = "EigenVec.size() < n_local * nev";
          } else if (EigenValue.size() < (size_t)2 * (size_t)k_keep) {
            bad = 1; why = "EigenValue.size() < 2 * k_keep";
          }
          long long n_global = (long long)n;
#if PARCEL_MPI
          {
            long long nl = (long long)n;
            MPI_Allreduce(&nl, &n_global, 1, MPI_LONG_LONG, MPI_SUM, comm->mpi_comm());
          }
#endif
          if (!bad && (long long)m > n_global) {
            bad = 1; why = "m > global number of rows";
          }
          int bad_any = bad;
#if PARCEL_MPI
          MPI_Allreduce(&bad, &bad_any, 1, MPI_INT, MPI_MAX, comm->mpi_comm());
#endif
          if (bad_any) {
            if (why.empty()) why = "invalid argument on another rank";
            throw std::invalid_argument("parcel::krylov::ks_arnoldi::solver: " + why
                                        + " (m=" + std::to_string((long long)m)
                                        + ", k_keep=" + std::to_string((long long)k_keep)
                                        + ", nev=" + std::to_string((long long)nev)
                                        + ", n_local=" + std::to_string((long long)n)
                                        + ", EigenVec.size()=" + std::to_string((long long)EigenVec.size())
                                        + ", EigenValue.size()=" + std::to_string((long long)EigenValue.size()) + ")");
          }
        }

        timer_solver.start();
        timer_solver_init.start();

        // ========== 前処理行列構築 ==========
        //   A K^-1 モード (set_precon_spectrum(true)) のときだけ set_precon_algo() の K を作る。
        //     z = K^{-1} v, 続いて v_{j+1} = A z → ks_arnoldi は A·K^{-1} の固有値に収束
        //     用途: 前処理 A K^-1 のスペクトル評価 (PCG/GMRES 反復数予測)
        //   既定 (A の固有値) では前処理なし (identity, z = v)。
        timer_solver_init_precon.start();
        parcel::precon_param &precon = *(get_outer_precon_param());
        parcel::precon_matrix<
          Tmat, Teigen_vec, Tprecon_mat, Tprecon_vec, Tprecon
        > precon_Mat(Mat, precon);
        timer_solver_init_precon.end();

        // ========== ヘルパスカラ ==========
        // スカラ定数は axpy/SpMV の Talpha/Tbeta として渡すため Tdot 型で持つ。
        // Teigen_vec で宣言すると Tdot != Teigen_vec の混合精度でテンプレートが一致しない。
        parcel::vector<Tdot> one_vec(1, que);
        parcel::vector<Tdot> zero_vec(1, que);
        { double v = 1.0; one_vec.setValue(&v); }
        { double v = 0.0; zero_vec.setValue(&v); }
        //   反復のたびに変わるスカラ (直交化係数 -h_ij、正規化係数 1/h) を渡すための領域。
        //   parcel::vector<Tdot>(1, que) を都度作ると確保・解放・同期の固定費がかさむので、
        //   ループ外で 1 個だけ持って使い回す。
        parcel::vector<Tdot> scal_vec(1, que);
        //   ブロック直交化 (CGS2) で使う領域。直交化は実行時に選ぶので、どちらの経路でも確保する。
        //   h は V[:,0..j]^T v_{j+1} の係数 (長さ m+1)。MPI では局所内積を h_tmp に
        //   受けてから allreduce で h_dev に集める。
        parcel::vector<Tdot> nega_one_vec(1, que);
        { double v = -1.0; nega_one_vec.setValue(&v); }
        parcel::vector<Tdot> h_dev((size_t)(m+1), que);
        parcel::vector<Tdot> h_tmp((size_t)(m+1), que);
        std::vector<parcel::scalar> h_host((size_t)(m+1), 0.0);

        // ========== Krylov 基底 V を parallel_vector (halo 込み) で確保 ==========
        //   (n + nadd) × (m+1) サイズ
        //   列 j 局所部分 (size n)        : vec_util.setptr(n,        V_storage, Vj,     stride*j)
        //   列 j 通信込み部分 (size n+nadd): vec_util.setptr(n+nadd,  V_storage, Vj_add, stride*j)
        //   SpMV 入力には Vj_add (halo 込み) を渡す。dot/axpy には Vj (local) を渡す。
        parcel::parallel_vector<Teigen_vec> V_parallel(n, nadd, m+1, que, PARCEL_BASIS_PAD);
        // リスタートを V の中で行うか (先頭の説明)。行うときは V_new を作業域 3 列だけにする。
        constexpr bool ks_inplace_ok =
          !std::is_same<Teigen_vec, parcel_doubledouble>::value
          && !std::is_same<Tdot,       parcel_doubledouble>::value;
        const bool ks_inplace = ks_inplace_ok && (k_keep + 1 <= 32);
        const PARCEL_INT v_new_cols = ks_inplace ? 3 : (m+1);
        parcel::parallel_vector<Teigen_vec> V_new_parallel(n, nadd, v_new_cols, que, PARCEL_BASIS_PAD);
        if( (size_t)V_parallel.ld() != stride || (size_t)V_new_parallel.ld() != stride ){
          throw std::logic_error("krylov::ks_arnoldi: basis stride mismatch (PARCEL_BASIS_PAD)");
        }
        parcel::vector<Teigen_vec> V_storage;
        parcel::vector<Teigen_vec> V_new_storage;
        vec_util.setptr(stride*(m+1),      V_parallel.vec_add(),     V_storage,     0);
        vec_util.setptr(stride*v_new_cols, V_new_parallel.vec_add(), V_new_storage, 0);
        // リスタートのたびに V と V_new を入れ替える (コピーしない) ためのフラグ。
        //   V_parallel / V_new_parallel はここの view 作成にしか使わず、ループ内は
        //   毎回 V_storage から列 view を取り直すので、view を張り替えるだけで足りる。
        bool v_swapped = false;
        parcel_blas.template zero_init<Teigen_vec>(V_storage);
        parcel_blas.template zero_init<Teigen_vec>(V_new_storage);
        // z_precon: 前処理出力 z = M^{-1} v_j を一時格納
        //   GMRES と同パターン: z_parallel.vec_add(0) を SpMV/precon の引数に直接使う
        //   (setptr view 経由だと halo exchange の context が伝わらない可能性あり)
        parcel::parallel_vector<Teigen_vec> z_parallel(n, nadd, 1, que);
        // halo 込み view (precon ↔ SpMV 用); 一度だけ setptr (zero_init 用)
        parcel::vector<Teigen_vec> z_add_for_init;
        vec_util.setptr(n + nadd, z_parallel.vec_add(), z_add_for_init, 0);
        parcel_blas.template zero_init<Teigen_vec>(z_add_for_init);
        // work は local のみ (halo 不要、初期 v_0 の正規化の temp)
        //   z の局所部分を借りる。work を使うのは下の初期 v_0 の正規化だけで、
        //   z は各ステップで前処理が全要素を書いてから読むので、ここで書いた値は結果に残らない。
        parcel::vector<Teigen_vec> work;
        vec_util.setptr(n, z_parallel.vec_add(), work, 0);

        // 初期 v_0: EigenVec の最初の n entries
        {
          parcel::vector<Teigen_vec> v0, ev_in;
          vec_util.setptr(n, V_storage, v0, 0);
          vec_util.setptr(n, EigenVec,  ev_in, 0);
          parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(one_vec, zero_vec, ev_in, v0);
        }
        // v_0 正規化
        {
          parcel::vector<Teigen_vec> v0;
          vec_util.setptr(n, V_storage, v0, 0);
          parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(v0, v0);
          double nrm = std::sqrt(parcel_blas.get_dot().get_T());
          // 初期ベクトルのノルムが 0 か非有限なら 1/0 = inf で 0*inf = NaN になるので、
          //   既定の sin/cos 種 (グローバル行番号から作る。power と同じ) に置き換える。
          //   nrm は allreduce 済みなので全ランクが同じ分岐を取る。
          if (!(nrm > 0.0) || !std::isfinite(nrm)) {
            if (myrank == 0)
              std::cout << " [KS] initial vector norm is zero or not finite; replaced by the default seed" << std::endl;
#if PARCEL_MPI
            const long long base = global_row_start(comm->mpi_comm(), (long long)n);
#else
            const long long base = 0;
#endif
            std::vector<double> seed((size_t)n);
            for (PARCEL_INT i = 0; i < n; i++) seed[(size_t)i] = default_seed_value(base + i);
            v0.setValue(seed.data());
            parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(v0, v0);
            nrm = std::sqrt(parcel_blas.get_dot().get_T());
          }
          double inv = 1.0 / nrm;
          scal_vec.setValue(&inv);
          parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, zero_vec, v0, work);
          parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(one_vec, zero_vec, work, v0);
        }

        // ========== Shift-invert 内部 pgmres + precon を一回だけ構築 (nested 高速化) ==========
        //   毎 Arnoldi step で pgmres/precon_matrix を再構築すると、BLOCKJACOBI 等の
        //   factor 化が毎回走り異常 overhead に。事前構築 → 全 step で再利用。
        std::unique_ptr<parcel::krylov::pgmres<Tspmv, Tprecon, Taxpy, Tdot>> shift_solver;
        std::unique_ptr<parcel::precon_matrix<Tmat, Teigen_vec, Tprecon_mat, Tprecon_vec, Tprecon>> shift_precon;
        if(shift_invert){
          shift_solver.reset(new parcel::krylov::pgmres<Tspmv, Tprecon, Taxpy, Tdot>());
          shift_solver->set_tolerance(get_linsolve_tolerance());
          shift_solver->set_max_iteration(get_linsolve_max_iteration());
          shift_solver->set_restart(get_linsolve_restart());
          shift_solver->set_qr_algo(parcel::qr::QR_MGS);
          // 内部 pgmres の前処理:
          //   A K^-1 モード : K (set_precon_algo 等。下で precon_Mat をそのまま渡す)
          //   A のモード    : set_linsolve_precon_algo 等 (既定はヤコビ前処理) で A - σI から作る
          //   K の設定は get_outer_precon_param() から取る。A K^-1 モードでは
          //   利用者の K、ShiftMat なしの A の shift-invert では前処理なし (K = I)。
          copy_precon_param(shift_pspec ? *get_outer_precon_param() : *get_linsolve_precon_param(),
                            *shift_solver->get_precon_param());
          if(shift_identity && linsolve_precon_is_set() &&
             get_linsolve_precon_param()->get_precon_algo() != PRECON_NONE && myrank == 0){
            std::cout << " [KS] note: shift-invert without ShiftMat uses K = I; set_linsolve_precon_algo is not applied (pass ShiftMat = A - sigma I to use it)" << std::endl;
          }
          if(shift_pspec){
            // A K^-1 モード: K は上で A から作った precon_Mat をそのまま使う
            //   (シフトなしの ks / power と同じ K)。内部 pgmres は z を返しシフト σ を持つ。
            shift_solver->set_precon_space_output(true);
            shift_solver->set_shift(shift_sigma_);
          } else {
          // precon_matrix を (A-σI) (= *ShiftMat) で一度だけ構築 → 全 Arnoldi step で再利用
          shift_precon.reset(new parcel::precon_matrix<
            Tmat, Teigen_vec, Tprecon_mat, Tprecon_vec, Tprecon
            >(*ShiftMat, *(shift_solver->get_precon_param())));
          }
        }

        timer_solver_init.end();
        timer_solver_main.start();

        // ========== KS restart loop ==========
        std::vector<double> H((m+1)*m, 0.0);          // col-major
        std::vector<double> Hcopy(m*m, 0.0);
        std::vector<double> Z(m*m, 0.0);
        std::vector<double> wr(m, 0.0), wi(m, 0.0);
        //   再始動の基底縮約 V_new = V·Z に渡す変換行列の置き場。保持数 k_new は
        //   リスタートごとに変わり得るが、最大でも m 列なので m×m で一度だけ確保して
        //   使い回す。毎回 resize すると 1 回につき解放と確保と同期が入る。
        //   使わない列はゼロのまま置き、gemm は先頭 k_new 列 (ldB=m) しか読まない。
        std::vector<double>  Z_col_buf((size_t)m*m, 0.0);
        parcel::vector<Tdot> Zdev((size_t)m*m, que);

        PARCEL_INT k_current = 0;
        bool converged = false;
        //   Arnoldi が不変部分空間に達した (h_{j+1,j} ~ 0) ときの有効次元。
        //   通常は m。破綻したリスタートでは j+1 になり、密行列の処理はこの次元で行う。
        PARCEL_INT md = m;
        bool st_invariant = false;
        // 収束状態 (krylov_param::set_status)
        //   st_breakdown : h_{j+1,j} = 0 (Arnoldi の破綻) または LAPACK の失敗を検出した
        //   st_nonfinite : 判定対象の Ritz 値・残差推定に NaN / Inf が出た
        //   H は allreduce 済みの内積から作るので、以下の値は全ランクで同じになる
        PARCEL_INT st_its    = 0;
        double     st_res    = std::numeric_limits<double>::quiet_NaN();
        bool st_breakdown    = false;
        bool st_nonfinite    = false;


        // ---- 不変部分空間の扱い ----
        //   h_{j+1,j} ~ 0 で「不変部分空間」と判定した Ritz 対は残差推定が 0 になる。ところが
        //   低精度の基底 (FP16/FP32 の Teigen_vec) では、直交化後の w が桁落ち・アンダーフローで
        //   厳密に 0 になり、偽の不変部分空間になりうる。そこで
        //     A: 不変部分空間に頼る収束は、真の残差 ||Op x - θ x|| / (|θ| ||x||) <= tol を
        //        nev 個すべてで確かめてから受け入れる (Op は Arnoldi と同じ演算子)。
        //     B: 確かめられない (または有効次元 < nev) ときは止めずに、乱数ベクトルを
        //        V_0..V_j に CGS2 で直交化して次の方向にし、h_{j+1,j} = 0 のまま m まで広げる
        //        (ARPACK dgetv0 と同じ。H はブロック上三角になる)。
        //   乱数方向を入れた基底では、残差推定 |h_{m+1,m} Z[m-1,i]| が 0 のブロックが残るので、
        //   その後の収束も真の残差で確かめる (injected_any)。
        //   すべての判定値は allreduce 済みの内積と rank 0 が配った密行列の結果から作るので、
        //   全ランクが同じ分岐を取る。
        bool injected_any = false;      // この solver() で乱数方向を入れたか
        bool inv_verified = false;      // st_invariant で止めたとき、真の残差で確かめたか
        double     inv_res   = 0.0;     // 確かめたときの真の相対残差の最大値
        PARCEL_INT inv_nconv = 0;
#if PARCEL_MPI
        //   乱数方向の種に使うグローバル行番号の先頭 (MPI_Exscan なので全ランクがここで呼ぶ)
        const long long ks_row_base = global_row_start(comm->mpi_comm(), (long long)n);
#else
        const long long ks_row_base = 0;
#endif

        //   直交化: v_{j+1} (V の列 j+1) を V_0..V_j に対して CGS2 (ブロック版) か MGS2 で
        //   直交化する。係数を hcol[0..j] に書く
        //   (1 パス目は代入、2 パス目は加算)。Arnoldi 本体と乱数方向の両方がこれを使う。
        //   戻り値: true なら正規化の内積 <v_{j+1}, v_{j+1}> (MPI 集約済み) を求め済みで get_dot() に入っている
        //   (CPU の MGS の融合経路)。呼び出し側はその場合 dot(Vjp1, Vjp1) を省く。false なら呼び出し側で求める。
        //   方式は set_qr_algo() で選ぶ (QR_CGS → CGS2、QR_MGS → MGS2。既定はビルド時の PARCEL_KS_ORTHO_DEFAULT。
        //   先頭の「直交化」の説明)。設定は全ランクで同じ値を与える前提 (他の設定関数と同じ)。
        const bool ks_use_cgs2 = (ks_qr_algo_ == parcel::qr::QR_CGS);
        auto ks_orthogonalize = [&](PARCEL_INT j, double *hcol) -> bool {
          parcel::vector<Teigen_vec> Vjp1;
          vec_util.setptr(n, V_storage, Vjp1, stride*(j+1));
          if (ks_use_cgs2) {
            // ブロック直交化: 古典グラムシュミットを 2 回 (CGS2)。
            //   1 パスを局所 dot (j+1 回) + allreduce 1 回 + gemv 1 回にまとめ、
            //   ベクトルごとに dot / axpy / allreduce を呼ぶ MGS より通信と呼び出しを減らす。
            //   gmres の CGS 経路 (parcel_krylov_gmres.hpp) と parcel_qr_cgs.hpp が同じ形。
            //
            //   注意 1: gemv は N と lda を「渡した y のビュー長」から決める (parcel_blas.hpp の
            //           gemv_main)。V の列ストライドは stride なので、y にはストライド長の
            //           ビューを渡す。局所長 n のビューを渡すと lda がずれて壊れる。
            //   注意 2: そのため gemv はハロ領域の行まで計算する。列 j+1 のハロには意味のない
            //           値が入るが、ハロは次の SpMVcomm で必ず詰め直され、dot と正規化は
            //           局所長 n のビューしか見ないので影響しない。
            for (int pass = 0; pass < 2; pass++) {
              // h = V[:,0..j]^T * v_{j+1}
              //   縮約そのものは既存の dot カーネルに任せる。gemm は出力要素 1 個につき
              //   1 ワークアイテムしか立てないため、この形 (出力 j+1 個) では並列度が出ない。
              //   allreduce だけを 1 回にまとめる (1 Arnoldi step あたり 2 回)。
              timer_solver_dot.start();
#if PARCEL_KS_FUSED
              // CPU: j+1 本の局所内積を 1 本のカーネルで取る。各内積の値は下の経路 (内積ごとに 1 本) と
              //   ビット単位で同じ (同じ分割・同じ足し順)。1 パスのカーネル起動が j+1 回から 1 回になる。
              if constexpr (!std::is_same<Teigen_vec,parcel_doubledouble>::value) {
                parcel::vector<Teigen_vec> V0;
                vec_util.setptr(n, V_storage, V0, 0);
                parcel_blas.template multi_dot_local<Teigen_vec,Teigen_vec,Teigen_vec>(
                    V0, (size_t)stride, j+1, (size_t)n, Vjp1, h_host);
              } else
#endif
              {
              for (PARCEL_INT i = 0; i <= j; i++) {
                parcel::vector<Teigen_vec> Vi;
                vec_util.setptr(n, V_storage, Vi, stride*i);
                //   第 3 引数 false で MPI 集約を省き、局所値だけを受け取る
                parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vi, Vjp1, false);
                h_host[(size_t)i] = parcel_blas.get_dot();
              }
              }
#if PARCEL_MPI
              h_tmp.setValue(h_host);
              comm->template allreduce<Tdot,Tdot>(h_tmp, h_dev, (int)(j+1), MPI_SUM);
              h_dev.getValue(h_host);
#else
              h_dev.setValue(h_host);
#endif
              timer_solver_dot.end();
              // H への積み上げ。ホストへ戻すのはこの 1 回だけ。
              h_dev.getValue(h_host);
              for (PARCEL_INT i = 0; i <= j; i++) {
                const double hv = h_host[(size_t)i].get_T();
                if (pass == 0) hcol[i]  = hv;
                else           hcol[i] += hv;
              }
              // v_{j+1} <- v_{j+1} - V[:,0..j] * h   (alpha=-1 なので符号反転は不要)
              timer_solver_gemv.start();
              {
                parcel::vector<Teigen_vec> V_add, Vjp1_add;
                vec_util.setptr(stride*(j+1), V_storage, V_add,    0);
                vec_util.setptr(stride,       V_storage, Vjp1_add, stride*(j+1));
                parcel_blas.template gemv<Tdot,Tdot,Teigen_vec,Tdot,Teigen_vec,Taxpy>(
                    j+1, nega_one_vec, one_vec, V_add, h_dev, Vjp1_add);
              }
              timer_solver_gemv.end();
            }
          } else {
            // MGS (2 回)。set_qr_algo(parcel::qr::QR_MGS) で選ぶ。
            //   CPU (PARCEL_GS_FUSED) では pgmres の MGS と同じく「i-1 列目の axpy」と「i 列目の内積」を 1 本のカーネルで行い、
            //   1 パス目の最後の axpy は 2 パス目の最初の内積と、2 パス目の最後の axpy は正規化の内積とまとめる。
            //   各演算と順序は下の融合しない MGS (内積 -> axpy を列ごと、2 パス、そのあと呼び出し側の正規化の内積) と
            //   同じなので、結果はビット単位で一致する (内積の積の順 V*w と w*V は同じ値)。
            //   係数は値渡し (axpy_value / axpy_dot_value)。時間は内積に計上する (axpy の分を含む)。
#if PARCEL_GS_FUSED
            if constexpr (!std::is_same<Teigen_vec,parcel_doubledouble>::value && !std::is_same<Tdot,parcel_doubledouble>::value
                          && !std::is_same<Taxpy,parcel_doubledouble>::value) {
              parcel::vector<Teigen_vec> Vi, Vprev;
              parcel::scalar a_prev;
              bool have_prev = false;
              for (int pass = 0; pass < 2; pass++) {
                for (PARCEL_INT i = 0; i <= j; i++) {
                  vec_util.setptr(n, V_storage, Vi, stride*i);
                  timer_solver_dot.start();
                  if (!have_prev) parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vi, Vjp1);
                  else            parcel_blas.template axpy_dot_value<Tdot,Teigen_vec,Taxpy,Teigen_vec>(a_prev, Vprev, Vjp1, Vi);
                  timer_solver_dot.end();
                  const double h = parcel_blas.get_dot().get_T();
                  if (pass == 0) hcol[i]  = h;
                  else           hcol[i] += h;
                  a_prev = parcel::scalar(-h);
                  vec_util.setptr(n, V_storage, Vprev, stride*i);
                  have_prev = true;
                }
              }
              // 2 パス目の最後の axpy と正規化の内積 <v_{j+1}, v_{j+1}> (集約込み)
              timer_solver_dot.start();
              parcel_blas.template axpy_dot_value<Tdot,Teigen_vec,Taxpy,Teigen_vec>(a_prev, Vprev, Vjp1, Vjp1);
              timer_solver_dot.end();
              return true;
            }
#endif
            // MGS 直交化 (local only; parcel_blas.dot は MPI_Allreduce 内蔵)
            for (PARCEL_INT i = 0; i <= j; i++) {
              parcel::vector<Teigen_vec> Vi;
              vec_util.setptr(n, V_storage, Vi, stride*i);
              timer_solver_dot.start();
              parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vi, Vjp1);
              double h_ij = parcel_blas.get_dot().get_T();
              timer_solver_dot.end();
              hcol[i] = h_ij;
              double neg = -h_ij;
              timer_solver_axpy.start();
              // 係数を値で渡す (1 要素のデバイス配列への書き込みと同期を省く。結果は同じ)
              if constexpr (!std::is_same<Teigen_vec,parcel_doubledouble>::value
                            && !std::is_same<Tdot,parcel_doubledouble>::value && !std::is_same<Taxpy,parcel_doubledouble>::value) {
                parcel_blas.template axpy_value<Tdot,Teigen_vec,Teigen_vec,Taxpy>(parcel::scalar(neg), Vi, Vjp1);
              } else {
                scal_vec.setValue(&neg);
                parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, one_vec, Vi, Vjp1);
              }
              timer_solver_axpy.end();
            }
            // 再直交化 (CGS2 / DGKS): 多 restart + 縮重スペクトルでの orthogonality loss 補償
            //   1 pass 目の MGS で生じた残差成分を 2 pass 目で除去 → V_m の直交性保持
            //   コスト ~2x だが ghost eigenvalue 抑制に必須
            for (PARCEL_INT i = 0; i <= j; i++) {
              parcel::vector<Teigen_vec> Vi;
              vec_util.setptr(n, V_storage, Vi, stride*i);
              timer_solver_dot.start();
              parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vi, Vjp1);
              double corr = parcel_blas.get_dot().get_T();
              timer_solver_dot.end();
              hcol[i] += corr;
              double neg = -corr;
              timer_solver_axpy.start();
              // 係数を値で渡す (1 要素のデバイス配列への書き込みと同期を省く。結果は同じ)
              if constexpr (!std::is_same<Teigen_vec,parcel_doubledouble>::value
                            && !std::is_same<Tdot,parcel_doubledouble>::value && !std::is_same<Taxpy,parcel_doubledouble>::value) {
                parcel_blas.template axpy_value<Tdot,Teigen_vec,Teigen_vec,Taxpy>(parcel::scalar(neg), Vi, Vjp1);
              } else {
                scal_vec.setValue(&neg);
                parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, one_vec, Vi, Vjp1);
              }
              timer_solver_axpy.end();
            }
          }
          return false;
        };

        //   Arnoldi と同じ演算子 Op を 1 回かける: y = Op x。
        //   x_add は halo 込み (n+nadd) の view、x_loc / y_loc は局所 (n) の view。
        //   shift-invert では内部 pgmres で (A-σI) y = x (A K^-1 モードは (A K^-1 - σI))、
        //   それ以外では y = A K^-1 x (前処理なしなら y = A x)。下の Arnoldi 本体と同じ手順
        //   (本体はタイマーを区間ごとに分けるため展開して書いている。変えるときは両方そろえる)。
        auto ks_apply_op = [&](parcel::vector<Teigen_vec> &x_add,
                               parcel::vector<Teigen_vec> &x_loc,
                               parcel::vector<Teigen_vec> &y_loc) {
          if (shift_invert) {
            parcel_blas.template zero_init<Teigen_vec>(y_loc);
            shift_solver->template solver<
              Tmat, Teigen_vec, Teigen_vec, Teigen_vec,
              Tprecon_mat, Tprecon_vec, Teigen_vec, Teigen_vec
            >(shift_pspec ? Mat : *ShiftMat, y_loc, x_loc,
              shift_pspec ? &precon_Mat : shift_precon.get());
            count_linsolve(*shift_solver);
          } else {
            {
              parcel::vector<Teigen_vec> z_view_in;
              vec_util.setptr(n + nadd, z_parallel.vec_add(), z_view_in, 0);
              precon_Mat.solver(x_add, z_view_in);
            }
            parcel_blas.template SpMV<Tdot,Tdot,Tmat,Teigen_vec,Teigen_vec,Tspmv>(one_vec, zero_vec, Mat, z_parallel.vec_add(0), y_loc);
          }
        };

        //   真の残差で Ritz 対を確かめる (A)。H の先頭 dim×dim ブロックを rank 0 が dgeev で
        //   分解して配り、|θ| 上位 nev 個について x = V_dim y (出力の Ritz ベクトルと同じ作り方)、
        //   r = Op x - θ x を作る。複素共役対 θ = a + ib、y = yr + i yi では
        //     rr = Op xr - a xr + b xi,  ri = Op xi - b xr - a xi,
        //     ||r||^2 = ||rr||^2 + ||ri||^2,  ||x||^2 = ||xr||^2 + ||xi||^2。
        //   相対残差 ||r|| / (|θ| ||x||) <= tol を満たした数を返し、最大値を worst に入れる。
        //   ノルムはすべて allreduce 済みの内積。念のため結果も rank 0 の値を配る。
        //   作業域には V_new_storage の列 0..2 を使う (m >= 2 なので 3 列ある。V_new は
        //   リスタートの縮約で上書きされるまで使っていない)。V の中で縮約する
        //   ときは V_new はこの作業域 3 列だけで、縮約には使わない。
        auto ks_verify_true_residual = [&](PARCEL_INT dim, double &worst) -> PARCEL_INT {
          worst = 0.0;
          if (dim < nev) { worst = std::numeric_limits<double>::infinity(); return 0; }
          std::vector<double> Hv((size_t)dim*dim), vwr((size_t)dim, 0.0), vwi((size_t)dim, 0.0), vZ((size_t)dim*dim, 0.0);
          for (PARCEL_INT i = 0; i < dim; i++) for (PARCEL_INT j2 = 0; j2 < dim; j2++)
            Hv[(size_t)i + (size_t)j2*dim] = H[i + j2*(m+1)];
          int info_v = 0;
          if (myrank == 0)
            info_v = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', dim,
                                   Hv.data(), dim, vwr.data(), vwi.data(),
                                   NULL, dim, vZ.data(), dim);
          ks_bcast_int(&info_v, comm_of(Mat));
          if (info_v != 0) {
            if (myrank == 0) std::cout << "[KS] dgeev info=" << info_v << " in the true-residual check" << std::endl;
            worst = std::numeric_limits<double>::infinity();
            return 0;
          }
          ks_bcast(vwr.data(), (int)dim, comm_of(Mat));
          ks_bcast(vwi.data(), (int)dim, comm_of(Mat));
          ks_bcast(vZ.data(),  (int)(dim*dim), comm_of(Mat));
          std::vector<int> vidx((size_t)dim);
          for (int i = 0; i < (int)dim; i++) vidx[i] = i;
          std::stable_sort(vidx.begin(), vidx.end(), [&](int a, int b) {
            return ks_mag_key(vwr[a], vwi[a]) > ks_mag_key(vwr[b], vwi[b]);
          });
          parcel::vector<Teigen_vec> xr_add, xr, xi_add, xi, y;
          vec_util.setptr(n + nadd, V_new_storage, xr_add, 0);
          vec_util.setptr(n,        V_new_storage, xr,     0);
          vec_util.setptr(n + nadd, V_new_storage, xi_add, stride);
          vec_util.setptr(n,        V_new_storage, xi,     stride);
          vec_util.setptr(n,        V_new_storage, y,      stride*2);
          //   x = sgn * V_dim * vZ[:, col]
          auto form_x = [&](parcel::vector<Teigen_vec> &dst, int col, double sgn) {
            parcel_blas.template zero_init<Teigen_vec>(dst);
            for (PARCEL_INT i = 0; i < dim; i++) {
              double zi = sgn * vZ[(size_t)i + (size_t)col*dim];
              if (std::fabs(zi) < 1e-300) continue;
              parcel::vector<Teigen_vec> v_src;
              vec_util.setptr(n, V_storage, v_src, stride*i);
              scal_vec.setValue(&zi);
              parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, one_vec, v_src, dst);
            }
          };
          //   dst += c * src
          auto add_to = [&](parcel::vector<Teigen_vec> &src, double c, parcel::vector<Teigen_vec> &dst) {
            scal_vec.setValue(&c);
            parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, one_vec, src, dst);
          };
          auto nrm2 = [&](parcel::vector<Teigen_vec> &v) -> double {
            parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(v, v);
            return parcel_blas.get_dot().get_T();
          };
          PARCEL_INT npass = 0;
          for (PARCEL_INT ii = 0; ii < nev; ii++) {
            const int i = vidx[(size_t)ii];
            const double a = vwr[i], b = vwi[i];
            const bool cplx = !(std::fabs(b) < 1e-15);   // 収束判定と同じ閾値
            double rel = std::numeric_limits<double>::infinity();
            int ire = i, iim = -1;
            double sgn = 1.0;
            bool ok_idx = true;
            if (cplx) {
              //   dgeev: wi > 0 の列 i が実部、i+1 が虚部。wi < 0 はその共役 (虚部の符号が逆)
              if (b > 0) { ire = i;   iim = i+1; sgn =  1.0; }
              else       { ire = i-1; iim = i;   sgn = -1.0; }
              if (ire < 0 || iim >= (int)dim) ok_idx = false;
            }
            if (ok_idx) {
              form_x(xr, ire, 1.0);
              double xn2 = nrm2(xr);
              ks_apply_op(xr_add, xr, y);
              add_to(xr, -a, y);
              double rn2;
              if (cplx) {
                form_x(xi, iim, sgn);
                xn2 += nrm2(xi);
                add_to(xi, b, y);             // rr = Op xr - a xr + b xi
                rn2 = nrm2(y);
                ks_apply_op(xi_add, xi, y);
                add_to(xi, -a, y);
                add_to(xr, -b, y);            // ri = Op xi - b xr - a xi
                rn2 += nrm2(y);
              } else {
                rn2 = nrm2(y);
              }
              const double mag = std::sqrt(a*a + b*b);
              rel = std::sqrt(rn2) / (mag * std::sqrt(xn2));
              if (!(mag >= 1e-300)) rel = std::numeric_limits<double>::infinity();
            }
            if (rel <= tol) npass++;
            if (!(rel <= worst)) worst = rel;   // NaN も最大側に残す
          }
          ks_bcast(&worst, 1, comm_of(Mat));
          ks_bcast_int(&npass, comm_of(Mat));
          return npass;
        };

        // 注: 縮重スペクトルでの ghost eigenvalue 抑制は ks_orthogonalize の CGS2 で
        //     完結しており、locking/deflation は不要。

        for (int restart = 0; restart < max_restart; restart++) {
          st_its = restart + 1;

          // -------- Arnoldi 拡張: k_current → m --------
          int n_random = 0;   // この Arnoldi 過程で入れた乱数方向の数
          for (PARCEL_INT j = k_current; j < m; j++) {
            parcel::vector<Teigen_vec> Vj, Vj_add, Vjp1;
            // local part (size n) for dot/axpy
            vec_util.setptr(n,        V_storage, Vj,     stride*j);
            // halo-inclusive part (size n+nadd) for SpMV input (SpMVcomm fills halo)
            vec_util.setptr(n + nadd, V_storage, Vj_add, stride*j);
            vec_util.setptr(n,        V_storage, Vjp1,   stride*(j+1));
            if (shift_invert) {
              // Shift-invert: 事前構築済 shift_solver + shift_precon で線形解
              //   precon_matrix を再利用するため BLOCKJACOBI factor 化が 1 回のみ
              timer_solver_SpMV.start();
              parcel_blas.template zero_init<Teigen_vec>(Vjp1);
              shift_solver->template solver<
                Tmat, Teigen_vec, Teigen_vec, Teigen_vec,
                Tprecon_mat, Tprecon_vec, Teigen_vec, Teigen_vec
              >(shift_pspec ? Mat : *ShiftMat, Vjp1, Vj,
                shift_pspec ? &precon_Mat : shift_precon.get());   // 外部 precon を渡す (再利用)
              count_linsolve(*shift_solver);
              timer_solver_SpMV.end();
            } else {
              // Step 1: 前処理 z = M^{-1} v_j (precon=NONE なら identity copy)
              //   z_parallel.vec_add(0) を毎回再取得 (GMRES と同パターン)
              timer_solver_precon.start();
              {
                parcel::vector<Teigen_vec> z_view_in;
                vec_util.setptr(n + nadd, z_parallel.vec_add(), z_view_in, 0);
                precon_Mat.solver(Vj_add, z_view_in);
              }
              timer_solver_precon.end();
              // Step 2: SpMV v_{j+1} = A z
              //   z_parallel.vec_add(0) 経由で SpMVcomm の halo exchange context が有効になる
              timer_solver_SpMV.start();
              parcel_blas.template SpMV<Tdot,Tdot,Tmat,Teigen_vec,Teigen_vec,Tspmv>(one_vec, zero_vec, Mat, z_parallel.vec_add(0), Vjp1);
              timer_solver_SpMV.end();
            }

            // 直交化。乱数方向でも同じ手順を使う。
            const bool norm_ready = ks_orthogonalize(j, &H[j*(m+1)]);
            timer_solver_dot.start();
            if (!norm_ready) parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vjp1, Vjp1);
            double h_jp1 = std::sqrt(parcel_blas.get_dot().get_T());
            timer_solver_dot.end();
            H[(j+1) + j*(m+1)] = h_jp1;
            // 破綻は相対値で判定する。
            //   直交化前の ||w|| は H の列から通信なしで求まる (||w||^2 = sum_i h_ij^2 + h_{j+1,j}^2)。
            //   h_{j+1,j} <= 1e-14 ||w|| なら Krylov 部分空間が不変になったとみなし、
            //   拡張を止めて有効次元 md = j+1 で密行列の処理に進む (その Ritz 対は残差 0)。
            //   h と H は allreduce 済みの内積から作るので、全ランクが同じ分岐を取る。
            {
              double wb2 = h_jp1 * h_jp1;
              for (PARCEL_INT i = 0; i <= j; i++) wb2 += H[i + j*(m+1)] * H[i + j*(m+1)];
              if (!std::isfinite(h_jp1) || !std::isfinite(wb2)) {
                if (myrank == 0) std::cout << "[KS] non-finite value in Arnoldi at j=" << j << std::endl;
                st_nonfinite = true;
                break;
              }
              if (h_jp1 == 0.0 || h_jp1 <= 1.0e-14 * std::sqrt(wb2)) {
                if (myrank == 0) std::cout << "[KS] breakdown j=" << j << " (invariant subspace, dimension " << j+1 << ")" << std::endl;
                H[(j+1) + j*(m+1)] = 0.0;
                // (A): 残差推定 0 をそのまま信じず、真の残差で確かめる。
                //   H はこの時点では各ランクが allreduce 済みの内積から作ったもの。検査の中で
                //   rank 0 のブロックを配ってから分解するので、判定は全ランクで同じになる。
                if (j + 1 >= nev) {
                  double worst = 0.0;
                  const PARCEL_INT np = ks_verify_true_residual(j + 1, worst);
                  if (np == nev) {
                    if (myrank == 0)
                      std::cout << "[KS] invariant subspace at j=" << j << " verified (max true residual "
                                << ks_sci(worst) << " <= tol " << ks_sci(tol) << ")" << std::endl;
                    md = j + 1;
                    st_invariant = true;
                    inv_verified = true;
                    inv_res = worst;
                    inv_nconv = np;
                    break;
                  }
                  if (myrank == 0)
                    std::cout << "[KS] invariant subspace at j=" << j << " not verified (true residual "
                              << ks_sci(worst) << " > tol " << ks_sci(tol)
                              << ", " << np << "/" << nev << " pairs pass), continuing with a random vector"
                              << std::endl;
                } else if (myrank == 0) {
                  std::cout << "[KS] invariant subspace at j=" << j << " has dimension " << j+1
                            << " < nev = " << nev << ", continuing with a random vector" << std::endl;
                }
                // (B): 止めずに乱数方向で広げる (ARPACK dgetv0 と同じ)。
                //   新しい方向は v_j と結合しないので h_{j+1,j} = 0 のまま (H はブロック上三角)。
                //   無限ループを避けるため 1 回の Arnoldi 過程で ks_max_random_dir 回まで。
                if (n_random >= ks_max_random_dir) {
                  if (myrank == 0)
                    std::cout << "[KS] random restarts of the Arnoldi direction exhausted ("
                              << ks_max_random_dir << "); stopping with dimension " << j+1 << std::endl;
                  md = j + 1;
                  st_invariant = true;
                  inv_verified = false;
                  break;
                }
                n_random++;
                {
                  //   種: グローバル行番号 + (リスタート番号, 列番号, 試行回数) の通し番号
                  const long long cseed = ((long long)restart * (long long)(m + 1) + (long long)(j + 1)) * 4LL
                                          + (long long)n_random;
                  std::vector<double> rv((size_t)n);
                  for (PARCEL_INT i = 0; i < n; i++) rv[(size_t)i] = ks_random_value(ks_row_base + i, cseed);
                  Vjp1.setValue(rv.data());
                }
                parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vjp1, Vjp1);
                const double r0 = std::sqrt(parcel_blas.get_dot().get_T());
                std::vector<double> hdummy((size_t)(j + 1), 0.0);   // 係数は捨てる (結合しない)
                if (!ks_orthogonalize(j, hdummy.data()))
                  parcel_blas.template dot<Teigen_vec,Teigen_vec,Teigen_vec>(Vjp1, Vjp1);
                double r1 = std::sqrt(parcel_blas.get_dot().get_T());
                //   直交化で乱数ベクトルがほぼ消えた = V_0..V_j が張る空間で尽きている
                //   (m が実効的な次元を超えた)。閾値は倍精度の丸めより十分大きく取る。
                //   判定値は allreduce 済みだが、念のため rank 0 の値で揃える。
                double rr[2] = { r0, r1 };
                ks_bcast(rr, 2, comm_of(Mat));
                r1 = rr[1];
                if (!std::isfinite(r1) || !(r1 > 1.0e-10 * rr[0])) {
                  if (myrank == 0)
                    std::cout << "[KS] random vector vanished after orthogonalization at j=" << j
                              << " (space exhausted); stopping with dimension " << j+1 << std::endl;
                  md = j + 1;
                  st_invariant = true;
                  inv_verified = false;
                  break;
                }
                injected_any = true;
                double inv_r = 1.0 / r1;
                scal_vec.setValue(&inv_r);
                parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, zero_vec, Vjp1, Vjp1);
                continue;   // H[(j+1) + j*(m+1)] は 0 のまま次の列へ
              }
            }
            double inv = 1.0 / h_jp1;
            scal_vec.setValue(&inv);
            timer_solver_axpy.start();
            //   v <- v / h。y[i] = a*x[i] + b*y[i] は x と y が同じでも要素ごとに閉じているので
            //   その場で 1 パスで行う。
            parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, zero_vec, Vjp1, Vjp1);
            timer_solver_axpy.end();
          }

          // -------- 実 Schur 分解: H_m = Q T Q^T (ROW_MAJOR) --------
          //   dgeev の固有ベクトルは複素共役対に対して正規直交でない
          //   (LAPACK は複素ベクトルとして ||v||=1 に正規化するため、
          //    実部・虚部に分けると各々のノルムは 1 未満で直交もしない)。
          //   これをリスタートの基底に使うと V_new の正規直交性が崩れ、
          //   A V = V H + h v_{m+1} e^T の関係が成り立たなくなる。
          //   Schur ベクトル Q は直交なのでこの問題が起きない。
          //   LAPACKE の row-major ラッパーは内部で転置と作業領域確保を行うが、
          //   dtrsen の job='N' で作業領域が 0 になり segfault する実装がある。
          //   H は元から列優先で持っているので、以降は COL_MAJOR で扱う。
          //
          //   収束判定には H の固有ベクトルが要る (Ritz 残差は
          //   |h_{m+1,m}| * |(固有ベクトルの最終成分)|)。Schur ベクトルでは
          //   この量にならないため、固有分解と Schur 分解の両方を行う。
          //   m は既定 30 なので密行列 2 回分のコストは無視できる。
          if (st_nonfinite) break;   // Arnoldi で非有限値が出たらその場で抜ける

          //   以降のホスト側 LAPACK の結果で全ランクが同じ分岐を取るよう、
          //   rank 0 の値を配る (H、固有分解、Schur 分解の結果。m は数十なので費用は無視できる)。
          //   各ランクが独立に LAPACK を呼ぶと、丸めの差で分岐が割れて集団通信で止まりうる。
          ks_bcast(H.data(), (int)H.size(), comm_of(Mat));
          ks_bcast_int(&md, comm_of(Mat));

          //   有効次元 md (不変部分空間に達したリスタートでは m より小さい) の先頭ブロックを分解する
          for (int i = 0; i < md; i++) for (int j = 0; j < md; j++)
            Hcopy[i + j*md] = H[i + j*(m+1)];
          std::fill(Z.begin(), Z.end(), 0.0);
          std::fill(wr.begin(), wr.end(), 0.0);
          std::fill(wi.begin(), wi.end(), 0.0);
          timer_dense_eig.start();
          int info = 0;
          if (myrank == 0)
            info = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', md,
                                 Hcopy.data(), md, wr.data(), wi.data(),
                                 NULL, md, Z.data(), md);
          ks_bcast_int(&info, comm_of(Mat));
          if (info == 0) {
            ks_bcast(wr.data(), (int)md, comm_of(Mat));
            ks_bcast(wi.data(), (int)md, comm_of(Mat));
            ks_bcast(Z.data(),  (int)(md*md), comm_of(Mat));
          }
          timer_dense_eig.end();
          if (info != 0) {
            if (myrank == 0) std::cout << "[KS] dgeev info=" << info << std::endl;
            st_breakdown = true;
            break;
          }

          //   不変部分空間では h_{md+1,md} = 0 なので Ritz 残差も 0
          double h_mp1_m = (md == m) ? H[m + (m-1)*(m+1)] : 0.0;

          // -------- 収束判定: 上位 nev 個 Ritz residual --------
          //   比較は NaN を最後に回す (NaN を含む比較は strict weak ordering を
          //   満たさず、std::stable_sort の動作が未定義になる)。
          std::vector<int> idx(md);
          for (int i = 0; i < md; i++) idx[i] = i;
          std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) {
            return ks_mag_key(wr[a], wi[a]) > ks_mag_key(wr[b], wi[b]);
          });


          //   Krylov-Schur では残差ベクトルが h_{m+1,m} * v_{m+1} * (Q の最終行) に
          //   なるため、Ritz 対 i の残差は |h_{m+1,m} * Q[m-1,i]|。
          //   複素共役対 (2x2 ブロック) は 2 成分のノルムを取る。
          bool all_conv = true;
          //   収束状態用: 判定対象の相対残差の最大値と、tol を満たした数 (NaN は満たさない側に数える)
          double res_max = 0.0;
          PARCEL_INT nconv = 0;
          bool nonfinite_now = false;
          //   不変部分空間の次元が nev 未満なら nev 個の Ritz 対はそろわない
          if (md < nev) {
            if (myrank == 0) std::cout << "[KS] invariant subspace dimension " << md << " < nev = " << nev << std::endl;
            st_breakdown = true;
            break;
          }
          for (int ii = 0; ii < nev; ii++) {
            int i = idx[ii];
            double resi;
            if (std::fabs(wi[i]) < 1e-15) {
              resi = std::fabs(h_mp1_m * Z[(md-1) + i*md]);
            } else {
              int i2 = (wi[i] > 0) ? i+1 : i-1;
              if (i2 < 0 || i2 >= md) resi = 1e300;
              else resi = std::fabs(h_mp1_m) * std::sqrt(
                Z[(md-1) + i*md]*Z[(md-1) + i*md] +
                Z[(md-1) + i2*md]*Z[(md-1) + i2*md]);
            }
            double mag = std::sqrt(wr[i]*wr[i] + wi[i]*wi[i]);
            //   NaN は「収束した」側に数えない (NaN との比較は偽になるので否定形で書く)
            if (!(mag >= 1e-300) || !(resi/mag <= tol)) all_conv = false;
            if (!std::isfinite(resi) || !std::isfinite(mag)) {
              nonfinite_now = true;
            } else {
              const double rel = resi/mag;    // mag = 0 なら +inf (非有限扱いにはしない)
              if (!(rel <= res_max)) res_max = rel;
              if (!(mag < 1e-300) && rel <= tol) nconv++;
            }
          }
          st_res = nonfinite_now ? std::numeric_limits<double>::quiet_NaN() : res_max;
          ks_nconv_ = nconv;
          //   非有限値が出たらその場で抜ける (DIVERGED_NANORINF)
          if (nonfinite_now) { st_nonfinite = true; break; }
          //   不変部分空間を真の残差で確かめられないまま止めた (乱数方向の上限、
          //   空間が尽きた)。残差推定は 0 でも収束とは扱わない。
          if (st_invariant && !inv_verified) {
            if (myrank == 0) std::cout << "[KS] invariant subspace was not verified; not converged" << std::endl;
            st_breakdown = true;
            break;
          }
          if (all_conv) {
            //   不変部分空間 (Arnoldi 内で真の残差を確認済み) では、状態の residual に
            //   真の相対残差の最大値を入れる。乱数方向を入れた基底では、残差推定が 0 のブロックが
            //   残るので、ここで真の残差を確かめてから受け入れる。
            if (st_invariant) {
              st_res = inv_res;
              ks_nconv_ = inv_nconv;
            } else if (injected_any) {
              double worst = 0.0;
              const PARCEL_INT np = ks_verify_true_residual(md, worst);
              st_res = worst;
              ks_nconv_ = np;
              if (np != nev) {
                if (myrank == 0)
                  std::cout << "[KS] Ritz estimates meet tol at restart " << restart
                            << " but the true residual " << ks_sci(worst) << " > tol " << ks_sci(tol)
                            << " (" << np << "/" << nev << " pairs pass; basis contains a random direction"
                            << " after an unverified invariant subspace); not converged" << std::endl;
                st_breakdown = true;
                break;
              }
              if (myrank == 0)
                std::cout << "[KS] true residual verified at restart " << restart
                          << " (max " << ks_sci(worst) << " <= tol " << ks_sci(tol) << ")" << std::endl;
            }
            converged = true;
            if (myrank == 0) std::cout << " [KS] converged at restart " << restart << std::endl;
            break;
          }
          //   不変部分空間に達したのに判定を満たさない (|λ| ~ 0 など)。これ以上広げられない
          if (st_invariant) {
            st_breakdown = true;
            break;
          }

          // -------- リスタート用の実 Schur 分解 H = Q T Q^T --------
          //   dgeev の固有ベクトルは複素共役対に対して正規直交でない
          //   (LAPACK は複素ベクトルとして ||v||=1 に正規化するため、
          //    実部・虚部に分けると各々のノルムは 1 未満で直交もしない)。
          //   これをリスタートの基底に使うと V_new の正規直交性が崩れ、
          //   A V = V H + h v_{m+1} e^T の関係が成り立たなくなる。
          //   Schur ベクトル Q は直交なのでこの問題が起きない。
          std::vector<double> Ts((size_t)m*m), Qs((size_t)m*m);
          std::vector<double> wr_s(m, 0.0), wi_s(m, 0.0);
          for (int i = 0; i < m; i++) for (int j = 0; j < m; j++)
            Ts[i + j*m] = H[i + j*(m+1)];
          //   リスタート後の H は「T + スパイク行」で上ヘッセンベルグではないため、
          //   ヘッセンベルグ専用の dhseqr ではなく一般行列用の dgees を使う。
          //   (dgees が内部でヘッセンベルグ化してから Schur 分解する)
          lapack_int sdim_s = 0;
          timer_dense_schur.start();
          //   Schur 分解と並べ替えは rank 0 だけが行い、結果を配る (上の dgeev と同じ理由)
          int info_hs = 0;
          if (myrank == 0)
            info_hs = LAPACKE_dgees(LAPACK_COL_MAJOR, 'V', 'N', NULL, m,
                                    Ts.data(), m, &sdim_s,
                                    wr_s.data(), wi_s.data(), Qs.data(), m);
          ks_bcast_int(&info_hs, comm_of(Mat));
          timer_dense_schur.end();
          if (info_hs != 0) {
            if (myrank == 0) std::cout << "[KS] dgees info=" << info_hs << std::endl;
            st_breakdown = true;
            break;
          }
          //   Schur 順序で |λ| 上位を選ぶ (NaN は最後。rank 0 の選択が下の dtrsen に入り、結果は配る)
          std::vector<int> idx_s(m);
          for (int i = 0; i < m; i++) idx_s[i] = i;
          std::stable_sort(idx_s.begin(), idx_s.end(), [&](int a, int b) {
            return ks_mag_key(wr_s[a], wi_s[a]) > ks_mag_key(wr_s[b], wi_s[b]);
          });

          // -------- 上位 k_keep を Schur 形の先頭へ並べ替える (dtrsen) --------
          //   select は 2x2 ブロックの両方を同時に選ぶ必要がある (LAPACK の要件)。
          std::vector<lapack_logical> sel_flag(m, 0);
          {
            int nsel = 0;
            std::vector<bool> picked(m, false);
            for (int ii = 0; ii < m && nsel < k_keep; ii++) {
              int i = idx_s[ii];
              if (picked[i]) continue;
              if (std::fabs(wi_s[i]) >= 1e-12) {
                int i2 = (wi_s[i] > 0) ? i+1 : i-1;
                if (i2 < 0 || i2 >= m || picked[i2]) continue;
                sel_flag[i] = sel_flag[i2] = 1;
                picked[i] = picked[i2] = true;
                nsel += 2;
              } else {
                sel_flag[i] = 1; picked[i] = true; nsel += 1;
              }
            }
          }
          lapack_int nsel_out = 0;
          double s_cond = 0.0, sep_cond = 0.0;
          timer_dense_schur.start();   // dtrsen も Schur 側の費用として同じタイマーに足す
          //   LAPACKE_dtrsen は作業領域を問い合わせで決めるが、job='N' では
          //   OpenBLAS が 0 を返し malloc(0) の結果を LAPACK に渡して segfault する。
          //   作業領域を明示して _work 版を直接呼ぶ。
          //   job='N' の要件: lwork >= max(1,n), liwork >= 1
          lapack_int lwork_tr  = (m > 1) ? (lapack_int)m : 1;
          lapack_int liwork_tr = 1;
          std::vector<double>     work_tr((size_t)lwork_tr);
          std::vector<lapack_int> iwork_tr((size_t)liwork_tr);
          int info_tr = 0;
          if (myrank == 0)
            info_tr = LAPACKE_dtrsen_work(LAPACK_COL_MAJOR, 'N', 'V', sel_flag.data(), m,
                                          Ts.data(), m, Qs.data(), m,
                                          wr_s.data(), wi_s.data(), &nsel_out,
                                          &s_cond, &sep_cond,
                                          work_tr.data(), lwork_tr,
                                          iwork_tr.data(), liwork_tr);
          ks_bcast_int(&info_tr, comm_of(Mat));
          if (info_tr == 0) {
            int nsel_i = (int)nsel_out;
            ks_bcast_int(&nsel_i, comm_of(Mat));
            nsel_out = (lapack_int)nsel_i;
            ks_bcast(Ts.data(), (int)Ts.size(), comm_of(Mat));
            ks_bcast(Qs.data(), (int)Qs.size(), comm_of(Mat));
          }
          timer_dense_schur.end();
          if (info_tr != 0) {
            if (myrank == 0) std::cout << "[KS] dtrsen info=" << info_tr << std::endl;
            st_breakdown = true;
            break;
          }
          //   並べ替え後、先頭 nsel_out 個が保持する Ritz 値。
          //   Hcopy = 並べ替え後の T、Z = 更新された Q (直交のまま)。
          PARCEL_INT k_new = (PARCEL_INT)nsel_out;
          if (k_new <= 0 || k_new >= m) {
            if (myrank == 0) std::cout << "[KS] dtrsen で保持数が不正 (" << k_new << ")" << std::endl;
            st_breakdown = true;
            break;
          }
          timer_restart_vec.start();
          std::vector<double> Z_new((size_t)m*k_new);
          for (int j = 0; j < k_new; j++)
            for (int i = 0; i < m; i++) Z_new[i*k_new + j] = Qs[i + j*m];
          timer_restart_vec.end();

          // -------- V_new = V_storage · Z_new[:, 0:k_new] (local 部分のみ) --------
          //   1 列ずつ axpy を最大 m*k_new 回呼ぶ代わりに gemm 1 回にまとめ、
          //   V を 1 度読んで V_new を 1 度書くだけで済ませる。
          //   規約は cagmres と同じ C[m,n] = alpha*A[m,k]*B[k,n] + beta*C[m,n]。
          //   beta=0 で C を上書きするので、事前の zero_init は要らない。
          {
            // Z_new は行優先 (i*k_new + j) で持っているので、gemm が期待する列優先 (ldB=m) に詰め直す。
            //   領域はリスタート間で使い回す (Z_col_buf / Zdev はループ外で m×m 確保済み)。
            //   k_new が前回より小さいときに古い値が残らないよう、詰める前にゼロで埋める。
            std::fill(Z_col_buf.begin(), Z_col_buf.end(), 0.0);
            for (int j = 0; j < k_new; j++)
              for (int i = 0; i < m; i++) Z_col_buf[(size_t)i + (size_t)j*m] = Z_new[i*k_new + j];
            Zdev.setValue(Z_col_buf.data());
            //   最後の "T" は縦長経路 (tgemm) の指定。V は n 行 x m 列 (n >> m) なので、
            //   出力の列ごとに V を読み直す既定経路ではなく、1 行 = 1 ワークアイテムで V を
            //   1 回だけ読み、k_new 個の出力へ同時に積む縦長経路を使う。
            //   この経路は beta を使わないが、ks は beta=0 (上書き) なので意味は変わらない。
            // 同じ配列への出力は tgemm の累算器 (32 列) に収まるときだけ (上の ks_inplace の条件)
            if (ks_inplace && k_new > 32)
              throw std::logic_error("krylov::ks_arnoldi: in-place restart needs k_new <= 32 (got "
                                     + std::to_string((long long)k_new) + ")");
            parcel_blas.template gemm<Tdot,Tdot,Teigen_vec,Tdot,Teigen_vec,Tdot>(
                n, k_new, m, one_vec, zero_vec,
                V_storage,     stride,
                Zdev,          m,
                ks_inplace ? V_storage : V_new_storage, stride,
                "N", "N", "N", "N", "T");
          }
          // v_{m+1} を V_new[:, k_new] にコピー (次 Arnoldi 開始 vec)
          {
            //   V の中で縮約したときは、列 m (縮約では読むだけ) を同じ V の列 k_new (< m) へ写す
            parcel::vector<Teigen_vec> v_mp1, v_new_kk;
            vec_util.setptr(n, V_storage,     v_mp1,    stride*m);
            vec_util.setptr(n, ks_inplace ? V_storage : V_new_storage, v_new_kk, stride*k_new);
            parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(one_vec, zero_vec, v_mp1, v_new_kk);
          }
          // V_storage と V_new_storage の役割を入れ替える (k_new+1 列のコピーを省く)。
          //   halo は次の SpMVcomm で再充填される。k_new+1 列より後ろは、次の Arnoldi が
          //   書き込む前に必ず zero_init される (上の Arnoldi ループ参照) のでゼロ埋めも要らない。
          //   V の中で縮約したときは入れ替えない。
          if (!ks_inplace) {
          v_swapped = !v_swapped;
          vec_util.setptr(stride*(m+1),
                          (v_swapped ? V_new_parallel : V_parallel).vec_add(),     V_storage,     0);
          vec_util.setptr(stride*(m+1),
                          (v_swapped ? V_parallel : V_new_parallel).vec_add(),     V_new_storage, 0);
          }

          // H_new: top-left k_new×k_new = 並べ替え後の T の先頭ブロック + spike row
          //   T は準三角なので、複素共役対の 2x2 ブロックもそのまま入る。
          std::vector<double> H_new((m+1)*m, 0.0);
          for (int j = 0; j < k_new; j++)
            for (int i = 0; i < k_new; i++)
              H_new[i + j*(m+1)] = Ts[i + j*m];
          for (int j = 0; j < k_new; j++)
            H_new[k_new + j*(m+1)] = h_mp1_m * Z_new[(m-1)*k_new + j];
          H = H_new;
          k_current = k_new;
        }  // end KS restart loop

        timer_solver_main.end();
        timer_solver.end();

        // ========== 出力用に固有ベクトルを求め直す ==========
        //   リスタート中の Z は Schur ベクトル Q であって固有ベクトルではない。
        //   Ritz ベクトル V*z を作るには H の固有ベクトルが要るので、
        //   最終的な H に対して dgeev をかけ直す。
        //   不変部分空間で止めた場合は有効次元 md で分解する。rank 0 の結果を配る。
        const int mm = (int)md;
        {
          for (int i = 0; i < mm; i++) for (int j = 0; j < mm; j++)
            Hcopy[i + j*mm] = H[i + j*(m+1)];
          std::fill(Z.begin(), Z.end(), 0.0);
          std::fill(wr.begin(), wr.end(), 0.0);
          std::fill(wi.begin(), wi.end(), 0.0);
          int info_out = 0;
          if (myrank == 0)
            info_out = LAPACKE_dgeev(LAPACK_COL_MAJOR, 'N', 'V', mm,
                                     Hcopy.data(), mm, wr.data(), wi.data(),
                                     NULL, mm, Z.data(), mm);
          ks_bcast_int(&info_out, comm_of(Mat));
          ks_bcast(wr.data(), mm, comm_of(Mat));
          ks_bcast(wi.data(), mm, comm_of(Mat));
          ks_bcast(Z.data(),  mm*mm, comm_of(Mat));
          if (info_out != 0 && myrank == 0)
            std::cout << "[KS] 出力用 dgeev info=" << info_out << std::endl;
        }

        // ========== 出力: 上位 k_keep_ Ritz pair を EigenValue, EigenVec へ ==========
        // wr, wi は上記ループでソート (sel) 後の値を保持
        // Z_new の対応する基底変換は V_storage に既に反映済
        //   stable_sort: dgeev は複素共役対 (wr 同じ, wi 反転) を隣接位置 p, p+1 に
        //   格納する。|λ| が等しいため stable_sort で順序保存 → idx_final で隣接維持。
        //   これにより verify 側で Re/Im 列を確実にペア取得可能。
        std::vector<int> idx_final(mm);
        for (int i = 0; i < mm; i++) idx_final[i] = i;
        std::stable_sort(idx_final.begin(), idx_final.end(), [&](int a, int b) {
          return ks_mag_key(wr[a], wi[a]) > ks_mag_key(wr[b], wi[b]);
        });
        //   有効次元が k_keep より小さいときは、足りない分を 0 で返す
        const int k_out = std::min<int>((int)k_keep, mm);
        if (k_out < (int)k_keep && myrank == 0)
          std::cout << "[KS] only " << k_out << " Ritz pairs available (invariant subspace); the rest are returned as 0" << std::endl;

        // EigenValue ← top k_keep の (Re, Im)
        //   shift-invert モード: μ = wr + i·wi は (A-σI)⁻¹ の eigval
        //                       → A の eigval は λ_A = σ + 1/μ = σ + (wr - i·wi)/(wr²+wi²)
        //   ev_host は EigenValue の長さで取る (setValue は EigenValue の全長を読む)。
        //   逆変換の分母 |μ|^2 が 1e-300 以下 (μ ~ 0 は |λ - σ| が無限大に相当) のときは +inf を返す。
        {
          std::vector<double> ev_host(EigenValue.size(), 0.0);
          for (int j = 0; j < k_out; j++) {
            int i = idx_final[j];
            double re = wr[i], im = wi[i];
            if (shift_invert) {
              double denom = re*re + im*im;
              if (denom > 1e-300) {
                double inv_re =  re/denom;
                double inv_im = -im/denom;
                re = shift_sigma_ + inv_re;
                im = inv_im;
              } else {
                re = std::numeric_limits<double>::infinity();
                im = 0.0;
              }
            }
            ev_host[j*2]   = re;
            ev_host[j*2+1] = im;
          }
          EigenValue.setValue(ev_host.data());
        }

        // EigenVec ← top k_keep の Ritz vector = V_m · Z[:, idx_final[j]]
        //   Z は最終 dgeev の col-major m×m, 列 p の固有ベクトルは {Z[0+p*m], Z[1+p*m], ...}
        //   注: Schur top-left 内の収束済 Ritz は Z[:, p] ≈ e_p となるが、
        //        非収束 Ritz では一般に Z[:, p] != e_p なので必ず V·Z 展開が必要
        //   V_storage は stride 幅 (halo 込み), EigenVec は local-only (n 幅)
        //   書くのは EigenVec に入る本数 (n*nev 以上 n*k_keep 未満の長さも受け付ける)
        const int ev_cols = (n > 0) ? (int)std::min<size_t>((size_t)k_keep, EigenVec.size() / (size_t)n)
                                    : k_keep;
        for (int j = 0; j < ev_cols; j++) {
          parcel::vector<Teigen_vec> v_dst;
          vec_util.setptr(n, EigenVec, v_dst, (size_t)n*j);
          parcel_blas.template zero_init<Teigen_vec>(v_dst);
          if (j >= k_out) continue;
          int p = idx_final[j];
          for (int i = 0; i < mm; i++) {
            double zip = Z[i + p*mm];
            if (std::fabs(zip) < 1e-300) continue;
            parcel::vector<Teigen_vec> v_src;
            vec_util.setptr(n, V_storage, v_src, stride*i);
            //   ここも k_keep×m 回まわるので、スカラは使い回しの領域で渡す
            scal_vec.setValue(&zip);
            parcel_blas.template axpy<Tdot,Tdot,Teigen_vec,Teigen_vec,Taxpy>(scal_vec, one_vec, v_src, v_dst);
          }
        }
        que[0].wait();

        // 収束状態 (表示の converged = YES/NO もここから決めるので、表示より前に求める)
        //   NaN / Inf はその場で抜けるので DIVERGED_NANORINF。不変部分空間 (h_{j+1,j} ~ 0) で
        //   nev 個が真の残差でも tol を満たせば CONVERGED_RTOL (残差推定 0 だけでは
        //   収束としない。確かめられないまま止めたときは DIVERGED_BREAKDOWN)。
        PARCEL_INT reason;
        if (st_nonfinite)               reason = DIVERGED_NANORINF;
        else if (converged)             reason = CONVERGED_RTOL;
        else if (st_breakdown)          reason = DIVERGED_BREAKDOWN;
        else                            reason = DIVERGED_ITS;

        // ========== 出力 timer (線形ソルバーと同フォーマット) ==========
        if (myrank == 0) {
          double t_total  = timer_solver.get();
          double t_init   = timer_solver_init.get();
          double t_initP  = timer_solver_init_precon.get();
          double t_initC  = timer_solver_init_convert.get();
          double t_loop   = timer_solver_main.get();
          double t_dot    = timer_solver_dot.get();
          double t_axpy   = timer_solver_axpy.get();
          double t_gemv   = timer_solver_gemv.get();
          double t_spmv   = timer_solver_SpMV.get();
          double t_precon = timer_solver_precon.get();
          double t_init_other = t_init - t_initP - t_initC;
          double t_loop_other = t_loop - (t_dot + t_axpy + t_gemv + t_spmv + t_precon);
          bool   loop_other_clamped = false;
          if (t_loop_other < 0.0) { t_loop_other = 0.0; loop_other_clamped = true; }
          double t_postloop   = t_total - t_init - t_loop;
          std::cout<<std::endl<<" ----------------- "<<std::endl;
          std::cout<<"ks_arnoldi time       = "<<t_total      <<" [sec]  (total solver wall time)"<<std::endl;
          std::cout<<"  operator            = "<<(!shift_invert ? (get_precon_spectrum() ? "A K^-1" : "A")
                                                 : (shift_identity ? "(A - sigma I)^-1 (no ShiftMat, K = I)"
                                                    : (shift_pspec ? "(A K^-1 - sigma I)^-1" : "(A - sigma I)^-1")))<<std::endl;
          std::cout<<"  init time           = "<<t_init       <<" [sec]  (before main loop)"<<std::endl;
          std::cout<<"    init precon       = "<<t_initP      <<" [sec]"<<std::endl;
          std::cout<<"    init convert      = "<<t_initC      <<" [sec]"<<std::endl;
          std::cout<<"    init other        = "<<t_init_other <<" [sec]  (vec alloc/zero, init SpMV/dot/precon)"<<std::endl;
          std::cout<<"  main loop time      = "<<t_loop       <<" [sec]  (sum over all KS restarts)"<<std::endl;
          std::cout<<"    loop dot          = "<<t_dot        <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
          std::cout<<"    loop axpy         = "<<t_axpy       <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
          std::cout<<"    loop gemv         = "<<t_gemv       <<" [sec]  calls = "<<timer_solver_gemv.get_count()<<"  (ブロック直交化 v <- v - V h)"<<std::endl;
          std::cout<<"    loop SpMV         = "<<t_spmv       <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
          std::cout<<"    loop precon       = "<<t_precon     <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
          std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (dgeev, Z*V, restart copy)"<<std::endl;
          //   「その他」の内訳を切り分けるため、リスタートの基底縮約 (gemm) を別に出す。
          //   残りは密行列の LAPACK (dgeev / dgees / dtrsen) と H の更新。
          std::cout<<"      restart gemm    = "<<parcel_blas.get_time_gemm() <<" [sec]"<<std::endl;
          std::cout<<"      dense eig       = "<<timer_dense_eig.get()   <<" [sec]  calls = "<<timer_dense_eig.get_count()  <<"  (dgeev)"<<std::endl;
          std::cout<<"      dense schur     = "<<timer_dense_schur.get() <<" [sec]  calls = "<<timer_dense_schur.get_count()<<"  (dgees + dtrsen)"<<std::endl;
          std::cout<<"      restart repack  = "<<timer_restart_vec.get() <<" [sec]  calls = "<<timer_restart_vec.get_count()<<"  (ホスト側の詰め替え)"<<std::endl;
          if (loop_other_clamped) {
            std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
          }
          std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (final EigenVec/Value copy)"<<std::endl;
          //   回数は get_iterations() と同じ (実行したリスタートの回数)。converged = YES/NO は
          //   converged_reason から決める (NaN を含む判定で YES を出さない)。
          std::cout<<"  restarts            = "<<st_its
                   <<" (converged = "<<(reason == CONVERGED_RTOL ? "YES" : "NO")<<")"<<std::endl;
        }

        set_status(reason, st_its, st_res);
        print_status(myrank, " , ks_nconv = " + std::to_string((long long)ks_nconv_) + "/" +
                             std::to_string((long long)nev) +
                             (shift_invert ? " , linsolve_failures = " +
                                             std::to_string((long long)get_linsolve_failures())
                                           : std::string()));
      }
    };

  }  // namespace krylov
}  // namespace parcel

#endif
