// =========================================================================
// parcel_krylov_inverse_power.hpp
//
// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
//
// 逆べき乗法 (inverse power method) — 最小絶対値固有値を計算
//
// アルゴリズム:
//   x_0: 初期 (sin/cos seed) を正規化
//   for itr:
//     solve A y = x          (内部 pgmres。set_linsolve_algo(LINSOLVE_PCG) で pcg)
//     θ = <y, x> / <y, y>
//     x_new = y / ||y||
//     if |θ - θ_prev|/|θ| < tol: break
//
//   y = A^-1 x より x = A y なので θ = <y, A y> / <y, y> であり、
//   これは現在の反復ベクトル y における A のレイリー商である。
//   逆反復で y が絶対値最小の固有値の固有ベクトルへ収束するため θ -> λ_min(A).
//
// 前処理について (既定の set_precon_spectrum(false)):
//   内部ソルバの前処理は set_linsolve_precon_algo() 等で指定する (既定はヤコビ前処理)。
//   内部ソルバが解くのはあくまで A y = x なので、前処理は内部反復の収束を速めるだけで、
//   返る固有値は前処理の有無によらず λ_min(A) である。set_precon_algo() はこのモードでは使わない。
//
// 適用範囲:
//   - 内部ソルバの既定は pgmres で、A の対称性は要らない。
//     set_linsolve_algo(LINSOLVE_PCG) で pcg にする場合は A が対称 (正定値または負定値) であること。
//   - 絶対値最小の固有値が複素共役対になる非対称行列では、レイリー商が収束しない。
//     その場合は ks_arnoldi の shift-invert を σ = 0 で使う:
//         solver.set_shift(0.0);
//         solver.solver<...>(Mat, EigenValue, EigenVec, &Mat);
//     μ = λ(A^-1), λ = 0 + 1/μ となり逆反復の一般化にあたる。
//     内部解法は pgmres で、複素共役対にも対応し、収束判定は Ritz 残差で行う。
//
// 前処理後の行列 A K^-1 の固有値 (set_precon_spectrum(true)、固有値ソルバ共通):
//   power / ks_arnoldi と同じく、前処理後の行列 A K^-1 の固有値を求めるモード。
//   power (絶対値最大) と対にして前処理後の条件数を評価できる。
//     solve (A K^-1) z = x   (内部 pgmres、右前処理。最後の K^-1 を掛けずに z を返す)
//     θ = <z, x> / <z, z>    (x = A K^-1 z なので z における A K^-1 のレイリー商)
//   K を陽に作らずに済むのは、右前処理 GMRES が内部でまさに A K^-1 を
//   係数行列として扱っているため。K は power と同じ前処理設定から作る。
//   このモードでは内部解法が pgmres なので A の対称性は要らない。
//   ただし A K^-1 が非対称で絶対値最小の固有値が複素共役対になると
//   レイリー商は収束しない。その場合は ks_arnoldi の同モードを使う。
//   返す固有ベクトルは A K^-1 の右固有ベクトル z の向き。
//
// 用途:
//   - PCG/GMRES の条件数評価: κ(A) = λ_max(A) / λ_min(A)
//     λ_max は power, λ_min は inverse_power で測定
//   - 前処理後: κ(A K^-1) = λ_max(A K^-1) / λ_min(A K^-1)
//     λ_max は前処理付き power, λ_min は set_precon_spectrum(true) の inverse_power
// =========================================================================
#ifndef PARCEL_KRYLOV_INVERSE_POWER_HPP_
#define PARCEL_KRYLOV_INVERSE_POWER_HPP_

#include <cmath>
#include <memory>
#include <vector>

namespace parcel{
  namespace krylov{

    template<
      class Tspmv   = parcel_double,
      class Tprecon = parcel_double,
      class Taxpy   = parcel_double,
      class Tdot    = parcel_double
    >
    class inverse_power : public eigen_param {

      ddlib::ddmath dd;
      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_mainloop;
      parcel::timer timer_solver_linsolve;   // 内部線形ソルバ (pgmres / pcg)
      parcel::timer timer_solver_axpy;
      parcel::timer timer_solver_dot;

      // 固有値の相対変化 |cur - prev| / |cur|。|cur| = 0 なら分母を |prev| にし、
      //   両方 0 なら 0 を返す (0/0 = NaN で収束判定が成立しなくなるのを避ける)。
      static double relative_change(double cur, double prev){
        double d = std::fabs(cur);
        if(d == 0.0) d = std::fabs(prev);
        if(d == 0.0) return 0.0;
        return std::fabs(cur - prev) / d;
      }

    public:
      // 内部線形ソルバの既定: 収束判定値 1e-10 (外側の固有値反復より厳しく)、
      //   最大反復 5000、pgmres のリスタート周期 30、ヤコビ前処理。
      //   設定は eigen_param の set_linsolve_* (固有値ソルバ共通)。
      //   set_precon_spectrum(true): A K^-1 の絶対値最小固有値 (内部 pgmres が z を返す)
      //   set_precon_spectrum(false、既定): A の絶対値最小固有値
      //   内部解法の既定は pgmres。
      //   set_linsolve_algo(LINSOLVE_PCG) で pcg にできる (A が対称定値のとき。A K^-1 モードでは不可)。
      inverse_power() : eigen_param(1.0e-10, 5000, 30) {}
      ~inverse_power(){}

      template<
        class Tmat,
        class Teigen_val,
        class Teigen_vec,
        class Tvec = parcel_double,
        class Tprecon_mat = parcel_double,
        class Tprecon_vec = parcel_double
      >
      void solver(
          parcel::matrix<Tmat>      &Mat,
          parcel::vector<Teigen_val> &EigenValue,   // size = 1: λ_min(A)
          parcel::vector<Teigen_vec> &EigenVec      // size = n: 対応固有ベクトル
      ) {
        sycl::queue *que = Mat.queue();
        require_in_order(que, "krylov::inverse_power");   // in-order キューが前提
        const PARCEL_INT n = EigenVec.size();
        que[0].wait();
        reset_status();
        reset_linsolve_failures();

#if PARCEL_MPI
        parcel::comm *comm = Mat.comm();
        int myrank   = comm->rank();
        comm->barrier();
        parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que, comm->mpi_comm());
        PARCEL_INT nadd = Mat.get_recv_size();
#else
        parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
        int myrank = 0;
        PARCEL_INT nadd = 0;
#endif

        timer_solver.start();
        timer_solver_init.start();

        parcel::scalar_util host_util;
        parcel::vector_util vec_util;

        const double tol     = get_tolerance();
        const PARCEL_INT itrmax = get_max_iteration();

        // スカラ定数
        parcel::vector<Tdot> one(1, que);          { double v=1.0; one.setValue(&v); }
        parcel::vector<Tdot> negative_one(1, que); { double v=-1.0; negative_one.setValue(&v); }
        parcel::vector<Tdot> zero(1, que);         { double v=0.0; zero.setValue(&v); }

        // 内部線形ソルバ
        //   A K^-1 モード : 右前処理 pgmres が (A K^-1) y = x を解いて内部変数 y (= z) を返す。
        //                   前処理は K そのもの (利用者の set_precon_algo 等で A から作る)。
        //   A のモード    : 既定は pgmres、set_linsolve_algo(LINSOLVE_PCG) で pcg。
        //                   前処理は set_linsolve_precon_algo 等 (既定はヤコビ前処理)。
        //   前処理行列は一度だけ作り、全反復で使い回す (pgmres に外部 precon として渡す)。
        const bool spec    = get_precon_spectrum();
        const bool use_pcg = (!spec) && (get_linsolve_algo() == LINSOLVE_PCG);
        parcel::krylov::pcg<Tspmv, Tprecon, Taxpy, Tdot> internal_pcg;
        std::unique_ptr<parcel::krylov::pgmres<Tspmv, Tprecon, Taxpy, Tdot>> internal_gmres;
        std::unique_ptr<parcel::precon_matrix<Tmat, Tvec, Tprecon_mat, Tprecon_vec, Tprecon>> precon_Mat;
        if(use_pcg){
          internal_pcg.set_tolerance(get_linsolve_tolerance());
          internal_pcg.set_max_iteration(get_linsolve_max_iteration());
          copy_precon_param(*get_linsolve_precon_param(), *internal_pcg.get_precon_param());
        } else {
          internal_gmres.reset(new parcel::krylov::pgmres<Tspmv, Tprecon, Taxpy, Tdot>());
          internal_gmres->set_tolerance(get_linsolve_tolerance());
          internal_gmres->set_max_iteration(get_linsolve_max_iteration());
          internal_gmres->set_restart(get_linsolve_restart());
          internal_gmres->set_qr_algo(parcel::qr::QR_MGS);
          copy_precon_param(spec ? *get_precon_param() : *get_linsolve_precon_param(),
                            *internal_gmres->get_precon_param());
          if(spec){
            internal_gmres->set_precon_space_output(true);
            internal_gmres->set_shift(0.0);
          }
          timer_solver_init_precon.start();
          precon_Mat.reset(new parcel::precon_matrix<
            Tmat, Tvec, Tprecon_mat, Tprecon_vec, Tprecon>(Mat, *(internal_gmres->get_precon_param())));
          timer_solver_init_precon.end();
        }

        // work vectors (halo 込み)
#if PARCEL_MPI
        parcel::parallel_vector<Tvec> x_parallel(n, nadd, 1, que);
        parcel::parallel_vector<Tvec> y_parallel(n, nadd, 1, que);
        parcel::vector<Tvec> x; vec_util.setptr(n, x_parallel.vec(0), x, 0);
        parcel::vector<Tvec> y; vec_util.setptr(n, y_parallel.vec(0), y, 0);
#else
        parcel::vector<Tvec> x(n, que); parcel_blas.template zero_init<Tvec>(x);
        parcel::vector<Tvec> y(n, que); parcel_blas.template zero_init<Tvec>(y);
#endif

        // 初期 x: sin/cos seed (全 mode と非ゼロ内積)
        //   グローバル行番号の先頭を MPI_Exscan で求める (不均等分割でも行番号がずれない)
        {
          std::vector<double> x_init(n);
#if PARCEL_MPI
          const long long base = global_row_start(comm->mpi_comm(), (long long)n);
#else
          const long long base = 0;
#endif
          for(PARCEL_INT i = 0; i < n; i++){
            x_init[i] = default_seed_value(base + i);
          }
          x.setValue(x_init.data());
        }
        // normalize x
        parcel_blas.template dot<Tvec, Tvec, Tdot>(x, x);
        double xnorm = std::sqrt(parcel_blas.get_dot().get_T());
        // ノルムが 0 か非有限 (全体が 0 行、または低精度で桁あふれ) なら 1/0 で
        //   NaN になる。正規化を省いて進める (値は全ランク共通の allreduce 済み)。
        if(!(xnorm > 0.0) || !std::isfinite(xnorm)){
          if(myrank == 0){
            std::cout<<" inverse_power : initial vector norm is zero or not finite; not normalized"<<std::endl;
          }
          xnorm = 1.0;
        }
        double inv_xnorm = 1.0 / xnorm;
        parcel::vector<Tdot> inv_xnorm_vec(1, que); inv_xnorm_vec.setValue(&inv_xnorm);
        parcel_blas.template axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(inv_xnorm_vec, zero, x, y);
        parcel_blas.template axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(one,           zero, y, x);

        timer_solver_init.end();
        que[0].wait();
        timer_solver_mainloop.start();

        PARCEL_MAX_FLOTING_TYPE eigen_value = 0.0;   // 前回値。relative_change() に直接渡す
        bool converged = false;
        int last_itr = 0;
        // 収束状態 (krylov_param::set_status)。power と同じく判定値は固有値の相対変化。
        //   反復回数は実行した外側反復の回数 (下の "iterations" 表示も同じ値)。
        //   終了時の出力
        //     CONVERGED_RTOL / DIVERGED_ITS : 固有値 = 最後の θ、固有ベクトル = 正規化した最後の y
        //     DIVERGED_NANORINF  : 非有限 (内部ソルバの DIVERGED_NANORINF を含む) を見つけた時点で
        //                          止め、固有値 = 最後の有限な推定値 (1 回目で止めたら 0)、
        //                          固有ベクトル = 直前に正規化した x (この反復の右辺。有限)
        //     DIVERGED_BREAKDOWN : y = 0 で止め、固有値・固有ベクトルは NANORINF と同じ扱い
        bool   st_nonfinite = false;   // 非有限で止めた
        bool   st_breakdown = false;   // y = 0 で止めた
        PARCEL_INT st_its   = 0;
        double st_res       = std::numeric_limits<double>::quiet_NaN();
        for(PARCEL_INT itr = 0; itr < itrmax; itr++){
          last_itr = itr;
          // 線形解 y = A^{-1} x (A K^-1 モードでは y = (A K^-1)^{-1} x)
          timer_solver_linsolve.start();
          parcel_blas.template zero_init<Tvec>(y);
          PARCEL_INT inner_reason = ITERATING;   // 内部ソルバの終了理由
          if(!use_pcg){
            // A K^-1 モードでは (A K^-1) y = x を解き y (= z) を受け取る。
            // A のモードでは A y = x を解く。
            internal_gmres->template solver<
              Tmat, Tvec, Tvec, Tvec,
              Tprecon_mat, Tprecon_vec
            >(Mat, y, x, precon_Mat.get());
            count_linsolve(*internal_gmres);
            inner_reason = internal_gmres->get_converged_reason();
          } else {
            internal_pcg.template solver<
              Tmat, Tvec, Tvec, Tvec,
              Tprecon_mat, Tprecon_vec
            >(Mat, y, x);
            count_linsolve(internal_pcg);
            inner_reason = internal_pcg.get_converged_reason();
          }
          timer_solver_linsolve.end();

          // 内部ソルバが非有限で止まった (DIVERGED_NANORINF) なら外側も止める。
          //   同じ演算精度・同じ係数行列で次の反復も同じく桁あふれするので、回しても
          //   失敗する内部解法を繰り返すだけになる。失敗は count_linsolve() で linsolve_failures に
          //   数え済み。内部ソルバの reason は全ランク共通 (allreduce 済みの残差から決まる)。
          //   x は正規化済みの右辺のまま (有限) で、固有値は最後の有限な推定値を返す。
          if(inner_reason == DIVERGED_NANORINF){
            if(myrank == 0){
              std::cout << " inverse_power : inner linear solver stopped with DIVERGED_NANORINF at itr = " << itr
                        << " ; stopped (DIVERGED_NANORINF)" << std::endl;
            }
            st_its = itr+1;
            st_nonfinite = true;
            st_res = std::numeric_limits<double>::quiet_NaN();
            break;
          }

          // θ = <y, x> / <y, y> = <y, A y> / <y, y> (y におけるレイリー商) → λ_min(A)
          //   前処理は内部ソルバの収束を速めるだけで、返る固有値は λ_min(A) のまま
          //   A K^-1 モードでは x = A K^-1 y なので θ → λ_min(A K^-1)
          timer_solver_dot.start();
          parcel_blas.template dot<Tvec,Tvec,Tdot>(y, x);
          double yx = parcel_blas.get_dot().get_T();
          parcel_blas.template dot<Tvec,Tvec,Tdot>(y, y);
          double yy = parcel_blas.get_dot().get_T();
          timer_solver_dot.end();
          // <y,x>, <y,y> が非有限なら止めて DIVERGED_NANORINF とする (下の !(yy > 0) より先に
          //   判定するので、<y,y> = NaN を DIVERGED_BREAKDOWN と取り違えない)。
          //   yx, yy は allreduce 済みなので全ランクが同じ分岐を取る。
          //   rank 0 が値を 1 行書く (回帰試験は nan / inf の語で NONFINITE と判定する)。
          //   y は捨て、x (直前に正規化した有限な反復ベクトル) と最後の有限な θ を返す。
          if(!std::isfinite(yx) || !std::isfinite(yy)){
            if(myrank == 0){
              std::cout << " inverse_power : non-finite value at itr = " << itr
                        << " (<y,x> = " << yx << " , <y,y> = " << yy << ") ; stopped (DIVERGED_NANORINF)" << std::endl;
            }
            st_its = itr+1;
            st_nonfinite = true;
            st_res = std::numeric_limits<double>::quiet_NaN();
            break;
          }
          // 内部ソルバの解 y が厳密に 0 (右辺 x がアンダーフローした、内部ソルバが
          //   0 を返した等) なら破綻として止める (そのまま進むと θ = 0/0 = NaN になる)。
          //   yy は allreduce 済みなので全ランクが同じ分岐を取る。x は最後の正規化済みのまま返す。
          if(!(yy > 0.0)){
            if(myrank == 0){
              std::cout << " inverse_power : linear solve returned y = 0 at itr = " << itr << " (breakdown)" << std::endl;
            }
            st_its = itr+1;
            st_breakdown = true;
            break;
          }
          // θ = 1 / (||y||² / <y,x>) of inverse iteration
          //   |<y,x>| <= ||y|| ||x|| = ||y|| なので |θ| <= 1/||y|| で、yy > 0 が有限なら
          //   θ も有限 (念のため確かめる)。相対変化は relative_change() (|θ_i| = 0 でも NaN にしない)。
          //   どちらかが非有限なら、x と固有値を更新する前に止める。
          const double theta = yx / yy;
          double rel_change = std::numeric_limits<double>::quiet_NaN();
          if(itr > 0) rel_change = relative_change(theta, (double)eigen_value);
          if(!std::isfinite(theta) || (itr > 0 && !std::isfinite(rel_change))){
            if(myrank == 0){
              std::cout << " inverse_power : non-finite value at itr = " << itr
                        << " (Rayleigh quotient = " << theta << " , relative change = " << rel_change
                        << ") ; stopped (DIVERGED_NANORINF)" << std::endl;
            }
            st_its = itr+1;
            st_nonfinite = true;
            st_res = std::numeric_limits<double>::quiet_NaN();
            break;
          }
          eigen_value = theta;

          // x_new = y / ||y||
          double yn = std::sqrt(yy);
          double inv_yn = 1.0 / yn;
          parcel::vector<Tdot> inv_yn_vec(1, que); inv_yn_vec.setValue(&inv_yn);
          timer_solver_axpy.start();
          parcel_blas.template axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(inv_yn_vec, zero, y, x);
          timer_solver_axpy.end();

          if(myrank == 0 && (itr < 10 || itr % 50 == 0)){
            std::cout << std::setw(8) << itr << " : Eigen Value = "
                      << std::scientific << std::setprecision(14) << eigen_value << std::endl;
          }

          st_its = itr+1;
          // 非有限は上で止めているので、ここに来る値は有限
          if(itr > 0){
            st_res = rel_change;
            if(rel_change < tol){
              converged = true;
              break;
            }
          }
        }
        timer_solver_mainloop.end();
        que[0].wait();

        // 出力
        EigenValue.setValue(eigen_value);
        vec_util.template convert<Tvec, Teigen_vec>(x, EigenVec);
        que[0].wait();
        timer_solver.end();

        if(myrank == 0){
          std::cout.unsetf(std::ios::scientific);
          std::cout.precision(6);
          double t_total = timer_solver.get();
          double t_init  = timer_solver_init.get();
          double t_loop  = timer_solver_mainloop.get();
          double t_lin   = timer_solver_linsolve.get();
          double t_dot   = timer_solver_dot.get();
          double t_axpy  = timer_solver_axpy.get();
          double t_loop_other = t_loop - (t_lin + t_dot + t_axpy);
          bool clamped = false;
          if(t_loop_other < 0.0){ t_loop_other = 0.0; clamped = true; }
          double t_post  = t_total - t_init - t_loop;
          std::cout<<std::endl<<" ----------------- "<<std::endl;
          std::cout<<"inverse_power time    = "<<t_total <<" [sec]  (total solver wall time)"<<std::endl;
          std::cout<<"  spectrum            = "<<(spec ? "A K^-1 (precon_spectrum)" : "A")<<std::endl;
          std::cout<<"  init time           = "<<t_init  <<" [sec]"<<std::endl;
          std::cout<<"  main loop time      = "<<t_loop  <<" [sec]  (sum over all inverse iterations)"<<std::endl;
          std::cout<<"    loop linsolve     = "<<t_lin   <<" [sec]  calls = "<<timer_solver_linsolve.get_count()<<(spec ? "  (内部 pgmres, z 出力)" : (use_pcg ? "  (内部 pcg)" : "  (内部 pgmres)"))<<std::endl;
          std::cout<<"    loop dot          = "<<t_dot   <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
          std::cout<<"    loop axpy         = "<<t_axpy  <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
          std::cout<<"    loop other        = "<<t_loop_other<<" [sec]"<<std::endl;
          if(clamped) std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
          std::cout<<"  post-loop time      = "<<t_post  <<" [sec]"<<std::endl;
          std::cout<<"  iterations          = "<<st_its
                   <<" (converged = "<<(converged?"YES":"NO")<<")"<<std::endl;
        }

        // 収束状態。内部ソルバの未収束は外側の reason に反映せず、回数だけ持つ
        set_status(st_breakdown ? DIVERGED_BREAKDOWN : decide_reason(converged, st_nonfinite), st_its, st_res);
        print_status(myrank, " , linsolve_failures = " + std::to_string((long long)get_linsolve_failures()));
      }
    };

  } // namespace krylov
} // namespace parcel

#endif
