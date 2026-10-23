// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_POWER_HPP_
#define PARCEL_KRYLOV_POWER_HPP_

#include <cmath>
#include <vector>

namespace parcel{
  namespace krylov{

// 計算精度を指定
template<
  class Tspmv   = parcel_double, // SpMV 計算精度
  class Tprecon = parcel_double, // 前処理計算精度
  class Taxpy   = parcel_double, // ベクトル演算計算精度
  class Tdot    = parcel_double  // 縮約演算計算精度
  >
class power : public eigen_param{

  ddlib::ddmath dd;
  parcel::timer timer_solver;
  parcel::timer timer_solver_init;
  parcel::timer timer_solver_init_precon;
  parcel::timer timer_solver_init_convert;
  parcel::timer timer_solver_scala;
  parcel::timer timer_solver_mainloop;

  parcel::timer timer_solver_SpMV;
  parcel::timer timer_solver_axpy;
  parcel::timer timer_solver_dot;
  parcel::timer timer_solver_precon;

  // 固有値の相対変化 |cur - prev| / |cur|。|cur| = 0 なら分母を |prev| にし、
  //   両方 0 なら 0 を返す (0/0 = NaN で収束判定が成立しなくなるのを避ける)。
  static double relative_change(double cur, double prev){
    double d = std::fabs(cur);
    if(d == 0.0) d = std::fabs(prev);
    if(d == 0.0) return 0.0;
    return std::fabs(cur - prev) / d;
  }

public :
  
  // 内部線形ソルバは持たないが、固有値ソルバ共通の設定 (set_precon_spectrum) を使う
  power() : eigen_param(1.0e-10, 5000, 30) {}
  ~power(){}
  
  // 前処理,IR等の設定をclassで定義する
  template<
    class Tmat,  // 係数行列データ精度
    class Teigen_val, // 固有値データ精度
    class Teigen_vec, // 固有ベクトルデータ精度
    class Tvec = parcel_double, // クリロフ部分空間法内部データ精度	
    class Tprecon_mat = parcel_double, // 前処理行列データ精度
    class Tprecon_vec = parcel_double  // 前処ベクトルデータ精度
    >
  void solver(
	      parcel::matrix<Tmat>  &Mat,
	      parcel::vector<Teigen_val> &EigenValue, // 固有値
	      parcel::vector<Teigen_vec> &EigenVec // 固有ベクトル
	      ){
    
    // 反復回数 : get_max_iteration()関数で取得
    
    // ------
    // べき乗法
    // x=x/||x||
    // do itr=1,itrmax
    //   y = AM^-1x
    //   Eigen = <x|A|x>/<x|x>
    //   x = y / ||y||
    // enddo
    // ------

    sycl::queue *que   = Mat.queue();

    require_in_order(que, "krylov::power");   // in-order キューが前提
    // べき乗法なので固有ベクトルは最大固有ベクトル
    const PARCEL_INT n = EigenVec.size();
    que[0].wait();
    reset_status();

#if PARCEL_MPI
    parcel::comm *comm = Mat.comm();
    int myrank   = comm->rank();
    int nprocess = comm->size();
    comm->barrier();
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,comm->mpi_comm());
    PARCEL_INT nadd = Mat.get_recv_size();
#else
    parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
    int myrank = 0;
    int nprocess = 1;
    PARCEL_INT nadd = 0;
#endif

    timer_solver.start();
    timer_solver_init.start();

    parcel::scalar_util host_util;
    parcel::vector_util vec_util;
    // ------------------------------
    // Init Preconditioning
    timer_solver_init_precon.start();

    double tol = get_tolerance();
    PARCEL_INT itrmax = get_max_iteration();
    // A K^-1 モード (set_precon_spectrum(true)) のときだけ K を使う。既定は前処理なし (A の固有値)
    precon_param &precon = *(get_outer_precon_param());
    parcel::precon_matrix<
      Tmat,
      Tvec,
      Tprecon_mat,
      Tprecon_vec,
      Tprecon
      > precon_Mat(Mat,precon);
    timer_solver_init_precon.end();

    // スカラ定数
    parcel::vector<Tdot> one;
    one.resize(1,que);
    double one_ = 1.0;one.setValue(&one_);

    parcel::vector<Tdot> negative_one;
    negative_one.resize(1,que);
    double negative_one_ = -1.0;negative_one.setValue(&negative_one_);

    parcel::vector<Tdot> zero;
    zero.resize(1,que);
    double zero_ = 0.0;zero.setValue(&zero_);
    
    // workベクトル確保    
#if PARCEL_MPI
    // 通信バッファ有
    parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);
    parcel::parallel_vector<Tvec> y_parallel(n,nadd,1,que);parcel::vector<Tvec> y;vec_util.setptr( n, y_parallel.vec(0), y, 0);    
    parcel::parallel_vector<Tvec> z_parallel(n,nadd,1,que);parcel::vector<Tvec> z;vec_util.setptr( n, z_parallel.vec(0), z, 0);
#else
    // 通信バッファ無
    parcel::vector<Tvec> x(n,que);parcel_blas.zero_init<Tvec>(x);
    parcel::vector<Tvec> y(n,que);parcel_blas.zero_init<Tvec>(y);
    parcel::vector<Tvec> z(n,que);parcel_blas.zero_init<Tvec>(z);    
#endif
    
    // 初期ベクトル
    //   全要素 1.0 とすると, (1,1,...,1) が特定の高周波固有ベクトルと厳密に直交する
    //   場合がある (例: 3 次元 7 点ポアソン・ディリクレ条件で NX が偶数のとき
    //   v_{NX,NX,NX} と直交) ため, その固有値に到達できない.
    //   sin/cos の組合せで, ほぼ任意の正規直交基底に対して非ゼロの射影を持つように
    //   初期化する. global index (先頭行 + i) を用いるので MPI 分割に依存しない.
    //   先頭行は MPI_Exscan で求める (不均等分割でも行番号がずれない).
    {
      std::vector<double> x_init(n);
#if PARCEL_MPI
      const long long base = global_row_start(comm->mpi_comm(), (long long)n);
#else
      const long long base = 0;
#endif
      for (PARCEL_INT i = 0; i < n; ++i) {
        x_init[i] = default_seed_value(base + i);
      }
      x.setValue(x_init.data());
    }
    y.setValue(0.0);
    z.setValue(0.0);
    
    parcel::vector<Tdot> yy_norm; yy_norm.resize(1,que);
      
    // 反復ループ
    PARCEL_MAX_FLOTING_TYPE eigen_value = 0.0;
    parcel::vector<Tdot> eigen_value_end;eigen_value_end.resize(1,que);
    timer_solver_init.end();

    // 収束状態 (krylov_param::set_status)。判定値は固有値の相対変化 |λ_i - λ_{i-1}| / |λ_i|
    //   (itr = 0 では前回値が無いので判定しない。反復が 1 回だけなら residual は NaN のまま)
    bool   st_converged = false;
    //   終了時の出力
    //     CONVERGED_RTOL     : 固有値 = 最後の λ_i、固有ベクトル = y = A K^-1 x (正規化前)
    //     DIVERGED_ITS       : 同上 (itrmax 回目の値)
    //     DIVERGED_NANORINF  : 非有限を見つけた時点で止め、固有値 = 最後の有限な推定値
    //                          (1 回目で止めたら 0)、固有ベクトル = 直前に正規化した x
    //     DIVERGED_BREAKDOWN : y = 0 で止め、固有値・固有ベクトルは NANORINF と同じ扱い
    bool   st_nonfinite = false;   // 非有限で止めた
    bool   st_breakdown = false;   // y = 0 で止めた
    PARCEL_INT st_its   = 0;
    double st_res       = std::numeric_limits<double>::quiet_NaN();

    que[0].wait();
    timer_solver_mainloop.start();
    for(PARCEL_INT itr = 0;itr < itrmax;itr++){
      // y = AM^-1x 
      // ------------------------------
      // preconditioning solve Mz = x; 
      timer_solver_precon.start();      
#if PARCEL_MPI      
      parcel::vector<Tvec> x_vecadd ;vec_util.setptr( (n+nadd), x_parallel.vec_add(), x_vecadd,0); 
      parcel::vector<Tvec> z_vecadd ;vec_util.setptr( (n+nadd), z_parallel.vec_add(), z_vecadd,0); 
      precon_Mat.solver(x_vecadd,z_vecadd);
#else
      precon_Mat.solver(x,z);
#endif
      timer_solver_precon.end();

      // ------------------------------
      // y = Az
      timer_solver_SpMV.start();
#if PARCEL_MPI      
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,z_vecadd,y);
#else
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,z,y);
#endif
      
      timer_solver_SpMV.end();
      // ------------------------------
      timer_solver_dot.start();
      //   Eigen = <x|A|x>/<x|x>
      
      parcel_blas.dot< Tvec,Tvec,Tdot >(x,x);
      parcel::scalar xx_host = parcel_blas.get_dot();

      parcel_blas.dot< Tvec,Tvec,Tdot >(y,y);
      parcel::scalar yy_host = parcel_blas.get_dot();
      
      parcel_blas.dot< Tvec,Tvec,Tdot >(x,y);
      parcel::scalar xAx_host = parcel_blas.get_dot();
      PARCEL_MAX_FLOTING_TYPE eigen_value_ = xAx_host.get_T() / xx_host.get_T();

      parcel::scalar yy_norm_host = host_util.sqrt(yy_host);
      timer_solver_dot.end();

      // 監視する量 (<x,x>, <y,y>, <x,y>, レイリー商) のどれかが NaN / Inf なら
      //   その場で止めて DIVERGED_NANORINF とする (下の「y = 0」の判定より先に行うので、
      //   <y,y> = NaN を DIVERGED_BREAKDOWN と取り違えない)。
      //   値はどれも allreduce 済みの内積から作るので、全ランクが同じ分岐を取る。
      //   返す値: 固有値は最後の有限な推定値 (itr = 0 で止めたら初期値 0)、固有ベクトルは
      //   直前に正規化した x (この反復の入力。<x,x> 自体が非有限なら x も非有限になりうる)。
      //   rank 0 が非有限になった値を 1 行書く (回帰試験は nan / inf の語で NONFINITE と判定する)。
      {
	const double xx_d  = (double)xx_host.get_T();
	const double yy_d  = (double)yy_host.get_T();
	const double xAx_d = (double)xAx_host.get_T();
	const double ev_d  = (double)eigen_value_;
	if(!std::isfinite(xx_d) || !std::isfinite(yy_d) || !std::isfinite(xAx_d) || !std::isfinite(ev_d)){
	  if(myrank == 0){
	    std::cout<<" power : non-finite value at itr = "<<itr
		     <<" (<x,x> = "<<xx_d<<" , <y,y> = "<<yy_d<<" , <x,y> = "<<xAx_d
		     <<" , Rayleigh quotient = "<<ev_d<<") ; stopped (DIVERGED_NANORINF)"<<std::endl;
	  }
	  st_its = itr+1;
	  st_nonfinite = true;
	  st_res = std::numeric_limits<double>::quiet_NaN();
	  // 出力は y から作るので x を写す。axpy は y = 1*x + 0*y で、y に NaN / Inf が残っていると
	  //   0*Inf = NaN が入るため、先に y を 0 にする
	  parcel_blas.zero_init<Tvec>(y);
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,zero,x,y);
	  break;
	}
      }

      // y = A K^-1 x が厳密に 0 (x が零空間にある、または低精度で y の全要素が
      //   アンダーフロー) なら破綻として止める (そのまま進むと 1/||y|| = inf で x が NaN になる)。
      //   yy は allreduce 済みなので全ランクが同じ分岐を取る。固有ベクトルには最後の正規化済み x を返す。
      if(!(yy_host.get_T() > 0.0)){
	if(myrank == 0){
	  std::cout<<" power : A K^-1 x is exactly zero at itr = "<<itr<<" (breakdown)"<<std::endl;
	}
	st_its = itr+1;
	st_breakdown = true;
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,zero,x,y);   // 出力は y から作るので x を写す
	break;
      }

      // 固有値の相対変化 (itr = 0 では前回値が無いので判定しない)
      //   分母を |λ_i| とし、それが 0 なら |λ_{i-1}| (前回値が非 0 なら相対変化 1 で未収束)、
      //   両方 0 なら推定値が変わっていないので相対変化 0 とする (relative_change())。
      //   両方とも有限なので NaN にはならないが、差が桁あふれすれば Inf になりうるので、
      //   そのときは上と同じく止める。
      double rel_change = std::numeric_limits<double>::quiet_NaN();
      if( 0 < itr ){
	rel_change = relative_change((double)eigen_value_, (double)eigen_value);
	if(!std::isfinite(rel_change)){
	  if(myrank == 0){
	    std::cout<<" power : non-finite value at itr = "<<itr
		     <<" (relative change = "<<rel_change<<") ; stopped (DIVERGED_NANORINF)"<<std::endl;
	  }
	  st_its = itr+1;
	  st_nonfinite = true;
	  st_res = std::numeric_limits<double>::quiet_NaN();
	  // 出力は y から作るので x を写す。axpy は y = 1*x + 0*y で、y に NaN / Inf が残っていると
	  //   0*Inf = NaN が入るため、先に y を 0 にする
	  parcel_blas.zero_init<Tvec>(y);
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,zero,x,y);
	  break;
	}
      }

      //   x = y / ||y||
      yy_norm.setValue( 1.0 / yy_norm_host.get_T() );
      timer_solver_axpy.start();
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (yy_norm,zero,y,x);
      timer_solver_axpy.end();

      // 反復履歴
      if( myrank == 0 && (itr < 10 || itr % 50 == 0) ){
	std::cout<<std::setw(8)<<itr<<" : Eigen Value = ";
	std::cout<<std::scientific << std::setprecision(14) << eigen_value_ <<std::endl;
      }

      st_its = itr+1;
      // 非有限は上で止めているので、ここに来る値は有限
      if( 0 < itr ){
	st_res = rel_change;
	if( rel_change < tol){
	  st_converged = true;
	  eigen_value = eigen_value_;  
	  break;
	}
      }
      
      eigen_value = eigen_value_;

    }
    timer_solver_mainloop.end();
    que[0].wait();

    // 固有値の出力
    EigenValue.setValue(eigen_value);

    // 固有ベクトルの出力
    vec_util.convert<Tvec,Teigen_vec>(y,EigenVec);

    que[0].wait();
    timer_solver.end();
    
    if( myrank == 0){
      // 反復トレースで scientific/precision(14) を設定済 → timer はデフォルト fixed に戻す
      std::cout.unsetf(std::ios::scientific);
      std::cout.precision(6);
      double t_total  = timer_solver.get();
      double t_init   = timer_solver_init.get();
      double t_initP  = timer_solver_init_precon.get();
      double t_initC  = timer_solver_init_convert.get();
      double t_loop   = timer_solver_mainloop.get();
      double t_dot    = timer_solver_dot.get();
      double t_axpy   = timer_solver_axpy.get();
      double t_spmv   = timer_solver_SpMV.get();
      double t_precon = timer_solver_precon.get();
      double t_init_other = t_init - t_initP - t_initC;
      double t_loop_other = t_loop - (t_dot + t_axpy + t_spmv + t_precon);
      bool   loop_other_clamped = false;
      if (t_loop_other < 0.0) { t_loop_other = 0.0; loop_other_clamped = true; }
      double t_postloop   = t_total - t_init - t_loop;
      std::cout<<std::endl<<" ----------------- "<<std::endl;
      std::cout<<"power time            = "<<t_total      <<" [sec]  (total solver wall time)"<<std::endl;
      std::cout<<"  init time           = "<<t_init       <<" [sec]  (before main loop)"<<std::endl;
      std::cout<<"    init precon       = "<<t_initP      <<" [sec]"<<std::endl;
      std::cout<<"    init convert      = "<<t_initC      <<" [sec]"<<std::endl;
      std::cout<<"    init other        = "<<t_init_other <<" [sec]  (vec alloc/zero, init SpMV/dot/precon)"<<std::endl;
      std::cout<<"  main loop time      = "<<t_loop       <<" [sec]  (sum over all power iterations)"<<std::endl;
      std::cout<<"    loop dot          = "<<t_dot        <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
      std::cout<<"    loop axpy         = "<<t_axpy       <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
      std::cout<<"    loop SpMV         = "<<t_spmv       <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
      std::cout<<"    loop precon       = "<<t_precon     <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
      std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (kernel launch/sync, cout per iter)"<<std::endl;
      if (loop_other_clamped) {
        std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
      }
      std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (final EigenVec/Value copy)"<<std::endl;
    }

    // 収束状態。真の残差は計算しない (true_residual は NaN)
    set_status(st_breakdown ? DIVERGED_BREAKDOWN : decide_reason(st_converged, st_nonfinite), st_its, st_res);
    print_status(myrank);
    return ;
    
  }
};

  }
}


#endif
