// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_CG_HPP_
#define PARCEL_KRYLOV_CG_HPP_

// 真の残差 r_true = b - Ax は q の領域に置く (省メモリ)。
//   r_true を計算するのは (a) 各反復の先頭 (TRUE_RESIDUAL) と (b) ループ後で、どちらも q が使われていない
//   区間 (前の反復の r -= alpha q の後、この反復の q = A p の前、およびループ終了後)。
//   q = A p は alpha*Ap + 0*q と計算するので、q に残った r_true の値は 0 倍されて消える。
//   p_history (pstep 本) は x の一括更新に使う本体のため減らせない。

#include <cstdlib>

// 解 x を何反復ぶんまとめて更新するか。
//   CG は p の履歴を pstep 本貯めてから gemv 1 回で x に反映する。
//   x の読み書きを 1/pstep に減らすため。
//   副作用として、真の残差 b-Ax は pstep 反復ごとにしか意味を持たない
//   (間の反復では x が前の値のままなので、同じ値が並ぶ)。
//   pstep=1 では x を毎反復更新し、p の履歴が 1 本になる (pstep=4 よりベクトル 3 本分メモリが少ない)。
//   pstep を大きくすると x の読み書きが減って少し速くなる代わりに、p の履歴の分だけメモリを使う。
//   x の足し合わせ方が変わるので丸めは変わる (反復数は同じ)。
//   既定は PARCEL_PCG_NO_XHIST が 1 (既定) なら 1、0 なら 4。環境変数 PARCEL_PSTEP (正の整数) が優先する。
#ifndef PARCEL_PCG_NO_XHIST
#define PARCEL_PCG_NO_XHIST 1
#endif
inline PARCEL_INT parcel_cg_pstep_default(){
  if(const char *e = std::getenv("PARCEL_PSTEP")){
    const long v = std::atol(e);
    if(v > 0){ return (PARCEL_INT)v; }
  }
  return PARCEL_PCG_NO_XHIST ? 1 : 4;
}


namespace parcel{
  namespace krylov{

// 計算精度を指定
template<
  class Tspmv   = parcel_double, // SpMV 計算精度
  class Tprecon = parcel_double, // 前処理計算精度
  class Taxpy   = parcel_double, // ベクトル演算計算精度
  class Tdot    = parcel_double  // 縮約演算計算精度
  >
class pcg : public krylov_param{
  ddlib::ddmath dd;
  parcel::timer timer_solver;
  parcel::timer timer_solver_init;
  parcel::timer timer_solver_init_precon;
  parcel::timer timer_solver_init_convert;
  parcel::timer timer_solver_scala;

  parcel::timer timer_solver_SpMV;
  parcel::timer timer_solver_axpy;
  // pstep 本まとめた x の更新は gemv なので、axpy とは別のタイマで測る
  //   (axpy に含めると 1 回あたりのバイト数の換算が崩れ、帯域が過小に出る)。
  parcel::timer timer_solver_gemv;
  parcel::timer timer_solver_dot;
  parcel::timer timer_solver_precon;
  parcel::timer timer_solver_mainloop;
  PARCEL_INT pstep = parcel_cg_pstep_default();
public :
  
  pcg(){}
  ~pcg(){}
  
  // 前処理,IR等の設定をclassで定義する
  template<
    class Tmat,  // 係数行列データ精度
    class Tvecx, // 入力ベクトルデータ精度
    class Tvecb, // 右辺ベクトルデータ精度
    class Tvec = parcel_double, // クリロフ部分空間法内部データ精度	
    class Tprecon_mat = parcel_double, // 前処理行列データ精度
    class Tprecon_vec = parcel_double  // 前処ベクトルデータ精度
    >
  void solver(
	      parcel::matrix<Tmat>  &Mat,
	      parcel::vector<Tvecx> &x0,
	      parcel::vector<Tvecb> &b0
	      ){
    sycl::queue *que   = Mat.queue();
    require_in_order(que, "krylov::pcg");   // in-order キューが前提
    const PARCEL_INT n = x0.size();
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
    
    std::cout<<" nprocess = "<<nprocess<<", rank = "<<myrank<<", n = "<< n <<", nadd = "<<nadd<<std::endl;
    
    timer_solver.start();
    timer_solver_init.start();

    parcel::scalar_util host_util;
    parcel::vector_util vec_util;
    // ------------------------------
    // Init Preconditioning
    timer_solver_init_precon.start();

    double tol = get_tolerance();
    PARCEL_INT itrmax = get_max_iteration();
    precon_param &precon = *(get_precon_param());
    parcel::precon_matrix<
      Tmat,
      Tvec,
      Tprecon_mat,
      Tprecon_vec,
      Tprecon
      > precon_Mat(Mat,precon);
    timer_solver_init_precon.end();
    // ------------------------------
#if PARCEL_MPI
    timer_solver_init_convert.start();
    parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x; vec_util.setptr( n, x_parallel.vec(0), x, 0);
    parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
    parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);    
    parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
    timer_solver_init_convert.end();
    parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
    parcel_blas.zero_init<Tvec>(r);

    // 列アドレスは ld_p (= parallel_vector の列間隔) で組み立てる。
    //   (n+nadd) を直に書くと、確保側の間隔が変わったときに黙って別の列を読む。
    parcel::parallel_vector<Tvec> p_history_parallel(n,nadd,pstep,que);
    // ld_p は size_t。ld_p*pstep_now などの積を int32 で作ると桁あふれする。
    const size_t ld_p = p_history_parallel.ld();
    parcel::vector<Tvec> p_history; 
    vec_util.setptr( ld_p*pstep, p_history_parallel.vec_add(),p_history, 0);
    parcel_blas.zero_init<Tvec>(p_history);
    parcel::parallel_vector<Tvec> q_parallel(n,nadd,1,que);parcel::vector<Tvec> q;vec_util.setptr( n, q_parallel.vec(0), q, 0);
    parcel::parallel_vector<Tvec> s_parallel(n,nadd,1,que);parcel::vector<Tvec> s;vec_util.setptr( n, s_parallel.vec(0), s, 0);
    parcel_blas.zero_init<Tvec>(q);
    parcel_blas.zero_init<Tvec>(s);
    // r_true は q の領域を使う (先頭の説明)
    parcel::vector<Tvec> r_true;vec_util.setptr( n, q_parallel.vec(0), r_true, 0);
#else    
    timer_solver_init_convert.start();
    parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
    parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
    timer_solver_init_convert.end();

    parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
    parcel::vector<Tvec> p_history((size_t)n*pstep,que);parcel_blas.zero_init<Tvec>(p_history);
    const size_t ld_p = n;   // 非 MPI は詰め物を入れない
    parcel::vector<Tvec> p;
    parcel::vector<Tvec> p_old;
    vec_util.setptr(n,p_history,p);
    parcel::vector<Tvec> q(n,que);parcel_blas.zero_init<Tvec>(q);
    parcel::vector<Tvec> s(n,que);parcel_blas.zero_init<Tvec>(s);
    parcel::vector<Tvec> r_true;vec_util.setptr(n,q,r_true);   // q の領域を使う
#endif

    // ------------------------------

    parcel::vector<Tdot> one;
    one.resize(1,que);
    double one_ = 1.0;one.setValue(&one_);

    parcel::vector<Tdot> zero;
    zero.resize(1,que);
    double zero_ = 0.0;zero.setValue(&zero_);

    parcel::vector<Tdot> negative_one;
    negative_one.resize(1,que);
    double negative_one_ = -1.0;negative_one.setValue(&negative_one_);
	
    parcel::vector<Tdot> work;
    work.resize(1,que);
    parcel_blas.zero_init<Tdot>(work);
	
    // ------------------------------	
    // || b ||
    parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
    parcel::scalar bnorm_host = parcel_blas.get_dot();
    bnorm_host = host_util.sqrt(bnorm_host);
    PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

    // ||b|| = 0 なら解は x = 0 (残差を 0 で割って NaN にしないため反復しない)。
    //   bnorm は allreduce 済みなので全ランクが同じ分岐を取る。
    if(bnorm_ == 0.0){
      parcel_blas.zero_init<Tvecx>(x0);
      timer_solver_init.end();
      timer_solver.end();
      if(myrank == 0){
	std::cout<<" ||b|| = 0 : x = 0 (no iteration)"<<std::endl;
      }
      set_status(CONVERGED_RTOL, 0, 0.0, 0.0);
      print_status(myrank);
      return ;
    }

    // r = b - Ax   
#if PARCEL_MPI
    parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat, x_parallel.vec_add(0),b,r);
#else
    parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif

    // || r ||
    parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
    parcel::scalar rnorm_host = parcel_blas.get_dot();
    rnorm_host = host_util.sqrt(rnorm_host);
    PARCEL_MAX_FLOTING_TYPE  rnorm_ = rnorm_host.get_T();

    // p = 0.0
#if PARCEL_MPI
    

#else
    parcel_blas.zero_init<Tvec>(p);
#endif    
    // solve Ms=r 
    // preconditioning
#if PARCEL_MPI    
      parcel::vector<Tvec> r_vecadd ;vec_util.setptr( (n+nadd), r_parallel.vec_add(), r_vecadd,0); 
      parcel::vector<Tvec> s_vecadd ;vec_util.setptr( (n+nadd), s_parallel.vec_add(), s_vecadd,0); 
      precon_Mat.solver( r_vecadd, s_vecadd);
#else
    precon_Mat.solver(r,s);
#endif

    // -----------
    // rho0=<s,r> 
    parcel::vector<Tdot> rho0;rho0.resize(1,que);
    parcel::vector<Tdot> rho1;rho1.resize(1,que);
    parcel::scalar rho1_host;

    parcel_blas.dot< Tvec,Tvec,Tdot >(s,r);
    parcel::scalar rho0_host = parcel_blas.get_dot();

    // -----------
    parcel::scalar alpha_host;
    parcel::scalar beta_host;
    parcel::scalar gamma_host;

    parcel::vector<Tdot> alpha; alpha.resize(1,que);
    parcel::vector<Tdot> beta;  beta.resize(1,que) ;
    parcel::vector<Tdot> gamma; gamma.resize(1,que);

    parcel::vector<Tdot> alpha_history; alpha_history.resize(pstep,que);

    alpha_host.set_Tdd(0.0);
    beta_host.set_Tdd(0.0);
    gamma_host.set_Tdd(0.0);

    alpha.setValue(alpha_host); 
    beta.setValue(beta_host)  ; 
    // 係数を値で axpy に渡すか。double-double を含む組み合わせでは 1 要素のデバイス配列を経由する。
    //   値渡しの経路では、毎反復の setValue のうち axpy だけが使うもの (work = -alpha、beta) と、
    //   デバイス側の値がどこからも読まれないもの (gamma、rho1) を省く。alpha は x の一括更新
    //   (alpha_history -> gemv) がデバイスの値を読むので、どちらの経路でも書き戻す。
    constexpr bool coef_by_value = parcel::blas<PARCEL_MAX_FLOTING_TYPE>::template coef_by_value_supported<Tdot,Tvec,Tvec,Taxpy>();
    parcel::scalar one_host; one_host.set_Tdd(1.0);
    double bnorm_FP64 = bnorm_;
    double rnorm_FP64 = rnorm_;

    if( myrank == 0){
      std::cout<<std::setw(8)<< 0 <<" : residual norm = ";
      std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
    }

    // 初期残差が収束判定値未満 (x0 がすでに解) なら反復に入らずに返す
    //   (r0 = 0 で反復すると alpha = rho0/<p,Ap> が 0/0 になる)。
    //   x0 は書き換えない。rnorm, bnorm は allreduce 済みなので全ランクが同じ分岐を取る。
    //   NaN は比較が偽になるので反復に入る。
    if((rnorm_/bnorm_)<tol){
      timer_solver_init.end();
      timer_solver.end();
      set_status(CONVERGED_RTOL, 0, rnorm_FP64/bnorm_FP64, rnorm_FP64/bnorm_FP64);
      print_status(myrank);
      return ;
    }

    PARCEL_INT pstep_end = (itrmax)%pstep+1; // 反復ループを抜けた後に一括更新するベクトル本数
    // 収束状態 (krylov_param::set_status)。判定に使うのは漸化式の残差 rnorm_/bnorm_
    bool   st_converged = false;
    bool   st_nonfinite = false;
    PARCEL_INT st_its   = 0;
    double st_res       = rnorm_FP64/bnorm_FP64;
    vec_util.copy_at<Tdot>(alpha,alpha_history,0,0);
    timer_solver_init.end();
    timer_solver_mainloop.start();
    for(PARCEL_INT itr = 0;itr < itrmax;itr++){
      //      x = x + alpha * p 
      PARCEL_INT pstep_now  =  itr%pstep;
      
#if PARCEL_MPI

#else
      vec_util.setptr(n,p_history,p,(size_t)pstep_now*n);
#endif      
#if PARCEL_CG_FUSED
      // pstep = 1 で係数を値で渡す組み合わせなら、x の更新を下の p の更新と 1 本のカーネルにまとめる (PARCEL_CG_FUSED)。
      //   TRUE_RESIDUAL では x を真の残差の前に更新する必要があるので使わない。
#if PARCEL_MPI && !TRUE_RESIDUAL
      const bool cg_fuse_xp = (pstep == 1) && coef_by_value;
#else
      const bool cg_fuse_xp = false;
#endif
      if(!cg_fuse_xp && pstep_now==(pstep-1)){
#else
      if(pstep_now==(pstep-1)){
#endif
	// pstep分のベクトルで更新
	//   実行されるのは pstep 反復に 1 回だけ。gemv 専用のタイマで測る。
	timer_solver_gemv.start();
#if PARCEL_MPI
	parcel::vector<Tvec> p_update ;vec_util.setptr( ld_p*pstep, p_history, p_update ,0); 
	// y は x_parallel (詰め物なし) なので、A の ld は明示的に渡す。
	parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep,one,one,
							 p_update,
							 alpha_history,
							 x_parallel.vec_add(0),
							 ld_p);
#else
	parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep,one,one,p_history,alpha_history,x);
#endif	
	timer_solver_gemv.end();
      }
      
      


#if TRUE_RESIDUAL
      
#if PARCEL_MPI
      parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r_true);	
#else
      parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r_true);
#endif      
      parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);
      parcel::scalar rnorm_host = parcel_blas.get_dot();
      rnorm_host = host_util.sqrt(rnorm_host);
      PARCEL_MAX_FLOTING_TYPE rnorm_true = rnorm_host.get_T();
#endif

      timer_solver_axpy.start();
      //      p = s + beta * p 
      PARCEL_INT pstep_next  = (itr+1)%pstep;
      parcel::vector<Tvec> p_now ;vec_util.setptr( n, p_history, p_now , ld_p*pstep_now); 
      parcel::vector<Tvec> p_next;vec_util.setptr( n, p_history, p_next, ld_p*pstep_next);

#if PARCEL_MPI
#if PARCEL_CG_FUSED
      if(cg_fuse_xp){
	if constexpr (coef_by_value){
	  // x = x + alpha p と p = s + beta p (p_now == p_next) を 1 本で
	  parcel_blas.cg_xp_fused<Tdot,Tvec,Taxpy>(one, alpha_history, one_host, beta_host, s, p_now, x_parallel.vec_add(0));
	}
      }else
#endif
      if constexpr (coef_by_value){
	parcel_blas.axpby_value<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one_host,beta_host,s,p_now,p_next);
      }else{
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one,beta,s,p_now,p_next);
      }
#else      
      vec_util.setptr(n,p_history,p_old,(size_t)pstep_now *n );
      vec_util.setptr(n,p_history,p    ,(size_t)pstep_next*n);
      if constexpr (coef_by_value){
	parcel_blas.axpby_value<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one_host,beta_host,s,p_old,p);
      }else{
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one,beta,s,p_old,p);
      }
#endif      

      timer_solver_axpy.end();

      //      q = Ap 
      timer_solver_SpMV.start();
#if PARCEL_MPI
      
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,p_history_parallel.vec_add(pstep_next),q);
      
#else
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,p,q);
#endif
      timer_solver_SpMV.end();

      //      gamma = <p,q> 
      timer_solver_dot.start();
#if PARCEL_MPI      
      parcel_blas.dot< Tvec,Tvec,Tdot>(p_history_parallel.vec(pstep_next),q);
#else
      parcel_blas.dot< Tvec,Tvec,Tdot>(p,q);
#endif      
      timer_solver_dot.end();

      timer_solver_scala.start();
      gamma_host = parcel_blas.get_dot();
      if constexpr (!coef_by_value) gamma.setValue(gamma_host);   // デバイスの gamma は読まれない (値渡しでは省く)

      //      alpha=rho0/gamma 
      alpha_host = host_util.div(rho0_host,gamma_host);
      alpha.setValue(alpha_host);
      timer_solver_scala.end();

      //      r = r - alpha * q 
      parcel::scalar nega_one_host;nega_one_host.set_Tdd(-1.0);
      parcel::scalar alpha_1_host = host_util.mul(nega_one_host,alpha_host);
      if constexpr (!coef_by_value) work.setValue(alpha_1_host); 

#if PARCEL_CG_FUSED
      // r = r + (-alpha) q と <r, r> を 1 本のカーネルで (MGS と同じ axpy_dot_value。時間は内積に計上する)
      if constexpr (coef_by_value){
	timer_solver_dot.start();
	parcel_blas.axpy_dot_value<Tdot,Tvec,Taxpy,Tdot>(alpha_1_host,q,r,r);
	timer_solver_dot.end();
      }else{
#endif
      timer_solver_axpy.start();
      if constexpr (coef_by_value){
	parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy> (alpha_1_host,q,r);   // r = r + (-alpha) q
      }else{
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (work,one,q,r);
      }
      timer_solver_axpy.end();

      timer_solver_dot.start();
      parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
      timer_solver_dot.end();
#if PARCEL_CG_FUSED
      }
#endif

      timer_solver_scala.start();
      rnorm_host = host_util.sqrt( parcel_blas.get_dot() );
      rnorm_ = rnorm_host.get_T();
      timer_solver_scala.end();


#if TRUE_RESIDUAL
      double res_norm      = rnorm_/bnorm_;
      double res_norm_true = rnorm_true/bnorm_;
      if( myrank == 0){
	std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	std::cout<<std::scientific << std::setprecision(14) << res_norm <<" , true = "<< res_norm_true <<std::endl;
      }
#else
      double res_norm      = rnorm_/bnorm_;
      if( myrank == 0){
	std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	std::cout<<std::scientific << std::setprecision(14) << res_norm <<std::endl;
      }
#endif

      st_its = itr+1;
      st_res = res_norm;
      vec_util.copy_at<Tdot>(alpha,alpha_history,0,pstep_next);
      // 非有限になったらその場で抜ける。
      //   x の一括更新本数は収束時と同じ数え方にそろえる。
      if(!std::isfinite(res_norm)){
	pstep_end = pstep_next+1;
	st_nonfinite = true;
	break;
      }
      if((rnorm_/bnorm_)<tol){
	pstep_end = pstep_next+1;
	st_converged = true;
	break;
      }
      //      solve Ms=r 
      // preconditioning
      timer_solver_precon.start();
#if PARCEL_MPI
      parcel::vector<Tvec> r_vecadd ;vec_util.setptr( (n+nadd), r_parallel.vec_add(), r_vecadd,0); 
      parcel::vector<Tvec> s_vecadd ;vec_util.setptr( (n+nadd), s_parallel.vec_add(), s_vecadd,0); 
      precon_Mat.solver( r_vecadd, s_vecadd);
#else      
      precon_Mat.solver(r,s);
#endif      
      timer_solver_precon.end();

      // rho1 = <s,r> 
      timer_solver_dot.start();
      parcel_blas.dot< Tvec,Tvec,Tdot >(s,r);
      timer_solver_dot.end();

      timer_solver_scala.start();
      rho1_host = parcel_blas.get_dot();
      if constexpr (!coef_by_value) rho1.setValue(rho1_host);   // デバイスの rho1 は読まれない (値渡しでは省く)

      beta_host = host_util.div(rho1_host,rho0_host); 
      if constexpr (!coef_by_value) beta.setValue(beta_host);   // 値渡しでは次の p の更新に beta_host を直接渡す

      rho0_host = rho1_host;
      timer_solver_scala.end();

    }
    timer_solver_mainloop.end();

    // x = x + alpha*p
    // pstep_end分のベクトルで更新
#if PARCEL_MPI
    parcel::vector<Tvec> p_update ;vec_util.setptr( ld_p*pstep_end, p_history, p_update ,0); 
    parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep_end,one,one,
						     p_update,
						     alpha_history,
						     x_parallel.vec_add(0),
						     ld_p );
    
#else
    parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep_end,one,one,p_history,alpha_history,x);
#endif
    


    vec_util.convert<Tvec,Tvecx>(x,x0);


#if PARCEL_MPI
    parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r_true);
#else
    parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(Mat,x0,b,r_true);
#endif    
    parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);
    parcel::scalar rnorm_host_ = parcel_blas.get_dot();
    rnorm_host_ = host_util.sqrt(rnorm_host_);

    if( myrank == 0){
      std::cout<<std::setw(8)<<" finish : residual true norm  || b - Ax || = ";
    }
    double finish_res_norm = rnorm_host_.get_T()/bnorm_;
    if( myrank == 0){
      std::cout<<std::scientific << std::setprecision(10) << finish_res_norm <<std::endl;
    }
    timer_solver.end();
      if( myrank == 0){
        double t_pcg    = timer_solver.get();
        double t_init   = timer_solver_init.get();
        double t_initP  = timer_solver_init_precon.get();
        double t_initC  = timer_solver_init_convert.get();
        double t_loop   = timer_solver_mainloop.get();
        double t_dot    = timer_solver_dot.get();
        double t_axpy   = timer_solver_axpy.get();
        double t_gemv   = timer_solver_gemv.get();
        double t_spmv   = timer_solver_SpMV.get();
        double t_precon = timer_solver_precon.get();
        double t_init_other = t_init - t_initP - t_initC;
        double t_loop_other = t_loop - (t_dot + t_axpy + t_gemv + t_spmv + t_precon);
        bool   loop_other_clamped = false;
        if (t_loop_other < 0.0) { t_loop_other = 0.0; loop_other_clamped = true; }
        double t_postloop   = t_pcg - t_init - t_loop;
        std::cout<<std::endl<<" ----------------- "<<std::endl;
        std::cout<<"pcg time              = "<<t_pcg        <<" [sec]  (total solver wall time)"<<std::endl;
        std::cout<<"  init time           = "<<t_init       <<" [sec]  (before main loop)"<<std::endl;
        std::cout<<"    init precon       = "<<t_initP      <<" [sec]"<<std::endl;
        std::cout<<"    init convert      = "<<t_initC      <<" [sec]"<<std::endl;
        std::cout<<"    init other        = "<<t_init_other <<" [sec]  (vec alloc/zero, init SpMV/dot/precon)"<<std::endl;
        std::cout<<"  main loop time      = "<<t_loop       <<" [sec]  (sum over all CG iterations)"<<std::endl;
        // dot / SpMV は「局所カーネル + MPI 集約」の合計。内訳を分けて出す。
        const double t_dot_comm  = parcel_blas.get_time_dot_comm();
        const double t_spmv_comm = parcel_blas.get_time_SpMV_comm();
        std::cout<<"    loop dot          = "<<t_dot        <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
        std::cout<<"      dot comm        = "<<t_dot_comm   <<" [sec]  (MPI_Allreduce)"<<std::endl;
        std::cout<<"      dot kernel      = "<<(t_dot - t_dot_comm) <<" [sec]  (局所内積)"<<std::endl;
        std::cout<<"    loop axpy         = "<<t_axpy       <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
        std::cout<<"    loop gemv         = "<<t_gemv       <<" [sec]  calls = "<<timer_solver_gemv.get_count()<<"  (pstep="<<pstep<<" 本まとめた x の更新)"<<std::endl;
        std::cout<<"    loop SpMV         = "<<t_spmv       <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
        std::cout<<"      SpMV comm       = "<<t_spmv_comm  <<" [sec]  (ハロ交換)"<<std::endl;
        std::cout<<"      SpMV kernel     = "<<(t_spmv - t_spmv_comm)<<" [sec]  (局所行列ベクトル積)"<<std::endl;
        std::cout<<"    loop precon       = "<<t_precon     <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
        std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (kernel launch/sync, cout per iter)"<<std::endl;
        if (loop_other_clamped) {
          std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
        }
        std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (final r_true compute, finish print)"<<std::endl;
      }

    // 収束状態。true_residual は上で表示した finish の値 (||b|| で割った相対値)
    set_status(decide_reason(st_converged, st_nonfinite), st_its, st_res, finish_res_norm);
    print_status(myrank);
    return ;

  }
};

  }
}


#endif
