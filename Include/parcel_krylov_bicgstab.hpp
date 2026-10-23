// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_BICGSTAB_HPP_
#define PARCEL_KRYLOV_BICGSTAB_HPP_

// 作業ベクトルの共有。
//   生存区間の重ならない作業ベクトルを同じ領域に置き、袖付きベクトルを 8 本に抑える。
//       e      は r の領域: e = r - alpha q をその場で作る。次に r を書くのは r = e - c3 v で、それまで
//                          古い r は読まない。そのため複写 e = r と r = e が要らない。
//       pe     は pp の領域: x = x + alpha pp を alpha が決まった直後 (e を作る前) に行い、pp をそこで
//                          使い終える。x の更新は x += alpha pp; x += c3 pe の 2 回の axpy で、
//                          間で x を読まない。v = 0 の打ち切りでは x += alpha pp を二重に足さない。
//       r_true は pp の領域: r_true を作るのは x を 2 回更新した後 (pe を使い終えた後) とループ後だけ。
namespace parcel{
  namespace krylov{

    template<
      // 他のソルバと同じ 4 引数 (SpMV / 前処理 / ベクトル演算 / 縮約演算の計算精度)。
      class Tspmv   = parcel_double,
      class Tprecon = parcel_double,
      class Taxpy   = parcel_double,
      class Tdot    = parcel_double 
      >
    class pbicgstab : public krylov_param{

      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_scala;
      
      parcel::timer timer_solver_SpMV;
      parcel::timer timer_solver_axpy;
      parcel::timer timer_solver_dot;
      parcel::timer timer_solver_precon;
      parcel::timer timer_solver_mainloop;

    public :
      pbicgstab(){}
      ~pbicgstab(){}

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


	require_in_order(que, "krylov::pbicgstab");   // in-order キューが前提
	const PARCEL_INT n = x0.size();
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

	timer_solver_init_convert.start();
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);    
	parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);	
	parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);    
	parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);

	parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel_blas.zero_init<Tvec>(r);
	
	parcel::parallel_vector<Tvec> r0_parallel(n,nadd,1,que);parcel::vector<Tvec> r0;vec_util.setptr( n, r0_parallel.vec(0), r0, 0);
	parcel::parallel_vector<Tvec> p_parallel(n,nadd,1,que);parcel::vector<Tvec> p;vec_util.setptr( n, p_parallel.vec(0), p, 0);
	parcel::parallel_vector<Tvec> q_parallel(n,nadd,1,que);parcel::vector<Tvec> q;vec_util.setptr( n, q_parallel.vec(0), q, 0);
	// e は r、pe と r_true は pp の領域 (先頭の「作業ベクトルの共有」の説明)
	parcel::parallel_vector<Tvec> v_parallel(n,nadd,1,que);parcel::vector<Tvec> v;vec_util.setptr( n, v_parallel.vec(0), v, 0);
	parcel::parallel_vector<Tvec> pp_parallel(n,nadd,1,que);parcel::vector<Tvec> pp;vec_util.setptr( n, pp_parallel.vec(0), pp, 0);
	parcel::vector<Tvec> pe;     vec_util.setptr( n, pp_parallel.vec(0), pe, 0);
	parcel::vector<Tvec> r_true; vec_util.setptr( n, pp_parallel.vec(0), r_true, 0);
	parcel::vector<Tvec> e;      vec_util.setptr( n, r_parallel.vec(0), e, 0);

	parcel_blas.zero_init<Tvec>(r0);
	parcel_blas.zero_init<Tvec>(p);
	parcel_blas.zero_init<Tvec>(q);
	parcel_blas.zero_init<Tvec>(v);
	parcel_blas.zero_init<Tvec>(pp);
  
#else
	parcel::vector<Tvec> x(n,que);parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que);parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);

	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);

	parcel::vector<Tvec> r0(n,que);parcel_blas.zero_init<Tvec>(r0);
	parcel::vector<Tvec> p (n,que);parcel_blas.zero_init<Tvec>(p);
	parcel::vector<Tvec> q (n,que);parcel_blas.zero_init<Tvec>(q);
	// e は r、pe と r_true は pp の領域 (先頭の「作業ベクトルの共有」の説明)
	parcel::vector<Tvec> v (n,que);parcel_blas.zero_init<Tvec>(v);
	parcel::vector<Tvec> pp(n,que);parcel_blas.zero_init<Tvec>(pp);
	parcel::vector<Tvec> pe;     vec_util.setptr( n, pp, pe, 0);
	parcel::vector<Tvec> r_true; vec_util.setptr( n, pp, r_true, 0);
	parcel::vector<Tvec> e;      vec_util.setptr( n, r, e, 0);
#endif	
	timer_solver_init_convert.end();
	// ------------------------------

	parcel::vector<Tdot> one;
	one.resize(1,que);
	double one_ = 1.0;
	one.setValue(&one_);

	parcel::vector<Tdot> zero;
	zero.resize(1,que);
	double zero_ = 0.0;
	zero.setValue(&zero_);

	parcel::vector<Tdot> negative_one;
	negative_one.resize(1,que);
	double negative_one_ = -1.0;
	negative_one.setValue(&negative_one_);

	parcel::vector<Tdot> work;
	work.resize(1,que);
	parcel_blas.zero_init<Tdot>(work);

	// ------------------------------

	parcel::vector<parcel_double> c1(1,que);
	parcel::vector<parcel_double> c2(1,que);
	// 係数は縮約演算の精度 Tdot で持つ (他のスカラと同じ)。
	parcel::vector<Tdot> c3(1,que);
	parcel::vector<Tdot> alpha(1,que);
	parcel::vector<Tdot> beta(1,que);

	// ! ------
	// ! pbicgstab start
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
    parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	// || r ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	parcel::scalar rnorm_host = parcel_blas.get_dot();
	rnorm_host = host_util.sqrt(rnorm_host);
	PARCEL_MAX_FLOTING_TYPE rnorm_ = rnorm_host.get_T();

	// 初期残差が収束判定値未満 (x0 がすでに解) なら反復に入らずに返す
	//   (r0 = 0 で反復すると c2 = <r0,q> = 0 で破綻する)。
	//   x0 は書き換えない。rnorm, bnorm は allreduce 済みなので全ランクが同じ分岐を取る。
	//   NaN は比較が偽になるので反復に入る。
	if((rnorm_/bnorm_)<tol){
	  const double res0 = (double)rnorm_/(double)bnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<< 0 <<" : residual norm = ";
	    std::cout<<std::scientific << std::setprecision(10) << res0 <<std::endl;
	  }
	  timer_solver_init.end();
	  timer_solver.end();
	  set_status(CONVERGED_RTOL, 0, res0, res0);
	  print_status(myrank);
	  return ;
	}

	// ! r0=r
	parcel_blas.copy<Tvec>(r,r0);

	// ! p=r
	parcel_blas.copy<Tvec>(r,p);

	// ! c1=<r0,r0>
	parcel_blas.dot< Tvec,Tvec,Tdot >(r0,r0);
	parcel::scalar c1_host = parcel_blas.get_dot();
	c1.setValue(c1_host);
	PARCEL_MAX_FLOTING_TYPE c1_ = parcel_blas.get_T();
	// 係数を値で axpy に渡すか。double-double を含む組み合わせでは 1 要素のデバイス配列を経由する。
	//   値渡しの経路では、毎反復の setValue のうち axpy だけが使うもの (alpha、c3、beta、work = -alpha / -c3) と、
	//   デバイス側の値がどこからも読まれないもの (c1、c2) を省く。
	constexpr bool coef_by_value = parcel::blas<PARCEL_MAX_FLOTING_TYPE>::template coef_by_value_supported<Tdot,Tvec,Tvec,Taxpy>();
	parcel::scalar one_host; one_host.set_Tdd(1.0);
	
	// 収束状態 (krylov_param::set_status)。判定に使うのは漸化式の残差 rnorm_/bnorm_
	//   破綻 (ゼロ割り) と非有限値を検出して反復を打ち切る。
	//     c2 = <r0,q> が 0 か非有限            -> DIVERGED_BREAKDOWN (alpha = c1/c2 が作れない)
	//     <v,v> = 0 (v = A M^-1 e = 0)          -> e = 0 の「幸運な破綻」。x += alpha*pp で解が
	//                                             求まっているので ||e||/||b|| < tol なら CONVERGED_RTOL
	//     c3 = 0 または c1 = <r0,r> = 0          -> DIVERGED_BREAKDOWN (beta = c1/(c3*c2) が作れない)
	//     残差が非有限                          -> その場で抜けて DIVERGED_NANORINF
	bool   st_converged = false;
	bool   st_nonfinite = false;
	bool   st_breakdown = false;
	PARCEL_INT st_its   = 0;
	double st_res       = rnorm_/bnorm_;
	timer_solver_init.end();
	timer_solver_mainloop.start();
	for(PARCEL_INT itr = 0;itr < itrmax;itr++){
	  // preconditioning
	  // !   solve Mpp=p
	  timer_solver_precon.start();
	  precon_Mat.solver(p,pp);
	  timer_solver_precon.end();

	  // !   q=App
	  timer_solver_SpMV.start();
	  parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>
	    (one,zero,Mat,pp,q);
	  timer_solver_SpMV.end();

	  // !   c2=<r0,q>
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r0,q);
	  parcel::scalar c2_host = parcel_blas.get_dot();
	  timer_solver_dot.end();
	  if constexpr (!coef_by_value) c2.setValue(c2_host);   // デバイスの c2 は読まれない (値渡しでは省く)
	  {
	    const double c2_d = (double)c2_host.get_T();
	    if(c2_d == 0.0 || !std::isfinite(c2_d)){
	      if(myrank == 0){
		// 非有限値は数値で書かない (回帰試験の nan / inf 検出に引っかけないため)
		std::cout<<" pbicgstab breakdown : <r0,q> "<<(c2_d == 0.0 ? "= 0" : "is not finite")
			 <<" at iteration "<<itr+1<<std::endl;
	      }
	      st_breakdown = true;
	      break;
	    }
	  }

	  // !   alpha = c1/c2
	  parcel::scalar alpha_host = host_util.div(c1_host,c2_host);
	  if constexpr (!coef_by_value) alpha.setValue(alpha_host);

	  // !   e=r-alpha*q
	  parcel::scalar nega_one_host;nega_one_host.set_Tdd(-1.0);
	  parcel::scalar alpha_1_host = host_util.mul(nega_one_host,alpha_host);
	  if constexpr (!coef_by_value) work.setValue(alpha_1_host);
	  timer_solver_axpy.start();
	  // x = x + alpha pp をここで行い pp を使い終える (pe が pp の領域を使うため)。
	  //   x += c3 pe は c3 が決まった後。x はその間で読まない
	  if constexpr (coef_by_value){
	    parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy>(alpha_host,pp,x);
	  }else{
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(alpha,one,pp,x);
	  }
	  // e は r の領域なので複写 e = r は要らない (その場で e = r - alpha q)
	  if constexpr (coef_by_value){
	    parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy>(alpha_1_host,q,e);
	  }else{
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,q,e);
	  }
	  timer_solver_axpy.end();

	  // preconditioning
	  // !   solve Mpe=e
	  timer_solver_precon.start();
	  precon_Mat.solver(e,pe);
	  timer_solver_precon.end();

	  // !   v=Ape
	  timer_solver_SpMV.start();
	  parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pe,v);
	  timer_solver_SpMV.end();

	  // !   c3=<e,v>/<v,v>
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(e,v);
	  parcel::scalar ev_host = parcel_blas.get_dot();
	  timer_solver_dot.end();

	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(v,v);
	  parcel::scalar vv_host = parcel_blas.get_dot();
	  timer_solver_dot.end();
	  if((double)vv_host.get_T() == 0.0){
	    // v = 0 : x += alpha*pp で打ち切る (c3 = <e,v>/<v,v> は 0/0 になるので作らない)
	    parcel_blas.dot< Tvec,Tvec,Tdot >(e,e);
	    parcel::scalar enorm_host = host_util.sqrt(parcel_blas.get_dot());
	    const double res_e = (double)enorm_host.get_T()/bnorm_;
	    if( myrank == 0){
	      std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	      std::cout<<std::scientific << std::setprecision(10) << res_e <<std::endl;
	    }
	    st_its = itr+1;
	    st_res = res_e;
	    if(!std::isfinite(res_e))  st_nonfinite = true;
	    else if(res_e < tol)       st_converged = true;
	    else                       st_breakdown = true;
	    break;
	  }
	  parcel::scalar c3_host = host_util.div(ev_host,vv_host);
	  if constexpr (!coef_by_value) c3.setValue(c3_host);

	  // !   x=x+alpha*pp+c3*pe
	  timer_solver_axpy.start();
	  if constexpr (coef_by_value){
	    parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy>(c3_host,pe,x);
	  }else{
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(c3,one,pe,x);
	  }
	  timer_solver_axpy.end();

#if TRUE_RESIDUAL

#if PARCEL_MPI
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r_true);
#else
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r_true);
#endif
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);

	  rnorm_host = parcel_blas.get_dot();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  PARCEL_MAX_FLOTING_TYPE rnorm_true = rnorm_host.get_T();

#endif

	  // !   r=e-c3*v
	  timer_solver_axpy.start();

	  parcel::scalar c3_1_host = host_util.mul(nega_one_host,c3_host);
	  if constexpr (coef_by_value){
	    parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy>(c3_1_host,v,r);
	  }else{
	    work.setValue(c3_1_host);
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,v,r);
	  }
	  timer_solver_axpy.end();

	  // !   c1=<r0,r>
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r0,r);
	  c1_host = parcel_blas.get_dot();
	  timer_solver_dot.end();
	  if constexpr (!coef_by_value){ c1.setValue(c1_host); c1.setValue(c1_host); }   // デバイスの c1 は読まれない

	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	  rnorm_host = parcel_blas.get_dot();
	  timer_solver_dot.end();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  rnorm_ = rnorm_host.get_T();
	  double res_norm      = rnorm_/bnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	    std::cout<<std::scientific << std::setprecision(10) << res_norm <<std::endl;
	  }
	  st_its = itr+1;
	  st_res = res_norm;
	  // 非有限になったらその場で抜ける
	  if(!std::isfinite(res_norm)){ st_nonfinite = true; break; }
	  // !   if(||r||<rmax) exit
	  if((rnorm_/bnorm_)<tol){ st_converged = true; break; }
	  // beta = c1/(c3*c2) が作れない
	  if((double)c3_host.get_T() == 0.0 || (double)c1_host.get_T() == 0.0){
	    if(myrank == 0){
	      std::cout<<" pbicgstab breakdown : "
		       <<((double)c3_host.get_T() == 0.0 ? "omega (c3) = 0" : "rho (<r0,r>) = 0")
		       <<" at iteration "<<itr+1<<std::endl;
	    }
	    st_breakdown = true;
	    break;
	  }

	  // !   beta=c1/(c3*c2)
	  parcel::scalar beta_host = host_util.div(c1_host, host_util.mul(c3_host,c2_host) );
	  if constexpr (!coef_by_value) beta.setValue(beta_host);

	  // !   p=r+beta*(p-c3*q)
	  c3_1_host = host_util.mul(nega_one_host,c3_host);
	  if constexpr (!coef_by_value) work.setValue(c3_1_host);

	  timer_solver_axpy.start();
	  if constexpr (coef_by_value){
	    parcel_blas.axpy_value<Tdot,Tvec,Tvec,Taxpy>(c3_1_host,q,p);                 // p = p + (-c3) q
	    parcel_blas.axpby_value<Tdot,Tdot,Tvec,Tvec,Taxpy>(one_host,beta_host,r,p);  // p = beta p + r
	  }else{
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,q,p);
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(one,beta,r,p);
	  }
	  timer_solver_axpy.end();

	  // ! enddo
	  // ! ------
	}
	timer_solver_mainloop.end();

	// post-loop true residual: || b - A x_final ||
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r_true);
#else
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r_true);
#endif
	parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);
	double finish_res_norm = 0.0;
	{
	  parcel::scalar rnorm_host_finish = parcel_blas.get_dot();
	  rnorm_host_finish = host_util.sqrt(rnorm_host_finish);
	  finish_res_norm = rnorm_host_finish.get_T()/bnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<<" finish : residual true norm  || b - Ax || = ";
	    std::cout<<std::scientific << std::setprecision(10) << finish_res_norm <<std::endl;
	  }
	}

	vec_util.convert<Tvec,Tvecx>(x,x0);

    timer_solver.end();
    if( myrank == 0){
        double t_solver = timer_solver.get();
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
        double t_postloop   = t_solver - t_init - t_loop;
        std::cout<<std::endl<<" ----------------- "<<std::endl;
        std::cout<<"pbicg-stab time       = "<<t_solver     <<" [sec]  (total solver wall time)"<<std::endl;
        std::cout<<"  init time           = "<<t_init       <<" [sec]  (before main loop)"<<std::endl;
        std::cout<<"    init precon       = "<<t_initP      <<" [sec]"<<std::endl;
        std::cout<<"    init convert      = "<<t_initC      <<" [sec]"<<std::endl;
        std::cout<<"    init other        = "<<t_init_other <<" [sec]  (vec alloc/zero, init SpMV/dot/precon)"<<std::endl;
        std::cout<<"  main loop time      = "<<t_loop       <<" [sec]  (sum over all iterations)"<<std::endl;
        std::cout<<"    loop dot          = "<<t_dot        <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
        std::cout<<"    loop axpy         = "<<t_axpy       <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
        std::cout<<"    loop SpMV         = "<<t_spmv       <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
        std::cout<<"    loop precon       = "<<t_precon     <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
        std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (kernel launch/sync, cout per iter)"<<std::endl;
        if (loop_other_clamped) {
          std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
        }
        std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (vec convert, finish print)"<<std::endl;
    }
    // 収束状態。true_residual は上で表示した finish の値 (||b|| で割った相対値)
    set_status(st_converged ? CONVERGED_RTOL
	       : (st_breakdown ? DIVERGED_BREAKDOWN : decide_reason(false, st_nonfinite)),
	       st_its, st_res, finish_res_norm);
    print_status(myrank);
    return ;
      }
    };
  }
}
#endif
