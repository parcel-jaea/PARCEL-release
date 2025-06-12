#ifndef PARCEL_KRYLOV_CG_HPP_
#define PARCEL_KRYLOV_CG_HPP_

// #define TRUE_RESIDUAL 0
#define pstep_update 1
// #define pstep_update 0

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
  // double tol_ = 1.0e-4;
  // PARCEL_INT itrmax_ = 100;
  // parcel::precon_param precon;

  ddlib::ddmath dd;
  parcel::timer timer_solver;
  parcel::timer timer_solver_init;
  parcel::timer timer_solver_init_precon;
  parcel::timer timer_solver_init_convert;
  parcel::timer timer_solver_scala;

  parcel::timer timer_solver_SpMV;
  parcel::timer timer_solver_axpy;
  parcel::timer timer_solver_dot;
  parcel::timer timer_solver_precon;
  //  PARCEL_INT pstep=100;
  //  PARCEL_INT pstep=8;
  //PARCEL_INT pstep=1;
  PARCEL_INT pstep=4;
public :
  
  pcg(){}
  ~pcg(){}
  
  // 前処理,IR等の設定をclassで定義する
  template<
    class Tmat,  // 係数行列データ精度
    class Tvecx, // 入力ベクトルデータ精度
    class Tvecb, // 右辺ベクトルデータ精度
    class Tvec , // クリロフ部分空間法内部データ精度	
    class Tprecon_mat, // 前処理行列データ精度
    class Tprecon_vec  // 前処ベクトルデータ精度
    >
  void solver(
	      parcel::matrix<Tmat>  &Mat,
	      parcel::vector<Tvecx> &x0,
	      parcel::vector<Tvecb> &b0
	      ){
    //    return ;
    sycl::queue *que   = Mat.queue();
    const PARCEL_INT n = x0.size();
    que[0].wait();

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
    //    return ;
    // ------------------------------
    // Init Preconditioning
    timer_solver_init_precon.start();

    //	precon_param &precon_0 = *( krylov_param.get_precon_param() );
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
    //    return ;
    timer_solver_init_precon.end();
    //	que[0].wait();
    //	parcel::matrix<Tprecon_mat> preconMat(que);
    //	parcel::matrix_util mat_util;
    //	mat_util.convert<Tmat,Tprecon_mat>(Mat,preconMat);
    // ------------------------------
#if PARCEL_MPI
    // parcel::parallel_vector<Tvec> @_parallel(n,nadd,1,que);parcel::vector<Tvec> @;vec_util.setptr( n, @_parallel, @, 0);
    //    return ;
    timer_solver_init_convert.start();
    // parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);
    parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x; vec_util.setptr( n, x_parallel.vec(0), x, 0);
    parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
    parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);    
    parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
    timer_solver_init_convert.end();
    parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
    parcel_blas.zero_init<Tvec>(r);
    parcel::parallel_vector<Tvec> r_true_parallel(n,nadd,1,que);parcel::vector<Tvec> r_true;vec_util.setptr( n, r_true_parallel.vec(0), r_true, 0);
    parcel_blas.zero_init<Tvec>(r_true);

#if pstep_update
    parcel::parallel_vector<Tvec> p_history_parallel(n,nadd,pstep,que);
    parcel::vector<Tvec> p_history; 
    vec_util.setptr( (n+nadd)*pstep, p_history_parallel.vec_add(),p_history, 0);
    parcel_blas.zero_init<Tvec>(p_history);
    //    parcel_blas.zero_init<Tvec>(p_history_parallel.vec_add(0));
#else
    parcel::parallel_vector<Tvec> p_parallel(n,nadd,1,que);parcel::vector<Tvec> p;vec_util.setptr( n, p_parallel.vec(0), p, 0);
    //    parcel::vector<Tvec> p(n,que);parcel_blas.zero_init<Tvec>(p);
    parcel_blas.zero_init<Tvec>(p);
#endif
    //    parcel::vector<Tvec> q(n,que);parcel_blas.zero_init<Tvec>(q);
    //    parcel::vector<Tvec> s(n,que);parcel_blas.zero_init<Tvec>(s);
    parcel::parallel_vector<Tvec> q_parallel(n,nadd,1,que);parcel::vector<Tvec> q;vec_util.setptr( n, q_parallel.vec(0), q, 0);
    parcel::parallel_vector<Tvec> s_parallel(n,nadd,1,que);parcel::vector<Tvec> s;vec_util.setptr( n, s_parallel.vec(0), s, 0);
    parcel_blas.zero_init<Tvec>(q);
    parcel_blas.zero_init<Tvec>(s);
#else    
    timer_solver_init_convert.start();
    parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
    parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
    timer_solver_init_convert.end();

    parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
    parcel::vector<Tvec> r_true(n,que);parcel_blas.zero_init<Tvec>(r_true);
#if pstep_update
    parcel::vector<Tvec> p_history(n*pstep,que);parcel_blas.zero_init<Tvec>(p_history);
    parcel::vector<Tvec> p;
    parcel::vector<Tvec> p_old;
    vec_util.setptr(n,p_history,p);
#else
    parcel::vector<Tvec> p(n,que);parcel_blas.zero_init<Tvec>(p);
#endif
    parcel::vector<Tvec> q(n,que);parcel_blas.zero_init<Tvec>(q);
    parcel::vector<Tvec> s(n,que);parcel_blas.zero_init<Tvec>(s);
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
	
    //	work.setValue(&one_);parcel_blas.set_ret(work);
    // ------------------------------	
    //     que[0].wait();return ;
    // || b ||
    parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
    parcel::scalar bnorm_host = parcel_blas.get_dot();
    bnorm_host = host_util.sqrt(bnorm_host);
    PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

    //    std::cout<<" bnorm = "<<bnorm_<<std::endl;
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
    
#if pstep_update
    // for(PARCEL_INT i = 0;i < pstep;i++){
    //   parcel_blas.zero_init<Tvec>(p_history_parallel.vec_add(i));
    // }
#else
    parcel_blas.zero_init<Tvec>(p);    
#endif

#else
    parcel_blas.zero_init<Tvec>(p);
#endif    
    // solve Ms=r 
    // preconditioning
    // parcel_blas.copy<Tvec>(r,s);
#if PARCEL_MPI    
      parcel::vector<Tvec> r_vecadd ;vec_util.setptr( (n+nadd), r_parallel.vec_add(), r_vecadd,0); 
      parcel::vector<Tvec> s_vecadd ;vec_util.setptr( (n+nadd), s_parallel.vec_add(), s_vecadd,0); 
      precon_Mat.solver( r_vecadd, s_vecadd);
      //      precon_Mat.solver( r, s);
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
    // std::cout<<"            alpha, beta, rho0 = "<< alpha_<< "," << beta_ << "," << rho0_ <<std::endl;
#if 1
    double bnorm_FP64 = bnorm_;
    double rnorm_FP64 = rnorm_;
    //    std::cout<<"bnorm = "<<std::scientific << std::setprecision(14) << bnorm_ <<std::endl;
    //    std::cout<<"rnorm = "<<std::scientific << std::setprecision(14) << rnorm_ <<std::endl;

    // std::cout<<std::setw(8)<< 0 <<" : residual norm = ";
    // std::cout<<std::scientific << std::setprecision(14) << rnorm_/bnorm_ <<" , true = "<< rnorm_/bnorm_ <<std::endl;
    if( myrank == 0){
      std::cout<<std::setw(8)<< 0 <<" : residual norm = ";
      std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
    }
#endif

    PARCEL_INT pstep_end = (itrmax)%pstep+1; // 反復ループを抜けた後に一括更新するベクトル本数
    vec_util.copy_at<Tdot>(alpha,alpha_history,0,0);
    timer_solver_init.end();
    //    que[0].wait();return ;
    for(PARCEL_INT itr = 0;itr < itrmax;itr++){
      //      x = x + alpha * p 
      timer_solver_axpy.start();
#if pstep_update
      PARCEL_INT pstep_now  =  itr%pstep;
#if 0
      //      parcel_blas.dot< Tvec,Tvec,Tdot >(p,p);
      {
	parcel_blas.dot< Tvec,Tvec,Tdot >(p_history_parallel.vec( pstep_now ),p_history_parallel.vec( pstep_now ));
	parcel::scalar pnorm_now_host = parcel_blas.get_dot();
	PARCEL_INT pstep_next = (itr+1)%pstep;	
	parcel_blas.dot< Tvec,Tvec,Tdot >(p_history_parallel.vec( pstep_next ),p_history_parallel.vec( pstep_next ));
	parcel::scalar pnorm_next_host = parcel_blas.get_dot();
	
	parcel_blas.dot< Tvec,Tvec,Tdot >(q,q);parcel::scalar qnorm_host = parcel_blas.get_dot();
	parcel_blas.dot< Tvec,Tvec,Tdot >(s,s);parcel::scalar snorm_host = parcel_blas.get_dot();
	parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);parcel::scalar rnorm_host = parcel_blas.get_dot();
	if( myrank == 0){
	  std::cout<<std::scientific << std::setprecision(14)<<
	    " pnorm_now = " << (double)pnorm_now_host.get_T() <<
	    ", pnorm _next= " << (double)pnorm_next_host.get_T() <<
	    ", qnorm = " << (double)qnorm_host.get_T() <<
	    ", rnorm = " << (double)rnorm_host.get_T() <<
	    ", snorm = " << (double)snorm_host.get_T() <<
	    std::endl;
	}
      }
#endif
      
#if PARCEL_MPI

#else
      vec_util.setptr(n,p_history,p,pstep_now*n);
#endif      
      //      parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(1,one,one,p,alpha,x);
      if(pstep_now==(pstep-1)){
	// pstep分のベクトルで更新
#if PARCEL_MPI
	parcel::vector<Tvec> p_update ;vec_util.setptr( (n+nadd)*pstep, p_history, p_update ,0); 
	parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep,one,one,
							 // p_history_parallel.vec_add(0,pstep),
							 p_update,
							 alpha_history,
							 x_parallel.vec_add(0));
#else
	parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep,one,one,p_history,alpha_history,x);
#endif	
      }
      
#else
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (alpha,one,p,x);
#endif
      timer_solver_axpy.end();
      
#if 0
    if( myrank == 0){
      std::cout<<std::scientific << std::setprecision(14)<<"rho0  = " << (double)rho0_host.get_T()  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"rho1  = " << (double)rho1_host.get_T()  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"beta  = " << (double)beta_host.get_T()  << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"alpha = " << (double)alpha_host.get_T() << std::endl;  
      std::cout<<std::scientific << std::setprecision(14)<<"gamma = " << (double)gamma_host.get_T() << std::endl;
    }      
#endif


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
#if pstep_update
      PARCEL_INT pstep_next  = (itr+1)%pstep;
      parcel::vector<Tvec> p_now ;vec_util.setptr( n, p_history, p_now , (n+nadd)*pstep_now); 
      parcel::vector<Tvec> p_next;vec_util.setptr( n, p_history, p_next, (n+nadd)*pstep_next);

#if PARCEL_MPI
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one,beta,s,p_now,p_next);
#else      
      vec_util.setptr(n,p_history,p_old,pstep_now *n );
      vec_util.setptr(n,p_history,p    ,pstep_next*n);
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one,beta,s,p_old,p);      
#endif      

#else
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy> (one,beta,s,p,p);
#endif
      timer_solver_axpy.end();

      //      q = Ap 
      timer_solver_SpMV.start();
#if PARCEL_MPI
      
#if pstep_update
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,p_history_parallel.vec_add(pstep_next),q);
#else
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,p_parallel.vec_add(0),q);
#endif
      
#else
      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,p,q);
#endif
      timer_solver_SpMV.end();

      //      gamma = <p,q> 
      timer_solver_dot.start();
#if pstep_update
#if PARCEL_MPI      
      parcel_blas.dot< Tvec,Tvec,Tdot>(p_history_parallel.vec(pstep_next),q);
#else
      parcel_blas.dot< Tvec,Tvec,Tdot>(p,q);
#endif      
#else
      parcel_blas.dot< Tvec,Tvec,Tdot>(p,q);
#endif
      timer_solver_dot.end();

      timer_solver_scala.start();
      gamma_host = parcel_blas.get_dot();
      gamma.setValue(gamma_host);

      //      alpha=rho0/gamma 
      alpha_host = host_util.div(rho0_host,gamma_host);
      alpha.setValue(alpha_host);
      timer_solver_scala.end();

      //      r = r - alpha * q 
      parcel::scalar nega_one_host;nega_one_host.set_Tdd(-1.0);
      parcel::scalar alpha_1_host = host_util.mul(nega_one_host,alpha_host);work.setValue(alpha_1_host); 

      timer_solver_axpy.start();
      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (work,one,q,r);
      timer_solver_axpy.end();

      timer_solver_dot.start();
      parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
      timer_solver_dot.end();

      timer_solver_scala.start();
      rnorm_host = host_util.sqrt( parcel_blas.get_dot() );
      rnorm_ = rnorm_host.get_T();
      timer_solver_scala.end();


#if TRUE_RESIDUAL
      double res_norm      = rnorm_/bnorm_;
      double res_norm_true = rnorm_true/bnorm_;
      if( myrank == 0){
	std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	//	  std::cout<<std::scientific << std::setprecision(10) << rnorm_/bnorm_<<" , true = "<<rnorm_true/bnorm_<<std::endl;
	std::cout<<std::scientific << std::setprecision(14) << res_norm <<" , true = "<< res_norm_true <<std::endl;
      }
#else
      double res_norm      = rnorm_/bnorm_;
      if( myrank == 0){
	std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	std::cout<<std::scientific << std::setprecision(14) << res_norm <<std::endl;
      }
#endif

#if pstep_update
      vec_util.copy_at<Tdot>(alpha,alpha_history,0,pstep_next);
      if((rnorm_/bnorm_)<tol){
	pstep_end = pstep_next+1;
	break;
      }
#else	  
      if((rnorm_/bnorm_)<tol){
	break;
      }
#endif
      //      solve Ms=r 
      // preconditioning
      timer_solver_precon.start();
      //	  parcel_blas.copy<Tvec>(r,s);
#if PARCEL_MPI
      parcel::vector<Tvec> r_vecadd ;vec_util.setptr( (n+nadd), r_parallel.vec_add(), r_vecadd,0); 
      parcel::vector<Tvec> s_vecadd ;vec_util.setptr( (n+nadd), s_parallel.vec_add(), s_vecadd,0); 
      precon_Mat.solver( r_vecadd, s_vecadd);
      //      precon_Mat.solver(r,s);
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
      rho1.setValue(rho1_host);

      //	  beta_ = rho1_/rho0_ ; beta.setValue(&beta_);
      beta_host = host_util.div(rho1_host,rho0_host); 
      beta.setValue(beta_host);

      // std::cout<< "rho0  = "<<rho0_ <<" , "<< rho0_host.get_T()  <<std::endl;
      // std::cout<< "rho1  = "<<rho1_ <<" , "<< rho1_host.get_T()  <<std::endl;
      // std::cout<< "alpha = "<<alpha_<<" , "<< alpha_host.get_T() <<std::endl;
      // std::cout<< "alpha = "<< alpha_host.get_T() <<std::endl;
      // std::cout<< "gamma = "<<gamma_<<" , "<< gamma_host.get_T() <<std::endl;
      // std::cout<< "beta  = "<<beta_ <<" , "<< beta_host.get_T()  <<std::endl;
      // beta.setValue(beta_host);

      rho0_host = rho1_host;
      timer_solver_scala.end();

    }

    // x = x + alpha*p 
#if pstep_update
    // pstep_end分のベクトルで更新
#if PARCEL_MPI
    parcel::vector<Tvec> p_update ;vec_util.setptr( (n+nadd)*pstep_end, p_history, p_update ,0); 
    parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep_end,one,one,
						     p_update,
						     alpha_history,
						     x_parallel.vec_add(0)    );
    
#else
    parcel_blas.gemv<Tdot,Tdot,Tvec,Tdot,Tvec,Taxpy>(pstep_end,one,one,p_history,alpha_history,x);
#endif
    
#if 0
    std::vector< double > alpha_history_(pstep);
    for(PARCEL_INT ii = 0;ii<pstep;ii++){alpha_history_[ii] = -1;}
    alpha_history.getValue(alpha_history_.data());
    std::cout<<"pstep = "<<pstep<<" , pstep_end = "<<pstep_end<<std::endl;
    for(PARCEL_INT ii = 0;ii<pstep;ii++){std::cout<<" alpha ["<<ii<<"] = "<<alpha_history_[ii]<<std::endl;}
    std::cout<< "alpha = "<< alpha_host.get_T() <<std::endl;
#endif

#else
    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (alpha,one,p,x);
#endif

    vec_util.convert<Tvec,Tvecx>(x,x0);

#if 1

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
    //	std::cout<<std::scientific << std::setprecision(10) << rnorm_host_.get_T()/bnorm_<<std::endl;
    double finish_res_norm = rnorm_host_.get_T()/bnorm_;
    if( myrank == 0){
      std::cout<<std::scientific << std::setprecision(10) << finish_res_norm <<std::endl;
    }
    // parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(A,x0,b,r_true);	
    // parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);
    // rnorm_host_ = parcel_blas.get_dot();
    // rnorm_host_ = host_util.sqrt(rnorm_host_);
    // std::cout<<std::setw(8)<<" finish : residual norm return = ";
    // std::cout<<std::scientific << std::setprecision(10) << rnorm_host_.get_T()/bnorm_<<std::endl;
#endif
    timer_solver.end();
#if 0
    std::cout<<"pcg time       = "<<timer_solver.get()         <<" [sec] "<<std::endl;
    std::cout<<"     init time = "<<timer_solver_init.get() <<" [sec] "<<std::endl;
    std::cout<<"     scal time = "<<timer_solver_scala.get()<<" [sec] "<<std::endl;
    std::cout<<"     dot time  = "<<parcel_blas.get_time_dot() <<" [sec] "<<std::endl;
    std::cout<<"     axpy time = "<<parcel_blas.get_time_axpy()<<" [sec] "<<std::endl;
    std::cout<<"     SpMV time = "<<parcel_blas.get_time_SpMV()<<" [sec] "<<std::endl;
#endif
      if( myrank == 0){
    std::cout<<std::endl<<" ----------------- "<<std::endl;
    std::cout<<"pcg time               = "<<timer_solver.get()      <<" [sec] "<<std::endl;
    std::cout<<"     init time         = "<<timer_solver_init.get() <<" [sec] "<<std::endl;
    std::cout<<"          init precon  = "<<timer_solver_init_precon.get()  <<" [sec] "<<std::endl;
    std::cout<<"          init convert = "<<timer_solver_init_convert.get() <<" [sec] "<<std::endl;
    std::cout<<"     dot time          = "<<timer_solver_dot.get()    <<" [sec] "<<std::endl;
    std::cout<<"     axpy time         = "<<timer_solver_axpy.get()   <<" [sec] "<<std::endl;
    std::cout<<"     SpMV time         = "<<timer_solver_SpMV.get()   <<" [sec] "<<std::endl;
    std::cout<<"     precon time       = "<<timer_solver_precon.get() <<" [sec] "<<std::endl;
  }

    return ;

  }
};

  }
}

#undef pstep_update

#endif
