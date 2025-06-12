#ifndef PARCEL_KRYLOV_BICGSTAB_HPP_
#define PARCEL_KRYLOV_BICGSTAB_HPP_
namespace parcel{
  namespace krylov{

    template<
      class Tdata   = parcel_double,
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

    public :
      pbicgstab(){}
      ~pbicgstab(){}

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


	sycl::queue *que   = Mat.queue();
	const PARCEL_INT n = x0.size();
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
	//	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
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
	parcel::parallel_vector<Tvec> r_true_parallel(n,nadd,1,que);parcel::vector<Tvec> r_true;vec_util.setptr( n, r_true_parallel.vec(0), r_true, 0);
	parcel_blas.zero_init<Tvec>(r_true);
	
	//    parcel::parallel_vector<Tvec> @_parallel(n,nadd,1,que);parcel::vector<Tvec> @;vec_util.setptr( n, @_parallel.vec(0), @, 0);
	//    parcel_blas.zero_init<Tvec>(@);

	parcel::parallel_vector<Tvec> r0_parallel(n,nadd,1,que);parcel::vector<Tvec> r0;vec_util.setptr( n, r0_parallel.vec(0), r0, 0);
	parcel::parallel_vector<Tvec> p_parallel(n,nadd,1,que);parcel::vector<Tvec> p;vec_util.setptr( n, p_parallel.vec(0), p, 0);
	parcel::parallel_vector<Tvec> q_parallel(n,nadd,1,que);parcel::vector<Tvec> q;vec_util.setptr( n, q_parallel.vec(0), q, 0);
	parcel::parallel_vector<Tvec> pe_parallel(n,nadd,1,que);parcel::vector<Tvec> pe;vec_util.setptr( n, pe_parallel.vec(0), pe, 0);
	parcel::parallel_vector<Tvec> v_parallel(n,nadd,1,que);parcel::vector<Tvec> v;vec_util.setptr( n, v_parallel.vec(0), v, 0);
	parcel::parallel_vector<Tvec> pp_parallel(n,nadd,1,que);parcel::vector<Tvec> pp;vec_util.setptr( n, pp_parallel.vec(0), pp, 0);
	parcel::parallel_vector<Tvec> e_parallel(n,nadd,1,que);parcel::vector<Tvec> e;vec_util.setptr( n, e_parallel.vec(0), e, 0);

	parcel_blas.zero_init<Tvec>(r0);
	parcel_blas.zero_init<Tvec>(p);
	parcel_blas.zero_init<Tvec>(q);
	parcel_blas.zero_init<Tvec>(pe);
	parcel_blas.zero_init<Tvec>(v);
	parcel_blas.zero_init<Tvec>(pp);
	parcel_blas.zero_init<Tvec>(e);
  
#else
	parcel::vector<Tvec> x(n,que);parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que);parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);

	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
	parcel::vector<Tvec> r_true(n,que);parcel_blas.zero_init<Tvec>(r_true);

	parcel::vector<Tvec> r0(n,que);parcel_blas.zero_init<Tvec>(r0);
	parcel::vector<Tvec> p (n,que);parcel_blas.zero_init<Tvec>(p);
	parcel::vector<Tvec> q (n,que);parcel_blas.zero_init<Tvec>(q);
	parcel::vector<Tvec> pe(n,que);parcel_blas.zero_init<Tvec>(pe);
	parcel::vector<Tvec> v (n,que);parcel_blas.zero_init<Tvec>(v);
	parcel::vector<Tvec> pp(n,que);parcel_blas.zero_init<Tvec>(pp);
	parcel::vector<Tvec> e (n,que);parcel_blas.zero_init<Tvec>(e);
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
	parcel::vector<parcel_double> c3(1,que);
	parcel::vector<parcel_double> alpha(1,que);
	parcel::vector<parcel_double> beta(1,que);

	// ! ------
	// ! pbicgstab start
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

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

	// ! r0=r
	parcel_blas.copy<Tvec>(r,r0);

	// ! p=r
	parcel_blas.copy<Tvec>(r,p);

	// ! c1=<r0,r0>
	parcel_blas.dot< Tvec,Tvec,Tdot >(r0,r0);
	parcel::scalar c1_host = parcel_blas.get_dot();
	c1.setValue(c1_host);
	PARCEL_MAX_FLOTING_TYPE c1_ = parcel_blas.get_T();
	
	timer_solver_init.end();
	for(PARCEL_INT itr = 0;itr < itrmax;itr++){
	  // preconditioning
	  // !   solve Mpp=p
	  precon_Mat.solver(p,pp);

	  // !   q=App
	  parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>
	    (one,zero,Mat,pp,q);

	  // !   c2=<r0,q>
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r0,q);
	  parcel::scalar c2_host = parcel_blas.get_dot();
	  c2.setValue(c2_host);

	  // !   alpha = c1/c2
	  parcel::scalar alpha_host = host_util.div(c1_host,c2_host); 
	  alpha.setValue(alpha_host);

	  // !   e=r-alpha*q
	  parcel::scalar nega_one_host;nega_one_host.set_Tdd(-1.0);
	  parcel::scalar alpha_1_host = host_util.mul(nega_one_host,alpha_host);work.setValue(alpha_1_host); 
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(one,zero,r,e);
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,q,e);

	  // preconditioning
	  // !   solve Mpe=e
	  //	  parcel_blas.copy<Tvec>(e,pe);
	  precon_Mat.solver(e,pe);

	  // !   v=Ape
	  parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pe,v);

	  // !   c3=<e,v>/<v,v>
	  parcel_blas.dot< Tvec,Tvec,Tdot >(e,v);
	  parcel::scalar ev_host = parcel_blas.get_dot();

	  parcel_blas.dot< Tvec,Tvec,Tdot >(v,v);
	  parcel::scalar vv_host = parcel_blas.get_dot();
	  parcel::scalar c3_host = host_util.div(ev_host,vv_host);
	  c3.setValue(c3_host);

	  // !   x=x+alpha*pp+c3*pe
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(alpha,one,pp,x);
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(c3,one,pe,x);

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
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(one,zero,e,r);

	  //	  PARCEL_MAX_FLOTING_TYPE c3_1 = -c3_; work.setValue(&c3_1);
	  parcel::scalar c3_1_host = host_util.mul(nega_one_host,c3_host);work.setValue(c3_1_host); 
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,v,r);

	  // !   c1=<r0,r>
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r0,r);
	  //	  c1_ = parcel_blas.get_T();
	  c1_host = parcel_blas.get_dot();c1.setValue(c1_host);
	  c1.setValue(c1_host);

	  parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	  //  rnorm_ = std::sqrt( parcel_blas.get_T() );
	  rnorm_host = parcel_blas.get_dot();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  rnorm_ = rnorm_host.get_T();
#if 1
	  double res_norm      = rnorm_/bnorm_;
	  //	  double res_norm_true = rnorm_true/bnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<< itr+1 <<" : residual norm = ";
	    std::cout<<std::scientific << std::setprecision(10) << res_norm <<std::endl;
	  }
#endif
	  // !   if(||r||<rmax) exit
	  if((rnorm_/bnorm_)<tol)break;

	  // !   beta=c1/(c3*c2)
	  //	  PARCEL_MAX_FLOTING_TYPE beta_ = c1_ / (c3_*c2_);beta.setValue(&beta_);
	  parcel::scalar beta_host = host_util.div(c1_host, host_util.mul(c3_host,c2_host) );beta.setValue(beta_host);

	  // !   p=r+beta*(p-c3*q)
	  c3_1_host = host_util.mul(nega_one_host,c3_host);work.setValue(c3_1_host); 

	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(work,one,q,p);
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy>(one,beta,r,p);

	  // ! enddo
	  // ! ------
	  // std::cout<< "c1     = "<< c1_host.get_T()  <<std::endl;
	  // std::cout<< "c2     = "<< c2_host.get_T()  <<std::endl;	
	  // std::cout<< "c3     = "<< c3_host.get_T()  <<std::endl;
	  // std::cout<< "alpha  = "<< alpha_host.get_T()  <<std::endl;
	  // std::cout<< "beta   = "<< beta_host.get_T()  <<std::endl;
	  
	}
	vec_util.convert<Tvec,Tvecx>(x,x0);
	
    timer_solver.end();
    if( myrank == 0){
      std::cout<<std::endl<<" ----------------- "<<std::endl;
      std::cout<<"pbicg-stab time               = "<<timer_solver.get()      <<" [sec] "<<std::endl;
#if 0
      std::cout<<"     init time         = "<<timer_solver_init.get() <<" [sec] "<<std::endl;
      std::cout<<"          init precon  = "<<timer_solver_init_precon.get()  <<" [sec] "<<std::endl;
      std::cout<<"          init convert = "<<timer_solver_init_convert.get() <<" [sec] "<<std::endl;
      std::cout<<"     dot time          = "<<timer_solver_dot.get()    <<" [sec] "<<std::endl;
      std::cout<<"     axpy time         = "<<timer_solver_axpy.get()   <<" [sec] "<<std::endl;
      std::cout<<"     SpMV time         = "<<timer_solver_SpMV.get()   <<" [sec] "<<std::endl;
      std::cout<<"     precon time       = "<<timer_solver_precon.get() <<" [sec] "<<std::endl;
#endif      
    }
    return ;
      }
    };
  }
}
#endif
