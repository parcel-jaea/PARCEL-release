
#ifndef PARCEL_KRYLOV_GMRES_HPP_
#define PARCEL_KRYLOV_GMRES_HPP_

#include "parcel_qr.hpp"

// double-double演算未対応

namespace parcel{
  namespace krylov{

    //    const PARCEL_INT CGS_ORTHONORMALIZATION = 0;
    //    const PARCEL_INT MGS_ORTHONORMALIZATION = 1;

    // 計算精度を指定
    template<
      class Tspmv   = parcel_double, // SpMV 計算精度
      class Tprecon = parcel_double, // 前処理計算精度
      class Taxpy   = parcel_double, // ベクトル演算計算精度
      class Tdot    = parcel_double  // 縮約演算計算精度
      >
    class pgmres : public krylov_param{

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
      parcel::vector_util vec_util;

      PARCEL_INT gmres_m_ = 4;
      PARCEL_INT orthogonal_algo_ = parcel::qr::QR_MGS;
      //      PARCEL_INT orthogonal_algo_ = parcel::qr::QR_CGS;
      
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;

    public :

      pgmres(){}
      ~pgmres(){}
      void set_qr_algo(PARCEL_INT qrdecomposition_algo){
	orthogonal_algo_ = qrdecomposition_algo;
	return ;
      }
      void set_restart(PARCEL_INT gmres_m){
	gmres_m_ = gmres_m;
	return ;
      }

      template<
	class Tvec,
	class Tcalc
	>
      void GS_ORTHONORMALIZATION(
				 parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
				 PARCEL_INT n,
				 PARCEL_INT m,
#if PARCEL_MPI
				 parcel::vector<Tvec> &v, PARCEL_INT ldv,
				 parcel::vector<Tvec> &w, PARCEL_INT ldw,
#else
				 parcel::vector<Tvec> &v,
				 parcel::vector<Tvec> &w,
#endif				 
				 std::vector< std::vector< parcel::scalar > > &hki
				 ){
#if PARCEL_MPI
	parcel::comm &parcel_comm = parcel_blas.parcel_comm();
	int myrank = parcel_comm.rank();
#endif
	// 古典グラムシュミット直交化
	if( orthogonal_algo_ == parcel::qr::QR_CGS ){

	  sycl::queue *que = v.queue();
	  parcel::vector<Tcalc> one;
	  one.resize(1,que);
	  double one_ = 1.0;one.setValue(&one_);
	  parcel::vector<Tcalc> zero;
	  zero.resize(1,que);
	  double zero_ = 0.0;zero.setValue(&zero_);

	  // hki = <wk,Vi> 
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if PARCEL_MPI
	  parcel::vector<Tcalc> wvi_tmp; wvi_tmp.resize(m+1,que);
	  parcel::vector<Tcalc> wvi; wvi.resize(m+1,que);
	  que[0].wait();
#if 0
	  // dot 計算で代用
	  {
	    std::vector< parcel::scalar > wvi_host(m+1,0.0);
	    for(PARCEL_INT i = 0;i<=m;i++){
	      parcel::vector<Tvec> vi;vec_util.setptr(n,v,vi,ldv*i);
	      parcel::vector<Tvec> wi;vec_util.setptr(n,w,wi,0);
	      que[0].wait();
	      parcel_blas.dot< Tvec,Tvec,Tcalc >(wi,vi);
	      que[0].wait();
	      wvi_host[i] = parcel_blas.get_dot();
	    }
	    wvi.setValue(wvi_host);
	  }
#else
	  que[0].wait();
	  parcel_blas.gemm<Tcalc,Tcalc,Tvec,Tvec,Tcalc,Tcalc>(
							      1,
							      m+1,
							      n,
							      one,zero,
							      w  ,1,
							      v  ,ldv,
							      wvi_tmp,1);
	  que[0].wait();
	  parcel_comm.allreduce(wvi_tmp,wvi,m+1,MPI_SUM);
#endif
	  
#else
	  parcel::vector<Tcalc> wvi; wvi.resize(m+1,que);
	  que[0].wait();
	  parcel_blas.gemm<Tcalc,Tcalc,Tvec,Tvec,Tcalc,Tcalc>(
							      1,
							      m+1,
							      n,
							      one,zero,
							      w  ,1,
							      v  ,n,
							      wvi,1);
	  que[0].wait();
#endif	  
	  std::vector< parcel::scalar > wvi_host(m+1,0.0);
	  wvi.getValue(wvi_host);
	  que[0].wait();
	  for(PARCEL_INT i = 0;i<=m;i++){
	    //	    std::cout<<i<<" : hki Host = "<<wvi_host[i]<<std::endl;
	    hki[i][m]   =  wvi_host[i];
	    wvi_host[i] = -wvi_host[i];
	    // if( myrank == 0){
	    //   std::cout<<i<<" : hki Host = "<<wvi_host[i].get_T()<<std::endl;
	    // }
	  }

	  // wi = wi - hki*Qk  
	  que[0].wait();
	  wvi.setValue(wvi_host);
	  que[0].wait();
#if PARCEL_MPI
	  parcel::vector<Tvec> v_add; vec_util.setptr(ldv*(m+1), v, v_add, 0);
	  parcel::vector<Tvec> w_add; vec_util.setptr(ldw      , w, w_add, 0);
	  parcel_blas.gemv<Tcalc,Tcalc,Tvec,Tcalc,Tvec,Taxpy>(m+1,one,one,v_add,wvi,w_add);
#else
	  parcel_blas.gemv<Tcalc,Tcalc,Tvec,Tcalc,Tvec,Taxpy>(m+1,one,one,v,wvi,w);
#endif
	  
	  return ;

	}

	// 修正グラムシュミット直交化
	if( orthogonal_algo_ == parcel::qr::QR_MGS ){
	  sycl::queue *que = v.queue();
	  parcel::vector<Tcalc> one;
	  one.resize(1,que);
	  double one_ = 1.0;one.setValue(&one_);
	  parcel::vector<Tvec> vi0;
	  parcel::vector<Tcalc> wvi; wvi.resize(1,que);
	  for(PARCEL_INT i = 0;i<=m;i++){
	    // hki = <wk,Vi> 
#if PARCEL_MPI
	    vec_util.setptr(n, v, vi0, ldv*i);
#else
	    vec_util.setptr(n, v, vi0, n*i);
#endif
	    que[0].wait();
	    parcel_blas.dot< Tvec,Tvec,Tcalc >(w,vi0);
	    que[0].wait();

	    parcel::scalar wvi_host = parcel_blas.get_dot();
	    que[0].wait();
	    //	    std::cout<<i<<"     MGS : "<<wvi_host.get_T()<<","<<parcel_blas.get_T()<<std::endl;
	    hki[i][m] = wvi_host;
	    wvi_host  = -wvi_host;

	    // wi = wi - hki*Qk  
	    //	    wvi_host.set_T(-wvi_);
	    que[0].wait();
	    wvi.setValue(wvi_host); 
	    que[0].wait();
	    parcel_blas.axpy<Tcalc,Tcalc,Tvec,Tvec,Taxpy>(wvi,one,vi0,w);
	    que[0].wait();
	  }
	  return ;
	}
	return ;

      }
      
  void GivensRotation(
		      PARCEL_INT m,
		      std::vector< 
		      std::vector< parcel::scalar > > &hki,
		      std::vector<parcel::scalar> &e,
		      std::vector<parcel::scalar> &co,
		      std::vector<parcel::scalar> &si
		      ){

    // ----- 
    // m+1行のヘッセンベルグ行列に対するギブンス回転
    // Givens rotation
    parcel::scalar one  = 1.0;
    parcel::scalar tmp1 = 0.0;
    parcel::scalar tmp2 = 0.0;
    //    std::cout<<"Givens rot : m = "<<m<<std::endl;
    //    for(PARCEL_INT k=0;k<m;k++){
    for(PARCEL_INT k=0;k<m;k++){
      tmp1 = hki[k][m]*co[k] - hki[k+1][m]*si[k];
      tmp2 = hki[k][m]*si[k] + hki[k+1][m]*co[k];
      hki[k][m]   = tmp1;
      hki[k+1][m] = tmp2;
    }

    // -------
    tmp1  = hki[m+1][m]/hki[m][m];
    //    co[m] = 1.0/sqrt(1.0+tmp1*tmp1 );
    co[m] = one / host_util.sqrt(one+tmp1*tmp1 );
    si[m] = -co[m]*tmp1;

    hki[m][m]   = co[m]*hki[m][m] - si[m]*hki[m+1][m];
    hki[m+1][m] = 0.0;

    tmp1 = co[m]*e[m]-si[m]*e[m+1];
    tmp2 = si[m]*e[m]+co[m]*e[m+1];
    e[m]   = tmp1;
    e[m+1] = tmp2;

    return ;

  }
  
      // 前処理,IR等の設定をclassで定義する
      template<
	class Tmat,  // 係数行列データ精度
	class Tvecx, // 入力ベクトルデータ精度
	class Tvecb, // 右辺ベクトルデータ精度
	class Tvec , // クリロフ部分空間法内部データ精度	
	class Tprecon_mat, // 前処理行列データ精度
	class Tprecon_vec, // 前処ベクトルデータ精度
	class TGS = parcel_double, // グラムシュミットの直交化精度
	class THessenberg = parcel_double // ヘッセンベルグ行列の精度
	>
      void solver(
		  parcel::matrix<Tmat>  &Mat,
		  parcel::vector<Tvecx> &x0,
		  parcel::vector<Tvecb> &b0
		  ){

	sycl::queue *que   = Mat.queue();
	const PARCEL_INT n = x0.size();
	//	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
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

	// ------------------------------
	// Init Preconditioning
	timer_solver_init_precon.start();

	//	precon_param &precon_0 = *( krylov_param.get_precon_param() );
	const double tol = get_tolerance();
	const PARCEL_INT itrmax = get_max_iteration();
	precon_param &precon = *(get_precon_param());
	parcel::precon_matrix<
	  Tmat,
	  Tvec,
	  Tprecon_mat,
	  Tprecon_vec,
	  Tprecon
	  > precon_Mat(Mat,precon);

	timer_solver_init_precon.end();

	timer_solver_init_convert.start();
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);    
	parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);    
	parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
#else	
	parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);
#endif
	timer_solver_init_convert.end();

	// ------
	// pgmres 
	// ------
	// do itr=1,....
	//   r=b-Ax
	//   v_1=r/||r||
	//   do i=1,m
	//     solve Mpv=v_i
	//     w=Apv
	//     グラムシュミット直交化
	//     v_i+1 = w/||w||
	//     ギブンス回転
	//     if(||r||<rmax)exit
	//   enddo
	//   解ベクトル更新
	//   if(||r||<rmax)exit
	// enddo

#if PARCEL_MPI
	parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel::parallel_vector<Tvec> r_true_parallel(n,nadd,1,que);parcel::vector<Tvec> r_true;vec_util.setptr( n, r_true_parallel.vec(0), r_true, 0);
	parcel_blas.zero_init<Tvec>(r);
	parcel_blas.zero_init<Tvec>(r_true);

	//	parcel::parallel_vector<Tvec> v_parallel(n*(gmres_m_+1),nadd,1,que);parcel::vector<Tvec> v;vec_util.setptr( n, v_parallel, v, 0);
	parcel::parallel_vector<Tvec> v_parallel(n,nadd,gmres_m_+1,que);
	parcel::vector<Tvec> v; 
	vec_util.setptr( (n+nadd)*(gmres_m_+1), v_parallel.vec_add(),v, 0);
	parcel_blas.zero_init<Tvec>(v);

	parcel::parallel_vector<Tvec> pv_parallel(n,nadd,1,que);parcel::vector<Tvec> pv;vec_util.setptr( n, pv_parallel.vec(0), pv, 0);
	parcel::parallel_vector<Tvec> w_parallel(n,nadd,1,que);parcel::vector<Tvec> w;vec_util.setptr( n, w_parallel.vec(0), w, 0);
	parcel::parallel_vector<Tvec> v0_parallel(n,nadd,1,que);parcel::vector<Tvec> v0;vec_util.setptr( n, v0_parallel.vec(0), v0, 0);
	parcel::parallel_vector<Tvec> x_update_parallel(n,nadd,1,que);parcel::vector<Tvec> x_update;vec_util.setptr( n, x_update_parallel.vec(0), x_update, 0);
 
	parcel_blas.zero_init<Tvec>(v);
	parcel_blas.zero_init<Tvec>(pv);
	parcel_blas.zero_init<Tvec>(w);
	parcel_blas.zero_init<Tvec>(v0);
	parcel_blas.zero_init<Tvec>(x_update);
#else
	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
	parcel::vector<Tvec> r_true(n,que);parcel_blas.zero_init<Tvec>(r_true);

	parcel::vector<Tvec> v(n*(gmres_m_+1),que);parcel_blas.zero_init<Tvec>(v);
	parcel::vector<Tvec> pv(n,que);parcel_blas.zero_init<Tvec>(pv);
	parcel::vector<Tvec> w(n,que);parcel_blas.zero_init<Tvec>(w);
	parcel::vector<Tvec> v0(n,que);parcel_blas.zero_init<Tvec>(v0);
	parcel::vector<Tvec> x_update(n,que);parcel_blas.zero_init<Tvec>(v0);
#endif
	parcel::vector<Tvec> y( (gmres_m_+1) ,que);parcel_blas.zero_init<Tvec>(y);
	parcel_blas.zero_init<Tvec>(y);
	
	std::vector< parcel::scalar > e(gmres_m_+1 ,0.0);
	std::vector< parcel::scalar > co(gmres_m_+1 ,0.0);
	std::vector< parcel::scalar > si(gmres_m_+1 ,0.0);
	std::vector< std::vector< parcel::scalar > > hki( gmres_m_+1  , std::vector< parcel::scalar > (gmres_m_)   );
	std::vector< std::vector< parcel::scalar > > Rtri( gmres_m_+1 , std::vector< parcel::scalar > (gmres_m_+1) );

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
	
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

	//    for(PARCEL_INT itr = 0;itr < itrmax/gmres_m_;itr++){
    
	// std::cout<<" itrmax = "<<itrmax<<std::endl;
	// std::cout<<" itrmax/gmres_m_ = "<<itrmax/gmres_m_<<std::endl;
	// for(PARCEL_INT itr = 0;itr < 4;itr++){
	for(PARCEL_INT itr = 0;itr < itrmax/gmres_m_;itr++){
	//	for(PARCEL_INT itr = 0;itr < 1;itr++){

	  for(PARCEL_INT i = 0;i<e.size() ;i++){e[i]  = 0.0;}
	  for(PARCEL_INT i = 0;i<co.size();i++){co[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<si.size();i++){si[i] = 0.0;}

	  for(int i = 0;i<hki.size();i++){
	    for(int j = 0;j<hki[i].size();j++){
	      hki[i][j] = 0.0;
	    }
	  }
	  for(int i = 0;i<Rtri.size();i++){
	    for(int j = 0;j<Rtri[i].size();j++){
	      Rtri[i][j] = 0.0;
	    }
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
	  e[0] = -rnorm_;
#if 1
	  double bnorm_FP64 = bnorm_;
	  double rnorm_FP64 = rnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<< itr*gmres_m_ <<" : residual norm = ";
	    std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	  }
	  if((rnorm_/bnorm_)<tol)break;
#endif

	  // v_1 = -r/||r||
	  parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(-1.0/rnorm_);
	  parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	  rnorm_inv.setValue(rnorm_inv_host); 

	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (rnorm_inv,zero,r,v);
	  parcel::vector<Tvec> vi0;
	  parcel::vector<Tvec> vi1;

	  for(PARCEL_INT i = 0;i < gmres_m_;i++){
	    // solve Mpv=v
#if PARCEL_MPI
	    vec_util.setptr( (n+nadd), v, vi0, (n+nadd)*i);
	    parcel::vector<Tvec> pv_vecadd ;vec_util.setptr( (n+nadd) , pv_parallel.vec_add(0), pv_vecadd,0); 	    
	    precon_Mat.solver(vi0,pv_vecadd) ;
#else
	    vec_util.setptr(n, v, vi0, n*i);
	    precon_Mat.solver(vi0,pv) ;
#endif

	    // w=Apv
#if PARCEL_MPI
	    parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pv_parallel.vec_add(0),w);
#else	    
	    parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pv,w);
#endif
	    
	    // グラムシュミット直交化
#if PARCEL_MPI
	    GS_ORTHONORMALIZATION<Tvec,TGS>(parcel_blas,n,i,v,n+nadd,w,n+nadd,hki);
#else
	    GS_ORTHONORMALIZATION<Tvec,TGS>(parcel_blas,n,i,v,w,hki);
#endif	    
	que[0].wait();
	// v_i+1 = w/||w||
#if PARCEL_MPI
	vec_util.setptr(n, v, vi1, (n+nadd)*(i+1) );
#else	
	vec_util.setptr(n, v, vi1, n*(i+1) );
#endif	
	parcel_blas.dot< Tvec,Tvec,Tdot >(w,w);
	parcel::scalar wnorm_host = parcel_blas.get_dot();
	wnorm_host = host_util.sqrt(wnorm_host);
	// 
	hki[i+1][i] = wnorm_host.get_T();

	parcel::scalar wnorm_inv_host;
	wnorm_inv_host.set_Tdd(1.0/wnorm_host.get_T());
	parcel::vector<Tdot> wnorm_inv; wnorm_inv.resize(1,que);
	wnorm_inv.setValue(wnorm_inv_host); 
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (wnorm_inv,zero,w,vi1);

#if 0
	    que[0].wait();
	    std::cout<<std::endl<<" ----------------- "<<std::endl;
	    std::cout<<" GR before : Hessenberg Matrix "<<std::endl;
	    for(PARCEL_INT j=0;j<gmres_m_;j++){
	      for(PARCEL_INT i=0;i<gmres_m_;i++){
		std::cout<<" i,j = "<<i<<" , "<<j<<" : "<<hki[i][j]<<std::endl;;
	      }
	    }
#endif

	    //Givens rotation
	    // GivensRotation<Tvec>(i,hki,e,co,si);
	    GivensRotation(i,hki,e,co,si);

#if 0
	    que[0].wait();
	    std::cout<<std::endl<<" ----------------- "<<std::endl;
	    std::cout<<" GR After : Hessenberg Matrix "<<std::endl;
	    for(PARCEL_INT j=0;j<gmres_m_;j++){
	      for(PARCEL_INT i=0;i<gmres_m_;i++){
		std::cout<<" i,j = "<<i<<" , "<<j<<" : "<<hki[i][j].get_T()<<std::endl;;
	      }
	    }
#endif

	  }


#if 0
	  que[0].wait();
	  if( myrank == 0){
	    std::cout<<std::endl<<" ----------------- "<<std::endl;
	    std::cout<<" Hessenberg Matrix "<<std::endl;
	    for(PARCEL_INT j=0;j<gmres_m_;j++){
	      for(PARCEL_INT i=0;i<gmres_m_;i++){
		std::cout<<" i,j = "<<i<<" , "<<j<<" : "<< hki[i][j].get_T()<<std::endl;;
	      }
	    }
	  }
#endif

#if 1
	  // ------------------------------------------------------------------
	  // 解ベクトルの更新(リスタート周期になるまでループを抜けない実装)
      std::vector< parcel::scalar > Hinv( gmres_m_*gmres_m_ );
      for(PARCEL_INT i=0;i<gmres_m_;i++){
	for(PARCEL_INT j=0;j<gmres_m_;j++){
	  //	  Hinv[j+i*gmres_m_]=hki[i][j];
	  Hinv[j+i*gmres_m_]=hki[j][i];
	}
      }
      std::string uplo("U");
      std::string diag("N");

      // trtriがホストで計算するstd::vector型なのでクラスを整理する
      //  * y = H^-1 eとして後退代入で解けばtrtriで逆行列を求める必要はない
#if 1
      parcel_blas_host.trtri(uplo,diag,Hinv,gmres_m_,gmres_m_);
#else
      mixed_precison::trtri<
	parcel::scalar,parcel::scalar
	>(
	  uplo,
	  diag,
	  Hinv.data(),
	  gmres_m_,gmres_m_
	  );
#endif      
#if 0
      que[0].wait();
      std::cout<<std::endl<<" ----------------- "<<std::endl;
      for(PARCEL_INT i=0;i<gmres_m_+1;i++){
      	std::cout<<"si : i = "<<i<<" , "<<si[i]<<std::endl;;
      }
      for(PARCEL_INT i=0;i<gmres_m_+1;i++){
      	std::cout<<"co : i = "<<i<<" , "<<co[i]<<std::endl;;
      }
      for(PARCEL_INT i=0;i<gmres_m_+1;i++){
      	std::cout<<"e  : i = "<<i<<" , "<<e[i]<<std::endl;;
      }
      std::cout<<" inverse Hessenberg Matrix "<<std::endl;
      for(PARCEL_INT j=0;j<gmres_m_;j++){
	for(PARCEL_INT i=0;i<gmres_m_;i++){
	  std::cout<<" i,j = "<<i<<" , "<<j<<" : "<<Hinv[i+j*gmres_m_]<<std::endl;;
	}
      }
#endif
      
      std::vector< parcel::scalar > y_host( (gmres_m_+1),0.0);
      for(PARCEL_INT i=0;i<gmres_m_;i++){
	y_host[i]=0.0;
	for(PARCEL_INT j=0;j<gmres_m_;j++){
	  y_host[i] = y_host[i] + Hinv[i+j*gmres_m_]*e[j];
	}
	// std::cout<<" y = ["<<i<<"] :"<<y_host[i]<<std::endl;
      }
      parcel::vector<THessenberg> y_;y_.resize(gmres_m_,que);y_.setValue(y_host.data());
      vec_util.convert<THessenberg,Tvec>(y_,y);

      // gemv計算
      // v0 = v[][]*y
#if PARCEL_MPI
      // 袖部分についても更新する実装
      //      parcel::vector<Tvec> v_update ;vec_util.setptr( (n+nadd)*(gmres_m_), v, v_update ,0);
      parcel::vector<Tvec> v_update ;vec_util.setptr( (n+nadd)*(gmres_m_), v, v_update ,0); 
      parcel_blas.gemv<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy>(gmres_m_,one,zero,
						       v_update,
						       y,
						       v0_parallel.vec_add(0));
						       // v0);
#else
      parcel_blas.gemv<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy>(gmres_m_,one,zero,v,y,v0);
#endif
      
#if 0
	  // Debug
	  parcel_blas.dot< Tvec,Tvec,Tdot >(v0,v0);
	  parcel::scalar v0_host = parcel_blas.get_dot();
	  que[0].wait();
	  std::cout<<std::scientific << std::setprecision(14)<<" v0 = " << (double)v0_host.get_T() << std::endl;
#endif

	  // 前処理計算
	  // Mx0 = v0
#if PARCEL_MPI
	  parcel::vector<Tvec> v0_vecadd ;vec_util.setptr( (n+nadd) , v0_parallel.vec_add(0), v0_vecadd,0); 	    	  
	  parcel::vector<Tvec> x_update_vecadd ;vec_util.setptr( (n+nadd) , x_update_parallel.vec_add(0), x_update_vecadd,0); 	    	  
	  precon_Mat.solver(
			    v0_vecadd,
			    x_update_vecadd);
#else
	  precon_Mat.solver(v0,x_update);	  
#endif

	  // 解ベクトルの更新 : x = x + x0
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,one,x_update,x);
	  // ------------------------------------------------------------------
#endif

	} // restart loop

	vec_util.convert<Tvec,Tvecx>(x,x0);
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);	
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
	  std::cout<<std::endl<<" ----------------- "<<std::endl;
	  std::cout<<"pgmres time               = "<<timer_solver.get()      <<" [sec] "<<std::endl;
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

