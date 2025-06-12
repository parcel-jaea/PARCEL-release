#ifndef PARCEL_PRECON_DDJACOBI_DIA_
#define PARCEL_PRECON_DDJACOBI_DIA_

#if 1
#include "parcel_precon_ddjacobi_dia_py.hpp"
#else

template<
  class T_vec,class T_precon_mat,class T_work_vec,class T_calc
  >
void ddjacobi_method_dia_main(
			    sycl::queue *q_ ,
			    PARCEL_INT N,
			    PARCEL_INT NND,
			    PARCEL_INT iterMax,
			    T_vec *x, // out  
			    T_vec *b, // in
			    T_precon_mat *A, // (D^-1)(L+U)
			    PARCEL_INT *offset,
			    PARCEL_INT max_up_offset,
			    PARCEL_INT max_down_offset,
			    T_precon_mat *Dinv, // (D^-1)
			    double *scaling_Inv,
			    T_work_vec *precon_x1_, // 
			    T_work_vec *precon_x2_, // 
			    T_work_vec *precon_b){

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 8192;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  
  const bool SCALING_MATRIX = SCALING_MATRIX_;
  const bool SCALING_VECTOR = SCALING_VECTOR_;

  q_[0].wait();

  // T_work_vec *x_now  = precon_x1_;
  // T_work_vec *x_next = precon_x2_;
  // solve Ax = b
  // x(i+1) = (D^-1) { b - {L+U}x(i) } 

  //      const double bnorm = 1.0;
  double bnorm_ = 1.0;
  if( SCALING_VECTOR == true ){ 
    setQueue(q_);
    if( SCALING_MATRIX == true ){ 
      bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N);
    }else{
      bnorm_ = max<T_vec,double>(N,b,N);
    }
    q_[0].wait();
  }
  const double bnorm_hi_ = bnorm_;
  const double bnorm_lo_ = 0.0;

  // double ddmul_var1; ddmul_calc1

#if _BLOCKING_LOOP_
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      //	  double bnorm = bnorm_ptr[0];
      ///	  double bnorm = 1.0;
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
#else
	  const size_t MAX = N;
	  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
	      const double Dinv_hi_ = Dinv[i];
	      const double Dinv_lo_ = 0.0;
	      double b_hi_ = b[i];
	      double b_lo_ = 0.0;
	      // const double b_    = b[i] / bnorm;
	      double dddiv_var1; 
	      dddiv_calc1(b_hi_,b_lo_,bnorm_hi_,bnorm_lo_,b_hi_,b_lo_);

	      if( SCALING_MATRIX == true ){ 
		const double val_hi_ = scaling_Inv[i];
		const double val_lo_ = 0.0;
		double precon_b_hi_ = 0.0;
		double precon_b_lo_ = 0.0;
		// precon_b[i] = Dinv_*b_*val;
		double ddmul_var1; 
		ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
		ddmul_calc1(precon_b_hi_,precon_b_lo_,val_hi_,val_lo_,precon_b_hi_,precon_b_lo_);
		precon_b[i] = precon_b_hi_;
	      }else{
		// precon_b[i] = Dinv_*b_;
		double precon_b_hi_ = 0.0;
		double precon_b_lo_ = 0.0;
		double ddmul_var1; 
		ddmul_calc1(b_hi_,b_lo_,Dinv_hi_,Dinv_lo_,precon_b_hi_,precon_b_lo_);
		precon_b[i] = precon_b_hi_;
	      }

#if _BLOCKING_LOOP_
	    }
	    }
	});
#else
    });
#endif
  q_[0].wait();

  // x(0) = (D^-1)b
  // x(1) = (D^-1)b - (D^-1){L+U}(D^-1)b
  //        ( I - (D^-1){L+U} )(D^-1)b  

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
#if _BLOCKING_LOOP_
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i_=iii;i_<imax;i_++){
#else
	  //	      const size_t MAX = N;
	  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
#endif
	      PARCEL_INT i = i_;
	      // T_calc x_i = precon_b[i];
	      double x_i_hi_ = precon_b[i];
	      double x_i_lo_ = 0.0;
	      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
		for(PARCEL_INT j = 0; j < NND; j++){
		  PARCEL_INT i0 = i + offset[j];
		  // T_calc A_ = A[i+j*N];
		  const double A_hi_ = A[i+j*N];
		  const double A_lo_ = 0.0;
		  if( (-1<i0) && (i0<N) ){
		    // T_calc x_ = precon_b[i0];
		    double x_hi_ = precon_b[i0];
		    double x_lo_ = 0.0;
		    //x_i = x_i - A_*x_;
		    double Ax_hi_ = 0.0;
		    double Ax_lo_ = 0.0;
		    double ddmul_var1; 
		    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);

		    double ddsub_var1; 
		    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
		  }
		}
	      }else{
		for(PARCEL_INT j = 0; j < NND; j++){
		  PARCEL_INT i0 = i + offset[j];
		  // T_calc A_ = A[i+j*N];
		  // T_calc x_ = precon_b[i0];

		  const double A_hi_ = A[i+j*N];
		  const double A_lo_ = 0.0;
		  double x_hi_ = precon_b[i0];
		  double x_lo_ = 0.0;

		  double Ax_hi_ = 0.0;
		  double Ax_lo_ = 0.0;
		  
		  // x_i = x_i - A_*x_;
		  double ddmul_var1; 
		  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);

		  double ddsub_var1; 
		  ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);

		}
	      }

	      if( iterMax==1 ){
		x[i] = x_i_hi_;
	      }else{
		x_next[i] = x_i_hi_;
	      }

#if _BLOCKING_LOOP_
	    }
	    }
	});
#else
    });
#endif
  q_[0].wait();

  // x(i+1) = (D^-1) { b - {L+U}x(i) } 
  for(PARCEL_INT iter = 1;iter<iterMax;iter++){

    swap_ptr<T_work_vec>(&x_now,&x_next);
#if _BLOCKING_LOOP_
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i_=iii;i_<imax;i_++){
#else
	    //	const size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
#endif
		PARCEL_INT i = i_;
		double x_i_hi_ = precon_b[i];
		double x_i_lo_ = 0.0;
		if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
		  for(PARCEL_INT j = 0; j < NND; j++){
		    PARCEL_INT i0 = i + offset[j];
		    double A_hi_ = A[i+j*N];
		    double A_lo_ = 0.0;
		    if( (-1<i0) && (i0<N) ){
		      double x_hi_ = x_now[i0];
		      double x_lo_ = 0.0;

		      // x_i = x_i - A_*x_;
		      double Ax_hi_ = 0.0;
		      double Ax_lo_ = 0.0;
		      double ddmul_var1; 
		      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
		      
		      double ddsub_var1; 
		      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
		    }
		  }
		}else{
		  for(PARCEL_INT j = 0; j < NND; j++){
		    PARCEL_INT i0 = i + offset[j];
		    // T_calc A_ = A[i+j*N];
		    // T_calc x_ = x_now[i0];
		    double A_hi_ = A[i+j*N];
		    double A_lo_ = 0.0;
		    double x_hi_ = x_now[i0];
		    double x_lo_ = 0.0;

		    // x_i = x_i - A_*x_;
		    double Ax_hi_ = 0.0;
		    double Ax_lo_ = 0.0;
		    double ddmul_var1; 
		    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
		      
		    double ddsub_var1; 
		    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);

		  }
		}

		if( ( (iter+1)==iterMax) ){
		  x[i] = x_i_hi_;
		}else{
		  x_next[i] = x_i_hi_;
		}
#if _BLOCKING_LOOP_
	      }
	      }
	  });
#else
      });
#endif
    q_[0].wait();
  }

  if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
#else
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		// x[i] = x[i] * bnorm;
		double x_hi_ = x[i];
		double x_lo_ = 0.0;
		double ddmul_var1; 
		ddmul_calc1(x_hi_,x_lo_,bnorm_hi_,bnorm_lo_,x_hi_,x_lo_);
		x[i] = x_hi_;

#if _BLOCKING_LOOP_
	      }
	      }
	  });
#else
      });
#endif
    q_[0].wait();
  }

  return ;

}


#endif


#endif
