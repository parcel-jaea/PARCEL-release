#ifndef PARCEL_PRECON_DDJACOBI_CRS_
#define PARCEL_PRECON_DDJACOBI_CRS_

#if 1
#include "parcel_precon_ddjacobi_crs_py.hpp"
#else

template<class T_vec,class T_mat,class T_work_vec>
void ddjacobi_method_crs_main(
 			    sycl::queue *q_ ,
			    PARCEL_INT N,
			    PARCEL_INT iterMax,
			    T_vec *x, // out  
			    T_vec *b, // in
			    T_mat *A, // (D^-1)(L+U)
			    PARCEL_INT *Rowptr,
			    PARCEL_INT *crsCol,
			    T_mat *Dinv, // (D^-1)
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

  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv[i];
	  const double Dinv_lo_ = 0.0;
	  const double b_hi_    = b[i];
	  const double b_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  precon_b[i] = precon_b_hi_;
	}
      }
    });
  q_[0].wait();

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b[i];
	  double x_i_lo_ = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A[j];
	    const double A_lo_  = 0.0;
	    const double x_hi_  = precon_b[col];
	    const double x_lo_  = 0.0;
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x[i] = x_i_hi_;
	  }else{
	    x_next[i] = x_i_hi_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<T_work_vec>(&x_now,&x_next);
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b[i];
	    double x_i_lo_ = 0.0;
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A[j];
	      const double A_lo_  = 0.0;
	      const double x_hi_  = x_now[col];
	      const double x_lo_  = 0.0;
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x[i] = x_i_hi_;
	    }else{
	      x_next[i] = x_i_hi_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

  return ;

}

#endif

#endif
