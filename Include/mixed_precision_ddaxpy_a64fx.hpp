// #if _A64FX_HIPSYCL_

#ifndef MIXED_PRECISON_DDAXPY_A64FX_HPP_
#define MIXED_PRECISON_DDAXPY_A64FX_HPP_
void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi ,double *beta_lo,
	    double *x_hi,double *x_lo,
	    PARCEL_INT ldx,
	    double *y_hi,double *y_lo,
	    PARCEL_INT ldy){

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  //      fapp_start("ddaxpy",1,0); 
  q_[0].wait();

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
						  
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	double ax_hi[nlocalblock],ax_lo[nlocalblock];
	double by_hi[nlocalblock],by_lo[nlocalblock];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  ddmul_calc1(x_hi[i],x_lo[i],alpha_hi_,alpha_lo_,ax_hi[i-iii],ax_lo[i-iii]);
	  ddmul_calc1(y_hi[i],y_lo[i],beta_hi_ ,beta_lo_ ,by_hi[i-iii],by_lo[i-iii]);
	}
	for(size_t i=iii;i<imax;i++){						     
	  double ddadd_var1;
	  ddadd_calc1(ax_hi[i-iii],ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo[i]);
	}						
      }
    });
      
  q_[0].wait();
  //      fapp_stop("ddaxpy",1,0);
      
  return ;

}

template<class TdataA,class TdataB,class TdataX,class TdataY>
void ddaxpy(
	    PARCEL_INT N,
	    TdataA *alpha_hi,
	    TdataB *beta_hi,
	    TdataX *x_hi,
	    PARCEL_INT ldx,
	    TdataY *y_hi,
	    PARCEL_INT ldy){      

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  //      fapp_start("ddaxpy",1,0); 
  q_[0].wait();

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
						  
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = 0.0;
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = 0.0;
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	double ax_hi[nlocalblock],ax_lo[nlocalblock];
	double by_hi[nlocalblock],by_lo[nlocalblock];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double x_lo = 0.0;
	  double y_lo = 0.0;
	  ddmul_calc1(x_hi[i],x_lo,alpha_hi_,alpha_lo_,ax_hi[i-iii],ax_lo[i-iii]);
	  ddmul_calc1(y_hi[i],y_lo,beta_hi_ ,beta_lo_ ,by_hi[i-iii],by_lo[i-iii]);
	}
	for(size_t i=iii;i<imax;i++){						     
	  double ddadd_var1;
	  double y_lo = 0.0;
	  ddadd_calc1(ax_hi[i-iii],ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo);
	}						
      }
    });
      
  q_[0].wait();
  //      fapp_stop("ddaxpy",1,0);
      
  return ;

}

template<class TdataX,class TdataY>
void ddaxpy(
	    PARCEL_INT N,
	    double *alpha_hi,double *alpha_lo,
	    double *beta_hi,double *beta_lo,
	    TdataX *x_hi,
	    PARCEL_INT ldx,
	    TdataY *y_hi,
	    PARCEL_INT ldy){      

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  //      fapp_start("ddaxpy",1,0); 
  q_[0].wait();

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
						  
      const double alpha_hi_ = alpha_hi[0];
      const double alpha_lo_ = alpha_lo[0];
      const double beta_hi_  = beta_hi[0];
      const double beta_lo_  = beta_lo[0];
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	double ax_hi[nlocalblock],ax_lo[nlocalblock];
	double by_hi[nlocalblock],by_lo[nlocalblock];
	for(size_t i=iii;i<imax;i++){
	  double ddmul_var1;
	  double x_lo = 0.0;
	  double y_lo = 0.0;
	  ddmul_calc1(x_hi[i],x_lo,alpha_hi_,alpha_lo_,ax_hi[i-iii],ax_lo[i-iii]);
	  ddmul_calc1(y_hi[i],y_lo,beta_hi_ ,beta_lo_ ,by_hi[i-iii],by_lo[i-iii]);
	}
	for(size_t i=iii;i<imax;i++){						     
	  double ddadd_var1;
	  double y_lo = 0.0;
	  ddadd_calc1(ax_hi[i-iii],ax_lo[i-iii],by_hi[i-iii],by_lo[i-iii],y_hi[i],y_lo);
	}
      }
    });
      
  q_[0].wait();
  //      fapp_stop("ddaxpy",1,0);
      
  return ;

}

#endif

// #endif
