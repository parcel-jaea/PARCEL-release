template<class TdataA,class TdataB,class TdataMat,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  // std::cout<<"     start ddgemv all ftype : "<<std::endl;
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat=bool,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat=bool,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat=bool,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB,class TdataMat=bool,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat=bool,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat=bool,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat=bool,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA,class TdataB=bool,class TdataMat=bool,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  TdataA *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat=bool,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat=bool,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat=bool,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB,class TdataMat=bool,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataB *beta, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataMat *A, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A[i+j*ldA]; 
	double A_lo_ =  0.0; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat=bool,class TdataX,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat=bool,class TdataX,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataX *x, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x[j]; 
	double x_lo_ =  0.0; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat=bool,class TdataX=bool,class TdataY>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  TdataY *y, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y[i]; 
      double y_lo_ =  0.0; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y[i] = y_hi_ + y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 

template<class TdataA=bool,class TdataB=bool,class TdataMat=bool,class TdataX=bool,class TdataY=bool>
void ddgemv( 
	  PARCEL_INT N, 
	  PARCEL_INT M, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi, double *A_lo, 
	  PARCEL_INT ldA, 
	  double *x_hi,double *x_lo, 
	  PARCEL_INT ldx, 
	  double *y_hi,double *y_lo, 
	  PARCEL_INT ldy){       
  size_t MAX   = N; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
#endif 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double Ax_hi_ = 0.0; 
      double Ax_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT j=0;j<M;j++){ 
	double x_hi_ =  x_hi[j]; 
	double x_lo_ =  x_lo[j]; 
	double A_hi_ =  A_hi[i+j*ldA]; 
	double A_lo_ =  A_lo[i+j*ldA]; 
	// Ax_ = Ax_ +  A_*x_; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); 
      } 
      double y_hi_ =  y_hi[i]; 
      double y_lo_ =  y_lo[i]; 
      //      y_  = beta_*y_; 
      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); 
      //      Ax_ = alpha_*Ax_; 
      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); 
      //      y_   = y_ + Ax_; 
      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); 
 
      y_hi[i] = y_hi_; 
      y_lo[i] = y_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
    }); 
 
} 


