template<class TdataAlpha,class TdataBeta,class TdataA,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA=bool,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA=bool,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA=bool,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta,class TdataA=bool,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA=bool,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA=bool,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA=bool,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha,class TdataBeta=bool,class TdataA=bool,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  TdataAlpha *alpha,
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha[0]; 
      const double alpha_lo_ = 0.0; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA=bool,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA=bool,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA=bool,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta,class TdataA=bool,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  TdataBeta *beta,
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta[0]; 
      const double beta_lo_  = 0.0; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  TdataA *A,
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A[i_a + j_a*ldA]; 
	const double A_lo_ = 0.0; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA=bool,class TdataB,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA=bool,class TdataB,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  TdataB *B,
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B[i_b + j_b*ldB]; 
	const double B_lo_ = 0.0; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA=bool,class TdataB=bool,class TdataC>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  TdataC *C,
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C[i_c + j_c*ldC]; 
      double C_lo_ = 0.0; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C[i_c + j_c*ldC] = C_hi_ + C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 
template<class TdataAlpha=bool,class TdataBeta=bool,class TdataA=bool,class TdataB=bool,class TdataC=bool>
void ddgemmTN( 
	  PARCEL_INT m, 
	  PARCEL_INT n, 
	  PARCEL_INT k, 
	  double *alpha_hi,double *alpha_lo, 
	  double *beta_hi,double *beta_lo, 
	  double *A_hi,double *A_lo, 
	  PARCEL_INT ldA, 
	  double *B_hi,double *B_lo, 
	  PARCEL_INT ldB, 
	  double *C_hi,double *C_lo, 
	  PARCEL_INT ldC){ 
  size_t MAX   = m*n; 
#if _BLOCKING_LOOP_ 
 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 1; 
#else 
  const size_t nblock      = m*n; 
  const size_t nlocalblock = 1; 
#endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ 
      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ 
	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); 
	for(size_t i=iii;i<imax;i++){ 
#else 
 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ 
 
#endif 
      const PARCEL_INT i_c = i%m; 
      const PARCEL_INT j_c = i/m; 
 
      const double alpha_hi_ = alpha_hi[0]; 
      const double alpha_lo_ = alpha_lo[0]; 
      const double beta_hi_  = beta_hi[0]; 
      const double beta_lo_  = beta_lo[0]; 
 
      double AB_hi_ = 0.0; 
      double AB_lo_ = 0.0; 
      double ddmul_var1; 
      double ddadd_var1; 
      for(PARCEL_INT k_=0;k_<k;k_++){ 
	const PARCEL_INT i_a = k_; 
	const PARCEL_INT j_a = i_c; 
	const PARCEL_INT i_b = k_; 
	const PARCEL_INT j_b = j_c; 
	const double A_hi_ = A_hi[i_a + j_a*ldA]; 
	const double A_lo_ = A_lo[i_a + j_a*ldA]; 
	const double B_hi_ = B_hi[i_b + j_b*ldB]; 
	const double B_lo_ = B_lo[i_b + j_b*ldB]; 
	double tmp_hi = 0.0; 
	double tmp_lo = 0.0; 
	// AB_ = AB_ + A_ * B_; 
	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); 
	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); 
      } 
      double C_hi_ = C_hi[i_c + j_c*ldC]; 
      double C_lo_ = C_lo[i_c + j_c*ldC]; 
 
      double alphaAB_hi_ = 0.0; 
      double alphaAB_lo_ = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); 
 
      double betaC_hi_ = 0.0; 
      double betaC_lo_ = 0.0; 
      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); 
 
 
      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); 
      C_hi[i_c + j_c*ldC] =C_hi_; 
      C_lo[i_c + j_c*ldC] =C_lo_; 
 
#if _BLOCKING_LOOP_ 
     } 
     } 
#else 
 
#endif 
 
});
} 

