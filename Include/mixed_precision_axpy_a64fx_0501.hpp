// #if _A64FX_HIPSYCL_

#ifndef MIXED_PRECISON_AXPY_A64FX_HPP_
#define MIXED_PRECISON_AXPY_A64FX_HPP_

template<class TdataA,class TdataB,class TdataX,class TdataY,class TdataZ,class Tcalc>
void axpy(
	  PARCEL_INT N,
	  TdataA *alpha,
	  TdataB *beta,
	  TdataX *x,
	  PARCEL_INT ldx,
	  TdataY *y,
	  PARCEL_INT ldy,
	  TdataZ *z,
	  PARCEL_INT ldz
	  ){

  //  std::cout<<" a64fx axpy 1 "<<std::endl;
  size_t MAX   = N;
#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  
#if 0
  Tcalc alpha_ = alpha[0];
  Tcalc beta_  = beta[0];
#pragma omp parallel for
  for(size_t i=0;i<N;i++){
    z[i] = beta_*y[i] + alpha_*x[i];
  }
#else  
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];

#pragma clang swp      
#pragma clang loop vectorize(enable)
#pragma clang loop unroll(enable)
#pragma fj loop prefetch
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	//	for(size_t i=iii;i<imax;i=i+2){					   
#if 0
	  z[i] = beta_*y[i] + alpha_*x[i];
#else
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  z[i] = y_;
#endif	  
	}
      }
    });
#endif
  
}


template<class TdataA,class TdataB,class TdataX,class TdataY,class Tcalc>
void axpy(
	  PARCEL_INT N,
	  TdataA *alpha,
	  TdataB *beta,
	  TdataX *x,
	  PARCEL_INT ldx,
	  TdataY *y,
	  PARCEL_INT ldy){      
  size_t MAX   = N;

  //  std::cout<<" a64fx axpy 2 "<<std::endl;
#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  
#if 0
  Tcalc alpha_ = alpha[0];
  Tcalc beta_  = beta[0];
#pragma omp parallel for
  for(size_t i=0;i<N;i++){
    y[i] = beta_*y[i] + alpha_*x[i];
  }
#else  
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);

#pragma clang swp      
#pragma clang loop vectorize(enable) 
#pragma clang loop unroll(enable)
#pragma fj loop prefetch
	for(size_t i=iii;i<imax;i++){
	//	for(size_t i=iii;i<imax;i=i+2){
#if 0
	  y[i] = beta_*y[i] + alpha_*x[i];
#else
	  Tcalc x_ =  x[i];
	  Tcalc y_ =  y[i];
	  y_   = beta_*y_ + alpha_*x_;
	  y[i] = y_;
#endif	  
	}
      }
    });
#endif
  
}

template<class Tdata>
void copy(
	  PARCEL_INT N,
	  Tdata *x,
	  PARCEL_INT ldx,
	  Tdata *y,
	  PARCEL_INT ldy){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      y[i] = x[i];
    });
}

template<class Tdata>
void zero(
	  PARCEL_INT N,
	  Tdata *x,
	  PARCEL_INT ldx
	  ){      
  size_t MAX   = N;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      x[i] = 0.0;
    });
}

#endif

// #endif
