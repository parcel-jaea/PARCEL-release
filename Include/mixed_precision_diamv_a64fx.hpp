// #if _A64FX_HIPSYCL_

#ifndef MIXED_PRECISON_DIAMV_A64FX_HPP_
#define MIXED_PRECISON_DIAMV_A64FX_HPP_

template<
  class TdataA,class TdataDIA,class TdataX,      
  class TdataB,class TdataY,      
  class Tcalc>
void diamv(
	   PARCEL_INT N,
	   PARCEL_INT NND,
	   TdataA *alpha,
	   TdataDIA *A,
	   PARCEL_INT *offset,
	   PARCEL_INT max_up_offset,
	   PARCEL_INT max_down_offset,
	   TdataX *x,
	   PARCEL_INT ldx,
	   TdataB *beta,
	   TdataY *y,
	   PARCEL_INT ldy){      
  size_t MAX   = N;
  // y = α*Ax + β*y;
  // std::cout<<"max_up_offset   = "<<max_up_offset<<std::endl;
  // std::cout<<"max_down_offset = "<<max_down_offset<<std::endl;

#if 1


#if 1

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 4096*(8/sizeof(TdataX));
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(PARCEL_INT iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const PARCEL_INT imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	if( ( 0<=(iii+max_up_offset) ) && ( (imax+max_down_offset)<N ) ){
	  Tcalc Ax[nlocalblock];
	  for(PARCEL_INT i=0;i<nlocalblock;i++){
	    Ax[i] = 0.0;
	  }

	  for(PARCEL_INT j = 0;j<NND;j++){
	    for(PARCEL_INT i=iii;i<imax;i++){
	      const Tcalc A_ = A[i+j*N];
	      const Tcalc x_ = x[i+offset[j]];
	      Ax[i-iii] = Ax[i-iii] + A_*x_;
	    }
	  }
	  const Tcalc alpha_ = alpha[0];
	  const Tcalc beta_  = beta[0];
	  for(PARCEL_INT i=iii;i<imax;i++){
	    Tcalc y_ = y[i];
	    y_ = alpha_*Ax[i-iii] + beta_*y_;
	    y[i] = y_;
	  }

	}else{
	  for(PARCEL_INT i=iii;i<imax;i++){
	    Tcalc Ax = 0.0;
	    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	      for(PARCEL_INT j = 0;j<NND;j++){
		if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
		  const Tcalc A_ = A[i+j*N];
		  const Tcalc x_ = x[i+offset[j]];
		  Ax = Ax + A_*x_;
		}
	      }
	    }else{
	      for(PARCEL_INT j = 0;j<NND;j++){
		const Tcalc A_ = A[i+j*N];
		const Tcalc x_ = x[i+offset[j]];
		Ax = Ax + A_*x_;
	      }
	    }
	    const Tcalc alpha_ = alpha[0];
	    const Tcalc beta_  = beta[0];
	    Tcalc y_ = y[i];
	    y_ = alpha_*Ax + beta_*y_;
	    y[i] = y_;
	  }
	}
      }
    });
	    
#endif


#if 0
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
      const PARCEL_INT i = i_;
      Tcalc Ax = 0.0;
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
	for(PARCEL_INT j = 0;j<NND;j++){
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){
	    const Tcalc A_ = A[i+j*N];
	    const Tcalc x_ = x[i+offset[j]];
	    Ax = Ax + A_*x_;
	  }
	}
      }else{
	for(PARCEL_INT j = 0;j<NND;j++){
	  const Tcalc A_ = A[i+j*N];
	  const Tcalc x_ = x[i+offset[j]];
	  Ax = Ax + A_*x_;
	}
      }
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
    });
#endif

#else

  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
      Tcalc Ax = 0.0;
      for(PARCEL_INT j = 0;j<NND;j++){
	if( (0<=((PARCEL_INT)i+offset[j]) ) && (((PARCEL_INT)i+offset[j])<N) ){
	  const Tcalc A_ = A[i+j*N];
	  const Tcalc x_ = x[i+offset[j]];
	  Ax = Ax + A_*x_;
	}
      }
      const Tcalc alpha_ = alpha[0];
      const Tcalc beta_  = beta[0];
      Tcalc y_ = y[i];
      y_ = alpha_*Ax + beta_*y_;
      y[i] = y_;
    });

#endif

  return ;

}

#endif

// #endif
