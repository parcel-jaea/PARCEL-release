#ifndef MIXED_PRECISON_GEMV_HPP_
#define MIXED_PRECISON_GEMV_HPP_

// 行列転置未対応
// y = alpha * Ax + beta * y 
template<class TdataA,class TdataB,class TdataMat,class TdataX,class TdataY,class Tcalc>
void gemv(
	  PARCEL_INT N, // 行数
	  PARCEL_INT M, // 列数
	  TdataA *alpha,
	  TdataB *beta,
	  TdataMat *A, // N行M列
	  PARCEL_INT ldA,
	  TdataX *x, // M行1列
	  PARCEL_INT ldx,
	  TdataY *y, // N行1列
	  PARCEL_INT ldy){      
  //  std::cout<<"M = "<<M<<std::endl;
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
      Tcalc alpha_ = alpha[0];
      Tcalc beta_  = beta[0];
      Tcalc Ax_ = 0.0;
      for(PARCEL_INT j=0;j<M;j++){
       	Tcalc x_ = x[j];
	Tcalc A_ = A[i+j*ldA];
	Ax_ = Ax_ + A_*x_;
      }
      Tcalc y_ =  y[i];
      y_   = beta_*y_;
      Ax_  = alpha_*Ax_;
      y_   = y_ + Ax_;
      y[i] = y_;

#if _BLOCKING_LOOP_
     }
     }
#else

#endif
    });

}

#endif
