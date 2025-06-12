// #if _A64FX_HIPSYCL_

#ifndef MIXED_PRECISON_CRSMV_A64FX_HPP_
#define MIXED_PRECISON_CRSMV_A64FX_HPP_

template<
  class TdataA,class TdataCRS,class TdataX,      
  class TdataB,class TdataY,      
  class Tcalc>
void crsmv(
	   PARCEL_INT N,
	   PARCEL_INT NNZ,
	   TdataA *alpha,
	   TdataCRS *A,
	   PARCEL_INT *Rowptr,
	   PARCEL_INT *crsCol,
	   TdataX *x,
	   PARCEL_INT ldx,
	   TdataB *beta,
	   TdataY *y,
	   PARCEL_INT ldy,
	   double *dwork_,size_t work_bloncksize){      

  size_t MAX   = N;
#if _A64FX_HIPSYCL_
  // GPUではworkメモリが取得できないサイズ
  const size_t nblock      = CRS_SPMV_NBLOCK;
  const size_t nlocalblock = CRS_SPMV_BLOCKSIZE * (8/sizeof(TdataCRS));
#else
  // CUDAでもnlocalblockを1にすれば性能が出る
  const size_t nblock      = CRS_SPMV_NBLOCK;
  const size_t nlocalblock = CRS_SPMV_BLOCKSIZE;
#endif

  Tcalc *work_ = (Tcalc *)dwork_;
  // y = α*Ax + β*y;
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);

	const Tcalc alpha_ = alpha[0];
	const Tcalc beta_  = beta[0];
	Tcalc y_[nlocalblock];
	//	Tcalc work[Rowptr[imax]-Rowptr[iii]+1];
	Tcalc *work = &work_[ii*work_bloncksize];
	for(PARCEL_INT j = Rowptr[iii]; j < Rowptr[imax]; j++){
	  const PARCEL_INT col = crsCol[j];
	  const Tcalc A_ = A[j];
	  const Tcalc x_ = x[col];
	  work[j-Rowptr[iii]] = A_*x_;
	}
	for(size_t i=iii;i<imax;i++){
	  Tcalc Ax = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    Ax += work[j-Rowptr[iii]]; 
	  }
	  y_[i-iii] = alpha_*Ax;
	}
	for(size_t i=iii;i<imax;i++){
	  Tcalc yy_ = y[i];
	  y[i] = y_[i-iii] + beta_*yy_;
	}
	       
      }
	  
    });

  return ;

}

#endif

// #endif
