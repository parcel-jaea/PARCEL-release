// #if _A64FX_HIPSYCL_
// #if 1

#ifndef MIXED_PRECISON_DOT_A64FX_HPP_
#define MIXED_PRECISON_DOT_A64FX_HPP_

template<class TdataX,class TdataY,class Tcalc,class Tret>
Tret dot(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){      
  // デバイス同期なし
#if 0
  return 1.0;
#else
  
#if _A64FX_HIPSYCL_
  const size_t nblock    = 12;
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)) ;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
      
  double *out = sycl::malloc_device<double>(nblock,q_[0]);

  double *sum_local_ = sycl::malloc_device<double>(nlocalblock*nblock,q_[0]);
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out[ii] = 0.0;
      double sum_local[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  sum_local[i-iii] = sum_local[i-iii] + x[i]*y[i];
	}						    
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] + sum_local[i];
      }
						  
    });
  q_[0].wait();
  double result[nblock];
  const size_t bytes = sizeof(double)*nblock;
  q_[0].memcpy(result, out, bytes);
  q_[0].wait();

  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret + result[i];
  }
      
  sycl::free(out, q_[0]);      
  sycl::free(sum_local_, q_[0]);      
      
  return ret;

#endif
  
}


#endif

// #endif
