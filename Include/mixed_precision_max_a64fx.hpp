#ifndef MIXED_PRECISON_MAX_A64FX_HPP_
#define MIXED_PRECISON_MAX_A64FX_HPP_


template<class TdataX,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx
	 ){
#if 1

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
	  Tret x_ = x[i];
	  x_ = std::abs(x_);
	  sum_local[i-iii] = sum_local[i-iii] < x_ ? x_ : sum_local[i-iii] ;
	}
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] < sum_local[i] ? sum_local[i] : out[ii];
      }
						  
    });
  q_[0].wait();
  double result[nblock];
  const size_t bytes = sizeof(double)*nblock;
  q_[0].memcpy(result, out, bytes);
  q_[0].wait();

  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret < result[i] ? result[i] : ret ;
  }
      
  sycl::free(out, q_[0]);      
  sycl::free(sum_local_, q_[0]);      
      
  if(ret == 0.0)ret = 1.0;

  return ret;
  
#else  
  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  Tret outbuf_[num_work_groups];
  sycl::buffer<Tret> outbuf(outbuf_,num_work_groups);

  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	Tret,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch(work_group_size, h); 
      auto out = outbuf.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id  = item.get_global_id(0);
		       size_t local_id   = item.get_local_id(0);
		       size_t group_id   = item.get_group(0);
		       scratch[local_id] = 0.0;
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 Tret x_ = x[i];
                         x_ = std::abs(x_);
                         scratch[local_id] = scratch[local_id] < x_ ? x_ :  scratch[local_id];
		       }
		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
// scratch[local_id] += scratch[local_id + i]; 
                           scratch[local_id] = scratch[local_id + i] < scratch[local_id + i] ? scratch[local_id + i] : scratch[local_id + i];
			 }
		       }
		       if (local_id == 0) {
			 // outbuf[group_id] = scratch[0];
			 out[group_id] = scratch[0];
		       }

		     });
    });
  q_[0].wait();

  //      auto result = outbuf.get_access();
  sycl::host_accessor result{outbuf,sycl::read_only};

  Tret ret = 0.0;
  for(size_t i=0;i<num_work_groups;i++){
    ret =  result[i] < ret ? ret : result[i];
    //	ret = ret + outbuf_[i];
  }

#endif
  
  if(ret == 0.0)ret = 1.0;

  return ret;

}

template<class TdataX,class TdataY,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){

#if 1

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
	  Tret x_  = x[i];
	  Tret y_  = y[i];
	  Tret xy_ = x_*y_; 
	  xy_ = std::abs(xy_);
	  sum_local[i-iii] = sum_local[i-iii] < xy_ ? xy_ : sum_local[i-iii] ;
	}
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] < sum_local[i] ? sum_local[i] : out[ii];
      }
						  
    });
  q_[0].wait();
  double result[nblock];
  const size_t bytes = sizeof(double)*nblock;
  q_[0].memcpy(result, out, bytes);
  q_[0].wait();

  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret < result[i] ? result[i] : ret ;
  }
      
  sycl::free(out, q_[0]);      
  sycl::free(sum_local_, q_[0]);      
      
  if(ret == 0.0)ret = 1.0;

  return ret;
  
#else  

  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  Tret outbuf_[num_work_groups];
  sycl::buffer<Tret> outbuf(outbuf_,num_work_groups);

  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	Tret,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch(work_group_size, h); 
      auto out = outbuf.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id  = item.get_global_id(0);
		       size_t local_id   = item.get_local_id(0);
		       size_t group_id   = item.get_group(0);
		       scratch[local_id] = 0.0;
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 Tret x_  = x[i];
			 Tret y_  = y[i];
			 Tret xy_ = x_*y_; 
                         xy_ = std::abs(xy_);
                         scratch[local_id] = scratch[local_id] < xy_ ? xy_ :  scratch[local_id];
		       }
		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
// scratch[local_id] += scratch[local_id + i]; 
                           scratch[local_id] = scratch[local_id + i] < scratch[local_id + i] ? scratch[local_id + i] : scratch[local_id + i];
			 }
		       }
		       if (local_id == 0) {
			 // outbuf[group_id] = scratch[0];
			 out[group_id] = scratch[0];
		       }

		     });
    });
  q_[0].wait();

  //      auto result = outbuf.get_access();
  sycl::host_accessor result{outbuf,sycl::read_only};

  Tret ret = 0.0;
  for(size_t i=0;i<num_work_groups;i++){
    ret =  result[i] < ret ? ret : result[i];
    //	ret = ret + outbuf_[i];
  }
#endif
  
  if(ret == 0.0)ret = 1.0;

  return ret;

// #endif


}


#endif

