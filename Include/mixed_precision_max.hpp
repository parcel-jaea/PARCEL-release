#ifndef MIXED_PRECISON_MAX_HPP_

#define MIXED_PRECISON_MAX_HPP_

// #define dot_cpu 1
// #define dot_reduction 0

#if _BLOCKING_LOOP_
#include "mixed_precision_max_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_MAX_A64FX_HPP_

template<class TdataX,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx
	 ){

// #if dot_cpu

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
  if(ret == 0.0)ret = 1.0;

  return ret;

// #endif


}

template<class TdataX,class TdataY,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){

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
  if(ret == 0.0)ret = 1.0;

  return ret;

// #endif


}

#endif

#endif
