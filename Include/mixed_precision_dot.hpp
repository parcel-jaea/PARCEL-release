#ifndef MIXED_PRECISON_DOT_HPP_
#define MIXED_PRECISON_DOT_HPP_


#define dot_cpu 1
// #define dot_cpu 1
// #define dot_tree 1

// Intel + fp16でうまくいかない?
// #define dot_reduction 1

#if _BLOCKING_LOOP_
#include "mixed_precision_dot_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_DOT_A64FX_HPP_
template<class TdataX,class TdataY,class Tcalc,class Tret>
Tret dot(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){

#if dot_cpu

  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  Tcalc outbuf_[num_work_groups];
  sycl::buffer<Tcalc> outbuf(outbuf_,num_work_groups);

  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	Tcalc,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch(work_group_size, h); 
      auto out = outbuf.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id  = item.get_global_id(0);
		       size_t local_id   = item.get_local_id(0);
		       size_t group_id   = item.get_group(0);
		       scratch[local_id] = (Tcalc)0.0;
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 Tcalc x_ = x[i];
			 Tcalc y_ = y[i];
			 scratch[local_id] += x_*y_;
		       }
		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   scratch[local_id] += scratch[local_id + i];
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

  Tcalc ret = 0.0;
  for(size_t i=0;i<num_work_groups;i++){
    ret =  ret + result[i];
    //	ret = ret + outbuf_[i];
  }

  return ret;

#endif


#if dot_tree
  Tcalc *work = sycl::malloc_device<Tcalc>((num_work_groups_+1),q_[0]);
  Tret ret_ = treeDot<TdataX,TdataY,Tcalc,Tret>(
						N, 
						x,ldx,
						y,ldy,
						work // 作業領域 : (num_work_groups + 1) 
						) ;
  return ret_;
#endif


#if dot_reduction

  // reduction
  size_t MAX = N;

#if 0

#ifdef HIPSYCL_SYCL_HPP

#else
  // Intel 
  if (std::is_same<Tcalc,sycl::half>::value) {
    const size_t work_group_size = 512; // simd lengh
    const size_t num_work_groups = 512; // cpu core
    const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

    Tcalc outbuf_[num_work_groups];
    sycl::buffer<Tcalc> outbuf(outbuf_,num_work_groups);

    q_[0].wait();      
    q_[0].submit([&] (sycl::handler &h){
	sycl::accessor<
	  Tcalc,1,
	  sycl::access::mode::read_write,
	  sycl::access::target::local> scratch(work_group_size, h); 
	auto out = outbuf.get_access(h);	  
	h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		       [=](sycl::nd_item<1> item) {
			 size_t global_id = item.get_global_id(0);
			 size_t local_id     = item.get_local_id(0);
			 size_t group_id     = item.get_group(0);
			 scratch[local_id] = 0.0;
			 for(size_t i = global_id; i<N;i=i+num_work_items){
			   Tcalc x_ = x[i];
			   Tcalc y_ = y[i];
			   scratch[local_id] += x_*y_;
			 }
			 // Do a tree reduction on items in work-group
			 for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			   item.barrier(sycl::access::fence_space::local_space);
			   if (local_id < i){
			     scratch[local_id] += scratch[local_id + i];
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
    sycl::host_accessor result_{outbuf,sycl::read_only};

    Tcalc ret_Tcalc = 0.0;
    for(size_t i=0;i<num_work_groups;i++){
      ret_Tcalc =  ret_Tcalc + result_[i];
      //	ret = ret + outbuf_[i];
    }
    return ret_Tcalc;
  }

#endif

#endif


#ifdef HIPSYCL_SYCL_HPP

  // hipSYCL 
  auto ret_ = sycl::malloc_device<Tcalc>(1,q_[0]);
  sycl::buffer<Tcalc> buff(&ret_[0],1);
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor sum{buff, h};
      h.parallel_for(sycl::range<1>{MAX},
		     sycl::reduction(sum,sycl::plus<Tcalc>()),
		     // sycl::reduction(sum,sycl::multiplies<Tcalc>()),
		     [=](sycl::id<1> i, auto &sum_){
		       sum_ += x[i]*y[i];
		       //sum_ *= x[i]*y[i];
		     });
    });
  Tcalc ret_h = 0.0; // = ret_[0];
  q_[0].wait();
  q_[0].memcpy(&ret_h, ret_,  sizeof(Tcalc));
  q_[0].wait();
  sycl::free(ret_ , q_[0]);
  Tret ret = ret_h;

  return ret;

#else

  Tcalc ret_ = 0.0;
  sycl::buffer<Tcalc> buff(&ret_,1);
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      //	  sycl::stream out(1024*10, 1024, h);
      h.parallel_for(sycl::range<1>{MAX},
		     sycl::reduction(buff, h, sycl::plus<>() ),
		     [=](sycl::id<1> i, auto &sum){
		       sum += x[i]*y[i];
		     });
    });
  q_[0].wait();
  sycl::host_accessor result{buff, sycl::read_only};
  Tret ret = result[0];

  return ret;

#endif

#endif

#if 0
  // GPU でerror
  // atomic reduction
  Tcalc ret_atomic = 0.0;
  ret_atomic =  dot_< TdataX, TdataY, Tcalc, Tret>(
						   N,
						   x,
						   ldx,
						   y,
						   ldy);
  return ret_atomic;
#endif

}
#endif

// atomic_refによるdot
template<class TdataX,class TdataY,class Tcalc,class Tret>
Tret dot_(
	  PARCEL_INT N,
	  TdataX *x,
	  PARCEL_INT ldx,
	  TdataY *y,
	  PARCEL_INT ldy){      
  const int num_work_items = 1024 * 1024;

  Tcalc res = 0.0;
  sycl::buffer<Tcalc> sum_buf(&res,1);

  q_[0].submit([&](auto &h) {
      sycl::accessor sum_acc(sum_buf, h, sycl::write_only, sycl::no_init);
      h.parallel_for(sycl::range<1>{num_work_items}, [=](sycl::id<1> index) {
	  Tcalc sum = 0.0;
	  for (size_t i = index; i < N; i=i+num_work_items){
	    Tcalc x_ = x[i];
	    Tcalc y_ = y[i];
	    sum += x_*y_;
	  }
	  auto v = sycl::atomic_ref<
	    Tcalc, sycl::memory_order::relaxed,
	    sycl::memory_scope::device,
	    sycl::access::address_space::global_space>(sum_acc[0]);
	  v.fetch_add(sum);
	});
    });

  q_[0].wait();
  sycl::host_accessor h_acc(sum_buf);
  res=h_acc[0];
 
  return res; 

}


template<class TdataX,class TdataY,class Tcalc>
void treeDot_(
	      sycl::queue *q,
	      TdataX *x,
	      TdataY *y,
	      Tcalc  *outbuf, // グループ毎の出力結果
	      const PARCEL_INT N,
	      const size_t num_work_groups, // グループ数
	      const size_t work_group_size  // 1グループのサイズ
	      ) {
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  // treeReduction 
  q[0].submit([&](auto &h) {
      // グループ内で参照可能なローカルメモリ
      sycl::accessor<Tcalc,1,sycl::access::mode::read_write,sycl::access::target::local> scratch(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);
		       scratch[local_id] = 0.0;
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 Tcalc x_ = x[i];
			 Tcalc y_ = y[i];
			 scratch[local_id] += x_*y_;
		       }
		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   scratch[local_id] += scratch[local_id + i];
			 }
		       }
		       if (local_id == 0) {
			 outbuf[group_id] = scratch[0];
		       }

		     });
    });
  q[0].wait();
  return ;
}

template<class TdataX,class TdataY,class Tcalc,class Tret>
Tret treeDot(
	     PARCEL_INT N, 
	     TdataX *x,PARCEL_INT ldx,
	     TdataY *y,PARCEL_INT ldy,
	     Tcalc  *work // 作業領域 : (num_work_groups + 1) 
	     ) {
  const size_t work_group_size = q_[0].get_device().get_info<sycl::info::device::max_work_group_size>(); // localグループ数
  size_t num_work_groups = num_work_groups_;
  if(num_work_groups<work_group_size){
    num_work_groups = 1;
  }
  const size_t num_work_items  = work_group_size*num_work_groups; // 全並列数

  Tret res  = 0.0;
  Tcalc res_ = 0.0;
  Tcalc *ret_buf = &work[0];
  Tcalc *outbuf  = &work[1];
  treeDot_<TdataX,TdataY,Tcalc>(
				q_,
				x,
				y,
				outbuf,
				N,
				num_work_groups, // グループ数
				work_group_size // 1グループのサイズ
				) ;
  treeReduction_<Tcalc>(
			q_,
			outbuf ,
			ret_buf,
			num_work_groups,
			1, // グループ数
			work_group_size // 1グループのサイズ
			) ;

  const size_t bytes = sizeof(Tcalc);
  q_[0].memcpy(&res_ , ret_buf, bytes);
  q_[0].wait();
  res = res_;
  return res;

} // end treeDot

template<class T>
void treeReduction_(
		    sycl::queue *q,
		    T *inbuf ,
		    T  *outbuf, // グループ毎の出力結果
		    const PARCEL_INT N,
		    const size_t num_work_groups, // グループ数
		    const size_t work_group_size  // 1グループのサイズ
		    ) {
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  // treeReduction 
  q[0].submit([&](auto &h) {
      // グループ内で参照可能なローカルメモリ
      sycl::accessor<T,1,sycl::access::mode::read_write,sycl::access::target::local> scratch(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch[local_id] = 0.0;
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 scratch[local_id] += inbuf[i];
		       }

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   scratch[local_id] += scratch[local_id + i];
			 }
		       }
		       if (local_id == 0) {
			 outbuf[group_id] = scratch[0];
		       }

		     });
    });
  q[0].wait();
  return ;
}

#endif
