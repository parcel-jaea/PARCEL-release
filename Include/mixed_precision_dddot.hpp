#ifndef MIXED_PRECISON_DDDOT_HPP_
#define MIXED_PRECISON_DDDOT_HPP_

#ifdef HIPSYCL_SYCL_HPP
// hipsycl(GPU) + DPC++ではddのリダクションがエラーとなる
#define dddot_reduction 0
#define dddot_cpu 1
#define dddot_tree 0
#else
#define dddot_cpu 1
#define dddot_tree 0
#endif

#if _BLOCKING_LOOP_
#include "mixed_precision_dddot_a64fx.hpp"
#endif

template<class TdataX,class TdataY>
void treeDddot_(
	       sycl::queue *q,
	       TdataX *x,
	       TdataY *y,
	       double *outbuf_hi, double *outbuf_lo, // グループ毎の出力結果
	       const PARCEL_INT N,
	       const size_t num_work_groups, // グループ数
	       const size_t work_group_size  // 1グループのサイズ
	       ) {
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数
  ddmath dd;
  // treeReduction 
  q[0].submit([&](auto &h) {
      // グループ内で参照可能なローカルメモリ
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_hi(work_group_size, h); 
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_lo(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch_hi[local_id] = 0.0;
		       scratch_lo[local_id] = 0.0;
		       ddtype scratch(0.0,0.0);
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 ddtype x_(x[i]);
			 ddtype y_(y[i]);
			 ddtype dd_xy  = x_ * y_;
			 ddtype dd_sum = scratch + dd_xy;
			 scratch = dd_sum;
		       }
		       scratch_hi[local_id] = scratch.get_hi();
		       scratch_lo[local_id] = scratch.get_lo();

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id],scratch_lo[local_id]);
			   ddtype dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
			   ddtype dd_sum = dd_lid + dd_i;
			   scratch_hi[local_id] = dd_sum.get_hi();
			   scratch_lo[local_id] = dd_sum.get_lo();
			 }
		       }
		       if (local_id == 0) {
			 // outbuf[group_id] = scratch[0];
			 outbuf_hi[group_id] = scratch_hi[0];
			 outbuf_lo[group_id] = scratch_lo[0];
			 //	 printf(" outbuf_hi = %f \n",outbuf_hi[group_id]);
		       }

		     });
    });
  q[0].wait();
  return ;
}

void treeDddot_(
	       sycl::queue *q,
	       double *x_hi,double *x_lo,
	       double *y_hi,double *y_lo,
	       double *outbuf_hi, double *outbuf_lo, // グループ毎の出力結果
	       const PARCEL_INT N,
	       const size_t num_work_groups, // グループ数
	       const size_t work_group_size  // 1グループのサイズ
	       ) {
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数
  ddmath dd;
  // treeReduction 
  q[0].submit([&](auto &h) {
      // グループ内で参照可能なローカルメモリ
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_hi(work_group_size, h); 
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_lo(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch_hi[local_id] = 0.0;
		       scratch_lo[local_id] = 0.0;
		       ddtype scratch(0.0,0.0);
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 ddtype x_(x_hi[i],x_lo[i]);
			 ddtype y_(y_hi[i],y_lo[i]);
			 ddtype dd_xy   = x_ * y_;
			 ddtype dd_sum = scratch + dd_xy;
			 scratch = dd_sum;
		       }
		       scratch_hi[local_id] = scratch.get_hi();
		       scratch_lo[local_id] = scratch.get_lo();

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id]    ,scratch_lo[local_id]);
			   ddtype   dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
			   ddtype dd_sum = dd_lid + dd_i;
			   scratch_hi[local_id] = dd_sum.get_hi();
			   scratch_lo[local_id] = dd_sum.get_lo();
			 }
		       }
		       if (local_id == 0) {
			 outbuf_hi[group_id] = scratch_hi[0];
			 outbuf_lo[group_id] = scratch_lo[0];
		       }

		     });
    });
  q[0].wait();
  return ;
}

void treeDDReduction_(
		      sycl::queue *q,
		      double *inbuf_hi ,double *inbuf_lo,
		      double *outbuf_hi,double *outbuf_lo, // グループ毎の出力結果
		      const PARCEL_INT N,
		      const size_t num_work_groups, // グループ数
		      const size_t work_group_size  // 1グループのサイズ
		      ) {
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数
  ddmath dd;
  // printf(" N = %d \n",N); 
  // treeReduction 
  q[0].submit([&](auto &h) {
      // グループ内で参照可能なローカルメモリ
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_hi(work_group_size, h); 
      sycl::accessor<double,1,sycl::access::mode::read_write,sycl::access::target::local> scratch_lo(work_group_size, h); 
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch_hi[local_id] = 0.0;
		       scratch_lo[local_id] = 0.0;
		       ddtype scratch(0.0,0.0);
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 ddtype inbuf_(inbuf_hi[i],inbuf_lo[i]);
			 ddtype dd_sum = scratch + inbuf_;
			 scratch = dd_sum;
		       }
		       scratch_hi[local_id] = scratch.get_hi();
		       scratch_lo[local_id] = scratch.get_lo();

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id]    ,scratch_lo[local_id]);
			   ddtype   dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
			   ddtype dd_sum = dd_lid + dd_i;
			   scratch_hi[local_id] = dd_sum.get_hi();
			   scratch_lo[local_id] = dd_sum.get_lo();
			 }
		       }

		       if (local_id == 0) {
			 outbuf_hi[group_id] = scratch_hi[0];
			 outbuf_lo[group_id] = scratch_lo[0];
		       }

		     });
    });
  q[0].wait();
  return ;
}

#ifndef MIXED_PRECISON_DDDOT_A64FX_HPP_
template<class TdataX,class TdataY>
ddtype dddot(
	    PARCEL_INT N,
	    TdataX *x,PARCEL_INT ldx,
	    TdataY *y,PARCEL_INT ldy,
	    double *work // 作業領域 : (num_work_groups + 1) * 2
	    ){
#if dddot_reduction

#ifdef HIPSYCL_SYCL_HPP
  // hipSYCL 
  size_t MAX = N;
  auto ret_ = sycl::malloc_device<ddtype>(1,q_[0]);
  sycl::buffer<ddtype> buff(&ret_[0],1);
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor sum{buff, h};
      h.parallel_for(sycl::range<1>{MAX},
		     sycl::reduction(sum,sycl::plus< ddtype >()),
		     [=](sycl::id<1> i, auto &sum_){
		       ddtype x_(x[i],0.0);
		       ddtype y_(y[i],0.0);
		       sum_ += x_*y_;
		     });
    });
  q_[0].wait();

  ddtype ret(0.0,0.0);
  q_[0].memcpy(&ret, ret_,  sizeof(ddtype));
  q_[0].wait();
  sycl::free(ret_ , q_[0]);

  return ret;

#else

  printf(" not support :: dddot_reduction d \n");
  ddtype ret(0.0,0.0);

  return ret;
#endif

#endif

#if dddot_cpu
  //#if 1
  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  double outbuf_hi_[num_work_groups];
  double outbuf_lo_[num_work_groups];
  sycl::buffer<double> outbuf_hi(outbuf_hi_,num_work_groups);
  sycl::buffer<double> outbuf_lo(outbuf_lo_,num_work_groups);

  ddmath dd;
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	double,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch_hi(work_group_size, h); 
      sycl::accessor<
	double,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch_lo(work_group_size, h); 

      auto out_hi = outbuf_hi.get_access(h);	  
      auto out_lo = outbuf_lo.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch_hi[local_id] = 0.0;
		       scratch_lo[local_id] = 0.0;
		       ddtype scratch(0.0,0.0);
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 ddtype x_(x[i]);
			 ddtype y_(y[i]);
			 ddtype dd_xy  = x_ * y_;
			 ddtype dd_sum = scratch + dd_xy;
			 scratch = dd_sum;
		       }
		       scratch_hi[local_id] = scratch.get_hi();
		       scratch_lo[local_id] = scratch.get_lo();

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id],scratch_lo[local_id]);
			   ddtype dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
			   ddtype dd_sum = dd_lid + dd_i;
			   scratch_hi[local_id] = dd_sum.get_hi();
			   scratch_lo[local_id] = dd_sum.get_lo();
			 }
		       }
		       if (local_id == 0) {
			 out_hi[group_id] = scratch_hi[0];
			 out_lo[group_id] = scratch_lo[0];
		       }

		     });
    });
  q_[0].wait();

  //      auto result = outbuf.get_access();
  sycl::host_accessor result_hi{outbuf_hi,sycl::read_only};
  sycl::host_accessor result_lo{outbuf_lo,sycl::read_only};

  ddtype ret(0.0,0.0);
  for(size_t i=0;i<num_work_groups;i++){
    ddtype dd_out(result_hi[i],result_lo[i]);
    ret = ret + dd_out;
  }

  return ret;

#endif

#if dddot_tree
  //#if 1

  const size_t work_group_size = q_[0].get_device().get_info<sycl::info::device::max_work_group_size>(); // localグループ数
  size_t num_work_groups = num_work_groups_;
  if(num_work_groups<work_group_size){
    num_work_groups = 1;
  }
  //      num_work_groups = 1;
  const size_t num_work_items  = work_group_size*num_work_groups; // 全並列数
  double *ret_buf_hi = &work[0]; 
  double *ret_buf_lo = &work[1];
  double *outbuf_hi  = &work[2];
  double *outbuf_lo  = &work[num_work_groups+2];

  treeDddot_<TdataX,TdataY>(
			   q_,
			   x,
			   y,
			   outbuf_hi,outbuf_lo,
			   N,
			   num_work_groups, // グループ数
			   work_group_size // 1グループのサイズ
			   ) ;
  q_[0].wait();
  treeDDReduction_(
		   q_,
		   outbuf_hi  , outbuf_lo,
		   ret_buf_hi , ret_buf_lo,
		   num_work_groups,
		   1, // グループ数
		   work_group_size // 1グループのサイズ
		   ) ;
  const size_t bytes = sizeof(double);
  double res_hi,res_lo;
  q_[0].wait();
  q_[0].memcpy(&res_hi , ret_buf_hi, bytes);
  q_[0].memcpy(&res_lo , ret_buf_lo, bytes);
  q_[0].wait();
  ddtype res(res_hi,res_lo);

  return res;
#endif

}

ddtype dddot(
	    PARCEL_INT N,
	    double *x_hi,double *x_lo,PARCEL_INT ldx,
	    double *y_hi,double *y_lo,PARCEL_INT ldy,
	    double *work // 作業領域 : (num_work_groups + 1) * 2
	    ){
  //      printf(" dddot_reduction dd \n");

#if dddot_reduction

#ifdef HIPSYCL_SYCL_HPP
  // hipSYCL 
  size_t MAX = N;
  auto ret_ = sycl::malloc_device<ddtype>(1,q_[0]);
  sycl::buffer<ddtype> buff(&ret_[0],1);
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor sum{buff, h};
      h.parallel_for(sycl::range<1>{MAX},
		     sycl::reduction(sum,sycl::plus< ddtype >()),
		     [=](sycl::id<1> i, auto &sum_){
		       ddtype x_(x_hi[i],x_lo[i]);
		       ddtype y_(y_hi[i],y_lo[i]);
		       sum_ += x_*y_;
		     });
    });
  q_[0].wait();

  ddtype ret(0.0,0.0);
  q_[0].memcpy(&ret, ret_,  sizeof(ddtype));
  q_[0].wait();
  sycl::free(ret_ , q_[0]);

  return ret;
#else
  // コンパイルは通るが実行時に失敗する
  size_t MAX = N;
  ddtype ret_(0.0,0.0);
  sycl::buffer<ddtype> buff(&ret_,1);
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      //	  sycl::accessor sum{buff, h};
      h.parallel_for(sycl::range<1>{MAX},
		     //			 sycl::reduction(buff, h, std::plus<ddtype>()
		     sycl::reduction(buff, h, sycl::plus< ddtype >(),
				     sycl::property::reduction::initialize_to_identity{}
				     ),
		     [=](sycl::id<1> i,auto &sum){
		       ddtype x_(x_hi[i],x_lo[i]);
		       ddtype y_(y_hi[i],y_lo[i]);
		       sum += x_*y_;
		     });
    });
  q_[0].wait();
  //      sycl::host_accessor result{buff, sycl::read_only};
  //  ddtype ret = result[0];
  ddtype ret(0.0,0.0);

  return ret;

  // size_t MAX = N;
  // auto ret_ = sycl::malloc_device<ddtype>(1,q_[0]);
  // sycl::buffer<ddtype> buff(&ret_[0],1);
  // q_[0].submit([&] (sycl::handler &h){
  // 	  sycl::accessor sum{buff, h};
  // 	  h.parallel_for(sycl::range<1>{MAX},
  // 			 sycl::reduction(sum,sycl::plus< ddtype >()),
  // 			 [=](sycl::id<1> i, auto &sum_){
  // 			   ddtype x_(x_hi[i],x_lo[i]);
  // 			   ddtype y_(y_hi[i],y_lo[i]);
  // 			   sum_ += x_*y_;
  // 			 });
  // 	});
  // q_[0].wait();

  //      ddtype ret(0.0,0.0);
  // q_[0].memcpy(&ret, ret_,  sizeof(ddtype));
  // q_[0].wait();
  // sycl::free(ret_ , q_[0]);

  //      return ret;
#endif

#endif


#if dddot_cpu
  //      #if 1
      
  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  double outbuf_hi_[num_work_groups];
  double outbuf_lo_[num_work_groups];
  sycl::buffer<double> outbuf_hi(outbuf_hi_,num_work_groups);
  sycl::buffer<double> outbuf_lo(outbuf_lo_,num_work_groups);

  ddmath dd;
  q_[0].wait();      
  q_[0].submit([&] (sycl::handler &h){
      sycl::accessor<
	double,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch_hi(work_group_size, h); 
      sycl::accessor<
	double,1,
	sycl::access::mode::read_write,
	sycl::access::target::local> scratch_lo(work_group_size, h); 

      auto out_hi = outbuf_hi.get_access(h);	  
      auto out_lo = outbuf_lo.get_access(h);	  
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id     = item.get_local_id(0);
		       size_t group_id     = item.get_group(0);

		       scratch_hi[local_id] = 0.0;
		       scratch_lo[local_id] = 0.0;
		       ddtype scratch(0.0,0.0);
		       for(size_t i = global_id; i<N;i=i+num_work_items){
			 ddtype x_(x_hi[i],x_lo[i]);
			 ddtype y_(y_hi[i],y_lo[i]);
			 ddtype dd_xy  = x_ * y_;
			 ddtype dd_sum = scratch + dd_xy;
			 scratch = dd_sum;
		       }
		       scratch_hi[local_id] = scratch.get_hi();
		       scratch_lo[local_id] = scratch.get_lo();

		       // Do a tree reduction on items in work-group
		       for(size_t i = work_group_size / 2; i > 0; i >>= 1) {
			 item.barrier(sycl::access::fence_space::local_space);
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id],scratch_lo[local_id]);
			   ddtype dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
			   ddtype dd_sum = dd_lid + dd_i;
			   scratch_hi[local_id] = dd_sum.get_hi();
			   scratch_lo[local_id] = dd_sum.get_lo();
			 }
		       }
		       if (local_id == 0) {
			 out_hi[group_id] = scratch_hi[0];
			 out_lo[group_id] = scratch_lo[0];
		       }

		     });
    });
  q_[0].wait();

  //      auto result = outbuf.get_access();
  sycl::host_accessor result_hi{outbuf_hi,sycl::read_only};
  sycl::host_accessor result_lo{outbuf_lo,sycl::read_only};

  ddtype ret(0.0,0.0);
  for(size_t i=0;i<num_work_groups;i++){
    ddtype dd_out(result_hi[i],result_lo[i]);
    ret = ret + dd_out;
  }

  return ret;
#endif

#if dddot_tree
  //     #if 1

  const size_t work_group_size = q_[0].get_device().get_info<sycl::info::device::max_work_group_size>(); // localグループ数
  size_t num_work_groups = num_work_groups_;
  if(num_work_groups<work_group_size){
    num_work_groups = 1;
  }
  const size_t num_work_items  = work_group_size*num_work_groups; // 全並列数
  double *ret_buf_hi = &work[0]; 
  double *ret_buf_lo = &work[1];
  double *outbuf_hi  = &work[2];
  double *outbuf_lo  = &work[num_work_groups+2];

  treeDddot_(
	    q_,
	    x_hi,x_lo,
	    y_hi,y_lo,
	    outbuf_hi,outbuf_lo,
	    N,
	    num_work_groups, // グループ数
	    work_group_size // 1グループのサイズ
	    ) ;

  treeDDReduction_(
		   q_,
		   outbuf_hi,outbuf_lo,
		   ret_buf_hi,ret_buf_lo,
		   num_work_groups,
		   1, // グループ数
		   work_group_size // 1グループのサイズ
		   ) ;
  const size_t bytes = sizeof(double);
  double res_hi,res_lo;
  q_[0].memcpy(&res_hi , ret_buf_hi, bytes);
  q_[0].memcpy(&res_lo , ret_buf_lo, bytes);
  q_[0].wait();
  ddtype res(res_hi,res_lo);

  return res;
#endif

}
#endif
        
#endif
