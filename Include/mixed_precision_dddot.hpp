// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDDOT_HPP_
#define MIXED_PRECISON_DDDOT_HPP_

#if PARCEL_BLOCKED_DOT
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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
			 if (local_id < i){
			   ddtype dd_lid(scratch_hi[local_id],scratch_lo[local_id]);
			   ddtype dd_i(scratch_hi[local_id + i],scratch_lo[local_id + i]);
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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
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

  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  // グループごとの部分和 (hi / lo) をデバイス上の領域に書き、ホスト側の固定領域 (malloc_host) へ
  // 1 回コピーする。どちらも dot と同じ作業域を使い回す。先頭 num_work_groups 個が hi、続きが lo。
  double *out_dev  = dot_work_<double>(dot_work_dev_ , 2*num_work_groups, false);
  double *out_host = dot_work_<double>(dot_work_host_, 2*num_work_groups, true);

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

      double *out_hi = out_dev;
      double *out_lo = out_dev + num_work_groups;
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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
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
  q_[0].memcpy(out_host, out_dev, sizeof(double)*2*num_work_groups);
  q_[0].wait();

  const double *result_hi = out_host;
  const double *result_lo = out_host + num_work_groups;
  // group 0 から順に足す

  ddtype ret(0.0,0.0);
  for(size_t i=0;i<num_work_groups;i++){
    ddtype dd_out(result_hi[i],result_lo[i]);
    ret = ret + dd_out;
  }

  return ret;



}

ddtype dddot(
	    PARCEL_INT N,
	    double *x_hi,double *x_lo,PARCEL_INT ldx,
	    double *y_hi,double *y_lo,PARCEL_INT ldy,
	    double *work // 作業領域 : (num_work_groups + 1) * 2
	    ){



      
  const size_t work_group_size = 512; // simd lengh
  const size_t num_work_groups = 512; // cpu core
  const size_t num_work_items = num_work_groups * work_group_size; // 全並列数

  // グループごとの部分和 (hi / lo) をデバイス上の領域に書き、ホスト側の固定領域 (malloc_host) へ
  // 1 回コピーする。どちらも dot と同じ作業域を使い回す。先頭 num_work_groups 個が hi、続きが lo。
  double *out_dev  = dot_work_<double>(dot_work_dev_ , 2*num_work_groups, false);
  double *out_host = dot_work_<double>(dot_work_host_, 2*num_work_groups, true);

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

      double *out_hi = out_dev;
      double *out_lo = out_dev + num_work_groups;
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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
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
  q_[0].memcpy(out_host, out_dev, sizeof(double)*2*num_work_groups);
  q_[0].wait();

  const double *result_hi = out_host;
  const double *result_lo = out_host + num_work_groups;
  // group 0 から順に足す

  ddtype ret(0.0,0.0);
  for(size_t i=0;i<num_work_groups;i++){
    ddtype dd_out(result_hi[i],result_lo[i]);
    ret = ret + dd_out;
  }

  return ret;


}
#endif
        
#endif
