// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_MAX_HPP_

#define MIXED_PRECISON_MAX_HPP_

#include "parcel_nblock.hpp"

// 縮約カーネル (dot / dddot / max) のブロック化。dot と同じスイッチで切り替える。
//   平坦版は sycl::buffer を呼び出しごとに生成し nd_range + barrier で
//   ツリー削減するため、問題サイズに依らない固定費が乗る。
//   max はスケーリング有効時 (前処理 2 / 4) に毎反復呼ばれる。
#if PARCEL_BLOCKED_DOT
#include "mixed_precision_max_a64fx.hpp"
#endif

#ifndef MIXED_PRECISON_MAX_A64FX_HPP_

template<class TdataX,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx
	 ){

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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
			 if (local_id < i){
                           scratch[local_id] = scratch[local_id] < scratch[local_id + i] ? scratch[local_id + i] : scratch[local_id];
			 }
		       }
		       if (local_id == 0) {
			 out[group_id] = scratch[0];
		       }

		     });
    });
  q_[0].wait();

  sycl::host_accessor result{outbuf,sycl::read_only};

  Tret ret = 0.0;
  for(size_t i=0;i<num_work_groups;i++){
    ret =  result[i] < ret ? ret : result[i];
  }
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
			 ::util_sycl::group_barrier(item, sycl::access::fence_space::local_space); // 例外を壊さない同期 (util_sycl.hpp)
			 if (local_id < i){
                           scratch[local_id] = scratch[local_id] < scratch[local_id + i] ? scratch[local_id + i] : scratch[local_id];
			 }
		       }
		       if (local_id == 0) {
			 out[group_id] = scratch[0];
		       }

		     });
    });
  q_[0].wait();

  sycl::host_accessor result{outbuf,sycl::read_only};

  Tret ret = 0.0;
  for(size_t i=0;i<num_work_groups;i++){
    ret =  result[i] < ret ? ret : result[i];
  }
  if(ret == 0.0)ret = 1.0;

  return ret;

}

#endif

#endif
