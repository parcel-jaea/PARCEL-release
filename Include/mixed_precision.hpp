// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_HPP_
#define MIXED_PRECISON_HPP_

#include <CL/sycl.hpp>
#include <memory>

#include "parcel.hpp"
#include "ddlib.hpp"

#include "parcel_crs.hpp"

namespace mixed_precison{
  using ddtype = ddlib::ddtype;
  using ddmath = ddlib::ddmath;

#include "mixed_precision_trtri.hpp"

  class mixed_precison_blas{
    sycl::queue *q_;
    size_t num_work_groups_ = 2048;

    // 内積などの作業域。
    //   内積は pgmres の MGS などで 1 反復に何十回も呼ばれるので、呼ぶたびの確保・解放の固定費を避けるため、
    //   一度確保した領域を使い回す。コピーされたオブジェクトとは shared_ptr で共有し、
    //   解放は最後の持ち主が確保時のコンテキストで行う。
    struct usm_work_{
      std::shared_ptr<void> p;
      size_t bytes = 0;
      sycl::queue *q = nullptr;
    };
    usm_work_ dot_work_dev_;
    usm_work_ dot_work_host_;
    // Gram 行列の gemm (mixed_precision_gemm.hpp) の部分和の作業域 (m*m*グループ数)。大きくなる方向にだけ確保し直す
    usm_work_ gemm_work_dev_;

    template<class T>
    T *dot_work_(usm_work_ &w, size_t count, bool host){
      const size_t bytes = count*sizeof(T);
      if(!w.p || w.bytes < bytes || w.q != q_){
	q_[0].wait();
	sycl::context ctx = q_[0].get_context();
	void *raw = host ? (void *)sycl::malloc_host(bytes, q_[0]) : (void *)sycl::malloc_device(bytes, q_[0]);
	if(raw == nullptr){
	  throw util_sycl::device_bad_alloc(std::string("PARCEL: dot work allocation failed: ")
					    + std::to_string((unsigned long long)bytes) + " bytes");
	}
	if(!host) util_sycl::track_device_alloc(raw, bytes);
	w.p = std::shared_ptr<void>(raw, [ctx](void *ptr){ util_sycl::free_tracked(ptr, ctx); });
	w.bytes = bytes;
	w.q = q_;
      }
      return (T *)w.p.get();
    }

  public:
    mixed_precison_blas(sycl::queue *q){
      q_ = q;
    }
    mixed_precison_blas(){}
    ~mixed_precison_blas(){}

    size_t bufsize(){
      size_t bufsize_ = (num_work_groups_ + 1) * 2;
      return bufsize_;
    };

    void set_num_work_groups(size_t num_work_groups){
      num_work_groups_ = num_work_groups;
    }

    void reset(sycl::queue *q){
      q_[0].wait();
      q_ = q;
    }

    void wait(){
      q_[0].wait();
    }

    void setQueue(sycl::queue *q){
      q_=q;
    }

    sycl::queue *queue(){
      return q_;
    }

#include "mixed_precision_max.hpp"

#include "mixed_precision_dot.hpp"
#include "mixed_precision_dddot.hpp"

#include "mixed_precision_axpy.hpp"
#include "mixed_precision_ddaxpy.hpp"

#include "mixed_precision_gemv.hpp"
#include "mixed_precision_ddgemv.hpp"

#include "mixed_precision_gemm.hpp"
#include "mixed_precision_ddgemm.hpp"

#include "mixed_precision_crsmv.hpp"
#include "mixed_precision_ddcrsmv.hpp"

#include "mixed_precision_diamv.hpp"
#include "mixed_precision_dddiamv.hpp"

  };

  class mixed_precison_func{

  public:
    template<class TdataA,class TdataB,class Tcalc,class Tret>
    Tret add(TdataA a,TdataB b){
      Tcalc a_ = a;
      Tcalc b_ = b;
      Tcalc c_ = a+b;
      Tret c   = c_;
      return c;
    }

  };

}

#endif
