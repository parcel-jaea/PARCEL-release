#ifndef MIXED_PRECISON_HPP_
#define MIXED_PRECISON_HPP_

#include <CL/sycl.hpp>
#include "parcel.hpp"
#include "ddlib.hpp"

#include "parcel_crs.hpp"

// #if _A64FX_HIPSYCL_
// #define CRS_SPMV_NBLOCK 12
// #define CRS_SPMV_BLOCKSIZE 4096
// #else
// #define CRS_SPMV_NBLOCK    N
// #define CRS_SPMV_BLOCKSIZE 1
// #endif

namespace mixed_precison{
  using ddtype = ddlib::ddtype;
  using ddmath = ddlib::ddmath;

#include "mixed_precision_trtri.hpp"

  class mixed_precison_blas{
    sycl::queue *q_;
    size_t num_work_groups_ = 2048;

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
