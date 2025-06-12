#ifndef PARCEL_HPP_
#define PARCEL_HPP_

// #define PARCEL_INT int64_t
// #define PARCEL_INT int32_t
// #define PARCEL_INT int

#ifndef PARCEL_INT
#define PARCEL_INT int32_t
#else
#define PARCEL_INT int64_t
#endif

#include "util_sycl.hpp"

namespace parcel{

  // -------------------------------
  template<class T>
  class floatingtype{
    util_sycl::vector<T> vec;
  public:
    void resize(size_t nnz, sycl::queue *q){
      vec.resize(nnz,q);
      return ;
    }
    void setValue(T *mem){
      vec.setValue(mem);
    }
    T *data(){
      return vec.data();
    }

  };

  class doubledouble{
    util_sycl::vector<double> vec_hi;
    util_sycl::vector<double> vec_lo;
  public:
    void resize(size_t nnz, sycl::queue *q){
      vec_hi.resize(nnz,q);
      vec_lo.resize(nnz,q);
      return ;
    }
    void setValue_hi(double *mem){
      vec_hi.setValue(mem);
    }
    void setValue_lo(double *mem){
      vec_lo.setValue(mem);
    }
    double *hi_data(){
      return vec_hi.data();
    }
    double *lo_data(){
      return vec_lo.data();
    }
  };

  template<class parceltype,class ID>
  class crsMat{
    parceltype mat;
    util_sycl::vector<ID> rowptr;
    util_sycl::vector<ID> crsCol;
    size_t n_;
    size_t nnz_;
    sycl::queue* q_;
    sycl::queue q_default{sycl::default_selector{}};
  public:

    void resize(size_t n,size_t nnz, sycl::queue* q){
      q_ = q;
      n_   = n>1? n:1;
      nnz_ = nnz>1? nnz:1;
      mat.resize(nnz_ , q_);
      rowptr.resize(n_+1 , q_);
      crsCol.resize(nnz_ , q_);
    }

    crsMat(){}
    crsMat(size_t n, size_t nnz,sycl::queue* q){
       resize(n,nnz,q);
    }
    ~crsMat(){
      q_[0].wait();
    }

    sycl::queue *queue(){
      return q_;
    }

    auto *mat_data(){
      return mat.data();
    }

    auto *mat_hi_data(){
      return mat.hi_data();
    }
    auto *mat_lo_data(){
      return mat.lo_data();
    }

    template<class Ttype>
    void mat_setValue(Ttype *mem){
      mat.setValue(mem);
    }
    void mat_setValue_hi(double *mem){
      mat.setValue_hi(mem);
    }
    void mat_setValue_lo(double *mem){
      mat.setValue_lo(mem);
    }

    ID *rowptr_data(){
      return rowptr.data();
    }

    ID *crsCol_data(){
      return crsCol.data();
    }
    void rowptr_setValue(ID *mem){
      rowptr.setValue(mem);
    }
    void crsCol_setValue(ID *mem){
      crsCol.setValue(mem);
    }
  };

}

using parcel_double     = parcel::floatingtype<double>;
using parcel_float      = parcel::floatingtype<float>;
using parcel_longdouble = parcel::floatingtype<long double>;
using parcel_fp16       = parcel::floatingtype<sycl::float16>;

#endif
