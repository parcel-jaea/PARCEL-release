#ifndef UTIL_SYCL_HPP_
#define UTIL_SYCL_HPP_

#ifdef HIPSYCL_SYCL_HPP
using namespace cl;
#endif

#include <CL/sycl.hpp>

namespace util_sycl{

  template<class T>
  class vector{
    T* mem_;
    size_t n_;
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};
    bool owner_ = true;
  public:

     void shallowcp(T* mem, size_t n, sycl::queue* q){
       mem_ = mem;
       n_ = n;
       q_ = q;
       owner_ = false;
     }

     vector(T* mem, size_t n, sycl::queue* q){
       shallowcp(mem,n,q);
     }

    // vector(util_sycl::vector &vec){
    //   mem_ = vec.data();
    //   n_   = vec.size();
    //   q_   = vec.queue();
    // }

    vector(size_t n, sycl::queue* q){
      q_ = q;
      n_ = n>1? n:1;
      mem_ = sycl::malloc_device<T>(n,q_[0]);
    }
    vector(){
      q_ = &q_default;
      n_ = 1;
      mem_ = sycl::malloc_device<T>(1,q_[0]);
    }
    ~vector(){
      q_[0].wait();
      if(owner_ == true){
	sycl::free(mem_, q_[0]);
      }
      q_[0].wait();
    }

    void resize(size_t n, sycl::queue* q){
      q_[0].wait();
      sycl::free(mem_, q_[0]);
      q_ = q;
      n_ = n>1? n:1;
      mem_ = sycl::malloc_device<T>(n,q_[0]);
    }

    void resize(size_t n){
      sycl::free(mem_, q_[0]);
      n_ = n>1? n:1;
      mem_ = sycl::malloc_device<T>(n,q_[0]);
    }

    sycl::queue *queue(){
      return q_;
    }

    T *data(){
      return mem_;
    }

    T *at(size_t i){
      return &mem_[i];
    }

    size_t size(){
      return n_;
    }

    void setValue(T *mem){
      const size_t bytes = n_ * sizeof(T);
      q_[0].memcpy(mem_, mem, bytes);
      q_[0].wait();
    }

    void getValue(T *mem){
      const size_t bytes = n_ * sizeof(T);
      q_[0].memcpy(mem, mem_, bytes);
      q_[0].wait();
    }

    template<class TT>
    void getValue(TT *mem,size_t n){
      const size_t bytes = n * sizeof(TT);
      q_[0].memcpy(mem, mem_, bytes);
      q_[0].wait();
    }
    template<class TT>
    void setValue(TT *mem,size_t n){
      const size_t bytes = n * sizeof(TT);
      q_[0].memcpy(mem_, mem, bytes);
      q_[0].wait();
    }

    void setVec(sycl::queue* q, T* mem, size_t n){
      sycl::free(mem_, q_[0]);
      mem_ = mem;
      n_ = n;
      q_ = q;
      return ;
    }

  };

}

#endif
