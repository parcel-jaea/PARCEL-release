// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_UTIL_COMMON_
#define PARCEL_UTIL_COMMON_

#include <type_traits>
#include "parcel_scalar.hpp"
#include "parcel_type.hpp"

namespace parcel{
  using ddtype = ddlib::ddtype;
  using ddmath = ddlib::ddmath;

  template<class T=void>
  inline void swap_ptr(T **x,T **y){
    T *tmp;
    tmp=*y;
    *y=*x;
    *x=tmp;
  }

  class vector_util{
  public :

    template<class TdataX,class TdataY>
    void convert_TtoT(
		      sycl::queue &q,
		      size_t N,
		      TdataX *x,
		      TdataY *y
		      ){      

      size_t MAX = N;
      q.wait();
      q.parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	  y[i] = x[i];
	});
      q.wait();

    }

    template<class TdataX>
    void convert_Ttodd(
		       sycl::queue &q,
		       size_t N,
		       TdataX *x,
		       double *y_hi,
		       double *y_lo
		       ){      
      size_t MAX   = N;
      q.wait();
      q.parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	  y_hi[i] = x[i];
	  y_lo[i] = 0.0;
	});
      q.wait();
    }

    template<class TdataY>
    void convert_ddtoT(
		       sycl::queue &q,
		       size_t N,
		       double *x_hi,
		       double *x_lo,
		       TdataY *y
		       ){      
      size_t MAX   = N;
      q.wait();
      q.parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	  y[i] = x_hi[i] + x_lo[i];
	});
      q.wait();
    }

    void convert_ddtodd(
			sycl::queue &q,
			size_t N,
			double *x_hi,
			double *x_lo,
			double *y_hi,
			double *y_lo
			){      
      size_t MAX   = N;
      q.wait();
      q.parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
	  y_hi[i] = x_hi[i];
	  y_lo[i] = x_lo[i];
	});
      q.wait();
    }

    template<class T>
    void copy(
	      vector<T>  &x_,
	      vector<T>  &y_
	      ){
      sycl::queue *q_ = x_.queue();
      sycl::queue &q = *q_; 

      const size_t nx = x_.size();
      const size_t ny = y_.size();
      const size_t n  = std::min(nx,ny);
      T *x = x_.data();
      T *y = y_.data();
      convert_TtoT<T,T>(q,n,x,y);

      return ;
    }

    template<class T>
    void copy(
	      sycl::queue *q_,
	      size_t n_,
	      T *x,
	      T *y
	      ){
      size_t n = n_;
      sycl::queue &q = *q_; 
      convert_TtoT<T,T>(q,n,x,y);      
      return ;
    }

    // 長さ n と先頭 offset は size_t。
    //   基底 (局所行数 x 列数) の列 j の先頭 ld*j は 32 ビットでは 2^31 を超えて桁あふれしうる。
    //   呼び出し側も積は size_t で作ること
    //   ((size_t)ld*j のように掛ける前にキャストする。PARCEL_INT 同士の積は掛けた時点で桁あふれする)。
    template<class T>
    void setptr(
		size_t n,
		vector<T> &x_,
		vector<T> &y_,
		size_t offset=0
		){
      sycl::queue *q_ = x_.queue();
      const size_t nx = n;      
      auto &y_parceltype = *(y_.mem()); // parcel::floatingtype<double>

      const auto T_name  = typeid(T).name();
      if( ddtype_name == T_name ){
	auto *x_hi = x_.data_hi();
	auto *x_lo = x_.data_lo();
	y_parceltype.shallowcp_hi(&x_hi[offset],nx,q_);
	y_parceltype.shallowcp_lo(&x_lo[offset],nx,q_);
      }else{
	auto *x = x_.data();
	y_parceltype.shallowcp(&x[offset],nx,q_);
      }

      return ;

    }

    template<class T>
    void copy_at( 
		 vector<T>  &x_,
		 vector<T>  &y_,
		 size_t offset_x = 0,
		 size_t offset_y = 0
		  ){
      const auto T_name  = typeid(T).name();
      sycl::queue *q_ = x_.queue();
      sycl::queue &q = *q_; 

      if( ddtype_name == T_name ){
	double *x_hi = ( (double *)x_.data_hi() + offset_x );
	double *x_lo = ( (double *)x_.data_lo() + offset_x );
	double *y_hi = ( (double *)y_.data_hi() + offset_y );
	double *y_lo = ( (double *)y_.data_lo() + offset_y );
	convert_ddtodd(q,1,x_hi,x_lo,y_hi,y_lo);
      }else{
	auto *x = ( x_.data() + offset_x );
	auto *y = ( y_.data() + offset_y );

	convert_TtoT<
	  typename std::result_of<parcel::parcel_type_to_baseType( T ) >::type ,
	  typename std::result_of<parcel::parcel_type_to_baseType( T ) >::type 
	  >(q,1,x,y);
      }

      return ;

    }

    template<class Tin,class Tout>
    void convert(
		 vector<Tin>  &x_,
		 vector<Tout> &y_
		 ){
      const auto Tin_name  = typeid(Tin).name();
      const auto Tout_name = typeid(Tout).name();

      sycl::queue *q_ = x_.queue();
      sycl::queue &q = *q_; 

      const size_t nx = x_.size();
      const size_t ny = y_.size();
      const size_t n  = std::min(nx,ny);

      // ------------------------------
      auto ddtodd_T = [&] {
	double *x_hi = (double *)x_.data_hi();
	double *x_lo = (double *)x_.data_lo();
	if( ddtype_name == Tout_name ){
	  // dd --> dd
	  double *y_hi = (double *)y_.data_hi();
	  double *y_lo = (double *)y_.data_lo();
	  convert_ddtodd(q,n,x_hi,x_lo,y_hi,y_lo);
	}else{
	  // dd --> T
	  auto y = y_.data();
	  convert_ddtoT<
	    typename std::result_of< parcel::parcel_type_to_baseType( vector<Tout>) >::type
	    >(q,n,x_hi,x_lo,y);
	}
      };
      auto Ttodd_T = [&] {
	auto x = x_.data();
	if( ddtype_name == Tout_name ){
	  // T --> dd
	  double *y_hi = (double *)y_.data_hi();
	  double *y_lo = (double *)y_.data_lo();
	  convert_Ttodd<
	    typename std::result_of< parcel::parcel_type_to_baseType( vector<Tin>) >::type
	    >(q,n,x,y_hi,y_lo);
	}else{
	  // T --> T
	  auto y = y_.data();
	  convert_TtoT<
	    typename std::result_of< parcel::parcel_type_to_baseType( vector<Tin> ) >::type ,
	    typename std::result_of< parcel::parcel_type_to_baseType( vector<Tout>) >::type
	    >(q,n,x,y);
	}
      };

      if( ddtype_name == Tin_name ){	
	ddtodd_T();
      }else{
	Ttodd_T();
      }

    }

  };


}

#endif
