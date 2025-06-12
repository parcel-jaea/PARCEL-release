#ifndef PARCEL_UTIL_COMMON_
#define PARCEL_UTIL_COMMON_

#include <type_traits>
#include "parcel_scalar.hpp"
#include "parcel_type.hpp"
// #include "parcel.hpp"
// #include "parcel_type.hpp"

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
		      PARCEL_INT N,
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
		       PARCEL_INT N,
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
		       PARCEL_INT N,
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
			PARCEL_INT N,
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
      PARCEL_INT n = n_;
      sycl::queue &q = *q_; 
      convert_TtoT<T,T>(q,n,x,y);      
      return ;
    }

    template<class T>
    void setptr(
		PARCEL_INT n,
		vector<T> &x_,
		vector<T> &y_,
		PARCEL_INT offset=0
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

#if 1
    template<class T>
    void copy_at( 
		 vector<T>  &x_,
		 vector<T>  &y_,
		 PARCEL_INT offset_x = 0,
		 PARCEL_INT offset_y = 0
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
#else
    template<class T>
    void convert_at( 
		    parcel::scalar val_,
		    vector<T>  &x_,
		    PARCEL_INT offset = 0
		     ){
      const auto T_name  = typeid(T).name();
      sycl::queue *q_ = x_.queue();
      sycl::queue &q = *q_; 
      vector<T> val(1,q_);

      if( ddtype_name == T_name ){
	double *x_hi = (double *)x_.data_hi();
	double *x_lo = (double *)x_.data_lo();
	double val_hi_ = val_.get_dd_hi();
	double val_lo_ = val_.get_dd_lo();
	val.setValue(&val_hi_,&val_lo_);
	convert_ddtodd(
		       q,
		       1,
		       &x_hi[offset],
		       &x_lo[offset],
		       val_hi,
		       val_lo
		       );
      }else{
	auto *x    = x_.data();
	auto val_T = val_.get_T();
	val.setValue(&val_T);
	convert_ddtodd(
		       q,
		       1,
		       &x,
		       val_T
		       );
      }

      return ;

    }
#endif

    template<class Tin,class Tout>
    void convert(
		 vector<Tin>  &x_,
		 vector<Tout> &y_
		 ){
      const auto Tin_name  = typeid(Tin).name();
      const auto Tout_name = typeid(Tout).name();

      //     std::cout<<"convert "<<Tin_name<<" --> "<<Tout_name<<std::endl;
      //      sycl::queue *q = x_.queue();
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
	  //	  std::cout<<"convert T --> dd"<<std::endl;
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
