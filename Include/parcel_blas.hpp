#ifndef PARCEL_BLAS_HPP_
#define PARCEL_BLAS_HPP_

#include <type_traits>
#include <algorithm>
#include "util_sycl.hpp"
#include "mixed_precision.hpp"
#include "parcel.hpp"
#include "parcel_util.hpp"
#include "parcel_time.hpp"

#if PARCEL_MPI
#include <mpi.h>
#include "./parcel_mpi/parcel_mpi.hpp"
#endif

namespace parcel{

  class blas_timer{
    sycl::queue *q_;
    parcel::timer timer_axpy;
    parcel::timer timer_gemv;
    parcel::timer timer_gemm;
    parcel::timer timer_dot;
    parcel::timer timer_SpMV_comm;
    parcel::timer timer_SpMV;

  public :
    blas_timer(){}
    blas_timer(sycl::queue *q){
      q_=q;
    }
    ~blas_timer(){}
    void setQueue(sycl::queue *q){
      q_=q;
    }

    void reset(){
      timer_axpy.reset();
      timer_gemv.reset();
      timer_gemm.reset();
      timer_dot.reset();
      timer_SpMV.reset();      
    }

    void SpMV_start(){q_[0].wait();timer_SpMV.start();}
    void SpMV_end()  {q_[0].wait();timer_SpMV.end();  }

    void SpMV_comm_start(){q_[0].wait();timer_SpMV_comm.start();}
    void SpMV_comm_end()  {q_[0].wait();timer_SpMV_comm.end();  }

    void axpy_start(){q_[0].wait();timer_axpy.start();}
    void axpy_end()  {q_[0].wait();timer_axpy.end();  }

    void gemv_start(){q_[0].wait();timer_gemv.start();}
    void gemv_end()  {q_[0].wait();timer_gemv.end();  }

    void gemm_start(){q_[0].wait();timer_gemm.start();}
    void gemm_end()  {q_[0].wait();timer_gemm.end();  }

    void dot_start(){q_[0].wait();timer_dot.start();}
    void dot_end()  {q_[0].wait();timer_dot.end();  }

    double get_axpy(){
      return timer_axpy.get();
    }
    double get_gemv(){
      return timer_gemv.get();
    }
    double get_gemm(){
      return timer_gemm.get();
    }
    double get_dot(){
      return timer_dot.get();
    }
    double get_SpMV(){
      return timer_SpMV.get();
    }
    double get_SpMV_comm(){
      return timer_SpMV_comm.get();
    }

  };

  // Tは浮動少数点型のreturn型
  template<class Tret=double>
#if PARCEL_MPI
  class blas : public parcel::comm {
#else
  class blas {
#endif
    mixed_precison::mixed_precison_blas blas_;
    util_sycl::vector<double> dwork_;

#if PARCEL_MPI
    parcel::comm comm_;
#endif
    Tret ret_T = 0.0;
    ddtype ret_dd;

    parcel::vector<parcel_double> zero;
    parcel::vector<parcel_double> one;
    parcel::vector<parcel_double> negative_one;

    parcel::vector<parcel_doubledouble> zero_dd;
    parcel::vector<parcel_doubledouble> one_dd;
    parcel::vector<parcel_doubledouble> negative_one_dd;
    //    parcel::vector<parcel_doubledouble> ddwork_;

    parcel::blas_timer blas_timer_;

  public :

    blas(){}
#if PARCEL_MPI
    blas(sycl::queue *q,MPI_Comm comm){
      set_comm(comm);
#else      
    blas(sycl::queue *q){
#endif      
      blas_.setQueue(q);
      blas_timer_.setQueue(q);

      //      ddwork_.resize(10,q);

      zero.resize(1,q);
      one.resize(1,q);
      negative_one.resize(1,q);

      zero_dd.resize(1,q);
      one_dd.resize(1,q);
      negative_one_dd.resize(1,q);

      Tret zero_     = 0.0;
      Tret one_      = 1.0;
      Tret negative_ = -1.0;

      zero.setValue(&zero_);
      one.setValue(&one_);
      negative_one.setValue(&negative_);
      double zero_d_     = 0.0;
      double one_d_      = 1.0;
      double negative_d_ = -1.0;
      zero_dd.setValue(&zero_d_,&zero_d_);
      one_dd.setValue(&one_d_,&zero_d_);
      negative_one_dd.setValue(&negative_d_,&zero_d_);

#if PARCEL_MPI
      comm_.setQueue(q);
#endif
      
    }

    ~blas(){}

#if PARCEL_MPI
    void set_comm(MPI_Comm comm){
      comm_.set_comm(comm);
      return ;
    }
    
    parcel::comm &parcel_comm(){
      return comm_;
    }
    
#endif

    void setQueue(sycl::queue *q){
      blas_.setQueue(q);
      blas_timer_.setQueue(q);

      //      ddwork_.resize(1,q);

      zero.resize(1,q);
      one.resize(1,q);
      negative_one.resize(1,q);

      Tret zero_     = 0.0;
      Tret one_      = 1.0;
      Tret negative_ = -1.0;
      zero.setValue(&zero_);
      one.setValue(&one_);
      negative_one.setValue(&negative_);

    }

    void wait(){
      blas_.wait();
    }
    mixed_precison::mixed_precison_blas &mixed_blas(){
      return blas_;
    }
    sycl::queue *queue(){
      return  blas_.queue();
    }

    void timer_reset(){
      blas_timer_.reset();
    }

    double get_time_axpy(){return blas_timer_.get_axpy();}
    double get_time_gemv(){return blas_timer_.get_gemv();}
    double get_time_gemm(){return blas_timer_.get_gemm();}
    double get_time_dot() {return blas_timer_.get_dot();}
    double get_time_SpMV(){return blas_timer_.get_SpMV();}

    void check_dwork_size(size_t n){
      size_t work_size = dwork_.size();
      if(work_size < n){
	dwork_.resize(n);
	wait();
      }
      return ;
    }

    // -------------------
#if 0
    template<class Tdot> 
    void get_dot(
		 parcel::vector<Tdot> ret
		 ){
      size_t n = ret.size();
      if(n!=1){
	std::cout<<"not correct vector size "<<std::endl;
      }
      const auto Tdot_name = typeid(Tdot).name();

      if( ddtype_name == Tdot_name ){
	double ret_hi = ret_dd.get_hi();
	double ret_lo = ret_dd.get_lo();
	ret.setValue(&ret_hi,&ret_lo);
      }else{
	typename std::result_of<parcel::parcel_type_to_baseType(Tdot) >::type ret_ = ret_T;
	ret.setValue(&ret_);
      }

      return ;
    }

    ddtype get_dot(parcel::vector<parcel_doubledouble> ret_){
      return ret_dd;
    }

    double get_dot(parcel::vector<parcel_double> ret_){
      return  static_cast<double>(ret_T);
    }
  
    float get_dot(parcel::vector<parcel_float> ret_){
      return  static_cast<float>(ret_T);
    }

    long double get_dot(parcel::vector<parcel_longdouble> ret_){
      return static_cast<long double>(ret_T);
    }
 
    sycl::half get_dot(parcel::vector<parcel_fp16> ret_){		  
      return static_cast<sycl::half>(ret_T);
    }

    std::tuple<Tret,ddtype>get_dot(){
      return {ret_T,ret_dd};
    }
#endif

    parcel::scalar get_dot(){
      parcel::scalar ret;
      ret.set_Tdd(ret_T,ret_dd);
      return ret;
    }

    // ---------
    Tret get_T(){
      return ret_T;
    }
    ddtype get_dd(){
      return ret_dd;
    }

    template<class T> T get_T();
    template<> ddtype get_T<ddtype>(){
      return ret_dd;
    }
    template<> double get_T<double>(){
      return  static_cast<double>(ret_T);
    }
    template<> float get_T<float>(){
      return  static_cast<float>(ret_T);
    }
    template<> long double get_T<long double>(){
      return static_cast<long double>(ret_T);
    }
    template<> sycl::half get_T<sycl::half>(){
      return static_cast<sycl::half>(ret_T);
    }
    // ---------

    template<class Tvecx>
    void set_ret(
		 parcel::vector<Tvecx> &x_
		 ){
      size_t n = x_.size();
      if(n!=1){
	std::cout<<"not correct vector size "<<std::endl;
	return ;
      }

      const auto Tvecx_name = typeid(Tvecx).name();
      if( ddtype_name == Tvecx_name ){ 
	double x_hi = 0.0;
	double x_lo = 0.0;
	x_.getValue_hi(&x_hi);
	x_.getValue_lo(&x_lo);
	ret_dd.set_dd(x_hi,x_lo);
	ret_T = x_hi+x_lo;
      }else{
	typename std::result_of<parcel::parcel_type_to_baseType(Tvecx) >::type x;
	x_.getValue(&x);
	// double x_hi = x;
	// ret_dd.set_dd(x_hi,0.0);
	ret_dd.set_dd(x,0.0);
	ret_T = x;
      }
      //      std::cout<<"ret_T  = "<<ret_T<<std::endl;
      //      std::cout<<"ret_dd = "<<double(ret_dd)<<std::endl;
    }

    // -------------------
    
    template<class Tvecx, class Tvecy>
    void dddot_main(
		   parcel::vector<Tvecx> &x_, 
		   parcel::vector<Tvecy> &y_){
      const auto Tvecx_name = typeid(Tvecx).name();
      const auto Tvecy_name = typeid(Tvecy).name();
      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      size_t dwork_size = blas_.bufsize(); 
      check_dwork_size(dwork_size);
      if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
	double *x_hi = (double *)x_.data_hi();
	double *x_lo = (double *)x_.data_lo();
	double *y_hi = (double *)y_.data_hi();
	double *y_lo = (double *)y_.data_lo(); 
	double *dwork = dwork_.data();
	ret_dd = blas_.dddot(n,
			    x_hi,x_lo,nx,
			    y_hi,y_lo,ny,
			    dwork);
#if PARCEL_MPI
	double sbuf_hi = ret_dd.get_hi();
	double sbuf_lo = ret_dd.get_lo();
	double rbuf_hi = 0.0;
	double rbuf_lo = 0.0;
	comm_.mpi_ddallreduce(&sbuf_hi,&sbuf_lo,&rbuf_hi,&rbuf_lo,1,MPI_SUM);
	ret_dd.set_dd(rbuf_hi,rbuf_lo);
#endif
	ret_T = ret_dd.get_T();
      }else{
	auto *x = x_.data();
	auto *y = y_.data();
	double *dwork = dwork_.data();

	ret_dd = blas_.dddot< 
	 typename std::result_of<parcel::parcel_type_to_baseType(Tvecx) >::type,
	 typename std::result_of<parcel::parcel_type_to_baseType(Tvecy) >::type
	  >(n,x,nx,y,ny,dwork);
	
#if PARCEL_MPI
	double sbuf_hi = ret_dd.get_hi();
	double sbuf_lo = ret_dd.get_lo();
	double rbuf_hi = 0.0;
	double rbuf_lo = 0.0;
	comm_.mpi_ddallreduce(&sbuf_hi,&sbuf_lo,&rbuf_hi,&rbuf_lo,  1,MPI_SUM);
	ret_dd.set_dd(rbuf_hi,rbuf_lo);
#endif
	ret_T = ret_dd.get_T();
      }
    }

    template<class Tvecx, class Tvecy, class Tcalc>
    void dot_main( 
		  parcel::vector<Tvecx> &x_, 
		  parcel::vector<Tvecy> &y_){

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      auto x = x_.data();
      auto y = y_.data();
      //      typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	auto
	  ret = 
	  blas_.dot<
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	  > (n,x,nx,y,ny);

	blas_.wait();

#if PARCEL_MPI
	using T_ret_type = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type;
	T_ret_type sbuf = ret;
	T_ret_type rbuf = 0.0;
	comm_.mpi_allreduce<T_ret_type>(&sbuf,&rbuf,1,MPI_SUM); 
	ret = rbuf;
#endif
	
	ret_T = ret;
	ret_dd.set_dd(ret,0.0);
	//      std::cout<<"nx = "<<nx<<", ny = "<<ny<<", n = "<<n<<std::endl;
	//      std::cout<<"dot ret_T = "<<ret_T<<std::endl;

    }

      template<
	class Tvecx,
      class Tvecy,
      class Tcalc >
    //    void dot(Tvecx &x_, Tvecy &y_){
    void dot(
	     parcel::vector<Tvecx> &x_, 
	     parcel::vector<Tvecy> &y_){

      blas_timer_.dot_start();

      const auto Tcalc_name = typeid(Tcalc).name();
      //      std::cout<<"ddtype_name = "<<ddtype_name<<" Tcalc_name = "<<Tcalc_name<<std::endl;
      if( ddtype_name == Tcalc_name ){ 
	dddot_main<Tvecx,Tvecy>(x_,y_);
	// std::cout<<"dddot = "<<ret_T<<" : "<<ret_dd.get_hi()<<" + "<<ret_dd.get_lo()<<std::endl;
	// std::cout<<"dddot = "<<ret_dd.get_hi()<<" + "<<ret_dd.get_lo()<<std::endl;
	//	return ret_dd;
      }else{
	dot_main<Tvecx,Tvecy,Tcalc>(x_,y_);	
	//	std::cout<<"dot = "<<ret_T<<std::endl;
	//	return ret_T;
      }

      blas_timer_.dot_end();
    }

#if 0
    template<
      class Tvecx,
      class Tvecy,
      class Tcalc >
    void norm(
	     parcel::vector<Tvecx> &x_, 
	     parcel::vector<Tvecy> &y_){
      dot<Tvecx,Tvecy,Tcalc>(x_,y_);
    }
#endif

    // ---------------------
    template<class Talpha,class Tbeta,class Tvecx, class Tvecy>
    void ddaxpy_main(
		     parcel::vector<Talpha> &alpha_,
		     parcel::vector<Tbeta>  &beta_,
		     parcel::vector<Tvecx>  &x_, 
		     parcel::vector<Tvecy>  &y_){
      const auto Tvecx_name = typeid(Tvecx).name();
      const auto Tvecy_name = typeid(Tvecy).name();
      const auto Talpha_name = typeid(Talpha).name();
      const auto Tbeta_name = typeid(Tbeta).name();
      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
	double *x_hi = (double *)x_.data_hi();
	double *x_lo = (double *)x_.data_lo();
	double *y_hi = (double *)y_.data_hi();
	double *y_lo = (double *)y_.data_lo(); 

	if( ddtype_name == Talpha_name ){ 
	  double *alpha_hi_ = (double *)alpha_.data_hi(); 
	  double *alpha_lo_ = (double *)alpha_.data_lo(); 
	  double *beta_hi_  = (double *)beta_.data_hi(); 
	  double *beta_lo_  = (double *)beta_.data_lo(); 
	  blas_.ddaxpy(
		       n,
		       alpha_hi_, alpha_lo_,
		       beta_hi_ , beta_lo_,
		       x_hi,x_lo,
		       nx,
		       y_hi,y_lo,
		       ny);
	}else{
	  double *alpha_hi = (double *)alpha_.data(); 
	  double *alpha_lo = (double *)zero.data();
	  double *beta_hi  = (double *)beta_.data(); 
	  double *beta_lo  = (double *)zero.data();
	  blas_.ddaxpy(
		       n,
		       alpha_hi, alpha_lo,
		       beta_hi , beta_lo,
		       x_hi,x_lo,
		       nx,
		       y_hi,y_lo,
		       ny);
	}
      }else{
	auto x = x_.data();
	auto y = y_.data();

	if( ddtype_name == Talpha_name ){ 
	  double *alpha_hi = (double *)alpha_.data_hi(); 
	  double *alpha_lo = (double *)alpha_.data_lo(); 

	  double *beta_hi = (double *)beta_.data_hi(); 
	  double *beta_lo = (double *)beta_.data_lo(); 

	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	    >(n,alpha_hi,alpha_lo,beta_hi,beta_lo,x,nx,y,ny);
	  
	}else{
	  auto *alpha = alpha_.data(); 	  	  
	  auto *beta = beta_.data(); 	  	  
	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	    >(n,alpha,beta,x,nx,y,ny);
	}

      }
    }

    template<class Talpha,class Tbeta,class Tvecx, class Tvecy, class Tvecz>
    void ddaxpy_main(
		     parcel::vector<Talpha> &alpha_,
		     parcel::vector<Tbeta>  &beta_,
		     parcel::vector<Tvecx>  &x_, 
		     parcel::vector<Tvecy>  &y_,
		     parcel::vector<Tvecy>  &z_){

      // (Talpha == Tbeta && Tvecx == Tvecy == Tvecz) が条件
      const auto Tvecx_name = typeid(Tvecx).name();
      const auto Tvecy_name = typeid(Tvecy).name();
      const auto Talpha_name = typeid(Talpha).name();
      const auto Tbeta_name  = typeid(Tbeta).name();

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t nz = z_.size(); 
      size_t n  = std::min(nx,ny);

      if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
	double *x_hi = (double *)x_.data_hi();
	double *x_lo = (double *)x_.data_lo();
	double *y_hi = (double *)y_.data_hi();
	double *y_lo = (double *)y_.data_lo();
	double *z_hi = (double *)z_.data_hi();
	double *z_lo = (double *)z_.data_lo();
	if( ddtype_name == Talpha_name ){ 
	  double *alpha_hi_ = (double *)alpha_.data_hi(); 
	  double *alpha_lo_ = (double *)alpha_.data_lo(); 
	  double *beta_hi_  = (double *)beta_.data_hi(); 
	  double *beta_lo_  = (double *)beta_.data_lo(); 
#if _A64FX_HIPSYCL_
	  //#if 0
        std::cout<<" not support : parcel_blas.hpp : ddaxpy "<<std::endl;	
#else
	  blas_.ddaxpy(
		       n,
		       alpha_hi_, alpha_lo_,
		       beta_hi_ , beta_lo_,
		       x_hi,x_lo,
		       nx,
		       y_hi,y_lo,
		       ny,
		       z_hi,z_lo,
		       nz
		       );
#endif
	}else{
	  double *alpha_hi = (double *)alpha_.data(); 
	  double *alpha_lo = (double *)zero.data();
	  double *beta_hi  = (double *)beta_.data(); 
	  double *beta_lo  = (double *)zero.data();
#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	  blas_.ddaxpy(
		       n,
		       alpha_hi, alpha_lo,
		       beta_hi , beta_lo,
		       x_hi,x_lo,
		       nx,
		       y_hi,y_lo,
		       ny,
		       z_hi,z_lo,
		       nz
		       );
#endif
	}
      }else{
	auto x = x_.data();
	auto y = y_.data();
	auto z = z_.data();

	if( ddtype_name == Talpha_name ){ 
	  double *alpha_hi = (double *)alpha_.data_hi(); 
	  double *alpha_lo = (double *)alpha_.data_lo(); 

	  double *beta_hi = (double *)beta_.data_hi(); 
	  double *beta_lo = (double *)beta_.data_lo(); 

#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	    >(n,alpha_hi,alpha_lo,beta_hi,beta_lo,x,nx,y,ny,z,nz);
#endif
	  
	}else{
	  auto *alpha = alpha_.data(); 	  	  
	  auto *beta = beta_.data(); 	  	  
#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	    >(n,alpha,beta,x,nx,y,ny,z,nz);
#endif	  
	}

      }
    }

    template<class Talpha, class Tbeta, class Tvecx, class Tvecy, class Tcalc>
    void axpy_main(
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<Tvecx>  &x_, 
		   parcel::vector<Tvecy>  &y_){

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      auto x = x_.data();
      auto y = y_.data();
      auto alpha = alpha_.data();
      auto beta  = beta_.data();

      blas_.axpy<
      	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	>(n,alpha,beta,x,nx,y,ny);

    }

    template<class Talpha, class Tbeta, class Tvecx, class Tvecy, class Tvecz, class Tcalc>
    void axpy_main(
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<Tvecx>  &x_, 
		   parcel::vector<Tvecy>  &y_,
		   parcel::vector<Tvecz>  &z_){

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t nz = z_.size(); 
      size_t n  = std::min(nx,ny);

      auto x = x_.data();
      auto y = y_.data();
      auto z = z_.data();
      auto alpha = alpha_.data();
      auto beta  = beta_.data();

      blas_.axpy<
      	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	>(n,alpha,beta,x,nx,y,ny,z,nz);
    }

    template<class Talpha, class Tbeta, class TmatA, class Tvecx, class Tvecy, class Tcalc>
    void gemv_main(
		   PARCEL_INT m,
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<TmatA>  &A_, 
		   parcel::vector<Tvecx>  &x_, 
		   parcel::vector<Tvecy>  &y_
		   ){

      // size_t nx = x_.size();
      size_t nx = m; 
      size_t ny = y_.size(); 
      size_t n   = ny;
      size_t lda = n;

      auto A = A_.data();
      auto x = x_.data();
      auto y = y_.data();
      auto alpha = alpha_.data();
      auto beta  = beta_.data();

      blas_.gemv<
      	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	>(n,m,alpha,beta,A,lda,x,nx,y,ny);
    }

    // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
    template<
      class Talpha, class Tbeta, 
      class TmatA, class TmatB, class TmatC, 
      class Tcalc>
    void ddgemm_main(
		   PARCEL_INT m,
		   PARCEL_INT n,
		   PARCEL_INT k,
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<TmatA>  &A_,PARCEL_INT ldA,
		   parcel::vector<TmatB>  &B_,PARCEL_INT ldB, 
		   parcel::vector<TmatC>  &C_,PARCEL_INT ldC,
		   std::string &TransA,
		   std::string &TransB,
		   std::string &TransC,
		   std::string &sym
		   ){
      const auto Talpha_name = typeid(Talpha).name();
      const auto Tbeta_name = typeid(Tbeta).name();
      const auto TmatA_name = typeid(TmatA).name();
      const auto TmatB_name = typeid(TmatB).name();
      const auto TmatC_name = typeid(TmatC).name();
#if 1
#include "make_parcel_blas_ddgemm_py.hpp"            
#else
      if( (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) ){
	// alpha,beta :: dd
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();
	if( ddtype_name == TmatA_name ){
	  // A : dd
	  double *A_hi = (double *)A_.data_hi();
	  double *A_lo = (double *)A_.data_lo();
	  if( ddtype_name == TmatB_name ){
	    // B : dd
	    double *B_hi = (double *)B_.data_hi();
	    double *B_lo = (double *)B_.data_lo();
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	      double *C_hi = (double *)C_.data_hi();
	      double *C_lo = (double *)C_.data_lo();
	      if( TransA == "T" ) {
		if( sym == "S" ) {
		  blas_.ddgemmTNsym(
		     m,n,k,
		     alpha_hi,alpha_lo,
		     beta_hi,beta_lo,
		     A_hi,A_lo,ldA,		     
		     B_hi,B_lo,ldB,
		     C_hi,C_lo,ldC);
		}else{
		  blas_.ddgemmTN(
		     m,n,k,
		     alpha_hi,alpha_lo,
		     beta_hi,beta_lo,
		     A_hi,A_lo,ldA,		     
		     B_hi,B_lo,ldB,
		     C_hi,C_lo,ldC);
		}
	      }else{
		blas_.ddgemm(
		     m,n,k,
		     alpha_hi,alpha_lo,
		     beta_hi,beta_lo,
		     A_hi,A_lo,ldA,		     
		     B_hi,B_lo,ldB,
		     C_hi,C_lo,ldC);
	      }
	    }else{
	      // C : d
	    }
	  }else{
	    // B : d
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }
	}else{
	  // A : d
	  if( ddtype_name == TmatB_name ){
	    // B : dd
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }else{
	    // B : d
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }
	}
      }else{
	// alpha,beta :: d 
	if( ddtype_name == TmatA_name ){
	  // A : dd
	  if( ddtype_name == TmatB_name ){
	    // B : dd
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	      
	    }else{
	      // C : d
	    }
	  }else{
	    // B : d
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }
	}else{
	  // A : d
	  if( ddtype_name == TmatB_name ){
	    // B : dd
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }else{
	    // B : d
	    if( ddtype_name == TmatC_name ){
	      // C : dd
	    }else{
	      // C : d
	    }
	  }
	}
      }
#endif            
      return ;
      
    }
    
    // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
    template<
      class Talpha, class Tbeta, 
      class TmatA, class TmatB, class TmatC, 
      class Tcalc>
    void gemm_main(
		   PARCEL_INT m,
		   PARCEL_INT n,
		   PARCEL_INT k,
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<TmatA>  &A_,PARCEL_INT ldA,
		   parcel::vector<TmatB>  &B_,PARCEL_INT ldB, 
		   parcel::vector<TmatC>  &C_,PARCEL_INT ldC,
		   std::string &TransA,
		   std::string &TransB,
		   std::string &TransC,
		   std::string &sym
		   ){

      auto A = A_.data();
      auto B = B_.data();
      auto C = C_.data();
      auto alpha = alpha_.data();
      auto beta  = beta_.data();
      
      if( TransA == "T" ){
	if( sym == "S" ) {
	  blas_.gemmTNsym<
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	    >(m,n,k,alpha,beta,
	      A,ldA,
	      B,ldB,
	      C,ldC
	      );
	}else{
	  blas_.gemmTN<
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	    >(m,n,k,alpha,beta,
	      A,ldA,
	      B,ldB,
	      C,ldC
	      );
	}
      }else{
	blas_.gemm<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	  >(m,n,k,alpha,beta,
	    A,ldA,
	    B,ldB,
	    C,ldC );
      }
    }


    template<class Talpha, class Tbeta, class TmatA, class Tvecx, class Tvecy>
    void ddgemv_main(
		   PARCEL_INT m,
		   parcel::vector<Talpha> &alpha_, 
		   parcel::vector<Tbeta>  &beta_, 
		   parcel::vector<TmatA>  &A_, 
		   parcel::vector<Tvecx>  &x_, 
		   parcel::vector<Tvecy>  &y_
		   ){
      const auto Talpha_name = typeid(Talpha).name();
      const auto Tbeta_name = typeid(Tbeta).name();
      const auto TmatA_name = typeid(TmatA).name();
      const auto Tvecx_name = typeid(Tvecx).name();
      const auto Tvecy_name = typeid(Tvecy).name();

      //      size_t nx = x_.size(); 
      size_t nx = m; 
      size_t ny = y_.size(); 
      size_t n   = ny;
      size_t lda = n;


#if 1
      if( (ddtype_name == Talpha_name) && ( Tvecx_name == Tbeta_name) ){ 	
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();
	if( (ddtype_name == TmatA_name) && ( Tvecx_name == Tvecy_name) ){ 	
	  double *A_hi = (double *)A_.data_hi();
	  double *A_lo = (double *)A_.data_lo();
	  double *y_hi = (double *)y_.data_hi();
	  double *y_lo = (double *)y_.data_lo();
	  if( ddtype_name == Tvecx_name) { 	
	    auto x = x_.data();
	    // std::cout<<" ddgemv @ 1 @"<<std::endl;
	    blas_.ddgemv<
	      bool,
	      bool,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	      bool
	      >(n,m,alpha_hi,alpha_lo,beta_hi,beta_lo,A_hi,A_lo,lda,x,nx,y_hi,y_lo,ny);
	  }else{
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    // std::cout<<" ddgemv @ 2 @"<<std::endl;
	    blas_.ddgemv(n,m,alpha_hi,alpha_lo,beta_hi,beta_lo,A_hi,A_lo,lda,x_hi,x_lo,nx,y_hi,y_lo,ny);
	  }
	}else{
	  auto A = A_.data();
	  auto x = x_.data();
	  auto y = y_.data();
	  if( ddtype_name == Tvecx_name) { 	
	    auto x = x_.data();
	    // std::cout<<" ddgemv @ 3 @"<<std::endl;
	    blas_.ddgemv<
	      bool,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	      >(n,m,alpha_hi,alpha_lo,beta_hi,beta_lo,A,lda,x,nx,y,ny);
	  }else{
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    // std::cout<<" ddgemv @ 4 @"<<std::endl;
	    blas_.ddgemv<
	      bool,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	      >(n,m,alpha_hi,alpha_lo,beta_hi,beta_lo,A,lda,x_hi,x_lo,nx,y,ny);
	  }
	}
      }else{
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
	if( (ddtype_name == TmatA_name) && ( Tvecx_name == Tvecy_name) ){ 	
	  double *A_hi = (double *)A_.data_hi();
	  double *A_lo = (double *)A_.data_lo();
	  double *y_hi = (double *)y_.data_hi();
	  double *y_lo = (double *)y_.data_lo();
	  if( ddtype_name == Tvecx_name) { 	
	    auto x = x_.data();
	    // std::cout<<" ddgemv @ 5 @"<<std::endl;
	    blas_.ddgemv<
	      typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	      bool
	      >(n,m,alpha,beta,A_hi,A_lo,lda,x,nx,y_hi,y_lo,ny);
	  }else{
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    // std::cout<<" ddgemv @ 6 @"<<std::endl;
	    blas_.ddgemv<
	      typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	      bool,
	      bool,
	      bool
	      >(n,m,alpha,beta,A_hi,A_lo,lda,x_hi,x_lo,nx,y_hi,y_lo,ny);
	  }
	}else{
	  auto A = A_.data();
	  auto y = y_.data();
	  if( ddtype_name == Tvecx_name) { 	
	    // std::cout<<" ddgemv @ 7 @"<<std::endl;
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    blas_.ddgemv<
	      typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	      bool,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	      >(n,m,alpha,beta,A,lda,x_hi,x_lo,nx,y,ny);
	  }else{
	    auto x = x_.data();
	    // std::cout<<" ddgemv @ 8 @"<<std::endl;
	    blas_.ddgemv<
	      typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	      >(n,m,alpha,beta,A,lda,x,nx,y,ny);
	  }
	}
      }


#else
      blas_.gemv<
      	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
      	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	>(n,m,alpha,beta,A,lda,x,nx,y,ny);
#endif

    }


    // y = beta * y + alpha * Ax;
    template<
      class Talpha,
      class Tbeta,
      class TmatA,
      class Tvecx,
      class Tvecy,
      class Tcalc >
    void gemv(
	      PARCEL_INT m,
	      parcel::vector<Talpha> &alpha_,
	      parcel::vector<Tbeta>  &beta_,
	      parcel::vector<TmatA>  &A_, 
	      parcel::vector<Tvecx>  &x_, 
	      parcel::vector<Tvecy>  &y_){
      blas_timer_.gemv_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){
	// std::cout<<"     start ddgemv"<<std::endl;
	ddgemv_main<Talpha,Tbeta,TmatA,Tvecx,Tvecy>(m,alpha_,beta_,A_,x_,y_);
      }else{
	gemv_main<Talpha,Tbeta,TmatA,Tvecx,Tvecy,Tcalc>(m,alpha_,beta_,A_,x_,y_);
      }
      blas_timer_.gemv_end();
    }

    // double-double未対応
    // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
    template<
      class Talpha,class Tbeta,
      class TmatA,class TmatB,class TmatC,
      class Tcalc >
    void gemm(
	      PARCEL_INT m,
	      PARCEL_INT n,
	      PARCEL_INT k,
	      parcel::vector<Talpha> &alpha_,
	      parcel::vector<Tbeta>  &beta_,
	      parcel::vector<TmatA>  &A_,PARCEL_INT ldA,
	      parcel::vector<TmatB>  &B_,PARCEL_INT ldB,
	      parcel::vector<TmatC>  &C_,PARCEL_INT ldC,
	      std::string TransA = "N",
	      std::string TransB = "N",
	      std::string TransC = "N",
	      std::string sym = "N"
	      ){
      blas_timer_.gemm_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){ 
	ddgemm_main<Talpha,Tbeta,TmatA,TmatB,TmatC,Tcalc>(m,n,k,alpha_,beta_,A_,ldA,B_,ldB,C_,ldC,TransA,TransB,TransC,sym);
      }else{	
	gemm_main<Talpha,Tbeta,TmatA,TmatB,TmatC,Tcalc>(m,n,k,alpha_,beta_,A_,ldA,B_,ldB,C_,ldC,TransA,TransB,TransC,sym);
      }
      blas_timer_.gemm_end();
    }

    // y = beta * y + alpha * x;
    template<
      class Talpha,
      class Tbeta,
      class Tvecx,
      class Tvecy,
      class Tcalc >
    void axpy(
	      parcel::vector<Talpha> &alpha_,
	      parcel::vector<Tbeta>  &beta_,
	      parcel::vector<Tvecx>  &x_, 
	      parcel::vector<Tvecy>  &y_){
	      //	      Tvecx &x_, Tvecy &y_){
      blas_timer_.axpy_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){ 
	ddaxpy_main<Talpha,Tbeta,Tvecx,Tvecy>(alpha_,beta_,x_,y_);
      }else{
	axpy_main<Talpha,Tbeta,Tvecx,Tvecy,Tcalc>(alpha_,beta_,x_,y_);
      }
      blas_timer_.axpy_end();
    }

    // z = beta * y + alpha * x;
    template<
      class Talpha,
      class Tbeta,
      class Tvecx,
      class Tvecy,
      class Tvecz,
      class Tcalc >
    void axpy(
	      parcel::vector<Talpha> &alpha_,
	      parcel::vector<Tbeta>  &beta_,
	      parcel::vector<Tvecx>  &x_, 
	      parcel::vector<Tvecy>  &y_,
	      parcel::vector<Tvecz>  &z_){
	      //	      Tvecx &x_, Tvecy &y_){
      blas_timer_.axpy_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){ 
	ddaxpy_main<Talpha,Tbeta,Tvecx,Tvecy,Tvecz>(alpha_,beta_,x_,y_,z_);
      }else{
	axpy_main<Talpha,Tbeta,Tvecx,Tvecy,Tvecz,Tcalc>(alpha_,beta_,x_,y_,z_);
      }
      blas_timer_.axpy_end();
    }

    template<class Tvec>      
    void copy(
	      parcel::vector<Tvec>  &x_, 
	      parcel::vector<Tvec>  &y_){

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      // copyの計算時間はaxpyに計上
      blas_timer_.axpy_start();
      const auto Tvec_name = typeid(Tvec).name();
      if( ddtype_name == Tvec_name ){ 
	auto x_hi = x_.data_hi();
	auto y_hi = y_.data_hi();
	blas_.copy<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x_hi,nx,y_hi,ny);

	auto x_lo = x_.data_lo();
	auto y_lo = y_.data_lo();
	blas_.copy<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x_lo,nx,y_lo,ny);
      }else{
	// std::cout<<" copy : x --> y "<<std::endl;
	auto x = x_.data();
	auto y = y_.data();
	blas_.copy<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x,nx,y,ny);
      }
      blas_timer_.axpy_end();
    }

    template<class Tvec>      
    void zero_init(
		   parcel::vector<Tvec>  &x_ 
		   ){

      size_t n = x_.size(); 

      const auto Tvec_name = typeid(Tvec).name();
      if( ddtype_name == Tvec_name ){ 
	auto x_hi = x_.data_hi();
	blas_.zero<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x_hi,n);

	auto x_lo = x_.data_lo();
	blas_.zero<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x_lo,n);

      }else{
	auto x   = x_.data();
	blas_.zero<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	  >(n,x,n);
      }

    }
    // ---------------------
    // Sparce Matrix 
#include "parcel_blas_crs.hpp"
#include "parcel_blas_dia.hpp"
    // ---------------------
    template<
      class Talpha,
      class Tbeta,
      class TmatA,
      class Tvecx,
      class Tvecy,
      class Tcalc >
    void SpMV(
	      parcel::vector<Talpha>  &alpha_,	
	      parcel::vector<Tbeta>   &beta_,	
	      // parcel::crsMat<TmatA> &A_,
	      parcel::matrix<TmatA> &Mat,
	      parcel::vector<Tvecx>   &x_, 
	      parcel::vector<Tvecy>   &y_){

      size_t matrix_type = Mat.matrix_type();
      if(matrix_type==CRS_MATRIX){
	parcel::crsMat<TmatA> *A  = Mat.crsMat_data();
	parcel::crsMat<TmatA> &A_ = *A;
	const auto Tcalc_name = typeid(Tcalc).name();

	blas_timer_.SpMV_start();
	
	blas_timer_.SpMV_comm_start();
#if PARCEL_MPI

	const auto Tvecx_name = typeid(Tvecx).name();
	using Tvecx_type = typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type;
	if( ddtype_name == Tvecx_name ){ 
	  A->template SpMVcomm< double >( x_.queue(), (double *)x_.data_hi());
	  A->template SpMVcomm< double >( x_.queue(), (double *)x_.data_lo());
	}else{	
	  A->template SpMVcomm< Tvecx_type >( x_.queue(), x_.data());
	}
	
#endif	
	blas_timer_.SpMV_comm_end();
	
	if( ddtype_name == Tcalc_name ){ 
	  ddSpMV_main_CRS
	    <Talpha,Tbeta,TmatA,Tvecx,Tvecy>
	    (alpha_,beta_,A_,x_,y_);
	}else{
	  SpMV_main_CRS
	    <Talpha,Tbeta,TmatA,Tvecx,Tvecy,Tcalc>
	    (alpha_,beta_,A_,x_,y_);
	}

	blas_timer_.SpMV_end();

	return ;

      }

      if(matrix_type==DIA_MATRIX){
	blas_timer_.SpMV_start();
	parcel::diaMat<TmatA> *A  = Mat.diaMat_data();
	parcel::diaMat<TmatA> &A_ = *A;
	// y = α*Ax + β*y;
	const auto Tcalc_name = typeid(Tcalc).name();
	if( ddtype_name == Tcalc_name ){ 
	  // 1.0はddとdで精度の差がないためdoubleを利用
	  ddSpMV_main_DIA
	    <Talpha,Tbeta,TmatA,Tvecx,Tvecy>
	    (alpha_,beta_,A_,x_,y_);wait();
	}else{
	  SpMV_main_DIA
	    <Talpha,Tbeta,TmatA,Tvecx,Tvecy,Tcalc>
	    (alpha_,beta_,A_,x_,y_);wait();
	}

	blas_timer_.SpMV_end();
	return ;
      }


    }

    template<
      class TmatA,
      class Tvecx,
      class Tvecb,
      class Tvecr,
      class Tcalc >
    void calc_res(
		  // parcel::crsMat<TmatA> &A_,
		  parcel::matrix<TmatA> &Mat,
		  parcel::vector<Tvecx>   &x_, 
		  parcel::vector<Tvecb>   &b_, 
		  parcel::vector<Tvecr>   &r_){

      size_t matrix_type = Mat.matrix_type();
      if(matrix_type==CRS_MATRIX){
	parcel::crsMat<TmatA> *A  = Mat.crsMat_data();
	parcel::crsMat<TmatA> &A_ = *A;

	copy<Tvecb>(b_,r_);wait();
	blas_timer_.SpMV_start();
	// y = α*Ax + β*y;
	const auto Tcalc_name = typeid(Tcalc).name();

	blas_timer_.SpMV_comm_start();
#if PARCEL_MPI

	const auto Tvecx_name = typeid(Tvecx).name();
	using Tvecx_type = typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type;
	if( ddtype_name == Tvecx_name ){ 
	  A->template SpMVcomm< double >( x_.queue(), (double *)x_.data_hi());
	  A->template SpMVcomm< double >( x_.queue(), (double *)x_.data_lo());
	}else{	
	  A->template SpMVcomm< Tvecx_type >( x_.queue(), x_.data());
	}
	
#endif	
	blas_timer_.SpMV_comm_end();

	
	if( ddtype_name == Tcalc_name ){ 
	  // 1.0はddとdで精度の差がないためdoubleを利用
	  // ddSpMV_main_CRS
	  //   <parcel_double,parcel_double,TmatA,Tvecx,Tvecr>
	  //   (negative_one,one,A_,x_,r_);wait();
	  ddSpMV_main_CRS
	    <parcel_doubledouble,parcel_doubledouble,TmatA,Tvecx,Tvecr>
	    (negative_one_dd,one_dd,A_,x_,r_);wait();
	}else{
	  SpMV_main_CRS
	    <parcel_double,parcel_double,TmatA,Tvecx,Tvecr,Tcalc>
	    (negative_one,one,A_,x_,r_);wait();
	}
	blas_timer_.SpMV_end();
	return ;
      }

      if(matrix_type==DIA_MATRIX){
	parcel::diaMat<TmatA> *A  = Mat.diaMat_data();
	parcel::diaMat<TmatA> &A_ = *A;

	copy<Tvecb>(b_,r_);wait();
	blas_timer_.SpMV_start();
	// y = α*Ax + β*y;
	const auto Tcalc_name = typeid(Tcalc).name();
	if( ddtype_name == Tcalc_name ){ 
	  ddSpMV_main_DIA
	    <parcel_doubledouble,parcel_doubledouble,TmatA,Tvecx,Tvecr>
	    (negative_one_dd,one_dd,A_,x_,r_);wait();
	}else{
	  SpMV_main_DIA
	    <parcel_double,parcel_double,TmatA,Tvecx,Tvecr,Tcalc>
	    (negative_one,one,A_,x_,r_);wait();
	}
	blas_timer_.SpMV_end();
	return ;
      }

    }

  };

}


#endif
