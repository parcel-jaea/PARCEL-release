// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_BLAS_HPP_
#define PARCEL_BLAS_HPP_

#include <type_traits>
#include <stdexcept>
#include <algorithm>
#include "util_sycl.hpp"
#include "mixed_precision.hpp"
#include "parcel.hpp"
#include "parcel_util.hpp"
#include "parcel_time.hpp"
#include <cstdlib>

// 内積と係数を値で渡す axpy (axpy_value / axpby_value) の時間計測の前後でデバイスを同期するか
//   (環境変数 PARCEL_TIMER_SYNC)。
//   未設定・0: 同期しない。内積は値を返す前に自分で同期し、キューは in-order なので、
//             計算の順序と結果は変わらない。同期の回数が減る (pgmres の MGS では 1 ステップに 4 回)。
//             その代わり内積の時間に、直前に積んだカーネル (MGS の axpy など) の時間が入る。
//   1       : 同期する (時間の内訳を正確に取りたいとき)。
//   SpMV・通信・その他の axpy / gemv / gemm の計測は、後続の MPI 通信の順序を保証しているので常に同期する。

#if PARCEL_MPI
#include <mpi.h>
#include "./parcel_mpi/parcel_mpi.hpp"
#endif

namespace parcel{

  class blas_timer{
    sycl::queue *q_;
    static bool sync_default_(){
      static const bool v = [](){
	if(const char *e = std::getenv("PARCEL_TIMER_SYNC")){ return std::atoi(e) != 0; }
	return false;
      }();
      return v;
    }
    bool sync_ = sync_default_();
    parcel::timer timer_axpy;
    parcel::timer timer_gemv;
    parcel::timer timer_gemm;
    parcel::timer timer_dot;
    parcel::timer timer_SpMV_comm;
    parcel::timer timer_SpMV;
    // dot の MPI_Allreduce だけを切り出して測る。
    //   dot は「局所内積カーネル + 全体集約」の合計なので、
    //   カーネルの効率と通信のコストを分けて見るために必要。
    parcel::timer timer_dot_comm;

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
      timer_SpMV_comm.reset();
      timer_dot_comm.reset();
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

    // 内積の前後の同期は PARCEL_TIMER_SYNC=1 のときだけ (上の説明)
    void dot_start(){if(sync_) q_[0].wait();timer_dot.start();}
    void dot_end()  {if(sync_) q_[0].wait();timer_dot.end();  }
    // axpy_value (MGS) 用。時間は axpy に計上する
    void axpy_value_start(){if(sync_) q_[0].wait();timer_axpy.start();}
    void axpy_value_end()  {if(sync_) q_[0].wait();timer_axpy.end();  }
    bool sync() const {return sync_;}

    // 集約のみ。内積のカーネルは集約の前に自分で同期している (mixed_precision_dot*.hpp) ので入れない。
    void dot_comm_start(){timer_dot_comm.start();}
    void dot_comm_end()  {timer_dot_comm.end();  }

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
    double get_dot_comm(){
      return timer_dot_comm.get();
    }
    long long get_dot_comm_count(){
      return timer_dot_comm.get_count();
    }
    long long get_SpMV_comm_count(){
      return timer_SpMV_comm.get_count();
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
      // 内積の作業域もこのキューのデバイスに置く (check_dwork_size の resize(n) はこのキューを使い続ける)
      dwork_.resize(1,q);


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
      // 内積の作業域もこのキューのデバイスに置く (check_dwork_size の resize(n) はこのキューを使い続ける)
      dwork_.resize(1,q);


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
    double get_time_dot_comm() {return blas_timer_.get_dot_comm();}
    double get_time_SpMV_comm(){return blas_timer_.get_SpMV_comm();}

    void check_dwork_size(size_t n){
      size_t work_size = dwork_.size();
      if(work_size < n){
	dwork_.resize(n);
	wait();
      }
      return ;
    }

    // -------------------

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
	ret_dd.set_dd(x,0.0);
	ret_T = x;
      }
    }

    // -------------------
    
    template<class Tvecx, class Tvecy>
    //   do_comm = false で MPI 集約を省く (dot_main と同じ約束)。
    //   dd 型でも局所値のみを返せるようにしておかないと、呼び出し側が
    //   まとめて allreduce する場合に二重集約になる。
    void dddot_main(
		   parcel::vector<Tvecx> &x_,
		   parcel::vector<Tvecy> &y_,
		   bool do_comm = true){
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
	if(do_comm){
	  double sbuf_hi = ret_dd.get_hi();
	  double sbuf_lo = ret_dd.get_lo();
	  double rbuf_hi = 0.0;
	  double rbuf_lo = 0.0;
	  blas_timer_.dot_comm_start();
	  comm_.mpi_ddallreduce(&sbuf_hi,&sbuf_lo,&rbuf_hi,&rbuf_lo,1,MPI_SUM);
	  blas_timer_.dot_comm_end();
	  ret_dd.set_dd(rbuf_hi,rbuf_lo);
	}
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
	if(do_comm){
	  double sbuf_hi = ret_dd.get_hi();
	  double sbuf_lo = ret_dd.get_lo();
	  double rbuf_hi = 0.0;
	  double rbuf_lo = 0.0;
	  blas_timer_.dot_comm_start();
	  comm_.mpi_ddallreduce(&sbuf_hi,&sbuf_lo,&rbuf_hi,&rbuf_lo,  1,MPI_SUM);
	  blas_timer_.dot_comm_end();
	  ret_dd.set_dd(rbuf_hi,rbuf_lo);
	}
#endif
	ret_T = ret_dd.get_T();
      }
    }

    //   do_comm = false にすると MPI 集約を行わず、そのランクの局所値だけを残す。
    //   長さ j+1 の内積をまとめて取りたいとき、dot を j+1 回呼ぶたびに allreduce が
    //   走るのを避けるために使う (ks_arnoldi の直交化)。呼び出し側が自分で
    //   allreduce を 1 回だけ掛ける責任を持つ。既定は true。
    template<class Tvecx, class Tvecy, class Tcalc>
    void dot_main(
		  parcel::vector<Tvecx> &x_,
		  parcel::vector<Tvecy> &y_,
		  bool do_comm = true){

      size_t nx = x_.size(); 
      size_t ny = y_.size(); 
      size_t n  = std::min(nx,ny);

      auto x = x_.data();
      auto y = y_.data();
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
	if(do_comm){
	  using T_ret_type = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type;
	  T_ret_type sbuf = ret;
	  T_ret_type rbuf = 0.0;
	  blas_timer_.dot_comm_start();
	  comm_.mpi_allreduce<T_ret_type>(&sbuf,&rbuf,1,MPI_SUM);
	  blas_timer_.dot_comm_end();
	  ret = rbuf;
	}
#endif
	
	ret_T = ret;
	ret_dd.set_dd(ret,0.0);

    }

      template<
	class Tvecx,
      class Tvecy,
      class Tcalc >
    //   do_comm = false で MPI 集約を省く (局所値のみ)。長さ j+1 の内積をまとめて
    //   取りたいときに、dot を呼ぶたびに allreduce が走るのを避けるために使う。
    //   呼び出し側が自分で allreduce を 1 回だけ掛ける責任を持つ。既定は true。
    void dot(
	     parcel::vector<Tvecx> &x_,
	     parcel::vector<Tvecy> &y_,
	     bool do_comm = true){

      blas_timer_.dot_start();

      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){ 
	dddot_main<Tvecx,Tvecy>(x_,y_,do_comm);
      }else{
	dot_main<Tvecx,Tvecy,Tcalc>(x_,y_,do_comm);
      }

      blas_timer_.dot_end();
    }


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
		       ny,
		       z_hi,z_lo,
		       nz
		       );
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

	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	    >(n,alpha_hi,alpha_lo,beta_hi,beta_lo,x,nx,y,ny,z,nz);
	  
	}else{
	  auto *alpha = alpha_.data(); 	  	  
	  auto *beta = beta_.data(); 	  	  
	  blas_.ddaxpy<
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	    typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	    >(n,alpha,beta,x,nx,y,ny,z,nz);
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
		   parcel::vector<Tvecy>  &y_,
		   PARCEL_INT ldA_in = 0
		   ){

      size_t nx = m; 
      size_t ny = y_.size(); 
      size_t n   = ny;
      // ldA_in: A の列の先頭間隔。0 なら y の長さから導く。
      //   parallel_vector に詰め物を入れると A の stride と y の長さが
      //   一致しなくなるので、呼び出し側が実際の ld を渡せるようにする。
      size_t lda = (ldA_in > 0) ? (size_t)ldA_in : n;

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
#include "make_parcel_blas_ddgemm_py.hpp"            
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
		   std::string &sym,
		   std::string &tall
		   ){

      auto A = A_.data();
      auto B = B_.data();
      auto C = C_.data();
      auto alpha = alpha_.data();
      auto beta  = beta_.data();

      // 縦長で beta を使う版 (tall == "B"): C = alpha*A*B + beta*C を 1 行 = 1 ワークアイテムで
      //   (A64FX は行ブロック版)。pcagmres の reOrth の更新 V -= Q R 用 (mixed_precision_gemm.hpp の tgemm_beta)。
      //   n > 32 は tgemm_beta の中で既定の gemm に落ちる (A64FX 版は n の制限なし)。
      if( tall == "B" ){
	blas_.tgemm_beta<
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
	return ;
      }

      // 縦長は転置未対応 : C = C + AB @ A,C:縦長行列
      if( tall == "T" ){
	blas_.tgemm<
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

	return ;
      }
      
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
		   parcel::vector<Tvecy>  &y_,
		   PARCEL_INT ldA_in = 0
		   ){
      const auto Talpha_name = typeid(Talpha).name();
      const auto Tbeta_name = typeid(Tbeta).name();
      const auto TmatA_name = typeid(TmatA).name();
      const auto Tvecx_name = typeid(Tvecx).name();
      const auto Tvecy_name = typeid(Tvecy).name();

      size_t nx = m; 
      size_t ny = y_.size(); 
      size_t n   = ny;
      // ldA_in: A の列の先頭間隔。0 なら y の長さから導く。
      //   parallel_vector に詰め物を入れると A の stride と y の長さが
      //   一致しなくなるので、呼び出し側が実際の ld を渡せるようにする。
      size_t lda = (ldA_in > 0) ? (size_t)ldA_in : n;


      // x (係数ベクトル) が double-double なら data_hi()/data_lo() を、そうでなければ data() を渡す
      //   (lo 部を落とさないため)。外側の条件は Talpha と Tbeta が同じ型であること、内側の条件は
      //   A が dd のとき y も dd であること (x と y が同じ型) を前提にしている。
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
	  if( ddtype_name != Tvecx_name) { 	
	    auto x = x_.data();
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
	    blas_.ddgemv(n,m,alpha_hi,alpha_lo,beta_hi,beta_lo,A_hi,A_lo,lda,x_hi,x_lo,nx,y_hi,y_lo,ny);
	  }
	}else{
	  auto A = A_.data();
	  auto x = x_.data();
	  auto y = y_.data();
	  if( ddtype_name != Tvecx_name) { 	
	    auto x = x_.data();
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
	  if( ddtype_name != Tvecx_name) { 	
	    auto x = x_.data();
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
	      parcel::vector<Tvecy>  &y_,
	      PARCEL_INT ldA_in = 0){
      blas_timer_.gemv_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){
	ddgemv_main<Talpha,Tbeta,TmatA,Tvecx,Tvecy>(m,alpha_,beta_,A_,x_,y_,ldA_in);
      }else{
	gemv_main<Talpha,Tbeta,TmatA,Tvecx,Tvecy,Tcalc>(m,alpha_,beta_,A_,x_,y_,ldA_in);
      }
      blas_timer_.gemv_end();
    }
    
    
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
	      std::string sym  = "N",
	      std::string tall = "N"
	      ){
      blas_timer_.gemm_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){
	// 縦長未対応
	ddgemm_main<Talpha,Tbeta,TmatA,TmatB,TmatC,Tcalc>(m,n,k,alpha_,beta_,A_,ldA,B_,ldB,C_,ldC,TransA,TransB,TransC,sym);
      }else{	
	gemm_main<Talpha,Tbeta,TmatA,TmatB,TmatC,Tcalc>(m,n,k,alpha_,beta_,A_,ldA,B_,ldB,C_,ldC,TransA,TransB,TransC,sym,tall);
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
      blas_timer_.axpy_start();
      const auto Tcalc_name = typeid(Tcalc).name();
      if( ddtype_name == Tcalc_name ){ 
	ddaxpy_main<Talpha,Tbeta,Tvecx,Tvecy>(alpha_,beta_,x_,y_);
      }else{
	axpy_main<Talpha,Tbeta,Tvecx,Tvecy,Tcalc>(alpha_,beta_,x_,y_);
      }
      blas_timer_.axpy_end();
    }

    // y = y + alpha * x。alpha はホストの値のまま渡す。
    //   axpy(alpha, one, x, y) と同じ計算だが、alpha を 1 要素のデバイス配列に書き戻す
    //   (setValue とその前後の wait) が要らない。alpha は Talpha の精度に丸めてから使うので、
    //   Talpha のデバイス配列を経由した場合と同じ値になる。
    //   double-double (Talpha / ベクトル / Tcalc のいずれか) には対応しない。呼び出し側が
    //   axpy_value_supported で確かめてから使う。
    template<class Talpha, class Tvecx, class Tvecy, class Tcalc>
    static constexpr bool axpy_value_supported(){
      return !std::is_same<Talpha,parcel_doubledouble>::value
	&& !std::is_same<Tvecx,parcel_doubledouble>::value
	&& !std::is_same<Tvecy,parcel_doubledouble>::value
	&& !std::is_same<Tcalc,parcel_doubledouble>::value;
    }

    template<class Talpha, class Tvecx, class Tvecy, class Tcalc>
    void axpy_value(
		    parcel::scalar alpha_,
		    parcel::vector<Tvecx> &x_,
		    parcel::vector<Tvecy> &y_){
      static_assert(axpy_value_supported<Talpha,Tvecx,Tvecy,Tcalc>(), "axpy_value: double-double is not supported");
      using Ta = typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc  ) >::type;
      blas_timer_.axpy_value_start();
      size_t nx = x_.size();
      size_t ny = y_.size();
      size_t n  = std::min(nx,ny);
      const Ta a  = alpha_.get_T();   // setValue(parcel::scalar) と同じ変換
      const Ta b  = 1.0;
      blas_.axpy_value<
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	Tc
	>(n,(Tc)a,(Tc)b,x_.data(),nx,y_.data(),ny);
      // PARCEL_TIMER_SYNC=0 では完了を待たずに返る (in-order キューの次の操作が待つ)。
      //   ホストから y を直接読む前や MPI に渡す前には呼び出し側が wait すること。
      blas_timer_.axpy_value_end();
    }

#if PARCEL_GS_FUSED
    // MGS 用: w = w + alpha * xprev (axpy_value と同じ計算) を行い、続けて内積 <w, vcur> を取る
    //   (dot<Tvec,Tvec,Tdot>(w, vcur) と同じ計算・同じ MPI 集約) のを 1 本のカーネルで行う。
    //   結果は get_dot() で取り出す (dot と同じ)。各要素の演算と順序は axpy_value のあと dot と同じ。
    //   時間は内積 (dot) に計上する (axpy の分も含む)。ブロック版の内積があるビルドだけ。
    template<class Talpha, class Tvec, class Taxpy, class Tdot>
    void axpy_dot_value(
			parcel::scalar alpha_,
			parcel::vector<Tvec> &xprev_,
			parcel::vector<Tvec> &w_,
			parcel::vector<Tvec> &vcur_){
      static_assert(axpy_value_supported<Talpha,Tvec,Tvec,Taxpy>() && !std::is_same<Tdot,parcel_doubledouble>::value,
		    "axpy_dot_value: double-double is not supported");
      using Ta = typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Taxpy  ) >::type;
      using Tv = typename std::result_of< parcel::parcel_type_to_baseType( Tvec   ) >::type;
      using Td = typename std::result_of< parcel::parcel_type_to_baseType( Tdot   ) >::type;
      const size_t n_axpy = std::min(xprev_.size(), w_.size());   // axpy_value と同じ長さ
      const size_t n_dot  = std::min(w_.size(), vcur_.size());    // dot と同じ長さ
      if(n_axpy != n_dot){
	throw std::runtime_error("PARCEL: axpy_dot_value: axpy length " + std::to_string(n_axpy)
				 + " != dot length " + std::to_string(n_dot));
      }
      blas_timer_.dot_start();
      const Ta a = alpha_.get_T();   // setValue(parcel::scalar) と同じ変換 (axpy_value と同じ)
      const Ta b = 1.0;
      auto ret = blas_.axpy_dot_fused<Tv,Tv,Tc,Td>(n_dot,(Tc)a,(Tc)b,xprev_.data(),w_.data(),vcur_.data());
      blas_.wait();
#if PARCEL_MPI
      {
	Td sbuf = ret;
	Td rbuf = 0.0;
	blas_timer_.dot_comm_start();
	comm_.mpi_allreduce<Td>(&sbuf,&rbuf,1,MPI_SUM);
	blas_timer_.dot_comm_end();
	ret = rbuf;
      }
#endif
      ret_T = ret;
      ret_dd.set_dd(ret,0.0);
      blas_timer_.dot_end();
    }
#endif

#if PARCEL_KS_FUSED
    // 局所内積 <x_d, y> (x_d = xbase の d*ldx 要素目から長さ n、d = 0..count-1) を 1 本のカーネルで取る。
    //   dot<Tvecx,Tvecy,Tcalc>(x_d, y, false) を count 回呼んで get_dot() した値と
    //   ビット単位で同じものを out[d] に返す (MPI 集約はしない。呼び出し側がまとめて 1 回行う)。
    //   時間は内積 (dot) に計上する。double-double には対応しない。
    template<class Tvecx, class Tvecy, class Tcalc>
    void multi_dot_local(
			 parcel::vector<Tvecx> &xbase_,
			 size_t ldx,
			 PARCEL_INT count,
			 size_t n,
			 parcel::vector<Tvecy> &y_,
			 std::vector<parcel::scalar> &out){
      static_assert(!std::is_same<Tvecx,parcel_doubledouble>::value && !std::is_same<Tvecy,parcel_doubledouble>::value
		    && !std::is_same<Tcalc,parcel_doubledouble>::value, "multi_dot_local: double-double is not supported");
      using Tx = typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type;
      using Ty = typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type;
      blas_timer_.dot_start();
      std::vector<double> sums((size_t)count);
      blas_.multi_dot<Tx,Ty>((PARCEL_INT)n, xbase_.data(), ldx, count, y_.data(), sums.data());
      for(PARCEL_INT d=0; d<count; d++){
	// dot_main と同じ変換: ブロック版 dot は double の和を Tcalc で返し、ret_T / ret_dd に入れる
	const Tc ret = sums[(size_t)d];
	ret_T = ret;
	ret_dd.set_dd(ret,0.0);
	out[(size_t)d] = get_dot();
      }
      blas_timer_.dot_end();
    }
#endif


    // y = beta * y + alpha * x。alpha / beta はホストの値のまま渡す。
    //   axpy(alpha, beta, x, y) と同じ計算で、係数は Talpha / Tbeta の精度に丸めてから使う
    //   (1 要素のデバイス配列を経由した場合と同じ値)。double-double には対応しない。
    template<class Talpha, class Tbeta, class Tvecx, class Tvecy, class Tcalc>
    void axpby_value(
		     parcel::scalar alpha_,
		     parcel::scalar beta_,
		     parcel::vector<Tvecx> &x_,
		     parcel::vector<Tvecy> &y_){
      static_assert(axpy_value_supported<Talpha,Tvecx,Tvecy,Tcalc>() && !std::is_same<Tbeta,parcel_doubledouble>::value,
		    "axpby_value: double-double is not supported");
      using Ta = typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type;
      using Tb = typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc  ) >::type;
      blas_timer_.axpy_value_start();
      size_t nx = x_.size();
      size_t ny = y_.size();
      size_t n  = std::min(nx,ny);
      const Ta a = alpha_.get_T();   // setValue(parcel::scalar) と同じ変換
      const Tb b = beta_.get_T();
      blas_.axpy_value<
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	Tc
	>(n,(Tc)a,(Tc)b,x_.data(),nx,y_.data(),ny);
      // PARCEL_TIMER_SYNC=0 では完了を待たずに返る (in-order キューの次の操作が待つ)。
      blas_timer_.axpy_value_end();
    }

    // z = beta * y + alpha * x。alpha / beta はホストの値のまま渡す。
    //   3 本版 axpy(alpha, beta, x, y, z) と同じ計算。double-double には対応しない。
    template<class Talpha, class Tbeta, class Tvecx, class Tvecy, class Tvecz, class Tcalc>
    void axpby_value(
		     parcel::scalar alpha_,
		     parcel::scalar beta_,
		     parcel::vector<Tvecx> &x_,
		     parcel::vector<Tvecy> &y_,
		     parcel::vector<Tvecz> &z_){
      static_assert(axpy_value_supported<Talpha,Tvecx,Tvecy,Tcalc>() && !std::is_same<Tbeta,parcel_doubledouble>::value
		    && !std::is_same<Tvecz,parcel_doubledouble>::value,
		    "axpby_value: double-double is not supported");
      using Ta = typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type;
      using Tb = typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc  ) >::type;
      blas_timer_.axpy_value_start();
      size_t nx = x_.size();
      size_t ny = y_.size();
      size_t nz = z_.size();
      size_t n  = std::min(nx,ny);
      const Ta a = alpha_.get_T();
      const Tb b = beta_.get_T();
      blas_.axpy_value<
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type,
	Tc
	>(n,(Tc)a,(Tc)b,x_.data(),nx,y_.data(),ny,z_.data(),nz);
      blas_timer_.axpy_value_end();
    }

#if PARCEL_CG_FUSED
    // CG (pstep = 1) 用: gemv(1, one, one, p, alpha_history, x) と axpby_value(ap, bp, s, p, p) を 1 本のカーネルで行う
    //   (PARCEL_CG_FUSED、parcel_nblock.hpp)。時間は axpy に計上する。double-double には対応しない。
    template<class Tdot, class Tvec, class Tcalc>
    void cg_xp_fused(parcel::vector<Tdot> &one_, parcel::vector<Tdot> &ah_, parcel::scalar ap_, parcel::scalar bp_,
                     parcel::vector<Tvec> &s_, parcel::vector<Tvec> &p_, parcel::vector<Tvec> &x_){
      static_assert(axpy_value_supported<Tdot,Tvec,Tvec,Tcalc>(), "cg_xp_fused: double-double is not supported");
      using Td = typename std::result_of< parcel::parcel_type_to_baseType( Tdot  ) >::type;
      using Tv = typename std::result_of< parcel::parcel_type_to_baseType( Tvec  ) >::type;
      using Tc = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type;
      blas_timer_.axpy_value_start();
      const size_t n = std::min(std::min(s_.size(), p_.size()), x_.size());
      const Td a = ap_.get_T();   // axpby_value と同じ変換 (Talpha = Tbeta = Tdot)
      const Td b = bp_.get_T();
      blas_.cg_xp_fused<Td,Td,Td,Tv,Tv,Tv,Tc>((PARCEL_INT)n, one_.data(), one_.data(), ah_.data(), p_.data(), x_.data(), (Tc)a, (Tc)b, s_.data());
      blas_timer_.axpy_value_end();
    }
#endif

    // 係数の値渡し (axpy_value / axpby_value) が使える型の組み合わせか (double-double を含まない)
    template<class Tcoef, class Tvecx, class Tvecy, class Tcalc>
    static constexpr bool coef_by_value_supported(){
      return axpy_value_supported<Tcoef,Tvecx,Tvecy,Tcalc>();
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
