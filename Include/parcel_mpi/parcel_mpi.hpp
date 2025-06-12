#ifndef PARCEL_MPI_HPP_
#define PARCEL_MPI_HPP_

#include <mpi.h>
#include <vector>
#include "mixed_precision_mpi.hpp"
#include <CL/sycl.hpp>
#include "../parcel.hpp"
#include "../parcel_util.hpp"

namespace parcel{

  class comm : public ::mixed_precison::comm {
    sycl::queue *q_;
  public : 
    comm(){};
    comm(MPI_Comm comm,sycl::queue *q){
      set_comm(comm);
      q_ = q;
    };
    ~comm(){};
    void setQueue(sycl::queue *q){
      q_=q;
    }
    sycl::queue *queue(){
      return q_;
    }

    int rank(){return mpi_rank();}
    int size(){return mpi_size();}
    void barrier(){
      mpi_barrier();
      return ;
    }
    template<class Tdata>
    std::vector<MPI_Request> isend(
		      parcel::vector<Tdata> &buff_,int send_rank,int n
		      ){
      std::vector<MPI_Request> req;
      auto buff = buff_.data();
      const auto Tdata_name = typeid(Tdata).name();
      if( Tdata_name == ddtype_name){
	req.resize(2);
	double *buff_hi = (double *)buff_.data_hi();
	double *buff_lo = (double *)buff_.data_lo();
	req[0] = mpi_isend<double>(buff_hi,send_rank,n);
	req[1] = mpi_isend<double>(buff_lo,send_rank,n);
      }else{
	req.resize(1);
	auto buff = buff_.data();
	using Tdata_type = typename std::result_of< parcel::parcel_type_to_baseType( Tdata ) >::type;
	req[0] = mpi_isend<Tdata_type>(buff,send_rank,n);
      }
      return req;
    } 

    template<class Tdata>
    std::vector<MPI_Request> irecv(
		      parcel::vector<Tdata> &buff_,int recv_rank,int n
		      ){
      std::vector<MPI_Request> req;
      auto buff = buff_.data();
      const auto Tdata_name = typeid(Tdata).name();
      if( Tdata_name == ddtype_name){
	req.resize(2);
	double *buff_hi = (double *)buff_.data_hi();
	double *buff_lo = (double *)buff_.data_lo();
	req[0] = mpi_irecv<double>(buff_hi,recv_rank,n);
	req[1] = mpi_irecv<double>(buff_lo,recv_rank,n);
      }else{
	req.resize(1);
	auto buff = buff_.data();
	using Tdata_type = typename std::result_of< parcel::parcel_type_to_baseType( Tdata ) >::type;
	req[0] = mpi_irecv<Tdata_type>(buff,recv_rank,n);
      }
      return req;
    } 
    
    template<class Tdata,class Tcalc>
    void allreduce(
		    parcel::vector<Tdata> &sbuf_,
		    parcel::vector<Tcalc> &rbuf_,
		    int n, MPI_Op op){
      
      using Tdata_type = typename std::result_of< parcel::parcel_type_to_baseType( Tdata ) >::type;
      using Tcalc_type = typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type;

      
      const auto Tdata_name = typeid(Tdata).name();
      const auto Tcalc_name = typeid(Tcalc).name();

      //       if( (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) ){ 
      if( Tdata_name == Tcalc_name ){
	// データ型と演算型が同じ
	if(Tcalc_name == ddtype_name){
	  // dd演算
	  if(Tdata_name == ddtype_name){
	    // 入力 : dd

#if 1
	    std::vector<double> sbuf_hi_host(sbuf_.size());
	    std::vector<double> sbuf_lo_host(sbuf_.size());
	    std::vector<double> rbuf_hi_host(rbuf_.size());
	    std::vector<double> rbuf_lo_host(rbuf_.size());
	    
	    double *sbuf_hi = (double *)sbuf_hi_host.data();
	    double *sbuf_lo = (double *)sbuf_lo_host.data();
	    double *rbuf_hi = (double *)rbuf_hi_host.data();
	    double *rbuf_lo = (double *)rbuf_lo_host.data();

	    sbuf_.getValue(sbuf_hi,sbuf_lo);
	    q_[0].wait();
	    mpi_ddallreduce(sbuf_hi,sbuf_lo,rbuf_hi,rbuf_lo,n,op);
	    rbuf_.setValue(rbuf_hi,rbuf_lo);
#else
	    double *sbuf_hi = (double *)sbuf_.data_hi();
	    double *sbuf_lo = (double *)sbuf_.data_lo();
	    double *rbuf_hi = (double *)rbuf_.data_hi();
	    double *rbuf_lo = (double *)rbuf_.data_lo();
	    q_[0].wait();
	    mpi_ddallreduce(sbuf_hi,sbuf_lo,rbuf_hi,rbuf_lo,n,op);
#endif	    
	  }else{
	    // 入力 : T
#if 1
	    std::vector<Tdata_type> sbuf_host(sbuf_.size());
	    std::vector<double> rbuf_hi_host(rbuf_.size());
	    std::vector<double> rbuf_lo_host(rbuf_.size());
	    auto sbuf = sbuf_host.data();
	    double *rbuf_hi = (double *)rbuf_hi_host.data();
	    double *rbuf_lo = (double *)rbuf_lo_host.data();
	    
	    sbuf_.getValue(sbuf);
	    q_[0].wait();
	    mpi_ddallreduce<Tdata_type>(sbuf,false,rbuf_hi,rbuf_lo,n,op);
	    rbuf_.setValue(rbuf_hi,rbuf_lo);
#else	    
	    auto sbuf = sbuf_.data();
	    double *rbuf_hi = (double *)rbuf_.data_hi();
	    double *rbuf_lo = (double *)rbuf_.data_lo();
	    q_[0].wait();
	    mpi_ddallreduce<Tdata_type>(sbuf,false,rbuf_hi,rbuf_lo,n,op);
#endif
	  }
	}else{
	  // T演算

#if 1
	  std::vector<Tdata_type> sbuf_host(sbuf_.size());
	  std::vector<Tcalc_type> rbuf_host(rbuf_.size());
	  Tdata_type *sbuf = (Tdata_type *)sbuf_host.data();
	  Tcalc_type *rbuf = (Tcalc_type *)rbuf_host.data();
	  sbuf_.getValue(sbuf);	  
	  q_[0].wait();
	  mpi_allreduce<Tcalc_type>(sbuf,rbuf,n,op);
	  rbuf_.setValue(rbuf);	  
	  
#else	  
	  Tcalc_type *sbuf = (Tcalc_type *)sbuf_.data();
	  Tcalc_type *rbuf = (Tcalc_type *)rbuf_.data();
	  q_[0].wait();
	  mpi_allreduce<Tcalc_type>(sbuf,rbuf,n,op);
#endif	  
	}
      }else{
	// データ型と演算型が異なる
	parcel::vector_util vec_util;
	// vec_util.convert<Tvecx,Tvec>(x0,x);
	if(Tcalc_name == ddtype_name){
	  // dd演算
	  if(Tdata_name == ddtype_name){
	    // 入力 : dd
#if 1
	    std::vector<double> sbuf_hi_host(sbuf_.size());
	    std::vector<double> sbuf_lo_host(sbuf_.size());
	    std::vector<Tcalc_type> rbuf_host(rbuf_.size());
	    
	    double *sbuf_hi = (double *)sbuf_hi_host.data();
	    double *sbuf_lo = (double *)sbuf_lo_host.data();
	    auto rbuf = rbuf_host.data();
	    
	    sbuf_.getValue(sbuf_hi,sbuf_lo);
	    q_[0].wait();
	    mpi_ddallreduce<Tcalc_type>(sbuf_hi,sbuf_lo,rbuf,false,n,op);
	    rbuf_.setValue(rbuf);	    
#else	    
	    double *sbuf_hi = (double *)sbuf_.data_hi();
	    double *sbuf_lo = (double *)sbuf_.data_lo();
	    auto rbuf    = rbuf_.data();
	    q_[0].wait();
	    mpi_ddallreduce<Tcalc_type>(sbuf_hi,sbuf_lo,rbuf,false,n,op);
#endif
	  }else{
	    // 入力 : T
#if 1
	    std::vector<Tdata_type> sbuf_host(sbuf_.size());
	    std::vector<double> rbuf_hi_host(rbuf_.size());
	    std::vector<double> rbuf_lo_host(rbuf_.size());
	    
	    auto sbuf = sbuf_host.data();
	    double *rbuf_hi = (double *)rbuf_hi_host.data();
	    double *rbuf_lo = (double *)rbuf_lo_host.data();
	    
	    sbuf_.getValue(sbuf);
	    mpi_ddallreduce<Tdata_type>(sbuf,false,rbuf_hi,rbuf_lo,n,op);
	    rbuf_.setValue(rbuf_hi,rbuf_lo);	    
	    
#else	    
	    auto sbuf = sbuf_.data();
	    double *rbuf_hi = (double *)rbuf_.data_hi();
	    double *rbuf_lo = (double *)rbuf_.data_lo();
	    mpi_ddallreduce<Tdata_type>(sbuf,false,rbuf_hi,rbuf_lo,n,op);
#endif	    
	  }
	}else{
	  // // T演算
	  // auto sbuf = sbuf_.data();
	  // auto rbuf = rbuf_.data();
#if 1
	  std::vector<Tdata_type> sbuf_host(sbuf_.size());
	  std::vector<Tcalc_type> rbuf_host(rbuf_.size());
	  Tdata_type *sbuf = (Tdata_type *)sbuf_host.data();
	  Tcalc_type *rbuf = (Tcalc_type *)rbuf_host.data();
	  
	  sbuf_.getValue(sbuf);
	  mpi_allreduce<Tcalc_type>(sbuf,rbuf,n,op);	  
	  rbuf_.setValue(rbuf);	    
#else	  
	  Tdata_type *sbuf = (Tdata_type *)sbuf_.data();
	  Tcalc_type *rbuf = (Tcalc_type *)rbuf_.data();	
	  mpi_allreduce<Tcalc_type>(sbuf,rbuf,n,op);
#endif	  
	}

      }

    }

    std::vector<MPI_Status> wait(std::vector<MPI_Request> &req){
      std::vector<MPI_Status > stat;
      int nreq = req.size();
      stat.resize(nreq);
      for(int i = 0;i<nreq;i++){
	stat[i] = mpi_wait_(req[i]);
      }
      return stat;
    }

    template<class Tdata>
    int alltoall(
		      parcel::vector<Tdata> &sendbuff_,int sendcount,
		      parcel::vector<Tdata> &recvbuff_,int recvcount
		      ){
      const auto Tdata_name = typeid(Tdata).name();
      if( Tdata_name == ddtype_name){
	double *sendbuff_hi = (double *)sendbuff_.data_hi();
	double *sendbuff_lo = (double *)sendbuff_.data_lo();
	double *recvbuff_hi = (double *)recvbuff_.data_hi();
	double *recvbuff_lo = (double *)recvbuff_.data_lo();
	int ierr_hi = mpi_alltoall<double>(sendbuff_hi,sendcount,recvbuff_hi,recvcount);
	int ierr_lo = mpi_alltoall<double>(sendbuff_lo,sendcount,recvbuff_lo,recvcount);
	return ierr_hi;
      }else{
	auto sendbuff = sendbuff_.data();
	auto recvbuff = recvbuff_.data();
	using Tdata_type = typename std::result_of< parcel::parcel_type_to_baseType( Tdata ) >::type;
	int ierr = mpi_alltoall<Tdata_type>(sendbuff,sendcount,recvbuff,recvcount);
	return ierr;
      }
      return -1;
    }
    
    template<class Tdata>
    int alltoall(
		      std::vector<Tdata> &sendbuff_,int sendcount,
		      std::vector<Tdata> &recvbuff_,int recvcount
		      ){
	auto sendbuff = sendbuff_.data();
	auto recvbuff = recvbuff_.data();
	int ierr = mpi_alltoall<Tdata>(sendbuff,sendcount,recvbuff,recvcount);
	return ierr;
    } 

    template<class Tdata>
    int allgather(
		      parcel::vector<Tdata> &sendbuff_,int sendcount,
		      parcel::vector<Tdata> &recvbuff_,int recvcount
		      ){
      const auto Tdata_name = typeid(Tdata).name();
      if( Tdata_name == ddtype_name){
	double *sendbuff_hi = (double *)sendbuff_.data_hi();
	double *sendbuff_lo = (double *)sendbuff_.data_lo();
	double *recvbuff_hi = (double *)recvbuff_.data_hi();
	double *recvbuff_lo = (double *)recvbuff_.data_lo();
	int ierr_hi = mpi_allgather<double>(sendbuff_hi,sendcount,recvbuff_hi,recvcount);
	int ierr_lo = mpi_allgather<double>(sendbuff_lo,sendcount,recvbuff_lo,recvcount);
	return ierr_hi;
      }else{
	auto sendbuff = sendbuff_.data();
	auto recvbuff = recvbuff_.data();
	using Tdata_type = typename std::result_of< parcel::parcel_type_to_baseType( Tdata ) >::type;
	int ierr = mpi_allgather<Tdata_type>(sendbuff,sendcount,recvbuff,recvcount);
	return ierr;
      }
      return -1;
    }
    
    template<class Tdata>
    int allgather(
		      std::vector<Tdata> &sendbuff_,int sendcount,
		      std::vector<Tdata> &recvbuff_,int recvcount
		      ){
	auto sendbuff = sendbuff_.data();
	auto recvbuff = recvbuff_.data();
	int ierr = mpi_allgather<Tdata>(sendbuff,sendcount,recvbuff,recvcount);
	return ierr;
    } 

    
  };


  class comm_ddm : public comm{
    std::vector<int> near_rank_; // 各次元近接ランク番号.-方向,+方向を持つ.
  public : 
    comm_ddm(){}
    ~comm_ddm(){}

    void set_comm_ddm(
		      MPI_Comm mpi_comm,
		      int dim,
		      std::vector<int> near_rank){
      near_rank_.resize(dim*2,0);
      for(int i=0;i<dim;i++){
	near_rank_[i] = near_rank[i];
      }
      set_comm(mpi_comm);
      return ;
    }

  };



}

#endif
