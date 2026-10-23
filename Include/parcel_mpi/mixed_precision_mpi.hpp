// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISION_MPI_HPP_
#define MIXED_PRECISION_MPI_HPP_

#include <mpi.h>
#include <vector>
#include <map>
#include "../ddlib.hpp"

namespace mixed_precison{

#define MIXED_PRECISION_MPI_TAG_DEFAULT 99

  class mpi_common {

  public :

    mpi_common(){}
    ~mpi_common(){}
    int size(MPI_Comm mpi_comm){
      int num_procs = 1;
      MPI_Comm_size(mpi_comm, &num_procs);
      return num_procs;
    }
    int rank(MPI_Comm mpi_comm){
      int my_rank = 0;
      MPI_Comm_rank(mpi_comm, &my_rank);
      return my_rank;
    }
    
    template<class T>
    MPI_Request mpi_isend_(MPI_Comm mpi_comm,void *buff,int send_rank,int n,int tag=MIXED_PRECISION_MPI_TAG_DEFAULT){
      MPI_Request req;
      int mpi_type_size = sizeof(T); 
      int nsize = n*mpi_type_size;
      MPI_Isend(buff, nsize, MPI_BYTE, send_rank, tag, mpi_comm, &req);
      return req;
    }

    template<class T>
    int mpi_allgather_(MPI_Comm mpi_comm,
			void *sendbuff,int sendcount,
			void *recvbuff,int recvcount,
			int tag=MIXED_PRECISION_MPI_TAG_DEFAULT){
      MPI_Request req;
      int mpi_type_size = sizeof(T); 
      int nsize = sendcount*mpi_type_size;
      int ierr  = MPI_Allgather(
		    sendbuff,nsize,MPI_BYTE,
		    recvbuff,nsize,MPI_BYTE,
		    mpi_comm);      
      return ierr;
    }

    template<class T>
    int mpi_alltoall_(MPI_Comm mpi_comm,
			void *sendbuff,int sendcount,
			void *recvbuff,int recvcount,
			int tag=MIXED_PRECISION_MPI_TAG_DEFAULT){
      MPI_Request req;
      int mpi_type_size = sizeof(T); 
      int nsize = sendcount*mpi_type_size;
      int ierr  = MPI_Alltoall(
		    sendbuff,nsize,MPI_BYTE,
		    recvbuff,nsize,MPI_BYTE,
		    mpi_comm);      
      return ierr;
    }

    template<class T>
    MPI_Request mpi_irecv_(MPI_Comm mpi_comm,void *buff,int recv_rank,int n,int tag=MIXED_PRECISION_MPI_TAG_DEFAULT){
      MPI_Request req;
      int mpi_type_size = sizeof(T); 
      int nsize = n*mpi_type_size;
      MPI_Irecv(buff, nsize, MPI_BYTE, recv_rank, tag, mpi_comm, &req);
      return req;
    }

    MPI_Status mpi_wait_(MPI_Request &req){
      MPI_Status stat;
      MPI_Wait(&req,&stat);
      return stat;
    }

    template<class T>
    int mpi_bcast_(MPI_Comm mpi_comm,void *buff,int n,int root){
      int nprocess;
      int myrank;
#if PARCEL_MPI
      MPI_Comm_size(mpi_comm,&nprocess); 
      MPI_Comm_rank(mpi_comm,&myrank); 
#else
      nprocess=1;
      myrank=0;
#endif
      if(myrank==root){
	for(int i = 0;i<nprocess;i++){
	  if(i!=root){
	    MPI_Request req = mpi_isend_<T>(mpi_comm,buff,i,n,myrank);
	    MPI_Status stat = mpi_wait_(req);
	  }
	}
      }else{
	MPI_Request req = mpi_irecv_<T>(mpi_comm,buff,root,n,root);
	MPI_Status stat = mpi_wait_(req);
      }
      return 0;
    }


  };

  class reduce : public mpi_common{

    enum class MPI_DatatypeID : int{
      T_MPI_LONG_DOUBLE,
	T_MPI_DOUBLE,
	T_MPI_FLOAT,
	T_MPI_INT32_T,
	T_MPI_INT64_T,
	T_NULL,
	};

    const char* long_double_name = typeid(long double).name();
    const char* double_name      = typeid(double).name();
    const char* float_name       = typeid(float).name();
    const char* int32_t_name     = typeid(int32_t).name();
    const char* int64_t_name     = typeid(int64_t).name();

    std::map<MPI_DatatypeID,MPI_Datatype> mpi_data_type = { 
      {MPI_DatatypeID::T_MPI_LONG_DOUBLE , MPI_LONG_DOUBLE},
      {MPI_DatatypeID::T_MPI_DOUBLE      , MPI_DOUBLE},
      {MPI_DatatypeID::T_MPI_FLOAT       , MPI_FLOAT},
      {MPI_DatatypeID::T_MPI_INT32_T     , MPI_INT32_T},
      {MPI_DatatypeID::T_MPI_INT64_T     , MPI_INT64_T},
    };

  public :

    reduce(){}
    ~reduce(){}

    template< class T> 
    MPI_DatatypeID Ttype_MPItype(){

      const auto T_name = typeid(T).name();

      if( T_name == long_double_name ){ return MPI_DatatypeID::T_MPI_LONG_DOUBLE; }
      if( T_name == double_name  )    { return MPI_DatatypeID::T_MPI_DOUBLE;  }
      if( T_name == float_name   )    { return MPI_DatatypeID::T_MPI_FLOAT;   }
      if( T_name == int32_t_name )    { return MPI_DatatypeID::T_MPI_INT32_T; }
      if( T_name == int64_t_name )    { return MPI_DatatypeID::T_MPI_INT64_T; }

      return MPI_DatatypeID::T_NULL;

    }

    // 演算精度とデータ精度は一致
    template<class T>
    void mpi_allreduce_(MPI_Comm mpi_comm,T *sbuf,T *rbuf,int n, MPI_Op op){
      const auto T_name = typeid(T).name();

      MPI_DatatypeID mpi_data_id = Ttype_MPItype<T>();
      if(MPI_DatatypeID::T_NULL != mpi_data_id){
	MPI_Datatype mpi_type = mpi_data_type[mpi_data_id];
	int ierr = MPI_Allreduce(sbuf,rbuf,n,mpi_type,op,mpi_comm);
      }else{
	// 未対応のデータ型PARCELの最大浮動小数点型を利用
	std::vector<PARCEL_MAX_FLOTING_TYPE> sbuf_(n);
	std::vector<PARCEL_MAX_FLOTING_TYPE> rbuf_(n);
	for(int i=0;i<n;i++){sbuf_[i] = sbuf[i];}
	int ierr = MPI_Allreduce(sbuf_.data(),rbuf_.data(),n,PARCEL_MPI_MAXFLOTING_TYPE,op,mpi_comm);
	for(int i=0;i<n;i++){rbuf[i]  = rbuf_[i];}
      }
      return ;
    }

    ddlib::ddtype ddMPI_Op(
			   double ddbuf_hi,double ddbuf_lo,
			   double Tbuf_hi ,double Tbuf_lo,
			   MPI_Op op){
      ddlib::ddtype ddx(ddbuf_hi,ddbuf_lo);
      ddlib::ddtype ddy(Tbuf_hi,Tbuf_lo);
      ddlib::ddtype ddbuf(0.0,0.0);

      // MPI_SUM
      // MPI_PROD
      // MPI_MAX
      // MPI_MIN

      if(op==MPI_SUM) { ddbuf = ddx+ddy; return ddbuf;}
      if(op==MPI_PROD){ ddbuf = ddx*ddy; return ddbuf;}

      if(op==MPI_MAX){ 
	if( ddx.get_hi() > ddy.get_hi() ){
	  ddbuf = ddx;
	}else{
	  if( ddx.get_hi() == ddy.get_hi() ){
	    if( ddx.get_lo() > ddy.get_lo() ){
	      ddbuf = ddx;
	    }else{
	      ddbuf = ddy;
	    }
	  }else{
	    ddbuf = ddy;
	  }
	}
	return ddbuf;
      }

      if(op==MPI_MIN){ 
	if( ddx.get_hi() < ddy.get_hi() ){
	  ddbuf = ddx;
	}else{
	  if( ddx.get_hi() == ddy.get_hi() ){
	    if( ddx.get_lo() < ddy.get_lo() ){
	      ddbuf = ddx;
	    }else{
	      ddbuf = ddy;
	    }
	  }else{
	    ddbuf = ddy;
	  }
	}
	return ddbuf;
      }
      std::cout<<"mixed_precison double-double reduce support is : "<<
	"MPI_SUM,MPI_PROD,MPI_MAX,MPI_MIN"
	       <<std::endl;
      return ddbuf;
    }

#include "mixed_precision_mpi_ddallreduce.hpp"
    
  };

  class comm : public reduce{
    MPI_Comm comm_;
    int rank_ = -1;
    int nprocess_ = -1;
    MPI_Status stat_send,stat_recv;
  public :

    void set_comm(MPI_Comm mpi_comm){
      comm_ = mpi_comm;
      int nprocess  = -1;
      int rank = -1;
#if PARCEL_MPI
      MPI_Comm_size(mpi_comm,&nprocess); 
      MPI_Comm_rank(mpi_comm,&rank); 
#else
      nprocess=1;
      rank=0;
#endif
      rank_     = rank;
      nprocess_ = nprocess;
      return ;
    }
    comm(){};
    comm(MPI_Comm comm){
      set_comm(comm);
    };

    ~comm(){};
    int mpi_rank(){
      return rank_;
    }

    int mpi_size(){
      return nprocess_;
    }
    
    MPI_Comm mpi_comm(){
      return comm_;
    }
    
    void mpi_barrier(){
      MPI_Barrier(comm_);
      return ;
    }
    template<class T>
    void mpi_allreduce(T *sbuf,T *rbuf,int n, MPI_Op op){
      mpi_allreduce_<T>(comm_,sbuf,rbuf,n,op);
      return ;
    }

    template<class T>
    void mpi_ddallreduce(T *sbuf,T *rbuf,int n, MPI_Op op){
      mpi_ddallreduce_<T>(comm_,sbuf,rbuf,n,op);
      return ;
    }

    template<class T>
    void mpi_ddallreduce(
		     double *sbuf_hi,double *sbuf_lo,
		     T      *rbuf,bool rbuf_null,
		     int n, MPI_Op op){
      mpi_ddallreduce_<T>(comm_,sbuf_hi,sbuf_lo,rbuf,rbuf_null,n,op);
      return ;
    }

    template<class T>
    void mpi_ddallreduce(
		     T      *sbuf,bool sbuf_null,
		     double *rbuf_hi,double *rbuf_lo,
		     int n, MPI_Op op){
      mpi_ddallreduce_<T>(comm_,sbuf,sbuf_null,rbuf_hi,rbuf_lo,n,op);
      return ;
    }


    void mpi_ddallreduce(double *sbuf_hi,double *sbuf_lo,double *rbuf_hi,double *rbuf_lo,int n, MPI_Op op){
      mpi_ddallreduce_(comm_,sbuf_hi,sbuf_lo,rbuf_hi,rbuf_lo,n,op);
      return ;
    }

    template<class T>
    int mpi_bcast(void *buff,int root,int n){
      int ierr = mpi_bcast_<T>(comm_,buff,n,root);
      return ierr;
    }

    template<class T>
    int mpi_allgather(void *sendbuff,int sendcount,void *recvbuff,int recvcount){
      int ierr = mpi_allgather_<T>(comm_,sendbuff,sendcount,recvbuff,recvcount);
      return ierr;
    }

    template<class T>
    int mpi_alltoall(void *sendbuff,int sendcount,void *recvbuff,int recvcount){
      int ierr = mpi_alltoall_<T>(comm_,sendbuff,sendcount,recvbuff,recvcount);
      return ierr;
    }
    
    template<class T>
    MPI_Request mpi_isend(void *buff,int send_rank,int n){
      MPI_Request req = mpi_isend_<T>(comm_,buff,send_rank,n,rank_);    
      return req;
    }

    template<class T>
    MPI_Request mpi_irecv(void *buff,int recv_rank,int n){
      MPI_Request req = mpi_irecv_<T>(comm_,buff,recv_rank,n,recv_rank);
      return req;
    }

    MPI_Status mpi_wait(MPI_Request &req){
      MPI_Status stat = mpi_wait_(req);
      return stat;
    }
    
  };

}

#endif
