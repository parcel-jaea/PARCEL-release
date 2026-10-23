// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
// データの場所によらずMPI_send,MPI_recvによりホスト側に落としてホスト側で計算
template<class T>
void mpi_ddallreduce_(MPI_Comm mpi_comm,T *sbuf,T *rbuf,int n, MPI_Op op){
  // 
  int nprocess;
  int myrank;

#if PARCEL_MPI
  MPI_Comm_size(mpi_comm,&nprocess); 
  MPI_Comm_rank(mpi_comm,&myrank); 
#else
      nprocess=1;
      myrank=0;
#endif

  if(myrank==0){

    std::vector<T> rbuf_host(n,0.0);
    std::vector<T> sbuf_host(n,0.0);
    std::vector<double> ddbuf_hi(n,0.0);
    std::vector<double> ddbuf_lo(n,0.0);
    {
      MPI_Request sreq = mpi_isend_<T>(mpi_comm,sbuf     ,0,n,0);
      MPI_Request rreq = mpi_irecv_<T>(mpi_comm,rbuf_host.data(),0,n,0);
      mpi_wait_(sreq);
      mpi_wait_(rreq);
    }

    for(int i=0;i<n;i++){
      ddbuf_hi[i] = rbuf_host[i]; 
      ddbuf_lo[i] = 0.0;
    }
    for(int recv_rank=1;recv_rank<nprocess;recv_rank++){
      MPI_Request req = mpi_irecv_<T>(mpi_comm,rbuf_host.data(),recv_rank,n,recv_rank);
      mpi_wait_(req);
      double rbuf_host_lo_ = 0.0;
      for(int i=0;i<n;i++){
	// op に応じた処理を実行
	double rbuf_host_hi_ = rbuf_host[i];
	ddlib::ddtype ddbuf = ddMPI_Op(
					  ddbuf_hi[i],ddbuf_lo[i],
					  rbuf_host_hi_,rbuf_host_lo_,
					  op);
	ddbuf_hi[i] = ddbuf.get_hi();
	ddbuf_lo[i] = ddbuf.get_lo();
      }
    }

    for(int i=0;i<n;i++){
      sbuf_host[i] = ddbuf_hi[i] + ddbuf_lo[i];
    }

    {
      MPI_Request sreq = mpi_isend_<T>(mpi_comm,sbuf_host.data(),0,n,0);
      MPI_Request rreq = mpi_irecv_<T>(mpi_comm,rbuf     ,0,n,0);
      mpi_wait_(sreq);
      mpi_wait_(rreq);
    }

  }else{
    MPI_Request req = mpi_isend_<T>(mpi_comm,sbuf,0,n,myrank);
    mpi_wait_(req);
  }

  // 計算結果を全プロセスに送信
  mpi_bcast_<T>(mpi_comm,rbuf,n,0);

  return ;

}

template<class T>
void mpi_ddallreduce_(MPI_Comm mpi_comm,
		  T *sbuf,bool sbuf_lo_null,
		  double *rbuf_hi,double *rbuf_lo,	
		  int n, MPI_Op op){

  int nprocess;
  int myrank;

#if PARCEL_MPI
  MPI_Comm_size(mpi_comm,&nprocess); 
  MPI_Comm_rank(mpi_comm,&myrank); 
#else
      nprocess=1;
      myrank=0;
#endif

  if(myrank==0){

    std::vector<T> rbuf_host(n,0.0);
    std::vector<double> ddbuf_hi(n,0.0);
    std::vector<double> ddbuf_lo(n,0.0);
    {
      MPI_Request sreq_hi = mpi_isend_<T>(mpi_comm,sbuf            ,0,n,0);
      MPI_Request rreq_hi = mpi_irecv_<T>(mpi_comm,rbuf_host.data(),0,n,0);
      mpi_wait_(sreq_hi);
      mpi_wait_(rreq_hi);
    }

    for(int i=0;i<n;i++){
      ddbuf_hi[i] = rbuf_host[i]; 
      ddbuf_lo[i] = 0.0; 
    }
    for(int recv_rank=1;recv_rank<nprocess;recv_rank++){
      MPI_Request req_hi = mpi_irecv_<T>(mpi_comm,rbuf_host.data(),recv_rank,n,recv_rank);
      mpi_wait_(req_hi);
      double rbuf_host_lo = 0.0;
      for(int i=0;i<n;i++){
	// op に応じた処理を実行
	ddlib::ddtype ddbuf = ddMPI_Op(
				       ddbuf_hi[i],ddbuf_lo[i],
				       rbuf_host[i],rbuf_host_lo,
				       op);
	ddbuf_hi[i] = ddbuf.get_hi();
	ddbuf_lo[i] = ddbuf.get_lo();
      }
    }

    std::vector<double> sbuf_host_hi(n,0.0);
    std::vector<double> sbuf_host_lo(n,0.0);
    for(int i=0;i<n;i++){
      sbuf_host_hi[i] = ddbuf_hi[i];
      sbuf_host_lo[i] = ddbuf_lo[i];
    }

    {
      MPI_Request sreq_hi = mpi_isend_<double>(mpi_comm,sbuf_host_hi.data(),0,n,0);
      MPI_Request rreq_hi = mpi_irecv_<double>(mpi_comm,rbuf_hi     ,0,n,0);
      mpi_wait_(sreq_hi);
      mpi_wait_(rreq_hi);

      MPI_Request sreq_lo = mpi_isend_<double>(mpi_comm,sbuf_host_lo.data(),0,n,0);
      MPI_Request rreq_lo = mpi_irecv_<double>(mpi_comm,rbuf_lo     ,0,n,0);
      mpi_wait_(sreq_lo);
      mpi_wait_(rreq_lo);
    }
  }else{
    MPI_Request req = mpi_isend_<T>(mpi_comm,sbuf,0,n,myrank);
    mpi_wait_(req);
  }

  // 計算結果を全プロセスに送信
  mpi_bcast_<double>(mpi_comm,rbuf_hi,n,0);
  mpi_bcast_<double>(mpi_comm,rbuf_lo,n,0);

  return ;

}

template<class T>
void mpi_ddallreduce_(MPI_Comm mpi_comm,
		  double *sbuf_hi,double *sbuf_lo,
		  T *rbuf        ,bool rbuf_lo_null,
		  int n, MPI_Op op){
  // 
  int nprocess;
  int myrank;

#if PARCEL_MPI
  MPI_Comm_size(mpi_comm,&nprocess); 
  MPI_Comm_rank(mpi_comm,&myrank); 
#else
      nprocess=1;
      myrank=0;
#endif

  if(myrank==0){

    std::vector<double> rbuf_host_hi(n,0.0);
    std::vector<double> rbuf_host_lo(n,0.0);
    std::vector<double> ddbuf_hi(n,0.0);
    std::vector<double> ddbuf_lo(n,0.0);
    {
      MPI_Request sreq_hi = mpi_isend_<double>(mpi_comm,sbuf_hi     ,0,n,0);
      MPI_Request rreq_hi = mpi_irecv_<double>(mpi_comm,rbuf_host_hi.data(),0,n,0);
      mpi_wait_(sreq_hi);
      mpi_wait_(rreq_hi);
      MPI_Request sreq_lo = mpi_isend_<double>(mpi_comm,sbuf_lo     ,0,n,0);
      MPI_Request rreq_lo = mpi_irecv_<double>(mpi_comm,rbuf_host_lo.data(),0,n,0);
      mpi_wait_(sreq_lo);
      mpi_wait_(rreq_lo);
    }

    for(int i=0;i<n;i++){
      ddbuf_hi[i] = rbuf_host_hi[i]; 
      ddbuf_lo[i] = rbuf_host_lo[i]; 
    }
    for(int recv_rank=1;recv_rank<nprocess;recv_rank++){
      MPI_Request req_hi = mpi_irecv_<double>(mpi_comm,rbuf_host_hi.data(),recv_rank,n,recv_rank);
      mpi_wait_(req_hi);
      MPI_Request req_lo = mpi_irecv_<double>(mpi_comm,rbuf_host_lo.data(),recv_rank,n,recv_rank);
      mpi_wait_(req_lo);
      for(int i=0;i<n;i++){
	// op に応じた処理を実行
	ddlib::ddtype ddbuf = ddMPI_Op(
					       ddbuf_hi[i],ddbuf_lo[i],
					       rbuf_host_hi[i],rbuf_host_lo[i],
					       op);
	ddbuf_hi[i] = ddbuf.get_hi();
	ddbuf_lo[i] = ddbuf.get_lo();
      }
    }

    std::vector<T> sbuf_host_hi(n,0.0);
    for(int i=0;i<n;i++){
      sbuf_host_hi[i] = ddbuf_hi[i];
    }

    {
      MPI_Request sreq_hi = mpi_isend_<T>(mpi_comm,sbuf_host_hi.data(),0,n,0);
      mpi_wait_(sreq_hi);
    }

  }else{
    MPI_Request req_hi = mpi_isend_<double>(mpi_comm,sbuf_hi,0,n,myrank);
    mpi_wait_(req_hi);
    MPI_Request req_lo = mpi_isend_<double>(mpi_comm,sbuf_lo,0,n,myrank);
    mpi_wait_(req_lo); 
  }

  // 計算結果を全プロセスに送信
  mpi_bcast_<T>(mpi_comm,rbuf,n,0);

  return ;

}

void mpi_ddallreduce_(MPI_Comm mpi_comm,
		  double *sbuf_hi,double *sbuf_lo,
		  double *rbuf_hi,double *rbuf_lo,	
		  int n, MPI_Op op){
  // 
  int nprocess;
  int myrank;

#if PARCEL_MPI
  MPI_Comm_size(mpi_comm,&nprocess); 
  MPI_Comm_rank(mpi_comm,&myrank); 
#else
      nprocess=1;
      myrank=0;
#endif

  if(myrank==0){

    std::vector<double> rbuf_host_hi(n,0.0);
    std::vector<double> rbuf_host_lo(n,0.0);
    std::vector<double> sbuf_host_hi(n,0.0);
    std::vector<double> sbuf_host_lo(n,0.0);
    std::vector<double> ddbuf_hi(n,0.0);
    std::vector<double> ddbuf_lo(n,0.0);
    {
      MPI_Request sreq_hi = mpi_isend_<double>(mpi_comm,sbuf_hi     ,0,n,0);
      MPI_Request rreq_hi = mpi_irecv_<double>(mpi_comm,rbuf_host_hi.data(),0,n,0);
      mpi_wait_(sreq_hi);
      mpi_wait_(rreq_hi);
      MPI_Request sreq_lo = mpi_isend_<double>(mpi_comm,sbuf_lo     ,0,n,0);
      MPI_Request rreq_lo = mpi_irecv_<double>(mpi_comm,rbuf_host_lo.data(),0,n,0);
      mpi_wait_(sreq_lo);
      mpi_wait_(rreq_lo);
    }

    for(int i=0;i<n;i++){
      ddbuf_hi[i] = rbuf_host_hi[i]; 
      ddbuf_lo[i] = rbuf_host_lo[i]; 
    }
    for(int recv_rank=1;recv_rank<nprocess;recv_rank++){
      MPI_Request req_hi = mpi_irecv_<double>(mpi_comm,rbuf_host_hi.data(),recv_rank,n,recv_rank);
      mpi_wait_(req_hi);
      MPI_Request req_lo = mpi_irecv_<double>(mpi_comm,rbuf_host_lo.data(),recv_rank,n,recv_rank);
      mpi_wait_(req_lo);
      for(int i=0;i<n;i++){
	// op に応じた処理を実行
	ddlib::ddtype ddbuf = ddMPI_Op(
				       ddbuf_hi[i],ddbuf_lo[i],
				       rbuf_host_hi[i],rbuf_host_lo[i],
				       op);
	ddbuf_hi[i] = ddbuf.get_hi();
	ddbuf_lo[i] = ddbuf.get_lo();
      }
    }

    for(int i=0;i<n;i++){
      sbuf_host_hi[i] = ddbuf_hi[i];
      sbuf_host_lo[i] = ddbuf_lo[i];
    }

    {
      MPI_Request sreq_hi = mpi_isend_<double>(mpi_comm,sbuf_host_hi.data(),0,n,0);
      MPI_Request rreq_hi = mpi_irecv_<double>(mpi_comm,rbuf_hi     ,0,n,0);
      mpi_wait_(sreq_hi);
      mpi_wait_(rreq_hi);

      MPI_Request sreq_lo = mpi_isend_<double>(mpi_comm,sbuf_host_lo.data(),0,n,0);
      MPI_Request rreq_lo = mpi_irecv_<double>(mpi_comm,rbuf_lo     ,0,n,0);
      mpi_wait_(sreq_lo);
      mpi_wait_(rreq_lo);
    }
  }else{
    MPI_Request req_hi = mpi_isend_<double>(mpi_comm,sbuf_hi,0,n,myrank);
    mpi_wait_(req_hi);
    MPI_Request req_lo = mpi_isend_<double>(mpi_comm,sbuf_lo,0,n,myrank);
    mpi_wait_(req_lo); 
  }

  // 計算結果を全プロセスに送信
  mpi_bcast_<double>(mpi_comm,rbuf_hi,n,0);
  mpi_bcast_<double>(mpi_comm,rbuf_lo,n,0);

  return ;

}
