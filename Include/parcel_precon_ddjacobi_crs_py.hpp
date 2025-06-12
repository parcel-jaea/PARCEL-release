template<class T_vec,class T_mat,class T_work_vec>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    T_vec *x, // out   
			    T_vec *b, // in 
			    T_mat *A, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    T_mat *Dinv, // (D^-1) 
			    double *scaling_Inv,
			    T_work_vec *precon_x1_,  
			    T_work_vec *precon_x2_,  
			    T_work_vec *precon_b){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<T_vec,double>(N,b,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv[i];
	  const double Dinv_lo_ = 0.0;
	  const double b_hi_    = b[i];
	  const double b_lo_    = 0.0;
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b[i] = precon_b_hi_;
	}
      }
    });
  q_[0].wait();

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b[i];
	  double x_i_lo_ = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A[j];
	    const double A_lo_  = 0.0;
	    const double x_hi_  = precon_b[col];
	    const double x_lo_  = 0.0;
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x[i] = x_i_hi_;
	  }else{
	    x_next[i] = x_i_hi_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<T_work_vec>(&x_now,&x_next);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, x_now); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b[i];
	    double x_i_lo_ = 0.0;
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A[j];
	      const double A_lo_  = 0.0;
	      const double x_hi_  = x_now[col];
	      const double x_lo_  = 0.0;
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x[i] = x_i_hi_;
	    }else{
	      x_next[i] = x_i_hi_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		x[i] = x[i] * bnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_vec,class T_mat>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    T_vec *x, // out   
			    T_vec *b, // in 
			    T_mat *A, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    T_mat *Dinv, // (D^-1) 
			    double *scaling_Inv,
			    double *precon_x1_hi_,  
			    double *precon_x1_lo_,  
			    double *precon_x2_hi_,  
			    double *precon_x2_lo_,  
			    double *precon_b_hi, 
			    double *precon_b_lo ){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<T_vec,double>(N,b,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv[i];
	  const double Dinv_lo_ = 0.0;
	  const double b_hi_    = b[i];
	  const double b_lo_    = 0.0;
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b_hi[i] = precon_b_hi_;
	  precon_b_lo[i] = precon_b_lo_;
	}
      }
    });
  q_[0].wait();

  double *x_now_hi  = precon_x1_hi_;
  double *x_now_lo  = precon_x1_lo_;
  double *x_next_hi = precon_x2_hi_;
  double *x_next_lo = precon_x2_lo_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,precon_b_hi); 
       crsMat_->template SpMVcomm< double >(q_,precon_b_lo); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b_hi[i];
	  double x_i_lo_ = precon_b_lo[i];
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A[j];
	    const double A_lo_  = 0.0;
	    const double x_hi_  = precon_b_hi[col];
	    const double x_lo_  = precon_b_lo[col];
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x[i] = x_i_hi_;
	  }else{
	    x_next_hi[i] = x_i_hi_;
	    x_next_lo[i] = x_i_lo_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<double>(&x_now_hi,&x_next_hi);
    swap_ptr<double>(&x_now_lo,&x_next_lo);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,x_now_hi); 
       crsMat_->template SpMVcomm< double >(q_,x_now_lo); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b_hi[i];
	    double x_i_lo_ = precon_b_lo[i];
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A[j];
	      const double A_lo_  = 0.0;
	      const double x_hi_  = x_now_hi[col];
	      const double x_lo_  = x_now_lo[col];
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x[i] = x_i_hi_;
	    }else{
	      x_next_hi[i] = x_i_hi_;
	      x_next_lo[i] = x_i_lo_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		x[i] = x[i] * bnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_vec,class T_work_vec>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    T_vec *x, // out   
			    T_vec *b, // in 
			    double *A_hi, // (D^-1)(L+U) 
			    double *A_lo, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    double *Dinv_hi, // (D^-1) 
			    double *Dinv_lo, // (D^-1) 
			    double *scaling_Inv,
			    T_work_vec *precon_x1_,  
			    T_work_vec *precon_x2_,  
			    T_work_vec *precon_b){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<T_vec,double>(N,b,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv_hi[i];
	  const double Dinv_lo_ = Dinv_lo[i];
	  const double b_hi_    = b[i];
	  const double b_lo_    = 0.0;
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b[i] = precon_b_hi_;
	}
      }
    });
  q_[0].wait();

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b[i];
	  double x_i_lo_ = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A_hi[j];
	    const double A_lo_  = A_lo[j];
	    const double x_hi_  = precon_b[col];
	    const double x_lo_  = 0.0;
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x[i] = x_i_hi_;
	  }else{
	    x_next[i] = x_i_hi_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<T_work_vec>(&x_now,&x_next);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, x_now); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b[i];
	    double x_i_lo_ = 0.0;
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A_hi[j];
	      const double A_lo_  = A_lo[j];
	      const double x_hi_  = x_now[col];
	      const double x_lo_  = 0.0;
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x[i] = x_i_hi_;
	    }else{
	      x_next[i] = x_i_hi_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		x[i] = x[i] * bnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_vec>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    T_vec *x, // out   
			    T_vec *b, // in 
			    double *A_hi, // (D^-1)(L+U) 
			    double *A_lo, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    double *Dinv_hi, // (D^-1) 
			    double *Dinv_lo, // (D^-1) 
			    double *scaling_Inv,
			    double *precon_x1_hi_,  
			    double *precon_x1_lo_,  
			    double *precon_x2_hi_,  
			    double *precon_x2_lo_,  
			    double *precon_b_hi, 
			    double *precon_b_lo ){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<T_vec,double>(N,b,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv_hi[i];
	  const double Dinv_lo_ = Dinv_lo[i];
	  const double b_hi_    = b[i];
	  const double b_lo_    = 0.0;
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b_hi[i] = precon_b_hi_;
	  precon_b_lo[i] = precon_b_lo_;
	}
      }
    });
  q_[0].wait();

  double *x_now_hi  = precon_x1_hi_;
  double *x_now_lo  = precon_x1_lo_;
  double *x_next_hi = precon_x2_hi_;
  double *x_next_lo = precon_x2_lo_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,precon_b_hi); 
       crsMat_->template SpMVcomm< double >(q_,precon_b_lo); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b_hi[i];
	  double x_i_lo_ = precon_b_lo[i];
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A_hi[j];
	    const double A_lo_  = A_lo[j];
	    const double x_hi_  = precon_b_hi[col];
	    const double x_lo_  = precon_b_lo[col];
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x[i] = x_i_hi_;
	  }else{
	    x_next_hi[i] = x_i_hi_;
	    x_next_lo[i] = x_i_lo_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<double>(&x_now_hi,&x_next_hi);
    swap_ptr<double>(&x_now_lo,&x_next_lo);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,x_now_hi); 
       crsMat_->template SpMVcomm< double >(q_,x_now_lo); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b_hi[i];
	    double x_i_lo_ = precon_b_lo[i];
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A_hi[j];
	      const double A_lo_  = A_lo[j];
	      const double x_hi_  = x_now_hi[col];
	      const double x_lo_  = x_now_lo[col];
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x[i] = x_i_hi_;
	    }else{
	      x_next_hi[i] = x_i_hi_;
	      x_next_lo[i] = x_i_lo_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		x[i] = x[i] * bnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_mat,class T_work_vec>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    double *x_hi, // out   
			    double *x_lo, // out   
			    double *b_hi, // in 
			    double *b_lo, // in 
			    T_mat *A, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    T_mat *Dinv, // (D^-1) 
			    double *scaling_Inv,
			    T_work_vec *precon_x1_,  
			    T_work_vec *precon_x2_,  
			    T_work_vec *precon_b){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<double,double,double>(N,b_hi,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<double,double>(N,b_hi,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv[i];
	  const double Dinv_lo_ = 0.0;
	  const double b_hi_    = b_hi[i];
	  const double b_lo_    = b_lo[i];
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b[i] = precon_b_hi_;
	}
      }
    });
  q_[0].wait();

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b[i];
	  double x_i_lo_ = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A[j];
	    const double A_lo_  = 0.0;
	    const double x_hi_  = precon_b[col];
	    const double x_lo_  = 0.0;
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x_hi[i] = x_i_hi_;
	    x_lo[i] = x_i_lo_;
	  }else{
	    x_next[i] = x_i_hi_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<T_work_vec>(&x_now,&x_next);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, x_now); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b[i];
	    double x_i_lo_ = 0.0;
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A[j];
	      const double A_lo_  = 0.0;
	      const double x_hi_  = x_now[col];
	      const double x_lo_  = 0.0;
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x_hi[i] = x_i_hi_;
	      x_lo[i] = x_i_lo_;
	    }else{
	      x_next[i] = x_i_hi_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		double bnorm_hi = bnorm;
		double bnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(bnorm_hi,bnorm_lo,x_hi[i],x_lo[i],x_hi[i],x_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_mat>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    double *x_hi, // out   
			    double *x_lo, // out   
			    double *b_hi, // in 
			    double *b_lo, // in 
			    T_mat *A, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    T_mat *Dinv, // (D^-1) 
			    double *scaling_Inv,
			    double *precon_x1_hi_,  
			    double *precon_x1_lo_,  
			    double *precon_x2_hi_,  
			    double *precon_x2_lo_,  
			    double *precon_b_hi, 
			    double *precon_b_lo ){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<double,double,double>(N,b_hi,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<double,double>(N,b_hi,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv[i];
	  const double Dinv_lo_ = 0.0;
	  const double b_hi_    = b_hi[i];
	  const double b_lo_    = b_lo[i];
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b_hi[i] = precon_b_hi_;
	  precon_b_lo[i] = precon_b_lo_;
	}
      }
    });
  q_[0].wait();

  double *x_now_hi  = precon_x1_hi_;
  double *x_now_lo  = precon_x1_lo_;
  double *x_next_hi = precon_x2_hi_;
  double *x_next_lo = precon_x2_lo_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,precon_b_hi); 
       crsMat_->template SpMVcomm< double >(q_,precon_b_lo); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b_hi[i];
	  double x_i_lo_ = precon_b_lo[i];
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A[j];
	    const double A_lo_  = 0.0;
	    const double x_hi_  = precon_b_hi[col];
	    const double x_lo_  = precon_b_lo[col];
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x_hi[i] = x_i_hi_;
	    x_lo[i] = x_i_lo_;
	  }else{
	    x_next_hi[i] = x_i_hi_;
	    x_next_lo[i] = x_i_lo_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<double>(&x_now_hi,&x_next_hi);
    swap_ptr<double>(&x_now_lo,&x_next_lo);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,x_now_hi); 
       crsMat_->template SpMVcomm< double >(q_,x_now_lo); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b_hi[i];
	    double x_i_lo_ = precon_b_lo[i];
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A[j];
	      const double A_lo_  = 0.0;
	      const double x_hi_  = x_now_hi[col];
	      const double x_lo_  = x_now_lo[col];
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x_hi[i] = x_i_hi_;
	      x_lo[i] = x_i_lo_;
	    }else{
	      x_next_hi[i] = x_i_hi_;
	      x_next_lo[i] = x_i_lo_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		double bnorm_hi = bnorm;
		double bnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(bnorm_hi,bnorm_lo,x_hi[i],x_lo[i],x_hi[i],x_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}

template<class T_work_vec>
void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    double *x_hi, // out   
			    double *x_lo, // out   
			    double *b_hi, // in 
			    double *b_lo, // in 
			    double *A_hi, // (D^-1)(L+U) 
			    double *A_lo, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    double *Dinv_hi, // (D^-1) 
			    double *Dinv_lo, // (D^-1) 
			    double *scaling_Inv,
			    T_work_vec *precon_x1_,  
			    T_work_vec *precon_x2_,  
			    T_work_vec *precon_b){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<double,double,double>(N,b_hi,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<double,double>(N,b_hi,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv_hi[i];
	  const double Dinv_lo_ = Dinv_lo[i];
	  const double b_hi_    = b_hi[i];
	  const double b_lo_    = b_lo[i];
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b[i] = precon_b_hi_;
	}
      }
    });
  q_[0].wait();

  T_work_vec *x_now  = precon_x1_;
  T_work_vec *x_next = precon_x2_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b[i];
	  double x_i_lo_ = 0.0;
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A_hi[j];
	    const double A_lo_  = A_lo[j];
	    const double x_hi_  = precon_b[col];
	    const double x_lo_  = 0.0;
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x_hi[i] = x_i_hi_;
	    x_lo[i] = x_i_lo_;
	  }else{
	    x_next[i] = x_i_hi_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<T_work_vec>(&x_now,&x_next);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< T_work_vec >(q_, x_now); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b[i];
	    double x_i_lo_ = 0.0;
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A_hi[j];
	      const double A_lo_  = A_lo[j];
	      const double x_hi_  = x_now[col];
	      const double x_lo_  = 0.0;
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x_hi[i] = x_i_hi_;
	      x_lo[i] = x_i_lo_;
	    }else{
	      x_next[i] = x_i_hi_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		double bnorm_hi = bnorm;
		double bnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(bnorm_hi,bnorm_lo,x_hi[i],x_lo[i],x_hi[i],x_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}


void ddjacobi_method_crs_main( 
 			    sycl::queue *q_ , 
			    PARCEL_INT N, 
			    PARCEL_INT iterMax, 
			    double *x_hi, // out   
			    double *x_lo, // out   
			    double *b_hi, // in 
			    double *b_lo, // in 
			    double *A_hi, // (D^-1)(L+U) 
			    double *A_lo, // (D^-1)(L+U) 
			    PARCEL_INT *Rowptr, 
			    PARCEL_INT *crsCol, 
			    double *Dinv_hi, // (D^-1) 
			    double *Dinv_lo, // (D^-1) 
			    double *scaling_Inv,
			    double *precon_x1_hi_,  
			    double *precon_x1_lo_,  
			    double *precon_x2_hi_,  
			    double *precon_x2_lo_,  
			    double *precon_b_hi, 
			    double *precon_b_lo ){ 
#if _A64FX_HIPSYCL_ 
  const size_t nblock      = 12; 
  const size_t nlocalblock = 8192; 
#else 
  const size_t nblock      = N; 
  const size_t nlocalblock = 1; 
#endif 
const bool SCALING_MATRIX = SCALING_MATRIX_; 
const bool SCALING_VECTOR = SCALING_VECTOR_; 
#if PARCEL_MPI
      parcel::comm *comm = crsMat_->comm();
#endif      
double bnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  bnorm_ = max<double,double,double>(N,b_hi,N,scaling_Inv,N); 
	}else{ 
	  bnorm_ = max<double,double>(N,b_hi,N); 
	} 
	q_[0].wait(); 
} 
#if PARCEL_MPI
double bnorm = 1.0;
MPI_Allreduce(&bnorm_,&bnorm,1,MPI_DOUBLE,MPI_MAX,comm->mpi_comm());
#else
const double bnorm = bnorm_; 
#endif
  q_[0].wait();
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  const double Dinv_hi_ = Dinv_hi[i];
	  const double Dinv_lo_ = Dinv_lo[i];
	  const double b_hi_    = b_hi[i];
	  const double b_lo_    = b_lo[i];
	  const double bnorm_hi_    = bnorm;
	  const double bnorm_lo_    = 0.0;
	  double ddmul_var1;
	  double precon_b_hi_ = 0.0;
	  double precon_b_lo_ = 0.0;
	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);
	  if( SCALING_MATRIX ){  
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    double dddiv_var1; 
	    dddiv_calc1(scalinv_val_hi,scalinv_val_lo,bnorm_hi_,bnorm_lo_,scalinv_val_hi,scalinv_val_lo);
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  }else{ 
	    double scalinv_val_hi = scaling_Inv[i]; 
	    double scalinv_val_lo = 0.0; 
	    ddmul_calc1(precon_b_hi_,precon_b_lo_,scalinv_val_hi,scalinv_val_lo,precon_b_hi_,precon_b_lo_);
	  } 
	  precon_b_hi[i] = precon_b_hi_;
	  precon_b_lo[i] = precon_b_lo_;
	}
      }
    });
  q_[0].wait();

  double *x_now_hi  = precon_x1_hi_;
  double *x_now_lo  = precon_x1_lo_;
  double *x_next_hi = precon_x2_hi_;
  double *x_next_lo = precon_x2_lo_;
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,precon_b_hi); 
       crsMat_->template SpMVcomm< double >(q_,precon_b_lo); 
  #endif 
  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double x_i_hi_ = precon_b_hi[i];
	  double x_i_lo_ = precon_b_lo[i];
	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	    const PARCEL_INT col = crsCol[j];
	    const double A_hi_  = A_hi[j];
	    const double A_lo_  = A_lo[j];
	    const double x_hi_  = precon_b_hi[col];
	    const double x_lo_  = precon_b_lo[col];
	    double Ax_hi_ = 0.0;
	    double Ax_lo_ = 0.0;
	    double ddmul_var1;
	    double ddsub_var1;
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	  }
	  if( iterMax==1 ){
	    x_hi[i] = x_i_hi_;
	    x_lo[i] = x_i_lo_;
	  }else{
	    x_next_hi[i] = x_i_hi_;
	    x_next_lo[i] = x_i_lo_;
	  }
	}
      }
    });
  q_[0].wait();

  for(PARCEL_INT iter = 1;iter<iterMax;iter++){
    swap_ptr<double>(&x_now_hi,&x_next_hi);
    swap_ptr<double>(&x_now_lo,&x_next_lo);
  #if PARCEL_MPI 
       crsMat_->template SpMVcomm< double >(q_,x_now_hi); 
       crsMat_->template SpMVcomm< double >(q_,x_now_lo); 
  #endif 
    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	  for(size_t i=iii;i<imax;i++){
	    double x_i_hi_ = precon_b_hi[i];
	    double x_i_lo_ = precon_b_lo[i];
	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
	      const PARCEL_INT col = crsCol[j];
	      const double A_hi_  = A_hi[j];
	      const double A_lo_  = A_lo[j];
	      const double x_hi_  = x_now_hi[col];
	      const double x_lo_  = x_now_lo[col];
	      double Ax_hi_ = 0.0;
	      double Ax_lo_ = 0.0;
	      double ddmul_var1;
	      double ddsub_var1;
	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);
	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);
	    }
	    if( ( (iter+1)==iterMax) ){
	      x_hi[i] = x_i_hi_;
	      x_lo[i] = x_i_lo_;
	    }else{
	      x_next_hi[i] = x_i_hi_;
	      x_next_lo[i] = x_i_lo_;
	    }
	  }
	}
      });
    q_[0].wait();
  }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		double bnorm_hi = bnorm;
		double bnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(bnorm_hi,bnorm_lo,x_hi[i],x_lo[i],x_hi[i],x_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
  return ;

}


