namespace parcel{
  namespace qr{
    // 計算精度を指定
    template<
      class TcalcQR  = parcel_double // グラムシュミット直交化精度
      >
    class choleskyqr{
      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;
    public :
      choleskyqr(){}
      ~choleskyqr(){}
      template<
	class Tvec, // QR分解する入力ベクトルV精度
	class Tbasis, // QR分解結果の直交ベクトルQ精度
	class Trmatrix // 上三角行列精度
	>      
      void solver(
		  parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
		  PARCEL_INT n_, // ベクトルサイズ
		  parcel::vector<Tvec> &V,PARCEL_INT ldv,
		  parcel::vector<Tbasis> &Q,PARCEL_INT ldq,
		  parcel::vector<Trmatrix> &R,PARCEL_INT ldr,
		  std::string Qflag = "Y"		  
		  ){
	//	std::cout<<"    main CholeskyQR :: "<<std::endl;

#if PARCEL_MPI
	parcel::comm &parcel_comm = parcel_blas.parcel_comm();
#endif

	const PARCEL_INT n = n_;
	const PARCEL_INT s = V.size() / ldv; // 直交化するベクトル数
	
	const auto TcalcQR_name  = typeid(TcalcQR).name();
	using QR_Type = typename std::result_of<parcel::parcel_type_to_baseType( TcalcQR ) >::type;

	sycl::queue *que   = V.queue();
	parcel::vector<TcalcQR> one;one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);
	parcel::vector<TcalcQR> nega_one;nega_one.resize(1,que);
	double nega_one_ = -1.0;nega_one.setValue(&nega_one_);
	parcel::vector<TcalcQR> zero;zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);
	
	// 三角行列のゼロ初期化
	parcel_blas.zero_init<Trmatrix>(R);
 	que[0].wait();
	//	std::cout<<"CholeskyQR :: START GEMM "<<std::endl;

	// <V|V>計算
	// C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if PARCEL_MPI
	// ALL REDUCE
	parcel::vector<Trmatrix> R_(R.size(),que);parcel_blas.zero_init<Trmatrix>(R_);
	parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Tvec,Trmatrix,TcalcQR>(
								s, // m
								s, // n
								n, // k
								one,zero,
								V,ldv, 
								V,ldv,
								R_,s,
#if 1
								// 対称性GEMMでの実装でA100で問題有り
								"T","N","N",
								"S");	
#else
								"T","N","N"
								);
#endif
 	que[0].wait();
	parcel_comm.allreduce<Trmatrix,TcalcQR>(R_,R,s*s,MPI_SUM);
#else
	parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Tvec,Trmatrix,TcalcQR>(
								s, // m
								s, // n
								n, // k
								one,zero,
								V,ldv, 
								V,ldv,
								R,s,
								"T","N","N",
								"S");
	
#endif
      //	std::cout<<"CholeskyQR :: END GEMM "<<std::endl;

	// コレスキー分解
	std::vector< parcel::scalar > Rhost(R.size());
	R.getValue(Rhost);
	std::vector< parcel::scalar > Lmat(Rhost.size());
	for(PARCEL_INT i = 0;i<Rhost.size();i++){
	  Lmat[i] = Rhost[i];
	  //	  std::cout<<"     CholeskyQR Lmat :: "<<i<<" : "<< Lmat[i].get_T()<<std::endl;
	}       
	parcel_blas_host.potrf(Lmat,s,s);
	
	// for(PARCEL_INT i = 0;i<R.size();i++){
	//   Rhost[i] = Lmat[i].get_T();
	// }
		
	for(PARCEL_INT i = 0;i<R.size();i++){
	  Rhost[i] = 0.0;
	}
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<=j;i++){
	    Rhost[i+j*s] = Lmat[j+i*s].get_T();
	  }
	}
	// for(PARCEL_INT i = 0;i<Rhost.size();i++){
	//   std::cout<<"     CholeskyQR Rmat :: "<<i<<" : "<< Lmat[i].get_T()<<std::endl;
	// }       
	R.setValue(Rhost.data());

	// 直交行列計算
	if( Qflag == "Y"){
	  std::vector< parcel::scalar > Rmat_inv(R.size());  
	  for(PARCEL_INT i = 0;i<Rmat_inv.size();i++){
	    Rmat_inv[i] = 0.0;
	  }       
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<=j;i++){
	      Rmat_inv[i+j*s] = Lmat[j+i*s].get_T();
	    }
	  }	
	  parcel_blas_host.trtri("U","N",Rmat_inv,s,s);
	  for(PARCEL_INT i = 0;i<Rhost.size();i++){
	    Rhost[i] = Rmat_inv[i].get_T();
	    //	  std::cout<<"Rinv : "<<i<<","<<" : "<<Rhost[i]<<std::endl;
	  }	
	  parcel::vector<Trmatrix> Rinv;Rinv.resize(R.size(),que);
	  Rinv.setValue(Rhost.data());
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	  parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Trmatrix,Tvec,TcalcQR>(
								       n, // m
								       s, // n
								       s, // k
								       one,zero,
								       V,ldv, 
								       Rinv,s,
								       Q,ldq);
	}
	
	return ;
	
      }
      
  };

    // 計算精度を指定
    template<
      class TcalcQR  = parcel_double // グラムシュミット直交化精度
      >
    class cagmers_choleskyqr{
      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;
    public :
      cagmers_choleskyqr(){}
      ~cagmers_choleskyqr(){}
      template<
	class Tvec, // QR分解する入力ベクトルV精度
	class Tbasis, // QR分解結果の直交ベクトルQ精度
	class Trmatrix // 上三角行列精度
	>      
      void solver(
		  parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
		  PARCEL_INT n_, // ベクトルサイズ
		  parcel::vector<Tvec> &VcholQR, 
		  parcel::vector<Tbasis> &Vbasis,
		  parcel::vector<Trmatrix> &R
		  ){
	
	// VcholQR^T * Vbasis --> R
	const PARCEL_INT n = n_;
	const PARCEL_INT s_half = Vbasis.size() / n; // 直交化するベクトル数

	const auto TcalcQR_name  = typeid(TcalcQR).name();
	using QR_Type = typename std::result_of<parcel::parcel_type_to_baseType( TcalcQR ) >::type;

	sycl::queue *que   = VcholQR.queue();
	parcel::vector<TcalcQR> one;one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);
	parcel::vector<TcalcQR> nega_one;nega_one.resize(1,que);
	double nega_one_ = -1.0;nega_one.setValue(&nega_one_);
	parcel::vector<TcalcQR> zero;zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);
	
	// 三角行列のゼロ初期化
	parcel_blas.zero_init<Trmatrix>(R);
	que[0].wait();

	// <Vchol|Vbasis>計算
	// C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	//	std::cout<<"s_half = "<<s_half<<std::endl;
	parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Tvec,Trmatrix,TcalcQR>(
								4,  // m
								s_half, // n
								n,  // k
								one,zero,
								VcholQR ,n, 
								Vbasis,n,
								R,4,
								"T","N","N"
								);

	// コレスキー分解
	std::vector< parcel::scalar > Rhost(R.size());
	std::vector< parcel::scalar > Rtmp(Rhost.size());
	std::vector< parcel::scalar > Lmat(Rhost.size());
	R.getValue(Rhost);
	 // for(PARCEL_INT i = 0;i<Rhost.size();i++){
	 //   std::cout<<i<<" - "<<Rhost[i].get_T()<<std::endl;
	 // }
	for(PARCEL_INT i = 0;i<s_half;i++){
	  Rtmp[i*2]   = Rhost[i*4+0];
	  Rtmp[i*2+1] = Rhost[i*4+1];
	}
	Rtmp[s_half*2] = Rhost[1*4+3];
	for(PARCEL_INT i = 2;i<s_half;i++){
	  Rtmp[(i-2)*2   + s_half*2+1] = Rhost[i*4+2];
	  Rtmp[(i-2)*2+1 + s_half*2+1] = Rhost[i*4+3];
	}
	const PARCEL_INT s =s_half*2-1;
	for(PARCEL_INT i = 0;i<s;i++){
	  for(PARCEL_INT j = i;j<s;j++){
	    Rhost[i+j*s] = Rtmp[i*2+(j-i)];
	    Rhost[j+i*s] = Rhost[i+j*s];
	  }
	}

	for(PARCEL_INT i = 0;i<Rhost.size();i++){
	  Lmat[i] = Rhost[i];
	  //	  std::cout<<"CAGMRES - CholeskyQR :: "<<i<<" : "<< Lmat[i].get_T()<<std::endl;
	}       
	parcel_blas_host.potrf(Lmat,s,s);
	
	// for(PARCEL_INT i = 0;i<R.size();i++){
	//   Rhost[i] = Lmat[i].get_T();
	// }
		
	for(PARCEL_INT i = 0;i<R.size();i++){
	  Rhost[i] = 0.0;
	}
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<=j;i++){
	    Rhost[i+j*s] = Lmat[j+i*s].get_T();
	  }
	}
	R.setValue(Rhost.data());

	return ;
	
      }
      
  };
    
}
}
