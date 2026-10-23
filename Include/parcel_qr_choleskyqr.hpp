// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
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
								// 対称性GEMMでの実装でA100で問題有り
								"T","N","N",
								"S");
	
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

	// コレスキー分解
	std::vector< parcel::scalar > Rhost(R.size());
	R.getValue(Rhost);
	std::vector< parcel::scalar > Lmat(Rhost.size());
	for(PARCEL_INT i = 0;i<Rhost.size();i++){
	  Lmat[i] = Rhost[i];
	}       
	parcel_blas_host.potrf(Lmat,s,s);
	
	for(PARCEL_INT i = 0;i<R.size();i++){
	  Rhost[i] = 0.0;
	}
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<=j;i++){
	    Rhost[i+j*s] = Lmat[j+i*s].get_T();
	  }
	}
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
	  }	
	  parcel::vector<Trmatrix> Rinv;Rinv.resize(R.size(),que);
	  Rinv.setValue(Rhost.data());
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]

	  // 縦長行列GEMM
	  parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Trmatrix,Tvec,TcalcQR>(
								       n, // m
								       s, // n
								       s, // k
								       one,zero,
								       V,ldv, 
								       Rinv,s,
								       Q,ldq,
								       "N","N","N",
								       "N","T");
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
      // CA-GMRES (単基底 v_{i+1} = A v_i、前処理なし) 専用の CholeskyQR。A が対称なら
      //   (v_i, v_j) = v_0^T A^{i+j} v_0 = mu_{i+j} なので、グラム行列 G ((s+1)x(s+1)) はハンケル行列になる。
      //   VcholQR = [v_0, v_1, v_{s-1}, v_s] (4 本) と Vbasis = [v_0, v_2, ..., v_s] (偶数番目 s/2+1 本) の
      //   内積 4 x (s/2+1) 個から mu_0 .. mu_{2s} を並べ、G = R^T R の上三角 R ((s+1)x(s+1), ld = s+1) を R に返す。
      //   Q は作らない。R は (s+1)^2 要素以上 (作業に 4*(s/2+1) 要素も使う)。
      //   ldv / ldq は VcholQR / Vbasis の列の間隔。MPI では局所の内積を allreduce する。
      //   VcholQR (Tvec) と Vbasis (Tbasis) の精度は違ってよい。
      void solver(
		  parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
		  PARCEL_INT n_, // ベクトルサイズ
		  parcel::vector<Tvec> &VcholQR, PARCEL_INT ldv,
		  parcel::vector<Tbasis> &Vbasis, PARCEL_INT ldq,
		  parcel::vector<Trmatrix> &R
		  ){
	
	// VcholQR^T * Vbasis --> R
	const PARCEL_INT n = n_;
	const PARCEL_INT s_half = (ldq > 0) ? (PARCEL_INT)(Vbasis.size() / ldq) : 0; // 偶数番目の基底の本数 (s/2+1)
	{
	  const size_t s1 = (size_t)(2*s_half - 1);   // s+1
	  if(s_half < 2 || (size_t)R.size() < s1*s1 || (size_t)R.size() < (size_t)4*s_half){
	    throw std::invalid_argument("parcel::qr::cagmers_choleskyqr::solver: need s >= 2 (even) and "
					"R of at least (s+1)^2 elements");
	  }
	}
#if PARCEL_MPI
	parcel::comm &parcel_comm = parcel_blas.parcel_comm();
#endif

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
#if PARCEL_MPI
	// 局所の内積を ALL REDUCE
	parcel::vector<Trmatrix> R_(R.size(),que);parcel_blas.zero_init<Trmatrix>(R_);
	parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Tbasis,Trmatrix,TcalcQR>(
								4,  // m
								s_half, // n
								n,  // k
								one,zero,
								VcholQR ,ldv, 
								Vbasis,ldq,
								R_,4,
								"T","N","N"
								);
	que[0].wait();
	parcel_comm.allreduce<Trmatrix,Trmatrix>(R_,R,4*s_half,MPI_SUM);
#else
	parcel_blas.gemm<TcalcQR,TcalcQR,Tvec,Tbasis,Trmatrix,TcalcQR>(
								4,  // m
								s_half, // n
								n,  // k
								one,zero,
								VcholQR ,ldv, 
								Vbasis,ldq,
								R,4,
								"T","N","N"
								);
#endif
	que[0].wait();

	// コレスキー分解
	std::vector< parcel::scalar > Rhost(R.size());
	std::vector< parcel::scalar > Rtmp(Rhost.size());
	std::vector< parcel::scalar > Lmat(Rhost.size());
	R.getValue(Rhost);
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
	}       
	parcel_blas_host.potrf(Lmat,s,s);
	
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
