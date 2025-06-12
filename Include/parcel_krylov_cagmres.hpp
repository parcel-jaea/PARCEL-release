#ifndef PARCEL_KRYLOV_CAGMRES_HPP_
#define PARCEL_KRYLOV_CAGMRES_HPP_

#include "parcel_qr.hpp"

// MPIコミュニケータを選択できるようにblasの引数を調整する

namespace parcel{
  namespace krylov{

    //    const PARCEL_INT CGS_ORTHONORMALIZATION = 0;
    //    const PARCEL_INT MGS_ORTHONORMALIZATION = 1;

    // 計算精度を指定
    template<
      class Tspmv   = parcel_double, // SpMV 計算精度
      class Tprecon = parcel_double, // 前処理計算精度
      class Taxpy   = parcel_double, // ベクトル演算計算精度
      class Tdot    = parcel_double  // 縮約演算計算精度
      >
    class pcagmres : public krylov_param{

      ddlib::ddmath dd;
      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_scala;

      parcel::timer timer_solver_main;
      parcel::timer timer_solver_SpMV;
      parcel::timer timer_solver_axpy;
      parcel::timer timer_solver_dot;
      parcel::timer timer_solver_precon;
      parcel::timer timer_solver_gemv;

      parcel::timer timer_solver_qr;
      parcel::timer timer_solver_reOrth;
      parcel::timer timer_solver_update;
      
      //      parcel::timer timer_solver_gemv;
      //      parcel::timer timer_solver_GramSchmidt;

      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;
      PARCEL_INT cagmres_s_ = 1;
      PARCEL_INT cagmres_t_ = 1;
      PARCEL_INT qrdecomposition_algo_ = parcel::qr::QR_CGS;

    public :

      void set_qr_algo(PARCEL_INT qrdecomposition_algo){
	qrdecomposition_algo_ = qrdecomposition_algo;
	return ;
      }

      void set_restart(PARCEL_INT cagmres_s,PARCEL_INT cagmres_t){
	cagmres_s_ =  cagmres_s;
	cagmres_t_ =  cagmres_t;
	return ;
      }
      pcagmres(){}
      ~pcagmres(){}

      void update_HessenbergMatrix(
				   PARCEL_INT t,
				   PARCEL_INT cagmres_t_,
				   PARCEL_INT cagmres_s_,
				   std::vector< parcel::scalar > &Hmat, // ヘッセンベルグ行列[s*t+1,s*t]
				   std::vector< parcel::scalar > &Bmat, // 基底変換行列[s+1,s]
				   std::vector< parcel::scalar > &Rmat, // QR分解で得られた上三角行列 [s+1,s+1]
				   std::vector< parcel::scalar > &BlockRmat, // ブロック古典グラムシュミット直交化で得られた上三角行列[s*t+1,s+1]
				   parcel::scalar &hk
				   ){
	//	const PARCEL_INT t = cagmret_t_;
	const PARCEL_INT s = cagmres_s_;
	
	std::vector< parcel::scalar > one(1);
	std::vector< parcel::scalar > zero(1);
	std::vector< parcel::scalar > negative_one(1);
		
	one[0] = 1.0;
	zero[0] = 0.0;	  
	negative_one[0] = -1.0;

	if(t==0){
	std::vector< parcel::scalar > Rmat_inv(s*s);
	for(PARCEL_INT i = 0;i<Rmat_inv.size();i++){ Rmat_inv[i] = 0.0; }
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<s;i++){
	    Rmat_inv[i+j*s] = Rmat[i+j*(s+1)];
	  }
	}
#if 0
	for(PARCEL_INT j = 0;j<s+1;j++){
	  for(PARCEL_INT i = 0;i<s+1;i++){
	    std::cout<<"     R : "<<i<<","<<j<<" : "<< Rmat[i+j*(s+1)].get_T() <<std::endl;
	  }	    
	}
#endif 	  
	std::string uplo("U");
	std::string diag("N");
	// 逆行列計算
	parcel_blas_host.trtri(uplo,diag,Rmat_inv,s,s);
#if 0
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<s;i++){
	    std::cout<<"     Rinv : "<<i<<","<<j<<" : "<< Rmat_inv[i+j*s].get_T() <<std::endl; 
	  }
	}
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<s+1;i++){
	    std::cout<<"     Bmat : "<<i<<","<<j<<" : "<< Bmat[i+j*(s+1)].get_T() <<std::endl; 
	  }
	}
#endif	
	  // H[s+1,s] = R_[s+1,s+1]*B1[s+1,s]*R^-1[s,s]	  
	  std::vector< parcel::scalar > RB((s+1)*s);
	  for(PARCEL_INT i = 0;i<RB.size();i++){ RB[i] = 0.0; }
	  if(t==0){
	    for(PARCEL_INT i = 0;i<Hmat.size();i++){ Hmat[i] = 0.0; }
	  }
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	  parcel_blas_host.gemm(
				s+1, // m
				s,   // n
				s+1, // k
				one,
				zero,
				Rmat,s+1,
				Bmat,s+1,
				RB  ,s+1 
				);
#if 0
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s+1;i++){
	      std::cout<<"     RB : "<<i<<","<<j<<" : "<< RB[i+j*(s+1)].get_T() <<std::endl; 
	    }
	  }
#endif
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	  parcel_blas_host.gemm(
				s+1, // m
				s, // n
				s, // k
				one,
				zero,
				RB,s+1,
				Rmat_inv,s,
				//Hmat,s+1
				Hmat,s*cagmres_t_+1
				);
#if 0
	  for(PARCEL_INT j = 0;j<s*cagmres_t_;j++){
	    for(PARCEL_INT i = 0;i<s*cagmres_t_+1;i++){
	      std::cout<<"     Hmat update : "<<i<<","<<j<<" : "<< Hmat[i+j*(s*cagmres_t_+1)].get_T() <<std::endl; 
	    }
	  }
#endif
	  if(t==0){
	    hk = Hmat[ s + (s-1)*(s*cagmres_t_+1) ];
	    return;
	  }
	  
	}else{
	  // オーバーラップした上三角行列成分
	  std::vector< parcel::scalar > Rmat_inv(s*s);
	  for(PARCEL_INT i = 0;i<Rmat_inv.size();i++){ Rmat_inv[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    Rmat_inv[0+j*s] = BlockRmat[ s*t + j*(s*t+1) ];
	  }
	  // オーバーラップしない上三角行列成分
	  for(PARCEL_INT j = 1;j<s;j++){
	    for(PARCEL_INT i = 1;i<s;i++){
	      //  Rtri0(1+i+(j-1+1)*(s+1))=Rtri(i+(j-1)*s)
	      // i+(j+1)*(s+1)
	      Rmat_inv[i+j*s] = Rmat[(i-1)+(j-1)*s];
	    }
	  }

	  std::string uplo("U");
	  std::string diag("N");
	  // 逆行列計算
	  parcel_blas_host.trtri(uplo,diag,Rmat_inv,s,s);
#if 0
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      std::cout<<"     Rinv : "<<i<<","<<j<<" : "<< Rmat_inv[i+j*s].get_T() <<std::endl; 
	    }
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s+1;i++){
	      std::cout<<"     Bmat : "<<i<<","<<j<<" : "<< Bmat[i+j*(s+1)].get_T() <<std::endl; 
	    }
	  }
#endif	
	  
	  std::vector< parcel::scalar > Hmat_all((s*(t+1)+1)*(s*(t+1)));
	  for(PARCEL_INT i = 0;i<Hmat_all.size();i++){ Hmat_all[i] = 0.0; }
	  // 計算済みの上ヘッセンベルグ行列を書き込み
	  for(PARCEL_INT j = 0;j<s*t;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      Hmat_all[i+j*(s*(t+1)+1)] = Hmat[i + j*(s*cagmres_t_+1)];
	    }
	  }

	  // R(1:s*t,1:s) = R_'(1:s*t,1:s)
	  std::vector< parcel::scalar > BlockR((s*t)*s);	  
	  for(PARCEL_INT i = 0;i<BlockR.size();i++){ BlockR[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      BlockR[i+j*(s*t)] = BlockRmat[i+j*(s*t+1)];
	    }
	  }
	  // R_(1:s*t,1:s+1) = R_'(1:s*t,1:s+1)
	  std::vector< parcel::scalar > BlockR_ubar((s*t)*(s+1));	  
	  for(PARCEL_INT i = 0;i<BlockR_ubar.size();i++){ BlockR_ubar[i] = 0.0; }             
	  for(PARCEL_INT j = 0;j<s+1;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      BlockR_ubar[i+j*(s*t)] = BlockRmat[i+j*(s*t+1)];
	    }
	  }
#if 0
	  for(PARCEL_INT j = 0;j<s*t;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      std::cout<<"     Hmat : "<<i<<","<<j<<" : "<< Hmat[i+j*(s*cagmres_t_+1)].get_T() <<std::endl; 
	    }
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      std::cout<<"     BlockR : "<<i<<","<<j<<" : "<< BlockR[i+j*(s*t)].get_T() <<std::endl; 
	    }
	  }
	  for(PARCEL_INT j = 0;j<s+1;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      std::cout<<"     BlockR_ubar : "<<i<<","<<j<<" : "<< BlockR_ubar[i+j*(s*t)].get_T() <<std::endl; 
	    }
	  }
#endif	  
	  // 第1項
	  // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
          std::vector< parcel::scalar > HR((s*t)*s);
	  for(PARCEL_INT i = 0;i<HR.size();i++){ HR[i] = 0.0; }             
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s,   // n
	  			s*t, // k
	  			negative_one,
	  			zero,
	  			Hmat  ,s*cagmres_t_+1,
	  			BlockR,s*t,
	  			HR,s*t
	  			);
	  // 第1項 + 第2項
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s,   // n
	  			s+1, // k
	  			one,
	  			one,
	  			BlockR_ubar,s*t,
	  			Bmat,s+1,
	  			HR,s*t
	  			);
	  // ( 第1項 + 第2項 ) * R^-1
	  std::vector< parcel::scalar > Hblock((s*t)*(s));
	  for(PARCEL_INT i = 0;i<Hblock.size();i++){ Hblock[i] = 0.0; }
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s, // n
	  			s, // k
	  			one,
	  			zero,
	  			HR,s*t,
	  			Rmat_inv,s,
				Hblock,s*t
	  			);
#if 0
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      std::cout<<"     Hblock line 25 : "<<i<<","<<j<<" : "<< Hblock[i+j*(s*t)].get_T() <<std::endl; 
	    }
	  }
#endif
	  std::vector< parcel::scalar > Rmat_ubar((s+1)*(s+1));
	  for(PARCEL_INT i = 0;i<Rmat_ubar.size();i++){ Rmat_ubar[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s+1;j++){
	    Rmat_ubar[0+j*(s+1)] = BlockRmat[ s*t + j*(s*t+1) ];
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      Rmat_ubar[i+1+(j+1)*(s+1)] = Rmat[i+j*s];
	    }
	  }
	  parcel::scalar bk = Bmat[s*(s+1)-1];
	  parcel::scalar rho_tilde = Rmat_ubar[s-1+(s-1)*(s+1)]; // Rtri0(s   + (s-1)*(s+1));
	  parcel::scalar rho       = Rmat_ubar[s  +    s*(s+1)]; // Rtri0(s+1 +     s*(s+1));
	  parcel::scalar hk_ = rho/rho_tilde*bk;
	  /// std::cout<<"hk_ = "<<hk_.get_T()<<std::endl;
	  std::vector< parcel::scalar > zk(s);
	  for(PARCEL_INT i = 0;i<zk.size();i++){ zk[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    zk[i] = Rmat_ubar[i+s*(s+1)];
	  }

	  // --------------------------------------------
	  // 単基底以外では要検討が必要
	  // 単基底では1.0
	  std::vector< parcel::scalar > e1(s);
	  for(PARCEL_INT i = 0;i<e1.size();i++){ e1[i] = 0.0; }
	  e1[0] = 1.0;
	  // for(PARCEL_INT i = 0;i<s+1;i++){
	  //   e1[i] = Bmat[i];
	  // }
	  
	  std::vector< parcel::scalar > es(s);
	  for(PARCEL_INT i = 0;i<es.size();i++){ es[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    es[i] = Bmat[i+1+(s-1)*(s+1)];
	    // std::cout<<" es "<<i<<" : "<< es[i].get_T()<<std::endl; 
	  }
	  // 単基底では固定 (ニュートン基底では変更する必要有り)
	  std::vector< parcel::scalar > es_(s);	 
	  for(PARCEL_INT i = 0;i<es_.size();i++){ es_[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    es_[i] = Bmat[i+1+(s-1)*(s+1)];
	    // std::cout<<" es_ "<<i<<" : "<< es_[i].get_T()<<std::endl; 	    
	  }
	  // --------------------------------------------

	  
	  // R B R ^-1
	  // 第1項
	  // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
          std::vector< parcel::scalar > RB(s*s);
	  for(PARCEL_INT i = 0;i<RB.size();i++){ RB[i] = 0.0; }             
	  parcel_blas_host.gemm(
	  			s, // m
	  			s, // n
	  			s, // k
	  			one,
	  			zero,
	  			Rmat_ubar,s+1,
	  			Bmat,s+1,
	  			RB,s
	  			);
	  
	  // rho b z es

	  // 第2項
	  // - h R R^-1
	  std::vector< parcel::scalar > ze( s*s  );
	  for(PARCEL_INT i = 0;i<ze.size();i++){ ze[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      ze[i+j*s] = bk/rho_tilde*zk[i]*es[j];
	    }
	  }
	  // 第3項
	  std::vector< parcel::scalar > Rmat_(s*s);
	  for(PARCEL_INT i = 0;i<Rmat_.size();i++){ Rmat_[i] = 0.0; }	  
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      // k-1で生成した基底ベクトルに対する再直交結果を書き込み
	      Rmat_[i+j*s] = BlockRmat[ i + s*(t-1) + j*(s*t+1) ];
	    }
	  }
	  
	  std::vector< parcel::scalar > e1es_( s*s  );
	  for(PARCEL_INT i = 0;i<e1es_.size();i++){ e1es_[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      e1es_[i+j*s] = -hk * e1[i] * es_[j];
	    }
	  }
	  // // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
	  parcel_blas_host.gemm(
				s, // m
				s, // n
				s, // k
				one,
				one,
				e1es_,s,
				Rmat_,s,
				RB,s 
				);
	  parcel_blas_host.gemm(
				s, // m
				s, // n
				s, // k
				one,
				one,
				RB,s,
				Rmat_inv,s,
				ze,s 
				);
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      // Hmat_all[i+(j+s*t)*(s*(t+1)+1)] = BlockRmat[ i + (j+1)*(s*t+1) ];
	      Hmat_all[i+(j+s*t)*(s*(t+1)+1)] = Hblock[i+j*(s*t)];	      
	    }
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      Hmat_all[i+s*t+(j+s*t)*(s*(t+1)+1)] = ze[i+j*s];
	      //	      std::cout<<"Hmat local "<<i<<","<<j<<" : "<< ze[i+j*s].get_T()<<std::endl; 
	    }
	  }
	  Hmat_all[ (s*(t+1)+1)*(s*(t+1)) -1 - ( s + s*(s*(t+1)+1)) ] = hk;
	  Hmat_all[ (s*(t+1)+1)*(s*(t+1)) -1 ] = hk_;
	  // --------------
	  for(PARCEL_INT j = 0;j<s*(t+1);j++){
	    for(PARCEL_INT i = 0;i<s*(t+1)+1;i++){
	      Hmat[i + j*(s*cagmres_t_+1)] = Hmat_all[i+j*(s*(t+1)+1)];
	      // std::cout<<"Hmat all "<<i<<","<<j<<" : "<<  Hmat_all[i+j*(s*(t+1)+1)].get_T() <<std::endl; 
	    }
	  }
	  
	  hk = hk_;
	  return ;
	}
      }

      template<
	class Tmat,  // 係数行列データ精度
	class Tvec, // クリロフ部分空間法内部データ精度	
	class Tprecon_mat, // 前処理行列データ精度
	class Tprecon_vec, // 前処ベクトルデータ精度
	class Tbasis // 基底ベクトルvのデータ精度
	>
      void update_xvector(
#if PARCEL_MPI
			  PARCEL_INT n,
#endif			  
			  PARCEL_INT cagmres_t_,
			  PARCEL_INT cagmres_s_,
			  parcel::precon_matrix<
			  Tmat,
			  // Tvec,
			  Tbasis,
			  Tprecon_mat,
			  Tprecon_vec,
			  Tprecon
			  >  &precon_Mat,		  
			  parcel::vector<Tbasis> &V,
			  parcel::vector<Tbasis> &Q,
#if PARCEL_MPI
			  parcel::vector<Tbasis> &v0_vecadd,			  
			  parcel::vector<Tbasis> &prev_vecadd,			  
			  parcel::vector<Tvec>   &x_vecadd,
#else			  
			  parcel::vector<Tbasis> &v0,			  
			  parcel::vector<Tbasis> &prev,			  
			  parcel::vector<Tvec>   &x,
#endif			  
			  std::vector< parcel::scalar > &H_UpperTri, // [s*t+1,s*t]
			  std::vector< parcel::scalar > &R_UpperTri, // [s+1,s+1]
			  std::vector< parcel::scalar > &zeta, // [s*t+1]
			  std::string Qflag = "Y"
			  ){

	// 解ベクトルの更新(リスタート周期になるまでループを抜けない実装)
	// std::cout<<" @ 9 @ "<<std::endl;
	
#if PARCEL_MPI
	sycl::queue *que = x_vecadd.queue();
#else
	sycl::queue *que = x.queue();
#endif
	
#if PARCEL_MPI
	MPI_Comm comm = MPI_COMM_WORLD;
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,comm);
#else
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
#endif
	//	parcel::precon_solver precon_solver;
	
	PARCEL_INT t = cagmres_t_;
	PARCEL_INT s = cagmres_s_;
	std::vector< parcel::scalar > Hinv_UpperTri( (s*t)*(s*t) );
	for(PARCEL_INT i=0;i<(s*t);i++){
	  for(PARCEL_INT j=0;j<(s*t);j++){
	    Hinv_UpperTri[j+i*(s*t)] = H_UpperTri[j+i*(s*t+1)];
	  }
	}
	std::string uplo("U");
	std::string diag("N");
#if 0
	for(PARCEL_INT i=0;i<(s*t);i++){
	  for(PARCEL_INT j=0;j<(s*t);j++){
	    std::cout<<"    Hmat     : "<<j<<","<<i<<" : "<<Hinv_UpperTri[j+i*(s*t)].get_T()<<std::endl; 
	  }
	}
#endif

	// trtriがホストで計算するstd::vector型なのでクラスを整理する
	//  * y = H^-1 eとして後退代入で解けばtrtriで逆行列を求める必要はない
	// std::cout<<" @ 10 @ "<<std::endl;
	parcel_blas_host.trtri(
			uplo,
			diag,
			Hinv_UpperTri,
			(s*t),(s*t)
			);
#if 0
	for(PARCEL_INT i=0;i<(s*t);i++){
	  for(PARCEL_INT j=0;j<(s*t);j++){
	    std::cout<<"    Hmat inv : "<<j<<","<<i<<" : "<<Hinv_UpperTri[j+i*(s*t)].get_T()<<std::endl; 
	  }
	}
#endif

#if 0
	for(PARCEL_INT i=0;i<(s*t);i++){
	  std::cout<<"    zeta : "<<i<<" : "<<zeta[i].get_T()<<std::endl;;
	}
#endif	
	std::vector< parcel::scalar > y( ((s*t)+1),0.0);
	for(PARCEL_INT i=0;i<(s*t);i++){
	  parcel::scalar  y_ = 0.0;
	  for(PARCEL_INT j=0;j<(s*t);j++){
	    parcel::scalar Hiv_ij = Hinv_UpperTri[i+j*(s*t)];
	    parcel::scalar e_j    = zeta[j];
	    y_ = y_ + Hiv_ij * e_j;
	  }
	  y[i] = y_;
	  // std::cout<<"     y : "<<i<<" : "<<y[i].get_T()<<std::endl;
	}
	// 直交化行列を陽に計算しない最適化(t==1)
	// Qy = V(R^-1y) = Vz : z = R^-1y --> Rz = y

	// if( ( cagmres_t_ == 1 ) && ( Qflag == "Y" ) ) {
	if( Qflag == "N" ) {
	  // 後方代入により逆行列を掛ける
	  for(PARCEL_INT i=s-1;i>-1;i--){	    
	    parcel::scalar yrinv_ = y[i]; 
	    for(PARCEL_INT j=i+1;j<s;j++){
	      yrinv_ = yrinv_ - y[j] * R_UpperTri[i+j*(s+1)];
	    }
	    y[i] = yrinv_ / R_UpperTri[ i + i*(s+1) ];
	  }
	}

	// yをデバイス側に転送
	parcel::vector<Tvec> y_;y_.resize((s*t),que);y_.setValue(y);
	que[0].wait();

	//	R_UpperTri
	// gemv計算
	parcel::vector<Tdot> one;one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);
	parcel::vector<Tdot> zero;zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);
	// v0 = v[][]*y
	// timer_solver_gemv.start();
#if PARCEL_MPI
 	parcel_blas.gemv<Tdot,Tdot,Tbasis,Tvec,Tbasis,Taxpy>(
							     (s*t),one,zero,
							     Q,
							     y_,
							     v0_vecadd
							     );
#else	
	parcel_blas.gemv<Tdot,Tdot,Tbasis,Tvec,Tbasis,Taxpy>((s*t),one,zero,Q,y_,v0);
#endif	
	// timer_solver_gemv.end();

	que[0].wait();
	// std::cout<<" @ 12 @ "<<std::endl;
	// 前処理計算
	// Mx0 = v0
	timer_solver_precon.start();
#if PARCEL_MPI
	precon_Mat.solver(v0_vecadd,prev_vecadd);
#else
	precon_Mat.solver(v0,prev);
#endif
	timer_solver_precon.end();

	que[0].wait();
	// 解ベクトルの更新 : x = x + x0
#if PARCEL_MPI
	parcel::vector<Tvec>      x;vec_util.setptr( n, x_vecadd   , x   , 0);    	
	parcel::vector<Tbasis> prev;vec_util.setptr( n, prev_vecadd, prev, 0);	
	parcel_blas.axpy<Tdot,Tdot,Tbasis,Tvec,Taxpy> (one,one,prev,x);
#else
	parcel_blas.axpy<Tdot,Tdot,Tbasis,Tvec,Taxpy> (one,one,prev,x);
#endif
	// ------------------------------------------------------------------

	return ;	
      }

      void GivensRotation(
			  PARCEL_INT m,
			  PARCEL_INT cagmres_t_,
			  PARCEL_INT cagmres_s_,
			  std::vector< parcel::scalar > &hki,
			  std::vector< parcel::scalar > &e,
			  std::vector< parcel::scalar > &co,
			  std::vector< parcel::scalar > &si
			  ){

	// ----- 
	// m+1行のヘッセンベルグ行列に対するギブンス回転
	// Givens rotation
	//    std::cout<<"Givens rot : m = "<<m<<std::endl;
	for(PARCEL_INT k=0;k<m;k++){
	  parcel::scalar hki_km  = hki[ k   + m*(cagmres_s_*cagmres_t_+1) ];
	  parcel::scalar hki_k1m = hki[ k+1 + m*(cagmres_s_*cagmres_t_+1) ];
	  parcel::scalar si_ = si[k];
	  parcel::scalar co_ = co[k];
	  parcel::scalar tmp1_ = hki_km * co_ - hki_k1m * si_;
	  parcel::scalar tmp2_ = hki_km * si_ + hki_k1m * co_;
	  hki[ k   + m*(cagmres_s_*cagmres_t_+1) ] = tmp1_;
	  hki[ k+1 + m*(cagmres_s_*cagmres_t_+1) ] = tmp2_;
	}

	// -------
	parcel::scalar hki_m1m = hki[m+1 + m*(cagmres_s_*cagmres_t_+1)];
	parcel::scalar hki_mm  = hki[m   + m*(cagmres_s_*cagmres_t_+1)];
	parcel::scalar tmp1_   = hki_m1m / hki_mm;
	parcel::scalar tmp2_   = 1.0;tmp2_ = tmp2_ + tmp1_*tmp1_;

	tmp2_ = host_util.sqrt(tmp2_);
	//     co[m] = 1.0 / host_util.sqrt(1.0+tmp1_*tmp1_);
	parcel::scalar co_ = 1.0;co_ = co_ /tmp2_;
	co[m] =  co_;
	si[m] = -co_*tmp1_;

	parcel::scalar si_ = si[m];
	parcel::scalar e_  = e[m];
	parcel::scalar e1_ = e[m+1];

	hki[m   + m*(cagmres_s_*cagmres_t_+1)] = co_*hki_mm - si_*hki_m1m;
	hki[m+1 + m*(cagmres_s_*cagmres_t_+1)] = 0.0;

	parcel::scalar tmp3_ = co_ * e_ - si_ * e1_;
	parcel::scalar tmp4_ = si_ * e_ + co_ * e1_;

	e[m]   = tmp3_;
	e[m+1] = tmp4_;

	return ;

      }

      // 前処理,IR等の設定をclassで定義する
      // CA-GMRES用に直交化前のベクトルの保存形式を指定できるようにする  !!!!
      template<
	class Tmat,  // 係数行列データ精度
	class Tvecx, // 入力ベクトルデータ精度
	class Tvecb, // 右辺ベクトルデータ精度
	class Tvec , // クリロフ部分空間法内部データ精度
	class Tprecon_mat, // 前処理行列データ精度
	class Tprecon_vec, // 前処ベクトルデータ精度
	class Tbasis, // 基底ベクトルのデータ精度
	class TcalcQR, // QR分解計算精度
	class THessenberg // ヘッセンベルグ行列の計算精度
	>
      void solver(
		  parcel::matrix<Tmat>  &Mat,
		  parcel::vector<Tvecx> &x0,
		  parcel::vector<Tvecb> &b0
		  ){


	sycl::queue *que   = Mat.queue();
	const PARCEL_INT n = x0.size();
#if PARCEL_MPI
	parcel::comm *comm = Mat.comm();
	int myrank   = comm->rank();
	int nprocess = comm->size();
	comm->barrier();
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,comm->mpi_comm());
	PARCEL_INT nadd = Mat.get_recv_size();
#else
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
	int myrank = 0;
	int nprocess = 1;
	PARCEL_INT nadd = 0;
#endif

	//	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
	// parcel::scalar_blas parcel_blas_host;
	parcel::qr::qr<TcalcQR,Tbasis,Tbasis,THessenberg> parcel_qr;
	parcel_qr.set_qr_algo(qrdecomposition_algo_);

	timer_solver.start();
	timer_solver_init.start();

	// ------------------------------
	// Init Preconditioning
	timer_solver_init_precon.start();

	const double tol = get_tolerance();
	const PARCEL_INT itrmax = get_max_iteration();
#if 1	
	precon_param &precon = *(get_precon_param());
	parcel::precon_matrix<
	  Tmat,
	  // Tvec,
	  Tbasis,
	  Tprecon_mat,
	  Tprecon_vec,
	  Tprecon
	  > precon_Mat(Mat,precon);
#else	
	parcel::precon_solver precon_solver;
#endif
	que[0].wait();
	timer_solver_init_precon.end();

	timer_solver_init_convert.start();
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);    
	parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);    
	parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
#else	
	parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);
#endif	
 	que[0].wait();
	timer_solver_init_convert.end();

	// ! ------------------------------------------------------------
	// ! CACAGMRES のメインループ
	// ! do itr = 1,itrmax,s*t
	// !   r = b-Ax
	// !   do k = 1,t
	// !     基底ベクトルの生成
	// !     do i=1,s
	// !       solve Ms=V_i
	// !       Newton基底,単基底
	// !       V_i+1=F(As)
	// !     enddo
	// !     if(k.eq.1)then
	// !       QR分解
	// !     else
	// !       ブロック古典グラムシュミット直交化
	// !       QR分解
	// !     endif
	// !     ヘッセンベルグ行列の計算
	// !     ギブンス回転
	// !     if(||r||<rmax) ;exit
	// !   enddo
	// !   解ベクトル更新
	// !   if(||r||<rmax) ;exit
	// ! enddo
	// ! ------------------------------------------------------------
	std::string Qflag = "Y";

#if PARCEL_MPI
	parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel::parallel_vector<Tbasis> QV_parallel(1,0,1,que);
	parcel::parallel_vector<Tbasis> Q_parallel(1,0,1,que);
	parcel::parallel_vector<Tbasis> V_parallel(1,0,1,que);
	
	parcel::vector<Tbasis> QV;
	parcel::vector<Tbasis> Q;
	parcel::vector<Tbasis> V;
	if( (cagmres_t_ == 1) && ( parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR) ){
	  Qflag = "N"; // 直交行列Qを陽に計算しない実装
	  QV_parallel.resize(n,nadd,(cagmres_s_*cagmres_t_+1),que);
	  vec_util.setptr( (n+nadd)*(cagmres_s_*cagmres_t_+1), QV_parallel.vec(0), QV, 0);
	  parcel_blas.zero_init<Tbasis>(QV);
	  vec_util.setptr( (n+nadd)*(cagmres_s_*cagmres_t_+1), QV, V, 0 );
	  vec_util.setptr( (n+nadd)*(cagmres_s_+1)           , QV, Q, (n+nadd)*(cagmres_s_*(cagmres_t_-1)) );
	}else{
	  V_parallel.resize(n,nadd,(cagmres_s_+1),que);
	  Q_parallel.resize(n,nadd,(cagmres_s_*cagmres_t_+1),que);
	  vec_util.setptr( (n+nadd)*(cagmres_s_+1)           , V_parallel.vec(0), V, 0);	  
	  vec_util.setptr( (n+nadd)*(cagmres_s_*cagmres_t_+1), Q_parallel.vec(0), Q, 0);
	  parcel_blas.zero_init<Tbasis>(V);
  	  parcel_blas.zero_init<Tbasis>(Q);
	}
	parcel::parallel_vector<Tbasis> v0_parallel(n,nadd,1,que);
	parcel::parallel_vector<Tbasis> pV_parallel(n,nadd,1,que);
	parcel::vector<Tbasis> v0; vec_util.setptr( n, v0_parallel.vec(0), v0, 0) ;parcel_blas.zero_init<Tbasis>(v0);
	parcel::vector<Tbasis> pV; vec_util.setptr( n, pV_parallel.vec(0), pV, 0) ;parcel_blas.zero_init<Tbasis>(pV);

#else	
	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
	parcel::vector<Tbasis> QV;
	parcel::vector<Tbasis> Q;
	parcel::vector<Tbasis> V;	
	if( (cagmres_t_ == 1) && ( parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR) ){
	  Qflag = "N"; // 直交行列Qを陽に計算しない実装
	  QV.resize( n*(cagmres_s_*cagmres_t_+1),que);parcel_blas.zero_init<Tbasis>(QV);
	  vec_util.setptr( n*(cagmres_s_*cagmres_t_+1), QV, Q, 0);
	  vec_util.setptr( n*(cagmres_s_+1)           , QV, V,  n*(cagmres_s_*(cagmres_t_-1)) );
	}else{
	  QV.resize(1,que);
	  V.resize( n*(cagmres_s_+1),que);parcel_blas.zero_init<Tbasis>(V);
	  Q.resize( n*(cagmres_s_*cagmres_t_+1),que);parcel_blas.zero_init<Tbasis>(Q);
	}
	parcel::vector<Tbasis> v0(n,que);parcel_blas.zero_init<Tbasis>(v0);
	parcel::vector<Tbasis> pV(n,que);parcel_blas.zero_init<Tbasis>(pV);
#endif

	parcel::vector<THessenberg> R((cagmres_s_+1)*(cagmres_s_+1), que);parcel_blas.zero_init<THessenberg>(R);
	parcel::vector<THessenberg> BlockR( (cagmres_s_*cagmres_t_+1)*(cagmres_s_+1) , que);parcel_blas.zero_init<THessenberg>(BlockR);
	
	parcel::vector<Tdot> one;
	one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);

	parcel::vector<Tdot> zero;
	zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);

	parcel::vector<Tdot> negative_one;
	negative_one.resize(1,que);
	double negative_one_ = -1.0;negative_one.setValue(&negative_one_);

	// --------------------------------------------
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

	// r = b - Ax
	// timer_solver_SpMV.start();
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif	
	// timer_solver_SpMV.end();

	// || r ||
	// timer_solver_dot.start();
	parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	// timer_solver_dot.end();

	// 収束判定
	PARCEL_MAX_FLOTING_TYPE rnorm_ = 1.0;
	{
	  parcel::scalar rnorm_host = parcel_blas.get_dot();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  rnorm_ = rnorm_host.get_T();
	  double bnorm_FP64 = bnorm_;
	  double rnorm_FP64 = rnorm_;
	  // std::cout<<std::setw(8)<< 0 <<" : residual norm = ";
	  // std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	  if((rnorm_/bnorm_)<tol) return;
	}
	
	// V0 = r/||r||
	{	  
	  parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(1.0/rnorm_);
	  parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	  rnorm_inv.setValue(rnorm_inv_host); 
	  // timer_solver_axpy.start();
#if PARCEL_MPI
 	  parcel::vector<Tbasis> V0;vec_util.setptr( n , V, V0,0); 	    	  
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,V0);
#else	  
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,V);
#endif	  
	  // timer_solver_axpy.end();
	}
	timer_solver_init.end();

	timer_solver_main.start();
	for(PARCEL_INT itr = 0;itr < itrmax/(cagmres_s_ * cagmres_t_);itr++){
	//	for(PARCEL_INT itr = 0;itr < 1;itr++){	  
	  // ------------
	  std::vector< parcel::scalar > Bmat( (cagmres_s_+1)*cagmres_s_ );
	  std::vector< parcel::scalar > Rmat( (cagmres_s_+1)*(cagmres_s_+1) );
	  std::vector< parcel::scalar > BlockRmat( (cagmres_s_*cagmres_t_+1)*(cagmres_s_+1) );
	  std::vector< parcel::scalar > Hmat( (cagmres_s_*cagmres_t_+1)*cagmres_s_*cagmres_t_ );	  
	  // ! 基底変換行列を作成
	  // 単基底のみ対応 :
	  for(PARCEL_INT i = 0;i<Bmat.size() ;i++){Bmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<cagmres_s_;i++){
	    Bmat[ (i+1) + i*(cagmres_s_+1) ] = 1.0;
	  }

	  std::vector< parcel::scalar > zeta(cagmres_s_ * cagmres_t_ + 1);
	  std::vector< parcel::scalar > co(cagmres_s_ * cagmres_t_ + 1);
	  std::vector< parcel::scalar > si(cagmres_s_ * cagmres_t_ + 1);
	  parcel::scalar hk;hk = 0.0;
	  // for(PARCEL_INT i = 0;i<@.size() ;i++){@[i] = 0.0;}	  
	  for(PARCEL_INT i = 0;i<co.size();i++   ){co[i]   = 0.0;}
	  for(PARCEL_INT i = 0;i<si.size();i++   ){si[i]   = 0.0;}
	  for(PARCEL_INT i = 0;i<zeta.size() ;i++){zeta[i] = 0.0;}

	  for(PARCEL_INT i = 0;i<Hmat.size() ;i++){Hmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<Rmat.size() ;i++){Rmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = 0.0;}

	  // r = b - Ax	  
	  timer_solver_SpMV.start();
#if PARCEL_MPI
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	  timer_solver_SpMV.end();
	  
	  // || r ||
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	  timer_solver_dot.end();
	  
	  // 収束判定
	  {
	    parcel::scalar rnorm_host = parcel_blas.get_dot();
	    rnorm_host = host_util.sqrt(rnorm_host);
	    rnorm_ = rnorm_host.get_T();
	    double bnorm_FP64 = bnorm_;
	    double rnorm_FP64 = rnorm_;
	    if( myrank == 0){
	      std::cout<<std::setw(8)<< itr*cagmres_s_ * cagmres_t_ <<" : residual norm = ";	      
	      std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	    }
	    if((rnorm_/bnorm_)<tol)break;
	  }
	  zeta[0] =  rnorm_;
	  // V0 = r / || r ||
	  {
	    parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(1.0/rnorm_);
	    parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	    rnorm_inv.setValue(rnorm_inv_host); 
	    timer_solver_axpy.start();
	    parcel::vector<Tbasis> Vi0;
	    vec_util.setptr(n, V, Vi0, 0);
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,Vi0);
	    timer_solver_axpy.end();
	  }
	  for(PARCEL_INT k = 0;k < cagmres_t_;k++){
	    
	    // 基底変換行列の作成(未対応)
	    for(PARCEL_INT i = 0;i < cagmres_s_;i++){
	      // 前処理
	      parcel::vector<Tbasis> Vi0;
#if PARCEL_MPI
	      vec_util.setptr(n, V, Vi0, (n+nadd)*i);
#else
	      vec_util.setptr(n, V, Vi0, n*i);
#endif	      
	      if( i==0 && k>0){
		timer_solver_axpy.start();
		parcel::vector<Tbasis> Qi0;
#if PARCEL_MPI
		vec_util.setptr( n+nadd, Q, Qi0, (n+nadd)*(cagmres_s_*k) );
		parcel_blas.axpy<Tdot,Tdot,Tbasis,Tbasis,Tvec> (one,zero,Qi0,Vi0);
#else		
		vec_util.setptr( n, Q, Qi0, n*(cagmres_s_*k) );
		parcel_blas.axpy<Tdot,Tdot,Tbasis,Tbasis,Tvec> (one,zero,Qi0,Vi0);
#endif		
		timer_solver_axpy.end();
	      }
	      
	      timer_solver_precon.start();
	      
#if PARCEL_MPI
	      parcel::vector<Tbasis> Vi0_vecadd;vec_util.setptr( (n+nadd), V, Vi0_vecadd, (n+nadd)*i); 
	      parcel::vector<Tbasis> pV_vecadd ;vec_util.setptr( (n+nadd), pV, pV_vecadd, 0); 
	      precon_Mat.solver(Vi0_vecadd,pV_vecadd);
#else	      
	      precon_Mat.solver(Vi0,pV);
#endif	      
	      timer_solver_precon.end();
	      
	      // V_i+1 = ApV : 単基底のみ対応
	      timer_solver_SpMV.start();
#if PARCEL_MPI
	      //	      parcel::vector<Tbasis> Vi1_vecadd;vec_util.setptr( (n+nadd), V, Vi1_vecadd, (n+nadd)*(i+1) );
	      parcel::vector<Tbasis> Vi1;vec_util.setptr( n, V, Vi1, (n+nadd)*(i+1) ); 
	      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Tbasis,Tspmv>(one,zero,Mat,pV_vecadd,Vi1);
#else	      
	      parcel::vector<Tbasis> Vi1;
	      vec_util.setptr(n, V, Vi1, n*(i+1));
	      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Tbasis,Tspmv>(one,zero,Mat,pV,Vi1);
#endif	      
	      timer_solver_SpMV.end();	      	      
	    }
	    
	    timer_solver_reOrth.start();
	    if( k > 0){
	      // ブロック古典グラムシュミット直交化
	      // R_' = Q_k-1 * Vk
	      // GEMM
	      
	      for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = 0.0;}
	      // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if PARCEL_MPI
	      parcel::vector<THessenberg> BlockR_( (cagmres_s_*k+1) * (cagmres_s_+1) , que);parcel_blas.zero_init<THessenberg>(BlockR_);
	      parcel_blas.gemm<Tdot,Tdot,Tbasis,Tbasis,THessenberg,THessenberg>(
										cagmres_s_*k+1, // m
										cagmres_s_+1, // n
										n, // k
										one,zero,
										Q,(n+nadd), 
										V,(n+nadd),
										BlockR_,cagmres_s_*k+1,
										"T","N","N");
	      comm->allreduce<THessenberg,THessenberg>(BlockR_, BlockR, (cagmres_s_*k+1) * (cagmres_s_+1),MPI_SUM);
#else
	      parcel_blas.gemm<Tdot,Tdot,Tbasis,Tbasis,THessenberg,THessenberg>(
										cagmres_s_*k+1, // m
										cagmres_s_+1, // n
										n, // k
										one,zero,
										Q,n, 
										V,n,
										BlockR,cagmres_s_*k+1,
										"T","N","N");	      
#endif
	      
	      std::vector< parcel::scalar > BlockRmat_host(BlockRmat.size(),0.0);
	      BlockR.getValue( BlockRmat_host );
	      for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = BlockRmat_host[i];}
	      {
		parcel::vector<Tbasis> Qblock;
		parcel::vector<Tbasis> Vblock;
		parcel::vector<THessenberg> Rblock;
#if PARCEL_MPI
		vec_util.setptr( (n+nadd)*(cagmres_s_*k+1), Q, Qblock, 0 );
		vec_util.setptr( (n+nadd)*(cagmres_s_)    , V, Vblock, n+nadd );
		vec_util.setptr( (cagmres_s_*k+1)*(cagmres_s_), BlockR, Rblock, cagmres_s_*k+1 );
		parcel_blas.gemm<Tdot,Tdot,Tbasis,THessenberg,Tbasis,THessenberg>(
										  n, // m
										  cagmres_s_, // n
										  cagmres_s_*k+1, // k
										  negative_one,one,
										  Qblock,n+nadd, 
										  Rblock,cagmres_s_*k+1,
										  Vblock,n+nadd);
#else		
		vec_util.setptr( n*(cagmres_s_*k+1), Q, Qblock, 0 );
		vec_util.setptr( n*(cagmres_s_)  , V, Vblock, n );
		vec_util.setptr( (cagmres_s_*k+1)*(cagmres_s_), BlockR, Rblock, cagmres_s_*k+1 );
		parcel_blas.gemm<Tdot,Tdot,Tbasis,THessenberg,Tbasis,THessenberg>(
										  n, // m
										  cagmres_s_, // n
										  cagmres_s_*k+1, // k
										  negative_one,one,
										  Qblock,n, 
										  Rblock,cagmres_s_*k+1,
										  Vblock,n);
#endif		
	      }
	    }
	    timer_solver_reOrth.end();
	    
	    {
	      parcel::vector<Tbasis> Vbasis;
	      parcel::vector<Tbasis> Qbasis;
	      timer_solver_qr.start();
#if PARCEL_MPI
	      if(k==0){
		vec_util.setptr( (n+nadd)*(cagmres_s_+1), V, Vbasis, 0 );
		vec_util.setptr( (n+nadd)*(cagmres_s_+1), Q, Qbasis, 0 );
	      }else{
		vec_util.setptr( (n+nadd)*(cagmres_s_)  , V, Vbasis, (n+nadd) );
		vec_util.setptr( (n+nadd)*(cagmres_s_)  , Q, Qbasis, (n+nadd)*( cagmres_s_*k + 1 ) );
	      }
	      // QR分解 : s+1本のベクトルを直交化 
	      parcel_qr.solver(
			       parcel_blas, n,
			       Vbasis ,(n+nadd),
			       Qbasis ,(n+nadd),
			       R,(cagmres_s_+1),
			       Qflag);
#else
	      if(k==0){
		vec_util.setptr( n*(cagmres_s_+1), V, Vbasis, 0 );
		vec_util.setptr( n*(cagmres_s_+1), Q, Qbasis, 0 );
	      }else{
		vec_util.setptr( n*(cagmres_s_)  , V, Vbasis, n );
		vec_util.setptr( n*(cagmres_s_)  , Q, Qbasis, n*( cagmres_s_*k + 1 ) );
	      }
	      // QR分解 : s+1本のベクトルを直交化
	      parcel_qr.solver( parcel_blas, n,
				Vbasis ,n,
				Qbasis,n,
				R,(cagmres_s_+1),
				Qflag);
#endif
	      
	      timer_solver_qr.end();
	    }
	    que[0].wait();

#if 0
	    // 直交化確認
	    {	     
	      for(PARCEL_INT j=0;j<cagmres_s_ * (k+1) + 1;j++){
		for(PARCEL_INT i = 0;i<=j ;i++){
		  parcel::vector<Tbasis> Qi;
		  parcel::vector<Tbasis> Qj;
		  vec_util.setptr(n, Q, Qi, n*i);
		  vec_util.setptr(n, Q, Qj, n*j);
		  parcel_blas.dot< Tbasis,Tbasis,Tdot >(Qi,Qj);
		  parcel::scalar Qij_host = parcel_blas.get_dot();
		  double Qij_host_ = Qij_host.get_T();
		  std::cout<<"     Qij : "<<i<<","<<j<<" : "<<Qij_host_<<std::endl;
		}
	      }
	    }
#endif	    
	    
	    std::vector< parcel::scalar > Rmat_host( (cagmres_s_+1)*(cagmres_s_+1),0.0 );
	    // R.getValue(Rmat_host.data());
	    R.getValue(Rmat_host);
	    que[0].wait();
	    for(PARCEL_INT i = 0;i<Rmat.size() ;i++){
	      Rmat[i] = Rmat_host[i];
	    }

	    // parcel_blas_host.trtri("U","N",Rmat_inv,cagmres_s_+1,cagmres_s_+1);
	    // !     ヘッセンベルグ行列の計算 -- t=1でも怪しい?
	    que[0].wait();
	    update_HessenbergMatrix(
				   k,
				   cagmres_t_,
				   cagmres_s_,
				   Hmat, // [s*t+1,s*t]
				   Bmat, // [s+1,s]
				   Rmat, // [s+1,s+1]
				   BlockRmat, // [s*t+1,s+1]
				   hk);
	    que[0].wait();
	    // Hessenberg Matrix確認
	    {
#if 0	      
	      for(PARCEL_INT j=0;j<cagmres_s_ * cagmres_t_;j++){
		for(PARCEL_INT i=0;i<cagmres_s_ * cagmres_t_ + 1;i++){
		  std::cout<<"     Hmat : "<<i<<","<<j<<" : "<< Hmat[i+j*(cagmres_s_ * cagmres_t_ + 1)].get_T() <<std::endl;
		}
	      }
#endif
	      
#if 0
	      std::vector<double> Q_check( n*(cagmres_s_*cagmres_t_+1),0.0 );
	      Q.getValue(Q_check.data());
	      //	      Q.getValue(Q_check);
	      for(PARCEL_INT i=0;i<3 ;i++){
		std::cout<<"     check Q ";
		for(PARCEL_INT j=0;j<cagmres_s_+1;j++){
		  std::cout<<Q_check[j*n+i]<<",";
		}
		std::cout<<std::endl;
	      }
#endif
	    }
	    //	    arnoldi_s(s,B1,R,BL_H_U);	    
	    // !     ギブンス回転
	    // t = 1のみ対応
	    //	    if(cagmres_t_==1){
#if 1
	  }
	  for(PARCEL_INT k = 0;k < cagmres_t_;k++){
	    for(PARCEL_INT i=k*cagmres_s_;i<(k+1)*cagmres_s_;i++){
	      GivensRotation(i,cagmres_t_,cagmres_s_,Hmat,zeta,co,si);
	    }
#if 0	    
	    // Hessenberg Matrix確認
	    {	     
	      for(PARCEL_INT j=0;j<cagmres_s_ * cagmres_t_;j++){
		for(PARCEL_INT i=0;i<cagmres_s_ * cagmres_t_ + 1;i++){
		  std::cout<<k<<" @ Hmat : "<<i<<","<<j<<" : "<< Hmat[i+j*(cagmres_s_ * cagmres_t_ + 1)].get_T() <<std::endl;
		}
	      }
	    }
#endif	    
	  } 
#else	    
	    // ギブンス回転については最後に実施しても同じ? 
	    for(PARCEL_INT i=k*cagmres_s_;i<(k+1)*cagmres_s_;i++){
	      GivensRotation(i,cagmres_t_,cagmres_s_,Hmat,zeta,co,si);
	    }
	    //	    }
	    // Hessenberg Matrix確認
	    {	     
	      for(PARCEL_INT j=0;j<cagmres_s_ * cagmres_t_;j++){
		for(PARCEL_INT i=0;i<cagmres_s_ * cagmres_t_ + 1;i++){
		  // std::cout<<"     Hmat : "<<i<<","<<j<<" : "<< Hmat[i+j*(cagmres_s_ * cagmres_t_ + 1)].get_T() <<std::endl;
		}
	      }
	    }
	  }
#endif	    

	  // 解ベクトル更新
	timer_solver_update.start();
#if PARCEL_MPI
	{
	  parcel::vector<Tbasis> v0_vecadd ;vec_util.setptr( (n+nadd), v0, v0_vecadd, 0); 
	  parcel::vector<Tbasis> pV_vecadd ;vec_util.setptr( (n+nadd), pV, pV_vecadd, 0); 
	  parcel::vector<Tvec> x_vecadd ;vec_util.setptr( (n+nadd), x, x_vecadd, 0); 
	  update_xvector<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tbasis>(
								   n,
								   cagmres_t_,
								   cagmres_s_,
								   precon_Mat,
								   V,
								   Q,
								   v0_vecadd,
								   pV_vecadd,
								   x_vecadd,
								   Hmat,Rmat,zeta,Qflag);
      }	
#else	
	  update_xvector<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tbasis>(
			   cagmres_t_,
			   cagmres_s_,
			   precon_Mat,
			   V,Q,v0,pV,x,Hmat,Rmat,zeta,Qflag);
#endif	  
	timer_solver_update.end();

	}
	que[0].wait();
	vec_util.convert<Tvec,Tvecx>(x,x0);
	que[0].wait();

      timer_solver_main.end();
      timer_solver.end();

      if( myrank == 0){
	std::cout<<std::endl<<" ----------------- "<<std::endl;
	std::cout<<"pcagmres time              = "<<timer_solver.get()      <<" [sec] "<<std::endl;
	std::cout<<"     init time             = "<<timer_solver_init.get() <<" [sec] "<<std::endl;
	std::cout<<"          init precon      = "<<timer_solver_init_precon.get()  <<" [sec] "<<std::endl;
	std::cout<<"          init convert     = "<<timer_solver_init_convert.get() <<" [sec] "<<std::endl;
	std::cout<<"     main time             = "<<timer_solver_main.get()   <<" [sec] "<<std::endl;
	// std::cout<<"          main dot time    = "<<timer_solver_dot.get()    <<" [sec] "<<std::endl;
	// std::cout<<"          main axpy time   = "<<timer_solver_axpy.get()   <<" [sec] "<<std::endl;
	std::cout<<"          main SpMV time   = "<<timer_solver_SpMV.get()   <<" [sec] "<<std::endl;
	std::cout<<"          main precon time = "<<timer_solver_precon.get() <<" [sec] "<<std::endl;      
	std::cout<<"          main QR time     = "<<timer_solver_qr.get()   <<" [sec] "<<std::endl;
	std::cout<<"          main reOrth time = "<<timer_solver_reOrth.get()   <<" [sec] "<<std::endl;
	std::cout<<"          main update time = "<<timer_solver_update.get()   <<" [sec] "<<std::endl;
      }
      
      return ;
      }

    };

  }
}

#endif

