// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include "parcel_nblock.hpp"
#ifndef PARCEL_PRECON_POINTJACOBI_
#define PARCEL_PRECON_POINTJACOBI_

namespace parcel{
  template <
    class Tmat,
    class Tvec,
    class Tprecon_mat,
    class Tprecon_vec,
    class Tprecon_calc_mat,
    class Tcalc
    >
  class precon_pointjacobi{
    vector<Tprecon_mat> DiagInv_matrix_;
  public:
    precon_pointjacobi(){}
    ~precon_pointjacobi(){}


    // dd行列ならddで計算
    void get_diag(parcel::crsMat<Tmat> &crsMat){

      sycl::queue *q_ = crsMat.queue();
      size_t N = crsMat.n();
      DiagInv_matrix_.resize(N,q_);

#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif
      vector<Tmat> &mat  = crsMat.mat();
      PARCEL_LINT *Rowptr = crsMat.rowptr_data();
      PARCEL_LINT *crsCol = crsMat.crsCol_data();

      const auto Tmat_name = typeid(Tmat).name();
      const auto Tprecon_mat_name = typeid(Tprecon_mat).name();

      if( ddtype_name == Tprecon_mat_name ){ 
	auto *DiagInv_hi = DiagInv_matrix_.data_hi();
	auto *DiagInv_lo = DiagInv_matrix_.data_lo();
	if( ddtype_name == Tmat_name ){ 
	  auto *A_hi = mat.data_hi();
	  auto *A_lo = mat.data_lo();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
		for(size_t i=iii;i<imax;i++){	    
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];
		    if(i==col){
		      ddtype ddone(1.0,0.0);
		      ddtype D(A_hi[j],A_lo[j]);
		      ddtype Dinv(0.0,0.0);
		      if( ((double)A_hi[j]==0.0) && ((double)A_lo[j]==0.0) ){
			Dinv =  ddone;
		      }else{
			Dinv =  ddone / D;
		      }
		      DiagInv_hi[i]  = Dinv.get_hi();
		      DiagInv_lo[i]  = Dinv.get_lo();
		      break;
		    }

		  }
		}
	      }
	    });
	}else{
	  auto *A = mat.data();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
		for(size_t i=iii;i<imax;i++){	    
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];

		    if(i==col){
		      ddtype ddone(1.0,0.0);
		      ddtype D(A[j],0.0);
		      ddtype Dinv(0.0,0.0);
		      if((double)A[j]==0.0){
			Dinv =  ddone;
		      }else{
			Dinv =  ddone / D;
		      }
		      DiagInv_hi[i]  = Dinv.get_hi();
		      DiagInv_lo[i]  = Dinv.get_lo();
		      break;
		    }

		  }
		}
	      }
	    });
	}
      }else{
	auto *DiagInv = DiagInv_matrix_.data();
	auto *A = mat.data();
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){	    
		for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_LINT col = crsCol[j];

		  if(i==col){
		    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type A_ = A[j];
		    DiagInv[i] = A[j] == (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type) 0.0 ?
		      (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type) 1.0 :
		      (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type) (1.0/A_);
		    break;
		  }

		}
	      }
	    }
	  });
      }

      q_[0].wait();

      return ;

    }

    void get_diag(parcel::diaMat<Tmat> &diaMat){

      sycl::queue *q_ = diaMat.queue();
      const size_t N = diaMat.n();
      const size_t NND = diaMat.nnd();
      DiagInv_matrix_.resize(N,q_);

#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif
      vector<Tmat> &mat  = diaMat.mat();
      PARCEL_INT *offset = diaMat.offset_data();

      const auto Tmat_name = typeid(Tmat).name();
      const auto Tprecon_mat_name = typeid(Tprecon_mat).name();

      if( ddtype_name == Tprecon_mat_name ){ 
	auto *DiagInv_hi = DiagInv_matrix_.data_hi();
	auto *DiagInv_lo = DiagInv_matrix_.data_lo();
	if( ddtype_name == Tmat_name ){ 
	  auto *A_hi = mat.data_hi();
	  auto *A_lo = mat.data_lo();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);		for(size_t i=iii;i<imax;i++){	    

		  for(PARCEL_INT j = 0; j < NND; j++){
		    const PARCEL_INT offset_ = offset[j];
		    if(offset_==0){
		      ddtype ddone(1.0,0.0);
		      ddtype D(A_hi[j],A_lo[j]);
		      ddtype Dinv(0.0,0.0);
		      if( ((double)A_hi[j]==0.0) && ((double)A_lo[j]==0.0) ){
			Dinv =  ddone;
		      }else{
			Dinv =  ddone / D;
		      }
		      DiagInv_hi[i]  = Dinv.get_hi();
		      DiagInv_lo[i]  = Dinv.get_lo();
		      break;
		    }

		  }
		}
	      }
	    });
	}else{
	  auto *A = mat.data();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
		for(size_t i=iii;i<imax;i++){	    

		  for(PARCEL_INT j = 0; j < NND; j++){
		    const PARCEL_INT offset_ = offset[j];
		    if(offset_==0){
		      ddtype ddone(1.0,0.0);
		      ddtype D(A[j],0.0);
		      ddtype Dinv(0.0,0.0);
		      if((double)A[j]==0.0){
			Dinv =  ddone;
		      }else{
			Dinv =  ddone / D;
		      }
		      DiagInv_hi[i]  = Dinv.get_hi();
		      DiagInv_lo[i]  = Dinv.get_lo();
		      break;
		    }

		  }
		}
	      }
	    });
	}
      }else{
	auto *DiagInv = DiagInv_matrix_.data();
	auto *A = mat.data();
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){	    

		for(PARCEL_INT j = 0; j < NND; j++){
		  const PARCEL_INT offset_ = offset[j];
		  if(offset_==0){
		    typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type A_ = A[j];
		    DiagInv[i] = A[j]==(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type)0.0 ?
		      (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type) 1.0 :
		      (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_calc_mat) >::type) (1.0/A_);
		    break;
		  }

		}
	      }
	    }
	  });
      }

      q_[0].wait();

      return ;

    }


#include "parcel_precon_ddpointjacobi.hpp"
    template<class T_vec,class T_mat,class T_calc>
    void pointjacobi(
		     sycl::queue *q_,
		     PARCEL_INT N,
		     T_vec *x,
		     T_mat *Dinv,
		     T_vec *y
		     ){

#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 4096 * (8/sizeof(T_vec));
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif

      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){
	      T_calc x_ =  x[i];
	      T_calc y_ =  y[i];
	      T_calc Dinv_ =  Dinv[i];
	      y_   = Dinv_*x_;
	      y[i] = y_;		 
	    }
	  }
	});

      return ;
    }

  
    // y = Dinvx
    void solver(
		parcel::vector<Tvec> &x_,
		parcel::vector<Tvec> &y_
		){

      const auto Tvec_name         = typeid(Tvec).name();
      const auto Tprecon_mat_name  = typeid(Tprecon_mat).name();
      const auto Tprecon_calc_name = typeid(Tcalc).name();

      sycl::queue *q_ = x_.queue();

      const size_t nx = x_.size();
      const size_t ny = y_.size();
      // 対角配列 (DiagInv_matrix_) は行列の局所行数ぶんしか確保していないので、そこで頭打ちにする。
      // MPI 実行ではソルバがハロ込み (n+nadd) のビューを渡すため、頭打ちが無いと Dinv を範囲外参照する。
      // ヤコビ (crsMat.n()) やブロックヤコビ (Lmatrix_crsMat.n()) と同じ規約。ハロ部分は
      // 前処理の直後の SpMV で通信により上書きされるため、計算結果は変わらない。
      const size_t n  = std::min(std::min(nx,ny), DiagInv_matrix_.size());

      if( ddtype_name == Tprecon_calc_name ){	

	// ddcalc
	if( ddtype_name == Tprecon_mat_name ){	
	  double *Dinv_hi = (double *)DiagInv_matrix_.data_hi();
	  double *Dinv_lo = (double *)DiagInv_matrix_.data_lo();
	  if( ddtype_name == Tvec_name ){	
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    double *y_hi = (double *)y_.data_hi();
	    double *y_lo = (double *)y_.data_lo();
	    ddpointjacobi(q_,n,x_hi,x_lo,Dinv_hi,Dinv_lo,y_hi,y_lo);
	  }else{
	    auto x = x_.data();
	    auto y = y_.data();
	    ddpointjacobi<
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type
	      >(q_,n,x,Dinv_hi,Dinv_lo,y);
	  }
	}else{
	  auto Dinv = DiagInv_matrix_.data();
	  if( ddtype_name == Tvec_name ){	
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    double *y_hi = (double *)y_.data_hi();
	    double *y_lo = (double *)y_.data_lo();
	    ddpointjacobi<
	      typename std::result_of< parcel::parcel_type_to_baseType( Tprecon_mat ) >::type
	      >(q_,n,x_hi,x_lo,Dinv,y_hi,y_lo);
	  }else{
	    auto x = x_.data();
	    auto y = y_.data();
	    ddpointjacobi<
	      typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type,
	      typename std::result_of< parcel::parcel_type_to_baseType( Tprecon_mat ) >::type
	      >(q_,n,x,Dinv,y);
	  }
	}

      }else{
	// Tcalc
	auto x = x_.data();
	auto Dinv = DiagInv_matrix_.data();
	auto y = y_.data();
	pointjacobi<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvec ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tprecon_mat ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tcalc ) >::type
	  >(q_,n,x,Dinv,y);
      }

      return ;

    }

    void init(parcel::crsMat<Tmat> &crsMat){
      get_diag(crsMat);
    }
    void init(parcel::diaMat<Tmat> &diaMat){
      get_diag(diaMat);
    }

  };

}

#endif
