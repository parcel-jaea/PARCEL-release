// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_PRECON_JACOBI_0_
#define PARCEL_PRECON_JACOBI_0_


#include "parcel_nblock.hpp"
#include "mixed_precision.hpp"

namespace parcel{
  template <
    class Tmat,
    class Tvec,
    class Tprecon_mat,
    class Tprecon_vec,
    class Tprecon_calc_mat, // 前処理行列作成時の精度(未使用)
    class Tcalc
    >
  class precon_jacobi : public mixed_precison::mixed_precison_blas{

    matrix<Tprecon_mat> precon_LUmatrix_;
    vector<Tprecon_mat> DiagInv_;
    // 行ごとの対角の位置 (行の先頭からの番号、255 は「飛ばさない」)。CRS の前処理行列で作る。
    util_sycl::vector<unsigned char> diagoff_;
    vector<Tprecon_vec> precon_x1_;
    vector<Tprecon_vec> precon_x2_;
    vector<Tprecon_vec> precon_b_;
    PARCEL_INT JACOBI_ITER_ = 1;

    bool SCALING_MATRIX_ = false;
    bool SCALING_VECTOR_ = false; 
    vector<parcel_double> scalingInv_vec_;
    
#if PARCEL_MPI
    crsMat<Tmat> *crsMat_;
#endif

    // PARCEL_JACOBI_ONTHEFLY で (D^-1)(L+U) の値を保存しないとき true。
    //   値は元の行列 otf_src_ の値と D^-1 から前処理のたびに計算する (構築時と同じ式なので同じ値になる)。
    //   前処理のカーネルは otf_ を見て OTF=true / false の 2 版から 1 回だけ選ぶ (jacobi_method_crs_main)。
    bool otf_ = false;
    crsMat<Tmat> *otf_src_ = nullptr;

    // 反復用作業ベクトル precon_x1_ / precon_x2_ の本来の長さ。
    //   反復回数 1 のときは参照されないので 1 要素だけ確保し、2 回以上で使うときに jacobi_method で確保し直す。
    size_t work_n_ = 0;

    void jacobi_work_alloc_(size_t n_full, sycl::queue *q){
      work_n_ = n_full;
      const size_t n_x = (JACOBI_ITER_ > 1) ? n_full : 1;   // 反復回数 1 では x_now / x_next を参照しない
      precon_x1_.resize(n_x,q);
      precon_x2_.resize(n_x,q);
      precon_b_.resize(n_full,q);
    }
    void jacobi_work_ensure_(PARCEL_INT iterMax, sycl::queue *q){
      if(iterMax > 1 && precon_x1_.size() < work_n_){
        precon_x1_.resize(work_n_,q);
        precon_x2_.resize(work_n_,q);
      }
    }
    
  public:
    precon_jacobi(){}
    ~precon_jacobi(){}

    // 全ての行で対角成分が存在することが条件
    // D + LU 行列を作成してprecon_matrix_のメモリを削除する
    void get_jacobi_DinvLUmatrix(
				 parcel::crsMat<Tmat> &crsMat
				 ){
      sycl::queue *q_ = crsMat.queue();
      size_t N   = crsMat.n();
      size_t NNZ = crsMat.nnz();

      // 値を保存せず前処理のたびに計算するか (条件は parcel_nblock.hpp の PARCEL_JACOBI_ONTHEFLY)
      otf_ = false;
      otf_src_ = nullptr;
#if PARCEL_JACOBI_ONTHEFLY
      if( ddtype_name != typeid(Tmat).name() && ddtype_name != typeid(Tprecon_mat).name() &&
          ddtype_name != typeid(Tcalc).name() && SCALING_MATRIX_ == false ){
        otf_ = true;
        otf_src_ = &crsMat;
      }
#endif
      const bool otf = otf_;

      // 対角要素
      DiagInv_.resize(N,q_);

      // 非対角要素
      precon_LUmatrix_.set_queue(q_);
      // (L+U) を A と同じ非零の並びで持ち、行ポインタ・列番号は A のものを共有する (parcel_nblock.hpp の説明)
      precon_LUmatrix_.resize(N,1,CRS_MATRIX);   // 行列の型だけを決める (値・添字は下で差し替える)
      precon_LUmatrix_.crsMat_data()->resize_share_structure(N, NNZ, q_, crsMat, !otf_);   // otf_ のとき値は 1 要素だけ
      {
        // 行ごとの対角の位置を控える。反復は対角 (値 0) を読まず、対角を除いた要素を A の行の順に足す。
        //   位置が 255 以上、または対角が無い行は 255 (その行は対角の 0 も含めて全部読む)。
        diagoff_.resize(N,q_);
        unsigned char *Doff = diagoff_.data();
        const PARCEL_LINT *Rp = crsMat.rowptr_data();
        const PARCEL_LINT *Cc = crsMat.crsCol_data();
        q_[0].parallel_for(sycl::range<1>{N},[=](sycl::id<1> i){
            unsigned char d = 255;
            for(PARCEL_LINT j = Rp[i]; j < Rp[i+1]; j++){
              if( (size_t)Cc[j] == (size_t)i ){
                const PARCEL_LINT o = j - Rp[i];
                d = (o < 255) ? (unsigned char)o : (unsigned char)255;
                break;
              }
            }
            Doff[i] = d;
          });
        q_[0].wait();
      }

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

      parcel::crsMat<Tprecon_mat> &precon_LU_crsMat = *precon_LUmatrix_.crsMat_data();
      vector<Tprecon_mat> &LU_mat = precon_LU_crsMat.mat();
      PARCEL_LINT *LU_Rowptr = precon_LU_crsMat.rowptr_data();
      PARCEL_LINT *LU_crsCol = precon_LU_crsMat.crsCol_data();

      const auto Tmat_name = typeid(Tmat).name();
      const auto Tprecon_mat_name = typeid(Tprecon_mat).name();

      // ----------
      // LU_Rowptr は A の行ポインタそのもの (共有)。書き換えない
      q_[0].wait();

      // ---------------------------------------------------
      // 行列に対するスケーリング
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      if( SCALING_MATRIX == true ){ 
	scalingInv_vec_.resize(N,q_);
	auto *A       = mat.data();
	auto *AmaxInv = scalingInv_vec_.data();
	crs_maxInv<
	  typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type,
	  double
	  >(N,A,AmaxInv,Rowptr,crsCol);
	q_[0].wait();
      }
      // ---------------------------------------------------

      if( ddtype_name == Tprecon_mat_name ){ 
	// dd
	if( ddtype_name == Tmat_name ){ 
	  // T
	  auto *A_hi    = mat.data_hi();
	  auto *A_lo    = mat.data_lo();
	  auto *LU_A_hi = LU_mat.data_hi();
	  auto *LU_A_lo = LU_mat.data_lo();
	  auto *DiagInv_hi = DiagInv_.data_hi();
	  auto *DiagInv_lo = DiagInv_.data_lo();
	  auto *AmaxInv    = scalingInv_vec_.data();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
		for(size_t i=iii;i<imax;i++){	    
		  
		  DiagInv_hi[i] = 1.0;
		  DiagInv_lo[i] = 0.0;
		  double Dinv_hi = 0.0;
		  double Dinv_lo = 0.0;
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];
		    if(i==col){
		      double A_hi_ = A_hi[j];
		      double A_lo_ = A_lo[j];
		      if( SCALING_MATRIX == true){ 
			double val_hi = AmaxInv[i];
			double val_lo = 0.0;
			double ddmul_var1;
			ddmul_calc1(A_hi_,A_lo_,val_hi,val_lo,A_hi_,A_lo_);
		      }

		      if( A_hi_ != 0.0 ){
			double one_hi_ = 1.0;
			double one_lo_ = 0.0;
			double dddiv_var1;
			dddiv_calc1(one_hi_,one_lo_,A_hi_,A_lo_,Dinv_hi,Dinv_lo);
			DiagInv_hi[i] = Dinv_hi;
			DiagInv_lo[i] = Dinv_lo;
		      }
		      break;
		    }
		  }
		  
		  PARCEL_LINT j0 = LU_Rowptr[i];
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];
		      // A と同じ位置 j に書く。対角は 0 (前処理の反復で 0*x_i を引くだけになる)
		      if(i==col){ LU_A_hi[j] = 0.0; LU_A_lo[j] = 0.0; continue; }
		      j0 = j;
		    if(i!=col){
		      double A_hi_ = A_hi[j];
		      double A_lo_ = A_lo[j];
		      if( SCALING_MATRIX == true ){ 
			double val_hi = AmaxInv[i];
			double val_lo = 0.0;
			double ddmul_var1;
			ddmul_calc1(A_hi_,A_lo_,val_hi,val_lo,A_hi_,A_lo_)
		      }

		      double LU_A_hi_ = 0.0;
		      double LU_A_lo_ = 0.0;
		      double ddmul_var1;
		      ddmul_calc1(A_hi_,A_lo_,Dinv_hi,Dinv_lo,LU_A_hi_,LU_A_lo_);
		      LU_A_hi[j0] = LU_A_hi_;
		      LU_A_lo[j0] = LU_A_lo_;

		      j0 = j0 + 1;
		    }
		  }
		}
	      }
	    });

	}else{
	  // T
	  auto *A       = mat.data();
	  auto *LU_A_hi = LU_mat.data_hi();
	  auto *LU_A_lo = LU_mat.data_lo();
	  auto *DiagInv_hi = DiagInv_.data_hi();
	  auto *DiagInv_lo = DiagInv_.data_lo();
	  auto *AmaxInv = scalingInv_vec_.data();
	  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
		const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
		for(size_t i=iii;i<imax;i++){	    
		  
		  DiagInv_hi[i] = 1.0;
		  DiagInv_lo[i] = 1.0;
		  double Dinv_hi = 0.0;
		  double Dinv_lo = 0.0;
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];
		    if(i==col){
		      double A_hi_ = A[j];
		      double A_lo_ = 0.0;
		      if( SCALING_MATRIX == true ){ 
			double val_hi = AmaxInv[i];
			double val_lo = 0.0;
			double ddmul_var1;
			ddmul_calc1(A_hi_,A_lo_,val_hi,val_lo,A_hi_,A_lo_)
		      }
		      if( A_hi_ != 0.0 ){
			double one_hi_ = 1.0;
			double one_lo_ = 0.0;
			double dddiv_var1;
			dddiv_calc1(one_hi_,one_lo_,A_hi_,A_lo_,Dinv_hi,Dinv_lo);
			DiagInv_hi[i] = Dinv_hi;
			DiagInv_lo[i] = Dinv_lo;
		      }
		      break;
		    }
		  }
		  
		  PARCEL_LINT j0 = LU_Rowptr[i];
		  for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_LINT col = crsCol[j];
		      // A と同じ位置 j に書く。対角は 0 (前処理の反復で 0*x_i を引くだけになる)
		      if(i==col){ LU_A_hi[j] = 0.0; LU_A_lo[j] = 0.0; continue; }
		      j0 = j;
		    if(i!=col){
		      double A_hi_ = A[j];
		      double A_lo_ = 0.0;
		      if( SCALING_MATRIX == true ){ 
			double val_hi = AmaxInv[i];
			double val_lo = 0.0;
			double ddmul_var1;
			ddmul_calc1(A_hi_,A_lo_,val_hi,val_lo,A_hi_,A_lo_)
		      }
		      double LU_A_hi_ = 0.0;
		      double LU_A_lo_ = 0.0;
		      double ddmul_var1;
		      ddmul_calc1(A_hi_,A_lo_,Dinv_hi,Dinv_lo,LU_A_hi_,LU_A_lo_)
			LU_A_hi[j0] = LU_A_hi_;
		      LU_A_lo[j0] = LU_A_lo_;

		      j0 = j0 + 1;
		    }
		  }
		}
	      }
	    });
	}

      }else{
	// T
	auto *A       = mat.data();
	auto *LU_A    = LU_mat.data();
	auto *DiagInv = DiagInv_.data();
	auto *AmaxInv = scalingInv_vec_.data();
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){	    

		DiagInv[i] = 1.0;
		typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type DiagInv_val = 1.0;
		for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_LINT col = crsCol[j];
		  if(i==col){
		    if( SCALING_MATRIX == true ){ 
		      DiagInv_val =
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type)(A[j]*AmaxInv[i])  ==
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 0.0 ?
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 1.0 :
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) (1.0/(A[j]*AmaxInv[i])) ;
		    }else{
		      DiagInv_val =
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type)A[j] ==
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type)  0.0 ?
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type)  1.0 :
			(typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type)  (1.0/A[j]) ;
		    }
		    DiagInv[i] = DiagInv_val;
		    break;
		  }
		}
		PARCEL_LINT j0 = LU_Rowptr[i];
		for(PARCEL_LINT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_LINT col = crsCol[j];
		    #if PARCEL_JACOBI_ONTHEFLY
		    if(otf){ break; }   // 値を保存しない (前処理のたびに同じ式で計算する)
		    #endif
		    // A と同じ位置 j に書く。対角は 0 (前処理の反復で 0*x_i を引くだけになる)
		    if(i==col){ LU_A[j] = 0.0; continue; }
		    j0 = j;
		  if(i!=col){
		    if( SCALING_MATRIX == true ){ 
		      LU_A[j0] = PARCEL_MAX_FLOTING_TYPE ( (A[j] )*AmaxInv[i]) * DiagInv_val ;
		    }else{
		      LU_A[j0] = PARCEL_MAX_FLOTING_TYPE ( A[j] ) * DiagInv_val ;
		    }
		    j0 = j0 + 1;
		  }
		}

	      }
	    }
	  });
      }

      q_[0].wait();


      return ;

    }

    // 全ての行で対角成分が存在することが条件
    // D + LU 行列を作成してprecon_matrix_のメモリを削除する
    void get_jacobi_DinvLUmatrix(
				 parcel::diaMat<Tmat> &diaMat
				 ){

    sycl::queue *q_ = diaMat.queue();
    const size_t N   = diaMat.n();
    const size_t NND = diaMat.nnd();
    const PARCEL_INT max_up_offset = diaMat.max_up_offset();
    const PARCEL_INT max_down_offset = diaMat.max_down_offset();

#if _A64FX_HIPSYCL_
    const size_t nblock      = parcel::nblock_runtime();
    const size_t nlocalblock = 8192;
#else
    const size_t nblock      = N;
    const size_t nlocalblock = 1;
#endif
    const PARCEL_INT NND_precon = NND -1;

    // 対角要素
    DiagInv_.resize(N,q_);

    // 非対角要素
    precon_LUmatrix_.set_queue(q_);
    precon_LUmatrix_.resize(N,NND_precon,DIA_MATRIX);

    vector<Tmat> &mat  = diaMat.mat();
    PARCEL_INT *offset = diaMat.offset_data();

    parcel::diaMat<Tprecon_mat> &precon_LU_diaMat = *precon_LUmatrix_.diaMat_data();
    vector<Tprecon_mat> &LU_mat = precon_LU_diaMat.mat();
    PARCEL_INT *LU_offset = precon_LU_diaMat.offset_data();

    const auto Tmat_name = typeid(Tmat).name();
    const auto Tprecon_mat_name = typeid(Tprecon_mat).name();

    // ---------------------------------------------------
    // 行列に対するスケーリング
    const bool SCALING_MATRIX = SCALING_MATRIX_;
    if( SCALING_MATRIX == true ){ 
      scalingInv_vec_.resize(N,q_);
      auto *A       = mat.data();
      auto *AmaxInv = scalingInv_vec_.data();
      dia_maxInv<
	typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type,
	double
	>(N,NND,A,AmaxInv);
    }
    q_[0].wait();

    // ---------------------------------------------------
    // offset
    const PARCEL_INT diag_id = get_diag_id(NND,offset);

    size_t NND_ = NND;
    q_[0].parallel_for(sycl::range<1>{NND_},[=](sycl::id<1> i){
	PARCEL_INT i_ = i;
	if(i_!=diag_id){
	  PARCEL_INT ii = diag_id < (PARCEL_INT)i ? i-1 : i;
	  LU_offset[ii] = offset[i];
	}
      });
    q_[0].wait();
    precon_LU_diaMat.set_max_up_offset( max_up_offset );
    precon_LU_diaMat.set_max_down_offset( max_down_offset );

    if( ddtype_name == Tprecon_mat_name ){ 
      // precon Mat : dd
      if( ddtype_name == Tmat_name ){ 
	// Mat : dd
	auto *A_hi    = mat.data_hi();
	auto *A_lo    = mat.data_lo();
	auto *LU_A_hi = LU_mat.data_hi();
	auto *LU_A_lo = LU_mat.data_lo();
	auto *DiagInv_hi = DiagInv_.data_hi();
	auto *DiagInv_lo = DiagInv_.data_lo();
	auto *AmaxInv    = scalingInv_vec_.data_hi();

	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){

	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){

		double DiagInv_val_hi_ = 1.0;
		double DiagInv_val_lo_ = 0.0;
		{
		  const PARCEL_INT j = diag_id;
		  double A_hi_ = A_hi[i+j*N];
		  double A_lo_ = A_lo[i+j*N];
		  
		  if( SCALING_MATRIX == true ){ 
		    // DiagInv_val = (A[i+j*N]*AmaxInv[i])==0.0 ? 1.0 : (1.0/(A[i+j*N]*AmaxInv[i])) ;
		    double AmaxInv_hi_ = AmaxInv[i]; 
		    double AmaxInv_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,DiagInv_val_hi_,DiagInv_val_lo_);

		    double one_hi_ = 1.0;
		    double one_lo_ = 0.0;
		    double dddiv_var1;
		    dddiv_calc1(one_hi_,one_lo_,DiagInv_val_hi_,DiagInv_val_lo_,DiagInv_val_hi_,DiagInv_val_lo_);

		  }else{
		    // DiagInv_val = A[i+j*N]==0.0 ? 1.0 : (1.0/A[i+j*N]) ;
		    double one_hi_ = 1.0;
		    double one_lo_ = 0.0;
		    double dddiv_var1;
		    dddiv_calc1(one_hi_,one_lo_,A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_);
		  }
		  DiagInv_hi[i] = DiagInv_val_hi_;
		  DiagInv_lo[i] = DiagInv_val_lo_;
		}

		for(PARCEL_INT j = 0; j <diag_id; j++){
		  double A_hi_ = A_hi[i+j*N];
		  double A_lo_ = A_lo[i+j*N];
		  if( SCALING_MATRIX == true ){ 
		    // LU_A[i+j*N] = (A[i+j*N]*AmaxInv[i]) * DiagInv_val;
		    double AmaxInv_hi_ = AmaxInv[i];
		    double AmaxInv_lo_ = 0.0;
		    double val_hi_ = 0.0;
		    double val_lo_ = 0.0;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,val_hi_,val_lo_);
		    ddmul_calc1(val_hi_,val_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+j*N] = LU_A_hi_;
		    LU_A_lo[i+j*N] = LU_A_lo_;
		  }else{
		    // LU_A[i+j*N] = A[i+j*N] * DiagInv_val;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+j*N] = LU_A_hi_;
		    LU_A_lo[i+j*N] = LU_A_lo_;
		  }
		}

		for(PARCEL_INT j = diag_id+1; j <NND; j++){
		  double A_hi_ = A_hi[i+j*N];
		  double A_lo_ = A_lo[i+j*N];
		  if( SCALING_MATRIX == true ){ 
		    // LU_A[i+(j-1)*N] = (A[i+j*N]*AmaxInv[i]) * DiagInv_val;
		    double AmaxInv_hi_ = AmaxInv[i];
		    double AmaxInv_lo_ = 0.0;
		    double val_hi_ = 0.0;
		    double val_lo_ = 0.0;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,val_hi_,val_lo_);
		    ddmul_calc1(val_hi_,val_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+(j-1)*N] = LU_A_hi_;
		    LU_A_lo[i+(j-1)*N] = LU_A_lo_;
		  }else{
		    // LU_A[i+(j-1)*N] = A[i+j*N] * DiagInv_val;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+(j-1)*N] = LU_A_hi_;
		    LU_A_lo[i+(j-1)*N] = LU_A_lo_;
		  }
		}

	      }
	    }
	  });

      }else{
	// Mat : T
	auto *A       = mat.data();
	auto *LU_A_hi = LU_mat.data_hi();
	auto *LU_A_lo = LU_mat.data_lo();
	auto *DiagInv_hi = DiagInv_.data_hi();
	auto *DiagInv_lo = DiagInv_.data_lo();
	auto *AmaxInv = scalingInv_vec_.data();
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){

	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){

		double DiagInv_val_hi_ = 1.0;
		double DiagInv_val_lo_ = 0.0;
		{
		  const PARCEL_INT j = diag_id;
		  double A_hi_ = A[i+j*N];
		  double A_lo_ = 0.0;
		  
		  if( SCALING_MATRIX == true ){ 
		    // DiagInv_val = (A[i+j*N]*AmaxInv[i])==0.0 ? 1.0 : (1.0/(A[i+j*N]*AmaxInv[i])) ;
		    double AmaxInv_hi_ = AmaxInv[i];
		    double AmaxInv_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,DiagInv_val_hi_,DiagInv_val_lo_);

		    double one_hi_ = 1.0;
		    double one_lo_ = 0.0;
		    double dddiv_var1;
		    dddiv_calc1(one_hi_,one_lo_,DiagInv_val_hi_,DiagInv_val_lo_,DiagInv_val_hi_,DiagInv_val_lo_);

		  }else{
		    // DiagInv_val = A[i+j*N]==0.0 ? 1.0 : (1.0/A[i+j*N]) ;
		    double one_hi_ = 1.0;
		    double one_lo_ = 0.0;
		    double dddiv_var1;
		    dddiv_calc1(one_hi_,one_lo_,A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_);
		  }
		  DiagInv_hi[i] = DiagInv_val_hi_;
		  DiagInv_lo[i] = DiagInv_val_lo_;
		}

		for(PARCEL_INT j = 0; j <diag_id; j++){
		  double A_hi_ = A[i+j*N];
		  double A_lo_ = 0.0;
		  if( SCALING_MATRIX == true ){ 
		    // LU_A[i+j*N] = (A[i+j*N]*AmaxInv[i]) * DiagInv_val;
		    double AmaxInv_hi_ = AmaxInv[i];
		    double AmaxInv_lo_ = 0.0;
		    double val_hi_ = 0.0;
		    double val_lo_ = 0.0;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,val_hi_,val_lo_);
		    ddmul_calc1(val_hi_,val_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+j*N] = LU_A_hi_;
		    LU_A_lo[i+j*N] = LU_A_lo_;
		  }else{
		    // LU_A[i+j*N] = A[i+j*N] * DiagInv_val;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+j*N] = LU_A_hi_;
		    LU_A_lo[i+j*N] = LU_A_lo_;
		  }
		}

		for(PARCEL_INT j = diag_id+1; j <NND; j++){
		  double A_hi_ = A[i+j*N];
		  double A_lo_ = 0.0;
		  if( SCALING_MATRIX == true ){ 
		    // LU_A[i+(j-1)*N] = (A[i+j*N]*AmaxInv[i]) * DiagInv_val;
		    double AmaxInv_hi_ = AmaxInv[i];
		    double AmaxInv_lo_ = 0.0;
		    double val_hi_ = 0.0;
		    double val_lo_ = 0.0;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,AmaxInv_hi_,AmaxInv_lo_,val_hi_,val_lo_);
		    ddmul_calc1(val_hi_,val_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+(j-1)*N] = LU_A_hi_;
		    LU_A_lo[i+(j-1)*N] = LU_A_lo_;
		  }else{
		    // LU_A[i+(j-1)*N] = A[i+j*N] * DiagInv_val;
		    double LU_A_hi_ = 0.0;
		    double LU_A_lo_ = 0.0;
		    double ddmul_var1;
		    ddmul_calc1(A_hi_,A_lo_,DiagInv_val_hi_,DiagInv_val_lo_,LU_A_hi_,LU_A_lo_);
		    LU_A_hi[i+(j-1)*N] = LU_A_hi_;
		    LU_A_lo[i+(j-1)*N] = LU_A_lo_;
		  }
		}

	      }
	    }
	  });

      }

    }else{
      // precon Mat : T
      // 前処理行列がdouble-doubleではないのでdouble-doubleの場合は上位ビットを利用
      auto *A       = mat.data();
      auto *LU_A    = LU_mat.data();
      auto *DiagInv = DiagInv_.data();
      auto *AmaxInv = scalingInv_vec_.data();
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){

	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){

	      DiagInv[i] = 1.0;
	      typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type DiagInv_val = 1.0;
	      {
		const PARCEL_INT j = diag_id;
		if( SCALING_MATRIX == true ){ 
		  DiagInv_val =
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) (A[i+j*N]*AmaxInv[i]) ==
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 0.0 ?
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 1.0 :
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) (1.0/(A[i+j*N]*AmaxInv[i])) ;
		}else{
		  DiagInv_val =
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) A[i+j*N] ==
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 0.0 ?
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) 1.0 :
		    (typename std::result_of<parcel::parcel_type_to_baseType(Tprecon_mat) >::type) (1.0/A[i+j*N]) ;
		}
		DiagInv[i] = DiagInv_val;
	      }

	      for(PARCEL_INT j = 0; j <diag_id; j++){
		if( SCALING_MATRIX == true ){ 
		  LU_A[i+j*N] = ( PARCEL_MAX_FLOTING_TYPE( A[i+j*N] ) *AmaxInv[i]) * DiagInv_val;
		}else{
		  LU_A[i+j*N] = PARCEL_MAX_FLOTING_TYPE( A[i+j*N] ) * DiagInv_val;
		}
	      }
	      for(PARCEL_INT j = diag_id+1; j <NND; j++){
		if( SCALING_MATRIX == true ){ 
		  LU_A[i+(j-1)*N] = ( PARCEL_MAX_FLOTING_TYPE( A[i+j*N] )*AmaxInv[i]) * DiagInv_val;
		}else{
		  LU_A[i+(j-1)*N] = PARCEL_MAX_FLOTING_TYPE( A[i+j*N] ) * DiagInv_val;
		}
	      }

	    }
	  }
	});

    }

    q_[0].wait();

    return ;

    }

    void set_JACOBI_ITER(PARCEL_INT ITER){
      JACOBI_ITER_ = ITER;
      return ;
    }

    void set_SCALING_MATRIX(bool scaling){
      SCALING_MATRIX_ = scaling;
      return ;
    }
    void set_SCALING_VECTOR(bool scaling){
      SCALING_VECTOR_ = scaling;
      return ;
    }
 
    PARCEL_INT get_JACOBI_ITER(){
      return JACOBI_ITER_;
    }
    PARCEL_INT iter(){
      return JACOBI_ITER_;
    }

#include "parcel_precon_ddjacobi_crs.hpp"
#include "parcel_precon_ddjacobi_dia.hpp"

    template<
      bool OTF,
      class T_vec,class T_precon_mat,class T_work_vec,class T_calc,class T_orig_ptr
      >
    void jacobi_method_crs_main(
				sycl::queue *q_ ,
				PARCEL_INT N,
				PARCEL_INT iterMax,
				T_vec *x, // out  
				T_vec *b, // in
				T_precon_mat *A, // (D^-1)(L+U)
				PARCEL_LINT *Rowptr,
				PARCEL_LINT *crsCol,
				T_precon_mat *Dinv, // (D^-1)
				double *scaling_Inv,
				T_work_vec *precon_x1_,
				T_work_vec *precon_x2_,
				T_work_vec *precon_b,
				T_orig_ptr Aorig,   // OTF のとき元の行列 A の値 (それ以外は読まない)
				const unsigned char *Doff){   // 行ごとの対角の位置 (diagoff_)
      // (D^-1)(L+U) の j 番目を T_calc の変数 v に入れる。OTF では構築時と同じ式 (対角は 0、非対角は
      //   PARCEL_MAX_FLOTING_TYPE(a_ij)*Dinv_i を前処理行列の型に丸める) で計算するので、保存した値と同じになる。
      //   行列スケーリングありでは OTF にならない。
      //   カーネルはテンプレート引数 OTF で 2 版に分け、呼び出し側 (jacobi_method) が otf_ を見て 1 回だけ選ぶ。
      //   カーネルの中に実行時の分岐を置くと、値を保存する構成でも GPU で前処理が遅くなるため。
      //   OTF=false 版は保存した値 A[j] を読むだけになり、PARCEL_JACOBI_ONTHEFLY=0 のビルドと同じ形になる。
      //   PARCEL_JACOBI_ONTHEFLY=0 では OTF=true 版を作らない。
#if PARCEL_JACOBI_ONTHEFLY
      //   OTF=true 版は積を無条件に作ってから対角だけ 0 に差し替える。列番号を比べてから A と Dinv を読む書き方は
      //   条件付きの読み出しになり、GPU のコンパイラによっては要素ごとの述語付きロードになって遅くなる。
      //   値は同じ (対角は 0、それ以外は同じ式・同じ型変換)。A の読み出しは行の中だけで範囲外はない。
#define PARCEL_JLU_LET(v, jj) \
      T_calc v; \
      if constexpr (OTF) { \
        const T_precon_mat otf_p_ = (T_precon_mat)( PARCEL_MAX_FLOTING_TYPE( Aorig[(jj)] ) * Dinv[i] ); \
        v = ( (crsCol[(jj)] == (PARCEL_LINT)(size_t)(i)) ? (T_precon_mat)0.0 : otf_p_ ); \
      } else { \
        v = A[(jj)]; \
      }
#else
      static_assert(!OTF, "jacobi_method_crs_main<true> needs PARCEL_JACOBI_ONTHEFLY");
#define PARCEL_JLU_LET(v, jj) T_calc v = A[(jj)]
#endif
      // 対角を飛ばす。行 i の k 番目 (対角を除いた並び) の物理位置は js_ + k (+1 if k >= dlim_)。
      //   Doff[i] == 255 の行は dlim_ = 行長で、どの k も飛ばさない (対角の 0 も読む)。
#define PARCEL_JROW_BEGIN \
      const PARCEL_LINT js_  = Rowptr[i]; \
      const PARCEL_LINT je_  = Rowptr[i+1]; \
      const unsigned int dk_ = Doff[i]; \
      const PARCEL_LINT dlim_ = (dk_ == 255u) ? (je_ - js_) : (PARCEL_LINT)dk_; \
      const PARCEL_LINT jn_  = (je_ - js_) - ((dk_ == 255u) ? 0 : 1);
#define PARCEL_JPOS(k) ( js_ + (k) + ( ((k) >= dlim_) ? 1 : 0 ) )
#if PARCEL_JACOBI_ROW2
      // 2 行 (i2_ と i2_+1) を交互に処理する。各行は「対角の前を先頭から、続いて対角の後ろを先頭から」で、下の 1 行版と同じ順・同じ式。
      //   共通の長さまでは 2 行を 1 本のループで交互に、余りは行ごとに続ける。保存した値 A を読む版 (OTF=false) だけで使う。
      constexpr bool row2_ = (!OTF) && std::is_same<T_precon_mat, float>::value;
#define PARCEL_JROW2_SEG(XA, XB, SA, NA, SB, NB, SRC) \
      { const PARCEL_LINT m2_ = ((NA) < (NB)) ? (NA) : (NB); \
        PARCEL_LINT t2_ = 0; \
        for(; t2_ < m2_; t2_++){ \
          const T_calc Aa2_ = A_rr_[(SA)+t2_]; const T_calc va2_ = SRC[crsCol_rr_[(SA)+t2_]]; \
          const T_calc Ab2_ = A_rr_[(SB)+t2_]; const T_calc vb2_ = SRC[crsCol_rr_[(SB)+t2_]]; \
          XA = XA - Aa2_*va2_; \
          XB = XB - Ab2_*vb2_; \
        } \
        for(PARCEL_LINT u2_ = t2_; u2_ < (NA); u2_++){ \
          const T_calc Aa2_ = A_rr_[(SA)+u2_]; const T_calc va2_ = SRC[crsCol_rr_[(SA)+u2_]]; \
          XA = XA - Aa2_*va2_; } \
        for(PARCEL_LINT u2_ = t2_; u2_ < (NB); u2_++){ \
          const T_calc Ab2_ = A_rr_[(SB)+u2_]; const T_calc vb2_ = SRC[crsCol_rr_[(SB)+u2_]]; \
          XB = XB - Ab2_*vb2_; } }
#define PARCEL_JROW2_PAIR(SRC, LAST) \
      { const size_t ia2_ = i2_, ib2_ = i2_ + 1; \
        T_calc xa2_ = precon_b_rr_[ia2_]; \
        T_calc xb2_ = precon_b_rr_[ib2_]; \
        const PARCEL_LINT jsa2_ = Rowptr_rr_[ia2_], jea2_ = Rowptr_rr_[ia2_+1]; \
        const PARCEL_LINT jsb2_ = Rowptr_rr_[ib2_], jeb2_ = Rowptr_rr_[ib2_+1]; \
        const unsigned int dka2_ = Doff_rr_[ia2_], dkb2_ = Doff_rr_[ib2_]; \
        const PARCEL_LINT jda2_ = jsa2_ + ((dka2_ == 255u) ? (jea2_ - jsa2_) : (PARCEL_LINT)dka2_); \
        const PARCEL_LINT jdb2_ = jsb2_ + ((dkb2_ == 255u) ? (jeb2_ - jsb2_) : (PARCEL_LINT)dkb2_); \
        PARCEL_JROW2_SEG(xa2_, xb2_, jsa2_, jda2_ - jsa2_, jsb2_, jdb2_ - jsb2_, SRC) \
        const PARCEL_LINT ka2_ = (jda2_ < jea2_) ? (jda2_ + 1) : jea2_; \
        const PARCEL_LINT kb2_ = (jdb2_ < jeb2_) ? (jdb2_ + 1) : jeb2_; \
        PARCEL_JROW2_SEG(xa2_, xb2_, ka2_, jea2_ - ka2_, kb2_, jeb2_ - kb2_, SRC) \
        if( LAST ){ x_rr_[ia2_] = xa2_; x_rr_[ib2_] = xb2_; } \
        else      { x_next_rr_[ia2_] = xa2_; x_next_rr_[ib2_] = xb2_; } }
#endif
#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif
  
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      const bool SCALING_VECTOR = SCALING_VECTOR_;

      q_[0].wait();

      // solve Ax = b
      // x(i+1) = (D^-1) { b - {L+U}x(i) } 

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

#if PARCEL_JACOBI_NODIV_UNIT && _BLOCKING_LOOP_
      // bnorm == 1 のときは割らない版 (DIV_ = false) を使う。T_vec が double / float のときだけ (x/1.0 == x が厳密)。
      auto jacobi_k1_ = [&](auto div_tag_){
        constexpr bool DIV_ = decltype(div_tag_)::value;
#endif
#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  // 連続した行の範囲 + restrict 付きのローカル変数 (下の #define で本体の名前を差し替える)
	  const auto * __restrict Dinv_rr_ = Dinv;
	  const auto * __restrict b_rr_ = b;
	  const auto * __restrict scaling_Inv_rr_ = scaling_Inv;
	  auto * __restrict precon_b_rr_ = precon_b;
	  const size_t jchunk_ = (N + nblock - 1) / nblock;
	  const size_t jis_ = ii[0]*jchunk_;
	  const size_t jie_ = (jis_ + jchunk_ < (size_t)N) ? (jis_ + jchunk_) : (size_t)N;
	  {
	    for(size_t i=jis_;i<jie_;i++){
#define Dinv Dinv_rr_
#define b b_rr_
#define scaling_Inv scaling_Inv_rr_
#define precon_b precon_b_rr_
#else
	      const size_t MAX = N;
	      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
	      const T_calc Dinv_ = Dinv[i];
#if PARCEL_JACOBI_NODIV_UNIT && _BLOCKING_LOOP_
	      const T_calc b_    = DIV_ ? (T_calc)(b[i] / bnorm) : (T_calc)(b[i]);
#else
	      const T_calc b_    = b[i] / bnorm;
#endif
	      if( SCALING_MATRIX == true ){ 
		const T_calc val = scaling_Inv[i];
		precon_b[i] = Dinv_*b_*val;
	      }else{
		precon_b[i] = Dinv_*b_;
	      }

#if _BLOCKING_LOOP_
	    }
	  }
	});
#else
	});
#endif
#if _BLOCKING_LOOP_
#undef Dinv
#undef b
#undef scaling_Inv
#undef precon_b
#endif
#if PARCEL_JACOBI_NODIV_UNIT && _BLOCKING_LOOP_
      };
      {
        constexpr bool unit_ok_ = std::is_same<T_vec,double>::value || std::is_same<T_vec,float>::value;
        if( unit_ok_ && bnorm == 1.0 ){ jacobi_k1_(std::false_type{}); }
        else                          { jacobi_k1_(std::true_type{}); }
      }
#endif
      q_[0].wait();


      // x(0) = (D^-1)b
      // x(1) = (D^-1)b - (D^-1){L+U}(D^-1)b
      //        ( I - (D^-1){L+U} )(D^-1)b  

      T_work_vec *x_now  = precon_x1_;
      T_work_vec *x_next = precon_x2_;
      q_[0].wait();
#if PARCEL_MPI
      crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b);
#endif
      q_[0].wait();
      
#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  // 連続した行の範囲 + restrict 付きのローカル変数 (下の #define で本体の名前を差し替える)
	  const auto * __restrict A_rr_ = A;
	  const auto * __restrict Rowptr_rr_ = Rowptr;
	  const auto * __restrict crsCol_rr_ = crsCol;
	  const auto * __restrict Doff_rr_ = Doff;
	  const auto * __restrict Dinv_rr_ = Dinv;
	  const auto * __restrict Aorig_rr_ = Aorig;
	  const auto * __restrict precon_b_rr_ = precon_b;
	  auto * __restrict x_rr_ = x;
	  auto * __restrict x_next_rr_ = x_next;
	  const size_t jchunk_ = (N + nblock - 1) / nblock;
	  const size_t jis_ = ii[0]*jchunk_;
	  const size_t jie_ = (jis_ + jchunk_ < (size_t)N) ? (jis_ + jchunk_) : (size_t)N;
	  {
#if PARCEL_JACOBI_ROW2
	    size_t i2_ = jis_;
	    if constexpr (row2_){
	      for(; i2_ + 1 < jie_; i2_ += 2){
	        PARCEL_JROW2_PAIR(precon_b_rr_, (iterMax==1))
	      }
	    }
	    for(size_t i=i2_;i<jie_;i++){
#else
	    for(size_t i=jis_;i<jie_;i++){
#endif
#define A A_rr_
#define Rowptr Rowptr_rr_
#define crsCol crsCol_rr_
#define Doff Doff_rr_
#define Dinv Dinv_rr_
#define Aorig Aorig_rr_
#define precon_b precon_b_rr_
#define x x_rr_
#define x_next x_next_rr_
#else
	      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
	      T_calc x_i = precon_b[i];
#if PARCEL_PRECON_LADDER
	      {
	        // 対角を飛ばした並びで SpMV と同じ梯子 4/2/1。減算は左結合なので、単純ループと同じ順序・同じ結果になる。
	        PARCEL_JROW_BEGIN
	        PARCEL_LINT k = 0;
	        for(; k + 4 <= jn_; k += 4){
	          const PARCEL_LINT j0 = PARCEL_JPOS(k  ), j1 = PARCEL_JPOS(k+1), j2 = PARCEL_JPOS(k+2), j3 = PARCEL_JPOS(k+3);
	          PARCEL_JLU_LET(A0, j0); T_calc x0 = precon_b[crsCol[j0]];
	          PARCEL_JLU_LET(A1, j1); T_calc x1 = precon_b[crsCol[j1]];
	          PARCEL_JLU_LET(A2, j2); T_calc x2 = precon_b[crsCol[j2]];
	          PARCEL_JLU_LET(A3, j3); T_calc x3 = precon_b[crsCol[j3]];
	          x_i = x_i - A0*x0 - A1*x1 - A2*x2 - A3*x3;
	        }
	        if(k + 2 <= jn_){
	          const PARCEL_LINT j0 = PARCEL_JPOS(k  ), j1 = PARCEL_JPOS(k+1);
	          PARCEL_JLU_LET(A0, j0); T_calc x0 = precon_b[crsCol[j0]];
	          PARCEL_JLU_LET(A1, j1); T_calc x1 = precon_b[crsCol[j1]];
	          x_i = x_i - A0*x0 - A1*x1;
	          k += 2;
	        }
	        if(k < jn_){
	          const PARCEL_LINT j0 = PARCEL_JPOS(k);
	          PARCEL_JLU_LET(A0, j0); T_calc x0 = precon_b[crsCol[j0]];
	          x_i = x_i - A0*x0;
	        }
	      }
#else
	      {
	        // 対角の前と後ろの 2 本のループ (対角を除いた並びを先頭から順に足す)
	        PARCEL_JROW_BEGIN
	        (void)jn_;
	        const PARCEL_LINT jd_ = js_ + dlim_;   // 対角の位置 (飛ばさない行は je_)
	        for(PARCEL_LINT j = js_; j < jd_; j++){
	          PARCEL_JLU_LET(A_, j); T_calc x_ = precon_b[crsCol[j]];
	          x_i = x_i - A_*x_;
	        }
	        for(PARCEL_LINT j = (jd_ < je_ ? jd_ + 1 : je_); j < je_; j++){
	          PARCEL_JLU_LET(A_, j); T_calc x_ = precon_b[crsCol[j]];
	          x_i = x_i - A_*x_;
	        }
	      }
#endif
	      if( iterMax==1 ){
		x[i] = x_i;
	      }else{
		x_next[i] = x_i;
	      }

#if _BLOCKING_LOOP_
	    }
	  }
	});
#else
	});
#endif
#if _BLOCKING_LOOP_
#undef A
#undef Rowptr
#undef crsCol
#undef Doff
#undef Dinv
#undef Aorig
#undef precon_b
#undef x
#undef x_next
#endif
      q_[0].wait();

      // x(i+1) = (D^-1) { b - {L+U}x(i) } 
      for(PARCEL_INT iter = 1;iter<iterMax;iter++){
	q_[0].wait();
	if(iter%2==0){
	  x_now  = precon_x1_;
	  x_next = precon_x2_;
	}else{
	  x_now  = precon_x2_;
	  x_next = precon_x1_;
	}
	
	
#if PARCEL_MPI
	crsMat_->template SpMVcomm< T_work_vec >(q_,x_now);
#endif
	q_[0].wait();
	
#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  // 連続した行の範囲 + restrict 付きのローカル変数 (下の #define で本体の名前を差し替える)
	  const auto * __restrict A_rr_ = A;
	  const auto * __restrict Rowptr_rr_ = Rowptr;
	  const auto * __restrict crsCol_rr_ = crsCol;
	  const auto * __restrict Doff_rr_ = Doff;
	  const auto * __restrict Dinv_rr_ = Dinv;
	  const auto * __restrict Aorig_rr_ = Aorig;
	  const auto * __restrict precon_b_rr_ = precon_b;
	  auto * __restrict x_rr_ = x;
	  auto * __restrict x_next_rr_ = x_next;
	  const auto * __restrict x_now_rr_ = x_now;
	  const size_t jchunk_ = (N + nblock - 1) / nblock;
	  const size_t jis_ = ii[0]*jchunk_;
	  const size_t jie_ = (jis_ + jchunk_ < (size_t)N) ? (jis_ + jchunk_) : (size_t)N;
	  {
#if PARCEL_JACOBI_ROW2
	    size_t i2_ = jis_;
	    if constexpr (row2_){
	      for(; i2_ + 1 < jie_; i2_ += 2){
	        PARCEL_JROW2_PAIR(x_now_rr_, ((iter+1)==iterMax))
	      }
	    }
	    for(size_t i=i2_;i<jie_;i++){
#else
	    for(size_t i=jis_;i<jie_;i++){
#endif
#define A A_rr_
#define Rowptr Rowptr_rr_
#define crsCol crsCol_rr_
#define Doff Doff_rr_
#define Dinv Dinv_rr_
#define Aorig Aorig_rr_
#define precon_b precon_b_rr_
#define x x_rr_
#define x_next x_next_rr_
#define x_now x_now_rr_
#else
        q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		T_calc x_i = precon_b[i];
#if PARCEL_PRECON_LADDER
		{
		  // 対角を飛ばした並びで SpMV と同じ梯子 4/2/1。減算は左結合なので、単純ループと同じ順序・同じ結果になる。
		  PARCEL_JROW_BEGIN
		  PARCEL_LINT k = 0;
		  for(; k + 4 <= jn_; k += 4){
		    const PARCEL_LINT j0 = PARCEL_JPOS(k  ), j1 = PARCEL_JPOS(k+1), j2 = PARCEL_JPOS(k+2), j3 = PARCEL_JPOS(k+3);
		    PARCEL_JLU_LET(A0, j0); T_calc x0 = x_now[crsCol[j0]];
		    PARCEL_JLU_LET(A1, j1); T_calc x1 = x_now[crsCol[j1]];
		    PARCEL_JLU_LET(A2, j2); T_calc x2 = x_now[crsCol[j2]];
		    PARCEL_JLU_LET(A3, j3); T_calc x3 = x_now[crsCol[j3]];
		    x_i = x_i - A0*x0 - A1*x1 - A2*x2 - A3*x3;
		  }
		  if(k + 2 <= jn_){
		    const PARCEL_LINT j0 = PARCEL_JPOS(k  ), j1 = PARCEL_JPOS(k+1);
		    PARCEL_JLU_LET(A0, j0); T_calc x0 = x_now[crsCol[j0]];
		    PARCEL_JLU_LET(A1, j1); T_calc x1 = x_now[crsCol[j1]];
		    x_i = x_i - A0*x0 - A1*x1;
		    k += 2;
		  }
		  if(k < jn_){
		    const PARCEL_LINT j0 = PARCEL_JPOS(k);
		    PARCEL_JLU_LET(A0, j0); T_calc x0 = x_now[crsCol[j0]];
		    x_i = x_i - A0*x0;
		  }
		}
#else
		{
		  // 対角の前と後ろの 2 本のループ (対角を除いた並びを先頭から順に足す)
		  PARCEL_JROW_BEGIN
		  (void)jn_;
		  const PARCEL_LINT jd_ = js_ + dlim_;   // 対角の位置 (飛ばさない行は je_)
		  for(PARCEL_LINT j = js_; j < jd_; j++){
		    PARCEL_JLU_LET(A_, j); T_calc x_ = x_now[crsCol[j]];
		    x_i = x_i - A_*x_;
		  }
		  for(PARCEL_LINT j = (jd_ < je_ ? jd_ + 1 : je_); j < je_; j++){
		    PARCEL_JLU_LET(A_, j); T_calc x_ = x_now[crsCol[j]];
		    x_i = x_i - A_*x_;
		  }
		}
#endif
		if( ( (iter+1)==iterMax) ){
		  x[i] = x_i;
		}else{
		  x_next[i] = x_i;
		}
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
#if _BLOCKING_LOOP_
#undef A
#undef Rowptr
#undef crsCol
#undef Doff
#undef Dinv
#undef Aorig
#undef precon_b
#undef x
#undef x_next
#undef x_now
#endif
	q_[0].wait();
      }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
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
	q_[0].wait();
      }

      return ;

    }
#undef PARCEL_JLU_LET
#undef PARCEL_JROW_BEGIN
#undef PARCEL_JPOS
#if PARCEL_JACOBI_ROW2
#undef PARCEL_JROW2_SEG
#undef PARCEL_JROW2_PAIR
#endif

    template<
      class T_vec,class T_precon_mat,class T_work_vec,class T_calc
      >
    void jacobi_method_dia_main(
				sycl::queue *q_ ,
				PARCEL_INT N,
				PARCEL_INT NND,
				PARCEL_INT iterMax,
				T_vec *x, // out  
				T_vec *b, // in
				T_precon_mat *A, // (D^-1)(L+U)
				PARCEL_INT *offset,
				PARCEL_INT max_up_offset,
				PARCEL_INT max_down_offset,
				T_precon_mat *Dinv, // (D^-1)
				double *scaling_Inv,
				T_work_vec *precon_x1_,
				T_work_vec *precon_x2_,
				T_work_vec *precon_b){
#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif
  
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      const bool SCALING_VECTOR = SCALING_VECTOR_;

      q_[0].wait();

      // solve Ax = b
      // x(i+1) = (D^-1) { b - {L+U}x(i) } 

      // MPI ALLREDUCEでスケーリング係数を計算する必要がある
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
      const double bnorm = bnorm_;

#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){
#else
	      const size_t MAX = N;
	      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		  const T_calc Dinv_ = Dinv[i];
		  const T_calc b_    = b[i] / bnorm;
		  if( SCALING_MATRIX == true ){ 
		    const T_calc val = scaling_Inv[i];
		    precon_b[i] = Dinv_*b_*val;
		  }else{
		    precon_b[i] = Dinv_*b_;
		  }


#if _BLOCKING_LOOP_
		}
		}
	    });
#else
	});
#endif
      q_[0].wait();

      // x(0) = (D^-1)b
      // x(1) = (D^-1)b - (D^-1){L+U}(D^-1)b
      //        ( I - (D^-1){L+U} )(D^-1)b  

      T_work_vec *x_now  = precon_x1_;
      T_work_vec *x_next = precon_x2_;
#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i_=iii;i_<imax;i_++){
#else
	      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
#endif
		  PARCEL_INT i = i_;
		  T_calc x_i = precon_b[i];
		  if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
		    for(PARCEL_INT j = 0; j < NND; j++){
		      PARCEL_INT i0 = i + offset[j];
		      T_calc A_ = A[i+j*N];
		      if( (-1<i0) && (i0<N) ){
			T_calc x_ = precon_b[i0];
			x_i = x_i - A_*x_;
		      }
		    }
		  }else{
		    for(PARCEL_INT j = 0; j < NND; j++){
		      PARCEL_INT i0 = i + offset[j];
		      T_calc A_ = A[i+j*N];
		      T_calc x_ = precon_b[i0];
		      x_i = x_i - A_*x_;
		    }
		  }

		  if( iterMax==1 ){
		    x[i] = x_i;
		  }else{
		    x_next[i] = x_i;
		  }

#if _BLOCKING_LOOP_
		}
		}
	    });
#else
	});
#endif
      q_[0].wait();

      // x(i+1) = (D^-1) { b - {L+U}x(i) } 
      for(PARCEL_INT iter = 1;iter<iterMax;iter++){

	swap_ptr<T_work_vec>(&x_now,&x_next);
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i_=iii;i_<imax;i_++){
#else
		q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){
#endif
		    PARCEL_INT i = i_;
		    T_calc x_i = precon_b[i];
		    if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){
		      for(PARCEL_INT j = 0; j < NND; j++){
			PARCEL_INT i0 = i + offset[j];
			T_calc A_ = A[i+j*N];
			if( (-1<i0) && (i0<N) ){
			  T_calc x_ = x_now[i0];
			  x_i = x_i - A_*x_;
			}
		      }
		    }else{
		      for(PARCEL_INT j = 0; j < NND; j++){
			PARCEL_INT i0 = i + offset[j];
			T_calc A_ = A[i+j*N];
			T_calc x_ = x_now[i0];
			x_i = x_i - A_*x_;
			}
		    }

		    if( ( (iter+1)==iterMax) ){
		      x[i] = x_i;
		    }else{
		      x_next[i] = x_i;
		    }
#if _BLOCKING_LOOP_
		  }
		  }
	      });
#else
	  });
#endif
	q_[0].wait();
      }

      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
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
	q_[0].wait();
      }

      return ;

    }

    template<
      class T_vec,class T_mat,class T_precon_mat,class T_work_vec,class T_calc
      >
    void jacobi_method(
		       vector<T_vec> &b_, // in
		       vector<T_vec> &x_, // out  
		       crsMat<T_precon_mat> &crsMat // (D^-1)(L+U)
		       ){

      
      PARCEL_INT iterMax = JACOBI_ITER_;

      if(iterMax<=0){
	iterMax=1;
      }
      const size_t N   = crsMat.n();

      sycl::queue *q_    = crsMat.queue();
      jacobi_work_ensure_(iterMax, q_);
      vector<T_precon_mat> &mat = crsMat.mat();
      PARCEL_LINT *Rowptr = crsMat.rowptr_data();
      PARCEL_LINT *crsCol = crsMat.crsCol_data();

      const auto Tvec_name              = typeid(T_vec).name();
      const auto T_precon_mat_name       = typeid(T_precon_mat).name();
      const auto Tprecon_work_vec_name  = typeid(T_work_vec).name();
      const auto Tprecon_calc_name      = typeid(T_calc).name();

      if( ddtype_name == Tprecon_calc_name ){ 
	// calc : dd
	auto scalingInv_vec = scalingInv_vec_.data();
	if( ddtype_name == Tvec_name ){ 
	  // in/out vec : dd
	  double *b_hi = (double *)b_.data_hi();
	  double *b_lo = (double *)b_.data_lo();
	  double *x_hi = (double *)x_.data_hi();
	  double *x_lo = (double *)x_.data_lo();
	  if( ddtype_name == T_precon_mat_name ){ 
	    //precon matrix : dd
	    double *A_hi    = (double *)mat.data_hi();
	    double *A_lo    = (double *)mat.data_lo();
	    double *Dinv_hi = (double *)DiagInv_.data_hi();
	    double *Dinv_lo = (double *)DiagInv_.data_lo();

	    if( ddtype_name == Tprecon_work_vec_name ){ 
	      //precon vector : dd
	      double *precon_b_hi  = (double *)precon_b_.data_hi();
	      double *precon_b_lo  = (double *)precon_b_.data_lo();
	      double *precon_x1_hi = (double *)precon_x1_.data_hi();
	      double *precon_x1_lo = (double *)precon_x1_.data_lo();
	      double *precon_x2_hi = (double *)precon_x2_.data_hi();
	      double *precon_x2_lo = (double *)precon_x2_.data_lo();
	      ddjacobi_method_crs_main(
				       q_,N,iterMax,
				       x_hi, x_lo, 
				       b_hi, b_lo, 
				       A_hi, A_lo, 
				       Rowptr,crsCol,
				       Dinv_hi,Dinv_lo,
				       scalingInv_vec,
				       precon_x1_hi,precon_x1_lo,
				       precon_x2_hi,precon_x2_lo,
				       precon_b_hi,precon_b_lo
				       );
	    }else{
	      //precon vector : d
	      auto precon_b  = precon_b_.data();
	      auto precon_x1 = precon_x1_.data();
	      auto precon_x2 = precon_x2_.data();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		>(
		  q_,N,iterMax,
		  x_hi, x_lo, 
		  b_hi, b_lo, 
		  A_hi, A_lo, 
		  Rowptr,crsCol,
		  Dinv_hi,Dinv_lo,
		  scalingInv_vec,
		  precon_x1,
		  precon_x2,
		  precon_b
		  );
	    }
	  }else{
	    //precon matrix : d
	    auto A    = mat.data();
	    auto Dinv = DiagInv_.data();

	    if( ddtype_name == Tprecon_work_vec_name ){ 
	      //precon vector : dd
	      double *precon_b_hi  = (double *)precon_b_.data_hi();
	      double *precon_b_lo  = (double *)precon_b_.data_lo();
	      double *precon_x1_hi = (double *)precon_x1_.data_hi();
	      double *precon_x1_lo = (double *)precon_x1_.data_lo();
	      double *precon_x2_hi = (double *)precon_x2_.data_hi();
	      double *precon_x2_lo = (double *)precon_x2_.data_lo();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type 
		>(
		  q_,N,iterMax,
		  x_hi, x_lo, 
		  b_hi, b_lo, 
		  A,
		  Rowptr,crsCol,
		  Dinv,
		  scalingInv_vec,
		  precon_x1_hi,precon_x1_lo,
		  precon_x2_hi,precon_x2_lo,
		  precon_b_hi,precon_b_lo
		  );
	    }else{
	      //precon vector : d
	      auto precon_x1 = precon_x1_.data();
	      auto precon_x2 = precon_x2_.data();
	      auto precon_b  = precon_b_.data();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
		typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		>(
		  q_,N,iterMax,
		  x_hi, x_lo, 
		  b_hi, b_lo,
		  A,
		  Rowptr,crsCol,
		  Dinv,
		  scalingInv_vec,
		  precon_x1,
		  precon_x2,
		  precon_b
		  );

	    }
	  }
	}else{
	  // in/out vec : d
	  auto b = b_.data();
	  auto x = x_.data();

	  if( ddtype_name == T_precon_mat_name ){ 
	    //precon matrix : dd
	    double *A_hi    = (double *)mat.data_hi();
	    double *A_lo    = (double *)mat.data_lo();
	    double *Dinv_hi = (double *)DiagInv_.data_hi();
	    double *Dinv_lo = (double *)DiagInv_.data_lo();

	    if( ddtype_name == Tprecon_work_vec_name ){ 
	      //precon vector : dd
	      double *precon_b_hi  = (double *) precon_b_.data_hi();
	      double *precon_b_lo  = (double *) precon_b_.data_lo();
	      double *precon_x1_hi = (double *) precon_x1_.data_hi();
	      double *precon_x1_lo = (double *) precon_x1_.data_lo();
	      double *precon_x2_hi = (double *) precon_x2_.data_hi();
	      double *precon_x2_lo = (double *) precon_x2_.data_lo();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type 
		>(
		  q_,N,iterMax,
		  x,
		  b,
		  A_hi, A_lo, 
		  Rowptr,crsCol,
		  Dinv_hi,Dinv_lo,
		  scalingInv_vec,
		  precon_x1_hi,precon_x1_lo,
		  precon_x2_hi,precon_x2_lo,
		  precon_b_hi,precon_b_lo
		  );
	    }else{
	      //precon vector : d
	      auto precon_b  = precon_b_.data();
	      auto precon_x1 = precon_x1_.data();
	      auto precon_x2 = precon_x2_.data();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		>(
		  q_,N,iterMax,
		  x, 
		  b, 
		  A_hi, A_lo, 
		  Rowptr,crsCol,
		  Dinv_hi,Dinv_lo,
		  scalingInv_vec,
		  precon_x1,
		  precon_x2,
		  precon_b
		  );
	    }
	  }else{
	    //precon matrix : d
	    auto A    = mat.data();
	    auto Dinv = DiagInv_.data();

	    if( ddtype_name == Tprecon_work_vec_name ){ 
	      //precon vector : dd
	      double *precon_b_hi  = (double *) precon_b_.data_hi();
	      double *precon_b_lo  = (double *) precon_b_.data_lo();
	      double *precon_x1_hi = (double *) precon_x1_.data_hi();
	      double *precon_x1_lo = (double *) precon_x1_.data_lo();
	      double *precon_x2_hi = (double *) precon_x2_.data_hi();
	      double *precon_x2_lo = (double *) precon_x2_.data_lo();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type 
		>(
		  q_,N,iterMax,
		  x,
		  b,
		  A,
		  Rowptr,crsCol,
		  Dinv,
		  scalingInv_vec,
		  precon_x1_hi,precon_x1_lo,
		  precon_x2_hi,precon_x2_lo,
		  precon_b_hi,precon_b_lo
		  );
	    }else{
	      //precon vector : d
	      auto precon_x1 = precon_x1_.data();
	      auto precon_x2 = precon_x2_.data();
	      auto precon_b  = precon_b_.data();
	      ddjacobi_method_crs_main<
		typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type,
		typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		>(
		  q_,N,iterMax,
		  x,
		  b,
		  A,
		  Rowptr,crsCol,
		  Dinv,
		  scalingInv_vec,
		  precon_x1,
		  precon_x2,
		  precon_b
		  );

	    }
	  }

	}

      }else{
	// calc : T
	// A : (D^-1){L+U} 行列
	auto A    = mat.data();
	auto Dinv = DiagInv_.data();
	auto b = b_.data();
	auto x = x_.data();
	auto precon_b  = precon_b_.data();
	auto precon_x1 = precon_x1_.data();
	auto precon_x2 = precon_x2_.data();
	auto scalingInv_vec = scalingInv_vec_.data();
	// otf_ のときだけ元の行列の値を渡す (otf_ は前処理の計算型・前処理行列・元の行列が dd でないときだけ立つ)
	decltype(std::declval<vector<Tmat>&>().data()) Aorig = nullptr;
	if(otf_){ Aorig = otf_src_->mat().data(); }
	// 値をその場で計算する版 (OTF=true) と保存した値を読む版 (OTF=false) をここで 1 回だけ選ぶ
	//   (カーネルの中に実行時の分岐を置かない)。PARCEL_JACOBI_ONTHEFLY=0 では OTF=true 版を作らない。
#define PARCEL_JACOBI_CRS_CALL_(OTF_V) \
	jacobi_method_crs_main< \
	    OTF_V, \
	    typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type , \
	    typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type , \
	    typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type , \
	    typename std::result_of<parcel::parcel_type_to_baseType(T_calc) >::type , \
	    decltype(Aorig) \
	    >( q_, N, iterMax, x, b, A, Rowptr, crsCol, Dinv, scalingInv_vec, \
	       precon_x1, precon_x2, precon_b, Aorig, diagoff_.data() )
#if PARCEL_JACOBI_ONTHEFLY
	if(otf_){ PARCEL_JACOBI_CRS_CALL_(true); }
	else    { PARCEL_JACOBI_CRS_CALL_(false); }
#else
	PARCEL_JACOBI_CRS_CALL_(false);
#endif
#undef PARCEL_JACOBI_CRS_CALL_

	    }

	  return ;
      }


      template<
	class T_vec,class T_mat,class T_precon_mat,class T_work_vec,class T_calc
	>
	void jacobi_method(
			   vector<T_vec> &b_, // in
			   vector<T_vec> &x_, // out  
			   diaMat<T_precon_mat> &diaMat // (D^-1)(L+U)
			   ){
	PARCEL_INT iterMax = JACOBI_ITER_;
	if(iterMax<=0){
	  iterMax=1;
	}
	const size_t N   = diaMat.n();
	const size_t NND = diaMat.nnd();
	jacobi_work_ensure_(iterMax, diaMat.queue());

	sycl::queue *q_    = diaMat.queue();
	vector<T_precon_mat> &mat = diaMat.mat();
	PARCEL_INT *offset = diaMat.offset_data();
	PARCEL_INT max_up_offset   = diaMat.max_up_offset();
	PARCEL_INT max_down_offset = diaMat.max_down_offset();

	const auto Tvec_name              = typeid(T_vec).name();
	const auto T_precon_mat_name      = typeid(T_precon_mat).name();
	const auto Tprecon_work_vec_name  = typeid(T_work_vec).name();
	const auto Tprecon_calc_name      = typeid(T_calc).name();


	if( ddtype_name == Tprecon_calc_name ){ 
	  // calc : dd
	  auto scalingInv_vec = scalingInv_vec_.data();
	  if( ddtype_name == Tvec_name ){ 
	    // in/out vec : dd
	    double *b_hi = (double *)b_.data_hi();
	    double *b_lo = (double *)b_.data_lo();
	    double *x_hi = (double *)x_.data_hi();
	    double *x_lo = (double *)x_.data_lo();
	    if( ddtype_name == T_precon_mat_name ){ 
	      //precon matrix : dd
	      double *A_hi    = (double *)mat.data_hi();
	      double *A_lo    = (double *)mat.data_lo();
	      double *Dinv_hi = (double *)DiagInv_.data_hi();
	      double *Dinv_lo = (double *)DiagInv_.data_lo();

	      if( ddtype_name == Tprecon_work_vec_name ){ 
		//precon vector : dd
		double *precon_b_hi  = (double *)precon_b_.data_hi();
		double *precon_b_lo  = (double *)precon_b_.data_lo();
		double *precon_x1_hi = (double *)precon_x1_.data_hi();
		double *precon_x1_lo = (double *)precon_x1_.data_lo();
		double *precon_x2_hi = (double *)precon_x2_.data_hi();
		double *precon_x2_lo = (double *)precon_x2_.data_lo();
		ddjacobi_method_dia_main(
					 q_,N,NND,iterMax,
					 x_hi, x_lo, 
					 b_hi, b_lo, 
					 A_hi, A_lo, 
					 offset,
					 max_up_offset,
					 max_down_offset,
					 Dinv_hi,Dinv_lo,
					 scalingInv_vec,
					 precon_x1_hi,precon_x1_lo,
					 precon_x2_hi,precon_x2_lo,
					 precon_b_hi,precon_b_lo
					 );
	      }else{
		//precon vector : d
		auto precon_b  = precon_b_.data();
		auto precon_x1 = precon_x1_.data();
		auto precon_x2 = precon_x2_.data();
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x_hi, x_lo, 
		    b_hi, b_lo, 
		    A_hi, A_lo, 
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv_hi,Dinv_lo,
		    scalingInv_vec,
		    precon_x1,
		    precon_x2,
		    precon_b
		    );
	      }
	    }else{
	      //precon matrix : d
	      auto A    = mat.data();
	      auto Dinv = DiagInv_.data();

	      if( ddtype_name == Tprecon_work_vec_name ){ 
		//precon vector : dd
		double *precon_b_hi  = (double *)precon_b_.data_hi();
		double *precon_b_lo  = (double *)precon_b_.data_lo();
		double *precon_x1_hi = (double *)precon_x1_.data_hi();
		double *precon_x1_lo = (double *)precon_x1_.data_lo();
		double *precon_x2_hi = (double *)precon_x2_.data_hi();
		double *precon_x2_lo = (double *)precon_x2_.data_lo();
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x_hi, x_lo, 
		    b_hi, b_lo, 
		    A,
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv,
		    scalingInv_vec,
		    precon_x1_hi,precon_x1_lo,
		    precon_x2_hi,precon_x2_lo,
		    precon_b_hi,precon_b_lo
		    );
	      }else{
		//precon vector : d
		auto precon_x1 = precon_x1_.data();
		auto precon_x2 = precon_x2_.data();
		auto precon_b  = precon_b_.data();
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x_hi, x_lo, 
		    b_hi, b_lo,
		    A,
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv,
		    scalingInv_vec,
		    precon_x1,
		    precon_x2,
		    precon_b
		    );
	      }
	    }
	  }else{
	    // in/out vec : d
	    auto b = b_.data();
	    auto x = x_.data();

	    if( ddtype_name == T_precon_mat_name ){ 
	      //precon matrix : dd
	      double *A_hi    = (double *)mat.data_hi();
	      double *A_lo    = (double *)mat.data_lo();
	      double *Dinv_hi = (double *)DiagInv_.data_hi();
	      double *Dinv_lo = (double *)DiagInv_.data_lo();

	      if( ddtype_name == Tprecon_work_vec_name ){ 
		//precon vector : dd
		double *precon_b_hi  = (double *) precon_b_.data_hi();
		double *precon_b_lo  = (double *) precon_b_.data_lo();
		double *precon_x1_hi = (double *) precon_x1_.data_hi();
		double *precon_x1_lo = (double *) precon_x1_.data_lo();
		double *precon_x2_hi = (double *) precon_x2_.data_hi();
		double *precon_x2_lo = (double *) precon_x2_.data_lo();

		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x,
		    b,
		    A_hi, A_lo, 
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv_hi,Dinv_lo,
		    scalingInv_vec,
		    precon_x1_hi,precon_x1_lo,
		    precon_x2_hi,precon_x2_lo,
		    precon_b_hi,precon_b_lo
		    );
	      }else{
		//precon vector : d
		auto precon_b  = precon_b_.data();
		auto precon_x1 = precon_x1_.data();
		auto precon_x2 = precon_x2_.data();
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x, 
		    b, 
		    A_hi, A_lo, 
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv_hi,Dinv_lo,
		    scalingInv_vec,
		    precon_x1,
		    precon_x2,
		    precon_b
		    );
	      }
	    }else{
	      //precon matrix : d
	      auto A    = mat.data();
	      auto Dinv = DiagInv_.data();

	      if( ddtype_name == Tprecon_work_vec_name ){ 
		//precon vector : dd
		double *precon_b_hi  = (double *) precon_b_.data_hi();
		double *precon_b_lo  = (double *) precon_b_.data_lo();
		double *precon_x1_hi = (double *) precon_x1_.data_hi();
		double *precon_x1_lo = (double *) precon_x1_.data_lo();
		double *precon_x2_hi = (double *) precon_x2_.data_hi();
		double *precon_x2_lo = (double *) precon_x2_.data_lo();

		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x,
		    b,
		    A,
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv,
		    scalingInv_vec,
		    precon_x1_hi,precon_x1_lo,
		    precon_x2_hi,precon_x2_lo,
		    precon_b_hi,precon_b_lo
		    );

	      }else{
		//precon vector : d
		auto precon_x1 = precon_x1_.data();
		auto precon_x2 = precon_x2_.data();
		auto precon_b  = precon_b_.data();

		jacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type ,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_calc) >::type 
		  >(
		    q_,N,NND,iterMax,
		    x,
		    b,
		    A,
		    offset,
		    max_up_offset,
		    max_down_offset,
		    Dinv,
		    scalingInv_vec,
		    precon_x1,
		    precon_x2,
		    precon_b
		    );
	      }
	    }

	  }

	}else{
	  // calc : T
	  // A : (D^-1){L+U} 行列
	  auto A    = mat.data();
	  auto Dinv = DiagInv_.data();
	  auto b = b_.data();
	  auto x = x_.data();
	  auto precon_b  = precon_b_.data();
	  auto precon_x1 = precon_x1_.data();
	  auto precon_x2 = precon_x2_.data();
	  auto scalingInv_vec = scalingInv_vec_.data();
	  jacobi_method_dia_main<
	    typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_calc) >::type 
	    >(
	      q_,
	      N,
	      NND,
	      iterMax,
	      x, 
	      b, 
	      A, 
	      offset,
	      max_up_offset,
	      max_down_offset,
	      Dinv,
	      scalingInv_vec,
	      precon_x1, 
	      precon_x2, 
	      precon_b
	      );

	}
	    return ;

	    }

    void solver(
		vector<Tvec> &b_,
		vector<Tvec> &x_
		){

      size_t matrix_type = precon_LUmatrix_.matrix_type();      
      if(matrix_type == CRS_MATRIX){
	parcel::crsMat<Tprecon_mat> &crsMat = *precon_LUmatrix_.crsMat_data();
	jacobi_method<
	  Tvec,Tmat,Tprecon_mat,Tprecon_vec,Tcalc
	  >(b_,x_,crsMat);
	return ;
      }

      if(matrix_type == DIA_MATRIX){
	parcel::diaMat<Tprecon_mat> &diaMat = *precon_LUmatrix_.diaMat_data();
	jacobi_method<
	  Tvec,Tmat,Tprecon_mat,Tprecon_vec,Tcalc
	  >(b_,x_,diaMat);
	return ;
      }

      return ;

    }

    void init(
		parcel::crsMat<Tmat> &crsMat,
		PARCEL_INT ITER,
		bool SCALING_MATRIX,
		bool SCALING_VECTOR
		){
#if PARCEL_MPI
        crsMat_ = &crsMat;
#endif      
	size_t N = crsMat.n();
	sycl::queue *q_ = crsMat.queue();
	setQueue(q_);
	set_JACOBI_ITER(ITER);
	set_SCALING_MATRIX(SCALING_MATRIX);
	set_SCALING_VECTOR(SCALING_VECTOR);

	get_jacobi_DinvLUmatrix(crsMat);

	// ヤコビ前処理用作業配列 (反復回数 1 では x1/x2 を 1 要素にする。jacobi_work_alloc_ 参照)
#if PARCEL_MPI
	PARCEL_INT nadd = crsMat.get_recv_size();
	jacobi_work_alloc_(N+nadd,q_);
#else
	jacobi_work_alloc_(N,q_);
#endif	


	return ;
      }

      void init(
		parcel::diaMat<Tmat> &diaMat,
		PARCEL_INT ITER,
		bool SCALING_MATRIX,
		bool SCALING_VECTOR
		){
	size_t N = diaMat.n();
	sycl::queue *q_ = diaMat.queue();
	setQueue(q_);
	set_JACOBI_ITER(ITER);
	set_SCALING_MATRIX(SCALING_MATRIX);
	set_SCALING_VECTOR(SCALING_VECTOR);

	get_jacobi_DinvLUmatrix(diaMat);

	// ヤコビ前処理用作業配列 (jacobi_work_alloc_ 参照)
	jacobi_work_alloc_(N,q_);

	return ;
      }


    };

  }

#endif

