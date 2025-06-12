#ifndef PARCEL_PRECON_JACOBI_0_
#define PARCEL_PRECON_JACOBI_0_

// jacobi_method_crs_main :: A64FX用のコードなので他の環境向けのコードを整備する
//   --> jacobi_method_crs_main,ddjacobi_method_crs_main

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
    
  public:
    precon_jacobi(){}
    ~precon_jacobi(){}

    // 全ての行で対角成分が存在することが条件
    // D + LU 行列を作成してprecon_matrix_のメモリを削除する
    // crsMatとしての入力は元の行列でないとダメ?
    void get_jacobi_DinvLUmatrix(
				 parcel::crsMat<Tmat> &crsMat
				 ){
      sycl::queue *q_ = crsMat.queue();
      size_t N   = crsMat.n();
      size_t NNZ = crsMat.nnz();

      // 対角要素
      DiagInv_.resize(N,q_);

      // 非対角要素
      precon_LUmatrix_.set_queue(q_);
      precon_LUmatrix_.resize(N,NNZ,CRS_MATRIX);

#if _A64FX_HIPSYCL_
      const size_t nblock      = 12;
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif

      vector<Tmat> &mat  = crsMat.mat();
      PARCEL_INT *Rowptr = crsMat.rowptr_data();
      PARCEL_INT *crsCol = crsMat.crsCol_data();

      parcel::crsMat<Tprecon_mat> &precon_LU_crsMat = *precon_LUmatrix_.crsMat_data();
      vector<Tprecon_mat> &LU_mat = precon_LU_crsMat.mat();
      PARCEL_INT *LU_Rowptr = precon_LU_crsMat.rowptr_data();
      PARCEL_INT *LU_crsCol = precon_LU_crsMat.crsCol_data();

      const auto Tmat_name = typeid(Tmat).name();
      const auto Tprecon_mat_name = typeid(Tprecon_mat).name();

      // ----------
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  for(size_t iii=ii*nlocalblock;iii<(N+1);iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > (N+1) ? (N+1) : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){	    
	      LU_Rowptr[i] = Rowptr[i]-i;
	    }
	  }
	});
      q_[0].wait();

      // ---------------------------------------------------
      // 行列に対するスケーリング
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      if( SCALING_MATRIX == true ){ 
	// std::cout<<"Matrix scaling"<<std::endl;
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
	//	std::cout<<" init jacobi dd "<<std::endl;
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
		  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_INT col = crsCol[j];
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
		  
		  PARCEL_INT j0 = LU_Rowptr[i];
		  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_INT col = crsCol[j];
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

		      LU_crsCol[j0] = col;
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
		  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_INT col = crsCol[j];
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
		  
		  PARCEL_INT j0 = LU_Rowptr[i];
		  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		    const PARCEL_INT col = crsCol[j];
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

		      LU_crsCol[j0] = col;
		      j0 = j0 + 1;
		    }
		  }
		}
	      }
	    });
	}

      }else{
	//	std::cout<<" init jacobi T "<<std::endl;
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
		for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_INT col = crsCol[j];
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
		PARCEL_INT j0 = LU_Rowptr[i];
		for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_INT col = crsCol[j];
		  if(i!=col){
		    if( SCALING_MATRIX == true ){ 
		      LU_A[j0] = PARCEL_MAX_FLOTING_TYPE ( (A[j] )*AmaxInv[i]) * DiagInv_val ;
		    }else{
		      LU_A[j0] = PARCEL_MAX_FLOTING_TYPE ( A[j] ) * DiagInv_val ;
		    }
		    LU_crsCol[j0] = col;
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
    // crsMatとしての入力は元の行列でないとダメ?
    void get_jacobi_DinvLUmatrix(
				 parcel::diaMat<Tmat> &diaMat
				 ){

    sycl::queue *q_ = diaMat.queue();
    const size_t N   = diaMat.n();
    const size_t NND = diaMat.nnd();
    const PARCEL_INT max_up_offset = diaMat.max_up_offset();
    const PARCEL_INT max_down_offset = diaMat.max_down_offset();

#if _A64FX_HIPSYCL_
    const size_t nblock      = 12;
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

    //    return ;
    // ---------------------------------------------------
    // 行列に対するスケーリング
    const bool SCALING_MATRIX = SCALING_MATRIX_;
    if( SCALING_MATRIX == true ){ 
      // std::cout<<"Matrix scaling"<<std::endl;
      scalingInv_vec_.resize(N,q_);
      auto *A       = mat.data();
      auto *AmaxInv = scalingInv_vec_.data();
      dia_maxInv<
	typename std::result_of<parcel::parcel_type_to_baseType(Tmat) >::type,
	double
	>(N,NND,A,AmaxInv);
    }
    q_[0].wait();

    //    return ;
    // ---------------------------------------------------
    // offset
    const PARCEL_INT diag_id = get_diag_id(NND,offset);

    //    std::cout<<"diag_id = "<<diag_id<<std::endl;
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
	std::cout<<" init jacobi dd "<<std::endl;
	// Mat : dd
	// std::cout<<"Mat : dd --> precon Mat : dd "<<std::endl;
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
	//std::cout<<"Mat : T --> precon Mat : dd "<<std::endl;
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
      //std::cout<<"Mat : T,dd --> precon Mat : T "<<std::endl;
      //	std::cout<<" init jacobi T "<<std::endl;
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
      class T_vec,class T_precon_mat,class T_work_vec,class T_calc
      >
    void jacobi_method_crs_main(
				sycl::queue *q_ ,
				PARCEL_INT N,
				PARCEL_INT iterMax,
				T_vec *x, // out  
				T_vec *b, // in
				T_precon_mat *A, // (D^-1)(L+U)
				PARCEL_INT *Rowptr,
				PARCEL_INT *crsCol,
				T_precon_mat *Dinv, // (D^-1)
				double *scaling_Inv,
				T_work_vec *precon_x1_, // 
				T_work_vec *precon_x2_, // 
				T_work_vec *precon_b){
      //      std::cout<<"JACOBI N = "<<N<<std::endl;
#if _A64FX_HIPSYCL_
      const size_t nblock      = 12;
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = N;
      const size_t nlocalblock = 1;
#endif
  
      const bool SCALING_MATRIX = SCALING_MATRIX_;
      const bool SCALING_VECTOR = SCALING_VECTOR_;

      q_[0].wait();

      // T_work_vec *x_now  = precon_x1_;
      // T_work_vec *x_next = precon_x2_;
      // solve Ax = b
      // x(i+1) = (D^-1) { b - {L+U}x(i) } 

      // const double bnorm = 1.0;
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
      //      std::cout<<" bnorm = "<<bnorm<<std::endl;
      // util_sycl::vector<double> work_bnorm;
      // work_bnorm.resize(1,q_);
      // work_bnorm.setValue(&bnorm_);
      // double *bnorm_ptr = work_bnorm.data();

#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  //	  double bnorm = bnorm_ptr[0];
	  ///	  double bnorm = 1.0;
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
	      // if( SCALING_VECTOR == true ){ 
	      //     precon_b[i] = precon_b[i]  ;
	      // }
#if 0
	      x[i] = Dinv_*b_;
	      //	      printf("Dinv = %f , %f \n",Dinv_,b_);
	      //	      x[i] = b_;
#endif

#if _BLOCKING_LOOP_
	    }
	  }
	});
#else
	});
#endif
      q_[0].wait();

      //      return ;

      
      // x(0) = (D^-1)b
      // x(1) = (D^-1)b - (D^-1){L+U}(D^-1)b
      //        ( I - (D^-1){L+U} )(D^-1)b  

      T_work_vec *x_now  = precon_x1_;
      T_work_vec *x_next = precon_x2_;
      q_[0].wait();
#if PARCEL_MPI
      //      crsMat_->template SpMVcomm< T_work_vec >(q_, x_now);
      crsMat_->template SpMVcomm< T_work_vec >(q_, precon_b);
#endif
      q_[0].wait();
      
#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){
#else
	      //	      const size_t MAX = N;
	      q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
	      T_calc x_i = precon_b[i];
	      for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		const PARCEL_INT col = crsCol[j];
		T_calc A_ = A[j];
		T_calc x_ = precon_b[col];
		x_i = x_i - A_*x_;
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
	q_[0].wait();
#if 1
	if(iter%2==0){
	  x_now  = precon_x1_;
	  x_next = precon_x2_;
	}else{
	  x_now  = precon_x2_;
	  x_next = precon_x1_;
	}
	
#else
	swap_ptr<T_work_vec>(&x_now,&x_next);
#endif
	
#if PARCEL_MPI
	crsMat_->template SpMVcomm< T_work_vec >(q_,x_now);
#endif
	q_[0].wait();
	
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
		//	const size_t MAX = N;
        q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		T_calc x_i = precon_b[i];
		for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){
		  const PARCEL_INT col = crsCol[j];
		  T_calc A_ = A[j];
		  T_calc x_ = x_now[col];
		  x_i = x_i - A_*x_;
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
	    //	    double bnorm = bnorm_ptr[0];
	    //   double bnorm = 1.0;
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
		//	    const size_t MAX = N;
	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){
#endif
		x[i] = x[i] * bnorm;
		//		x[i] = x[i] * x[i] * x[i] ;
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
				T_work_vec *precon_x1_, // 
				T_work_vec *precon_x2_, // 
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

      q_[0].wait();

      // T_work_vec *x_now  = precon_x1_;
      // T_work_vec *x_next = precon_x2_;
      // solve Ax = b
      // x(i+1) = (D^-1) { b - {L+U}x(i) } 

      //      const double bnorm = 1.0;
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
      //      std::cout<<"NND = "<<NND<<std::endl;
      //      std::cout<<"iterMax = "<<iterMax<<std::endl;

#if _BLOCKING_LOOP_
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  //	  double bnorm = bnorm_ptr[0];
	  ///	  double bnorm = 1.0;
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

#if 0
		  x[i] = b[i] * Dinv_;
#endif

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
	      //	      const size_t MAX = N;
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
		//	const size_t MAX = N;
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

      
      //      q_[0].wait();
      //      PARCEL_INT iterMax = JACOBI_ITER_;
      PARCEL_INT iterMax = JACOBI_ITER_;

      if(iterMax<=0){
	iterMax=1;
      }
      const size_t N   = crsMat.n();
      //  const size_t nnz = crsMat.nnz();

      sycl::queue *q_    = crsMat.queue();
      vector<T_precon_mat> &mat = crsMat.mat();
      PARCEL_INT *Rowptr = crsMat.rowptr_data();
      PARCEL_INT *crsCol = crsMat.crsCol_data();

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
	//	std::cout<<"calc jacobi"<<std::endl;
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
#if 0
	ddjacobi_method_crs_main<
	  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type ,
	  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
	  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
	  >(
#else
	jacobi_method_crs_main<
	    typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_calc) >::type 
	    >(
#endif
	      q_,
	      N,
	      iterMax,
	      x, 
	      b, 
	      A, 
	      Rowptr,
	      crsCol,
	      Dinv,
	      scalingInv_vec,
	      precon_x1, 
	      precon_x2, 
	      precon_b
	      );

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
	//	q_[0].wait();
	PARCEL_INT iterMax = JACOBI_ITER_;
	if(iterMax<=0){
	  iterMax=1;
	}
	const size_t N   = diaMat.n();
	const size_t NND = diaMat.nnd();

	//	std::cout<<"calc jacobi"<<std::endl;
	sycl::queue *q_    = diaMat.queue();
	vector<T_precon_mat> &mat = diaMat.mat();
	PARCEL_INT *offset = diaMat.offset_data();
	PARCEL_INT max_up_offset   = diaMat.max_up_offset();
	PARCEL_INT max_down_offset = diaMat.max_down_offset();

	const auto Tvec_name              = typeid(T_vec).name();
	const auto T_precon_mat_name      = typeid(T_precon_mat).name();
	const auto Tprecon_work_vec_name  = typeid(T_work_vec).name();
	const auto Tprecon_calc_name      = typeid(T_calc).name();

#if 1

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
		 // std::cout<<"precon calc : "<<std::endl;
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
		 // std::cout<<"precon calc : T_work_vec"<<std::endl;
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
		 // std::cout<<"precon calc : T_precon_mat"<<std::endl;
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
		 // std::cout<<"precon calc : T_precon_mat,T_work_vec"<<std::endl;
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

		 // std::cout<<"precon calc : T_vec"<<std::endl;
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
		 // std::cout<<"precon calc : T_vec,T_work_vec"<<std::endl;
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

		 // std::cout<<"precon calc : T_vec,T_precon_mat"<<std::endl;
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

		 // std::cout<<"precon calc : T_vec,T_precon_mat,T_work_vec"<<std::endl;
#if 1
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
#else
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
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
#endif
	      }
	    }

	  }

	}else{
	  //	// std::cout<<"calc jacobi"<<std::endl;
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
#if 0
		ddjacobi_method_dia_main<
		  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type,
		  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
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

#else
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
#endif

	}
#else

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
#if 0
	ddjacobi_method_dia_main<
	  typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type ,
	  typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
	  typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type 
	  >(
#else
	    jacobi_method_dia_main<
	    typename std::result_of<parcel::parcel_type_to_baseType(T_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_precon_mat) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_work_vec) >::type ,
	    typename std::result_of<parcel::parcel_type_to_baseType(T_calc) >::type 
	    >(
#endif
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
#endif
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
	// size_t nnz =  crsMat.nnz();
	sycl::queue *q_ = crsMat.queue();
	setQueue(q_);
	set_JACOBI_ITER(ITER);
	set_SCALING_MATRIX(SCALING_MATRIX);
	set_SCALING_VECTOR(SCALING_VECTOR);

	// parcel::crsMat<Tprecon_mat> precon_crsMat(n,nnz,q);
	get_jacobi_DinvLUmatrix(crsMat);
	//      precon_matrix_.resize(1,1,matrix_type);

	// ヤコビ前処理用作業配列
#if PARCEL_MPI
	PARCEL_INT nadd = crsMat.get_recv_size();
	precon_x1_.resize(N+nadd,q_);
	precon_x2_.resize(N+nadd,q_);
	precon_b_.resize(N+nadd,q_);
#else
	precon_x1_.resize(N,q_);
	precon_x2_.resize(N,q_);
	precon_b_.resize(N,q_);
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
	// size_t nnz =  diaMat.nnz();
	sycl::queue *q_ = diaMat.queue();
	setQueue(q_);
	set_JACOBI_ITER(ITER);
	set_SCALING_MATRIX(SCALING_MATRIX);
	set_SCALING_VECTOR(SCALING_VECTOR);

	// parcel::diaMat<Tprecon_mat> precon_diaMat(n,nnz,q);
	get_jacobi_DinvLUmatrix(diaMat);
	//      precon_matrix_.resize(1,1,matrix_type);

	// ヤコビ前処理用作業配列
	precon_x1_.resize(N,q_);
	precon_x2_.resize(N,q_);
	precon_b_.resize(N,q_);

	return ;
      }


    };

  }

#endif

