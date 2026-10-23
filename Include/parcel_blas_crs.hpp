// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_BLAS_CRS_
#define PARCEL_BLAS_CRS_

template<
  class Talpha,
  class Tbeta,
  class TcrsMat,
  class Tvecx,
  class Tvecy
  >
void ddSpMV_main_CRS(
		     parcel::vector<Talpha>  &alpha_,	
		     parcel::vector<Tbeta>   &beta_,	
		     parcel::crsMat<TcrsMat> &A_,
		     parcel::vector<Tvecx>   &x_, 
		     parcel::vector<Tvecy>   &y_
		     ){
  size_t maxCol = A_.maxCol();
  const size_t  dwork_bloncksize = maxCol*CRS_SPMV_BLOCKSIZE*8*2;
  // dwork_ を読み書きするのはブロック化版の crsmv / ddcrsmv だけで、
  // それが使われるかどうかは _BLOCKING_LOOP_SPMV_ が決める
  // (mixed_precision_crsmv.hpp の include 条件)。平坦版は引数に取るだけで
  // 一度も参照しない。確保の条件は必ずカーネルの選択と一致させること。
  //
  // 既定 (_BLOCKING_LOOP_SPMV_ = 0) では確保しない。ブロック化版の作業配列は
  // 32*maxCol*n 要素 (7 点ステンシルで 1 点あたり 1792 B) と大きいため。
#if _BLOCKING_LOOP_SPMV_
  check_dwork_size(dwork_bloncksize *2 *CRS_SPMV_NBLOCK);
#endif
  double *dwork = dwork_.data();

  const auto TcrsMat_name = typeid(TcrsMat).name();
  const auto Tvecx_name   = typeid(Tvecx).name();
  const auto Tvecy_name   = typeid(Tvecy).name();
  const auto Talpha_name  = typeid(Talpha).name();
  const auto Tbeta_name   = typeid(Tbeta).name();


  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t n  = std::min(nx,ny);
  size_t nnz = A_.nnz();
  auto Rowptr = A_.rowptr_data();
  auto crsCol = A_.crsCol_data();

  if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
    // vector : dd
    double *x_hi = (double *)x_.data_hi();
    double *x_lo = (double *)x_.data_lo();
    double *y_hi = (double *)y_.data_hi();
    double *y_lo = (double *)y_.data_lo(); 
    if( ddtype_name == TcrsMat_name ){
      // matrix : dd
      double *A_hi = (double *)A_.mat_data_hi();
      double *A_lo = (double *)A_.mat_data_lo();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();
	blas_.ddcrsmv(
		      n,
		      nnz,
		      alpha_hi,alpha_lo,
		      A_hi,A_lo,
		      Rowptr,
		      crsCol,
		      x_hi,x_lo,
		      nx,
		      beta_hi,beta_lo,
		      y_hi,y_lo,
		      ny,
		      dwork,dwork_bloncksize);
      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type
	  >(
	    n,
	    nnz,
	    alpha,
	    A_hi,A_lo,
	    Rowptr,
	    crsCol,
	    x_hi,x_lo,
	    nx,
	    beta,
	    y_hi,y_lo,ny,
	    dwork,dwork_bloncksize);
      }
    }else{
      // matrix : d
      auto A = A_.mat_data();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();
	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( TcrsMat ) >::type
	  >(
	    n,
	    nnz,
	    alpha_hi,alpha_lo,
	    A,
	    Rowptr,
	    crsCol,
	    x_hi,x_lo,
	    nx,
	    beta_hi,beta_lo,
	    y_hi,y_lo,ny,
	    dwork,dwork_bloncksize);

      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TcrsMat ) >::type
	  >(
	    n,
	    nnz,
	    alpha,
	    A,
	    Rowptr,
	    crsCol,
	    x_hi,x_lo,
	    nx,
	    beta,
	    y_hi,y_lo,
	    ny,
	    dwork,dwork_bloncksize);

      }
    }
  }else{
    // vector : d
    auto *x = x_.data();
    auto *y = y_.data();
    if( ddtype_name == TcrsMat_name ){
      // matrix : dd
      double *A_hi = (double *)A_.mat_data_hi();
      double *A_lo = (double *)A_.mat_data_lo();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();

	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy  ) >::type
	  >(
	    n,
	    nnz,
	    alpha_hi,alpha_lo,
	    A_hi,A_lo,
	    Rowptr,
	    crsCol,
	    x,
	    nx,
	    beta_hi,beta_lo,
	    y,
	    ny,
	    dwork,dwork_bloncksize);


      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();

	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy  ) >::type
	  >(
	    n,
	    nnz,
	    alpha,
	    A_hi,A_lo,
	    Rowptr,
	    crsCol,
	    x,
	    nx,
	    beta,
	    y,
	    ny,
	    dwork,dwork_bloncksize);

      }
    }else{
      // matrix : d
      auto A = A_.mat_data();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();

	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( TcrsMat ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type
	  >(
	    n,
	    nnz,
	    alpha_hi,alpha_lo,
	    A,
	    Rowptr,
	    crsCol,
	    x,
	    nx,
	    beta_hi,beta_lo,
	    y,
	    ny,
	    dwork,dwork_bloncksize);

      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
	blas_.ddcrsmv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TcrsMat ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type
	  >(
	    n,
	    nnz,
	    alpha,
	    A,
	    Rowptr,
	    crsCol,
	    x,
	    nx,
	    beta,
	    y,
	    ny,
	    dwork,dwork_bloncksize);

      }
    }
  }

}

template<
  class Talpha,
  class Tbeta,
  class TcrsMat,
  class Tvecx,
  class Tvecy,
  class Tcalc >
void SpMV_main_CRS(
		   parcel::vector<Talpha>  &alpha_,	
		   parcel::vector<Tbeta>   &beta_,	
		   parcel::crsMat<TcrsMat> &A_,
		   parcel::vector<Tvecx>   &x_, 
		   parcel::vector<Tvecy>   &y_){
  size_t maxCol = A_.maxCol();
  const size_t  dwork_bloncksize = maxCol*CRS_SPMV_BLOCKSIZE*8*2;
  // dwork_ を読み書きするのはブロック化版の crsmv / ddcrsmv だけで、
  // それが使われるかどうかは _BLOCKING_LOOP_SPMV_ が決める
  // (mixed_precision_crsmv.hpp の include 条件)。平坦版は引数に取るだけで
  // 一度も参照しない。確保の条件は必ずカーネルの選択と一致させること。
  //
  // 既定 (_BLOCKING_LOOP_SPMV_ = 0) では確保しない。ブロック化版の作業配列は
  // 32*maxCol*n 要素 (7 点ステンシルで 1 点あたり 1792 B) と大きいため。
#if _BLOCKING_LOOP_SPMV_
  check_dwork_size(dwork_bloncksize *2 *CRS_SPMV_NBLOCK);
#endif

  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t n  = std::min(nx,ny);
  size_t nnz = A_.nnz();
  auto A = A_.mat_data();
  PARCEL_LINT *Rowptr = A_.rowptr_data();
  PARCEL_LINT *crsCol = A_.crsCol_data();
  auto x = x_.data();
  auto y = y_.data();
  auto alpha = alpha_.data();
  auto beta  = beta_.data();
  double *dwork = dwork_.data();

  blas_.crsmv<
    typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( TcrsMat ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tcalc   ) >::type
    > (
       n,
       nnz,
       alpha,
       A,
       Rowptr,
       crsCol,
       x,
       nx,
       beta,
       y,
       ny,
       dwork,dwork_bloncksize);
}


#endif
