#ifndef PARCEL_BLAS_DIA_
#define PARCEL_BLAS_DIA_

template<
  class Talpha,
  class Tbeta,
  class TdiaMat,
  class Tvecx,
  class Tvecy
  >
void ddSpMV_main_DIA(
		     parcel::vector<Talpha>  &alpha_,	
		     parcel::vector<Tbeta>   &beta_,	
		     parcel::diaMat<TdiaMat> &A_,
		     parcel::vector<Tvecx>   &x_, 
		     parcel::vector<Tvecy>   &y_
		     ){
  const auto TdiaMat_name = typeid(TdiaMat).name();
  const auto Tvecx_name   = typeid(Tvecx).name();
  const auto Tvecy_name   = typeid(Tvecy).name();
  const auto Talpha_name  = typeid(Talpha).name();
  const auto Tbeta_name   = typeid(Tbeta).name();

  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t n  = std::min(nx,ny);
  size_t nnd = A_.nnd();

  PARCEL_INT *offset = A_.offset_data();
  PARCEL_INT max_up_offset = A_.max_up_offset();
  PARCEL_INT max_down_offset = A_.max_down_offset();

#if 0
  // 入力が全てdouble-doubleの結果

  // vector : dd
  double *x_hi = (double *)x_.data_hi();
  double *x_lo = (double *)x_.data_lo();
  double *y_hi = (double *)y_.data_hi();
  double *y_lo = (double *)y_.data_lo(); 

  // matrix : dd
  double *A_hi = (double *)A_.mat_data_hi();
  double *A_lo = (double *)A_.mat_data_lo();

  // scalar : dd 
  double *alpha_hi = (double *)alpha_.data_hi();
  double *alpha_lo = (double *)alpha_.data_lo();
  double *beta_hi  = (double *)beta_.data_hi();
  double *beta_lo  = (double *)beta_.data_lo();

  PARCEL_INT *offset = A_.offset_data();
  PARCEL_INT max_up_offset = A_.max_up_offset();
  PARCEL_INT max_down_offset = A_.max_down_offset();
  blas_.dddiamv(
  		n,
  		nnd,
  		alpha_hi,alpha_lo,
  		A_hi,A_lo,
  		offset,max_up_offset,max_down_offset,
  		x_hi,x_lo,
  		nx,
  		beta_hi,beta_lo,
  		y_hi,y_lo,
  		ny);

#else

  if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
    // vector : dd
    double *x_hi = (double *)x_.data_hi();
    double *x_lo = (double *)x_.data_lo();
    double *y_hi = (double *)y_.data_hi();
    double *y_lo = (double *)y_.data_lo(); 
    if( ddtype_name == TdiaMat_name ){
      // matrix : dd
      double *A_hi = (double *)A_.mat_data_hi();
      double *A_lo = (double *)A_.mat_data_lo();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();
	blas_.dddiamv(
		      n,
		      nnd,
		      alpha_hi,alpha_lo,
		      A_hi,A_lo,
		      offset,max_up_offset,max_down_offset,
		      x_hi,x_lo,
		      nx,
		      beta_hi,beta_lo,
		      y_hi,y_lo,
		      ny);
      }else{
	// scalar : d
	// std::cout<<"alpha,beta : scala"<<std::endl;
	auto alpha = alpha_.data();
	auto beta  = beta_.data();

#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type,
	  bool,
	  bool,
	  bool
	  >(
	    n,
	    nnd,
	    alpha,
	    A_hi,A_lo,
	    offset,max_up_offset,max_down_offset,
	    x_hi,x_lo,
	    nx,
	    beta,
	    y_hi,y_lo,
	    ny);
#endif
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
#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  bool,
	  bool,
	  typename std::result_of< parcel::parcel_type_to_baseType( TdiaMat ) >::type,
	  bool,
	  bool
	  >(
	    n,
	    nnd,
	    alpha_hi,alpha_lo,
	    A,
	    offset,max_up_offset,max_down_offset,
	    x_hi,x_lo,
	    nx,
	    beta_hi,beta_lo,
	    y_hi,y_lo,
	    ny);
#endif
	
      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TdiaMat ) >::type,
	  bool,
	  bool
	  >(
	    n,
	    nnd,
	    alpha,
	    A,
offset,max_up_offset,max_down_offset,
	    x_hi,x_lo,
	    nx,
	    beta,
	    y_hi,y_lo,
	    ny);
#endif	
      }      
    }
  }else{
    // vector : d
    auto *x = x_.data();
    auto *y = y_.data();
    if( ddtype_name == TdiaMat_name ){
      // matrix : dd
      double *A_hi = (double *)A_.mat_data_hi();
      double *A_lo = (double *)A_.mat_data_lo();
      if( (ddtype_name == Talpha_name) && (Talpha_name == Tbeta_name) ){ 
	// scalar : dd 
	double *alpha_hi = (double *)alpha_.data_hi();
	double *alpha_lo = (double *)alpha_.data_lo();
	double *beta_hi  = (double *)beta_.data_hi();
	double *beta_lo  = (double *)beta_.data_lo();

#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  bool,
	  bool,
	  bool,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy  ) >::type
	  >(
	    n,
	    nnd,
	    alpha_hi,alpha_lo,
	    A_hi,A_lo,
offset,max_up_offset,max_down_offset,
	    x,
	    nx,
	    beta_hi,beta_lo,
	    y,
	    ny);
#endif
      }else{
	// scalar : d
	auto alpha = alpha_.data();
	auto beta  = beta_.data();

#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta  ) >::type,
	  bool,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy  ) >::type
	  >(
	    n,
	    nnd,
	    alpha,
	    A_hi,A_lo,
offset,max_up_offset,max_down_offset,
	    x,
	    nx,
	    beta,
	    y,
	    ny);
#endif
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

#if _A64FX_HIPSYCL_
        std::cout<<" not support : parcel_blas_dia.hpp "<<std::endl;	
#else
	blas_.dddiamv<
	  bool,
	  bool,
	  typename std::result_of< parcel::parcel_type_to_baseType( TdiaMat ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type
	  >(
	    n,
	    nnd,
	    alpha_hi,alpha_lo,
	    A,
offset,max_up_offset,max_down_offset,
	    x,
	    nx,
	    beta_hi,beta_lo,
	    y,
	    ny);
#endif
      }else{
	// scalar : d
	//	    std::cout<<" ddddd "<<std::endl;
	auto alpha = alpha_.data();
	auto beta  = beta_.data();
	blas_.dddiamv<
	  typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( TdiaMat ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
	  typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type
	  >(
	    n,
	    nnd,
	    alpha,
	    A,
offset,max_up_offset,max_down_offset,
	    x,
	    nx,
	    beta,
	    y,
	    ny);

      }
    }
  }


#endif

}

template<
  class Talpha,
  class Tbeta,
  class TdiaMat,
  class Tvecx,
  class Tvecy,
  class Tcalc >
void SpMV_main_DIA(
		   parcel::vector<Talpha>  &alpha_,	
		   parcel::vector<Tbeta>   &beta_,	
		   parcel::diaMat<TdiaMat> &A_,
		   parcel::vector<Tvecx>   &x_, 
		   parcel::vector<Tvecy>   &y_){

  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t n  = std::min(nx,ny);
  size_t nnd = A_.nnd();
  auto A = A_.mat_data();
  PARCEL_INT *offset = A_.offset_data();
  PARCEL_INT max_up_offset = A_.max_up_offset();
  PARCEL_INT max_down_offset = A_.max_down_offset();
  auto x = x_.data();
  auto y = y_.data();
  auto alpha = alpha_.data();
  auto beta  = beta_.data();

  //      std::cout<<" SpMV_main_DIA::blas_.diamv "<<std::endl;
  blas_.diamv<
    typename std::result_of< parcel::parcel_type_to_baseType( Talpha  ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( TdiaMat ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tvecx   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tbeta   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tvecy   ) >::type,
    typename std::result_of< parcel::parcel_type_to_baseType( Tcalc   ) >::type
    > (
       n,
       nnd,
       alpha,
       A,
       offset,max_up_offset,max_down_offset,
       x,
       nx,
       beta,
       y,
       ny);

}

#endif

