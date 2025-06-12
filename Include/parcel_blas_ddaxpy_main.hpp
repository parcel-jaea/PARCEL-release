#ifndef PARCEL_BLAS_DDAXPY_MAIN_HPP_
#define PARCEL_BLAS_DDAXPY_MAIN_HPP_

template<class Talpha,class Tbeta,class Tvecx, class Tvecy, class Tvecz>
void ddaxpy_main(
		 parcel::vector<Talpha> &alpha_,
		 parcel::vector<Tbeta>  &beta_,
		 parcel::vector<Tvecx>  &x_, 
		 parcel::vector<Tvecy>  &y_,
		 parcel::vector<Tvecy>  &z_,){

  // (Talpha == Tbeta && Tvecx == Tvecy == Tvecz) が条件
  const auto Tvecx_name = typeid(Tvecx).name();
  const auto Tvecy_name = typeid(Tvecy).name();
  const auto Talpha_name = typeid(Talpha).name();
  const auto Tbeta_name  = typeid(Tbeta).name();

  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t nz = z_.size(); 
  size_t n  = std::min(nx,ny);

  if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
    double *x_hi = (double *)x_.data_hi();
    double *x_lo = (double *)x_.data_lo();
    double *y_hi = (double *)y_.data_hi();
    double *y_lo = (double *)y_.data_lo();
    double *z_hi = (double *)z_.data_hi();
    double *z_lo = (double *)z_.data_lo();
    if( ddtype_name == Talpha_name ){ 
      double *alpha_hi_ = (double *)alpha_.data_hi(); 
      double *alpha_lo_ = (double *)alpha_.data_lo(); 
      double *beta_hi_  = (double *)beta_.data_hi(); 
      double *beta_lo_  = (double *)beta_.data_lo(); 
      blas_.ddaxpy(
		   n,
		   alpha_hi_, alpha_lo_,
		   beta_hi_ , beta_lo_,
		   x_hi,x_lo,
		   nx,
		   y_hi,y_lo,
		   ny,
		   z_hi,z_lo,
		   nz
		   );
    }else{
      double *alpha_hi = (double *)alpha_.data(); 
      double *alpha_lo = (double *)zero.data();
      double *beta_hi  = (double *)beta_.data(); 
      double *beta_lo  = (double *)zero.data();
      blas_.ddaxpy(
		   n,
		   alpha_hi, alpha_lo,
		   beta_hi , beta_lo,
		   x_hi,x_lo,
		   nx,
		   y_hi,y_lo,
		   ny,
		   z_hi,z_lo,
		   nz
		   );
    }
  }else{
    auto x = x_.data();
    auto y = y_.data();
    auto z = z_.data();

    if( ddtype_name == Talpha_name ){ 
      double *alpha_hi = (double *)alpha_.data_hi(); 
      double *alpha_lo = (double *)alpha_.data_lo(); 

      double *beta_hi = (double *)beta_.data_hi(); 
      double *beta_lo = (double *)beta_.data_lo(); 

      blas_.ddaxpy<
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	>(n,alpha_hi,alpha_lo,beta_hi,beta_lo,x,nx,y,ny,z,nz);
	  
    }else{
      auto *alpha = alpha_.data(); 	  	  
      auto *beta = beta_.data(); 	  	  
      blas_.ddaxpy<
	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecz ) >::type
	>(n,alpha,beta,x,nx,y,ny,z,nz);
    }

  }
}

template<class Talpha,class Tbeta,class Tvecx, class Tvecy>
void ddaxpy_main(
		 parcel::vector<Talpha> &alpha_,
		 parcel::vector<Tbeta>  &beta_,
		 parcel::vector<Tvecx>  &x_, 
		 parcel::vector<Tvecy>  &y_){
  // (Talpha == Tbeta && Tvecx == Tvecy == Tvecz) が条件
  const auto Tvecx_name = typeid(Tvecx).name();
  const auto Tvecy_name = typeid(Tvecy).name();
  const auto Talpha_name = typeid(Talpha).name();
  const auto Tbeta_name  = typeid(Tbeta).name();

  size_t nx = x_.size(); 
  size_t ny = y_.size(); 
  size_t n  = std::min(nx,ny);

  if( (ddtype_name == Tvecx_name) && ( Tvecx_name == Tvecy_name) ){ 	
    double *x_hi = (double *)x_.data_hi();
    double *x_lo = (double *)x_.data_lo();
    double *y_hi = (double *)y_.data_hi();
    double *y_lo = (double *)y_.data_lo();
    if( ddtype_name == Talpha_name ){ 
      double *alpha_hi_ = (double *)alpha_.data_hi(); 
      double *alpha_lo_ = (double *)alpha_.data_lo(); 
      double *beta_hi_  = (double *)beta_.data_hi(); 
      double *beta_lo_  = (double *)beta_.data_lo(); 
      blas_.ddaxpy(
		   n,
		   alpha_hi_, alpha_lo_,
		   beta_hi_ , beta_lo_,
		   x_hi,x_lo,
		   nx,
		   y_hi,y_lo,
		   ny);
    }else{
      double *alpha_hi = (double *)alpha_.data(); 
      double *alpha_lo = (double *)zero.data();
      double *beta_hi  = (double *)beta_.data(); 
      double *beta_lo  = (double *)zero.data();
      blas_.ddaxpy(
		   n,
		   alpha_hi, alpha_lo,
		   beta_hi , beta_lo,
		   x_hi,x_lo,
		   nx,
		   y_hi,y_lo,
		   ny);
    }
  }else{
    auto x = x_.data();
    auto y = y_.data();

    if( ddtype_name == Talpha_name ){ 
      double *alpha_hi = (double *)alpha_.data_hi(); 
      double *alpha_lo = (double *)alpha_.data_lo(); 

      double *beta_hi = (double *)beta_.data_hi(); 
      double *beta_lo = (double *)beta_.data_lo(); 

      blas_.ddaxpy<
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	>(n,alpha_hi,alpha_lo,beta_hi,beta_lo,x,nx,y,ny);
	  
    }else{
      auto *alpha = alpha_.data(); 	  	  
      auto *beta = beta_.data(); 	  	  
      blas_.ddaxpy<
	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecx ) >::type,
	typename std::result_of< parcel::parcel_type_to_baseType( Tvecy ) >::type
	>(n,alpha,beta,x,nx,y,ny);
    }

  }
}

#endif
