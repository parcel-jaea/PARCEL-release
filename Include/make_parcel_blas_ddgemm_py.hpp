    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
bool, 
bool, 
bool 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
bool, 
bool, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
bool, 
bool, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  auto B = B_.data(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  auto B = B_.data(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  auto A = A_.data(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  auto A = A_.data(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  auto A = A_.data(); 
  auto B = B_.data(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	double *alpha_hi = (double *)alpha_.data_hi(); 
	double *alpha_lo = (double *)alpha_.data_lo(); 
	double *beta_hi = (double *)beta_.data_hi(); 
	double *beta_lo = (double *)beta_.data_lo(); 
  auto A = A_.data(); 
  auto B = B_.data(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha_hi,alpha_lo, 
beta_hi,beta_lo, 
A,ldA, 
B,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
bool 
>(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  auto B = B_.data(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
>(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name == TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  double *A_hi = (double *)A_.data_hi(); 
  double *A_lo = (double *)A_.data_lo(); 
  auto B = B_.data(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A_hi,A_lo,ldA, 
B,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  auto A = A_.data(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
>(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name == TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  auto A = A_.data(); 
  double *B_hi = (double *)B_.data_hi(); 
  double *B_lo = (double *)B_.data_lo(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
bool, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B_hi,B_lo,ldB, 
C,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name == TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  auto A = A_.data(); 
  auto B = B_.data(); 
  double *C_hi = (double *)C_.data_hi(); 
  double *C_lo = (double *)C_.data_lo(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
>(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
bool 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C_hi,C_lo,ldC 
); }
 return ;
 }
    if( 
 (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && 
 (ddtype_name != TmatA_name) && 
 (ddtype_name != TmatB_name) && 
 (ddtype_name != TmatC_name) 
 ){ 
	auto alpha = alpha_.data(); 
	auto beta = beta_.data(); 
  auto A = A_.data(); 
  auto B = B_.data(); 
  auto C = C_.data(); 
  if( TransA == "T" ) { 
    if( sym == "S" ) { 
	blas_.ddgemmTNsym< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
>(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C,ldC 
); 
		}else{ 
		  blas_.ddgemmTN< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C,ldC 
); 
		} 
  }else{ 
      blas_.ddgemm< 
typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, 
typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type 
	 >(m,n,k, 
alpha, 
beta, 
A,ldA, 
B,ldB, 
C,ldC 
); }
 return ;
 }

