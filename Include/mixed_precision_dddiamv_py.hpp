// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
template<class TdataA,class TdataB,class TdataDIA,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA=bool,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA=bool,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA=bool,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB,class TdataDIA=bool,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA=bool,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA=bool,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA=bool,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA,class TdataB=bool,class TdataDIA=bool,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     TdataA *alpha, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha[0]; 
      double alpha_lo_ = 0.0; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA=bool,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA=bool,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA=bool,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB,class TdataDIA=bool,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     TdataB *beta, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta[0]; 
      double beta_lo_  = 0.0; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     TdataDIA *A, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A[i+j*N]; 
	    double A_lo_ = 0.0; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A[i+j*N]; 
	  double A_lo_ = 0.0; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA=bool,class TdataX,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA=bool,class TdataX,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     TdataX *x, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x[i+offset[j]]; 
	    double x_lo_ = 0.0; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x[i+offset[j]]; 
	  double x_lo_ = 0.0; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA=bool,class TdataX=bool,class TdataY>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     TdataY *y, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y[i]; 
      double y_lo_ = 0.0; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y[i] = yy_hi; 
 
    }); 
  
  return ; 
 
} 
template<class TdataA=bool,class TdataB=bool,class TdataDIA=bool,class TdataX=bool,class TdataY=bool>
void dddiamv( 
	     PARCEL_INT N, 
	     PARCEL_INT NND, 
	     double *alpha_hi, 
	     double *alpha_lo, 
	     double *A_hi, 
	     double *A_lo, 
	     PARCEL_INT *offset, 
	     PARCEL_INT max_up_offset, 
	     PARCEL_INT max_down_offset, 
	     double *x_hi, 
	     double *x_lo, 
	     PARCEL_INT ldx, 
	     double *beta_hi, 
	     double *beta_lo, 
	     double *y_hi, 
	     double *y_lo, 
	     PARCEL_INT ldy 
	     ){ 
  size_t MAX   = N; 
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ 
      const PARCEL_INT i = i_; 
      double Ax_hi=0.0; 
      double Ax_lo=0.0; 
      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ 
	    double A_hi_ = A_hi[i+j*N]; 
	    double A_lo_ = A_lo[i+j*N]; 
	    double x_hi_ = x_hi[i+offset[j]]; 
	    double x_lo_ = x_lo[i+offset[j]]; 
	    double c_hi = 0.0; 
	    double c_lo = 0.0; 
	    double ddadd_var1; 
	    double ddmul_var1; 
	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); 
	  } 
	} 
      }else{ 
	for(PARCEL_INT j = 0;j<NND;j++){ 
	  double A_hi_ = A_hi[i+j*N]; 
	  double A_lo_ = A_lo[i+j*N]; 
	  double x_hi_ = x_hi[i+offset[j]]; 
	  double x_lo_ = x_lo[i+offset[j]]; 
	  double c_hi = 0.0; 
	  double c_lo = 0.0; 
	  double ddadd_var1; 
	  double ddmul_var1; 
	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); 
	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   
	} 
      } 
      double alpha_hi_ = alpha_hi[0]; 
      double alpha_lo_ = alpha_lo[0]; 
      double beta_hi_  = beta_hi[0]; 
      double beta_lo_  = beta_lo[0]; 
      double y_hi_ = y_hi[i]; 
      double y_lo_ = y_lo[i]; 
 
      double ddadd_var1; 
      double ddmul_var1; 
      double ddmul_var2; 
 
      double by_hi = 0.0; 
      double by_lo = 0.0; 
      double aAx_hi = 0.0; 
      double aAx_lo = 0.0; 
      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); 
      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); 
 
      double yy_hi = 0.0; 
      double yy_lo = 0.0; 
      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); 
      y_hi[i] = yy_hi; 
      y_lo[i] = yy_lo; 
 
    }); 
  
  return ; 
 
} 

