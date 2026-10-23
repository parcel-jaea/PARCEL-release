// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#include "parcel_nblock.hpp"
template<class T_matrix,class T_vec,class T_work>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    T_vec *r_hi, 
			    T_work *y_hi, 
			    T_vec *z_hi, 
			    T_matrix *L_hi, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    T_matrix *U_hi, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<T_vec,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<T_vec,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = 0.0;
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = 0.0; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = 0.0; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = 0.0; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = 0.0; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = 0.0; 
				   double U_lo_  = 0.0; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 

			       z_hi[i] = z_hi_; 

			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z_hi[i] = z_hi[i] * rnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_matrix,class T_vec>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    T_vec *r_hi, 
			    double *y_hi, 
			    double *y_lo, 
			    T_vec *z_hi, 
			    T_matrix *L_hi, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    T_matrix *U_hi, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<T_vec,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<T_vec,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = 0.0;
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = y_lo[Col]; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = 0.0; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
				 y_lo[i] = y_lo_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = y_lo[i]; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = 0.0; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = y_lo[Col]; 
				   double U_lo_  = 0.0; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 
			       y_lo[i] = z_lo_; 
			       z_hi[i] = z_hi_; 

			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z_hi[i] = z_hi[i] * rnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_matrix,class T_work>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    double *r_hi, 
			    double *r_lo, 
			    T_work *y_hi, 
			    double *z_hi, 
			    double *z_lo, 
			    T_matrix *L_hi, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    T_matrix *U_hi, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<double,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<double,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = r_lo[i];
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = 0.0; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = 0.0; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = 0.0; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = 0.0; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = 0.0; 
				   double U_lo_  = 0.0; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 

			       z_hi[i] = z_hi_; 
			       z_lo[i] = z_lo_; 
			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		double rnorm_hi = rnorm;
		double rnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(rnorm_hi,rnorm_lo,z_hi[i],z_lo[i],z_hi[i],z_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_matrix>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    double *r_hi, 
			    double *r_lo, 
			    double *y_hi, 
			    double *y_lo, 
			    double *z_hi, 
			    double *z_lo, 
			    T_matrix *L_hi, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    T_matrix *U_hi, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<double,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<double,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = r_lo[i];
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = y_lo[Col]; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = 0.0; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
				 y_lo[i] = y_lo_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = y_lo[i]; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = 0.0; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = y_lo[Col]; 
				   double U_lo_  = 0.0; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 
			       y_lo[i] = z_lo_; 
			       z_hi[i] = z_hi_; 
			       z_lo[i] = z_lo_; 
			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		double rnorm_hi = rnorm;
		double rnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(rnorm_hi,rnorm_lo,z_hi[i],z_lo[i],z_hi[i],z_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_vec,class T_work>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    T_vec *r_hi, 
			    T_work *y_hi, 
			    T_vec *z_hi, 
			    double *L_hi, 
			    double *L_lo, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    double *U_hi, 
			    double *U_lo, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<T_vec,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<T_vec,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = 0.0;
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = 0.0; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = L_lo[j]; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = 0.0; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = U_lo[j]; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = 0.0; 
				   double U_lo_  = U_lo[j]; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 

			       z_hi[i] = z_hi_; 

			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z_hi[i] = z_hi[i] * rnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_vec>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    T_vec *r_hi, 
			    double *y_hi, 
			    double *y_lo, 
			    T_vec *z_hi, 
			    double *L_hi, 
			    double *L_lo, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    double *U_hi, 
			    double *U_lo, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<T_vec,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<T_vec,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = 0.0;
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = y_lo[Col]; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = L_lo[j]; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
				 y_lo[i] = y_lo_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = y_lo[i]; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = U_lo[j]; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = y_lo[Col]; 
				   double U_lo_  = U_lo[j]; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 
			       y_lo[i] = z_lo_; 
			       z_hi[i] = z_hi_; 

			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		z_hi[i] = z_hi[i] * rnorm;
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 
template<class T_work>
void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    double *r_hi, 
			    double *r_lo, 
			    T_work *y_hi, 
			    double *z_hi, 
			    double *z_lo, 
			    double *L_hi, 
			    double *L_lo, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    double *U_hi, 
			    double *U_lo, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<double,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<double,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = r_lo[i];
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = 0.0; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = L_lo[j]; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = 0.0; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = U_lo[j]; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = 0.0; 
				   double U_lo_  = U_lo[j]; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 

			       z_hi[i] = z_hi_; 
			       z_lo[i] = z_lo_; 
			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		double rnorm_hi = rnorm;
		double rnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(rnorm_hi,rnorm_lo,z_hi[i],z_lo[i],z_hi[i],z_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 

void ddblockjacobi( 
 			    sycl::queue *q_, 
 			    PARCEL_INT N, 
			    double *r_hi, 
			    double *r_lo, 
			    double *y_hi, 
			    double *y_lo, 
			    double *z_hi, 
			    double *z_lo, 
			    double *L_hi, 
			    double *L_lo, 
			    PARCEL_LINT *L_Rowptr, 
			    PARCEL_LINT *L_crsCol, 
			    double *U_hi, 
			    double *U_lo, 
			    PARCEL_LINT *U_Rowptr, 
			    PARCEL_LINT *U_crsCol, 
			    double *scaling_Inv 
			     ){ 
#if _A64FX_HIPSYCL_ 
      const size_t nblock      = parcel::nblock_runtime(); 
      const size_t nlocalblock = 8192; 
#else 
      const size_t nblock      = N; 
      const size_t nlocalblock = 1; 
#endif 
  const bool SCALING_MATRIX = SCALING_MATRIX_; 
  const bool SCALING_VECTOR = SCALING_VECTOR_; 
  const size_t work_group_size = 1; // simd lengh 
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core 
  const size_t num_work_items  = num_work_groups * work_group_size; 
  size_t *nhead = blockjacobi_buffer_.nhead_data(); 
  size_t *ntail = blockjacobi_buffer_.ntail_data(); 
 
double rnorm_ = 1.0; 
if( SCALING_VECTOR == true ){  
	setQueue(q_); 
	if( SCALING_MATRIX == true ){  
	  rnorm_ = max<double,double,double>(N,r_hi,N,scaling_Inv,N); 
	}else{ 
	  rnorm_ = max<double,double>(N,r_hi,N); 
	} 
	q_[0].wait(); 
} 
const double rnorm = rnorm_; 
#if _A64FX_HIPSYCL_ 
  size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ 
      size_t global_id = ii; 
      size_t local_id  = 0; 
      size_t group_id  = ii; 
#else 
      q_[0].submit([&] (sycl::handler &h){ 
	  h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), 
			 [=](sycl::nd_item<1> item) { 
			   size_t global_id = item.get_global_id(0); 
			   size_t local_id  = item.get_local_id(0); 
			   size_t group_id  = item.get_group(0); 
#endif 
			   PARCEL_INT ihead = nhead[group_id]; 
			   PARCEL_INT itail = ntail[group_id]; 
 
			   if(local_id==0){ 
			     // 1. solve Ly = r 
			     for(PARCEL_INT i = ihead;i<itail;i++){ 
	  const double rnorm_hi_ = rnorm;
	  const double rnorm_lo_ = 0.0;
	  const double r_hi_ = r_hi[i];
	  const double r_lo_ = r_lo[i];
  double y_hi_ = 0.0; 
  double y_lo_ = 0.0; 
  double dddiv_var1;
  dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 
if( SCALING_MATRIX == true ){  
  double ddmul_var1; 
  double scal_Inv_hi = scaling_Inv[i]; 
  double scal_Inv_lo = 0.0; 
  ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
} 
			       for(PARCEL_LINT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = L_crsCol[j]; 
				 if(Col!=i){ 
				   // y_ = y_ - yL_*L_; 
				   double yL_hi_ = y_hi[Col]; 
				   double yL_lo_ = y_lo[Col]; 
				   double L_hi_  = L_hi[j]; 
				   double L_lo_  = L_lo[j]; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 
				 } 
			       } 
				 y_hi[i] = y_hi_; 
				 y_lo[i] = y_lo_; 
			     } 
 
			     // 2. solve Uz = y 
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ 
			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = y_lo[i]; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
			       for(PARCEL_LINT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){ 
				 PARCEL_LINT Col = U_crsCol[j]; 
				 if(Col==i){ 
				   // DiagInv = 1.0/U[j]; 
				   double a_hi = 1.0; 
				   double a_lo = 0.0; 
				   double U_hi_ = U_hi[j]; 
				   double U_lo_ = U_lo[j]; 
				   double dddiv_var1; 
				   dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 
				 }else{ 
				   // z_ = z_ - zL_*U_; 
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[j]; 
				   double zL_lo_ = y_lo[Col]; 
				   double U_lo_  = U_lo[j]; 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo); 
				   double d_hi = 0.0; 
				   double d_lo = 0.0; 
				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_); 
				 } 
			       } 
 
			       // z_=z_*DiagInv; 
			       double ddmul_var1; 
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 
			       y_lo[i] = z_lo_; 
			       z_hi[i] = z_hi_; 
			       z_lo[i] = z_lo_; 
			     } 
 
			   } 
 
			 }); 
#if _A64FX_HIPSYCL_ 
 
#else 
	}); 
#endif 
      q_[0].wait(); 
 
      if( SCALING_VECTOR == true ){ 
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
	    size_t N_ = N;
	    q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		double rnorm_hi = rnorm;
		double rnorm_lo = 0.0;
	        double ddmul_var1;
	        ddmul_calc1(rnorm_hi,rnorm_lo,z_hi[i],z_lo[i],z_hi[i],z_lo[i]);
#if _BLOCKING_LOOP_
	      }
	    }
	  });
#else
	  });
#endif
 }
      return ; 
    } 

