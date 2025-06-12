#ifndef PARCEL_PRECON_DDBLOCKJACOBI_
#define PARCEL_PRECON_DDBLOCKJACOBI_

#if 1

#include "parcel_precon_ddblockjacobi_crs_py.hpp"
#include "parcel_precon_ddblockjacobi_dia_py.hpp"


#else
// template<class T_matrix,class T_vec,class T_work>
template<class T_matrix,class T_vec,class T_work>
void ddblockjacobi(
		   sycl::queue *q_,
		   T_vec  *r_hi,
		   T_work *y_hi,
		   T_vec  *z_hi,
		   T_matrix *L_hi,
		   PARCEL_INT *L_Rowptr,
		   PARCEL_INT *L_crsCol,
		   T_matrix *U_hi,
		   PARCEL_INT *U_Rowptr,
		   PARCEL_INT *U_crsCol
		   ){

  const size_t work_group_size = 1; // simd lengh
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
  const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
  size_t *nhead = blockjacobi_buffer_.nhead_data();
  size_t *ntail = blockjacobi_buffer_.ntail_data();

#if _A64FX_HIPSYCL_
  q_[0].parallel_for(sycl::range<1>{BLOCK_JACOBI_NUMBLOCK_},[=](sycl::id<1> ii){
      // work_group_size=1限定で動作
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

			   // ブロック内並列化無し
			   if(local_id==0){
			     // 1. solve Ly = r
			     for(PARCEL_INT i = ihead;i<itail;i++){
			       // Lの対角要素は1.0が条件
			       double y_hi_ = r_hi[i];
			       double y_lo_ = 0.0;
			       for(PARCEL_INT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){
				 PARCEL_INT Col = L_crsCol[j];
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
				 // y[i] = y_;
				 y_hi[i] = y_hi_;
				 z_hi[i] = y_hi_;
			       }
			     }

			     // 2. solve Uz = y
			     // for(int i = ntail-1;i>(nhead-1);i--){
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){
			       double z_hi_ = y_hi[i];
			       double z_lo_ = 0.0;
			       double DiagInv_hi = 1.0;
			       double DiagInv_lo = 0.0;
			       for(PARCEL_INT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){
				 PARCEL_INT Col = U_crsCol[j];
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
				   double zL_lo_ = 0.0;
				   double U_hi_  = U_hi[j];
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

      return ;
    }

// double-double 
void ddblockjacobi(
		   sycl::queue *q_,
		   double *r_hi,
		   double *r_lo,
		   double *y_hi,
		   double *y_lo,
		   double *z_hi,
		   double *z_lo,
		   double *L_hi,
		   double *L_lo,
		   PARCEL_INT *L_Rowptr,
		   PARCEL_INT *L_crsCol,
		   double *U_hi,
		   double *U_lo,
		   PARCEL_INT *U_Rowptr,
		   PARCEL_INT *U_crsCol
		   ){

  const size_t work_group_size = 1; // simd lengh
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
  const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
  size_t *nhead = blockjacobi_buffer_.nhead_data();
  size_t *ntail = blockjacobi_buffer_.ntail_data();

#if _A64FX_HIPSYCL_
  q_[0].parallel_for(sycl::range<1>{BLOCK_JACOBI_NUMBLOCK_},[=](sycl::id<1> ii){
      // work_group_size=1限定で動作
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

			   // ブロック内並列化無し
			   if(local_id==0){
			     // 1. solve Ly = r
			     for(PARCEL_INT i = ihead;i<itail;i++){
			       // Lの対角要素は1.0が条件
			       double y_hi_ = r_hi[i];
			       double y_lo_ = r_lo[i];
			       for(PARCEL_INT j = L_Rowptr[i]; j < L_Rowptr[i+1]; j++){
				 PARCEL_INT Col = L_crsCol[j];
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
				 // y[i] = y_;
				 y_hi[i] = y_hi_;
				 y_lo[i] = y_lo_;

				 z_hi[i] = y_hi_;
				 z_lo[i] = y_lo_;
			       }
			     }

			     // 2. solve Uz = y
			     // for(int i = ntail-1;i>(nhead-1);i--){
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){
			       double z_hi_ = y_hi[i];
			       double z_lo_ = y_lo[i];
			       double DiagInv_hi = 1.0;
			       double DiagInv_lo = 0.0;
			       for(PARCEL_INT j = U_Rowptr[i]; j < U_Rowptr[i+1]; j++){
				 PARCEL_INT Col = U_crsCol[j];
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
				   double zL_lo_ = y_lo[Col];
				   double U_hi_  = U_hi[j];
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

      return ;
    }


template<class T_matrix,class T_vec,class T_work,class T_calc>
void ddblockjacobi(
		   sycl::queue *q_,
		   PARCEL_INT N,
		   T_vec  *r_hi,
		   T_work *y_hi,
		   T_vec  *z_hi,
		   T_matrix *L_hi,
		   PARCEL_INT *L_offset,
		   PARCEL_INT nndL,
		   T_matrix *U_hi,
		   PARCEL_INT *U_offset,
		   PARCEL_INT nndU,
		   double *scaling_Inv
		   ){

#if _A64FX_HIPSYCL_
  const size_t nblock      = 12;
  const size_t nlocalblock = 8192;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif

  const auto Tmat_name  = typeid(T_matrix).name();
  const auto Tcalc_name = typeid(T_calc).name();

  const bool SCALING_MATRIX = SCALING_MATRIX_;
  const bool SCALING_VECTOR = SCALING_VECTOR_;

  const size_t work_group_size = 1; // simd lengh
  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core
  const size_t num_work_items  = num_work_groups * work_group_size; // 全並列数
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
  q_[0].submit([&] (sycl::handler &h){
      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size),
		     [=](sycl::nd_item<1> item) {
		       size_t global_id = item.get_global_id(0);
		       size_t local_id  = item.get_local_id(0);
		       size_t group_id  = item.get_group(0);
#else
		       size_t MAX = BLOCK_JACOBI_NUMBLOCK_;
		       q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){
			   // work_group_size=1限定で動作
			   size_t global_id = ii;
			   size_t local_id  = 0;
			   size_t group_id  = ii;
#endif
			   PARCEL_INT ihead = nhead[group_id];
			   PARCEL_INT itail = ntail[group_id];
			   
			   // まずは並列化無し
			   if(local_id==0){
#if 0
			     for(size_t i = ihead;i<itail;i++){
			       z[i] = r[i];
			     }
#else

			     // 1. solve Ly = r
			     for(PARCEL_INT i = ihead;i<itail;i++){
			       // Lの対角要素は1.0が条件
			       // T_calc y_ = r[i] / rnorm;

			       const double rnorm_hi_ = rnorm;
			       const double rnorm_lo_ = 0.0;
			       const double r_hi_ = r_hi[i];
			       const double r_lo_ = 0.0;
			       double y_hi_ = 0.0; 
			       double y_lo_ = 0.0; 
			       double dddiv_var1;
			       dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_); 

			       if( SCALING_MATRIX == true ){ 
				 // y_ = y_*scaling_Inv[i];
				 double ddmul_var1; 
				 double scal_Inv_hi = scaling_Inv[i]; 
				 double scal_Inv_lo = 0.0; 
				 ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_); 
			       }

			       // 非対角ループ
			       for(PARCEL_INT j = 0; j < nndL-1; j++){
				 const PARCEL_INT offset = L_offset[j];
				 const PARCEL_INT Col    = i + offset;
				 if( ihead<=Col ){
				   // const T_calc yL_ = y[Col];
				   // const T_calc L_  = L[i+j*N];
				   //  y_ = y_ - yL_*L_;

				   double yL_hi_ = y_hi[Col];
				   double yL_lo_ = 0.0;
				   double L_hi_  = L_hi[i+j*N];
				   double L_lo_  = 0.0; 
 
				   double ddmul_var1; 
				   double ddsub_var1; 
				   double c_hi = 0.0; 
				   double c_lo = 0.0; 
				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo); 
				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_); 

				 }
			       }
			       // y[i] = y_;
			       y_hi[i] = y_hi_; 
			     }

			     // 2. solve Uz = y
			     // for(int i = ntail-1;i>(nhead-1);i--){
			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){
			       // T_calc z_ = y[i];
			       // T_calc DiagInv = 1.0;

			       double z_hi_ = y_hi[i]; 
			       double z_lo_ = 0.0; 
			       double DiagInv_hi = 1.0; 
			       double DiagInv_lo = 0.0; 
	      
			       // 非対角ループ
			       for(PARCEL_INT j = 1; j < nndU; j++){
				 const PARCEL_INT offset = U_offset[j];
				 const PARCEL_INT Col = i + offset;
				 if( Col<itail ){
				   // T_calc zL_ = y[Col];
				   // T_calc U_  = U[i+j*N];
				   // z_ = z_ - zL_*U_;
				   double zL_hi_ = y_hi[Col]; 
				   double U_hi_  = U_hi[i+j*N]; 
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

			       // 対角要素の逆数
			       // DiagInv = 1.0/U[i+0*N];
			       double a_hi = 1.0; 
			       double a_lo = 0.0; 
			       double U_hi_ = U_hi[i+0*N]; 
			       double U_lo_ = 0.0; 
			       double dddiv_var1; 
			       dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo); 

			       // z_   = z_*DiagInv;
			       // y[i] = z_;
			       // z[i] = z_;
			       //printf("r[%d] = %f, z[%d] = %f\n",i,r[i],i,z[i]);
			       double ddmul_var1;
			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_); 
			       y_hi[i] = z_hi_; 
			       z_hi[i] = z_hi_; 
			     }
#endif
			   }
	  
			 });

#if _A64FX_HIPSYCL_
		     });
#else

#endif
      q_[0].wait();

      if( SCALING_VECTOR == true ){ 
	const size_t N_ = N ;
#if _BLOCKING_LOOP_
	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	      for(size_t i=iii;i<imax;i++){
#else
		q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){
#endif
		    z_hi[i] = z_hi[i] * rnorm;
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

 
#endif

#endif
