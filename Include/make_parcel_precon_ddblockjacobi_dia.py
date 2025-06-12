
def make_template(file_w,T_matrix,T_vec,T_work):
    temp="template<" 
    if T_matrix: 
        temp=temp+"class T_matrix,"
    if T_vec: 
        temp=temp+"class T_vec,"
    if T_work: 
        temp=temp+"class T_work,"
    temp=temp+">"
    temp=temp.replace(",>",">\n")
    temp=temp.replace("template<>","\n")
    file_w.write(temp)
    return

def make_arg(file_w,T_matrix,T_vec,T_work):

    file_w.write("void ddblockjacobi( \n")
    file_w.write(" 			    sycl::queue *q_, \n")
    file_w.write(" 			    PARCEL_INT N, \n")
    if T_vec:
        file_w.write("			    T_vec *r_hi, \n")
    else:
        file_w.write("			    double *r_hi, \n")
        file_w.write("			    double *r_lo, \n")
    if T_work:
        file_w.write("			    T_work *y_hi, \n")
    else:
        file_w.write("			    double *y_hi, \n")
        file_w.write("			    double *y_lo, \n")
    if T_vec:
        file_w.write("			    T_vec *z_hi, \n")
    else:
        file_w.write("			    double *z_hi, \n")
        file_w.write("			    double *z_lo, \n")
    if T_matrix:
        file_w.write("			    T_matrix *L_hi, \n")
    else:
        file_w.write("			    double *L_hi, \n")
        file_w.write("			    double *L_lo, \n")
    file_w.write("			    PARCEL_INT *L_offset, \n")
    file_w.write("			    PARCEL_INT nndL, \n")
    if T_matrix:
        file_w.write("			    T_matrix *U_hi, \n")
    else:
        file_w.write("			    double *U_hi, \n")
        file_w.write("			    double *U_lo, \n")
    file_w.write("			    PARCEL_INT *U_offset, \n")
    file_w.write("			    PARCEL_INT nndU, \n")
    file_w.write("			    double *scaling_Inv \n")
    file_w.write("			     ){ \n")

    return

def make_main(file_w,T_matrix,T_vec,T_work):

    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  const size_t nblock      = 12; \n")
    file_w.write("  const size_t nlocalblock = 8192; \n")
    file_w.write("#else \n")
    file_w.write("  const size_t nblock      = N; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#endif \n")
    file_w.write(" \n")
    file_w.write(" \n")
    file_w.write("  const bool SCALING_MATRIX = SCALING_MATRIX_; \n")
    file_w.write("  const bool SCALING_VECTOR = SCALING_VECTOR_; \n")
    file_w.write(" \n")
    file_w.write("  const size_t work_group_size = 1; // simd lengh \n")
    file_w.write("  const size_t num_work_groups = BLOCK_JACOBI_NUMBLOCK_; // cpu core \n")
    file_w.write("  const size_t num_work_items  = num_work_groups * work_group_size; \n")
    file_w.write("  size_t *nhead = blockjacobi_buffer_.nhead_data(); \n")
    file_w.write("  size_t *ntail = blockjacobi_buffer_.ntail_data(); \n")
    file_w.write(" \n")
    file_w.write("  double rnorm_ = 1.0; \n")
    file_w.write("  if( SCALING_VECTOR == true ){  \n")
    file_w.write("    setQueue(q_); \n")
    file_w.write("    if( SCALING_MATRIX == true ){  \n")
    if T_vec:
        file_w.write("      rnorm_ = max<T_vec,double,double>(N,r_hi,N,scaling_Inv,N); \n")
    else:
        file_w.write("      rnorm_ = max<double,double,double>(N,r_hi,N,scaling_Inv,N); \n")
    file_w.write("    }else{ \n")
    if T_vec:
        file_w.write("      rnorm_ = max<T_vec,double>(N,r_hi,N); \n")
    else:
        file_w.write("      rnorm_ = max<double,double>(N,r_hi,N); \n")
    file_w.write("    } \n")
    file_w.write("    q_[0].wait(); \n")
    file_w.write("  } \n")
    file_w.write("  const double rnorm = rnorm_; \n")
    file_w.write(" \n")
    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  q_[0].submit([&] (sycl::handler &h){ \n")
    file_w.write("      h.parallel_for(sycl::nd_range<1>(num_work_items, work_group_size), \n")
    file_w.write("		     [=](sycl::nd_item<1> item) { \n")
    file_w.write("		       size_t global_id = item.get_global_id(0); \n")
    file_w.write("		       size_t local_id  = item.get_local_id(0); \n")
    file_w.write("		       size_t group_id  = item.get_group(0); \n")
    file_w.write("#else \n")
    file_w.write("		       size_t MAX = BLOCK_JACOBI_NUMBLOCK_; \n")
    file_w.write("		       q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> ii){ \n")
    file_w.write("			   size_t global_id = ii; \n")
    file_w.write("			   size_t local_id  = 0; \n")
    file_w.write("			   size_t group_id  = ii; \n")
    file_w.write("#endif \n")
    file_w.write("			   PARCEL_INT ihead = nhead[group_id]; \n")
    file_w.write("			   PARCEL_INT itail = ntail[group_id]; \n")
    file_w.write("			    \n")
    file_w.write("			   if(local_id==0){ \n")
    file_w.write("#if 0 \n")
    file_w.write("			     for(size_t i = ihead;i<itail;i++){ \n")
    file_w.write("			       z[i] = r[i]; \n")
    file_w.write("			     } \n")
    file_w.write("#else \n")
    file_w.write(" \n")
    file_w.write("			     // 1. solve Ly = r \n")
    file_w.write("			     for(PARCEL_INT i = ihead;i<itail;i++){ \n")
    file_w.write("			       const double rnorm_hi_ = rnorm; \n")
    file_w.write("			       const double rnorm_lo_ = 0.0; \n")
    file_w.write("			       const double r_hi_ = r_hi[i]; \n")
    if T_vec:
        file_w.write("			       const double r_lo_ = 0.0; \n")
    else:
        file_w.write("			       const double r_lo_ = r_lo[i]; \n")
    file_w.write("			       double y_hi_ = 0.0;  \n")
    file_w.write("			       double y_lo_ = 0.0;  \n")
    file_w.write("			       double dddiv_var1; \n")
    file_w.write("			       dddiv_calc1(r_hi_,r_lo_,rnorm_hi_,rnorm_lo_,y_hi_,y_lo_);  \n")
    file_w.write(" \n")
    file_w.write("			       if( SCALING_MATRIX == true ){  \n")
    file_w.write("				 double ddmul_var1;  \n")
    file_w.write("				 double scal_Inv_hi = scaling_Inv[i];  \n")
    file_w.write("				 double scal_Inv_lo = 0.0;  \n")
    file_w.write("				 ddmul_calc1(y_hi_,y_lo_,scal_Inv_hi,scal_Inv_lo,y_hi_,y_lo_);  \n")
    file_w.write("			       } \n")
    file_w.write(" \n")
    file_w.write("			       for(PARCEL_INT j = 0; j < nndL-1; j++){ \n")
    file_w.write("				 const PARCEL_INT offset = L_offset[j]; \n")
    file_w.write("				 const PARCEL_INT Col    = i + offset; \n")
    file_w.write("				 if( ihead<=Col ){ \n")
    file_w.write("				   double yL_hi_ = y_hi[Col]; \n")
    if T_work:
        file_w.write("				   double yL_lo_ = 0.0; \n")
    else:
        file_w.write("				   double yL_lo_ = y_lo[Col]; \n")
    file_w.write("				   double L_hi_  = L_hi[i+j*N]; \n")
    if T_matrix:
        file_w.write("				   double L_lo_  = 0.0;  \n")
    else:
        file_w.write("				   double L_lo_  = L_lo[i+j*N];  \n")
    file_w.write("  \n")
    file_w.write("				   double ddmul_var1;  \n")
    file_w.write("				   double ddsub_var1;  \n")
    file_w.write("				   double c_hi = 0.0;  \n")
    file_w.write("				   double c_lo = 0.0;  \n")
    file_w.write("				   ddmul_calc1(yL_hi_,yL_lo_,L_hi_,L_lo_,c_hi,c_lo);  \n")
    file_w.write("				   ddsub_calc1(y_hi_,y_lo_,c_hi,c_lo,y_hi_,y_lo_);  \n")
    file_w.write(" \n")
    file_w.write("				 } \n")
    file_w.write("			       } \n")
    if T_work:
        file_w.write("			       y_hi[i] = y_hi_;  \n")
    else:
        file_w.write("			       y_hi[i] = y_hi_;  \n")
        file_w.write("			       y_lo[i] = y_lo_;  \n")
    file_w.write("			     } \n")
    file_w.write(" \n")
    file_w.write("			     // 2. solve Uz = y \n")
    file_w.write("			     for(PARCEL_INT i = itail-1;i>=(ihead);i--){ \n")
    file_w.write("			       double z_hi_ = y_hi[i];  \n")
    if T_work:
        file_w.write("			       double z_lo_ = 0.0;  \n")
    else:
        file_w.write("			       double z_lo_ = y_lo[i];  \n")
    file_w.write("			       double DiagInv_hi = 1.0;  \n")
    file_w.write("			       double DiagInv_lo = 0.0; 	       \n")
    file_w.write("			       for(PARCEL_INT j = 1; j < nndU; j++){ \n")
    file_w.write("				 const PARCEL_INT offset = U_offset[j]; \n")
    file_w.write("				 const PARCEL_INT Col = i + offset; \n")
    file_w.write("				 if( Col<itail ){ \n")
    file_w.write("				   double zL_hi_ = y_hi[Col];  \n")
    if T_work:
        file_w.write("				   double zL_lo_ = 0.0;  \n")
    else:
        file_w.write("				   double zL_lo_ = y_lo[Col];  \n")

    file_w.write("				   double U_hi_  = U_hi[i+j*N];  \n")
    if T_matrix:
        file_w.write("				   double U_lo_  = 0.0;  \n")
    else:
        file_w.write("				   double U_lo_  = U_lo[i+j*N];  \n")

    file_w.write("				   double ddmul_var1;  \n")
    file_w.write("				   double ddsub_var1;  \n")
    file_w.write("				   double c_hi = 0.0;  \n")
    file_w.write("				   double c_lo = 0.0;  \n")
    file_w.write("				   ddmul_calc1(zL_hi_,zL_lo_,U_hi_,U_lo_,c_hi,c_lo);  \n")
    file_w.write("				   double d_hi = 0.0;  \n")
    file_w.write("				   double d_lo = 0.0;  \n")
    file_w.write("				   ddsub_calc1(z_hi_,z_lo_,c_hi,c_lo,z_hi_,z_lo_);  \n")
    file_w.write("				 } \n")
    file_w.write("			       } \n")
    file_w.write(" \n")
    file_w.write("			       double a_hi = 1.0;  \n")
    file_w.write("			       double a_lo = 0.0;  \n")
    file_w.write("			       double U_hi_ = U_hi[i+0*N];  \n")
    if T_matrix:
        file_w.write("			       double U_lo_ = 0.0;  \n")
    else:
        file_w.write("			       double U_lo_ = U_lo[i+0*N];  \n")

    file_w.write("			       double dddiv_var1;  \n")
    file_w.write("			       dddiv_calc1(a_hi,a_lo,U_hi_,U_lo_,DiagInv_hi,DiagInv_lo);  \n")
    file_w.write(" \n")
    file_w.write("			       double ddmul_var1; \n")
    file_w.write("			       ddmul_calc1(z_hi_,z_lo_,DiagInv_hi,DiagInv_lo,z_hi_,z_lo_);  \n")
    if T_work:
        file_w.write("			       y_hi[i] = z_hi_;  \n")
    else:
        file_w.write("			       y_hi[i] = z_hi_;  \n")
        file_w.write("			       y_lo[i] = z_lo_;  \n")

    if T_vec:
        file_w.write("			       z_hi[i] = z_hi_;  \n")
    else:
        file_w.write("			       z_hi[i] = z_hi_;  \n")
        file_w.write("			       z_lo[i] = z_lo_;  \n")

    file_w.write("			     } \n")
    file_w.write("#endif \n")
    file_w.write("			   } \n")
    file_w.write("	   \n")
    file_w.write("			 }); \n")
    file_w.write(" \n")
    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("		     }); \n")
    file_w.write("#else \n")
    file_w.write(" \n")
    file_w.write("#endif \n")
    file_w.write("      q_[0].wait(); \n")
    file_w.write(" \n")
    file_w.write("      if( SCALING_VECTOR == true ){  \n")
    file_w.write("	const size_t N_ = N ; \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("	q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("	    for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	      const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	      for(size_t i=iii;i<imax;i++){ \n")
    file_w.write("#else \n")
    file_w.write("		q_[0].parallel_for(sycl::range<1>{N_},[=](sycl::id<1> i){ \n")
    file_w.write("#endif \n")

    if T_vec:
        file_w.write("		    z_hi[i] = z_hi[i] * rnorm; \n")
    else:
        file_w.write("		double rnorm_hi = rnorm;\n")
        file_w.write("		double rnorm_lo = 0.0;\n")
        file_w.write("	        double ddmul_var1;\n")
        file_w.write("	        ddmul_calc1(rnorm_hi,rnorm_lo,z_hi[i],z_lo[i],z_hi[i],z_lo[i]);\n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("		  } \n")
    file_w.write("		  } \n")
    file_w.write("	      }); \n")
    file_w.write("#else \n")
    file_w.write("	  }); \n")
    file_w.write("#endif \n")
    file_w.write("	q_[0].wait(); \n")
    file_w.write(" \n")
    file_w.write("      } \n")
    file_w.write(" \n")
    file_w.write("      return ; \n")
    file_w.write("    } \n")

    return
    
ofile="parcel_precon_ddblockjacobi_dia_py.hpp"
file_w = open(ofile,'w') 

TF = [True,False]
for T_matrix in TF:
    for T_vec in TF:
        for T_work in TF:
            make_template(file_w,T_matrix,T_vec,T_work)
            make_arg(file_w,T_matrix,T_vec,T_work)
            make_main(file_w,T_matrix,T_vec,T_work)

file_w.write('\n')
file_w.close()
