def make_template(file_w,Tvec,Tmat,Twork):
    temp="template<" 
    if Tvec: 
        temp=temp+"class T_vec,"
    if Tmat: 
        temp=temp+"class T_mat,"
    if Twork: 
        temp=temp+"class T_work_vec,"
    temp=temp+">"
    temp=temp.replace(",>",">\n")
    temp=temp.replace("template<>","\n")
    file_w.write(temp)
    return

def make_arg(file_w,Tvec,Tmat,Twork):

    file_w.write("void ddjacobi_method_crs_main( \n")
    file_w.write(" 			    sycl::queue *q_ , \n")
    file_w.write("			    PARCEL_INT N, \n")
    file_w.write("			    PARCEL_INT iterMax, \n")
    if Tvec:
        file_w.write("			    T_vec *x, // out   \n")
        file_w.write("			    T_vec *b, // in \n")
    else:
        file_w.write("			    double *x_hi, // out   \n")
        file_w.write("			    double *x_lo, // out   \n")
        file_w.write("			    double *b_hi, // in \n")
        file_w.write("			    double *b_lo, // in \n")
    if Tmat:
        file_w.write("			    T_mat *A, // (D^-1)(L+U) \n")
    else:
        file_w.write("			    double *A_hi, // (D^-1)(L+U) \n")
        file_w.write("			    double *A_lo, // (D^-1)(L+U) \n")
    file_w.write("			    PARCEL_INT *Rowptr, \n")
    file_w.write("			    PARCEL_INT *crsCol, \n")
    if Tmat:
        file_w.write("			    T_mat *Dinv, // (D^-1) \n")
    else:
        file_w.write("			    double *Dinv_hi, // (D^-1) \n")
        file_w.write("			    double *Dinv_lo, // (D^-1) \n")
    if Twork:
        file_w.write("			    T_work_vec *precon_x1_,  \n")
        file_w.write("			    T_work_vec *precon_x2_,  \n")
        file_w.write("			    T_work_vec *precon_b){ \n")
    else:
        file_w.write("			    double *precon_x1_hi_,  \n")
        file_w.write("			    double *precon_x1_lo_,  \n")
        file_w.write("			    double *precon_x2_hi_,  \n")
        file_w.write("			    double *precon_x2_lo_,  \n")
        file_w.write("			    double *precon_b_hi, \n")
        file_w.write("			    double *precon_b_lo ){ \n")

    return

def make_main(file_w,Tvec,Tmat,Twork):

    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  const size_t nblock      = 12; \n")
    file_w.write("  const size_t nlocalblock = 8192; \n")
    file_w.write("#else \n")
    file_w.write("  const size_t nblock      = N; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#endif \n")

    file_w.write("  q_[0].wait();\n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){\n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){\n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);\n")
    file_w.write("	for(size_t i=iii;i<imax;i++){\n")
    if Tmat:
        file_w.write("	  const double Dinv_hi_ = Dinv[i];\n")
        file_w.write("	  const double Dinv_lo_ = 0.0;\n")
    else:
        file_w.write("	  const double Dinv_hi_ = Dinv_hi[i];\n")
        file_w.write("	  const double Dinv_lo_ = Dinv_lo[i];\n")
    if Tvec:
        file_w.write("	  const double b_hi_    = b[i];\n")
        file_w.write("	  const double b_lo_    = 0.0;\n")
    else:
        file_w.write("	  const double b_hi_    = b_hi[i];\n")
        file_w.write("	  const double b_lo_    = b_lo[i];\n")
    file_w.write("	  double ddmul_var1;\n")
    file_w.write("	  double precon_b_hi_ = 0.0;\n")
    file_w.write("	  double precon_b_lo_ = 0.0;\n")
    file_w.write("	  ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_);\n")
    if Twork:
        file_w.write("	  precon_b[i] = precon_b_hi_;\n")
    else:
        file_w.write("	  precon_b_hi[i] = precon_b_hi_;\n")
        file_w.write("	  precon_b_lo[i] = precon_b_lo_;\n")
    file_w.write("	}\n")
    file_w.write("      }\n")
    file_w.write("    });\n")
    file_w.write("  q_[0].wait();\n")
    file_w.write("\n")

    if Twork:
        file_w.write("  T_work_vec *x_now  = precon_x1_;\n")
        file_w.write("  T_work_vec *x_next = precon_x2_;\n")
    else:
        file_w.write("  double *x_now_hi  = precon_x1_hi_;\n")
        file_w.write("  double *x_now_lo  = precon_x1_lo_;\n")
        file_w.write("  double *x_next_hi = precon_x2_hi_;\n")
        file_w.write("  double *x_next_lo = precon_x2_lo_;\n")

    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){\n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){\n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);\n")
    file_w.write("	for(size_t i=iii;i<imax;i++){\n")
    if Twork:
        file_w.write("	  double x_i_hi_ = precon_b[i];\n")
        file_w.write("	  double x_i_lo_ = 0.0;\n")
    else:
        file_w.write("	  double x_i_hi_ = precon_b_hi[i];\n")
        file_w.write("	  double x_i_lo_ = precon_b_lo[i];\n")
    file_w.write("	  for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){\n")
    file_w.write("	    const PARCEL_INT col = crsCol[j];\n")
    if Tmat:
        file_w.write("	    const double A_hi_  = A[j];\n")
        file_w.write("	    const double A_lo_  = 0.0;\n")
    else:
        file_w.write("	    const double A_hi_  = A_hi[j];\n")
        file_w.write("	    const double A_lo_  = A_lo[j];\n")
    if Twork:
        file_w.write("	    const double x_hi_  = precon_b[col];\n")
        file_w.write("	    const double x_lo_  = 0.0;\n")
    else:
        file_w.write("	    const double x_hi_  = precon_b_hi[col];\n")
        file_w.write("	    const double x_lo_  = precon_b_lo[col];\n")
    file_w.write("	    double Ax_hi_ = 0.0;\n")
    file_w.write("	    double Ax_lo_ = 0.0;\n")
    file_w.write("	    double ddmul_var1;\n")
    file_w.write("	    double ddsub_var1;\n")
    file_w.write("	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);\n")
    file_w.write("	    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);\n")
    file_w.write("	  }\n")
    file_w.write("	  if( iterMax==1 ){\n")
    if Tvec:
        file_w.write("	    x[i] = x_i_hi_;\n")
    else:
        file_w.write("	    x_hi[i] = x_i_hi_;\n")
        file_w.write("	    x_lo[i] = x_i_lo_;\n")
    file_w.write("	  }else{\n")
    if Twork:
        file_w.write("	    x_next[i] = x_i_hi_;\n")
    else:
        file_w.write("	    x_next_hi[i] = x_i_hi_;\n")
        file_w.write("	    x_next_lo[i] = x_i_lo_;\n")
    file_w.write("	  }\n")
    file_w.write("	}\n")
    file_w.write("      }\n")
    file_w.write("    });\n")
    file_w.write("  q_[0].wait();\n")
    file_w.write("\n")
    file_w.write("  for(PARCEL_INT iter = 1;iter<iterMax;iter++){\n")
    if Twork:
        file_w.write("    swap_ptr<T_work_vec>(&x_now,&x_next);\n")
    else:
        file_w.write("    swap_ptr<double>(&x_now_hi,&x_next_hi);\n")
        file_w.write("    swap_ptr<double>(&x_now_lo,&x_next_lo);\n")

    file_w.write("    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){\n")
    file_w.write("	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){\n")
    file_w.write("	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);\n")
    file_w.write("	  for(size_t i=iii;i<imax;i++){\n")
    if Twork:
        file_w.write("	    double x_i_hi_ = precon_b[i];\n")
        file_w.write("	    double x_i_lo_ = 0.0;\n")
    else:
        file_w.write("	    double x_i_hi_ = precon_b_hi[i];\n")
        file_w.write("	    double x_i_lo_ = precon_b_lo[i];\n")
    file_w.write("	    for(PARCEL_INT j = Rowptr[i]; j < Rowptr[i+1]; j++){\n")
    file_w.write("	      const PARCEL_INT col = crsCol[j];\n")
    if Tmat:
        file_w.write("	      const double A_hi_  = A[j];\n")
        file_w.write("	      const double A_lo_  = 0.0;\n")
    else:
        file_w.write("	      const double A_hi_  = A_hi[j];\n")
        file_w.write("	      const double A_lo_  = A_lo[j];\n")
    if Twork:
        file_w.write("	      const double x_hi_  = x_now[col];\n")
        file_w.write("	      const double x_lo_  = 0.0;\n")
    else:
        file_w.write("	      const double x_hi_  = x_now_hi[col];\n")
        file_w.write("	      const double x_lo_  = x_now_lo[col];\n")
    file_w.write("	      double Ax_hi_ = 0.0;\n")
    file_w.write("	      double Ax_lo_ = 0.0;\n")
    file_w.write("	      double ddmul_var1;\n")
    file_w.write("	      double ddsub_var1;\n")
    file_w.write("	      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_);\n")
    file_w.write("	      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_);\n")
    file_w.write("	    }\n")
    file_w.write("	    if( ( (iter+1)==iterMax) ){\n") 
    if Tvec:
        file_w.write("	      x[i] = x_i_hi_;\n")
    else:
        file_w.write("	      x_hi[i] = x_i_hi_;\n")
        file_w.write("	      x_lo[i] = x_i_lo_;\n")
    file_w.write("	    }else{\n")
    if Twork:
        file_w.write("	      x_next[i] = x_i_hi_;\n")
    else:
        file_w.write("	      x_next_hi[i] = x_i_hi_;\n")
        file_w.write("	      x_next_lo[i] = x_i_lo_;\n")
    file_w.write("	    }\n")
    file_w.write("	  }\n")
    file_w.write("	}\n")
    file_w.write("      });\n")
    file_w.write("    q_[0].wait();\n")
    file_w.write("  }\n")
    file_w.write("\n")
    file_w.write("  return ;\n")
    file_w.write("\n")
    file_w.write("}\n")
    file_w.write("\n")

    return
    
ofile="parcel_precon_ddjacobi_crs_py.hpp"
file_w = open(ofile,'w') 

TF = [True,False]
for Tvec in TF:
    for Tmat in TF:
        for Twork in TF:
            make_template(file_w,Tvec,Tmat,Twork)
            make_arg(file_w,Tvec,Tmat,Twork)
            make_main(file_w,Tvec,Tmat,Twork)


file_w.write('\n')
file_w.close()
