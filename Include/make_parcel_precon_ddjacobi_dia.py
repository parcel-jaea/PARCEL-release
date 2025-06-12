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

    file_w.write("void ddjacobi_method_dia_main( \n")
    file_w.write(" 			    sycl::queue *q_ , \n")
    file_w.write("			    PARCEL_INT N, \n")
    file_w.write("			    PARCEL_INT NND, \n")
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
    file_w.write("			    PARCEL_INT *offset, \n")
    file_w.write("			    PARCEL_INT max_up_offset, \n")
    file_w.write("			    PARCEL_INT max_down_offset, \n")
    if Tmat:
        file_w.write("			    T_mat *Dinv, // (D^-1) \n")
    else:
        file_w.write("			    double *Dinv_hi, // (D^-1) \n")
        file_w.write("			    double *Dinv_lo, // (D^-1) \n")

    file_w.write("			    double *scaling_Inv,\n")

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

    # file_w.write("std::cout<< ")
    # if Tvec:
    #     file_w.write(" \" T_vec \" << ")        
    # if Tmat:
    #     file_w.write(" \" T_mat \" << ")
    # if Twork:
    #     file_w.write(" \" T_work_vec \" << ")
    # file_w.write("std::endl; \n")

    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  const size_t nblock      = 12; \n")
    file_w.write("  const size_t nlocalblock = 8192; \n")
    file_w.write("#else \n")
    file_w.write("  const size_t nblock      = N; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#endif \n")
    file_w.write("   \n")
    file_w.write("  const bool SCALING_MATRIX = SCALING_MATRIX_; \n")
    file_w.write("  const bool SCALING_VECTOR = SCALING_VECTOR_; \n")
    file_w.write(" \n")
    file_w.write("  q_[0].wait(); \n")
    file_w.write(" \n")
    file_w.write("  // T_work_vec *x_now  = precon_x1_; \n")
    file_w.write("  // T_work_vec *x_next = precon_x2_; \n")
    file_w.write("  // solve Ax = b \n")
    file_w.write("  // x(i+1) = (D^-1) { b - {L+U}x(i) }  \n")
    file_w.write(" \n")
    file_w.write("  //      const double bnorm = 1.0; \n")
    file_w.write("  double bnorm_ = 1.0; \n")
    file_w.write("  if( SCALING_VECTOR == true ){  \n")
    file_w.write("    setQueue(q_); \n")
    if Tvec:
        file_w.write("    if( SCALING_MATRIX == true ){  \n")
        file_w.write("      bnorm_ = max<T_vec,double,double>(N,b,N,scaling_Inv,N); \n")
        file_w.write("    }else{ \n")
        file_w.write("      bnorm_ = max<T_vec,double>(N,b,N); \n")
        file_w.write("    } \n")
    else:
        file_w.write("    if( SCALING_MATRIX == true ){  \n")
        file_w.write("      bnorm_ = max<double,double,double>(N,b_hi,N,scaling_Inv,N); \n")
        file_w.write("    }else{ \n")
        file_w.write("      bnorm_ = max<double,double>(N,b_hi,N); \n")
        file_w.write("    } \n")

    file_w.write("    q_[0].wait(); \n")
    file_w.write("  } \n")
    file_w.write("  const double bnorm_hi_ = bnorm_; \n")
    file_w.write("  const double bnorm_lo_ = 0.0; \n")
    file_w.write(" \n")
    file_w.write("  // double ddmul_var1; ddmul_calc1 \n")
    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("      //	  double bnorm = bnorm_ptr[0]; \n")
    file_w.write("      ///	  double bnorm = 1.0; \n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	for(size_t i=iii;i<imax;i++){ \n")
    file_w.write("#else \n")
    file_w.write("	  const size_t MAX = N; \n")
    file_w.write("	  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ \n")
    file_w.write("#endif \n")
    if Tmat:
        file_w.write("	      const double Dinv_hi_ = Dinv[i]; \n")
        file_w.write("	      const double Dinv_lo_ = 0.0; \n")
    else:
        file_w.write("	      const double Dinv_hi_ = Dinv_hi[i]; \n")
        file_w.write("	      const double Dinv_lo_ = Dinv_lo[i]; \n")
    if Tvec:
        file_w.write("	      double b_hi_ = b[i]; \n")
        file_w.write("	      double b_lo_ = 0.0; \n")
    else:
        file_w.write("	      double b_hi_ = b_hi[i]; \n")
        file_w.write("	      double b_lo_ = b_lo[i]; \n")
    file_w.write("	      // const double b_    = b[i] / bnorm; \n")
    file_w.write("	      double dddiv_var1;  \n")
    file_w.write("	      dddiv_calc1(b_hi_,b_lo_,bnorm_hi_,bnorm_lo_,b_hi_,b_lo_); \n")
    file_w.write(" \n")
    file_w.write("	      if( SCALING_MATRIX == true ){  \n")
    file_w.write("		const double val_hi_ = scaling_Inv[i]; \n")
    file_w.write("		const double val_lo_ = 0.0; \n")
    file_w.write("		double precon_b_hi_ = 0.0; \n")
    file_w.write("		double precon_b_lo_ = 0.0; \n")
    file_w.write("		// precon_b[i] = Dinv_*b_*val; \n")
    file_w.write("		double ddmul_var1;  \n")
    file_w.write("		ddmul_calc1(Dinv_hi_,Dinv_lo_,b_hi_,b_lo_,precon_b_hi_,precon_b_lo_); \n")
    file_w.write("		ddmul_calc1(precon_b_hi_,precon_b_lo_,val_hi_,val_lo_,precon_b_hi_,precon_b_lo_); \n")
    if Twork:
        file_w.write("		precon_b[i] = precon_b_hi_; \n")
    else:
        file_w.write("		precon_b_hi[i] = precon_b_hi_; \n")
        file_w.write("		precon_b_lo[i] = precon_b_lo_; \n")
    file_w.write("	      }else{ \n")
    file_w.write("		// precon_b[i] = Dinv_*b_; \n")
    file_w.write("		double precon_b_hi_ = 0.0; \n")
    file_w.write("		double precon_b_lo_ = 0.0; \n")
    file_w.write("		double ddmul_var1;  \n")
    file_w.write("		ddmul_calc1(b_hi_,b_lo_,Dinv_hi_,Dinv_lo_,precon_b_hi_,precon_b_lo_); \n")
    if Twork:
        file_w.write("		precon_b[i] = precon_b_hi_; \n")
    else:
        file_w.write("		precon_b_hi[i] = precon_b_hi_; \n")
        file_w.write("		precon_b_lo[i] = precon_b_lo_; \n")
    file_w.write("	      } \n")
    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("	    } \n")
    file_w.write("	    } \n")
    file_w.write("	}); \n")
    file_w.write("#else \n")
    file_w.write("    }); \n")
    file_w.write("#endif \n")
    file_w.write("  q_[0].wait(); \n")
    file_w.write(" \n")
    file_w.write("  // x(0) = (D^-1)b \n")
    file_w.write("  // x(1) = (D^-1)b - (D^-1){L+U}(D^-1)b \n")
    file_w.write("  //        ( I - (D^-1){L+U} )(D^-1)b   \n")
    file_w.write(" \n")

    if Twork:
        file_w.write("  T_work_vec *x_now  = precon_x1_; \n")
        file_w.write("  T_work_vec *x_next = precon_x2_; \n")
    else:
        file_w.write("  double *x_now_hi  = precon_x1_hi_; \n")
        file_w.write("  double *x_now_lo  = precon_x1_lo_; \n")
        file_w.write("  double *x_next_hi = precon_x2_hi_; \n")
        file_w.write("  double *x_next_lo = precon_x2_lo_; \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	for(size_t i_=iii;i_<imax;i_++){ \n")
    file_w.write("#else \n")
    file_w.write("	  //	      const size_t MAX = N; \n")
    file_w.write("	  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ \n")
    file_w.write("#endif \n")
    file_w.write("	      PARCEL_INT i = i_; \n")
    file_w.write("	      // T_calc x_i = precon_b[i]; \n")
    if Twork:
        file_w.write("	      double x_i_hi_ = precon_b[i]; \n")
        file_w.write("	      double x_i_lo_ = 0.0; \n")
    else:
        file_w.write("	      double x_i_hi_ = precon_b_hi[i]; \n")
        file_w.write("	      double x_i_lo_ = precon_b_lo[i]; \n")
    file_w.write("	      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ \n")
    file_w.write("		for(PARCEL_INT j = 0; j < NND; j++){ \n")
    file_w.write("		  PARCEL_INT i0 = i + offset[j]; \n")
    file_w.write("		  // T_calc A_ = A[i+j*N]; \n")
    if Tmat:
        file_w.write("		  const double A_hi_ = A[i+j*N]; \n")
        file_w.write("		  const double A_lo_ = 0.0; \n")
    else:
        file_w.write("		  const double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("		  const double A_lo_ = A_lo[i+j*N]; \n")
    file_w.write("		  if( (-1<i0) && (i0<N) ){ \n")
    file_w.write("		    // T_calc x_ = precon_b[i0]; \n")
    if Twork:
        file_w.write("		    double x_hi_ = precon_b[i0]; \n")
        file_w.write("		    double x_lo_ = 0.0; \n")
    else:
        file_w.write("		    double x_hi_ = precon_b_hi[i0]; \n")
        file_w.write("		    double x_lo_ = precon_b_lo[i0]; \n")
    file_w.write("		    //x_i = x_i - A_*x_; \n")
    file_w.write("		    double Ax_hi_ = 0.0; \n")
    file_w.write("		    double Ax_lo_ = 0.0; \n")
    file_w.write("		    double ddmul_var1;  \n")
    file_w.write("		    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write(" \n")
    file_w.write("		    double ddsub_var1;  \n")
    file_w.write("		    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_); \n")
    file_w.write("		  } \n")
    file_w.write("		} \n")
    file_w.write("	      }else{ \n")
    file_w.write("		for(PARCEL_INT j = 0; j < NND; j++){ \n")
    file_w.write("		  PARCEL_INT i0 = i + offset[j]; \n")
    file_w.write("		  // T_calc A_ = A[i+j*N]; \n")
    file_w.write("		  // T_calc x_ = precon_b[i0]; \n")
    file_w.write(" \n")
    if Tmat:
        file_w.write("		  const double A_hi_ = A[i+j*N]; \n")
        file_w.write("		  const double A_lo_ = 0.0; \n")
    else:
        file_w.write("		  const double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("		  const double A_lo_ = A_lo[i+j*N]; \n")
    if Twork:
        file_w.write("		  double x_hi_ = precon_b[i0]; \n")
        file_w.write("		  double x_lo_ = 0.0; \n")
    else:
        file_w.write("		  double x_hi_ = precon_b_hi[i0]; \n")
        file_w.write("		  double x_lo_ = precon_b_lo[i0]; \n")
    file_w.write(" \n")
    file_w.write("		  double Ax_hi_ = 0.0; \n")
    file_w.write("		  double Ax_lo_ = 0.0; \n")
    file_w.write("		   \n")
    file_w.write("		  // x_i = x_i - A_*x_; \n")
    file_w.write("		  double ddmul_var1;  \n")
    file_w.write("		  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write(" \n")
    file_w.write("		  double ddsub_var1;  \n")
    file_w.write("		  ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_); \n")
    file_w.write(" \n")
    file_w.write("		} \n")
    file_w.write("	      } \n")
    file_w.write(" \n")
    file_w.write("	      if( iterMax==1 ){ \n")
    if Tvec:
        file_w.write("		x[i] = x_i_hi_; \n")
    else:
        file_w.write("		x_hi[i] = x_i_hi_; \n")
        file_w.write("		x_lo[i] = x_i_lo_; \n")
    file_w.write("	      }else{ \n")
    if Twork:
        file_w.write("		x_next[i] = x_i_hi_; \n")
    else:
        file_w.write("		x_next_hi[i] = x_i_hi_; \n")
        file_w.write("		x_next_lo[i] = x_i_lo_; \n")
    file_w.write("	      } \n")
    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("	    } \n")
    file_w.write("	    } \n")
    file_w.write("	}); \n")
    file_w.write("#else \n")
    file_w.write("    }); \n")
    file_w.write("#endif \n")
    file_w.write("  q_[0].wait(); \n")
    file_w.write(" \n")
    file_w.write("  // x(i+1) = (D^-1) { b - {L+U}x(i) }  \n")
    file_w.write("  for(PARCEL_INT iter = 1;iter<iterMax;iter++){ \n")
    file_w.write(" \n")
    #   file_w.write("    swap_ptr<T_work_vec>(&x_now,&x_next); \n")
    if Twork:
        file_w.write("    swap_ptr<T_work_vec>(&x_now,&x_next);\n")
    else:
        file_w.write("    swap_ptr<double>(&x_now_hi,&x_next_hi);\n")
        file_w.write("    swap_ptr<double>(&x_now_lo,&x_next_lo);\n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	  for(size_t i_=iii;i_<imax;i_++){ \n")
    file_w.write("#else \n")
    file_w.write("	    //	const size_t MAX = N; \n")
    file_w.write("	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ \n")
    file_w.write("#endif \n")
    file_w.write("		PARCEL_INT i = i_; \n")
    if Twork:
        file_w.write("		double x_i_hi_ = precon_b[i]; \n")
        file_w.write("		double x_i_lo_ = 0.0; \n")
    else:
        file_w.write("		double x_i_hi_ = precon_b_hi[i]; \n")
        file_w.write("		double x_i_lo_ = precon_b_lo[i]; \n")
    file_w.write("		if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ \n")
    file_w.write("		  for(PARCEL_INT j = 0; j < NND; j++){ \n")
    file_w.write("		    PARCEL_INT i0 = i + offset[j]; \n")
    if Tmat:
        file_w.write("		    double A_hi_ = A[i+j*N]; \n")
        file_w.write("		    double A_lo_ = 0.0; \n")
    else:
        file_w.write("		    double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("		    double A_lo_ = A_lo[i+j*N]; \n")
    file_w.write("		    if( (-1<i0) && (i0<N) ){ \n")
    if Twork:
        file_w.write("		      double x_hi_ = x_now[i0]; \n")
        file_w.write("		      double x_lo_ = 0.0; \n")
    else:
        file_w.write("		      double x_hi_ = x_now_hi[i0]; \n")
        file_w.write("		      double x_lo_ = x_now_lo[i0]; \n")
    file_w.write(" \n")
    file_w.write("		      // x_i = x_i - A_*x_; \n")
    file_w.write("		      double Ax_hi_ = 0.0; \n")
    file_w.write("		      double Ax_lo_ = 0.0; \n")
    file_w.write("		      double ddmul_var1;  \n")
    file_w.write("		      ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write("		       \n")
    file_w.write("		      double ddsub_var1;  \n")
    file_w.write("		      ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_); \n")
    file_w.write("		    } \n")
    file_w.write("		  } \n")
    file_w.write("		}else{ \n")
    file_w.write("		  for(PARCEL_INT j = 0; j < NND; j++){ \n")
    file_w.write("		    PARCEL_INT i0 = i + offset[j]; \n")
    if Tmat:
        file_w.write("		    double A_hi_ = A[i+j*N]; \n")
        file_w.write("		    double A_lo_ = 0.0; \n")
    else:
        file_w.write("		    double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("		    double A_lo_ = A_lo[i+j*N]; \n")
    if Twork:
        file_w.write("		    double x_hi_ = x_now[i0]; \n")
        file_w.write("		    double x_lo_ = 0.0; \n")
    else:
        file_w.write("		    double x_hi_ = x_now_hi[i0]; \n")
        file_w.write("		    double x_lo_ = x_now_lo[i0]; \n")
    file_w.write(" \n")
    file_w.write("		    // x_i = x_i - A_*x_; \n")
    file_w.write("		    double Ax_hi_ = 0.0; \n")
    file_w.write("		    double Ax_lo_ = 0.0; \n")
    file_w.write("		    double ddmul_var1;  \n")
    file_w.write("		    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write("		       \n")
    file_w.write("		    double ddsub_var1;  \n")
    file_w.write("		    ddsub_calc1(x_i_hi_,x_i_lo_,Ax_hi_,Ax_lo_,x_i_hi_,x_i_lo_); \n")
    file_w.write(" \n")
    file_w.write("		  } \n")
    file_w.write("		} \n")
    file_w.write(" \n")
    file_w.write("		if( ( (iter+1)==iterMax) ){ \n")
    if Tvec:
        file_w.write("		  x[i] = x_i_hi_; \n")
    else:
        file_w.write("		  x_hi[i] = x_i_hi_; \n")
        file_w.write("		  x_lo[i] = x_i_lo_; \n")
    file_w.write("		}else{ \n")
    if Twork:
        file_w.write("		  x_next[i] = x_i_hi_; \n")
    else:
        file_w.write("		  x_next_hi[i] = x_i_hi_; \n")
        file_w.write("		  x_next_lo[i] = x_i_lo_; \n")
    file_w.write("		} \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("	      } \n")
    file_w.write("	      } \n")
    file_w.write("	  }); \n")
    file_w.write("#else \n")
    file_w.write("      }); \n")
    file_w.write("#endif \n")
    file_w.write("    q_[0].wait(); \n")
    file_w.write("  } \n")
    file_w.write(" \n")
    file_w.write("  if( SCALING_VECTOR == true ){  \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("    q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("	for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	  const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	  for(size_t i=iii;i<imax;i++){ \n")
    file_w.write("#else \n")
    file_w.write("	    q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ \n")
    file_w.write("#endif \n")
    file_w.write("		// x[i] = x[i] * bnorm; \n")
    if Tvec:
        file_w.write("		double x_hi_ = x[i]; \n")
        file_w.write("		double x_lo_ = 0.0; \n")
    else:
        file_w.write("		double x_hi_ = x_hi[i]; \n")
        file_w.write("		double x_lo_ = x_lo[i]; \n")
    file_w.write("		double ddmul_var1;  \n")
    file_w.write("		ddmul_calc1(x_hi_,x_lo_,bnorm_hi_,bnorm_lo_,x_hi_,x_lo_); \n")
    if Tvec:
        file_w.write("		x[i] = x_hi_; \n")
    else:
        file_w.write("		x_hi[i] = x_hi_; \n")
        file_w.write("		x_lo[i] = x_lo_; \n")
    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("	      } \n")
    file_w.write("	      } \n")
    file_w.write("	  }); \n")
    file_w.write("#else \n")
    file_w.write("      }); \n")
    file_w.write("#endif \n")
    file_w.write("    q_[0].wait(); \n")
    file_w.write("  } \n")
    file_w.write(" \n")
    file_w.write("  return ; \n")
    file_w.write(" \n")
    file_w.write("} \n")

    return
    
ofile="parcel_precon_ddjacobi_dia_py.hpp"
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
