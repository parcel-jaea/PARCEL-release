def make_template(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY):
    temp="template<" 
    if TdataA: 
        temp=temp+"class TdataA,"
    else:
        temp=temp+"class TdataA=bool,"

    if TdataB: 
        temp=temp+"class TdataB,"
    else:
        temp=temp+"class TdataB=bool,"

    if TdataDIA: 
        temp=temp+"class TdataDIA,"
    else:
        temp=temp+"class TdataDIA=bool,"

    if TdataX: 
        temp=temp+"class TdataX,"
    else:
        temp=temp+"class TdataX=bool,"

    if TdataY: 
        temp=temp+"class TdataY,"
    else:
        temp=temp+"class TdataY=bool,"

    temp=temp+">"
    temp=temp.replace(",>",">\n")
    temp=temp.replace("template<>","\n")
    file_w.write(temp)
    return

def make_arg(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY):
    file_w.write("void dddiamv( \n")
    file_w.write("	     PARCEL_INT N, \n")
    file_w.write("	     PARCEL_INT NND, \n")
    if TdataA:
        file_w.write("	     TdataA *alpha, \n")
    else:
        file_w.write("	     double *alpha_hi, \n")
        file_w.write("	     double *alpha_lo, \n")
    if TdataDIA:
        file_w.write("	     TdataDIA *A, \n")
    else:
        file_w.write("	     double *A_hi, \n")
        file_w.write("	     double *A_lo, \n")
    file_w.write("	     PARCEL_INT *offset, \n")
    file_w.write("	     PARCEL_INT max_up_offset, \n")
    file_w.write("	     PARCEL_INT max_down_offset, \n")
    if TdataX:
        file_w.write("	     TdataX *x, \n")
    else:
        file_w.write("	     double *x_hi, \n")
        file_w.write("	     double *x_lo, \n")

    file_w.write("	     PARCEL_INT ldx, \n")
    if TdataB:
        file_w.write("	     TdataB *beta, \n")
    else:
        file_w.write("	     double *beta_hi, \n")
        file_w.write("	     double *beta_lo, \n")

    if TdataY:
        file_w.write("	     TdataY *y, \n")
    else:
        file_w.write("	     double *y_hi, \n")
        file_w.write("	     double *y_lo, \n")

    file_w.write("	     PARCEL_INT ldy \n")
    file_w.write("	     ){ \n")

    return

def make_main(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY):
    file_w.write("  size_t MAX   = N; \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i_){ \n")
    file_w.write("      const PARCEL_INT i = i_; \n")
    file_w.write("      double Ax_hi=0.0; \n")
    file_w.write("      double Ax_lo=0.0; \n")
    file_w.write("      if( ( (i+max_up_offset)<0 ) || ( (i+max_down_offset) >= N ) ){ \n")
    file_w.write("	for(PARCEL_INT j = 0;j<NND;j++){ \n")
    file_w.write("	  if( (0<=(i+offset[j]) ) && ((i+offset[j])<N) ){ \n")

    if TdataDIA:
        file_w.write("	    double A_hi_ = A[i+j*N]; \n")
        file_w.write("	    double A_lo_ = 0.0; \n")
    else:
        file_w.write("	    double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("	    double A_lo_ = A_lo[i+j*N]; \n")

    if TdataX:
        file_w.write("	    double x_hi_ = x[i+offset[j]]; \n")
        file_w.write("	    double x_lo_ = 0.0; \n")
    else:
        file_w.write("	    double x_hi_ = x_hi[i+offset[j]]; \n")
        file_w.write("	    double x_lo_ = x_lo[i+offset[j]]; \n")
    file_w.write("	    double c_hi = 0.0; \n")
    file_w.write("	    double c_lo = 0.0; \n")
    file_w.write("	    double ddadd_var1; \n")
    file_w.write("	    double ddmul_var1; \n")
    file_w.write("	    ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); \n")
    file_w.write("	    ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo); \n")
    file_w.write("	  } \n")
    file_w.write("	} \n")
    file_w.write("      }else{ \n")
    file_w.write("	for(PARCEL_INT j = 0;j<NND;j++){ \n")

    if TdataDIA:
        file_w.write("	  double A_hi_ = A[i+j*N]; \n")
        file_w.write("	  double A_lo_ = 0.0; \n")
    else :
        file_w.write("	  double A_hi_ = A_hi[i+j*N]; \n")
        file_w.write("	  double A_lo_ = A_lo[i+j*N]; \n")

    if TdataX:
        file_w.write("	  double x_hi_ = x[i+offset[j]]; \n")
        file_w.write("	  double x_lo_ = 0.0; \n")
    else :
        file_w.write("	  double x_hi_ = x_hi[i+offset[j]]; \n")
        file_w.write("	  double x_lo_ = x_lo[i+offset[j]]; \n")

    file_w.write("	  double c_hi = 0.0; \n")
    file_w.write("	  double c_lo = 0.0; \n")
    file_w.write("	  double ddadd_var1; \n")
    file_w.write("	  double ddmul_var1; \n")
    file_w.write("	  ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,c_hi,c_lo); \n")
    file_w.write("	  ddadd_calc1(Ax_hi,Ax_lo,c_hi,c_lo,Ax_hi,Ax_lo);	   \n")
    file_w.write("	} \n")
    file_w.write("      } \n")

    if TdataA:
        file_w.write("      double alpha_hi_ = alpha[0]; \n")
        file_w.write("      double alpha_lo_ = 0.0; \n")
    else :
        file_w.write("      double alpha_hi_ = alpha_hi[0]; \n")
        file_w.write("      double alpha_lo_ = alpha_lo[0]; \n")

    if TdataB:
        file_w.write("      double beta_hi_  = beta[0]; \n")
        file_w.write("      double beta_lo_  = 0.0; \n")
    else :
        file_w.write("      double beta_hi_  = beta_hi[0]; \n")
        file_w.write("      double beta_lo_  = beta_lo[0]; \n")

    if TdataY:
        file_w.write("      double y_hi_ = y[i]; \n")
        file_w.write("      double y_lo_ = 0.0; \n")
    else :
        file_w.write("      double y_hi_ = y_hi[i]; \n")
        file_w.write("      double y_lo_ = y_lo[i]; \n")

    file_w.write(" \n")
    file_w.write("      double ddadd_var1; \n")
    file_w.write("      double ddmul_var1; \n")
    file_w.write("      double ddmul_var2; \n")
    file_w.write(" \n")
    file_w.write("      double by_hi = 0.0; \n")
    file_w.write("      double by_lo = 0.0; \n")
    file_w.write("      double aAx_hi = 0.0; \n")
    file_w.write("      double aAx_lo = 0.0; \n")
    file_w.write("      ddmul_calc1(alpha_hi_,alpha_lo_,Ax_hi,Ax_lo,aAx_hi,aAx_lo); \n")
    file_w.write("      ddmul_calc2(beta_hi_,beta_lo_,y_hi_,y_lo_,by_hi,by_lo); \n")
    file_w.write(" \n")
    file_w.write("      double yy_hi = 0.0; \n")
    file_w.write("      double yy_lo = 0.0; \n")
    file_w.write("      ddadd_calc1(aAx_hi,aAx_lo,by_hi,by_lo,yy_hi,yy_lo); \n")
    if TdataY:
        file_w.write("      y[i] = yy_hi; \n")
    else:
        file_w.write("      y_hi[i] = yy_hi; \n")
        file_w.write("      y_lo[i] = yy_lo; \n")
    file_w.write(" \n")
    file_w.write("    }); \n")
    file_w.write("  \n")
    file_w.write("  return ; \n")
    file_w.write(" \n")
    file_w.write("} \n")

    return
    
ofile="mixed_precision_dddiamv_py.hpp"
file_w = open(ofile,'w') 

TF = [True,False]
for TdataA in TF:
    for TdataB in TF:
        for TdataDIA in TF:
            for TdataX in TF:
                for TdataY in TF:
                    make_template(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY)
                    make_arg(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY)
                    make_main(file_w,TdataA,TdataB,TdataDIA,TdataX,TdataY)

file_w.write('\n')
file_w.close()
