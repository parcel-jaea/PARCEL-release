
def make_template(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY):
    temp="template<" 
    if TdataA: 
        temp=temp+"class TdataA,"
    else:
        temp=temp+"class TdataA=bool,"
    if TdataB: 
        temp=temp+"class TdataB,"
    else:
        temp=temp+"class TdataB=bool,"
    if TdataMat: 
        temp=temp+"class TdataMat,"
    else:
        temp=temp+"class TdataMat=bool,"
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

def make_arg(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY):

    file_w.write("void ddgemv( \n")
    file_w.write("	  PARCEL_INT N, \n")
    file_w.write("	  PARCEL_INT M, \n")
    if TdataA: 
        file_w.write("	  TdataA *alpha,\n")
    else:
        file_w.write("	  double *alpha_hi,double *alpha_lo, \n")
    if TdataB: 
        file_w.write("	  TdataB *beta, \n")
    else:
        file_w.write("	  double *beta_hi,double *beta_lo, \n")
    if TdataMat: 
        file_w.write("	  TdataMat *A, \n")
    else:
        file_w.write("	  double *A_hi, double *A_lo, \n")
    file_w.write("	  PARCEL_INT ldA, \n")
    if TdataX: 
        file_w.write("	  TdataX *x, \n")
    else:
        file_w.write("	  double *x_hi,double *x_lo, \n")
    file_w.write("	  PARCEL_INT ldx, \n")
    if TdataY: 
        file_w.write("	  TdataY *y, \n")
    else:
        file_w.write("	  double *y_hi,double *y_lo, \n")
    file_w.write("	  PARCEL_INT ldy){       \n")

    return

def make_main(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY):

    file_w.write("  size_t MAX   = N; \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write(" \n")
    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  const size_t nblock      = 12; \n")
    file_w.write("  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)); \n")
    file_w.write("#else \n")
    file_w.write("  const size_t nblock      = N; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#endif \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock); \n")
    file_w.write("	for(size_t i=iii;i<imax;i++){ \n")
    file_w.write("#else \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ \n")
    file_w.write("#endif \n")

    if TdataA: 
        file_w.write("      double alpha_hi_ = alpha[0]; \n")
        file_w.write("      double alpha_lo_ = 0.0; \n")
    else:
        file_w.write("      double alpha_hi_ = alpha_hi[0]; \n")
        file_w.write("      double alpha_lo_ = alpha_lo[0]; \n")

    if TdataB: 
        file_w.write("      double beta_hi_  = beta[0]; \n")
        file_w.write("      double beta_lo_  = 0.0; \n")
    else:
        file_w.write("      double beta_hi_  = beta_hi[0]; \n")
        file_w.write("      double beta_lo_  = beta_lo[0]; \n")

    file_w.write("      double Ax_hi_ = 0.0; \n")
    file_w.write("      double Ax_lo_ = 0.0; \n")

    file_w.write("      double ddmul_var1; \n")
    file_w.write("      double ddadd_var1; \n")
    file_w.write("      for(PARCEL_INT j=0;j<M;j++){ \n")

    if TdataX: 
        file_w.write("	double x_hi_ =  x[j]; \n")
        file_w.write("	double x_lo_ =  0.0; \n")
    else:
        file_w.write("	double x_hi_ =  x_hi[j]; \n")
        file_w.write("	double x_lo_ =  x_lo[j]; \n")

    if TdataMat: 
        file_w.write("	double A_hi_ =  A[i+j*ldA]; \n")
        file_w.write("	double A_lo_ =  0.0; \n")
    else:
        file_w.write("	double A_hi_ =  A_hi[i+j*ldA]; \n")
        file_w.write("	double A_lo_ =  A_lo[i+j*ldA]; \n")

    file_w.write("	// Ax_ = Ax_ +  A_*x_; \n")
    file_w.write("	double tmp_hi = 0.0; \n")
    file_w.write("	double tmp_lo = 0.0; \n")
    file_w.write("	ddmul_calc1(A_hi_,A_lo_,x_hi_,x_lo_,tmp_hi,tmp_lo); \n")
    file_w.write("	ddadd_calc1(tmp_hi,tmp_lo,Ax_hi_,Ax_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write("      } \n")

    if TdataY: 
        file_w.write("      double y_hi_ =  y[i]; \n")
        file_w.write("      double y_lo_ =  0.0; \n")
    else:
        file_w.write("      double y_hi_ =  y_hi[i]; \n")
        file_w.write("      double y_lo_ =  y_lo[i]; \n")

    file_w.write("      //      y_  = beta_*y_; \n")
    file_w.write("      ddmul_calc1(y_hi_,y_lo_,beta_hi_,beta_lo_,y_hi_,y_lo_); \n")
    file_w.write("      //      Ax_ = alpha_*Ax_; \n")
    file_w.write("      ddmul_calc1(Ax_hi_,Ax_lo_,alpha_hi_,alpha_lo_,Ax_hi_,Ax_lo_); \n")
    file_w.write("      //      y_   = y_ + Ax_; \n")
    file_w.write("      ddadd_calc1(y_hi_,y_lo_,Ax_hi_,Ax_lo_,y_hi_,y_lo_); \n")
    file_w.write(" \n")
    if TdataY: 
        file_w.write("      y[i] = y_hi_ + y_lo_; \n")
    else:
        file_w.write("      y_hi[i] = y_hi_; \n")
        file_w.write("      y_lo[i] = y_lo_; \n")

    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("     } \n")
    file_w.write("     } \n")
    file_w.write("#else \n")
    file_w.write(" \n")
    file_w.write("#endif \n")
    file_w.write("    }); \n")
    file_w.write(" \n")
    file_w.write("} \n")
    
    file_w.write("\n")

    return
    
ofile="mixed_precision_ddgemv_py.hpp"
file_w = open(ofile,'w') 

TF = [True,False]
for TdataA in TF:
    for TdataB in TF:
        for TdataMat in TF:
            for TdataX in TF:
                for TdataY in TF:
                     make_template(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY)
                     make_arg(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY)
                     make_main(file_w,TdataA,TdataB,TdataMat,TdataX,TdataY)

file_w.write('\n')
file_w.close()
