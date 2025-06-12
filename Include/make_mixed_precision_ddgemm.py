
def make_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):
    temp="template<" 
    if TdataAlpha: 
        temp=temp+"class TdataAlpha,"
    else:
        temp=temp+"class TdataAlpha=bool,"
    if TdataBeta: 
        temp=temp+"class TdataBeta,"
    else:
        temp=temp+"class TdataBeta=bool,"
    if TdataA: 
        temp=temp+"class TdataA,"
    else:
        temp=temp+"class TdataA=bool,"
    if TdataB: 
        temp=temp+"class TdataB,"
    else:
        temp=temp+"class TdataB=bool,"
    if TdataC: 
        temp=temp+"class TdataC,"
    else:
        temp=temp+"class TdataC=bool,"
        
    temp=temp+">"
    temp=temp.replace(",>",">\n")
    temp=temp.replace("template<>","\n")
    file_w.write(temp)
    return

def make_arg(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):

    file_w.write("void ddgemm( \n")
    file_w.write("	  PARCEL_INT m, \n")
    file_w.write("	  PARCEL_INT n, \n")
    file_w.write("	  PARCEL_INT k, \n")
    if TdataAlpha: 
        file_w.write("	  TdataAlpha *alpha,\n")
    else:
        file_w.write("	  double *alpha_hi,double *alpha_lo, \n")
    if TdataBeta: 
        file_w.write("	  TdataBeta *beta,\n")
    else:
        file_w.write("	  double *beta_hi,double *beta_lo, \n")
    if TdataA: 
        file_w.write("	  TdataA *A,\n")
    else:
        file_w.write("	  double *A_hi,double *A_lo, \n")
    file_w.write("	  PARCEL_INT ldA, \n")
    if TdataB: 
        file_w.write("	  TdataB *B,\n")
    else:
        file_w.write("	  double *B_hi,double *B_lo, \n")
    file_w.write("	  PARCEL_INT ldB, \n")
    if TdataC: 
        file_w.write("	  TdataC *C,\n")
    else:
        file_w.write("	  double *C_hi,double *C_lo, \n")
    file_w.write("	  PARCEL_INT ldC){ \n")

    return

def make_main(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):
    file_w.write("  size_t MAX   = m*n; \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write(" \n")
    file_w.write("#if _A64FX_HIPSYCL_ \n")
    file_w.write("  const size_t nblock      = 12; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#else \n")
    file_w.write("  const size_t nblock      = m*n; \n")
    file_w.write("  const size_t nlocalblock = 1; \n")
    file_w.write("#endif \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){ \n")
    file_w.write("      for(size_t iii=ii*nlocalblock;iii<MAX;iii=iii+nblock*nlocalblock){ \n")
    file_w.write("	const size_t imax = (iii+nlocalblock) > MAX ? MAX : (iii+nlocalblock); \n")
    file_w.write("	for(size_t i=iii;i<imax;i++){ \n")
    file_w.write("#else \n")
    file_w.write(" \n")
    file_w.write("  q_[0].parallel_for(sycl::range<1>{MAX},[=](sycl::id<1> i){ \n")
    file_w.write(" \n")
    file_w.write("#endif \n")
    file_w.write("      const PARCEL_INT i_c = i%m; \n")
    file_w.write("      const PARCEL_INT j_c = i/m; \n")
    file_w.write(" \n")
    if TdataAlpha: 
        file_w.write("      const double alpha_hi_ = alpha[0]; \n")
        file_w.write("      const double alpha_lo_ = 0.0; \n")
    else:
        file_w.write("      const double alpha_hi_ = alpha_hi[0]; \n")
        file_w.write("      const double alpha_lo_ = alpha_lo[0]; \n")
    if TdataBeta: 
        file_w.write("      const double beta_hi_  = beta[0]; \n")
        file_w.write("      const double beta_lo_  = 0.0; \n")
    else:        
        file_w.write("      const double beta_hi_  = beta_hi[0]; \n")
        file_w.write("      const double beta_lo_  = beta_lo[0]; \n") 
    file_w.write(" \n")
    file_w.write("      double AB_hi_ = 0.0; \n")
    file_w.write("      double AB_lo_ = 0.0; \n")
    file_w.write("      double ddmul_var1; \n")
    file_w.write("      double ddadd_var1; \n")
    file_w.write("      for(PARCEL_INT k_=0;k_<k;k_++){ \n")
    file_w.write("	const PARCEL_INT i_a = i_c; \n")
    file_w.write("	const PARCEL_INT j_a = k_; \n")
    file_w.write("	const PARCEL_INT i_b = k_; \n")
    file_w.write("	const PARCEL_INT j_b = j_c; \n")

    if TdataA: 
        file_w.write("	const double A_hi_ = A[i_a + j_a*ldA]; \n")
        file_w.write("	const double A_lo_ = 0.0; \n")
    else:
        file_w.write("	const double A_hi_ = A_hi[i_a + j_a*ldA]; \n")
        file_w.write("	const double A_lo_ = A_lo[i_a + j_a*ldA]; \n") 
       
    if TdataB: 
        file_w.write("	const double B_hi_ = B[i_b + j_b*ldB]; \n")
        file_w.write("	const double B_lo_ = 0.0; \n")
    else: 
        file_w.write("	const double B_hi_ = B_hi[i_b + j_b*ldB]; \n")
        file_w.write("	const double B_lo_ = B_lo[i_b + j_b*ldB]; \n")
       
    file_w.write("	double tmp_hi = 0.0; \n")
    file_w.write("	double tmp_lo = 0.0; \n")
    file_w.write("	// AB_ = AB_ + A_ * B_; \n")
    file_w.write("	ddmul_calc1(A_hi_,A_lo_,B_hi_,B_lo_,tmp_hi,tmp_lo); \n")
    file_w.write("	ddadd_calc1(tmp_hi,tmp_lo,AB_hi_,AB_lo_,AB_hi_,AB_lo_); \n")
    file_w.write("      } \n")
    if TdataC: 
        file_w.write("      double C_hi_ = C[i_c + j_c*ldC]; \n")
        file_w.write("      double C_lo_ = 0.0; \n")
    else:        
        file_w.write("      double C_hi_ = C_hi[i_c + j_c*ldC]; \n")
        file_w.write("      double C_lo_ = C_lo[i_c + j_c*ldC]; \n")
    file_w.write(" \n")
    file_w.write("      double alphaAB_hi_ = 0.0; \n")
    file_w.write("      double alphaAB_lo_ = 0.0; \n")
    file_w.write("      ddmul_calc1(alpha_hi_,alpha_lo_,AB_hi_,AB_lo_,alphaAB_hi_,alphaAB_lo_); \n")
    file_w.write(" \n")
    file_w.write("      double betaC_hi_ = 0.0; \n")
    file_w.write("      double betaC_lo_ = 0.0; \n")
    file_w.write("      ddmul_calc1(beta_hi_,beta_lo_,C_hi_,C_lo_,betaC_hi_,betaC_lo_); \n")
    file_w.write(" \n")
    file_w.write(" \n")
    file_w.write("      ddadd_calc1(alphaAB_hi_,alphaAB_lo_,betaC_hi_,betaC_lo_,C_hi_,C_lo_); \n")

    if TdataC: 
        file_w.write("      C[i_c + j_c*ldC] = C_hi_ + C_lo_; \n")
    else:
        file_w.write("      C_hi[i_c + j_c*ldC] =C_hi_; \n")
        file_w.write("      C_lo[i_c + j_c*ldC] =C_lo_; \n")
        
    file_w.write(" \n")
    file_w.write("#if _BLOCKING_LOOP_ \n")
    file_w.write("     } \n")
    file_w.write("     } \n")
    file_w.write("#else \n")
    file_w.write(" \n")
    file_w.write("#endif \n")
    file_w.write(" \n")
    file_w.write("});\n")
    file_w.write("} \n")

    return
    
ofile="mixed_precision_ddgemm_py.hpp"
file_w = open(ofile,'w') 

TF = [True,False]
for TdataAlpha in TF:
    for TdataBeta in TF:
        for TdataA in TF:
            for TdataB in TF:
                for TdataC in TF:
                     make_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
                     make_arg(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
                     make_main(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)

file_w.write('\n')
file_w.close()
