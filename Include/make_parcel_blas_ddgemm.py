def make_ddgemm_main_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):
    if TdataAlpha:
	file_w.write("bool, \n")
    else:
	file_w.write("typename std::result_of< parcel::parcel_type_to_baseType( Talpha ) >::type, \n")
    if TdataBeta:
	file_w.write("bool, \n")
    else:
	file_w.write("typename std::result_of< parcel::parcel_type_to_baseType( Tbeta ) >::type, \n")
    if TdataA:
	file_w.write("bool, \n")
    else:
	file_w.write("typename std::result_of< parcel::parcel_type_to_baseType( TmatA ) >::type, \n")
    if TdataB:
	file_w.write("bool, \n")
    else:
	file_w.write("typename std::result_of< parcel::parcel_type_to_baseType( TmatB ) >::type, \n")
    if TdataC:
	file_w.write("bool \n")
    else:
	file_w.write("typename std::result_of< parcel::parcel_type_to_baseType( TmatC ) >::type \n")
    
def make_ddgemm_main(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):
    
    file_w.write("  if( TransA == \"T\" ) { \n")
    file_w.write("    if( sym == \"S\" ) { \n")
    
    file_w.write("	blas_.ddgemmTNsym< \n")
    make_ddgemm_main_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
    file_w.write(">(m,n,k, \n")
    if TdataAlpha: 
	 file_w.write("alpha_hi,alpha_lo, \n")
    else:
	 file_w.write("alpha, \n")
    if TdataBeta: 
	 file_w.write("beta_hi,beta_lo, \n")
    else:
	 file_w.write("beta, \n")
    if TdataA: 
	 file_w.write("A_hi,A_lo,ldA, \n")
    else:
	 file_w.write("A,ldA, \n")
    if TdataB: 
	 file_w.write("B_hi,B_lo,ldB, \n")
    else:
	 file_w.write("B,ldB, \n")
    if TdataC: 
	 file_w.write("C_hi,C_lo,ldC \n")
    else:
	 file_w.write("C,ldC \n")
    file_w.write("); \n")
    file_w.write("		}else{ \n")
    file_w.write("		  blas_.ddgemmTN< \n")
    make_ddgemm_main_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
    file_w.write("	 >(m,n,k, \n")
    if TdataAlpha: 
	 file_w.write("alpha_hi,alpha_lo, \n")
    else:
	 file_w.write("alpha, \n")
    if TdataBeta: 
	 file_w.write("beta_hi,beta_lo, \n")
    else:
	 file_w.write("beta, \n")
    if TdataA: 
	 file_w.write("A_hi,A_lo,ldA, \n")
    else:
	 file_w.write("A,ldA, \n")
    if TdataB: 
	 file_w.write("B_hi,B_lo,ldB, \n")
    else:
	 file_w.write("B,ldB, \n")
    if TdataC: 
	 file_w.write("C_hi,C_lo,ldC \n")
    else:
	 file_w.write("C,ldC \n")
    file_w.write("); \n")
    file_w.write("		} \n")
    file_w.write("  }else{ \n")
    file_w.write("      blas_.ddgemm< \n")
    make_ddgemm_main_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
    file_w.write("	 >(m,n,k, \n")
    if TdataAlpha: 
	 file_w.write("alpha_hi,alpha_lo, \n")
    else:
	 file_w.write("alpha, \n")
    if TdataBeta: 
	 file_w.write("beta_hi,beta_lo, \n")
    else:
	 file_w.write("beta, \n")
    if TdataA: 
	 file_w.write("A_hi,A_lo,ldA, \n")
    else:
	 file_w.write("A,ldA, \n")
    if TdataB: 
	 file_w.write("B_hi,B_lo,ldB, \n")
    else:
	 file_w.write("B,ldB, \n")
    if TdataC: 
	 file_w.write("C_hi,C_lo,ldC \n")
    else:
	 file_w.write("C,ldC \n")
    file_w.write("); }\n")
    
    return
    
def make_template(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC):
    
    file_w.write("    if( \n")
    if TdataAlpha:
        file_w.write(" (ddtype_name == Talpha_name) && (ddtype_name == Tbeta_name) && \n")
    else:
        file_w.write(" (ddtype_name != Talpha_name) && (ddtype_name != Tbeta_name) && \n")

    if TdataA:
        file_w.write(" (ddtype_name == TmatA_name) && \n")
    else:
        file_w.write(" (ddtype_name != TmatA_name) && \n")

    if TdataB:
        file_w.write(" (ddtype_name == TmatB_name) && \n")
    else:
        file_w.write(" (ddtype_name != TmatB_name) && \n")

    if TdataC:
        file_w.write(" (ddtype_name == TmatC_name) \n")
    else:
        file_w.write(" (ddtype_name != TmatC_name) \n")
    file_w.write(" ){ \n")

    if TdataAlpha:
        file_w.write("	double *alpha_hi = (double *)alpha_.data_hi(); \n")
        file_w.write("	double *alpha_lo = (double *)alpha_.data_lo(); \n")
        file_w.write("	double *beta_hi = (double *)beta_.data_hi(); \n")
        file_w.write("	double *beta_lo = (double *)beta_.data_lo(); \n")
    else:
        file_w.write("	auto alpha = alpha_.data(); \n")
        file_w.write("	auto beta = beta_.data(); \n")

    if TdataA:
        file_w.write("  double *A_hi = (double *)A_.data_hi(); \n")
        file_w.write("  double *A_lo = (double *)A_.data_lo(); \n")
    else:
        file_w.write("  auto A = A_.data(); \n")

    if TdataB:
        file_w.write("  double *B_hi = (double *)B_.data_hi(); \n")
        file_w.write("  double *B_lo = (double *)B_.data_lo(); \n")
    else:
        file_w.write("  auto B = B_.data(); \n")

    if TdataC:
        file_w.write("  double *C_hi = (double *)C_.data_hi(); \n")
        file_w.write("  double *C_lo = (double *)C_.data_lo(); \n")
    else:
        file_w.write("  auto C = C_.data(); \n")
        
    make_ddgemm_main(file_w,TdataAlpha,TdataBeta,TdataA,TdataB,TdataC)
    file_w.write(" return ;\n }\n")

    return ;

ofile="make_parcel_blas_ddgemm_py.hpp"

file_w = open(ofile,'w') 

TF = [True,False]
TF = [True,False]

for TdataAlphaBeta in TF:
    for TdataA in TF:
        for TdataB in TF:
            for TdataC in TF:
                make_template(file_w,TdataAlphaBeta,TdataAlphaBeta,TdataA,TdataB,TdataC)

file_w.write('\n')
file_w.close()

