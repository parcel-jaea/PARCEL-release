#ifndef PARCEL_QR_HPP_
#define PARCEL_QR_HPP_

#include "parcel_blas.hpp"

#include "parcel_qr_cgs.hpp"
#include "parcel_qr_mgs.hpp"
#include "parcel_qr_choleskyqr.hpp"
// #include "parcel_qr_tsqr.hpp"

namespace parcel{
  namespace qr{
    const PARCEL_INT QR_CGS = 0;
    const PARCEL_INT QR_MGS = 1;
    const PARCEL_INT QR_CHOLESKYQR = 2;
    const PARCEL_INT QR_CAGMRES_CHOLESKYQR = 202;


    // 計算精度を指定
    template<
      class TcalQR = parcel_double, // QR分解精度
      class Tvec = parcel_double, // QR分解する入力ベクトルV精度
      class Tbasis = parcel_double, // QR分解結果の直交ベクトルQ精度
      class Trmatrix = parcel_double // 上三角行列精度
      >    
    class qr{
      parcel::qr::cgs<TcalQR> parcel_cgs_;
      parcel::qr::mgs<TcalQR> parcel_mgs_;
      parcel::qr::choleskyqr<TcalQR> parcel_choleskyqr_;
      parcel::qr::cagmers_choleskyqr<TcalQR> parcel_cagmers_choleskyqr_;
      PARCEL_INT qr_algo_ = QR_MGS;
    public :
      qr(){}
      ~qr(){}
      void set_qr_algo(PARCEL_INT qr_algo){
	qr_algo_ = qr_algo;
	return ;
      }
      PARCEL_INT get_qr_algo(){
	return qr_algo_;	
      }
      
      void solver(
		  blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
		  PARCEL_INT n, // ベクトルサイズ
		  vector<Tvec> &V, PARCEL_INT ldv,
		  vector<Tbasis> &Q, PARCEL_INT ldq,
		  vector<Trmatrix> &R, PARCEL_INT ldr,
		  std::string Qflag = "Y"		  
		  ){
	switch (qr_algo_) {
	case QR_CGS:
	  //	  std::cout<<" CGS "<<std::endl;	  
	  parcel_cgs_.solver(parcel_blas,n,V,ldv,Q,ldq,R,ldr);
	  return ;
	case QR_MGS:
	  //	  std::cout<<" MGS "<<std::endl;	  
	  parcel_mgs_.solver(parcel_blas,n,V,ldv,Q,ldq,R,ldr);
	  return ;
	case QR_CHOLESKYQR:
	  parcel_choleskyqr_.solver(parcel_blas,n,V,ldv,Q,ldq,R,ldr,Qflag);
	  //	  parcel_mgs_.solver(parcel_blas,n,V,ldv,Q,ldq,R,ldr);
	  return ;
	case QR_CAGMRES_CHOLESKYQR:
	  parcel_cagmers_choleskyqr_.solver(parcel_blas,n,V,Q,R);
	  return ;
	}
	return ;
      }
    };
  // parcel_cgs.solver<
  //   parcel_solver_data_type,
  //   parcel_solver_data_type,
  //   parcel_solver_data_type>
  //   ( parcel_blas,problem_size,V,Q,R );
    
  }
}


#endif
