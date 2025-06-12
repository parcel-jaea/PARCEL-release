#ifndef PARCEL_QR_MGS_HPP_
#define PARCEL_QR_MGS_HPP_

namespace parcel{
  namespace qr{
    // 計算精度を指定
    template<
      class TcalcGS  = parcel_double // グラムシュミット直交化精度
      >
    class mgs{
      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
    public : 
      mgs(){}
      ~mgs(){}
      template<
	class Tvec, // QR分解する入力ベクトルV精度
	class Tbasis, // QR分解結果の直交ベクトルQ精度
	class Trmatrix // 上三角行列精度
	>      
      void solver(
		  parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
		  PARCEL_INT n_, // ベクトルサイズ
		  parcel::vector<Tvec> &V,PARCEL_INT ldv,
		  parcel::vector<Tbasis> &Q,PARCEL_INT ldq,
		  parcel::vector<Trmatrix> &R,PARCEL_INT ldr
		  ){
	//	std::cout<<"MGS START"<<std::endl;
	const PARCEL_INT n = n_;
	const PARCEL_INT s = V.size() / ldv; // 直交化するベクトル数
	
	const auto TcalcGS_name  = typeid(TcalcGS).name();
	using GS_Type      = typename std::result_of<parcel::parcel_type_to_baseType( TcalcGS ) >::type;

	sycl::queue *que   = V.queue();
	parcel::vector<TcalcGS> one;one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);
	parcel::vector<TcalcGS> nega_one;nega_one.resize(1,que);
	double nega_one_ = -1.0;nega_one.setValue(&nega_one_);
	parcel::vector<TcalcGS> zero;zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);
	
	// 古典グラムシュミット直交化
	// 三角行列のゼロ初期化
	parcel_blas.zero_init<Trmatrix>(R);
	que[0].wait();
	
	for(PARCEL_INT i = 0;i<s;i++){
	  // qi~ = vi - sum k=1,i-1 ( <vi,qk> ) * qk
	  // <V,Q>計算
	  parcel::vector<Tvec> Vi0;
	  vec_util.setptr(n, V, Vi0, ldv*i); // ベクトルサイズn,Vのn*iポインタを持つベクトルを参照コピー
	  parcel::vector<Trmatrix> hi;
	  vec_util.setptr(s, R, hi, s*i);
	  
	  if(i>0){	  
	    // 1列分(m+1回)の内積計算	

	    for(PARCEL_INT j = 0;j<i;j++){
	      parcel::vector<Trmatrix> hij;
	      vec_util.setptr(1, hi, hij, j);
	      parcel::vector<Tbasis> Qj0;
	      //	      vec_util.setptr(n, Q, Qj0, ldq*j); // ベクトルサイズn,Vのn*iポインタを持つベクトルを参照コピー
	      vec_util.setptr(n, Q, Qj0, ldq*j); // ベクトルサイズn,Vのn*iポインタを持つベクトルを参照コピー
	      
	      // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if 1
	      parcel_blas.dot< Tvec,Tvec,TcalcGS >(Vi0,Qj0);
	      parcel::scalar VQij = parcel_blas.get_dot();
	      hij.setValue(VQij);
#else
	      PARCEL_INT m_ = 1;
	      PARCEL_INT n_ = 1;
	      PARCEL_INT k_ = n;
	      parcel_blas.gemm<TcalcGS,TcalcGS,Tvec,Tbasis,Trmatrix,TcalcGS>(
									    m_,n_,k_,
									    one,zero,
									    Vi0,1, 
									    Qj0,ldq,
									    hij,1);
#endif	      
	      que[0].wait();
	       
	      // V( k + (i-1)*n) = V( k + (i-1)*n) - hij(j)*Q( k + (j-1)*n)	      
	      parcel_blas.gemv<TcalcGS,TcalcGS,Tbasis,Trmatrix,Tvec,TcalcGS>(1,nega_one,one,Qj0,hij,Vi0);
	      que[0].wait();
	    }
	  }

	  // 直交ベクトルの規格化
	  parcel::scalar one_host = 1.0;
	  parcel_blas.dot< Tvec,Tvec,TcalcGS >(Vi0,Vi0);

	  parcel::scalar qnorm_host = parcel_blas.get_dot();
	  qnorm_host = host_util.sqrt(qnorm_host);
	  parcel::scalar qnorm_host_inv = one_host / qnorm_host;
	  parcel::vector<TcalcGS> qnorm_inv;qnorm_inv.resize(1,que);
	  qnorm_inv.setValue(qnorm_host_inv);

	  parcel::vector<Trmatrix> Rii;
	  vec_util.setptr(1, R, Rii, i + s*i);
	  // Rの対角位置を更新
	  Rii.setValue(qnorm_host);

	  // 直交化ベクトルの規格化
	  parcel::vector<Tbasis> Qi;
	  vec_util.setptr(n, Q, Qi, ldq*i); 
	  parcel_blas.axpy<TcalcGS,TcalcGS,Tvec,Tbasis,TcalcGS>(qnorm_inv,zero,Vi0,Qi);

	}
	
	que[0].wait();
	
	return ;
	
      }
    };
      
  }
}
#endif
