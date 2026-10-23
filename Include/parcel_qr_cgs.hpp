// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_QR_CGS_HPP_
#define PARCEL_QR_CGS_HPP_

namespace parcel{
  namespace qr{
    // 計算精度を指定
    template<
      class TcalcGS  = parcel_double // グラムシュミット直交化精度
      >
    class cgs{
      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
    public : 
      cgs(){}
      ~cgs(){}
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
#if PARCEL_MPI
	parcel::comm &parcel_comm = parcel_blas.parcel_comm();
#endif
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
	  vec_util.setptr(n, V, Vi0, (size_t)ldv*i); // ベクトルサイズn,Vのn*iポインタを持つベクトルを参照コピー
	  parcel::vector<Trmatrix> hi;
	  vec_util.setptr(s, R, hi, s*i);
	  
	  if(i>0){	  
	    // 1列分(m+1回)の内積計算	
	    PARCEL_INT m_ = 1;
	    PARCEL_INT n_ = i;
	    PARCEL_INT k_ = n;

	    // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if PARCEL_MPI	    
	    parcel::vector<Trmatrix> hi_(hi.size(),que);parcel_blas.zero_init<Trmatrix>(hi_);
	    parcel_blas.gemm<TcalcGS,TcalcGS,Tvec,Tbasis,Trmatrix,TcalcGS>(
									  m_,n_,k_,
									  one,zero,
									  Vi0,1, 
									  Q  ,ldq,
									  hi_,1);
	    parcel_comm.allreduce<Trmatrix,TcalcGS>(hi_,hi,i,MPI_SUM);
#else
	    parcel_blas.gemm<TcalcGS,TcalcGS,Tvec,Tbasis,Trmatrix,TcalcGS>(
									  m_,n_,k_,
									  one,zero,
									  Vi0,1, 
									  Q  ,ldq,
									  hi,1);
#endif
	    
	    que[0].wait();
	    // y = beta * y + alpha * Ax;	    
	    // V( k + (i-1)*n) = V( k + (i-1)*n) - rij(j)*Q( k + (j-1)*n)
#if PARCEL_MPI	    
	    parcel::vector<Tvec> Vi0_add;
	    parcel::vector<Tvec> Q_add;
	    vec_util.setptr(ldv  , V, Vi0_add, (size_t)ldv*i); 
	    vec_util.setptr((size_t)ldq*s, Q, Q_add  , 0); 
	    parcel_blas.gemv<TcalcGS,TcalcGS,Tbasis,Trmatrix,Tvec,TcalcGS>(i,nega_one,one,Q_add,hi,Vi0_add);
#else
	    parcel_blas.gemv<TcalcGS,TcalcGS,Tbasis,Trmatrix,Tvec,TcalcGS>(i,nega_one,one,Q,hi,Vi0);
#endif	    
	    que[0].wait();
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
	  vec_util.setptr(n, Q, Qi, (size_t)ldq*i); 
	  parcel_blas.axpy<TcalcGS,TcalcGS,Tvec,Tbasis,TcalcGS>(qnorm_inv,zero,Vi0,Qi);

	}
	
	que[0].wait();
	
	return ;
	
      }
    };
      
  }
}
#endif
