// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_CHOLESKY_CAGMRES_HPP_
#define PARCEL_KRYLOV_CHOLESKY_CAGMRES_HPP_

// ------------------------------------------------------------------------------------------
// detail::cagmres_implicit_q : 直交行列 Q を陽に作らない CholeskyQR 版 CA-GMRES
//   (pcagmres の CAGMRES_Q_IMPLICIT の実装)
//
//   利用者はこのクラスを直接使わず、
//     parcel::krylov::pcagmres<...> s;
//     s.set_qr_algo(parcel::qr::QR_CHOLESKYQR);            // または QR_CAGMRES_CHOLESKYQR
//     s.set_explicit_q(parcel::krylov::CAGMRES_Q_IMPLICIT);
//   と書く。pcagmres::solver() は CAGMRES_Q_IMPLICIT のとき設定をこのクラスに写して solver() を呼び、
//   収束状態を写し戻す (parcel_krylov_cagmres.hpp)。このクラスは公開しない (名前空間 detail)。
//
//   pcagmres と同じ右前処理付き CA-GMRES(s,t) (単基底 v_{i+1} = A K^-1 v_i、周回ごとに s*t 本) で、
//   solver() のインタフェースと収束状態 (converged_reason / iterations / residual / true_residual)、
//   幸運な破綻の扱い (cagmres_breakdown)、||b|| = 0 / 非有限での打ち切りも pcagmres と同じ。違いは次の 3 点。
//
//   (1) Q を陽に作らない。基底 V (周回あたり s*t+1 本) をそのまま保持し、グラム行列 G = V^T V の
//       コレスキー分解 G = R^T R から R だけを求める (V = Q R)。H は pcagmres と同じ式 (R B R^-1) で組み、
//       解の更新は Q y = V (R^-1 y) とする。pcagmres の CholeskyQR (t = 1 のときの Qflag = "N") と同じ考え方を
//       t > 1 にも広げたもの。ブロック k > 0 では、新しい s 本 W と既存の基底 V_old のグラム行列から
//         R12 = R11^-T (V_old^T W) (= Q_old^T W、ブロック古典 GS の係数)、R22 = chol(W^T W - R12^T R12)
//       を求め、R 全体を上三角のまま 1 本ずつ延ばす。ブロックの始点 q_{sk} だけは V R^-1 e_{sk} として作る。
//   (2) 基底を偶数番目 (Teven_basis) と奇数番目 (Todd_basis) に分けて格納し、それぞれの精度を選べる
//       (基底を低精度で持つ CB-GMRES の考え方。既定はどちらも Tbasis で、pcagmres と同じ精度)。
//   (3) set_qr_algo(parcel::qr::QR_CAGMRES_CHOLESKYQR) を指定すると、最初のブロック (k = 0) のグラム行列を
//       parcel::qr::cagmers_choleskyqr で「モーメント」から作る。A が対称で前処理なし (K = I) なら
//         (v_i, v_j) = (A^i v_0, A^j v_0) = v_0^T A^{i+j} v_0 = mu_{i+j}
//       なので、G はハンケル行列で、4 本 (v_0, v_1, v_{s-1}, v_s) と偶数番目の基底 (v_0, v_2, ..., v_s) の
//       内積 4 x (s/2+1) 個から mu_0 .. mu_{2s} がすべて得られる ((s+1)^2 個の内積が要らない)。
//       **A が対称でないと答えが誤る** (検出はしない。利用者の責任)。前処理あり・s が奇数では使えないので
//       solver() の入口で std::invalid_argument を投げる。k > 0 のブロックは (1) の一般の方法で求める。
// ------------------------------------------------------------------------------------------
#include "parcel_qr.hpp"
// cagmres_breakdown と detail::cagmres_implicit_q の前方宣言 (単独で読み込まれた場合のため。
//   parcel_krylov_cagmres.hpp は末尾でこのファイルを読み込む。どちらもインクルードガード付き)
#include "parcel_krylov_cagmres.hpp"
namespace parcel{
  namespace krylov{
  namespace detail{

    // 計算精度を指定 (既定値は pcagmres 側が持つ。ここは pcagmres から型を受け取るだけ)
    template<
      class Tspmv,   // SpMV 計算精度
      class Tprecon, // 前処理計算精度
      class Taxpy,   // ベクトル演算計算精度
      class Tdot     // 縮約演算計算精度
      >
    class cagmres_implicit_q : public krylov_param{

      ddlib::ddmath dd;
      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_scala;

      parcel::timer timer_solver_main;
      parcel::timer timer_solver_SpMV;
      // r = b - Ax は SpMV に加えて b の読み出しと r の書き込みを含む。
      //   SpMV のタイマに入れると回数もトラフィックも食い違うので分ける。
      parcel::timer timer_solver_calcres;
      parcel::timer timer_solver_axpy;
      parcel::timer timer_solver_dot;
      parcel::timer timer_solver_precon;
      parcel::timer timer_solver_gemv;

      parcel::timer timer_solver_qr;
      parcel::timer timer_solver_reOrth;
      // グラム行列の内訳 (内積 / 集約)。reOrth (前のブロックとの内積) と
      //   自己グラム (k = 0 の (s+1)^2、k > 0 の新しい s 本どうし) に分けて測る。どちらも QR の内数
      parcel::timer timer_solver_reOrth_dot;
      parcel::timer timer_solver_reOrth_comm;
      parcel::timer timer_solver_gram_dot;
      parcel::timer timer_solver_gram_comm;
      parcel::timer timer_solver_update;

      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;
      // 既定値は s=8, t=4 (リスタート周期 32)。s=1 では通信削減が効かない。
      PARCEL_INT cagmres_s_ = 8;
      PARCEL_INT cagmres_t_ = 4;
      // QR は常に CholeskyQR (Q を陽に作らない)。
      //   QR_CHOLESKYQR          : グラム行列を内積で作る (一般の行列・前処理あり)
      //   QR_CAGMRES_CHOLESKYQR  : k = 0 のブロックをモーメント (ハンケル行列) で作る (対称行列・前処理なし専用)
      PARCEL_INT qrdecomposition_algo_ = parcel::qr::QR_CHOLESKYQR;

    public :

      // 範囲外は std::invalid_argument。CholeskyQR しか持たないので QR_CGS / QR_MGS も拒否する。
      void set_qr_algo(PARCEL_INT qrdecomposition_algo){
	if(qrdecomposition_algo != parcel::qr::QR_CHOLESKYQR &&
	   qrdecomposition_algo != parcel::qr::QR_CAGMRES_CHOLESKYQR){
	  throw std::invalid_argument("parcel::krylov::pcagmres (CAGMRES_Q_IMPLICIT)::set_qr_algo: invalid value "
				      + std::to_string((long long)qrdecomposition_algo)
				      + " (expected QR_CHOLESKYQR or QR_CAGMRES_CHOLESKYQR)");
	}
	qrdecomposition_algo_ = qrdecomposition_algo;
	return ;
      }

      void set_restart(PARCEL_INT cagmres_s,PARCEL_INT cagmres_t){
	check_positive(cagmres_s, "pcagmres::set_restart (s)");
	check_positive(cagmres_t, "pcagmres::set_restart (t)");
	cagmres_s_ =  cagmres_s;
	cagmres_t_ =  cagmres_t;
	return ;
      }
      // 既定値の参照用 (pcagmres と同じ)
      PARCEL_INT get_restart_s() const { return cagmres_s_; }
      PARCEL_INT get_restart_t() const { return cagmres_t_; }
      PARCEL_INT get_qr_algo()   const { return qrdecomposition_algo_; }
      cagmres_implicit_q(){}
      ~cagmres_implicit_q(){}

      void update_HessenbergMatrix(
				   PARCEL_INT t,
				   PARCEL_INT cagmres_t_,
				   PARCEL_INT cagmres_s_,
				   std::vector< parcel::scalar > &Hmat, // ヘッセンベルグ行列[s*t+1,s*t]
				   std::vector< parcel::scalar > &Bmat, // 基底変換行列[s+1,s]
				   std::vector< parcel::scalar > &Rmat, // QR分解で得られた上三角行列 [s+1,s+1]
				   std::vector< parcel::scalar > &BlockRmat, // ブロック古典グラムシュミット直交化で得られた上三角行列[s*t+1,s+1]
				   parcel::scalar &hk
				   ){
	const PARCEL_INT s = cagmres_s_;
	
	std::vector< parcel::scalar > one(1);
	std::vector< parcel::scalar > zero(1);
	std::vector< parcel::scalar > negative_one(1);
		
	one[0] = 1.0;
	zero[0] = 0.0;	  
	negative_one[0] = -1.0;

	if(t==0){
	std::vector< parcel::scalar > Rmat_inv(s*s);
	for(PARCEL_INT i = 0;i<Rmat_inv.size();i++){ Rmat_inv[i] = 0.0; }
	for(PARCEL_INT j = 0;j<s;j++){
	  for(PARCEL_INT i = 0;i<s;i++){
	    Rmat_inv[i+j*s] = Rmat[i+j*(s+1)];
	  }
	}
	std::string uplo("U");
	std::string diag("N");
	// 逆行列計算
	parcel_blas_host.trtri(uplo,diag,Rmat_inv,s,s);
	  // H[s+1,s] = R_[s+1,s+1]*B1[s+1,s]*R^-1[s,s]	  
	  std::vector< parcel::scalar > RB((s+1)*s);
	  for(PARCEL_INT i = 0;i<RB.size();i++){ RB[i] = 0.0; }
	  if(t==0){
	    for(PARCEL_INT i = 0;i<Hmat.size();i++){ Hmat[i] = 0.0; }
	  }
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	  parcel_blas_host.gemm(
				s+1, // m
				s,   // n
				s+1, // k
				one,
				zero,
				Rmat,s+1,
				Bmat,s+1,
				RB  ,s+1 
				);
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
	  parcel_blas_host.gemm(
				s+1, // m
				s, // n
				s, // k
				one,
				zero,
				RB,s+1,
				Rmat_inv,s,
				Hmat,s*cagmres_t_+1
				);
	  if(t==0){
	    hk = Hmat[ s + (s-1)*(s*cagmres_t_+1) ];
	    return;
	  }
	  
	}else{
	  // オーバーラップした上三角行列成分
	  std::vector< parcel::scalar > Rmat_inv(s*s);
	  for(PARCEL_INT i = 0;i<Rmat_inv.size();i++){ Rmat_inv[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    Rmat_inv[0+j*s] = BlockRmat[ s*t + j*(s*t+1) ];
	  }
	  // オーバーラップしない上三角行列成分
	  for(PARCEL_INT j = 1;j<s;j++){
	    for(PARCEL_INT i = 1;i<s;i++){
	      //  Rtri0(1+i+(j-1+1)*(s+1))=Rtri(i+(j-1)*s)
	      // i+(j+1)*(s+1)
	      Rmat_inv[i+j*s] = Rmat[(i-1)+(j-1)*s];
	    }
	  }

	  std::string uplo("U");
	  std::string diag("N");
	  // 逆行列計算
	  parcel_blas_host.trtri(uplo,diag,Rmat_inv,s,s);
	  
	  std::vector< parcel::scalar > Hmat_all((s*(t+1)+1)*(s*(t+1)));
	  for(PARCEL_INT i = 0;i<Hmat_all.size();i++){ Hmat_all[i] = 0.0; }
	  // 計算済みの上ヘッセンベルグ行列を書き込み
	  for(PARCEL_INT j = 0;j<s*t;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      Hmat_all[i+j*(s*(t+1)+1)] = Hmat[i + j*(s*cagmres_t_+1)];
	    }
	  }

	  // R(1:s*t,1:s) = R_'(1:s*t,1:s)
	  std::vector< parcel::scalar > BlockR((s*t)*s);	  
	  for(PARCEL_INT i = 0;i<BlockR.size();i++){ BlockR[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      BlockR[i+j*(s*t)] = BlockRmat[i+j*(s*t+1)];
	    }
	  }
	  // R_(1:s*t,1:s+1) = R_'(1:s*t,1:s+1)
	  std::vector< parcel::scalar > BlockR_ubar((s*t)*(s+1));	  
	  for(PARCEL_INT i = 0;i<BlockR_ubar.size();i++){ BlockR_ubar[i] = 0.0; }             
	  for(PARCEL_INT j = 0;j<s+1;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      BlockR_ubar[i+j*(s*t)] = BlockRmat[i+j*(s*t+1)];
	    }
	  }
	  // 第1項
	  // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
          std::vector< parcel::scalar > HR((s*t)*s);
	  for(PARCEL_INT i = 0;i<HR.size();i++){ HR[i] = 0.0; }             
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s,   // n
	  			s*t, // k
	  			negative_one,
	  			zero,
	  			Hmat  ,s*cagmres_t_+1,
	  			BlockR,s*t,
	  			HR,s*t
	  			);
	  // 第1項 + 第2項
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s,   // n
	  			s+1, // k
	  			one,
	  			one,
	  			BlockR_ubar,s*t,
	  			Bmat,s+1,
	  			HR,s*t
	  			);
	  // ( 第1項 + 第2項 ) * R^-1
	  std::vector< parcel::scalar > Hblock((s*t)*(s));
	  for(PARCEL_INT i = 0;i<Hblock.size();i++){ Hblock[i] = 0.0; }
	  parcel_blas_host.gemm(
	  			s*t, // m
	  			s, // n
	  			s, // k
	  			one,
	  			zero,
	  			HR,s*t,
	  			Rmat_inv,s,
				Hblock,s*t
	  			);
	  std::vector< parcel::scalar > Rmat_ubar((s+1)*(s+1));
	  for(PARCEL_INT i = 0;i<Rmat_ubar.size();i++){ Rmat_ubar[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s+1;j++){
	    Rmat_ubar[0+j*(s+1)] = BlockRmat[ s*t + j*(s*t+1) ];
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      Rmat_ubar[i+1+(j+1)*(s+1)] = Rmat[i+j*s];
	    }
	  }
	  parcel::scalar bk = Bmat[s*(s+1)-1];
	  parcel::scalar rho_tilde = Rmat_ubar[s-1+(s-1)*(s+1)]; // Rtri0(s   + (s-1)*(s+1));
	  parcel::scalar rho       = Rmat_ubar[s  +    s*(s+1)]; // Rtri0(s+1 +     s*(s+1));
	  parcel::scalar hk_ = rho/rho_tilde*bk;
	  std::vector< parcel::scalar > zk(s);
	  for(PARCEL_INT i = 0;i<zk.size();i++){ zk[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    zk[i] = Rmat_ubar[i+s*(s+1)];
	  }

	  // --------------------------------------------
	  // 単基底以外では要検討が必要
	  // 単基底では1.0
	  std::vector< parcel::scalar > e1(s);
	  for(PARCEL_INT i = 0;i<e1.size();i++){ e1[i] = 0.0; }
	  e1[0] = 1.0;
	  
	  std::vector< parcel::scalar > es(s);
	  for(PARCEL_INT i = 0;i<es.size();i++){ es[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    es[i] = Bmat[i+1+(s-1)*(s+1)];
	  }
	  // 単基底では固定 (ニュートン基底では変更する必要有り)
	  std::vector< parcel::scalar > es_(s);	 
	  for(PARCEL_INT i = 0;i<es_.size();i++){ es_[i] = 0.0; }
	  for(PARCEL_INT i = 0;i<s;i++){
	    es_[i] = Bmat[i+1+(s-1)*(s+1)];
	  }
	  // --------------------------------------------

	  
	  // R B R ^-1
	  // 第1項
	  // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
          std::vector< parcel::scalar > RB(s*s);
	  for(PARCEL_INT i = 0;i<RB.size();i++){ RB[i] = 0.0; }             
	  parcel_blas_host.gemm(
	  			s, // m
	  			s, // n
	  			s, // k
	  			one,
	  			zero,
	  			Rmat_ubar,s+1,
	  			Bmat,s+1,
	  			RB,s
	  			);
	  
	  // rho b z es

	  // 第2項
	  // - h R R^-1
	  std::vector< parcel::scalar > ze( s*s  );
	  for(PARCEL_INT i = 0;i<ze.size();i++){ ze[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      ze[i+j*s] = bk/rho_tilde*zk[i]*es[j];
	    }
	  }
	  // 第3項
	  std::vector< parcel::scalar > Rmat_(s*s);
	  for(PARCEL_INT i = 0;i<Rmat_.size();i++){ Rmat_[i] = 0.0; }	  
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      // k-1で生成した基底ベクトルに対する再直交結果を書き込み
	      Rmat_[i+j*s] = BlockRmat[ i + s*(t-1) + j*(s*t+1) ];
	    }
	  }
	  
	  std::vector< parcel::scalar > e1es_( s*s  );
	  for(PARCEL_INT i = 0;i<e1es_.size();i++){ e1es_[i] = 0.0; }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      e1es_[i+j*s] = -hk * e1[i] * es_[j];
	    }
	  }
	  // C[m,n] = one * A[m,k] * B[k,n] + zero * C[m,n]
	  parcel_blas_host.gemm(
				s, // m
				s, // n
				s, // k
				one,
				one,
				e1es_,s,
				Rmat_,s,
				RB,s 
				);
	  parcel_blas_host.gemm(
				s, // m
				s, // n
				s, // k
				one,
				one,
				RB,s,
				Rmat_inv,s,
				ze,s 
				);
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s*t;i++){
	      Hmat_all[i+(j+s*t)*(s*(t+1)+1)] = Hblock[i+j*(s*t)];	      
	    }
	  }
	  for(PARCEL_INT j = 0;j<s;j++){
	    for(PARCEL_INT i = 0;i<s;i++){
	      Hmat_all[i+s*t+(j+s*t)*(s*(t+1)+1)] = ze[i+j*s];
	    }
	  }
	  Hmat_all[ (s*(t+1)+1)*(s*(t+1)) -1 - ( s + s*(s*(t+1)+1)) ] = hk;
	  Hmat_all[ (s*(t+1)+1)*(s*(t+1)) -1 ] = hk_;
	  // --------------
	  for(PARCEL_INT j = 0;j<s*(t+1);j++){
	    for(PARCEL_INT i = 0;i<s*(t+1)+1;i++){
	      Hmat[i + j*(s*cagmres_t_+1)] = Hmat_all[i+j*(s*(t+1)+1)];
	    }
	  }
	  
	  hk = hk_;
	  return ;
	}
      }

      void GivensRotation(
			  PARCEL_INT m,
			  PARCEL_INT cagmres_t_,
			  PARCEL_INT cagmres_s_,
			  std::vector< parcel::scalar > &hki,
			  std::vector< parcel::scalar > &e,
			  std::vector< parcel::scalar > &co,
			  std::vector< parcel::scalar > &si
			  ){

	// ----- 
	// m+1行のヘッセンベルグ行列に対するギブンス回転
	// Givens rotation
	for(PARCEL_INT k=0;k<m;k++){
	  parcel::scalar hki_km  = hki[ k   + m*(cagmres_s_*cagmres_t_+1) ];
	  parcel::scalar hki_k1m = hki[ k+1 + m*(cagmres_s_*cagmres_t_+1) ];
	  parcel::scalar si_ = si[k];
	  parcel::scalar co_ = co[k];
	  parcel::scalar tmp1_ = hki_km * co_ - hki_k1m * si_;
	  parcel::scalar tmp2_ = hki_km * si_ + hki_k1m * co_;
	  hki[ k   + m*(cagmres_s_*cagmres_t_+1) ] = tmp1_;
	  hki[ k+1 + m*(cagmres_s_*cagmres_t_+1) ] = tmp2_;
	}

	// -------
	parcel::scalar hki_m1m = hki[m+1 + m*(cagmres_s_*cagmres_t_+1)];
	parcel::scalar hki_mm  = hki[m   + m*(cagmres_s_*cagmres_t_+1)];
	parcel::scalar tmp1_   = hki_m1m / hki_mm;
	parcel::scalar tmp2_   = 1.0;tmp2_ = tmp2_ + tmp1_*tmp1_;

	tmp2_ = host_util.sqrt(tmp2_);
	parcel::scalar co_ = 1.0;co_ = co_ /tmp2_;
	co[m] =  co_;
	si[m] = -co_*tmp1_;

	parcel::scalar si_ = si[m];
	parcel::scalar e_  = e[m];
	parcel::scalar e1_ = e[m+1];

	hki[m   + m*(cagmres_s_*cagmres_t_+1)] = co_*hki_mm - si_*hki_m1m;
	hki[m+1 + m*(cagmres_s_*cagmres_t_+1)] = 0.0;

	parcel::scalar tmp3_ = co_ * e_ - si_ * e1_;
	parcel::scalar tmp4_ = si_ * e_ + co_ * e1_;

	e[m]   = tmp3_;
	e[m+1] = tmp4_;

	return ;

      }

      // ---------------------------------------------------------------------------------
      // Q を陽に作らない CholeskyQR の部品
      //   基底 v_g (g = 0 .. s*t) は偶数番目を Veven の g/2 列、奇数番目を Vodd の g/2 列に、
      //   列の間隔 n (局所行数、袖なし) で持つ。偶数・奇数の添字の範囲は格納先で連続した列になる。
      // ---------------------------------------------------------------------------------

      // グラム行列の部分 C = A(:, ca .. ca+na-1)^T B(:, cb .. cb+nb-1) を局所の dot (allreduce しない版) で
      //   (i, j) ごとに求め、host の Gloc[off + i + j*na] に書く (列優先。集約は呼び出し側)。
      //   計算精度は TcalcQR。tri = true (A と B が同じ列の範囲) なら上三角 (i <= j) だけ計算して
      //   下三角に写す (局所値のまま写すので、集約後も厳密に対称)。
      template<class TcalcQR, class TA, class TB>
      void gram_dot_block(parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas, PARCEL_INT n,
			  parcel::vector<TA> &A, PARCEL_INT ca, PARCEL_INT na,
			  parcel::vector<TB> &B, PARCEL_INT cb, PARCEL_INT nb,
			  bool tri, std::vector< parcel::scalar > &Gloc, size_t off){
	for(PARCEL_INT j = 0; j < nb; j++){
	  parcel::vector<TB> Bj; vec_util.setptr( (size_t)n, B, Bj, (size_t)n*(cb+j) );
	  const PARCEL_INT imax = tri ? (j+1) : na;
	  for(PARCEL_INT i = 0; i < imax; i++){
	    parcel::vector<TA> Ai; vec_util.setptr( (size_t)n, A, Ai, (size_t)n*(ca+i) );
	    parcel_blas.dot<TA,TB,TcalcQR>(Ai, Bj, false);
	    Gloc[off + (size_t)i + (size_t)j*na] = parcel_blas.get_dot();
	  }
	}
	if(tri){
	  for(PARCEL_INT j = 0; j < nb; j++){
	    for(PARCEL_INT i = j+1; i < na; i++){
	      Gloc[off + (size_t)i + (size_t)j*na] = Gloc[off + (size_t)j + (size_t)i*na];
	    }
	  }
	}
      }

      // 内積 G(a, b) = (v_a, v_b) (a = a0..a1, b = b0..b1) を host の G (列優先, ld = ldg, 大域の基底番号) に書く。
      //   偶数・奇数の格納の組ごとに最大 4 組の局所内積 (gram_dot_block) を並べ、1 回の MPI_Allreduce で集約する。
      //   a と b が同じ範囲なら対称性で (奇, 偶) の組を省き、転置を書く。対角の組は上三角だけ計算する。
      template<class TcalcQR, class THessenberg, class Teven_basis, class Todd_basis>
      void gram_block(parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
#if PARCEL_MPI
		      MPI_Comm mpi_comm,
#endif
		      PARCEL_INT n,
		      parcel::vector<Teven_basis> &Veven,
		      parcel::vector<Todd_basis>  &Vodd,
		      PARCEL_INT a0, PARCEL_INT a1, PARCEL_INT b0, PARCEL_INT b1,
		      parcel::vector<THessenberg> &Gdev,
		      std::vector<double> &G, PARCEL_INT ldg,
		      parcel::timer &t_dot, parcel::timer &t_comm){
	sycl::queue *que = Gdev.queue();
	const bool sym = (a0 == b0) && (a1 == b1);
	// 範囲 [lo, hi] のうち偶数 (par = 0) / 奇数 (par = 1) の番号: 先頭 first と本数 cnt
	auto sub = [](PARCEL_INT lo, PARCEL_INT hi, PARCEL_INT par, PARCEL_INT &first, PARCEL_INT &cnt){
	  first = lo + (((lo % 2) == par) ? 0 : 1);
	  cnt   = (first <= hi) ? (hi - first)/2 + 1 : 0;
	};
	PARCEL_INT fa[2], na[2], fb[2], nb[2];
	for(PARCEL_INT p = 0; p < 2; p++){
	  sub(a0, a1, p, fa[p], na[p]);
	  sub(b0, b1, p, fb[p], nb[p]);
	}
	PARCEL_INT pc_a[4], pc_b[4];
	size_t     pc_off[4];
	PARCEL_INT npc = 0;
	size_t off = 0;
	std::vector< parcel::scalar > Gloc( (size_t)Gdev.size(), 0.0 );
	t_dot.start();
	for(PARCEL_INT pa = 0; pa < 2; pa++){
	  for(PARCEL_INT pb = 0; pb < 2; pb++){
	    if(na[pa] == 0 || nb[pb] == 0) continue;
	    if(sym && pa == 1 && pb == 0) continue;   // (奇, 偶) は (偶, 奇) の転置
	    if(off + (size_t)na[pa]*nb[pb] > (size_t)Gdev.size()){
	      throw std::logic_error("parcel::krylov::pcagmres (CAGMRES_Q_IMPLICIT)::gram_block: work buffer too small");
	    }
	    const bool tri = sym && (pa == pb);
	    if(pa == 0 && pb == 0){
	      gram_dot_block<TcalcQR>(parcel_blas,n,Veven,fa[0]/2,na[0],Veven,fb[0]/2,nb[0],tri,Gloc,off);
	    }else if(pa == 0 && pb == 1){
	      gram_dot_block<TcalcQR>(parcel_blas,n,Veven,fa[0]/2,na[0],Vodd ,fb[1]/2,nb[1],tri,Gloc,off);
	    }else if(pa == 1 && pb == 0){
	      gram_dot_block<TcalcQR>(parcel_blas,n,Vodd ,fa[1]/2,na[1],Veven,fb[0]/2,nb[0],tri,Gloc,off);
	    }else{
	      gram_dot_block<TcalcQR>(parcel_blas,n,Vodd ,fa[1]/2,na[1],Vodd ,fb[1]/2,nb[1],tri,Gloc,off);
	    }
	    pc_a[npc] = pa; pc_b[npc] = pb; pc_off[npc] = off; npc++;
	    off += (size_t)na[pa]*nb[pb];
	  }
	}
	que[0].wait();
	t_dot.end();
	t_comm.start();
	// 局所値を THessenberg の Gdev に一度置いてから読み戻す (格納の丸めを THessenberg にそろえる)
	if(off > 0){
	  parcel::vector<THessenberg> Gv; vec_util.setptr( off, Gdev, Gv, 0 );
	  Gv.setValue(Gloc);
	}
	que[0].wait();
	std::vector< parcel::scalar > Gs(Gdev.size());
	Gdev.getValue(Gs);
	std::vector<double> buf(off > 0 ? off : 1, 0.0);
	for(size_t i = 0; i < off; i++){ buf[i] = (double)Gs[i].get_T(); }
#if PARCEL_MPI
	MPI_Allreduce(MPI_IN_PLACE, buf.data(), (int)off, MPI_DOUBLE, MPI_SUM, mpi_comm);
#endif
	t_comm.end();
	for(PARCEL_INT q = 0; q < npc; q++){
	  const PARCEL_INT pa = pc_a[q], pb = pc_b[q];
	  for(PARCEL_INT j = 0; j < nb[pb]; j++){
	    for(PARCEL_INT i = 0; i < na[pa]; i++){
	      const PARCEL_INT ga = fa[pa] + 2*i;
	      const PARCEL_INT gb = fb[pb] + 2*j;
	      const double v = buf[pc_off[q] + (size_t)i + (size_t)j*na[pa]];
	      G[(size_t)ga + (size_t)gb*ldg] = v;
	      if(sym){ G[(size_t)gb + (size_t)ga*ldg] = v; }
	    }
	  }
	}
      }

      // w = V(:, 0 .. m-1) c (偶数・奇数の格納ごとに GEMV 1 回ずつ)。w は局所 n 要素
      template<class Tvec, class Teven_basis, class Todd_basis, class Tout>
      void combine_basis(parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas, PARCEL_INT n, PARCEL_INT m,
			 const std::vector<double> &c,
			 parcel::vector<Teven_basis> &Veven,
			 parcel::vector<Todd_basis>  &Vodd,
			 parcel::vector<Tout> &w){
	sycl::queue *que = w.queue();
	const PARCEL_INT ne = (m+1)/2;
	const PARCEL_INT no = m/2;
	std::vector< parcel::scalar > ce(ne);
	std::vector< parcel::scalar > co(no > 0 ? no : 1);
	for(PARCEL_INT j = 0; j < m; j++){
	  if(j % 2 == 0){ ce[j/2] = c[j]; }else{ co[j/2] = c[j]; }
	}
	parcel::vector<Tdot> one;  one.resize(1,que);  double one_  = 1.0; one.setValue(&one_);
	parcel::vector<Tdot> zero; zero.resize(1,que); double zero_ = 0.0; zero.setValue(&zero_);
	parcel::vector<Tvec> ye; ye.resize(ne,que); ye.setValue(ce);
	parcel::vector<Teven_basis> Ve; vec_util.setptr( (size_t)n*ne, Veven, Ve, 0 );
	que[0].wait();
	parcel_blas.gemv<Tdot,Tdot,Teven_basis,Tvec,Tout,Taxpy>(ne,one,zero,Ve,ye,w,n);
	if(no > 0){
	  parcel::vector<Tvec> yo; yo.resize(no,que); yo.setValue(co);
	  parcel::vector<Todd_basis> Vo; vec_util.setptr( (size_t)n*no, Vodd, Vo, 0 );
	  que[0].wait();
	  parcel_blas.gemv<Tdot,Tdot,Todd_basis,Tvec,Tout,Taxpy>(no,one,one,Vo,yo,w,n);
	}
	que[0].wait();
      }

      // 対称行列 S (m x m, 列優先, ld = lds) のコレスキー分解 S = U^T U。上三角 U を
      //   U (ld = ldu) の (u0, u0) から書く。parcel_blas_host.potrf と同じく対角は sqrt(|.|) とし、
      //   負や 0 になっても止めない (階数落ちは cagmres_breakdown::find_rank_deficient で検出する)。
      static void chol_upper(const std::vector<double> &S, PARCEL_INT m, PARCEL_INT lds,
			     std::vector<double> &U, PARCEL_INT ldu, PARCEL_INT u0){
	for(PARCEL_INT j = 0; j < m; j++){
	  double d = S[(size_t)j + (size_t)j*lds];
	  for(PARCEL_INT k = 0; k < j; k++){
	    const double u = U[(size_t)(u0+k) + (size_t)(u0+j)*ldu];
	    d -= u*u;
	  }
	  const double ujj = std::sqrt(std::fabs(d));
	  U[(size_t)(u0+j) + (size_t)(u0+j)*ldu] = ujj;
	  for(PARCEL_INT i = j+1; i < m; i++){
	    double v = S[(size_t)j + (size_t)i*lds];
	    for(PARCEL_INT k = 0; k < j; k++){
	      v -= U[(size_t)(u0+k) + (size_t)(u0+j)*ldu] * U[(size_t)(u0+k) + (size_t)(u0+i)*ldu];
	    }
	    U[(size_t)(u0+j) + (size_t)(u0+i)*ldu] = v / ujj;
	  }
	}
      }

      // 前処理,IR等の設定をclassで定義する
      //   Todd_basis / Teven_basis の既定は Tbasis (指定しなければ pcagmres と同じ精度で基底を持つ)。
      template<
	class Tmat,  // 係数行列データ精度
	class Tvecx, // 入力ベクトルデータ精度
	class Tvecb, // 右辺ベクトルデータ精度
	class Tvec = parcel_double, // クリロフ部分空間法内部データ精度
	class Tprecon_mat = parcel_double, // 前処理行列データ精度
	class Tprecon_vec = parcel_double, // 前処ベクトルデータ精度
	class Tbasis = parcel_double, // 前処理の入出力 (基底から作る作業ベクトル) のデータ精度
	class TcalcQR = parcel_double, // QR分解 (グラム行列) 計算精度
	class THessenberg = parcel_double, // ヘッセンベルグ行列の計算精度
	class Todd_basis  = Tbasis, // 奇数番目の基底ベクトルの格納精度
	class Teven_basis = Tbasis  // 偶数番目の基底ベクトルの格納精度
	>
      void solver(
		  parcel::matrix<Tmat>  &Mat,
		  parcel::vector<Tvecx> &x0,
		  parcel::vector<Tvecb> &b0
		  ){

	sycl::queue *que   = Mat.queue();

	require_in_order(que, "krylov::pcagmres");   // in-order キューが前提
	const PARCEL_INT n = x0.size();
	reset_status();

	const PARCEL_INT s  = cagmres_s_;
	const PARCEL_INT t  = cagmres_t_;
	const PARCEL_INT st = s*t;       // 1 周回の反復回数 (H の列数)
	const PARCEL_INT Nt = st + 1;    // 1 周回の基底ベクトルの本数

	precon_param &precon = *(get_precon_param());
	// モーメント法 (QR_CAGMRES_CHOLESKYQR) の前提は入口で確かめる (全ランクで同じ判定)
	const bool use_moment = (qrdecomposition_algo_ == parcel::qr::QR_CAGMRES_CHOLESKYQR);
	if(use_moment){
	  if(precon.get_precon_algo() != parcel::PRECON_NONE){
	    throw std::invalid_argument("parcel::krylov::pcagmres::solver: QR_CAGMRES_CHOLESKYQR requires "
					"PRECON_NONE (the moment Gram matrix assumes a symmetric A K^-1 = A)");
	  }
	  if(s < 2 || (s % 2) != 0){
	    throw std::invalid_argument("parcel::krylov::pcagmres::solver: QR_CAGMRES_CHOLESKYQR requires "
					"an even s >= 2 (got s = " + std::to_string((long long)s) + ")");
	  }
	}

#if PARCEL_MPI
	parcel::comm *comm = Mat.comm();
	int myrank   = comm->rank();
	comm->barrier();
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,comm->mpi_comm());
	PARCEL_INT nadd = Mat.get_recv_size();
#else
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
	int myrank = 0;
	PARCEL_INT nadd = 0;
#endif

	timer_solver.start();
	timer_solver_init.start();

	// ------------------------------
	// Init Preconditioning
	timer_solver_init_precon.start();

	const double tol = get_tolerance();
	const PARCEL_INT itrmax = get_max_iteration();
	parcel::precon_matrix<
	  Tmat,
	  Tbasis,
	  Tprecon_mat,
	  Tprecon_vec,
	  Tprecon
	  > precon_Mat(Mat,precon);
	que[0].wait();
	timer_solver_init_precon.end();

	timer_solver_init_convert.start();
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> x_parallel(n,nadd,1,que);parcel::vector<Tvec> x;vec_util.setptr( n, x_parallel.vec(0), x, 0);
	parcel_blas.zero_init<Tvec>(x);vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::parallel_vector<Tvec> b_parallel(n,nadd,1,que);parcel::vector<Tvec> b;vec_util.setptr( n, b_parallel.vec(0), b, 0);
	parcel_blas.zero_init<Tvec>(b);vec_util.convert<Tvecb,Tvec>(b0,b);
	parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel_blas.zero_init<Tvec>(r);
#else
	parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);
	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
#endif
 	que[0].wait();
	timer_solver_init_convert.end();

	// ! ------------------------------------------------------------
	// ! CA-GMRES (Q を陽に作らない CholeskyQR 版) のメインループ
	// ! do itr = 1,itrmax,s*t
	// !   r = b-Ax, v_0 = r/||r||
	// !   do k = 1,t
	// !     w = ブロックの始点 (k=1 : v_0, k>1 : q_{s(k-1)} = V R^-1 e)
	// !     do i=1,s
	// !       solve M p = w_i ; v_{s(k-1)+i} = A p (単基底)
	// !     enddo
	// !     グラム行列 → コレスキー分解で R を 1 ブロック延ばす (k>1 はブロック古典 GS の係数 R12 も)
	// !     ヘッセンベルグ行列の計算
	// !   enddo
	// !   ギブンス回転, y = H^-1 zeta, z = R^-1 y
	// !   解ベクトル更新 x = x + M^-1 V z
	// ! enddo
	// ! ------------------------------------------------------------

	// 基底 v_g (g = 0 .. s*t): 偶数番目は Veven、奇数番目は Vodd (列の間隔 n、袖なし)
	const PARCEL_INT ncol_even = (Nt+1)/2;
	const PARCEL_INT ncol_odd  = Nt/2;
	parcel::vector<Teven_basis> Veven( (size_t)n*ncol_even, que); parcel_blas.zero_init<Teven_basis>(Veven);
	parcel::vector<Todd_basis>  Vodd ( (size_t)n*ncol_odd , que); parcel_blas.zero_init<Todd_basis>(Vodd);

	// 前処理の入力 w と出力 pV。pV は SpMV の入力、w は前処理 (ヤコビ等) の入力になるので袖付き
	// Tvec と Tbasis が同じ型なら w は r の領域を使う (省メモリ)。
	//   r は周回の先頭で r = b - Ax を作り v_0 = r/||r|| を Veven に書いた後は次の周回まで読まない。
	//   w を書くのはその後 (ブロックの始点の複写・V c・基底の複写) で、周回の最後の解の更新 (w = V z) の後は
	//   次の周回の r = b - Ax まで読まない。
#if PARCEL_MPI
	constexpr bool w_alias = std::is_same<Tvec,Tbasis>::value;
	parcel::parallel_vector<Tbasis> w_parallel(w_alias ? 1 : n, w_alias ? 0 : nadd, 1, que);
	parcel::parallel_vector<Tbasis> pV_parallel(n,nadd,1,que);
	parcel::vector<Tbasis> w;
	parcel::vector<Tbasis> w_vecadd;
	if constexpr (w_alias){
	  vec_util.setptr( n       , r_parallel.vec(0)     , w        , 0 );
	  vec_util.setptr( (n+nadd), r_parallel.vec_add(0) , w_vecadd , 0 );
	}else{
	  vec_util.setptr( n       , w_parallel.vec(0)     , w        , 0 );
	  vec_util.setptr( (n+nadd), w_parallel.vec_add(0) , w_vecadd , 0 );
	}
	parcel::vector<Tbasis> pV;        vec_util.setptr( n       , pV_parallel.vec(0)    , pV       , 0 );
	parcel::vector<Tbasis> pV_vecadd; vec_util.setptr( (n+nadd), pV_parallel.vec_add(0), pV_vecadd, 0 );
#else
	parcel::vector<Tbasis> w;
	if constexpr (std::is_same<Tvec,Tbasis>::value){ vec_util.setptr( n, r, w, 0 ); }
	else{ w.resize(n,que); }
	parcel::vector<Tbasis> pV(n,que);
	parcel::vector<Tbasis> &w_vecadd  = w;
	parcel::vector<Tbasis> &pV_vecadd = pV;
#endif
	parcel_blas.zero_init<Tbasis>(w_vecadd);
	parcel_blas.zero_init<Tbasis>(pV_vecadd);

	// グラム行列の局所部分の作業域 (1 回の gram_block で最大 max((s+1)^2, (s*(t-1)+1)*s) 個)
	parcel::vector<THessenberg> Gdev( (size_t)Nt*(cagmres_s_+1), que); parcel_blas.zero_init<THessenberg>(Gdev);
	// モーメント法の作業域: 4 本 (v_0, v_1, v_{s-1}, v_s) と R ((s+1)x(s+1))
	parcel::vector<Teven_basis> VcholQR;
	parcel::vector<THessenberg> Rdev;
	if(use_moment){
	  VcholQR.resize( (size_t)n*4, que); parcel_blas.zero_init<Teven_basis>(VcholQR);
	  Rdev.resize( (size_t)(s+1)*(s+1), que); parcel_blas.zero_init<THessenberg>(Rdev);
	}

	parcel::vector<Tdot> one;
	one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);

	parcel::vector<Tdot> zero;
	zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);

	// --------------------------------------------
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

	// ||b|| = 0 なら解は x = 0 (0/0 = NaN にしないため反復しない)
	if(bnorm_ == 0.0){
	  parcel_blas.zero_init<Tvecx>(x0);
	  timer_solver_init.end();
	  timer_solver.end();
	  if(myrank == 0){
	    std::cout<<" ||b|| = 0 : x = 0 (no iteration)"<<std::endl;
	  }
	  set_status(CONVERGED_RTOL, 0, 0.0, 0.0);
	  print_status(myrank);
	  return ;
	}

	// r = b - Ax
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	// || r ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);

	// 収束判定
	PARCEL_MAX_FLOTING_TYPE rnorm_ = 1.0;
	{
	  parcel::scalar rnorm_host = parcel_blas.get_dot();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  rnorm_ = rnorm_host.get_T();
	  double bnorm_FP64 = bnorm_;
	  double rnorm_FP64 = rnorm_;
	  if((rnorm_/bnorm_)<tol){
	    // 初期残差で収束済み。x は変えていないので残差はそのまま真の残差
	    set_status(CONVERGED_RTOL, 0, rnorm_FP64/bnorm_FP64, rnorm_FP64/bnorm_FP64);
	    print_status(myrank);
	    return;
	  }
	}
	timer_solver_init.end();

	// 収束状態 (krylov_param::set_status)。pcagmres と同じ数え方。
	//   反復上限で抜けたときは最後の更新後の値 (finish) で判定する。
	bool   st_converged = false;
	bool   st_nonfinite = false;
	bool   st_breakdown = false;
	// 反復の上限 itrmax は s*t の倍数でなくてよい (pcagmres の Q を作る経路と同じ扱い。
	//   説明は parcel_krylov_cagmres.hpp の pcagmres::solver)。周回数は ceil(itrmax/st)、最後の周回は
	//   m_cyc = itrmax - itr*st 列だけ使い、基底は t_cyc = ceil(m_cyc/s) ブロック (各 s ステップ) だけ作る。
	//   最後のブロックの残り t_cyc*s - m_cyc (< s) 本は作るが使わない。上限まで回ったときの its は itrmax。
	//   itrmax が st の倍数なら n_cycle = itrmax/st、m_cyc = st、t_cyc = t。
	const PARCEL_INT n_cycle = itrmax/st + ((itrmax % st) != 0 ? 1 : 0);
	PARCEL_INT st_its   = itrmax;
	double st_res       = std::numeric_limits<double>::quiet_NaN();
	// 階数落ちの閾値: QR の計算精度の sqrt(u) の桁 (CholeskyQR) と、基底の格納精度・作業ベクトルの精度の揺らぎ (10 u) の大きい方
	const double tau = std::max( std::max( cagmres_breakdown::tolerance<Teven_basis,TcalcQR>(true),
					       cagmres_breakdown::tolerance<Todd_basis ,TcalcQR>(true) ),
				     cagmres_breakdown::tolerance<Tbasis,TcalcQR>(true) );
	timer_solver_main.start();
	for(PARCEL_INT itr = 0;itr < n_cycle;itr++){
	  // ------------
	  std::vector< parcel::scalar > Bmat( (s+1)*s );
	  std::vector< parcel::scalar > Rmat( (s+1)*(s+1) );
	  std::vector< parcel::scalar > BlockRmat( (size_t)Nt*(s+1) );
	  std::vector< parcel::scalar > Hmat( (size_t)Nt*st );
	  // ! 基底変換行列を作成
	  // 単基底のみ対応 :
	  for(size_t i = 0;i<Bmat.size() ;i++){Bmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<s;i++){
	    Bmat[ (i+1) + i*(s+1) ] = 1.0;
	  }

	  std::vector< parcel::scalar > zeta(Nt);
	  std::vector< parcel::scalar > co(Nt);
	  std::vector< parcel::scalar > si(Nt);
	  parcel::scalar hk;hk = 0.0;
	  for(size_t i = 0;i<co.size();i++   ){co[i]   = 0.0;}
	  for(size_t i = 0;i<si.size();i++   ){si[i]   = 0.0;}
	  for(size_t i = 0;i<zeta.size() ;i++){zeta[i] = 0.0;}
	  for(size_t i = 0;i<Hmat.size() ;i++){Hmat[i] = 0.0;}
	  for(size_t i = 0;i<Rmat.size() ;i++){Rmat[i] = 0.0;}
	  for(size_t i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = 0.0;}
	  // グラム行列と V = Q Rall の上三角 Rall (どちらも Nt x Nt、大域の基底番号、host)
	  std::vector<double> Gh  ( (size_t)Nt*Nt, 0.0 );
	  std::vector<double> Rall( (size_t)Nt*Nt, 0.0 );

	  // r = b - Ax
	  timer_solver_calcres.start();
#if PARCEL_MPI
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	  timer_solver_calcres.end();

	  // || r ||
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	  timer_solver_dot.end();

	  // 収束判定
	  {
	    parcel::scalar rnorm_host = parcel_blas.get_dot();
	    rnorm_host = host_util.sqrt(rnorm_host);
	    rnorm_ = rnorm_host.get_T();
	    double bnorm_FP64 = bnorm_;
	    double rnorm_FP64 = rnorm_;
	    // 表示は rank 0 だけ
	    if( myrank == 0){
	      std::cout<<std::setw(8)<< itr*st <<" : residual norm = ";
	      std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	    }
	    st_res = rnorm_FP64/bnorm_FP64;
	    // 非有限になったらその場で抜ける
	    if(!std::isfinite(st_res)){ st_nonfinite = true; st_its = itr*st; break; }
	    if((rnorm_/bnorm_)<tol){ st_converged = true; st_its = itr*st; break; }
	  }
	  zeta[0] =  rnorm_;
	  // v_0 = r / || r || (Veven の 0 列)
	  {
	    parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(1.0/rnorm_);
	    parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	    rnorm_inv.setValue(rnorm_inv_host);
	    timer_solver_axpy.start();
	    parcel::vector<Teven_basis> V0; vec_util.setptr( n, Veven, V0, 0 );
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Teven_basis,Taxpy> (rnorm_inv,zero,r,V0);
	    timer_solver_axpy.end();
	  }
	  // この周回で使う列数 m_cyc と作るブロック数 t_cyc (最後の周回だけ上限の端数に減らす)
	  const PARCEL_INT m_cyc = std::min(st, itrmax - itr*st);
	  const PARCEL_INT t_cyc = (m_cyc + s - 1)/s;
	  // ブロック内の階数落ち (幸運な破綻) で有効な H の列数 (pcagmres と同じ)。
	  //   無ければ m_cyc (最後の周回で端数に減らしたとき以外は st)
	  PARCEL_INT m_cap = m_cyc;

	  for(PARCEL_INT k = 0;k < t_cyc;k++){
	    const PARCEL_INT base = s*k;   // このブロックの始点の基底番号
	    // ---- ブロックの始点を w に置く ----
	    //   k = 0 : v_0。k > 0 : q_{sk} = V(:, 0..sk) Rall^-1 e_{sk} (Q の sk 列目だけを作る)
	    timer_solver_axpy.start();
	    if(k == 0){
	      parcel::vector<Teven_basis> V0; vec_util.setptr( n, Veven, V0, 0 );
	      vec_util.convert<Teven_basis,Tbasis>(V0, w);
	    }else{
	      std::vector<double> c( base+1, 0.0 );
	      for(PARCEL_INT i = base; i >= 0; i--){
		double v = (i == base) ? 1.0 : 0.0;
		for(PARCEL_INT j = i+1; j <= base; j++){ v -= Rall[(size_t)i + (size_t)j*Nt] * c[j]; }
		c[i] = v / Rall[(size_t)i + (size_t)i*Nt];
	      }
	      combine_basis<Tvec>(parcel_blas, n, base+1, c, Veven, Vodd, w);
	    }
	    timer_solver_axpy.end();

	    // ---- 基底の生成 v_{g+1} = A K^-1 v_g (単基底) ----
	    for(PARCEL_INT i = 0;i < s;i++){
	      const PARCEL_INT g = base + i;   // 入力の基底番号 (i = 0 は始点で、w に置き済み)
	      if(i > 0){
		timer_solver_axpy.start();
		if(g % 2 == 0){
		  parcel::vector<Teven_basis> Vg; vec_util.setptr( n, Veven, Vg, (size_t)n*(g/2) );
		  vec_util.convert<Teven_basis,Tbasis>(Vg, w);
		}else{
		  parcel::vector<Todd_basis>  Vg; vec_util.setptr( n, Vodd , Vg, (size_t)n*(g/2) );
		  vec_util.convert<Todd_basis,Tbasis>(Vg, w);
		}
		timer_solver_axpy.end();
	      }
	      // 前処理
	      timer_solver_precon.start();
	      precon_Mat.solver(w_vecadd,pV_vecadd);
	      timer_solver_precon.end();

	      // V_g+1 = A pV : 袖付きの pV を渡す (隣接ランクの値を SpMV の中で受け取る)
	      timer_solver_SpMV.start();
	      const PARCEL_INT g1 = g + 1;
	      if(g1 % 2 == 0){
		parcel::vector<Teven_basis> Vg1; vec_util.setptr( n, Veven, Vg1, (size_t)n*(g1/2) );
		parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Teven_basis,Tspmv>(one,zero,Mat,pV_vecadd,Vg1);
	      }else{
		parcel::vector<Todd_basis>  Vg1; vec_util.setptr( n, Vodd , Vg1, (size_t)n*(g1/2) );
		parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Todd_basis,Tspmv>(one,zero,Mat,pV_vecadd,Vg1);
	      }
	      timer_solver_SpMV.end();
	    }
	    que[0].wait();

	    // ---- グラム行列 → R (Q は作らない) ----
	    timer_solver_qr.start();
	    if(k == 0){
	      if(use_moment){
		// 4 本 (v_0, v_1, v_{s-1}, v_s) を VcholQR に集め、偶数番目の基底 (v_0, v_2, ..., v_s) との
		//   内積 4 x (s/2+1) 個からハンケル行列 G_ij = mu_{i+j} を作る (parcel_qr_choleskyqr.hpp)
		auto put = [&](PARCEL_INT g, PARCEL_INT slot){
		  parcel::vector<Teven_basis> Vs; vec_util.setptr( n, VcholQR, Vs, (size_t)n*slot );
		  if(g % 2 == 0){
		    parcel::vector<Teven_basis> Vg; vec_util.setptr( n, Veven, Vg, (size_t)n*(g/2) );
		    vec_util.convert<Teven_basis,Teven_basis>(Vg, Vs);
		  }else{
		    parcel::vector<Todd_basis>  Vg; vec_util.setptr( n, Vodd , Vg, (size_t)n*(g/2) );
		    vec_util.convert<Todd_basis,Teven_basis>(Vg, Vs);
		  }
		};
		put(0,0); put(1,1); put(s-1,2); put(s,3);
		parcel::vector<Teven_basis> Vbasis_QR; vec_util.setptr( (size_t)n*(s/2+1), Veven, Vbasis_QR, 0 );
		parcel::qr::qr<TcalcQR,Teven_basis,Teven_basis,THessenberg> parcel_qr_;
		parcel_qr_.set_qr_algo( parcel::qr::QR_CAGMRES_CHOLESKYQR );
		parcel_qr_.solver( parcel_blas, n, VcholQR, n, Vbasis_QR, n, Rdev, s+1, "N");
		que[0].wait();
		std::vector< parcel::scalar > Rh( Rdev.size() );
		Rdev.getValue(Rh);
		for(PARCEL_INT j = 0; j <= s; j++){
		  for(PARCEL_INT i = 0; i <= j; i++){
		    Rall[(size_t)i + (size_t)j*Nt] = (double)Rh[i + j*(s+1)].get_T();
		  }
		}
	      }else{
		gram_block<TcalcQR,THessenberg>(parcel_blas,
#if PARCEL_MPI
						comm->mpi_comm(),
#endif
						n, Veven, Vodd, 0, s, 0, s, Gdev, Gh, Nt,
						timer_solver_gram_dot, timer_solver_gram_comm);
		std::vector<double> S( (size_t)(s+1)*(s+1) );
		for(PARCEL_INT j = 0; j <= s; j++){
		  for(PARCEL_INT i = 0; i <= s; i++){ S[i + j*(s+1)] = Gh[(size_t)i + (size_t)j*Nt]; }
		}
		chol_upper(S, s+1, s+1, Rall, Nt, 0);
	      }
	      // update_HessenbergMatrix へ渡す R ((s+1)x(s+1), ld = s+1)
	      for(PARCEL_INT j = 0; j <= s; j++){
		for(PARCEL_INT i = 0; i <= s; i++){ Rmat[i + j*(s+1)] = Rall[(size_t)i + (size_t)j*Nt]; }
	      }
	    }else{
	      // 新しい s 本 W = v_{sk+1 .. sk+s} と既存の基底 V_old = v_{0 .. sk}
	      const PARCEL_INT m1 = base + 1;
	      timer_solver_reOrth.start();
	      gram_block<TcalcQR,THessenberg>(parcel_blas,
#if PARCEL_MPI
					      comm->mpi_comm(),
#endif
					      n, Veven, Vodd, 0, base, base+1, base+s, Gdev, Gh, Nt,
					      timer_solver_reOrth_dot, timer_solver_reOrth_comm);
	      timer_solver_reOrth.end();
	      gram_block<TcalcQR,THessenberg>(parcel_blas,
#if PARCEL_MPI
					      comm->mpi_comm(),
#endif
					      n, Veven, Vodd, base+1, base+s, base+1, base+s, Gdev, Gh, Nt,
					      timer_solver_gram_dot, timer_solver_gram_comm);
	      // R12 = R11^-T (V_old^T W) = Q_old^T W (ブロック古典 GS の係数。前進代入)
	      for(PARCEL_INT c = 0; c < s; c++){
		const size_t gc = (size_t)(base+1+c);
		for(PARCEL_INT i = 0; i < m1; i++){
		  double v = Gh[(size_t)i + gc*Nt];
		  for(PARCEL_INT j = 0; j < i; j++){ v -= Rall[(size_t)j + (size_t)i*Nt] * Rall[(size_t)j + gc*Nt]; }
		  Rall[(size_t)i + gc*Nt] = v / Rall[(size_t)i + (size_t)i*Nt];
		}
	      }
	      // R22 = chol(W^T W - R12^T R12) (射影後の W の R)
	      std::vector<double> S( (size_t)s*s );
	      for(PARCEL_INT c2 = 0; c2 < s; c2++){
		for(PARCEL_INT c1 = 0; c1 < s; c1++){
		  const size_t g1 = (size_t)(base+1+c1), g2 = (size_t)(base+1+c2);
		  double v = Gh[g1 + g2*Nt];
		  for(PARCEL_INT i = 0; i < m1; i++){ v -= Rall[(size_t)i + g1*Nt] * Rall[(size_t)i + g2*Nt]; }
		  S[c1 + c2*s] = v;
		}
	      }
	      chol_upper(S, s, s, Rall, Nt, base+1);
	      // update_HessenbergMatrix へ渡す形 (pcagmres と同じ):
	      //   BlockRmat (ld = s*k+1, s+1 列) = Q_old^T [q_{sk}, W] = [e_{sk}, R12]、Rmat (s x s, ld = s) = R22
	      const PARCEL_INT ldb = m1;
	      for(PARCEL_INT rr = 0; rr < ldb; rr++){ BlockRmat[rr] = (rr == base) ? 1.0 : 0.0; }
	      for(PARCEL_INT j = 0; j < s; j++){
		for(PARCEL_INT rr = 0; rr < ldb; rr++){
		  BlockRmat[rr + (j+1)*ldb] = Rall[(size_t)rr + (size_t)(base+1+j)*Nt];
		}
	      }
	      for(PARCEL_INT j = 0; j < s; j++){
		for(PARCEL_INT i = 0; i < s; i++){
		  Rmat[i + j*s] = Rall[(size_t)(base+1+i) + (size_t)(base+1+j)*Nt];
		}
	      }
	    }
	    timer_solver_qr.end();

	    // ブロック内の階数落ちの検出 (pcagmres と同じ。説明は parcel_krylov_cagmres.hpp の
	    //   cagmres_breakdown)。k = 0 : R は (s+1)x(s+1)、列 j が v_j。k > 0 : R22 は s x s、列 j が
	    //   v_{sk+1+j} で、前のブロックに取り除いた成分 (R12) も列ノルムに入れる。
	    //   R と R12 は全ランクで同じ値 (内積は allreduce 済み) なので判定も全ランクで一致する。
	    bool blk_bd = false, blk_offdiag_bad = false;
	    PARCEL_INT blk_g = 0;
	    {
	      const PARCEL_INT ns = (k == 0) ? s+1 : s;
	      std::vector<double> extra2(ns, 0.0);
	      if(k > 0){
		const PARCEL_INT ldb = s*k+1;
		for(PARCEL_INT j = 0; j < ns; j++){
		  for(PARCEL_INT rr = 0; rr < ldb; rr++){
		    const double v = (double)BlockRmat[rr + (j+1)*ldb].get_T();
		    extra2[j] += v*v;
		  }
		}
	      }
	      const PARCEL_INT jb = cagmres_breakdown::find_rank_deficient(
				      Rmat, ns, (k == 0) ? 1 : 0, extra2, tau, blk_offdiag_bad);
	      if(jb < ns){
		blk_bd = true;
		blk_g  = (k == 0) ? jb : s*k + 1 + jb;
		cagmres_breakdown::sanitize_R(Rmat, ns, jb, blk_offdiag_bad);
	      }
	    }

	    // !     ヘッセンベルグ行列の計算
	    update_HessenbergMatrix(
				   k,
				   t,
				   s,
				   Hmat, // [s*t+1,s*t]
				   Bmat, // [s+1,s]
				   Rmat, // [s+1,s+1]
				   BlockRmat, // [s*t+1,s+1]
				   hk);
	    if(blk_bd){
	      // 先頭 blk_g 列だけ残して、このリスタート周期の基底生成を打ち切る (以降の基底はゴミ)
	      //   使わない基底 (列 m_cyc 以降) の階数落ちは無視してよい。m_cap は m_cyc を超えない
	      m_cap = std::min(m_cyc, cagmres_breakdown::truncate_H(Hmat, Nt, st, blk_g, blk_offdiag_bad));
	      if(myrank == 0 && blk_g <= m_cyc){
		std::cout<<" pcagmres (Q = implicit) : rank-deficient basis block (k = "<<k<<"), Krylov subspace invariant after "
			 <<blk_g<<" basis vectors"<<std::endl;
	      }
	      break;
	    }
	  }

	  // 幸運な破綻の検出 (pcagmres と同じ。説明は parcel_krylov_cagmres.hpp)
	  //   破綻が無ければ m_eff = m_cyc (最後の周回で端数に減らしたとき以外は st)
	  PARCEL_INT m_eff = m_cyc;
	  {
	    const PARCEL_INT ldh = Nt;
	    bool bd = false;
	    for(PARCEL_INT i=0;i<m_cap && !bd;i++){   // 階数落ちがあれば先頭 m_cap 列まで
	      double cn2 = 0.0;
	      bool   fin = true;
	      for(PARCEL_INT rr = 0; rr <= i+1; rr++){
		const double h = (double)Hmat[rr + i*ldh].get_T();
		if(!std::isfinite(h)) fin = false;
		cn2 += h*h;
	      }
	      const double hn = std::fabs((double)Hmat[i+1 + i*ldh].get_T());
	      if(!fin){
		m_eff = i; bd = true;
	      }else if(hn == 0.0 || hn <= 1.0e-14*std::sqrt(cn2)){
		Hmat[i+1 + i*ldh] = 0.0;
		GivensRotation(i,t,s,Hmat,zeta,co,si);
		const double hd = (double)Hmat[i + i*ldh].get_T();
		m_eff = (hd != 0.0 && std::isfinite(hd)) ? i+1 : i;
		bd = true;
	      }else{
		GivensRotation(i,t,s,Hmat,zeta,co,si);
	      }
	    }
	    if(!bd && m_cap < m_cyc){ m_eff = m_cap; bd = true; }
	    if(bd && myrank == 0){
	      std::cout<<" pcagmres (Q = implicit) : Arnoldi breakdown, solve with "<<m_eff<<" of "
		       <<m_cyc<<" columns"<<std::endl;
	    }
	  }
	  if(m_eff == 0){
	    st_breakdown = true;
	    st_its = itr*st;
	    break;
	  }

	  // 解ベクトル更新 : x = x + K^-1 Q y、Q y = V (Rall^-1 y) (Q を陽に作らない)
	  timer_solver_update.start();
	  {
	    const PARCEL_INT me  = m_eff;
	    const PARCEL_INT ldh = Nt;
	    std::vector<double> y( me, 0.0 );
	    // y = H(0:me-1, 0:me-1)^-1 zeta (ギブンス回転後の H は上三角。後退代入)
	    for(PARCEL_INT i = me-1; i >= 0; i--){
	      double v = (double)zeta[i].get_T();
	      for(PARCEL_INT j = i+1; j < me; j++){ v -= (double)Hmat[i + j*ldh].get_T() * y[j]; }
	      y[i] = v / (double)Hmat[i + i*ldh].get_T();
	    }
	    // z = Rall(0:me-1, 0:me-1)^-1 y (後退代入。z は y に上書き)
	    for(PARCEL_INT i = me-1; i >= 0; i--){
	      double v = y[i];
	      for(PARCEL_INT j = i+1; j < me; j++){ v -= Rall[(size_t)i + (size_t)j*Nt] * y[j]; }
	      y[i] = v / Rall[(size_t)i + (size_t)i*Nt];
	    }
	    // w = V z
	    timer_solver_gemv.start();
	    combine_basis<Tvec>(parcel_blas, n, me, y, Veven, Vodd, w);
	    timer_solver_gemv.end();
	    // pV = K^-1 w、x = x + pV
	    precon_Mat.solver(w_vecadd,pV_vecadd);
	    que[0].wait();
	    parcel_blas.axpy<Tdot,Tdot,Tbasis,Tvec,Taxpy> (one,one,pV,x);
	  }
	  timer_solver_update.end();

	}
	que[0].wait();
	vec_util.convert<Tvec,Tvecx>(x,x0);
	que[0].wait();

	timer_solver_main.end();

	// 終了時の真の残差 || b - A x_final || (pcagmres と同じ。r を作業域に使う)
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	parcel_blas.dot<Tvec,Tvec,Tdot>(r,r);
	double finish_res_norm = 0.0;
	{
	  parcel::scalar rnorm_host_finish = parcel_blas.get_dot();
	  rnorm_host_finish = host_util.sqrt(rnorm_host_finish);
	  finish_res_norm = rnorm_host_finish.get_T()/bnorm_;
	  if(myrank == 0){
	    std::cout<<std::setw(8)<<" finish : residual true norm  || b - Ax || = ";
	    std::cout<<std::scientific << std::setprecision(10) << finish_res_norm <<std::endl;
	  }
	}
	timer_solver.end();

	if( myrank == 0){
	  std::cout<<std::endl<<" ----------------- "<<std::endl;
	  // 回帰試験の集計が拾う形 "pcagmres time = <値> [sec]" に合わせ、Q の作り方は後ろに書く
	  std::cout<<"pcagmres time              = "<<timer_solver.get()      <<" [sec]  (Q = implicit: not formed, CholeskyQR"
		   <<(qrdecomposition_algo_ == parcel::qr::QR_CAGMRES_CHOLESKYQR ? " moment" : "")<<")"<<std::endl;
	  std::cout<<"     init time             = "<<timer_solver_init.get() <<" [sec] "<<std::endl;
	  std::cout<<"          init precon      = "<<timer_solver_init_precon.get()  <<" [sec] "<<std::endl;
	  std::cout<<"          init convert     = "<<timer_solver_init_convert.get() <<" [sec] "<<std::endl;
	  std::cout<<"     main time             = "<<timer_solver_main.get()   <<" [sec] "<<std::endl;
	  std::cout<<"          main calc_res    = "<<timer_solver_calcres.get() <<" [sec]  calls = "<<timer_solver_calcres.get_count()<<std::endl;
	  std::cout<<"          main SpMV time   = "<<timer_solver_SpMV.get()   <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
	  std::cout<<"          main precon time = "<<timer_solver_precon.get() <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
	  std::cout<<"          main QR time     = "<<timer_solver_qr.get()   <<" [sec]  calls = "<<timer_solver_qr.get_count()<<"  (グラム行列 + コレスキー分解)"<<std::endl;
	  std::cout<<"          main reOrth time = "<<timer_solver_reOrth.get()   <<" [sec]  (QR の内数: 前のブロックとの内積)"<<std::endl;
	  // グラム行列の内訳。reOrth は reOrth の内数、gram は自己グラム (QR の内数)
	  std::cout<<"            (内) reOrth dot    = "<<timer_solver_reOrth_dot.get()<<" [sec]  calls = "<<timer_solver_reOrth_dot.get_count()
		   <<"  (局所内積。dot)"<<std::endl;
	  std::cout<<"            (内) reOrth allred = "<<timer_solver_reOrth_comm.get()<<" [sec]  calls = "<<timer_solver_reOrth_comm.get_count()
		   <<"  (ホストへの取り出し + MPI_Allreduce)"<<std::endl;
	  std::cout<<"          (内) QR gram dot     = "<<timer_solver_gram_dot.get()<<" [sec]  calls = "<<timer_solver_gram_dot.get_count()
		   <<"  (自己グラムの局所内積。dot、対称なので上三角だけ)"<<std::endl;
	  std::cout<<"          (内) QR gram allred  = "<<timer_solver_gram_comm.get()<<" [sec]  calls = "<<timer_solver_gram_comm.get_count()
		   <<"  (ホストへの取り出し + MPI_Allreduce)"<<std::endl;
	  std::cout<<"          main gemv time   = "<<timer_solver_gemv.get()   <<" [sec]  calls = "<<timer_solver_gemv.get_count()<<std::endl;
	  std::cout<<"          main update time = "<<timer_solver_update.get()   <<" [sec] "<<std::endl;
	}

	if(!st_converged && !st_nonfinite){
	  st_res = finish_res_norm;
	  if(!std::isfinite(st_res)) st_nonfinite = true;
	  else if(st_res < tol)      st_converged = true;
	}
	set_status(st_converged ? CONVERGED_RTOL
		   : (st_breakdown ? DIVERGED_BREAKDOWN : decide_reason(false, st_nonfinite)),
		   st_its, st_res, finish_res_norm);
	print_status(myrank);
	return ;
      }

    };

  } // namespace detail
  }
}

#endif

