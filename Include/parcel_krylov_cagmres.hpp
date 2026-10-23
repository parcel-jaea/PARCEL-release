// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_KRYLOV_CAGMRES_HPP_
#define PARCEL_KRYLOV_CAGMRES_HPP_

#include "parcel_qr.hpp"

// explicit Q (Qflag = "Y") の基底ブロック V は Q の中に直接作る (省メモリ)。
//   QR が QR_MGS / QR_CGS のとき、V (s+1 本、袖付き) を別に確保せず、ブロック k の V を
//   Q の列 s*k .. s*k+s に置く。V の列 0 は q_{sk} そのもので、列 1..s は QR でその場で
//   Q の列 s*k+1 .. s*k+s になる。MGS / CGS は列 i を作るとき列 0..i-1 の Q と列 i の V しか読まず、
//   列 i を規格化して書いた後は V の列 i を読まないので、V と Q が同じ領域でも演算の順序・種類は
//   変わらない。ブロック直交化 (BlockR = Q^T V と V -= Q R) は Q の列 0..s*k と V の列
//   (Q の列 s*k 以降) を読み書きし、書く側と読む側は重ならない。
//   QR_CHOLESKYQR は Q = V R^-1 を 1 回の gemm で作る (その場では書けない) ので V を別に確保する。
// 解の更新の作業ベクトル v0 (v0 = Q y、前処理の入力) は専用に確保しない。
//   Tvec と Tbasis が同じ型なら r の領域を使う。r は周回の先頭で r = b - Ax として作り、
//   V の列 0 を作った後は次の周回まで読まない。解の更新は周回の最後なので重ならない。
//   型が違うときは、V を別に確保していれば (Qflag = "Y") V の列 0 を使う (V は解の更新で読まず、
//   列 0 は次の周回の先頭で作り直す)。どちらも使えなければ専用に確保する。

namespace parcel{
  namespace krylov{

    // CA-GMRES の幸運な破綻の処理。
    //   CA-GMRES は s 本の基底 v_{i+1} = A K^-1 v_i をまとめて作ってから QR 分解する。
    //   ブロックの途中で Krylov 部分空間が不変になると、それ以降の基底は前の基底の一次結合になり、
    //   破綻は Arnoldi の h_{i+1,i} = 0 ではなく「R の対角 R_jj がほぼ 0 (階数落ち)」として現れる。
    //   このとき MGS/CGS は R_jj ~ 丸め誤差で割って以降の Q と R をゴミにし、CholeskyQR は
    //   sqrt(|負の値|) で同じくゴミを作る。そのまま H = R B R^-1 を組むと、有効なはずの先頭の列にも
    //   ゴミ (や 0 * inf = NaN) が混じり、x の更新が壊れる。
    //   そこで R の対角を列ノルムと比べて最初の階数落ちの列を見つけ、それより後ろを単位行列に
    //   置き換えてから H を組む。呼び出し側は h_{g,g-1} を 0 にして先頭 g 列だけで最小二乗を解く。
    namespace cagmres_breakdown {

      // 基底の精度の単位丸め誤差 (double-double は double の 2 乗、それ以外は基の型)
      template<class T>
      inline double unit_roundoff(){
	if constexpr (std::is_same<T, parcel_doubledouble>::value){
	  const double u = std::numeric_limits<double>::epsilon();
	  return u*u;
	}else if constexpr (std::is_same<T, parcel_float>::value){
	  return (double)std::numeric_limits<float>::epsilon();
	}else if constexpr (std::is_same<T, parcel_fp16>::value){
	  return 9.765625e-4; // 2^-10
	}else if constexpr (std::is_same<T, parcel_longdouble>::value){
	  return (double)std::numeric_limits<long double>::epsilon();
	}else{
	  return std::numeric_limits<double>::epsilon();
	}
      }

      // 階数落ちとみなす R_jj / ||v_j|| の閾値。
      //   QR の分解の精度で決まる部分は QR の計算 (TcalcQR) の u で決める。CholeskyQR は Gram 行列経由なので
      //   sqrt(u) 程度までしか R_jj を分解できない。
      //   基底を粗い精度 (Tbasis) で持つことによる揺らぎは 10 u_basis とし、大きい方を取る
      //   (基底の u に 1e3 を掛けると FP16 の基底では閾値が 1 近くになり、正常な列まで階数落ちと判定する)。
      template<class Tbasis, class TcalcQR>
      inline double tolerance(bool choleskyqr){
	const double uq = unit_roundoff<TcalcQR>();
	const double ub = unit_roundoff<Tbasis>();
	const double tq = choleskyqr ? 1.0e2*std::sqrt(uq) : 1.0e3*uq;
	return std::max(tq, 1.0e1*ub);
      }

      // R (ns x ns, 列優先, ld = ns) の列 j0.. を調べ、最初に階数落ちした列を返す (無ければ ns)。
      //   extra2[j] : ブロック直交化 (k > 0) で前のブロックの Q に取り除いた成分の 2 乗和。
      //   offdiag_bad : その列の非対角 (前の基底への射影) まで非有限なら true
      inline PARCEL_INT find_rank_deficient(std::vector< parcel::scalar > &R, PARCEL_INT ns,  // get_T が非 const のため参照
					    PARCEL_INT j0, const std::vector<double> &extra2,
					    double tau, bool &offdiag_bad){
	offdiag_bad = false;
	for(PARCEL_INT j = j0; j < ns; j++){
	  double cn2 = (j < (PARCEL_INT)extra2.size()) ? extra2[j] : 0.0;
	  bool   fin = std::isfinite(cn2);
	  for(PARCEL_INT r = 0; r < j; r++){
	    const double v = (double)R[r + j*ns].get_T();
	    if(!std::isfinite(v)) fin = false;
	    cn2 += v*v;
	  }
	  const double d = std::fabs((double)R[j + j*ns].get_T());
	  if(!fin){ offdiag_bad = true; return j; }
	  if(!std::isfinite(d)) return j;
	  cn2 += d*d;
	  if(d == 0.0 || d <= tau*std::sqrt(cn2)) return j;
	}
	return ns;
      }

      // 列 j 以降を単位行列に置き換える。列 j は前の基底への射影 (行 0..j-1) を残し、対角だけ 1 にする
      //   (本来 ~0 の対角は h_{g,g-1} にしか効かず、それは呼び出し側で 0 にする)。
      //   非対角も非有限なら列 j も単位列にする (呼び出し側はその列の H も使わない)。
      inline void sanitize_R(std::vector< parcel::scalar > &R, PARCEL_INT ns, PARCEL_INT j, bool offdiag_bad){
	for(PARCEL_INT c = j; c < ns; c++){
	  for(PARCEL_INT r = 0; r < ns; r++){
	    if(c == j && r < j && !offdiag_bad) continue;
	    R[r + c*ns] = (r == c) ? 1.0 : 0.0;
	  }
	}
      }

      // 破綻したブロックの後始末: 先頭 m_cap 列だけ有効にする。
      //   g = 階数落ちした基底の通し番号 (v_g が v_0..v_{g-1} の一次結合)。h_{g,g-1} = 0、列 g 以降は 0。
      //   射影まで壊れていたら列 g-1 も 0 にする (m_cap = g-1)。
      inline PARCEL_INT truncate_H(std::vector< parcel::scalar > &Hmat, PARCEL_INT ldh, PARCEL_INT ncol,
				   PARCEL_INT g, bool offdiag_bad){
	PARCEL_INT m_cap = offdiag_bad ? g-1 : g;
	if(m_cap < 0) m_cap = 0;
	for(PARCEL_INT c = m_cap; c < ncol; c++){
	  for(PARCEL_INT r = 0; r < ldh; r++){ Hmat[r + c*ldh] = 0.0; }
	}
	if(!offdiag_bad && g >= 1 && g < ldh){ Hmat[g + (g-1)*ldh] = 0.0; }
	return m_cap;
      }
    }

    // 直交基底 Q を陽に作るかどうか (pcagmres::set_explicit_q)。
    //   CAGMRES_Q_AUTO     (0, 既定) : QR_CHOLESKYQR かつ t == 1 のときだけ
    //                                  Q を作らない (Qflag = "N")。それ以外は Q を作る。
    //   CAGMRES_Q_EXPLICIT (1)       : 常に Q を作る (QR_CHOLESKYQR で t == 1 でも作る)。
    //   CAGMRES_Q_IMPLICIT (2)       : どの t でも Q を作らない CA-GMRES
    //                                  (parcel_krylov_choleskyqr_cagmres.hpp の detail::cagmres_implicit_q に委ねる)。
    //                                  QR は QR_CHOLESKYQR か QR_CAGMRES_CHOLESKYQR (モーメント法) だけ。
    //   不正な組合せ (IMPLICIT + QR_CGS / QR_MGS、QR_CAGMRES_CHOLESKYQR + AUTO / EXPLICIT) は
    //   solver() の入口で std::invalid_argument (設定関数は順不同で呼べるので、組合せは入口で調べる)。
    const PARCEL_INT CAGMRES_Q_AUTO     = 0;
    const PARCEL_INT CAGMRES_Q_EXPLICIT = 1;
    const PARCEL_INT CAGMRES_Q_IMPLICIT = 2;

    inline const char *cagmres_q_name(PARCEL_INT mode){
      switch(mode){
      case CAGMRES_Q_AUTO:     return "auto";
      case CAGMRES_Q_EXPLICIT: return "explicit";
      case CAGMRES_Q_IMPLICIT: return "implicit";
      default:                 return "?";
      }
    }

    namespace detail {
      // Q を陽に作らない CA-GMRES の本体 (公開しない。定義は parcel_krylov_choleskyqr_cagmres.hpp)
      template<class Tspmv, class Tprecon, class Taxpy, class Tdot> class cagmres_implicit_q;
    }

    // 計算精度を指定
    template<
      class Tspmv   = parcel_double, // SpMV 計算精度
      class Tprecon = parcel_double, // 前処理計算精度
      class Taxpy   = parcel_double, // ベクトル演算計算精度
      class Tdot    = parcel_double  // 縮約演算計算精度
      >
    class pcagmres : public krylov_param{

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
      // reOrth の内訳 (内積 / 集約 / 更新 V -= Q R)。reOrth の内数
      parcel::timer timer_solver_reOrth_dot;
      parcel::timer timer_solver_reOrth_comm;
      parcel::timer timer_solver_reOrth_update;
      parcel::timer timer_solver_update;
      
      parcel::vector_util vec_util;
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;
      // 既定値は s=8, t=4 (リスタート周期 32)。s=1 では通信削減が効かない。
      PARCEL_INT cagmres_s_ = 8;
      PARCEL_INT cagmres_t_ = 4;
      // 既定の QR は MGS (CGS は (s, t) = (8, 4) で最初の周期に残差が大きく増えることがある)
      PARCEL_INT qrdecomposition_algo_ = parcel::qr::QR_MGS;
      // set_qr_algo を呼んだか。IMPLICIT で未指定なら QR_CHOLESKYQR を使う (effective_qr_algo)
      bool       qr_algo_set_ = false;
      // Q を陽に作るか (CAGMRES_Q_*)。既定は AUTO
      PARCEL_INT explicit_q_ = CAGMRES_Q_AUTO;
      // CAGMRES_Q_IMPLICIT の本体。solver() のたびに設定を写して呼ぶ。
      //   メンバに持つのはタイマをオブジェクトごとに累積するため。
      detail::cagmres_implicit_q<Tspmv,Tprecon,Taxpy,Tdot> implicit_q_;

      // 実際に使う QR。IMPLICIT で set_qr_algo を呼んでいなければ QR_CHOLESKYQR
      //   (Q を作らない方式は CholeskyQR 系でしか成り立たないため。明示した CGS / MGS は下で例外)。
      PARCEL_INT effective_qr_algo() const {
	if(explicit_q_ == CAGMRES_Q_IMPLICIT && !qr_algo_set_) return parcel::qr::QR_CHOLESKYQR;
	return qrdecomposition_algo_;
      }

      // Q の作り方と QR の組合せの検査 (solver() の入口で呼ぶ。全ランクで同じ設定なので全ランクが投げる)
      void check_q_combination() const {
	const PARCEL_INT qa = effective_qr_algo();
	if(explicit_q_ == CAGMRES_Q_IMPLICIT &&
	   (qa == parcel::qr::QR_CGS || qa == parcel::qr::QR_MGS)){
	  throw std::invalid_argument(std::string("parcel::krylov::pcagmres::solver: CAGMRES_Q_IMPLICIT requires "
						  "QR_CHOLESKYQR or QR_CAGMRES_CHOLESKYQR (got ")
				      + (qa == parcel::qr::QR_CGS ? "QR_CGS" : "QR_MGS")
				      + "; call set_qr_algo(parcel::qr::QR_CHOLESKYQR))");
	}
	if(qa == parcel::qr::QR_CAGMRES_CHOLESKYQR && explicit_q_ != CAGMRES_Q_IMPLICIT){
	  throw std::invalid_argument(std::string("parcel::krylov::pcagmres::solver: QR_CAGMRES_CHOLESKYQR requires "
						  "set_explicit_q(CAGMRES_Q_IMPLICIT) (got CAGMRES_Q_")
				      + (explicit_q_ == CAGMRES_Q_AUTO ? "AUTO" : "EXPLICIT") + ")");
	}
      }

    public :

      // 範囲外の値は std::invalid_argument (set_restart の s, t は 1 以上)。
      //   QR_CAGMRES_CHOLESKYQR (モーメント法) は CAGMRES_Q_IMPLICIT 専用 (組合せは solver() の入口で検査する)
      void set_qr_algo(PARCEL_INT qrdecomposition_algo){
	if(qrdecomposition_algo != parcel::qr::QR_CGS &&
	   qrdecomposition_algo != parcel::qr::QR_MGS &&
	   qrdecomposition_algo != parcel::qr::QR_CHOLESKYQR &&
	   qrdecomposition_algo != parcel::qr::QR_CAGMRES_CHOLESKYQR){
	  throw std::invalid_argument("parcel::krylov::pcagmres::set_qr_algo: invalid value "
				      + std::to_string((long long)qrdecomposition_algo)
				      + " (expected QR_CGS, QR_MGS, QR_CHOLESKYQR or QR_CAGMRES_CHOLESKYQR)");
	}
	qrdecomposition_algo_ = qrdecomposition_algo;
	qr_algo_set_ = true;
	return ;
      }

      // Q を陽に作るか (CAGMRES_Q_AUTO / CAGMRES_Q_EXPLICIT / CAGMRES_Q_IMPLICIT)。
      //   範囲外の値はここで std::invalid_argument。QR との組合せは solver() の入口で検査する。
      void set_explicit_q(PARCEL_INT mode){
	if(mode != CAGMRES_Q_AUTO && mode != CAGMRES_Q_EXPLICIT && mode != CAGMRES_Q_IMPLICIT){
	  throw std::invalid_argument("parcel::krylov::pcagmres::set_explicit_q: invalid value "
				      + std::to_string((long long)mode)
				      + " (expected CAGMRES_Q_AUTO=0, CAGMRES_Q_EXPLICIT=1 or CAGMRES_Q_IMPLICIT=2)");
	}
	explicit_q_ = mode;
	return ;
      }
      PARCEL_INT get_explicit_q() const { return explicit_q_; }

      void set_restart(PARCEL_INT cagmres_s,PARCEL_INT cagmres_t){
	check_positive(cagmres_s, "pcagmres::set_restart (s)");
	check_positive(cagmres_t, "pcagmres::set_restart (t)");
	cagmres_s_ =  cagmres_s;
	cagmres_t_ =  cagmres_t;
	return ;
      }
      // 既定値の参照用 (C API の parcel_options_init がここから既定値を取る)
      PARCEL_INT get_restart_s() const { return cagmres_s_; }
      PARCEL_INT get_restart_t() const { return cagmres_t_; }
      PARCEL_INT get_qr_algo()   const { return effective_qr_algo(); }   // 実際に使う値
      pcagmres(){}
      ~pcagmres(){}

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

      template<
	class Tmat,  // 係数行列データ精度
	class Tvec, // クリロフ部分空間法内部データ精度	
	class Tprecon_mat, // 前処理行列データ精度
	class Tprecon_vec, // 前処ベクトルデータ精度
	class Tbasis // 基底ベクトルvのデータ精度
	>
      void update_xvector(
#if PARCEL_MPI
			  MPI_Comm mpi_comm, // 行列のコミュニケータ (Mat.comm()->mpi_comm())
			  PARCEL_INT n,
#endif			  
			  PARCEL_INT cagmres_t_,
			  PARCEL_INT cagmres_s_,
			  parcel::precon_matrix<
			  Tmat,
			  Tbasis,
			  Tprecon_mat,
			  Tprecon_vec,
			  Tprecon
			  >  &precon_Mat,		  
			  parcel::vector<Tbasis> &V,
			  parcel::vector<Tbasis> &Q,
#if PARCEL_MPI
			  parcel::vector<Tbasis> &v0_vecadd,			  
			  parcel::vector<Tbasis> &prev_vecadd,			  
			  parcel::vector<Tvec>   &x_vecadd,
#else			  
			  parcel::vector<Tbasis> &v0,			  
			  parcel::vector<Tbasis> &prev,			  
			  parcel::vector<Tvec>   &x,
#endif			  
			  std::vector< parcel::scalar > &H_UpperTri, // [s*t+1,s*t]
			  std::vector< parcel::scalar > &R_UpperTri, // [s+1,s+1]
			  std::vector< parcel::scalar > &zeta, // [s*t+1]
			  std::string Qflag = "Y",
			  // 使う列数 (幸運な破綻で途中まで。負なら s*t 全部)
			  PARCEL_INT m_eff = -1
			  ){

	// 解ベクトルの更新(リスタート周期になるまでループを抜けない実装)
	
#if PARCEL_MPI
	sycl::queue *que = x_vecadd.queue();
#else
	sycl::queue *que = x.queue();
#endif
	
#if PARCEL_MPI
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,mpi_comm);
#else
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
#endif
	
	PARCEL_INT t = cagmres_t_;
	PARCEL_INT s = cagmres_s_;
	const PARCEL_INT me = (m_eff >= 0 && m_eff < s*t) ? m_eff : s*t;
	std::vector< parcel::scalar > Hinv_UpperTri( me*me );
	for(PARCEL_INT i=0;i<me;i++){
	  for(PARCEL_INT j=0;j<me;j++){
	    Hinv_UpperTri[j+i*me] = H_UpperTri[j+i*(s*t+1)];
	  }
	}
	std::string uplo("U");
	std::string diag("N");

	// 上三角の H の逆行列をホストで trtri で求め、y = H^-1 zeta を作る
	parcel_blas_host.trtri(
			uplo,
			diag,
			Hinv_UpperTri,
			me,me
			);

	std::vector< parcel::scalar > y( ((s*t)+1),0.0);
	for(PARCEL_INT i=0;i<me;i++){
	  parcel::scalar  y_ = 0.0;
	  for(PARCEL_INT j=0;j<me;j++){
	    parcel::scalar Hiv_ij = Hinv_UpperTri[i+j*me];
	    parcel::scalar e_j    = zeta[j];
	    y_ = y_ + Hiv_ij * e_j;
	  }
	  y[i] = y_;
	}
	// 直交化行列を陽に計算しない最適化(t==1)
	// Qy = V(R^-1y) = Vz : z = R^-1y --> Rz = y

	if( Qflag == "N" ) {
	  // 後方代入により逆行列を掛ける
	  for(PARCEL_INT i=me-1;i>-1;i--){	    
	    parcel::scalar yrinv_ = y[i]; 
	    for(PARCEL_INT j=i+1;j<me;j++){
	      yrinv_ = yrinv_ - y[j] * R_UpperTri[i+j*(s+1)];
	    }
	    y[i] = yrinv_ / R_UpperTri[ i + i*(s+1) ];
	  }
	}

	// yをデバイス側に転送
	parcel::vector<Tvec> y_;y_.resize(me,que);y_.setValue(y);
	que[0].wait();

	// gemv計算
	parcel::vector<Tdot> one;one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);
	parcel::vector<Tdot> zero;zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);
	// v0 = v[][]*y
	timer_solver_gemv.start();
#if PARCEL_MPI
 	parcel_blas.gemv<Tdot,Tdot,Tbasis,Tvec,Tbasis,Taxpy>(
							     me,one,zero,
							     Q,
							     y_,
							     v0_vecadd
							     );
#else	
	parcel_blas.gemv<Tdot,Tdot,Tbasis,Tvec,Tbasis,Taxpy>(me,one,zero,Q,y_,v0);
#endif	
	timer_solver_gemv.end();

	que[0].wait();
	// 前処理計算
	// Mx0 = v0
	// (timer_solver_precon には入れない: loop update の内数として数える)
#if PARCEL_MPI
	precon_Mat.solver(v0_vecadd,prev_vecadd);
#else
	precon_Mat.solver(v0,prev);
#endif

	que[0].wait();
	// 解ベクトルの更新 : x = x + x0
#if PARCEL_MPI
	parcel::vector<Tvec>      x;vec_util.setptr( n, x_vecadd   , x   , 0);    	
	parcel::vector<Tbasis> prev;vec_util.setptr( n, prev_vecadd, prev, 0);	
	parcel_blas.axpy<Tdot,Tdot,Tbasis,Tvec,Taxpy> (one,one,prev,x);
#else
	parcel_blas.axpy<Tdot,Tdot,Tbasis,Tvec,Taxpy> (one,one,prev,x);
#endif
	// ------------------------------------------------------------------

	return ;	
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

      // 前処理,IR等の設定をclassで定義する
      template<
	class Tmat,  // 係数行列データ精度
	class Tvecx, // 入力ベクトルデータ精度
	class Tvecb, // 右辺ベクトルデータ精度
	class Tvec = parcel_double, // クリロフ部分空間法内部データ精度
	class Tprecon_mat = parcel_double, // 前処理行列データ精度
	class Tprecon_vec = parcel_double, // 前処ベクトルデータ精度
	class Tbasis = parcel_double, // 基底ベクトルのデータ精度
	class TcalcQR = parcel_double, // QR分解計算精度
	class THessenberg = parcel_double, // ヘッセンベルグ行列の計算精度
	// 以下 2 つは CAGMRES_Q_IMPLICIT のときだけ使う (それ以外では無視)
	class Todd_basis  = Tbasis, // 奇数番目の基底ベクトルの格納精度
	class Teven_basis = Tbasis  // 偶数番目の基底ベクトルの格納精度
	>
      void solver(
		  parcel::matrix<Tmat>  &Mat,
		  parcel::vector<Tvecx> &x0,
		  parcel::vector<Tvecb> &b0
		  ){

	// 設定の組合せの検査 (設定関数は順不同で呼べるので入口で見る)
	check_q_combination();

	// CAGMRES_Q_IMPLICIT は detail::cagmres_implicit_q に委ねる。
	//   設定 (tol / 反復上限 / s,t / QR / 前処理) を写して解き、収束状態を写し戻す。
	if(explicit_q_ == CAGMRES_Q_IMPLICIT){
	  reset_status();
	  implicit_q_.set_tolerance(get_tolerance());
	  implicit_q_.set_max_iteration(get_max_iteration());
	  implicit_q_.set_restart(cagmres_s_, cagmres_t_);
	  implicit_q_.set_qr_algo(effective_qr_algo());
	  copy_precon_param(*get_precon_param(), *implicit_q_.get_precon_param());
	  implicit_q_.template solver<Tmat,Tvecx,Tvecb,Tvec,Tprecon_mat,Tprecon_vec,Tbasis,
				      TcalcQR,THessenberg,Todd_basis,Teven_basis>(Mat,x0,b0);
	  set_status(implicit_q_.get_converged_reason(), implicit_q_.get_iterations(),
		     implicit_q_.get_residual(), implicit_q_.get_true_residual());
	  return ;
	}

	sycl::queue *que   = Mat.queue();


	require_in_order(que, "krylov::pcagmres");   // in-order キューが前提
	const PARCEL_INT n = x0.size();
	reset_status();
#if PARCEL_MPI
	parcel::comm *comm = Mat.comm();
	int myrank   = comm->rank();
	int nprocess = comm->size();
	comm->barrier();
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que,comm->mpi_comm());
	PARCEL_INT nadd = Mat.get_recv_size();
#else
	parcel::blas<PARCEL_MAX_FLOTING_TYPE> parcel_blas(que);
	int myrank = 0;
	int nprocess = 1;
	PARCEL_INT nadd = 0;
#endif

	parcel::qr::qr<TcalcQR,Tbasis,Tbasis,THessenberg> parcel_qr;
	parcel_qr.set_qr_algo(qrdecomposition_algo_);

	timer_solver.start();
	timer_solver_init.start();

	// ------------------------------
	// Init Preconditioning
	timer_solver_init_precon.start();

	const double tol = get_tolerance();
	const PARCEL_INT itrmax = get_max_iteration();
	precon_param &precon = *(get_precon_param());
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
#else	
	parcel::vector<Tvec> x(n,que); parcel_blas.zero_init<Tvec>(x);
	vec_util.convert<Tvecx,Tvec>(x0,x);
	parcel::vector<Tvec> b(n,que); parcel_blas.zero_init<Tvec>(b);
	vec_util.convert<Tvecb,Tvec>(b0,b);
#endif	
 	que[0].wait();
	timer_solver_init_convert.end();

	// ! ------------------------------------------------------------
	// ! CACAGMRES のメインループ
	// ! do itr = 1,itrmax,s*t
	// !   r = b-Ax
	// !   do k = 1,t
	// !     基底ベクトルの生成
	// !     do i=1,s
	// !       solve Ms=V_i
	// !       Newton基底,単基底
	// !       V_i+1=F(As)
	// !     enddo
	// !     if(k.eq.1)then
	// !       QR分解
	// !     else
	// !       ブロック古典グラムシュミット直交化
	// !       QR分解
	// !     endif
	// !     ヘッセンベルグ行列の計算
	// !     ギブンス回転
	// !     if(||r||<rmax) ;exit
	// !   enddo
	// !   解ベクトル更新
	// !   if(||r||<rmax) ;exit
	// ! enddo
	// ! ------------------------------------------------------------
	std::string Qflag = "Y";
	// V を Q の中に置くか (先頭の説明)。
	//   Qflag = "N" (AUTO・t == 1・CholeskyQR) は V と Q がもともと同じ領域なので対象外
	const bool v_in_q = !( (explicit_q_ == CAGMRES_Q_AUTO) && (cagmres_t_ == 1) && ( parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR) )
	  && ( parcel_qr.get_qr_algo() == qr::QR_MGS || parcel_qr.get_qr_algo() == qr::QR_CGS );

#if PARCEL_MPI
	parcel::parallel_vector<Tvec> r_parallel(n,nadd,1,que);parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel::parallel_vector<Tbasis> QV_parallel(1,0,1,que);
	parcel::parallel_vector<Tbasis> Q_parallel(1,0,1,que);
	parcel::parallel_vector<Tbasis> V_parallel(1,0,1,que);
	
	parcel::vector<Tbasis> QV;
	parcel::vector<Tbasis> Q;
	parcel::vector<Tbasis> V;
	// CAGMRES_Q_EXPLICIT では t == 1 の CholeskyQR でも Q を作る
	if( (explicit_q_ == CAGMRES_Q_AUTO) &&
	    (cagmres_t_ == 1) && ( parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR) ){
	  Qflag = "N"; // 直交行列Qを陽に計算しない実装
	  QV_parallel.resize(n,nadd,(cagmres_s_*cagmres_t_+1),que);
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_*cagmres_t_+1), QV_parallel.vec(0), QV, 0);
	  parcel_blas.zero_init<Tbasis>(QV);
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_*cagmres_t_+1), QV, V, 0 );
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1)           , QV, Q, (size_t)(n+nadd)*(cagmres_s_*(cagmres_t_-1)) );
	}else if(v_in_q){
	  // V は Q の中 (先頭の説明)。ブロック k ごとに位置を付け直す
	  Q_parallel.resize(n,nadd,(cagmres_s_*cagmres_t_+1),que);
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_*cagmres_t_+1), Q_parallel.vec(0), Q, 0);
  	  parcel_blas.zero_init<Tbasis>(Q);
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1)           , Q, V, 0);
	}else{
	  V_parallel.resize(n,nadd,(cagmres_s_+1),que);
	  Q_parallel.resize(n,nadd,(cagmres_s_*cagmres_t_+1),que);
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1)           , V_parallel.vec(0), V, 0);	  
	  vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_*cagmres_t_+1), Q_parallel.vec(0), Q, 0);
	  parcel_blas.zero_init<Tbasis>(V);
  	  parcel_blas.zero_init<Tbasis>(Q);
	}
	// v0 は r (同じ型のとき) か V の列 0 (先頭の説明)
	parcel::parallel_vector<Tbasis> v0_parallel(1,0,1,que);
	parcel::vector<Tbasis> v0;
	if constexpr (std::is_same<Tvec,Tbasis>::value){
	  vec_util.setptr( n, r_parallel.vec(0), v0, 0);
	}else{
	  if(Qflag == "Y" && !v_in_q){
	    vec_util.setptr( n, V_parallel.vec(0), v0, 0);
	  }else{
	    v0_parallel.resize(n,nadd,1,que);
	    vec_util.setptr( n, v0_parallel.vec(0), v0, 0) ;parcel_blas.zero_init<Tbasis>(v0);
	  }
	}
	parcel::parallel_vector<Tbasis> pV_parallel(n,nadd,1,que);
	parcel::vector<Tbasis> pV; vec_util.setptr( n, pV_parallel.vec(0), pV, 0) ;parcel_blas.zero_init<Tbasis>(pV);

#else	
	parcel::vector<Tvec> r(n,que);parcel_blas.zero_init<Tvec>(r);
	parcel::vector<Tbasis> QV;
	parcel::vector<Tbasis> Q;
	parcel::vector<Tbasis> V;	
	// CAGMRES_Q_EXPLICIT では t == 1 の CholeskyQR でも Q を作る
	if( (explicit_q_ == CAGMRES_Q_AUTO) &&
	    (cagmres_t_ == 1) && ( parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR) ){
	  Qflag = "N"; // 直交行列Qを陽に計算しない実装
	  QV.resize( (size_t)n*(cagmres_s_*cagmres_t_+1),que);parcel_blas.zero_init<Tbasis>(QV);
	  vec_util.setptr( (size_t)n*(cagmres_s_*cagmres_t_+1), QV, Q, 0);
	  vec_util.setptr( (size_t)n*(cagmres_s_+1)           , QV, V,  (size_t)n*(cagmres_s_*(cagmres_t_-1)) );
	}else if(v_in_q){
	  // V は Q の中 (先頭の説明)
	  QV.resize(1,que);
	  Q.resize( (size_t)n*(cagmres_s_*cagmres_t_+1),que);parcel_blas.zero_init<Tbasis>(Q);
	  vec_util.setptr( (size_t)n*(cagmres_s_+1), Q, V, 0 );
	}else{
	  QV.resize(1,que);
	  V.resize( (size_t)n*(cagmres_s_+1),que);parcel_blas.zero_init<Tbasis>(V);
	  Q.resize( (size_t)n*(cagmres_s_*cagmres_t_+1),que);parcel_blas.zero_init<Tbasis>(Q);
	}
	// v0 は r (同じ型のとき) か V の列 0 (先頭の説明)
	parcel::vector<Tbasis> v0;
	if constexpr (std::is_same<Tvec,Tbasis>::value){
	  vec_util.setptr( n, r, v0, 0);
	}else{
	  if(Qflag == "Y" && !v_in_q){
	    vec_util.setptr( n, V, v0, 0);
	  }else{
	    v0.resize(n,que);parcel_blas.zero_init<Tbasis>(v0);
	  }
	}
	parcel::vector<Tbasis> pV(n,que);parcel_blas.zero_init<Tbasis>(pV);
#endif

	parcel::vector<THessenberg> R((cagmres_s_+1)*(cagmres_s_+1), que);parcel_blas.zero_init<THessenberg>(R);
	parcel::vector<THessenberg> BlockR( (cagmres_s_*cagmres_t_+1)*(cagmres_s_+1) , que);parcel_blas.zero_init<THessenberg>(BlockR);
	
	parcel::vector<Tdot> one;
	one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);

	parcel::vector<Tdot> zero;
	zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);

	parcel::vector<Tdot> negative_one;
	negative_one.resize(1,que);
	double negative_one_ = -1.0;negative_one.setValue(&negative_one_);

	// --------------------------------------------
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

	// ||b|| = 0 なら解は x = 0 (0/0 = NaN や 1/||r|| = inf にしないため反復しない)。
	//   全ランクで同じ分岐 (bnorm は allreduce 済み)。
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
	
	// V0 = r/||r||
	{	  
	  parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(1.0/rnorm_);
	  parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	  rnorm_inv.setValue(rnorm_inv_host); 
#if PARCEL_MPI
 	  parcel::vector<Tbasis> V0;vec_util.setptr( n , V, V0,0); 	    	  
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,V0);
#else	  
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,V);
#endif	  
	}
	timer_solver_init.end();

	// 収束状態 (krylov_param::set_status)。判定に使うのは周回先頭の真の残差
	//   反復回数は基底ベクトルの総数 (表示のカウンタ itr*s*t と同じ数え方)
	bool   st_converged = false;
	bool   st_nonfinite = false;
	bool   st_breakdown = false;   // 最初の列で H が特異になり x を更新できなかった
	// 反復の上限 itrmax は s*t の倍数でなくてよい (倍数に切り捨てない。itrmax < s*t でも itrmax 反復する)。
	//   周回数を ceil(itrmax/(s*t)) にし、最後の周回は m_cyc = itrmax - itr*s*t 列だけを使う:
	//     基底は t_cyc = ceil(m_cyc/s) ブロックだけ作る (ブロックの中は常に s ステップ。行列べき乗・QR・
	//     update_HessenbergMatrix は s 本単位のまま)。H の先頭 m_cyc 列 (と Q の先頭 m_cyc+1 本) は
	//     m_cyc 反復の Arnoldi 関係そのものなので、ギブンス回転と最小二乗を先頭 m_cyc 列で解けば
	//     m_cyc 反復の GMRES の解になる (幸運な破綻で列を減らすのと同じ仕組み)。
	//     最後のブロックの残り t_cyc*s - m_cyc (< s) 本の基底は作るが使わない (SpMV・前処理が最大 s-1 回余分)。
	//   上限まで回ったときの反復回数 (its) は itrmax ちょうど。
	//   itrmax が s*t の倍数なら n_cycle = itrmax/(s*t)、m_cyc = s*t、t_cyc = t。
	const PARCEL_INT st_len  = cagmres_s_ * cagmres_t_;
	const PARCEL_INT n_cycle = itrmax/st_len + ((itrmax % st_len) != 0 ? 1 : 0);
	PARCEL_INT st_its   = itrmax;   // 上限まで回ったときの値
	double st_res       = std::numeric_limits<double>::quiet_NaN();
	timer_solver_main.start();
	for(PARCEL_INT itr = 0;itr < n_cycle;itr++){
	  // ------------
	  std::vector< parcel::scalar > Bmat( (cagmres_s_+1)*cagmres_s_ );
	  std::vector< parcel::scalar > Rmat( (cagmres_s_+1)*(cagmres_s_+1) );
	  std::vector< parcel::scalar > BlockRmat( (cagmres_s_*cagmres_t_+1)*(cagmres_s_+1) );
	  std::vector< parcel::scalar > Hmat( (cagmres_s_*cagmres_t_+1)*cagmres_s_*cagmres_t_ );	  
	  // ! 基底変換行列を作成
	  // 単基底のみ対応 :
	  for(PARCEL_INT i = 0;i<Bmat.size() ;i++){Bmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<cagmres_s_;i++){
	    Bmat[ (i+1) + i*(cagmres_s_+1) ] = 1.0;
	  }

	  std::vector< parcel::scalar > zeta(cagmres_s_ * cagmres_t_ + 1);
	  std::vector< parcel::scalar > co(cagmres_s_ * cagmres_t_ + 1);
	  std::vector< parcel::scalar > si(cagmres_s_ * cagmres_t_ + 1);
	  parcel::scalar hk;hk = 0.0;
	  for(PARCEL_INT i = 0;i<co.size();i++   ){co[i]   = 0.0;}
	  for(PARCEL_INT i = 0;i<si.size();i++   ){si[i]   = 0.0;}
	  for(PARCEL_INT i = 0;i<zeta.size() ;i++){zeta[i] = 0.0;}

	  for(PARCEL_INT i = 0;i<Hmat.size() ;i++){Hmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<Rmat.size() ;i++){Rmat[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = 0.0;}

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
	    if( myrank == 0){
	      std::cout<<std::setw(8)<< itr*cagmres_s_ * cagmres_t_ <<" : residual norm = ";	      
	      std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	    }
	    st_res = rnorm_FP64/bnorm_FP64;
	    // 非有限になったらその場で抜ける
	    if(!std::isfinite(st_res)){ st_nonfinite = true; st_its = itr*cagmres_s_ * cagmres_t_; break; }
	    if((rnorm_/bnorm_)<tol){ st_converged = true; st_its = itr*cagmres_s_ * cagmres_t_; break; }
	  }
	  zeta[0] =  rnorm_;
	  // V を Q の中に置くときは、ブロック 0 の位置 (Q の列 0..s) に戻す
	  if(v_in_q){
#if PARCEL_MPI
	    vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1), Q, V, 0 );
#else
	    vec_util.setptr( (size_t)n*(cagmres_s_+1), Q, V, 0 );
#endif
	  }
	  // V0 = r / || r ||
	  {
	    parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(1.0/rnorm_);
	    parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	    rnorm_inv.setValue(rnorm_inv_host); 
	    timer_solver_axpy.start();
	    parcel::vector<Tbasis> Vi0;
	    vec_util.setptr(n, V, Vi0, 0);
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tbasis,Taxpy> (rnorm_inv,zero,r,Vi0);
	    timer_solver_axpy.end();
	  }
	  // この周回で使う列数 m_cyc と作るブロック数 t_cyc (最後の周回だけ上限の端数に減らす)
	  const PARCEL_INT m_cyc = std::min(st_len, itrmax - itr*st_len);
	  const PARCEL_INT t_cyc = (m_cyc + cagmres_s_ - 1)/cagmres_s_;
	  // ブロック内の階数落ち (幸運な破綻) で有効な H の列数。無ければ m_cyc
	  //   (最後の周回で端数に減らしたとき以外は s*t)
	  PARCEL_INT m_cap = m_cyc;
	  for(PARCEL_INT k = 0;k < t_cyc;k++){
	    // V を Q の中に置くときは、ブロック k の V を Q の列 s*k .. s*k+s にする
	    //   (列 0 が q_{sk} そのものになるので、下の q_{sk} の複写は要らない)
	    if(v_in_q){
#if PARCEL_MPI
	      vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1), Q, V, (size_t)(n+nadd)*(cagmres_s_*k) );
#else
	      vec_util.setptr( (size_t)n*(cagmres_s_+1), Q, V, (size_t)n*(cagmres_s_*k) );
#endif
	    }
	    
	    // 基底変換行列の作成(未対応)
	    for(PARCEL_INT i = 0;i < cagmres_s_;i++){
	      // 前処理
	      parcel::vector<Tbasis> Vi0;
#if PARCEL_MPI
	      vec_util.setptr(n, V, Vi0, (size_t)(n+nadd)*i);
#else
	      vec_util.setptr(n, V, Vi0, (size_t)n*i);
#endif	      
	      if( i==0 && k>0 && !v_in_q){
		timer_solver_axpy.start();
		parcel::vector<Tbasis> Qi0;
#if PARCEL_MPI
		vec_util.setptr( n+nadd, Q, Qi0, (size_t)(n+nadd)*(cagmres_s_*k) );
		parcel_blas.axpy<Tdot,Tdot,Tbasis,Tbasis,Tvec> (one,zero,Qi0,Vi0);
#else		
		vec_util.setptr( n, Q, Qi0, (size_t)n*(cagmres_s_*k) );
		parcel_blas.axpy<Tdot,Tdot,Tbasis,Tbasis,Tvec> (one,zero,Qi0,Vi0);
#endif		
		timer_solver_axpy.end();
	      }
	      
	      timer_solver_precon.start();
	      
#if PARCEL_MPI
	      parcel::vector<Tbasis> Vi0_vecadd;vec_util.setptr( (n+nadd), V, Vi0_vecadd, (size_t)(n+nadd)*i); 
	      parcel::vector<Tbasis> pV_vecadd ;vec_util.setptr( (n+nadd), pV, pV_vecadd, 0); 
	      precon_Mat.solver(Vi0_vecadd,pV_vecadd);
#else	      
	      precon_Mat.solver(Vi0,pV);
#endif	      
	      timer_solver_precon.end();
	      
	      // V_i+1 = ApV : 単基底のみ対応
	      timer_solver_SpMV.start();
#if PARCEL_MPI
	      parcel::vector<Tbasis> Vi1;vec_util.setptr( n, V, Vi1, (size_t)(n+nadd)*(i+1) ); 
	      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Tbasis,Tspmv>(one,zero,Mat,pV_vecadd,Vi1);
#else	      
	      parcel::vector<Tbasis> Vi1;
	      vec_util.setptr(n, V, Vi1, (size_t)n*(i+1));
	      parcel_blas.SpMV<Tdot,Tdot,Tmat,Tbasis,Tbasis,Tspmv>(one,zero,Mat,pV,Vi1);
#endif	      
	      timer_solver_SpMV.end();	      	      
	    }
	    
	    timer_solver_reOrth.start();
	    if( k > 0){
	      // ブロック古典グラムシュミット直交化
	      // R_' = Q_k-1 * Vk
	      // GEMM
	      
	      for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = 0.0;}
	      // R_' = Q^T V (mR x nR、列優先、ld = mR)。列 0 は v_0 = q_{sk} の写しとの内積 (H の組み立てで使う)
	      const PARCEL_INT mR = cagmres_s_*k+1;
	      const PARCEL_INT nR = cagmres_s_+1;
#if PARCEL_MPI
	      const size_t ldQV = (size_t)(n+nadd);
#else
	      const size_t ldQV = (size_t)n;
#endif
	      // 局所の dot (allreduce しない版) を mR x nR 回呼び、集約は allreduce 1 回にまとめる
	      //   (ks_arnoldi のブロック直交化と同じ形)。出力 1 要素 1 スレッドの gemmTN は起動スレッドが
	      //   mR x nR 個しかなく、局所行数を逐次に足すため GPU で遅い。
	      //   計算精度は THessenberg。値は THessenberg の BlockR に置く。
	      std::vector< parcel::scalar > BlockR_loc( (size_t)mR*nR, 0.0 );
	      timer_solver_reOrth_dot.start();
	      for(PARCEL_INT j = 0;j < nR;j++){
		parcel::vector<Tbasis> Vj; vec_util.setptr( n, V, Vj, ldQV*j );
		for(PARCEL_INT i = 0;i < mR;i++){
		  parcel::vector<Tbasis> Qi; vec_util.setptr( n, Q, Qi, ldQV*i );
		  parcel_blas.dot<Tbasis,Tbasis,THessenberg>(Qi, Vj, false);
		  BlockR_loc[(size_t)i + (size_t)j*mR] = parcel_blas.get_dot();
		}
	      }
	      timer_solver_reOrth_dot.end();
	      timer_solver_reOrth_comm.start();
#if PARCEL_MPI
	      parcel::vector<THessenberg> BlockR_( (size_t)mR*nR , que);
	      BlockR_.setValue( BlockR_loc );
	      comm->allreduce<THessenberg,THessenberg>(BlockR_, BlockR, mR*nR, MPI_SUM);
#else
	      {
		parcel::vector<THessenberg> BlockR_v; vec_util.setptr( (size_t)mR*nR, BlockR, BlockR_v, 0 );
		BlockR_v.setValue( BlockR_loc );
	      }
#endif
	      
	      std::vector< parcel::scalar > BlockRmat_host(BlockRmat.size(),0.0);
	      BlockR.getValue( BlockRmat_host );
	      for(PARCEL_INT i = 0;i<BlockRmat.size() ;i++){BlockRmat[i] = BlockRmat_host[i];}
	      timer_solver_reOrth_comm.end();
	      // V(:, 1..s) -= Q R_'(:, 1..s)
	      //   縦長 gemm (tall = "B"、mixed_precision_gemm.hpp の tgemm_beta) で Q の各行を 1 回だけ読む。
	      //   出力 1 要素の足し算の順序は既定の gemm と同じ。Tcalc が dd のときは既定の gemm (ddgemm)。
	      timer_solver_reOrth_update.start();
	      {
		parcel::vector<Tbasis> Qblock;
		parcel::vector<Tbasis> Vblock;
		parcel::vector<THessenberg> Rblock;
		vec_util.setptr( ldQV*mR          , Q, Qblock, 0 );
		vec_util.setptr( ldQV*(cagmres_s_), V, Vblock, ldQV );
		vec_util.setptr( (size_t)mR*(cagmres_s_), BlockR, Rblock, mR );
		parcel_blas.gemm<Tdot,Tdot,Tbasis,THessenberg,Tbasis,THessenberg>(
										  n, // m
										  cagmres_s_, // n
										  mR, // k
										  negative_one,one,
										  Qblock,ldQV, 
										  Rblock,mR,
										  Vblock,ldQV,
										  "N","N","N","N","B");
	      }
	      timer_solver_reOrth_update.end();
	    }
	    timer_solver_reOrth.end();
	    
	    {
	      parcel::vector<Tbasis> Vbasis;
	      parcel::vector<Tbasis> Qbasis;
	      timer_solver_qr.start();
#if PARCEL_MPI
	      if(k==0){
		vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1), V, Vbasis, 0 );
		vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_+1), Q, Qbasis, 0 );
	      }else{
		vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_)  , V, Vbasis, (n+nadd) );
		vec_util.setptr( (size_t)(n+nadd)*(cagmres_s_)  , Q, Qbasis, (size_t)(n+nadd)*( cagmres_s_*k + 1 ) );
	      }
	      // QR分解 : s+1本のベクトルを直交化 
	      parcel_qr.solver(
			       parcel_blas, n,
			       Vbasis ,(n+nadd),
			       Qbasis ,(n+nadd),
			       R,(cagmres_s_+1),
			       Qflag);
#else
	      if(k==0){
		vec_util.setptr( (size_t)n*(cagmres_s_+1), V, Vbasis, 0 );
		vec_util.setptr( (size_t)n*(cagmres_s_+1), Q, Qbasis, 0 );
	      }else{
		vec_util.setptr( (size_t)n*(cagmres_s_)  , V, Vbasis, n );
		vec_util.setptr( (size_t)n*(cagmres_s_)  , Q, Qbasis, (size_t)n*( cagmres_s_*k + 1 ) );
	      }
	      // QR分解 : s+1本のベクトルを直交化
	      parcel_qr.solver( parcel_blas, n,
				Vbasis ,n,
				Qbasis,n,
				R,(cagmres_s_+1),
				Qflag);
#endif
	      
	      timer_solver_qr.end();
	    }
	    que[0].wait();

	    
	    std::vector< parcel::scalar > Rmat_host( (cagmres_s_+1)*(cagmres_s_+1),0.0 );
	    R.getValue(Rmat_host);
	    que[0].wait();
	    for(PARCEL_INT i = 0;i<Rmat.size() ;i++){
	      Rmat[i] = Rmat_host[i];
	    }

	    // ブロック内の階数落ちの検出 (説明は cagmres_breakdown)。
	    //   k = 0 : R は (s+1)x(s+1)、列 j が基底 v_j (R_00 = ||v_0|| = 1 は調べない)
	    //   k > 0 : R は s x s、列 j が基底 v_{s*k+1+j}。前のブロックに取り除いた成分 (BlockR) も列ノルムに入れる
	    //   R と BlockR は全ランクで同じ値 (allreduce 済み) なので判定も全ランクで一致する。
	    bool blk_bd = false, blk_offdiag_bad = false;
	    PARCEL_INT blk_g = 0;
	    {
	      const PARCEL_INT ns = (k == 0) ? cagmres_s_+1 : cagmres_s_;
	      std::vector<double> extra2(ns, 0.0);
	      if(k > 0){
		const PARCEL_INT ldb = cagmres_s_*k+1;
		for(PARCEL_INT j = 0; j < ns; j++){
		  for(PARCEL_INT r = 0; r < ldb; r++){
		    const double v = (double)BlockRmat[r + (j+1)*ldb].get_T();
		    extra2[j] += v*v;
		  }
		}
	      }
	      const double tau = cagmres_breakdown::tolerance<Tbasis,TcalcQR>(
				   parcel_qr.get_qr_algo() == qr::QR_CHOLESKYQR);
	      const PARCEL_INT jb = cagmres_breakdown::find_rank_deficient(
				      Rmat, ns, (k == 0) ? 1 : 0, extra2, tau, blk_offdiag_bad);
	      if(jb < ns){
		blk_bd = true;
		blk_g  = (k == 0) ? jb : cagmres_s_*k + 1 + jb;
		cagmres_breakdown::sanitize_R(Rmat, ns, jb, blk_offdiag_bad);
	      }
	    }

	    // !     ヘッセンベルグ行列の計算
	    que[0].wait();
	    update_HessenbergMatrix(
				   k,
				   cagmres_t_,
				   cagmres_s_,
				   Hmat, // [s*t+1,s*t]
				   Bmat, // [s+1,s]
				   Rmat, // [s+1,s+1]
				   BlockRmat, // [s*t+1,s+1]
				   hk);
	    que[0].wait();
	    if(blk_bd){
	      // 先頭 blk_g 列だけ残して、このリスタート周期の基底生成を打ち切る (以降の基底はゴミ)
	      //   最後の周回を端数に減らしたときは、使わない基底 (列 m_cyc 以降) の階数落ちは
	      //   無視してよい (blk_g > m_cyc なら先頭 m_cyc 列は有効)。m_cap は m_cyc を超えない
	      m_cap = std::min(m_cyc, cagmres_breakdown::truncate_H(Hmat, cagmres_s_*cagmres_t_+1, cagmres_s_*cagmres_t_,
								     blk_g, blk_offdiag_bad));
	      if(myrank == 0 && blk_g <= m_cyc){
		std::cout<<" pcagmres : rank-deficient basis block (k = "<<k<<"), Krylov subspace invariant after "
			 <<blk_g<<" basis vectors"<<std::endl;
	      }
	      break;
	    }
	    {
	      
	    }
	  }
	  // ギブンス回転と幸運な破綻の検出。Krylov 部分空間が不変になると h_{i+1,i} ~ 0 となり、
	  //   それ以降の基底は QR の 0 割りで NaN かゴミになる。
	  //   列 i の h_{i+1,i} が列ノルムの 1e-14 倍以下 (または 0) ならその列で止め、
	  //   列が非有限ならその手前で止める。使う列数 m_eff で最小二乗を解き x を更新する。
	  //   収束は次の周回先頭 (または反復上限なら finish) の真の残差で判定する。
	  //   破綻が無ければ m_eff = m_cyc (最後の周回で端数に減らしたとき以外は s*t)
	  PARCEL_INT m_eff = m_cyc;
	  {
	    const PARCEL_INT ldh = cagmres_s_*cagmres_t_+1;
	    bool bd = false;
	    // ブロック内の階数落ちがあれば先頭 m_cap 列まで (列 m_cap-1 の h は 0 にしてある)
	    for(PARCEL_INT i=0;i<m_cap && !bd;i++){
	      // それまでの回転は行 0..i にしか作用しないので、h_{i+1,i} と列ノルムは回転前後で同じ
	      double cn2 = 0.0;
	      bool   fin = true;
	      for(PARCEL_INT r = 0; r <= i+1; r++){
		const double h = (double)Hmat[r + i*ldh].get_T();
		if(!std::isfinite(h)) fin = false;
		cn2 += h*h;
	      }
	      const double hn = std::fabs((double)Hmat[i+1 + i*ldh].get_T());
	      if(!fin){
		m_eff = i; bd = true;
	      }else if(hn == 0.0 || hn <= 1.0e-14*std::sqrt(cn2)){
		Hmat[i+1 + i*ldh] = 0.0;
		GivensRotation(i,cagmres_t_,cagmres_s_,Hmat,zeta,co,si);
		const double hd = (double)Hmat[i + i*ldh].get_T();
		m_eff = (hd != 0.0 && std::isfinite(hd)) ? i+1 : i;
		bd = true;
	      }else{
		GivensRotation(i,cagmres_t_,cagmres_s_,Hmat,zeta,co,si);
	      }
	    }
	    if(!bd && m_cap < m_cyc){ m_eff = m_cap; bd = true; }
	    if(bd && myrank == 0){
	      std::cout<<" pcagmres : Arnoldi breakdown, solve with "<<m_eff<<" of "
		       <<m_cyc<<" columns"<<std::endl;
	    }
	  }
	  if(m_eff == 0){
	    st_breakdown = true;
	    st_its = itr*cagmres_s_ * cagmres_t_;
	    break;
	  }
	  for(PARCEL_INT k = 0;k < cagmres_t_;k++){
	  } 

	  // 解ベクトル更新
	timer_solver_update.start();
#if PARCEL_MPI
	{
	  parcel::vector<Tbasis> v0_vecadd ;vec_util.setptr( (n+nadd), v0, v0_vecadd, 0); 
	  parcel::vector<Tbasis> pV_vecadd ;vec_util.setptr( (n+nadd), pV, pV_vecadd, 0); 
	  parcel::vector<Tvec> x_vecadd ;vec_util.setptr( (n+nadd), x, x_vecadd, 0); 
	  update_xvector<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tbasis>(
								   comm->mpi_comm(),
								   n,
								   cagmres_t_,
								   cagmres_s_,
								   precon_Mat,
								   V,
								   Q,
								   v0_vecadd,
								   pV_vecadd,
								   x_vecadd,
								   Hmat,Rmat,zeta,Qflag,m_eff);
      }	
#else	
	  update_xvector<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tbasis>(
			   cagmres_t_,
			   cagmres_s_,
			   precon_Mat,
			   V,Q,v0,pV,x,Hmat,Rmat,zeta,Qflag,m_eff);
#endif	  
	timer_solver_update.end();

	}
	que[0].wait();
	vec_util.convert<Tvec,Tvecx>(x,x0);
	que[0].wait();

      timer_solver_main.end();

      // post-loop true residual: || b - A x_final ||  (reuse vector r as scratch)
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
        double t_solver = timer_solver.get();
        double t_init   = timer_solver_init.get();
        double t_initP  = timer_solver_init_precon.get();
        double t_initC  = timer_solver_init_convert.get();
        double t_loop   = timer_solver_main.get();
        double t_dot    = timer_solver_dot.get();
        double t_axpy   = timer_solver_axpy.get();
        double t_spmv   = timer_solver_SpMV.get();
        double t_calcres= timer_solver_calcres.get();
        double t_precon = timer_solver_precon.get();
        double t_qr     = timer_solver_qr.get();
        double t_reOrth = timer_solver_reOrth.get();
        double t_update = timer_solver_update.get();
        double t_gemv   = timer_solver_gemv.get();
        double t_init_other = t_init - t_initP - t_initC;
        double t_loop_other = t_loop - (t_dot + t_axpy + t_spmv + t_precon + t_qr + t_reOrth + t_update + t_calcres);
        bool   loop_other_clamped = false;
        if (t_loop_other < 0.0) { t_loop_other = 0.0; loop_other_clamped = true; }
        double t_postloop   = t_solver - t_init - t_loop;
        std::cout<<std::endl<<" ----------------- "<<std::endl;
        std::cout<<"pcagmres time         = "<<t_solver     <<" [sec]  (total solver wall time, Q = "
                 <<cagmres_q_name(explicit_q_)<<(Qflag == "N" ? ": not formed" : ": formed")<<")"<<std::endl;
        std::cout<<"  init time           = "<<t_init       <<" [sec]  (before main loop)"<<std::endl;
        std::cout<<"    init precon       = "<<t_initP      <<" [sec]"<<std::endl;
        std::cout<<"    init convert      = "<<t_initC      <<" [sec]"<<std::endl;
        std::cout<<"    init other        = "<<t_init_other <<" [sec]  (vec alloc/zero, init SpMV/dot/precon)"<<std::endl;
        std::cout<<"  main loop time      = "<<t_loop       <<" [sec]  (sum over all restart iterations)"<<std::endl;
        std::cout<<"    loop dot          = "<<t_dot        <<" [sec]  calls = "<<timer_solver_dot.get_count()<<std::endl;
        std::cout<<"    loop axpy         = "<<t_axpy       <<" [sec]  calls = "<<timer_solver_axpy.get_count()<<std::endl;
        std::cout<<"    loop SpMV         = "<<t_spmv       <<" [sec]  calls = "<<timer_solver_SpMV.get_count()<<std::endl;
        std::cout<<"    loop calc_res     = "<<t_calcres    <<" [sec]  calls = "<<timer_solver_calcres.get_count()<<"  (r = b - Ax。SpMV + b 読み + r 書き)"<<std::endl;
        std::cout<<"    loop precon       = "<<t_precon     <<" [sec]  calls = "<<timer_solver_precon.get_count()<<std::endl;
        std::cout<<"    loop QR           = "<<t_qr         <<" [sec]"<<std::endl;
        std::cout<<"    loop reOrth       = "<<t_reOrth     <<" [sec]"<<std::endl;
        // reOrth の内訳 (reOrth の内数)
        std::cout<<"      (内) reOrth dot    = "<<timer_solver_reOrth_dot.get()<<" [sec]  calls = "<<timer_solver_reOrth_dot.get_count()
                 <<"  (R = Q^T V の局所内積。dot を (s*k+1)(s+1) 回)"<<std::endl;
        std::cout<<"      (内) reOrth allred = "<<timer_solver_reOrth_comm.get()<<" [sec]  calls = "<<timer_solver_reOrth_comm.get_count()
                 <<"  (R の allreduce + ホストへの取り出し)"<<std::endl;
        std::cout<<"      (内) reOrth update = "<<timer_solver_reOrth_update.get()<<" [sec]  calls = "<<timer_solver_reOrth_update.get_count()
                 <<"  (V -= Q R、縦長 gemm (dd は既定の gemm))"<<std::endl;
        std::cout<<"    loop update       = "<<t_update     <<" [sec]"<<std::endl;
        std::cout<<"      (内) loop gemv   = "<<t_gemv       <<" [sec]  calls = "<<timer_solver_gemv.get_count()<<"  (update 内の解更新 v0 = Q y、M = s*t。loop update の内数)"<<std::endl;
        std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (kernel launch/sync, cout per iter, ...)"<<std::endl;
        if (loop_other_clamped) {
          std::cout<<"      NOTE: sub-timers overlap (e.g. precon nested inside update_xvector). 'loop other' clamped to 0."<<std::endl;
        }
        std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (vec convert, finish print)"<<std::endl;
      }

      // 収束状態。反復上限で抜けたときは pgmres と同じく finish の値 (真の残差 / ||b||) で判定する
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

  }
}

// CAGMRES_Q_IMPLICIT の本体 (parcel::krylov::detail::cagmres_implicit_q)。
//   上の cagmres_breakdown を使うので、pcagmres の後に読み込む。
#include "parcel_krylov_choleskyqr_cagmres.hpp"

#endif

