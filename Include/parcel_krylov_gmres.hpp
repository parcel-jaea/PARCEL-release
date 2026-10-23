// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター

#ifndef PARCEL_KRYLOV_GMRES_HPP_
#define PARCEL_KRYLOV_GMRES_HPP_

#include "parcel_qr.hpp"
#include <memory>   // std::unique_ptr (for optional external precon_matrix)

// double-double演算未対応

namespace parcel{
  namespace krylov{

    // 計算精度を指定
    template<
      class Tspmv   = parcel_double, // SpMV 計算精度
      class Tprecon = parcel_double, // 前処理計算精度
      class Taxpy   = parcel_double, // ベクトル演算計算精度
      class Tdot    = parcel_double  // 縮約演算計算精度
      >
    class pgmres : public krylov_param{

      ddlib::ddmath dd;
      parcel::timer timer_solver;
      parcel::timer timer_solver_init;
      parcel::timer timer_solver_init_precon;
      parcel::timer timer_solver_init_convert;
      parcel::timer timer_solver_scala;

      parcel::timer timer_solver_SpMV;
      // r = b - Ax は SpMV に加えて b の読み出しと r の書き込みを含む。
      //   SpMV のタイマに入れると回数もトラフィックも食い違うので分ける。
      parcel::timer timer_solver_calcres;
      parcel::timer timer_solver_axpy;
      parcel::timer timer_solver_dot;
      parcel::timer timer_solver_precon;
      parcel::timer timer_solver_mainloop;
      parcel::timer timer_solver_GS;
      parcel::timer timer_solver_givens;
      parcel::timer timer_solver_gemv;
      parcel::vector_util vec_util;

      // リスタート周期の既定値は 30 (PETSc の既定と同じ)。作業ベクトルは (m+1) 本。
      PARCEL_INT gmres_m_ = 30;
      PARCEL_INT orthogonal_algo_ = parcel::qr::QR_MGS;
      // 前処理空間出力モード (固有値ソルバの A K^-1 モード用)。
      //   右前処理 GMRES は内部で (A K^-1) z = b を解き、最後に x = K^-1 z を返す。
      //   true にすると最後の K^-1 を掛けずに z を x0 に返す (x0 の初期値も z0 と解釈)。
      //   これで K を陽に持たずに (A K^-1)^-1 b が得られる。
      bool precon_space_output_ = false;
      // シフト σ: (A K^-1 - σI) z = b を解く。precon_space_output_ が true のときだけ有効。
      //   Arnoldi は w = A K^-1 v - σ v、残差は r = b - A K^-1 z + σ z。
      double shift_ = 0.0;
      
      parcel::scalar_util host_util;
      parcel::scalar_blas parcel_blas_host;

    public :

      pgmres(){}
      ~pgmres(){}
      // 固有値ソルバ (inverse_power / ks_arnoldi) の A K^-1 モードから使う
      void set_precon_space_output(bool f){ precon_space_output_ = f; }
      bool get_precon_space_output() const { return precon_space_output_; }
      void set_shift(double sigma){ shift_ = sigma; }
      double get_shift() const { return shift_; }

      // 範囲外の値は std::invalid_argument (set_qr_algo は QR_CGS / QR_MGS、set_restart は 1 以上)
      void set_qr_algo(PARCEL_INT qrdecomposition_algo){
	if(qrdecomposition_algo != parcel::qr::QR_CGS && qrdecomposition_algo != parcel::qr::QR_MGS){
	  throw std::invalid_argument("parcel::krylov::pgmres::set_qr_algo: invalid value "
				      + std::to_string((long long)qrdecomposition_algo)
				      + " (expected QR_CGS or QR_MGS)");
	}
	orthogonal_algo_ = qrdecomposition_algo;
	return ;
      }
      void set_restart(PARCEL_INT gmres_m){
	check_positive(gmres_m, "pgmres::set_restart");
	gmres_m_ = gmres_m;
	return ;
      }
      // 既定値の参照用 (C API の parcel_options_init がここから既定値を取る)
      PARCEL_INT get_restart() const { return gmres_m_; }
      PARCEL_INT get_qr_algo() const { return orthogonal_algo_; }

      template<
	class Tvec,
	class Tcalc
	>
      void GS_ORTHONORMALIZATION(
				 parcel::blas<PARCEL_MAX_FLOTING_TYPE> &parcel_blas,
				 PARCEL_INT n,
				 PARCEL_INT m,
#if PARCEL_MPI
				 parcel::vector<Tvec> &v, size_t ldv,   // size_t (ldv*(m+1) の桁あふれ防止)
				 parcel::vector<Tvec> &w, PARCEL_INT ldw,
#else
				 parcel::vector<Tvec> &v,
				 parcel::vector<Tvec> &w,
#endif				 
				 std::vector< std::vector< parcel::scalar > > &hki
				 ){
#if PARCEL_MPI
	parcel::comm &parcel_comm = parcel_blas.parcel_comm();
	int myrank = parcel_comm.rank();
#endif
	// 古典グラムシュミット直交化
	if( orthogonal_algo_ == parcel::qr::QR_CGS ){

	  sycl::queue *que = v.queue();
	  parcel::vector<Tcalc> one;
	  one.resize(1,que);
	  double one_ = 1.0;one.setValue(&one_);
	  parcel::vector<Tcalc> zero;
	  zero.resize(1,que);
	  double zero_ = 0.0;zero.setValue(&zero_);

	  // hki = <wk,Vi> 
	  // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
#if PARCEL_MPI
	  parcel::vector<Tcalc> wvi_tmp; wvi_tmp.resize(m+1,que);
	  parcel::vector<Tcalc> wvi; wvi.resize(m+1,que);
	  que[0].wait();
	  que[0].wait();
	  parcel_blas.gemm<Tcalc,Tcalc,Tvec,Tvec,Tcalc,Tcalc>(
							      1,
							      m+1,
							      n,
							      one,zero,
							      w  ,1,
							      v  ,ldv,
							      wvi_tmp,1);
	  que[0].wait();
	  parcel_comm.allreduce(wvi_tmp,wvi,m+1,MPI_SUM);
	  
#else
	  parcel::vector<Tcalc> wvi; wvi.resize(m+1,que);
	  que[0].wait();
	  parcel_blas.gemm<Tcalc,Tcalc,Tvec,Tvec,Tcalc,Tcalc>(
							      1,
							      m+1,
							      n,
							      one,zero,
							      w  ,1,
							      v  ,n,
							      wvi,1);
	  que[0].wait();
#endif	  
	  std::vector< parcel::scalar > wvi_host(m+1,0.0);
	  wvi.getValue(wvi_host);
	  que[0].wait();
	  for(PARCEL_INT i = 0;i<=m;i++){
	    hki[i][m]   =  wvi_host[i];
	    wvi_host[i] = -wvi_host[i];
	  }

	  // wi = wi - hki*Qk  
	  que[0].wait();
	  wvi.setValue(wvi_host);
	  que[0].wait();
#if PARCEL_MPI
	  parcel::vector<Tvec> v_add; vec_util.setptr(ldv*(m+1), v, v_add, 0);
	  parcel::vector<Tvec> w_add; vec_util.setptr(ldw      , w, w_add, 0);
	  // v の列間隔 ldv は w の長さ ldw とは一致しない (v だけ詰め物が入る)。
	  // gemv に ldv を明示しないと y の長さから導かれて別の列を読む。
	  parcel_blas.gemv<Tcalc,Tcalc,Tvec,Tcalc,Tvec,Taxpy>(m+1,one,one,v_add,wvi,w_add,ldv);
#else
	  parcel_blas.gemv<Tcalc,Tcalc,Tvec,Tcalc,Tvec,Taxpy>(m+1,one,one,v,wvi,w);
#endif
	  
	  return ;

	}

	// 修正グラムシュミット直交化
	if( orthogonal_algo_ == parcel::qr::QR_MGS ){
	  sycl::queue *que = v.queue();
	  parcel::vector<Tvec> vi0;
	  // 係数 -hki をホストの値のまま axpy に渡す。係数を 1 要素のデバイス配列へ書き戻す (setValue) 手間と、
	  //   その前後の wait が要らない (dot は値を返す前に自分で同期し、キューは in-order)。
	  //   計算の中身 (dot の値、axpy の式と係数の丸め) は下の書き戻す経路と同じ。
	  //   double-double を含む組み合わせは下の書き戻す経路を使う。
	  if constexpr (parcel::blas<PARCEL_MAX_FLOTING_TYPE>::template axpy_value_supported<Tcalc,Tvec,Tvec,Taxpy>()){
#if PARCEL_GS_FUSED
	    // CPU: 「i-1 本目の axpy」と「i 本目の内積」を 1 本のカーネルで行う。
	    //   計算の順序は下の 2 本版と同じ (i=0 は内積だけ、最後の axpy は単独)。
	    {
	      parcel::vector<Tvec> vprev;
	      parcel::scalar a_prev;
	      for(PARCEL_INT i = 0;i<=m;i++){
#if PARCEL_MPI
		vec_util.setptr(n, v, vi0, (size_t)ldv*i);
#else
		vec_util.setptr(n, v, vi0, (size_t)n*i);
#endif
		if(i == 0){
		  parcel_blas.dot< Tvec,Tvec,Tcalc >(w,vi0);
		}else{
		  parcel_blas.axpy_dot_value<Tcalc,Tvec,Taxpy,Tcalc>(a_prev,vprev,w,vi0);
		}
		parcel::scalar wvi_host = parcel_blas.get_dot();
		hki[i][m] = wvi_host;
		a_prev = -wvi_host;
#if PARCEL_MPI
		vec_util.setptr(n, v, vprev, (size_t)ldv*i);
#else
		vec_util.setptr(n, v, vprev, (size_t)n*i);
#endif
	      }
	      parcel_blas.axpy_value<Tcalc,Tvec,Tvec,Taxpy>(a_prev,vprev,w);
	      que[0].wait();
	      return ;
	    }
#endif
	    for(PARCEL_INT i = 0;i<=m;i++){
#if PARCEL_MPI
	      vec_util.setptr(n, v, vi0, (size_t)ldv*i);
#else
	      vec_util.setptr(n, v, vi0, (size_t)n*i);
#endif
	      parcel_blas.dot< Tvec,Tvec,Tcalc >(w,vi0);
	      parcel::scalar wvi_host = parcel_blas.get_dot();
	      hki[i][m] = wvi_host;
	      wvi_host  = -wvi_host;
	      parcel_blas.axpy_value<Tcalc,Tvec,Tvec,Taxpy>(wvi_host,vi0,w);
	    }
	    // axpy_value は PARCEL_TIMER_SYNC=0 では待たずに返る。w を返す前に 1 回だけ同期する。
	    que[0].wait();
	    return ;
	  }
	  parcel::vector<Tcalc> one;
	  one.resize(1,que);
	  double one_ = 1.0;one.setValue(&one_);
	  parcel::vector<Tcalc> wvi; wvi.resize(1,que);
	  for(PARCEL_INT i = 0;i<=m;i++){
	    // hki = <wk,Vi> 
#if PARCEL_MPI
	    vec_util.setptr(n, v, vi0, (size_t)ldv*i);
#else
	    vec_util.setptr(n, v, vi0, (size_t)n*i);
#endif
	    que[0].wait();
	    parcel_blas.dot< Tvec,Tvec,Tcalc >(w,vi0);
	    que[0].wait();

	    parcel::scalar wvi_host = parcel_blas.get_dot();
	    que[0].wait();
	    hki[i][m] = wvi_host;
	    wvi_host  = -wvi_host;

	    // wi = wi - hki*Qk  
	    que[0].wait();
	    wvi.setValue(wvi_host); 
	    que[0].wait();
	    parcel_blas.axpy<Tcalc,Tcalc,Tvec,Tvec,Taxpy>(wvi,one,vi0,w);
	    que[0].wait();
	  }
	  return ;
	}
	return ;

      }
      
  void GivensRotation(
		      PARCEL_INT m,
		      std::vector< 
		      std::vector< parcel::scalar > > &hki,
		      std::vector<parcel::scalar> &e,
		      std::vector<parcel::scalar> &co,
		      std::vector<parcel::scalar> &si
		      ){

    // ----- 
    // m+1行のヘッセンベルグ行列に対するギブンス回転
    // Givens rotation
    parcel::scalar one  = 1.0;
    parcel::scalar tmp1 = 0.0;
    parcel::scalar tmp2 = 0.0;
    for(PARCEL_INT k=0;k<m;k++){
      tmp1 = hki[k][m]*co[k] - hki[k+1][m]*si[k];
      tmp2 = hki[k][m]*si[k] + hki[k+1][m]*co[k];
      hki[k][m]   = tmp1;
      hki[k+1][m] = tmp2;
    }

    // -------
    tmp1  = hki[m+1][m]/hki[m][m];
    co[m] = one / host_util.sqrt(one+tmp1*tmp1 );
    si[m] = -co[m]*tmp1;

    hki[m][m]   = co[m]*hki[m][m] - si[m]*hki[m+1][m];
    hki[m+1][m] = 0.0;

    tmp1 = co[m]*e[m]-si[m]*e[m+1];
    tmp2 = si[m]*e[m]+co[m]*e[m+1];
    e[m]   = tmp1;
    e[m+1] = tmp2;

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
	class TGS = parcel_double, // グラムシュミットの直交化精度
	class THessenberg = parcel_double // ヘッセンベルグ行列の精度
	>
      void solver(
		  parcel::matrix<Tmat>  &Mat,
		  parcel::vector<Tvecx> &x0,
		  parcel::vector<Tvecb> &b0,
		  // 外部で構築済 precon_matrix を渡す (nested 用; 毎呼出 init を回避)
		  // nullptr の場合は内部で構築
		  parcel::precon_matrix<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tprecon> *external_precon = nullptr
		  ){

	sycl::queue *que   = Mat.queue();

	require_in_order(que, "krylov::pgmres");   // in-order キューが前提
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

	timer_solver.start();
	timer_solver_init.start();

	// ------------------------------
	// Init Preconditioning
	timer_solver_init_precon.start();

	const double tol = get_tolerance();
	const PARCEL_INT itrmax = get_max_iteration();
	// 外部 precon が指定された場合はそれを使う (nested 呼出時に重複 init を回避)
	precon_param &precon = *(get_precon_param());
	std::unique_ptr<parcel::precon_matrix<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tprecon> > local_precon;
	if(external_precon == nullptr){
	  local_precon.reset(new parcel::precon_matrix<
	    Tmat, Tvec, Tprecon_mat, Tprecon_vec, Tprecon
	    >(Mat, precon));
	}
	parcel::precon_matrix<Tmat,Tvec,Tprecon_mat,Tprecon_vec,Tprecon> &precon_Mat =
	  external_precon ? *external_precon : *local_precon;

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
	timer_solver_init_convert.end();

	// ------
	// pgmres 
	// ------
	// do itr=1,....
	//   r=b-Ax
	//   v_1=r/||r||
	//   do i=1,m
	//     solve Mpv=v_i
	//     w=Apv
	//     グラムシュミット直交化
	//     v_i+1 = w/||w||
	//     ギブンス回転
	//     if(||r||<rmax)exit
	//   enddo
	//   解ベクトル更新
	//   if(||r||<rmax)exit
	// enddo

	// 使う区間が重ならない作業ベクトルを同じ領域に置く (省メモリ)。
	//   r        -> pv の領域: r は再始動の頭で b-Ax を入れ、v_1 = r/||r|| を作ったら読まない。pv は Arnoldi 内 (前処理の出力) だけ。
	//   v0       -> pv の領域: v0 = V y は Arnoldi を抜けた後の解の更新だけで使う (次の再始動では r が先に上書きする)。
	//   x_update -> w  の領域: x_update = M^-1 v0 も解の更新だけ。w は Arnoldi 内 (A pv と直交化) だけ。
	//   r_true   -> w  の領域: 真の残差は再始動ループを抜けた後に一度だけ計算する。
	//   前処理 M^-1 v0 の入力 (pv の領域) と出力 (w の領域) は別の領域なので重ならない。
	//   計算の順序も値も変わらない。袖付きベクトルが 4 本減る。
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> pv_parallel(n,nadd,1,que);parcel::vector<Tvec> pv;vec_util.setptr( n, pv_parallel.vec(0), pv, 0);
	parcel::parallel_vector<Tvec> w_parallel(n,nadd,1,que);parcel::vector<Tvec> w;vec_util.setptr( n, w_parallel.vec(0), w, 0);
	parcel::parallel_vector<Tvec> &r_parallel      = pv_parallel;
	parcel::parallel_vector<Tvec> &r_true_parallel = w_parallel;
	parcel::parallel_vector<Tvec> &v0_parallel       = pv_parallel;
	parcel::parallel_vector<Tvec> &x_update_parallel = w_parallel;
	parcel::vector<Tvec> r;vec_util.setptr( n, r_parallel.vec(0), r, 0);
	parcel::vector<Tvec> r_true;vec_util.setptr( n, r_true_parallel.vec(0), r_true, 0);
	parcel_blas.zero_init<Tvec>(r);
	parcel_blas.zero_init<Tvec>(r_true);

	// 基底 v の列アドレスは ld_v (= parallel_vector の列間隔) で組み立て、確保側と一致させる。
	//   列の後ろに詰め物 PARCEL_BASIS_PAD 要素を入れる (キャッシュセットの衝突を避ける)。
	parcel::parallel_vector<Tvec> v_parallel(n,nadd,gmres_m_+1,que,PARCEL_BASIS_PAD);
	// ld_v は size_t (ld_v*i などの積を int32 で作ると桁あふれする)
	const size_t ld_v = v_parallel.ld();
	parcel::vector<Tvec> v; 
	vec_util.setptr( ld_v*(gmres_m_+1), v_parallel.vec_add(),v, 0);
	parcel_blas.zero_init<Tvec>(v);

	// pv / w は上で確保済み
	parcel::vector<Tvec> v0;vec_util.setptr( n, v0_parallel.vec(0), v0, 0);
	parcel::vector<Tvec> x_update;vec_util.setptr( n, x_update_parallel.vec(0), x_update, 0);
 
	parcel_blas.zero_init<Tvec>(v);
	parcel_blas.zero_init<Tvec>(pv);
	parcel_blas.zero_init<Tvec>(w);
	parcel_blas.zero_init<Tvec>(v0);
	parcel_blas.zero_init<Tvec>(x_update);
#else
	parcel::vector<Tvec> pv(n,que);parcel_blas.zero_init<Tvec>(pv);
	parcel::vector<Tvec> w(n,que);parcel_blas.zero_init<Tvec>(w);
	// 領域の割り当ては MPI 側の説明と同じ
	parcel::vector<Tvec> &r        = pv;
	parcel::vector<Tvec> &r_true   = w;
	parcel::vector<Tvec> &v0       = pv;
	parcel::vector<Tvec> &x_update = w;

	parcel::vector<Tvec> v((size_t)n*(gmres_m_+1),que);parcel_blas.zero_init<Tvec>(v);
	const size_t ld_v = n;   // 非 MPI は詰め物を入れない
#endif
	parcel::vector<Tvec> y( (gmres_m_+1) ,que);parcel_blas.zero_init<Tvec>(y);
	parcel_blas.zero_init<Tvec>(y);

	// 前処理空間出力モード: 反復中の x は z (A K^-1 z = b の解) を持つ。
	//   残差 r = b - A K^-1 z + σ z の計算に K^-1 z の置き場が要る。
	//   通常モードでは使わないので 1 要素だけ確保する。
	const bool zmode = precon_space_output_;
	const bool zshift = zmode && (shift_ != 0.0);
	const PARCEL_INT n_zk = zmode ? n : 1;
#if PARCEL_MPI
	parcel::parallel_vector<Tvec> xk_parallel(n_zk, zmode ? nadd : 0, 1, que);
	parcel::vector<Tvec> xk; vec_util.setptr( n_zk, xk_parallel.vec(0), xk, 0);
#else
	parcel::vector<Tvec> xk(n_zk,que);
#endif
	parcel_blas.zero_init<Tvec>(xk);
	parcel::vector<Tdot> sigma_vec;      sigma_vec.resize(1,que);
	parcel::vector<Tdot> nega_sigma_vec; nega_sigma_vec.resize(1,que);
	{ double s_ = shift_; sigma_vec.setValue(&s_); s_ = -shift_; nega_sigma_vec.setValue(&s_); }
	
	std::vector< parcel::scalar > e(gmres_m_+1 ,0.0);
	std::vector< parcel::scalar > co(gmres_m_+1 ,0.0);
	std::vector< parcel::scalar > si(gmres_m_+1 ,0.0);
	std::vector< std::vector< parcel::scalar > > hki( gmres_m_+1  , std::vector< parcel::scalar > (gmres_m_)   );
	std::vector< std::vector< parcel::scalar > > Rtri( gmres_m_+1 , std::vector< parcel::scalar > (gmres_m_+1) );

	// ------------------------------
	parcel::vector<Tdot> one;
	one.resize(1,que);
	double one_ = 1.0;one.setValue(&one_);

	parcel::vector<Tdot> zero;
	zero.resize(1,que);
	double zero_ = 0.0;zero.setValue(&zero_);

	parcel::vector<Tdot> negative_one;
	negative_one.resize(1,que);
	double negative_one_ = -1.0;negative_one.setValue(&negative_one_);
	
	// || b ||
	parcel_blas.dot< Tvec,Tvec,Tdot >(b,b);
	parcel::scalar bnorm_host = parcel_blas.get_dot();
	bnorm_host = host_util.sqrt(bnorm_host);
	PARCEL_MAX_FLOTING_TYPE  bnorm_ = bnorm_host.get_T();

	// ||b|| = 0 なら解は x = 0 (前処理空間出力モードでは z = 0)。残差を 0 で割って NaN に
	//   しないため反復しない。bnorm は allreduce 済みで全ランク同じ分岐。
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

	// 収束状態 (krylov_param::set_status)。判定に使うのはリスタート先頭の真の残差
	//   反復回数は内側反復の総数 (表示のカウンタ itr*gmres_m_ と同じ数え方)
	bool   st_converged = false;
	bool   st_nonfinite = false;
	bool   st_breakdown = false;   // 幸運でない破綻 (H が特異) で打ち切った
	// 反復の上限 itrmax は restart (gmres_m_) の倍数でなくてよい (倍数に切り捨てない)。
	//   周回数を ceil(itrmax/gmres_m_) にし、最後の周回だけ m_cyc = itrmax - itr*gmres_m_ 列で Arnoldi を止め、
	//   その列数で最小二乗を解いて x を更新する。上限まで回ったときの反復回数は itrmax。
	//   itrmax < gmres_m_ でも itrmax 反復する。
	const PARCEL_INT n_cycle = itrmax/gmres_m_ + ((itrmax % gmres_m_) != 0 ? 1 : 0);
	PARCEL_INT st_its   = itrmax;   // 上限まで回ったときの値
	double st_res       = std::numeric_limits<double>::quiet_NaN();
	timer_solver_init.end();
	timer_solver_mainloop.start();
	for(PARCEL_INT itr = 0;itr < n_cycle;itr++){
	  for(PARCEL_INT i = 0;i<e.size() ;i++){e[i]  = 0.0;}
	  for(PARCEL_INT i = 0;i<co.size();i++){co[i] = 0.0;}
	  for(PARCEL_INT i = 0;i<si.size();i++){si[i] = 0.0;}

	  for(int i = 0;i<hki.size();i++){
	    for(int j = 0;j<hki[i].size();j++){
	      hki[i][j] = 0.0;
	    }
	  }
	  for(int i = 0;i<Rtri.size();i++){
	    for(int j = 0;j<Rtri[i].size();j++){
	      Rtri[i][j] = 0.0;
	    }
	  }

	  // r = b - Ax
	  //   前処理空間出力モードでは x は z なので r = b - A (K^-1 z) + σ z
	  if(zmode){
	    timer_solver_precon.start();
#if PARCEL_MPI
	    parcel::vector<Tvec> z_vecadd ;vec_util.setptr( (n+nadd) , x_parallel.vec_add(0), z_vecadd,0);
	    parcel::vector<Tvec> xk_vecadd;vec_util.setptr( (n+nadd) , xk_parallel.vec_add(0), xk_vecadd,0);
	    precon_Mat.solver(z_vecadd,xk_vecadd);
#else
	    precon_Mat.solver(x,xk);
#endif
	    timer_solver_precon.end();
	  }
	  timer_solver_calcres.start();
#if PARCEL_MPI
	  if(zmode) parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,xk_parallel.vec_add(0),b,r);
	  else      parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r);
#else
	  if(zmode) parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,xk,b,r);
	  else      parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,x,b,r);
#endif
	  timer_solver_calcres.end();
	  if(zshift){
	    timer_solver_axpy.start();
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (sigma_vec,one,x,r);
	    timer_solver_axpy.end();
	  }
	  // || r ||
	  timer_solver_dot.start();
	  parcel_blas.dot< Tvec,Tvec,Tdot >(r,r);
	  parcel::scalar rnorm_host = parcel_blas.get_dot();
	  timer_solver_dot.end();
	  rnorm_host = host_util.sqrt(rnorm_host);
	  PARCEL_MAX_FLOTING_TYPE rnorm_ = rnorm_host.get_T();
	  e[0] = -rnorm_;
	  double bnorm_FP64 = bnorm_;
	  double rnorm_FP64 = rnorm_;
	  if( myrank == 0){
	    std::cout<<std::setw(8)<< itr*gmres_m_ <<" : residual norm = ";
	    std::cout<<std::scientific << std::setprecision(14) << rnorm_FP64/bnorm_FP64 <<" , true = "<< rnorm_FP64/bnorm_FP64 <<std::endl;
	  }
	  st_res = rnorm_FP64/bnorm_FP64;
	  // 非有限になったらその場で抜ける
	  if(!std::isfinite(st_res)){ st_nonfinite = true; st_its = itr*gmres_m_; break; }
	  if((rnorm_/bnorm_)<tol){ st_converged = true; st_its = itr*gmres_m_; break; }

	  // v_1 = -r/||r||
	  parcel::scalar rnorm_inv_host ; rnorm_inv_host.set_Tdd(-1.0/rnorm_);
	  parcel::vector<Tdot> rnorm_inv; rnorm_inv.resize(1,que);
	  rnorm_inv.setValue(rnorm_inv_host); 

	  timer_solver_axpy.start();
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (rnorm_inv,zero,r,v);
	  timer_solver_axpy.end();
	  parcel::vector<Tvec> vi0;
	  parcel::vector<Tvec> vi1;

	  // 幸運な破綻 (Krylov 部分空間が不変になり ||w|| ~ 0) では 1/||w|| が inf になるので、
	  //   その列で Arnoldi を止め、m_eff = i+1 列で最小二乗を解いて x を更新する。収束は次のリスタート先頭
	  //   (または反復上限なら finish) の真の残差で判定する。
	  // m_cyc はこの周回の列数。最後の周回だけ上限の端数 (itrmax - itr*gmres_m_) に減らす
	  const PARCEL_INT m_cyc = std::min(gmres_m_, itrmax - itr*gmres_m_);
	  PARCEL_INT m_eff = m_cyc;
	  for(PARCEL_INT i = 0;i < m_cyc;i++){
	    // solve Mpv=v
	    timer_solver_precon.start();
#if PARCEL_MPI
	    //   view の長さは局所 + halo (n+nadd)。ld_v には詰め物が入りうる。
	    vec_util.setptr( (size_t)(n+nadd), v, vi0, ld_v*i);
	    parcel::vector<Tvec> pv_vecadd ;vec_util.setptr( (n+nadd) , pv_parallel.vec_add(0), pv_vecadd,0);
	    precon_Mat.solver(vi0,pv_vecadd) ;
#else
	    vec_util.setptr(n, v, vi0, (size_t)n*i);
	    precon_Mat.solver(vi0,pv) ;
#endif
	    timer_solver_precon.end();

	    // w=Apv
	    timer_solver_SpMV.start();
#if PARCEL_MPI
	    parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pv_parallel.vec_add(0),w);
#else
	    parcel_blas.SpMV<Tdot,Tdot,Tmat,Tvec,Tvec,Tspmv>(one,zero,Mat,pv,w);
#endif
	    timer_solver_SpMV.end();
	    // シフト: w = A K^-1 v_i - σ v_i
	    if(zshift){
	      parcel::vector<Tvec> vi_loc;
#if PARCEL_MPI
	      vec_util.setptr(n, v, vi_loc, ld_v*i);
#else
	      vec_util.setptr(n, v, vi_loc, (size_t)n*i);
#endif
	      timer_solver_axpy.start();
	      parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (nega_sigma_vec,one,vi_loc,w);
	      timer_solver_axpy.end();
	    }
	    
	    // グラムシュミット直交化
	    timer_solver_GS.start();
#if PARCEL_MPI
	    GS_ORTHONORMALIZATION<Tvec,TGS>(parcel_blas,n,i,v,ld_v,w,n+nadd,hki);
#else
	    GS_ORTHONORMALIZATION<Tvec,TGS>(parcel_blas,n,i,v,w,hki);
#endif
	    timer_solver_GS.end();
	que[0].wait();
	// v_i+1 = w/||w||
#if PARCEL_MPI
	vec_util.setptr(n, v, vi1, ld_v*(i+1) );
#else	
	vec_util.setptr(n, v, vi1, (size_t)n*(i+1) );
#endif	
	timer_solver_dot.start();
	parcel_blas.dot< Tvec,Tvec,Tdot >(w,w);
	parcel::scalar wnorm_host = parcel_blas.get_dot();
	timer_solver_dot.end();
	wnorm_host = host_util.sqrt(wnorm_host);
	//
	hki[i+1][i] = wnorm_host.get_T();

	{
	  // 直交化前の ||w|| は h の列から通信なしで求まる (||w_before||^2 = sum_k h_{k,i}^2 + ||w||^2)
	  const double wn = (double)wnorm_host.get_T();
	  double wb2 = wn*wn;
	  for(PARCEL_INT k = 0; k <= i; k++){
	    const double h = (double)hki[k][i].get_T();
	    wb2 += h*h;
	  }
	  if(!std::isfinite(wn) || !std::isfinite(wb2)){
	    st_nonfinite = true;
	    m_eff = 0;
	    break;
	  }
	  if(wn == 0.0 || wn <= 1.0e-14*std::sqrt(wb2)){
	    hki[i+1][i] = 0.0;
	    timer_solver_givens.start();
	    GivensRotation(i,hki,e,co,si);
	    timer_solver_givens.end();
	    // 回転後の対角が 0 なら H が特異 (A K^-1 v_i が既存の基底に含まれ、かつ寄与 0)。
	    //   その列は最小二乗から外す。
	    const double hd = (double)hki[i][i].get_T();
	    m_eff = (hd != 0.0 && std::isfinite(hd)) ? i+1 : i;
	    if( myrank == 0){
	      std::cout<<" pgmres : Arnoldi breakdown at column "<<i+1<<" (||w|| = "<<wn
		       <<"), solve with "<<m_eff<<" columns"<<std::endl;
	    }
	    break;
	  }
	}

	parcel::scalar wnorm_inv_host;
	wnorm_inv_host.set_Tdd(1.0/wnorm_host.get_T());
	parcel::vector<Tdot> wnorm_inv; wnorm_inv.resize(1,que);
	wnorm_inv.setValue(wnorm_inv_host);
	timer_solver_axpy.start();
	parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (wnorm_inv,zero,w,vi1);
	timer_solver_axpy.end();


	    //Givens rotation
	    timer_solver_givens.start();
	    GivensRotation(i,hki,e,co,si);
	    timer_solver_givens.end();


	  }



	  if(st_nonfinite){ st_its = itr*gmres_m_; break; }
	  if(m_eff == 0){
	    // 最初の列で特異。x は更新できない
	    st_breakdown = true;
	    st_its = itr*gmres_m_;
	    break;
	  }

	  // ------------------------------------------------------------------
	  // 解ベクトルの更新(リスタート周期になるまでループを抜けない実装)
	  //   列数は m_eff (幸運な破綻で途中で止めた場合や、上限の端数で最後の周回を短くした場合は gmres_m_ より少ない)
      std::vector< parcel::scalar > Hinv( m_eff*m_eff );
      for(PARCEL_INT i=0;i<m_eff;i++){
	for(PARCEL_INT j=0;j<m_eff;j++){
	  Hinv[j+i*m_eff]=hki[j][i];
	}
      }
      std::string uplo("U");
      std::string diag("N");

      // 上三角の H の逆行列をホストで trtri で求め、y = H^-1 e を作る
      parcel_blas_host.trtri(uplo,diag,Hinv,m_eff,m_eff);
      
      std::vector< parcel::scalar > y_host( (gmres_m_+1),0.0);
      for(PARCEL_INT i=0;i<m_eff;i++){
	y_host[i]=0.0;
	for(PARCEL_INT j=0;j<m_eff;j++){
	  y_host[i] = y_host[i] + Hinv[i+j*m_eff]*e[j];
	}
      }
      parcel::vector<THessenberg> y_;y_.resize(m_eff,que);y_.setValue(y_host.data());
      vec_util.convert<THessenberg,Tvec>(y_,y);

      // gemv計算
      // v0 = v[][]*y
      timer_solver_gemv.start();
#if PARCEL_MPI
      // 袖部分についても更新する実装
      parcel::vector<Tvec> v_update ;vec_util.setptr( ld_v*(m_eff), v, v_update ,0);
      // y は v0_parallel (詰め物なし) なので、A の ld は明示的に渡す。
      parcel_blas.gemv<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy>(m_eff,one,zero,
						       v_update,
						       y,
						       v0_parallel.vec_add(0),
						       ld_v);
#else
      parcel_blas.gemv<Tdot,Tdot,Tvec,Tvec,Tvec,Taxpy>(m_eff,one,zero,v,y,v0);
#endif
      timer_solver_gemv.end();
      

	  if(zmode){
	    // 前処理空間出力モード: z = z + V y (K^-1 は掛けない)
	    timer_solver_axpy.start();
	    parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,one,v0,x);
	    timer_solver_axpy.end();
	  } else {
	  // 前処理計算
	  // Mx0 = v0
	  timer_solver_precon.start();
#if PARCEL_MPI
	  parcel::vector<Tvec> v0_vecadd ;vec_util.setptr( (n+nadd) , v0_parallel.vec_add(0), v0_vecadd,0);
	  parcel::vector<Tvec> x_update_vecadd ;vec_util.setptr( (n+nadd) , x_update_parallel.vec_add(0), x_update_vecadd,0);
	  precon_Mat.solver(
			    v0_vecadd,
			    x_update_vecadd);
#else
	  precon_Mat.solver(v0,x_update);
#endif
	  timer_solver_precon.end();

	  // 解ベクトルの更新 : x = x + x0
	  timer_solver_axpy.start();
	  parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (one,one,x_update,x);
	  timer_solver_axpy.end();
	  }
	  // ------------------------------------------------------------------

	} // restart loop
	timer_solver_mainloop.end();

	vec_util.convert<Tvec,Tvecx>(x,x0);
	if(zmode){
	  // 真の残差 || b - A K^-1 z + σ z ||
#if PARCEL_MPI
	  parcel::vector<Tvec> z_vecadd ;vec_util.setptr( (n+nadd) , x_parallel.vec_add(0), z_vecadd,0);
	  parcel::vector<Tvec> xk_vecadd;vec_util.setptr( (n+nadd) , xk_parallel.vec_add(0), xk_vecadd,0);
	  precon_Mat.solver(z_vecadd,xk_vecadd);
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,xk_parallel.vec_add(0),b,r_true);
#else
	  precon_Mat.solver(x,xk);
	  parcel_blas.calc_res<Tmat,Tvec,Tvec,Tvec,Tspmv>(Mat,xk,b,r_true);
#endif
	  if(zshift) parcel_blas.axpy<Tdot,Tdot,Tvec,Tvec,Taxpy> (sigma_vec,one,x,r_true);
	} else {
#if PARCEL_MPI
	parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(Mat,x_parallel.vec_add(0),b,r_true);
#else
	parcel_blas.calc_res<Tmat,Tvecx,Tvec,Tvec,Tspmv>(Mat,x0,b,r_true);
#endif
	}
	parcel_blas.dot< Tvec,Tvec,Tdot >(r_true,r_true);
	parcel::scalar rnorm_host_ = parcel_blas.get_dot();
	rnorm_host_ = host_util.sqrt(rnorm_host_);

	if( myrank == 0){
	  std::cout<<std::setw(8)<<" finish : residual true norm  || b - Ax || = ";
	}
	double finish_res_norm = rnorm_host_.get_T()/bnorm_;
	if( myrank == 0){
	  std::cout<<std::scientific << std::setprecision(10) << finish_res_norm <<std::endl;
	}

	timer_solver.end();
	if( myrank == 0){
        double t_solver = timer_solver.get();
        double t_init   = timer_solver_init.get();
        double t_initP  = timer_solver_init_precon.get();
        double t_initC  = timer_solver_init_convert.get();
        double t_loop   = timer_solver_mainloop.get();
        double t_dot    = timer_solver_dot.get();
        double t_axpy   = timer_solver_axpy.get();
        double t_spmv   = timer_solver_SpMV.get();
        double t_calcres= timer_solver_calcres.get();
        double t_precon = timer_solver_precon.get();
        double t_GS     = timer_solver_GS.get();
        double t_givens = timer_solver_givens.get();
        double t_gemv   = timer_solver_gemv.get();
        double t_init_other = t_init - t_initP - t_initC;
        double t_loop_other = t_loop - (t_dot + t_axpy + t_spmv + t_precon + t_GS + t_givens + t_gemv + t_calcres);
        bool   loop_other_clamped = false;
        if (t_loop_other < 0.0) { t_loop_other = 0.0; loop_other_clamped = true; }
        double t_postloop   = t_solver - t_init - t_loop;
        std::cout<<std::endl<<" ----------------- "<<std::endl;
        std::cout<<"pgmres time           = "<<t_solver     <<" [sec]  (total solver wall time)"<<std::endl;
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
        std::cout<<"    loop GS           = "<<t_GS         <<" [sec]  (Gram-Schmidt orthonormalization)"<<std::endl;
        std::cout<<"    loop givens       = "<<t_givens     <<" [sec]  (Givens rotation, Hess matrix update)"<<std::endl;
        std::cout<<"    loop gemv         = "<<t_gemv       <<" [sec]  (solution update: x += V*y)"<<std::endl;
        std::cout<<"    loop other        = "<<t_loop_other <<" [sec]  (kernel launch/sync, cout per iter, ...)"<<std::endl;
        if (loop_other_clamped) {
          std::cout<<"      NOTE: sub-timers overlap; 'loop other' clamped to 0."<<std::endl;
        }
        std::cout<<"  post-loop time      = "<<t_postloop   <<" [sec]  (vec convert, finish print)"<<std::endl;
	}

	// 収束状態。反復上限で抜けたときは最後の更新をループ内で判定していないので、
	//   同じ量 (真の残差 / ||b||) である finish の値で判定する。
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

#endif

