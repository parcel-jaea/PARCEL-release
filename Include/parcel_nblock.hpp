// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_NBLOCK_HPP_
#define PARCEL_NBLOCK_HPP_

#include <cstddef>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

// ブロック化カーネル (_BLOCKING_LOOP_ 系) の分割数。
//
//   work item 1 個 = スレッド 1 本 = コア 1 個 になるよう、
//   MPI ランクあたりの物理コア数を指定する。
//   既定の 12 は A64FX の CMG あたりのコア数。
//
//   例:  A64FX  48 コア / 4 CMG        → 12 (既定)
//        Xeon MAX 56 コア / 4 ランク   → -DPARCEL_NBLOCK=14
//        Xeon 6980P 128 コア / 4 ランク → -DPARCEL_NBLOCK=32
//
//   1 work item あたりの要素数 (nlocalblock) はカーネルごとに
//   個別にチューニングされているため、ここでは触らない。
//   スタック消費量が nlocalblock に比例する点にも注意 (KMP_STACKSIZE)。
#ifndef PARCEL_NBLOCK
#define PARCEL_NBLOCK 12
#endif

// ブロック化カーネルの分割数を実行時に決める。
//
//   work item 1 個 = スレッド 1 本 になるのが狙いなので、実行時のスレッド数を
//   そのまま使う。
//
//   優先順位:
//     1. 環境変数 PARCEL_NBLOCK   明示指定。性能調査で値を振るときに使う
//     2. omp_get_max_threads()    OpenMP バックエンド。OMP_NUM_THREADS に従う
//     3. コンパイル時の PARCEL_NBLOCK   上記が使えない場合の最終手段
//
//   分割数がスレッド数より多い分は動く (部分和が増えて最終縮約が伸びるだけ)。
//   少ないとコアが遊ぶので、迷う場合は多い側に倒すこと。
//
//   1 回だけ決めて以後は使い回す。カーネルごとに getenv を呼ぶと反復のたびに
//   費用がかかる。ホスト側でのみ呼ぶこと (カーネル内からは呼ばない)。
namespace parcel {

  inline std::size_t nblock_runtime(){
    static const std::size_t n_ = [](){
      if(const char *e = std::getenv("PARCEL_NBLOCK")){
	const long v = std::atol(e);
	if(v > 0){ return (std::size_t)v; }
      }
#ifdef _OPENMP
      const int t = omp_get_max_threads();
      if(t > 0){ return (std::size_t)t; }
#endif
      return (std::size_t)PARCEL_NBLOCK;
    }();
    return n_;
  }

}

// 縮約カーネル (dot / dddot / max) をブロック化するスイッチ。
//
//   平坦版の dot / dddot は nd_range + barrier でツリー削減する。AdaptiveCpp の OpenMP バックエンドは
//   barrier を boost::fiber で実装するため、512 work group x 512 work item に
//   9 段のバリアを張る固定費が乗る (問題サイズに依らず 1 回 約 1 ms)。
//   ブロック化版は barrier もバッファも使わず、部分和を work item ごとに持つ。
//   CPU では dot が 1.6〜2.8 倍、ソルバ全体が 1.1〜1.5 倍速くなる。
//
//   部分和の個数は実行構成 (ランク数 x スレッド数) で変わるので、縮約順序も
//   それで変わる。倍精度では結果に出ないが、前処理のデータ精度を fp32 にすると
//   丸めの差が反復数に現れることがある。精度を下げた構成を環境間で比べるときは、
//   このフラグと実行構成を揃えること。
//
//   max も同じ形のツリー削減で、スケーリング有効時 (前処理 BLOCKJACOBI /
//   JACOBI) に毎反復呼ばれるため、dot と同じスイッチで切り替える。
//
//   _BLOCKING_LOOP_ は 9 カーネルすべてを差し替えるが、効果は dot / dddot に
//   集中している。変更範囲を絞りたい場合はこちらを使う。
#ifndef _BLOCKING_LOOP_DOT_
#define _BLOCKING_LOOP_DOT_ 0
#endif

// ブロック化した縮約カーネル (dot / dddot / max) を使うかどうか。
// _BLOCKING_LOOP_ (全カーネル) でも有効になる。
#if _BLOCKING_LOOP_ || _BLOCKING_LOOP_DOT_
#define PARCEL_BLOCKED_DOT 1
#else
#define PARCEL_BLOCKED_DOT 0
#endif

// Krylov 基底の列の先頭間隔に入れる詰め物 [要素]。
//   ks_arnoldi の基底 V / V_new と pgmres (MPI) の基底 v は、列を stride = n + nadd + PARCEL_BASIS_PAD おきに置く。
//   n + nadd が 2 の大きな冪の倍数になると、gemv / 縦長 gemm で同時に読む多数の列が同じキャッシュセットに
//   落ちて CPU で遅くなる。64 バイト (double 8 要素) ずらすとこれを避けられる。
//   詰め物は列の後ろ (halo のさらに後ろ) に置くだけで、計算の中身は変わらない。
//   既定は CPU のビルド (PARCEL_BLOCKED_DOT) だけ 8、GPU は 0 (GPU では衝突が起きず、詰め物で列の先頭の
//   整列がずれると逆に少し遅くなる)。
#ifndef PARCEL_BASIS_PAD
#if PARCEL_BLOCKED_DOT
#define PARCEL_BASIS_PAD 8
#else
#define PARCEL_BASIS_PAD 0
#endif
#endif

// CPU 向けの行ブロック版 gemv / 縦長 gemm (tgemm / tgemm_beta) を使うか。
//   「1 行 = 1 ワークアイテムで M 本の列を同時に読む」形は CPU ではメモリ帯域を使い切れない。
//   行ブロック版は 1 ワークアイテムが行のブロックを受け持ち、ブロックの中を「列が外側・行が内側」で
//   回して局所配列に積む。読む列は常に 1 本で、行方向にベクトル化される。
//   各要素の足し合わせの順序と式は 1 行 = 1 ワークアイテムの版と同じ (列の昇順に積み、最後に同じ式で合成)。
//   既定は CPU のビルド (PARCEL_BLOCKED_DOT、すなわち _BLOCKING_LOOP_ / _BLOCKING_LOOP_DOT_) だけ 1。
#ifndef PARCEL_CPU_ROWBLOCK
#define PARCEL_CPU_ROWBLOCK PARCEL_BLOCKED_DOT
#endif

// MGS (pgmres、ks_arnoldi の set_qr_algo(QR_MGS)) の「前のステップの axpy」と「このステップの内積」を
//   1 本のカーネルにまとめるか。
//   CPU (OpenMP バックエンド) ではカーネル 1 本の起動と待ちの固定費が MGS の 1 ステップの大半を占める。
//   各要素の演算と順序は 2 本に分けた場合と同じ (ブロック版 dot と同じ分割・足し順)。
//   ブロック版の内積 (PARCEL_BLOCKED_DOT) があるビルドだけで使える。既定は CPU のビルドで 1、GPU は 0。
#ifndef PARCEL_GS_FUSED
#define PARCEL_GS_FUSED PARCEL_BLOCKED_DOT
#endif
#if PARCEL_GS_FUSED && !PARCEL_BLOCKED_DOT
#error "PARCEL_GS_FUSED=1 needs the blocked CPU dot (_BLOCKING_LOOP_ or _BLOCKING_LOOP_DOT_)"
#endif

// CG の 1 反復のカーネルの融合 (PARCEL_CG_FUSED)。A64FX の経路 (_BLOCKING_LOOP_) だけで既定 1、それ以外 (x86 の CPU・GPU) は 0。
//   (a) pstep = 1 (既定) のとき、x の更新 (x = x + alpha p。gemv の axpy 分解版 M = 1) と p の更新 (p = s + beta p) を 1 本のカーネルで
//       (MPI ありのビルドだけ)。
//   (b) r の更新 (r = r - alpha q) と <r, r> を 1 本のカーネルで (MGS の axpy_dot_value。PARCEL_GS_FUSED が要る)。
//   各要素の式と足す順は分けた場合と同じで、結果はビット単位で一致する (A64FX、200 反復の残差の推移で確認)。
//   対象は係数を値で渡す型の組み合わせ (double-double を含まない) で、TRUE_RESIDUAL (RESIDUAL_REPLACE) のときは (a) を使わない。
//   時間の内訳では、(a) は axpy、(b) は内積に入る。-DPARCEL_CG_FUSED=0 で切れる。
//   融合したカーネルは A64FX 向けのファイル (mixed_precision_axpy_a64fx.hpp) にあるので、_BLOCKING_LOOP_ のビルドでしか 1 にできない。
// A64FX 向けの最適化 C (axpy_value / axpby_value の A64FX 版の本体 parcel_axpy_value_rr_ と、gemv の axpy 分解版の
//   restrict 付きの書き直し) を使うか。1 (既定): 書き直した版。0: 書き直す前の形
//   (axpy_value は nblock x nlocalblock の飛び飛びの分割で元の式, gemv の分解版は係数・x を内側のループで読む元の形)。
//   各要素の式と足す順は同じで, 結果はビット単位で同じ。_BLOCKING_LOOP_ のビルドでだけ意味がある。
#ifndef PARCEL_A64FX_AXPY_C
#define PARCEL_A64FX_AXPY_C 1
#endif

#ifndef PARCEL_CG_FUSED
#if defined(_BLOCKING_LOOP_) && _BLOCKING_LOOP_ && PARCEL_GS_FUSED
#define PARCEL_CG_FUSED 1
#else
#define PARCEL_CG_FUSED 0
#endif
#endif
#if PARCEL_CG_FUSED && !((defined(_BLOCKING_LOOP_) && _BLOCKING_LOOP_) && PARCEL_GS_FUSED)
#error "PARCEL_CG_FUSED=1 needs _BLOCKING_LOOP_=1 and PARCEL_GS_FUSED=1"
#endif

// ks_arnoldi のブロック直交化 (CGS2) で、1 パスの局所内積 j+1 本を 1 本のカーネルにまとめるか。
//   まとめたカーネルは内積ごとにブロック版 dot と同じ分割・同じ sum_local の添字・同じ最後の足し順で
//   計算するので、各内積の値は j+1 回に分けた場合とビット単位で一致する。
//   既定は CPU のビルド (PARCEL_BLOCKED_DOT) で 1、GPU は 0。
#ifndef PARCEL_KS_FUSED
#define PARCEL_KS_FUSED PARCEL_BLOCKED_DOT
#endif
#if PARCEL_KS_FUSED && !PARCEL_BLOCKED_DOT
#error "PARCEL_KS_FUSED=1 needs the blocked CPU dot (_BLOCKING_LOOP_ or _BLOCKING_LOOP_DOT_)"
#endif

// SpMV の内側ループを梯子状に分割する (既定は無効)。
//
//   CRS の内側ループは Rowptr で回数が決まるため、コンパイラが回数を知らず
//   展開も先読みもできない。4 -> 2 -> 1 と段階的に処理すると、どの非零数でも
//   全量が固定長の本体に入る (nnz=7 なら 4+2+1)。行あたり非零数が一定である
//   必要はなく、追加メモリも要らない。
//
//   GPU 向けビルドで有効にする (試験の生成器 tests/regression/gen_regression.py,
//   tests/perf/gen_perf.py が付ける)。A64FX では SpMV 側だけ有効にする。
#ifndef PARCEL_SPMV_LADDER
#define PARCEL_SPMV_LADDER 0
#endif

// A64FX (_BLOCKING_LOOP_) の SpMV をブロック化し、配列のポインタを restrict 付きのローカル変数に写す。
//   _BLOCKING_LOOP_SPMV_ (二段構え) を使わないときに有効。parallel_for は nblock_runtime() 個 (= スレッド数) の
//   work item で、各 item が連続した ceil(N / nblock) 行を受け持つ。行ごとの計算と足す順序は平坦版
//   (1 行 = 1 work item) と同じ。カーネルのラムダはポインタを取り込んだ閉包としてライブラリに渡るので、
//   コンパイラは y への書き込みがポインタ自体を書き換えうると見てループ内で読み直す。ローカルの restrict
//   ポインタに写すとこれが消える。beta が 0 のときは y を読まない (y = alpha*Ax)。
//   Jacobi 前処理 (precon 4) のカーネル 3 本 (D^-1 b、1 回目、2 回目以降) も _BLOCKING_LOOP_ では同じく
//   連続した行の範囲に分け、restrict 付きのローカル変数に写す (前処理の入力と出力は別の配列であることが前提)。
#if defined(_BLOCKING_LOOP_) && _BLOCKING_LOOP_ && !(defined(_BLOCKING_LOOP_SPMV_) && _BLOCKING_LOOP_SPMV_)
#define PARCEL_SPMV_A64FX_BLOCK_ACTIVE 1
#else
#define PARCEL_SPMV_A64FX_BLOCK_ACTIVE 0
#endif

// 前処理の反復ループを梯子状に分割する (既定は PARCEL_SPMV_LADDER に追従)。
//
//   前処理は setup でシステム行列から自前の CRS 行列を組み立て、反復では
//   それだけを読む (Jacobi: precon_LU_crsMat、Block Jacobi: Lmatrix_crsMat /
//   Umatrix_crsMat)。このループも SpMV と同じ行に沿った累積なので梯子が効く。
#ifndef PARCEL_PRECON_LADDER
#define PARCEL_PRECON_LADDER PARCEL_SPMV_LADDER
#endif

// Jacobi 前処理 (precon 4) の (D^-1)(L+U) は、行列 A の行ポインタ・列番号を共有して持つ (省メモリ)。
//   A と同じ非零の並びで値だけを持ち、対角の位置に 0 を置く。前処理の反復は行ごとに
//   対角の位置 (1 行 1 バイト、parcel_precon_jacobi.hpp の diagoff_) を控えておき、対角を飛ばして
//   A の行の順に x_i - Σ a_ij x_j を足す。位置が 255 以上の行・対角の無い行は対角の 0 も読む。
//   前処理は行列 A より長く生きてはいけない (A の構造を共有するため)。

// Jacobi 前処理 (precon 4) の (D^-1)(L+U) の値を持たず、A の値と D^-1 から前処理のたびに計算する (既定 1、A64FX では 0)。
//   値の配列 (A の非零 1 個あたり前処理行列の型 1 個) が要らなくなる。前処理の計算量は非零 1 個あたり
//   掛け算 1 回と対角の判定が増え、GPU では前処理が遅くなる (CPU ではほぼ変わらない)。
//   0 にすると値を保存する (メモリを使って前処理を速くする)。
//   その場の計算は構築時と同じ型・同じ式 (PARCEL_MAX_FLOTING_TYPE(a_ij) * Dinv_i を前処理行列の型に丸める。
//   対角は 0) なので、値は保存した場合と同じになる。ただし積を反復の式の中で作るので、積和の融合 (FMA) を
//   許すビルドでは融合のされ方が変わって丸めが変わりうる (-ffp-contract=off ではビット単位で一致する)。
//   対象: CRS 行列で、行列スケーリングなし (実行時)、係数行列・前処理行列・前処理の計算型のいずれも
//   double-double でないとき。float の前処理行列も対象。外れる組み合わせは 0 と同じく値を保存する。
//   半精度の前処理 (前処理行列・前処理の計算型) は既定で行列スケーリングが有効になる (parcel_precon.hpp の
//   scaling_default) ので、スケーリングを明示的に切らない限り値を保存する。
//   前処理は反復のたびに A の値を読むので、前処理を使い終えるまで A を破棄・変更してはいけない。
//   前処理のカーネルは「保存した値を読む版」と「その場で計算する版」に分けてあり、どちらを使うかは呼び出す前に
//   1 回だけ決める。値を保存する組み合わせの前処理は 0 のビルドと同じ速さになる。0 のビルドではその場で計算する版を作らない。
// A64FX の経路 (_A64FX_HIPSYCL_ が 1) では既定を 0 (値を保存する) にする。A64FX 1 ノード (NX=256、pcg + ヤコビ前処理) で
//   前処理が 0.77 倍、収束までの総時間が 0.81 倍になった (その場の掛け算と対角の判定が A64FX では隠れない)。
//   代わりにメモリは A の非零 1 個あたり前処理行列の型 1 個 (倍精度なら 8 B) 増える。それ以外の環境の既定は 1。
//   -DPARCEL_JACOBI_ONTHEFLY=0/1 を明示すればどの環境でもそれが優先される。
//   _A64FX_HIPSYCL_ は parcel_crs.hpp で _BLOCKING_LOOP_ / _BLOCKING_LOOP_SPMV_ から決まるが、parcel_crs.hpp は
//   このファイルを読んだ後で定義するので、ここでは同じ規則 (明示の _A64FX_HIPSYCL_ を優先し、無ければブロック化の有無) で判定する。
#ifndef PARCEL_JACOBI_ONTHEFLY
#  if defined(_A64FX_HIPSYCL_)
#    if _A64FX_HIPSYCL_
#      define PARCEL_JACOBI_ONTHEFLY 0
#    else
#      define PARCEL_JACOBI_ONTHEFLY 1
#    endif
#  elif (defined(_BLOCKING_LOOP_) && _BLOCKING_LOOP_) || (defined(_BLOCKING_LOOP_SPMV_) && _BLOCKING_LOOP_SPMV_)
#    define PARCEL_JACOBI_ONTHEFLY 0
#  else
#    define PARCEL_JACOBI_ONTHEFLY 1
#  endif
#endif

// A64FX (_BLOCKING_LOOP_) のヤコビ前処理の最適化。どちらも計算の順序と丸めを変えない (結果はビット単位で同じ)。
//   PARCEL_JACOBI_NODIV_UNIT (既定 1): precon_b = Dinv*b/bnorm の bnorm が 1 (ベクトルのスケーリングなし。倍精度・単精度の既定) のとき、
//     割り算をしない版を呼び出し側で選ぶ (x/1.0 == x)。A64FX の SVE の割り算はパイプライン化されず、前処理の約 3 割を占めていた。
//   PARCEL_JACOBI_ROW2 (既定 1): 前処理行列が float (precon_dataF など) のとき、2 行を交互に処理して 2 本の積和の依存の鎖を重ねる。
//     各行の中の足し順と式は元と同じ。倍精度の行列ではコンパイラが既に行を重ねているので使わない (遅くなる)。
//     その場計算 (OTF) の版と梯子 (PARCEL_PRECON_LADDER=1) では使わない。
//   _BLOCKING_LOOP_ が 0 のビルド (x86 CPU・GPU) のコードは変わらない。
#ifndef PARCEL_JACOBI_NODIV_UNIT
#define PARCEL_JACOBI_NODIV_UNIT 1
#endif
#ifndef PARCEL_JACOBI_ROW2
#if _BLOCKING_LOOP_ && !PARCEL_PRECON_LADDER
#define PARCEL_JACOBI_ROW2 1
#else
#define PARCEL_JACOBI_ROW2 0
#endif
#endif
#if PARCEL_JACOBI_ROW2 && !(_BLOCKING_LOOP_ && !PARCEL_PRECON_LADDER)
#error "PARCEL_JACOBI_ROW2=1 needs _BLOCKING_LOOP_=1 and PARCEL_PRECON_LADDER=0"
#endif

// 同時には使われない作業ベクトルには同じ領域を割り当てる (計算は変わらない)。
//   - Jacobi 前処理の反復用 2 本 (precon_x1_ / precon_x2_) は反復回数 1 (既定) のとき参照されないので 1 要素だけ確保する
//   - pgmres: r と v0 を pv の領域に、x_update と r_true を w の領域に置く (詳細は parcel_krylov_gmres.hpp)
//   - ks: 初期ベクトルの正規化にだけ使う work を z (前処理の出力) の局所部分に置く

#endif
