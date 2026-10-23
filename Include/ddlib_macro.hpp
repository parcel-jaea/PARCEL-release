// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef DDLIBMACRO_HPP_
#define DDLIBMACRO_HPP_

// double-double 演算のマクロ。カーネルの中で展開して使う。
//   (1) 積の誤差なし変換 (TwoProd) は ddlib::two_prod (ddlib.hpp) で行う。Dekker の分割
//       (2^27+1 を掛けて上下に分ける) は、acpp が付ける -ffp-contract=fast で GPU の
//       バックエンドが積と和を FMA に融合すると、分割が崩れたり誤差項を二重に足したりして
//       dd の積が double 相当に落ちる。two_prod は fma で積と誤差項を作るので縮約されても
//       値が変わらない。
//   (2) 各 calc マクロは { } で囲み、先頭に PARCEL_DD_FP_BLOCK を置く。マクロはカーネル
//       (各 *_py.hpp, mixed_precision_dd*.hpp, parcel_precon_dd*.hpp など) の中で展開
//       されるので、ddlib.hpp 側の pragma はここには効かない。ブロック先頭の pragma は
//       そのブロックだけに効くので、利用者のコードの浮動小数点の扱いは変えない。
//       TwoSum / Fast-TwoSum の列を結合則の書き換えと縮約から守る保険
//       (-ffp-contract=fast のバックエンド融合はこれでは止まらない。止めているのは
//       (1) で融合できる積を残さないこと)。
//   マクロの展開先で ddlib::two_prod を使うので、ddlib.hpp を include してから使うこと
//   (このファイルは ddlib.hpp からだけ include される)。
//
//   PARCEL_DD_FP_BLOCK: clang では float_control(precise) と fp contract(off)。
//   CUDA のデバイス側 (__CUDA_ARCH__) では float_control は無視され警告になるだけなので
//   付けない (__NVPTX__ はホスト側でも定義されるので使えない)。clang 以外では空。
#if defined(__clang__) && !defined(__CUDA_ARCH__)
#define PARCEL_DD_FP_BLOCK _Pragma("float_control(precise, on)") _Pragma("clang fp contract(off)")
#else
#define PARCEL_DD_FP_BLOCK
#endif

#define ddadd_calc1(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_ddadd_1 = a_hi;X1L_ddadd_1 = a_lo;X2H_ddadd_1 = b_hi;X2L_ddadd_1 = b_lo;p1_ddadd_1 = X1H_ddadd_1 + X2H_ddadd_1;p2_ddadd_1 = p1_ddadd_1  - X1H_ddadd_1;T1L_ddadd_1 = (X1H_ddadd_1 - (p1_ddadd_1 - p2_ddadd_1) ) + ( X2H_ddadd_1 - p2_ddadd_1 );T1L_ddadd_1 = T1L_ddadd_1 + X1L_ddadd_1 + X2L_ddadd_1;s1_ddadd_1  = p1_ddadd_1+T1L_ddadd_1;c_hi = s1_ddadd_1;c_lo = T1L_ddadd_1 - (s1_ddadd_1 - p1_ddadd_1); }
#define ddadd_var1 T1H_ddadd_1, T1L_ddadd_1, X1H_ddadd_1, X1L_ddadd_1, X2H_ddadd_1, X2L_ddadd_1, p1_ddadd_1, p2_ddadd_1, s1_ddadd_1;

#define ddsub_calc1(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_ddsub_1=a_hi;X1L_ddsub_1=a_lo;X2H_ddsub_1=b_hi;X2L_ddsub_1=b_lo;p1_ddsub_1=X1H_ddsub_1-X2H_ddsub_1;p2_ddsub_1=p1_ddsub_1-X1H_ddsub_1;T1L_ddsub_1=(X1H_ddsub_1-(p1_ddsub_1-p2_ddsub_1))-(X2H_ddsub_1+p2_ddsub_1);T1H_ddsub_1=p1_ddsub_1;T1L_ddsub_1=T1L_ddsub_1+X1L_ddsub_1-X2L_ddsub_1;s1_ddsub_1=T1H_ddsub_1+T1L_ddsub_1;T1L_ddsub_1=T1L_ddsub_1-(s1_ddsub_1-T1H_ddsub_1);T1H_ddsub_1=s1_ddsub_1;c_hi=T1H_ddsub_1;c_lo=T1L_ddsub_1; }
#define ddsub_var1 T1H_ddsub_1,T1L_ddsub_1,X1H_ddsub_1,X1L_ddsub_1,X2H_ddsub_1,X2L_ddsub_1,p1_ddsub_1,p2_ddsub_1,s1_ddsub_1

#define ddmul_calc1(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_ddmul_1=a_hi;X1L_ddmul_1=a_lo;X2H_ddmul_1=b_hi;X2L_ddmul_1=b_lo;::ddlib::two_prod(X1H_ddmul_1,X2H_ddmul_1,T1H_ddmul_1,T1L_ddmul_1);T1L_ddmul_1=T1L_ddmul_1+(X1H_ddmul_1*X2L_ddmul_1+X1L_ddmul_1*X2H_ddmul_1);s1_ddmul_1=T1H_ddmul_1+T1L_ddmul_1;T1L_ddmul_1=T1L_ddmul_1-(s1_ddmul_1-T1H_ddmul_1);T1H_ddmul_1=s1_ddmul_1;c_hi=T1H_ddmul_1;c_lo=T1L_ddmul_1; }
#define ddmul_var1 T1H_ddmul_1,T1L_ddmul_1,X1H_ddmul_1,X1L_ddmul_1,X2H_ddmul_1,X2L_ddmul_1,p1_ddmul_1,p2_ddmul_1,r1_ddmul_1,r2_ddmul_1,u1_ddmul_1,s1_ddmul_1

// ------------


// ---------
#define ddadd_calc2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_ddadd_2=a_hi;X1L_ddadd_2=a_lo;X2H_ddadd_2=b_hi;X2L_ddadd_2=b_lo;p1_ddadd_2=X1H_ddadd_2+X2H_ddadd_2;p2_ddadd_2=p1_ddadd_2-X1H_ddadd_2;T1L_ddadd_2=(X1H_ddadd_2-(p1_ddadd_2-p2_ddadd_2))+(X2H_ddadd_2-p2_ddadd_2);T1H_ddadd_2=p1_ddadd_2;T1L_ddadd_2=T1L_ddadd_2+X1L_ddadd_2+X2L_ddadd_2;s1_ddadd_2=T1H_ddadd_2+T1L_ddadd_2;T1L_ddadd_2=T1L_ddadd_2-(s1_ddadd_2-T1H_ddadd_2);T1H_ddadd_2=s1_ddadd_2;c_hi=T1H_ddadd_2;c_lo=T1L_ddadd_2; }

#define ddadd_var2 T1H_ddadd_2, T1L_ddadd_2, X1H_ddadd_2, X1L_ddadd_2, X2H_ddadd_2, X2L_ddadd_2, p1_ddadd_2, p2_ddadd_2, s1_ddadd_2

#define ddmul_calc2(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_ddmul_2=a_hi;X1L_ddmul_2=a_lo;X2H_ddmul_2=b_hi;X2L_ddmul_2=b_lo;::ddlib::two_prod(X1H_ddmul_2,X2H_ddmul_2,T1H_ddmul_2,T1L_ddmul_2);T1L_ddmul_2=T1L_ddmul_2+(X1H_ddmul_2*X2L_ddmul_2+X1L_ddmul_2*X2H_ddmul_2);s1_ddmul_2=T1H_ddmul_2+T1L_ddmul_2;T1L_ddmul_2=T1L_ddmul_2-(s1_ddmul_2-T1H_ddmul_2);T1H_ddmul_2=s1_ddmul_2;c_hi=T1H_ddmul_2;c_lo=T1L_ddmul_2; }

#define ddmul_var2 T1H_ddmul_2,T1L_ddmul_2,X1H_ddmul_2,X1L_ddmul_2,X2H_ddmul_2,X2L_ddmul_2,p1_ddmul_2,p2_ddmul_2,r1_ddmul_2,r2_ddmul_2,u1_ddmul_2,s1_ddmul_2

#define dddiv_calc1(a_hi,a_lo,b_hi,b_lo,c_hi,c_lo) { PARCEL_DD_FP_BLOCK X1H_dddiv_1=a_hi;X1L_dddiv_1=a_lo;X2H_dddiv_1=b_hi;X2L_dddiv_1=b_lo;q1_dddiv_1=X1H_dddiv_1/X2H_dddiv_1;::ddlib::two_prod(X2H_dddiv_1,q1_dddiv_1,T1H_dddiv_1,T1L_dddiv_1);T1L_dddiv_1=T1L_dddiv_1+(X2L_dddiv_1*q1_dddiv_1);s1_dddiv_1=T1H_dddiv_1+T1L_dddiv_1;T1L_dddiv_1=T1L_dddiv_1-(s1_dddiv_1-T1H_dddiv_1);T1H_dddiv_1=s1_dddiv_1;s1_dddiv_1=X1H_dddiv_1-T1H_dddiv_1;r1_dddiv_1=s1_dddiv_1-X1H_dddiv_1;s2_dddiv_1=(X1H_dddiv_1-(s1_dddiv_1-r1_dddiv_1))-(T1H_dddiv_1+r1_dddiv_1);s2_dddiv_1=s2_dddiv_1-T1L_dddiv_1;s2_dddiv_1=s2_dddiv_1+X1L_dddiv_1;q2_dddiv_1=(s1_dddiv_1+s2_dddiv_1)/X2H_dddiv_1;T1H_dddiv_1=q1_dddiv_1+q2_dddiv_1;T1L_dddiv_1=q2_dddiv_1-(T1H_dddiv_1-q1_dddiv_1);c_hi=T1H_dddiv_1;c_lo=T1L_dddiv_1; }

#define dddiv_var1 X1H_dddiv_1,X1L_dddiv_1,X2H_dddiv_1,X2L_dddiv_1,T1H_dddiv_1,T1L_dddiv_1,p1_dddiv_1,p2_dddiv_1,q1_dddiv_1,q2_dddiv_1,r1_dddiv_1,r2_dddiv_1,s1_dddiv_1,s2_dddiv_1,u1_dddiv_1
#endif
