// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef DDLIB_HPP_
#define DDLIB_HPP_

#include "ddlib_macro.hpp"
#include <cmath>

// dd の演算は結合則による書き換えで誤差項が消えるので、この範囲は利用者が -ffast-math などを
//   付けても厳密な浮動小数点の意味で計算させる (clang)。parcel_sycl.hpp は -ffast-math を既定で
//   拒否するので、これは PARCEL_ALLOW_FAST_MATH で解除した場合の保険 (デバイス側カーネルの
//   dd 演算は別ファイルにあり、ここでは守れない)。
// fp contract(off) も付ける (float_control(precise) は 1 つの式の中の縮約を許す)。ただし acpp は
//   -ffp-contract=fast を付けるため、clang のバックエンドが積と和を FMA に融合するのは pragma では
//   止まらない (FCC は止まる)。dd の積が壊れないのは two_prod (下) を fma で組んでいるため。
//   積和の縮約 (FMA) の禁止 (fp contract(off)) はファイルの範囲に置かず、dd の各関数の本体の先頭
//   (PARCEL_DD_NO_CONTRACT) に置く。clang の fp contract は push / pop を持たず、ファイルの範囲に置くと
//   翻訳単位の終わりまで効く。float_control(pop) がそれを戻すのは float_control を知る clang 11 以降だけで、
//   富士通 FCC の clang モード (LLVM 7 ベース) は float_control を知らずに無視するため、以前は dd の後に続く
//   PARCEL の全カーネル (SpMV・axpy・内積・前処理) で FMA が禁止されていた (A64FX で確認)。
//   複合文の先頭に置いた fp contract はその複合文の中だけで効く (FCC の LLVM 7 と clang 18 で確認)。
//   PARCEL_DD_NO_CONTRACT は fp contract だけを置く (float_control を知らない FCC で警告を増やさない)。
//   fp contract も float_control と同じく CUDA のデバイス側では付けない (以前と同じ条件)。
//   CUDA のデバイス側 (__CUDA_ARCH__) では float_control は無視され警告になるだけなので付けない
//   (__NVPTX__ はホスト側でも定義されるので条件に使えない)。
#if defined(__clang__) && !defined(__CUDA_ARCH__)
#pragma float_control(precise, on, push)
#define PARCEL_DD_NO_CONTRACT _Pragma("clang fp contract(off)")
#else
#define PARCEL_DD_NO_CONTRACT
#endif
namespace ddlib{

  constexpr double split_C = 134217729.0;
  constexpr double ONE     = 1.0;
  constexpr double ZERO    = 0.0;
  constexpr double HALF    = 0.5;

  // 誤差なし変換の積 (TwoProd)。 p = fl(a*b), e = a*b - p (厳密)。
  //   ddtype::mul / div と ddlib_macro.hpp の ddmul / ddaxpy / dddiv マクロはすべてここを通る。
  //
  //   e = fma(a, b, -p) は a*b を丸めずに p を引くので、縮約されてもされなくても厳密
  //   (Dekker の分割は途中の積と差を別々に丸めることが前提で、FMA に融合されると壊れる)。
  //   p も fma(a, b, 0) で作る。p を a*b のままにすると、後段の s1 = p + e や s1 - p が
  //   FMA に融合され、誤差項を二重に足す (GPU で起きる)。fma の結果はそれ以上融合されない。
  //   fma(a, b, +0) は a*b の丸めと同じ値 (積がちょうど -0 のときだけ +0 になる)。
  //   ハードウェア FMA のある CPU では積 1 回と同じ速さ。
  //
  //   fma は SYCL のヘッダを先に読んでいれば cl::sycl::fma (AdaptiveCpp ではホストでは
  //   std::fma、CUDA / HIP のデバイスでは fma 命令)、そうでなければ std::fma。
  //   PARCEL_DD_NO_FMA を定義すると Dekker の分割を使う (FMA を持たない環境向け。
  //   GPU では上記の融合で壊れるので使わないこと)。
  inline double dd_fma(const double a, const double b, const double c){
#if defined(SYCL_LANGUAGE_VERSION)
    return cl::sycl::fma(a, b, c);
#else
    return std::fma(a, b, c);
#endif
  }

  inline void two_prod(const double a, const double b, double &p, double &e){
    PARCEL_DD_NO_CONTRACT
#if defined(PARCEL_DD_NO_FMA)
    const double T1H = a * b;
    double u1 = split_C*a;
    const double p1 = u1 - (u1-a);
    const double p2 = a - p1;
    u1 = split_C*b;
    const double r1 = u1 - (u1-b);
    const double r2 = b - r1;
    p = T1H;
    e = ((p1*r1-T1H) + p1*r2 + p2*r1) + p2*r2;
#else
    p = dd_fma(a, b, 0.0);
    e = dd_fma(a, b, -p);
#endif
  }

  class ddtype{
    double hi;
    double lo;
    
  public:
    operator double() { return static_cast<double>(hi+lo); };
    operator long double() { return static_cast<long double>(hi+lo); };
    operator float() { return static_cast<float>(hi+lo); };

    ddtype(){
      hi = 0.0;
      lo = 0.0;
    }
    
    template<class T>
    explicit ddtype(T hi_){
      hi = static_cast<double> (hi_);
      lo = static_cast<double> (0);
    }

    ddtype(double hi_,double lo_){
      hi = hi_;
      lo = lo_;
    }

    inline void set_dd(double hi_,double lo_) {
      hi = hi_;
      lo = lo_;
    }

    inline ddtype get_dd() const{
      ddtype ret(hi,lo);
      return ret;
    }

    template<class T = double>
    inline T get_T() const{
      T ret = static_cast<T>(hi+lo);
      return ret;
    }
    
    inline double get_hi() const{
      return hi;
    }

    inline double get_lo() const{
      return lo;
    }

    // -------------------------
    inline  ddtype add(
		      const double a_hi, const double a_lo, 
		      const double b_hi, const double b_lo ) const{
      PARCEL_DD_NO_CONTRACT

      double X1H  = a_hi;
      double X1L  = a_lo;
      double X2H  = b_hi;
      double X2L  = b_lo;
      double p1   = X1H + X2H;
      double p2   = p1 - X1H;
      double T1L  = (X1H - (p1 - p2)) + (X2H - p2);
      double T1H  = p1;
      T1L  = T1L + X1L + X2L;
      double s1   = T1H + T1L;
      T1L  = T1L - (s1 - T1H);
      T1H  = s1;

      ddtype ddtype_(T1H,T1L);
      return ddtype_;
    }

    inline  ddtype sub(
		      const double a_hi, const double a_lo, 
		      const double b_hi, const double b_lo ) const{
      PARCEL_DD_NO_CONTRACT

      double X1H  = a_hi;
      double X1L  = a_lo;
      double X2H  = b_hi;
      double X2L  = b_lo;
      double p1   = X1H - X2H;
      double p2   = p1 - X1H;
      double T1L  = (X1H - (p1 - p2)) - (X2H + p2);
      double T1H  = p1;
      T1L  = T1L + X1L - X2L;
      double s1   = T1H + T1L;
      T1L  = T1L - (s1 - T1H);
      T1H  = s1;

      ddtype ddtype_(T1H,T1L);
      return ddtype_;      
    }

    inline  ddtype mul(
		      const double a_hi, const double a_lo, 
		      const double b_hi, const double b_lo ) const{
      PARCEL_DD_NO_CONTRACT

      double X1H  =  a_hi;
      double X1L  =  a_lo;
      double X2H  =  b_hi;
      double X2L  =  b_lo;
      double T1H, T1L;
      two_prod(X1H, X2H, T1H, T1L);
      T1L  = T1L + (X1H * X2L + X1L * X2H);
      double s1   =  T1H + T1L;
      T1L  = T1L - (s1 - T1H);
      T1H  = s1;

      ddtype ddtype_(T1H,T1L);
      return ddtype_;      
    }

    inline  ddtype div(
		      const double a_hi, const double a_lo, 
		      const double b_hi, const double b_lo ) const{
      PARCEL_DD_NO_CONTRACT
      
      double X1H  = a_hi;
      double X1L  = a_lo;
      
      double X2H  = b_hi;
      double X2L  = b_lo;
      double q1   = X1H/X2H;
      double T1H, T1L;
      two_prod(X2H, q1, T1H, T1L);
      T1L  = T1L + (X2L * q1);
      double s1   = T1H + T1L;
      T1L  = T1L - (s1 - T1H);
      T1H  = s1;
      s1   = X1H - T1H;
      double r1   = s1 - X1H;
      double s2   = (X1H - (s1 - r1)) - (T1H + r1);
      s2   = s2 - T1L;
      s2   = s2 + X1L;
      double q2   = (s1 + s2)/X2H;
      T1H  = q1 + q2;
      T1L  = q2 - (T1H - q1);

      ddtype ddtype_(T1H,T1L);
      return ddtype_;      
   
    }

    inline ddtype operator +(const ddtype a) const{
      return add( hi,lo,a.get_hi(),a.get_lo());
    }

    inline ddtype operator *(const ddtype a) const{
      return mul( hi,lo,a.get_hi(),a.get_lo());
    }
    inline ddtype operator -(const ddtype a) const{
      return sub( hi,lo,a.get_hi(),a.get_lo());
    }
    inline ddtype operator /(const ddtype a) const{
      return div( hi,lo,a.get_hi(),a.get_lo());
    }

  };

  struct ddplus {
    ddlib::ddtype operator()(const ddlib::ddtype &x, const ddlib::ddtype &y) 
      const { return x + y; }
  };


  class ddmath{

  public:

    inline ddtype sqrt(ddtype a) const{
      PARCEL_DD_NO_CONTRACT
      
      double a_hi = a.get_hi(); 
      double a_lo = a.get_lo(); 
      if( (a_hi==ZERO) && (a_lo==ZERO) ){
	double b_hi = ZERO;
	double b_lo = ZERO;
      }
      if ( (a_hi<ZERO ) ||  ( (a_hi==ZERO) && (a_lo<ZERO) ) ) {
	std::cout<<"SQRT CALLED WITH NEGATIVE ARGUMENT"<<std::endl;
	double b_hi = ZERO;
	double b_lo = ZERO;
	ddtype ddtype_(b_hi,b_lo);
	return ddtype_;
      }

      double x  = ONE/std::sqrt(a_hi);
      double ax = a_hi * x;

      ddtype ax_(ax,ZERO);
      ddtype x_(x,ZERO);
      ddtype HALF_(HALF,ZERO);

      ddtype y  = x_  * HALF_;
      ddtype z  = ax_ * ax_;
      ddtype T2 = a-z;
      ddtype T1 = T2*y;

      ddtype b  = T1+ax_;
       			
      return b;
    }

  };


}
#if defined(__clang__) && !defined(__CUDA_ARCH__)
#pragma float_control(pop)
#endif
#undef PARCEL_DD_NO_CONTRACT

// c++ compiler
#ifdef __INTEL_COMPILER

#pragma optimize("", on)
#else
#ifdef __GNUC__
#pragma GCC reset_options
#endif

#endif

// llvm 
#ifdef __clang__

#ifdef __INTEL_LLVM_COMPILER

#else
#ifdef __CUDACC__

#endif

#ifdef DPCPP_CUDA
  // CUDAバックエンドでは最適化しても問題無し

#else
#pragma clang optimize on
#endif

#endif

#endif

#endif
