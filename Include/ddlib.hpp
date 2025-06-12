#ifndef DDLIB_HPP_
#define DDLIB_HPP_

#include "ddlib_macro.hpp"
#include <cmath>

// // c++ compiler
// #ifdef __INTEL_COMPILER
// #pragma optimize("",off)
// #else
// #ifdef __GNUC__
// #pragma GCC optimize ("O3")
// #endif
// #endif

// // llvm 
// #ifdef __clang__

// #ifdef __INTEL_LLVM_COMPILER
// // Intel llvmでうまくいかないのでデフォルトでfp-modeを指定する
// // #pragma clang optimize off

// #else

// #ifdef DPCPP_CUDA
//   // CUDAバックエンドでは最適化しても問題無し

// #else
// #pragma clang optimize off
// #endif

// #endif

// #endif


namespace ddlib{

  constexpr double split_C = 134217729.0;
  constexpr double ONE     = 1.0;
  constexpr double ZERO    = 0.0;
  constexpr double HALF    = 0.5;

  class ddtype{
    //    ddlib::ddfunc ddfunc_;
    // mutable double hi;
    // mutable double lo;
    double hi;
    double lo;
    
  public:
    operator double() { return static_cast<double>(hi+lo); };
    operator long double() { return static_cast<long double>(hi+lo); };
    operator float() { return static_cast<float>(hi+lo); };

    //    operator @() { return static_cast<@>(hi+lo); };
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

    //    explicit ddtype(ddtype a_){
    // ddtype(ddtype a_) const{
    //   hi = a_.get_hi();
    //   lo = a_.get_lo();
    // }

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

      double X1H  =  a_hi;
      double X1L  =  a_lo;
      double X2H  =  b_hi;
      double X2L  =  b_lo;
      double T1H  =  X1H * X2H;
      double u1   =  split_C*X1H;
      double p1   =  u1 - (u1-X1H);
      double p2   =  X1H - p1;
      u1   =  split_C*X2H;
      double r1   =  u1 - (u1-X2H);
      double r2   =  X2H - r1;
      double T1L  = ((p1*r1-T1H) + p1*r2 + p2*r1) + p2*r2;
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
      
      double X1H  = a_hi;
      double X1L  = a_lo;
      
      double X2H  = b_hi;
      double X2L  = b_lo;
      double q1   = X1H/X2H;
      double T1H  = X2H * q1;
      double u1   = split_C*X2H;
      double p1   = u1 - (u1-X2H);
      double p2   = X2H - p1;
      u1   = split_C*q1;
      double r1   = u1 - (u1-q1);
      double r2   = q1 - r1;
      double T1L  = ((p1*r1-T1H) + p1*r2 + p2*r1) + p2*r2;
      T1L  = T1L + (X2L * q1);
      double s1   = T1H + T1L;
      T1L  = T1L - (s1 - T1H);
      T1H  = s1;
      s1   = X1H - T1H;
      r1   = s1 - X1H;
      double s2   = (X1H - (s1 - r1)) - (T1H + r1);
      s2   = s2 - T1L;
      s2   = s2 + X1L;
      double q2   = (s1 + s2)/X2H;
      T1H  = q1 + q2;
      T1L  = q2 - (T1H - q1);

      ddtype ddtype_(T1H,T1L);
      return ddtype_;      
   
    }

      // ddtype &operator +=(const ddtype &a) const{
      //   const ddtype tmp = add( hi,lo,a.get_hi(),a.get_lo());
      //   hi = tmp.get_hi();
      //   lo = tmp.get_lo();
      //   return *this;
      // }

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

// #pragma clang optimize on

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


// #pragma GCC optimize ("O3")
// #pragma optimize("",off)
