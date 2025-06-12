#ifndef DDLIB_HPP_
#define DDLIB_HPP_

#include <cmath>

namespace ddlib{
  
  constexpr double split_C = 134217729.0;
  constexpr double ONE     = 1.0;
  constexpr double ZERO    = 0.0;
  constexpr double HALF    = 0.5;

  class ddtype{
    double hi;
    double lo;
  public:
    ddtype(){
      hi = 0.0;
      lo = 0.0;
    }
    
    template<class T>
    ddtype(T hi_){
      hi = static_cast<double> (hi_);
      lo = static_cast<double> (0);
    }

    ddtype(double hi_,double lo_){
      hi = hi_;
      lo = lo_;
    }

    void set_dd(double hi_,double lo_){
      hi = hi_;
      lo = lo_;
    }

    ddtype get_dd() const{
      ddtype ret(hi,lo);
      return ret;
    }

    template<class T>
    T get_T() const{
      T ret = static_cast<T>(hi+lo);
      return ret;
    }
    
    double get_hi() const{
      return hi;
    }
    double get_lo() const{
      return lo;
    }


  };

  class ddfunc{

  public:

    ddtype add(ddtype a,ddtype b) const{

      double a_hi = a.get_hi(); 
      double a_lo = a.get_lo(); 
      double b_hi = b.get_hi(); 
      double b_lo = b.get_lo();

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

    ddtype sub(ddtype a,ddtype b) const{
      double a_hi = a.get_hi(); 
      double a_lo = a.get_lo(); 
      double b_hi = b.get_hi(); 
      double b_lo = b.get_lo();
      
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

    ddtype mul(ddtype a,ddtype b) const{
      double a_hi = a.get_hi(); 
      double a_lo = a.get_lo(); 
      double b_hi = b.get_hi(); 
      double b_lo = b.get_lo();

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

    ddtype div(ddtype a,ddtype b) const{

      double a_hi = a.get_hi(); 
      double a_lo = a.get_lo(); 
      double b_hi = b.get_hi(); 
      double b_lo = b.get_lo();
      
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

    ddtype sqrt(ddtype a) const{
      
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

      ddtype y  = mul( x,ZERO,HALF,ZERO);
      ddtype z  = mul(ax,ZERO,  ax,ZERO);
      ddtype T2 = sub(a,z );
      ddtype T1 = mul(T2,y);
      ddtype ax_(ax,ZERO);
      ddtype b  = add(T1,ax_);
       			
      return b;
    }

    // ----------------------------

    ddtype add(double a_hi,double a_lo,double b_hi,double b_lo) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_hi,b_lo);
      return add(a,b);
    }
    // a :: double-double ,b ::
    template<class T>
    ddtype add(double a_hi,double a_lo,T b_) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_);
      return add(a,b);
    }

    template<class T>
    T add(T a_, T b_) const{
      ddtype a(a_);
      ddtype b(b_);
      ddtype ret = add(a,b);
      return ret.get_T<T>();
    }

    ddtype sub(double a_hi,double a_lo,double b_hi,double b_lo) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_hi,b_lo);
      return sub(a,b);
    }

    template<class T>
    T sub(T a_, T b_) const{
      ddtype a(a_);
      ddtype b(b_);
      ddtype ret = sub(a,b);
      return ret.get_T<T>();
    }

    ddtype mul(double a_hi,double a_lo,double b_hi,double b_lo) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_hi,b_lo);
      return mul(a,b);
    }

    template<class T>
    ddtype mul(double a_hi,double a_lo,T b_) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_);
      return mul(a,b);
    }

    template<class T>
    T mul(T a_, T b_) const{
      ddtype a(a_);
      ddtype b(b_);
      ddtype ret = mul(a,b);
      return ret.get_T<T>();
    }

    ddtype div(double a_hi,double a_lo,double b_hi,double b_lo) const{
      ddtype a(a_hi,a_lo);
      ddtype b(b_hi,b_lo);
      return div(a,b);
    }

    template<class T>
    T div(T a_, T b_) const{
      ddtype a(a_);
      ddtype b(b_);
      ddtype ret = div(a,b);
      return ret.get_T<T>();
    }

    ddtype sqrt(double a_hi,double a_lo) const{
      ddtype a(a_hi,a_lo);
      return sqrt(a);
    }

    template<class T>
    T sqrt(T a_) const{
      ddtype a(a_);
      ddtype ret = sqrt(a);
      return ret.get_T<T>();
    }

  };

}

#endif

