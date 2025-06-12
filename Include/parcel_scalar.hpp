#ifndef PARCEL_SCALAR__
#define PARCEL_SCALAR__

#include <type_traits>
// #include "parcel.hpp"
// #include "parcel_type.hpp"
#include "util_sycl.hpp"

namespace parcel{
  using ddtype = ddlib::ddtype;
  using ddmath = ddlib::ddmath;

  //  template<class T=PARCEL_MAX_FLOTING_TYPE>

  class scalar{
    PARCEL_MAX_FLOTING_TYPE T_;
    ddtype dd_;
  public :
    scalar(){
      T_  = 0.0;
      dd_.set_dd(0.0,0.0);
    };

    scalar(float T){
      T_  = T;
      dd_.set_dd(T,0.0);
    };

    scalar(sycl::half T){
      T_  = T;
      dd_.set_dd(T,0.0);
    };

    scalar(double T){
      T_  = T;
      dd_.set_dd(T,0.0);
    };

    scalar(double T_hi,double T_lo){
      T_  = T_hi + T_lo;
      dd_.set_dd(T_hi,T_lo);
    };

    ~scalar(){};

    PARCEL_MAX_FLOTING_TYPE get_T(){
      return T_;
    }
    ddtype get_dd(){
      return dd_;
    }
    double get_dd_hi(){
      return dd_.get_hi();
    }
    double get_dd_lo(){
      return dd_.get_lo();
    }
    void set_Tdd(scalar Tdd){
      T_ = Tdd.get_T();
      dd_ = Tdd.get_dd();
    }

    void set_Tdd(PARCEL_MAX_FLOTING_TYPE T){
      T_ = T;
      dd_.set_dd(T,0.0);
    }
    void set_Tdd(PARCEL_MAX_FLOTING_TYPE T,ddtype dd){
      T_ = T;
      dd_ = dd;
    }
    void set_T(PARCEL_MAX_FLOTING_TYPE T){
      T_ = T;
    }
    void set_dd(double dd_hi,double dd_lo){
      dd_.set_dd(dd_hi,dd_lo);
    }
    void set_dd(ddtype dd){
      dd_ = dd;
    }

    // -----------------------------
    // 演算子オーバーロード

#if 1

#if 0
    inline const scalar operator =(scalar a)  const {      
      PARCEL_MAX_FLOTING_TYPE T = a.get_T();
      ddtype dd = a.get_dd();
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator =(double a)  const {      
      PARCEL_MAX_FLOTING_TYPE T = a;
      ddtype dd;dd.set_dd((double)a,0.0);
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator =(float a)  const {
      PARCEL_MAX_FLOTING_TYPE T = a;
      ddtype dd;dd.set_dd((double)a,0.0);
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator =(sycl::half a)  const {
      PARCEL_MAX_FLOTING_TYPE T = a;
      ddtype dd;dd.set_dd((double)a,0.0);
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator =(ddtype a)  const {
      PARCEL_MAX_FLOTING_TYPE T = a.get_T();
      ddtype dd = a;
      scalar b;b.set_Tdd(T,dd);
      return b;
    }
#endif
    inline const scalar operator +(scalar a)  const {
      PARCEL_MAX_FLOTING_TYPE T  = T_ + a.get_T();
      ddtype dd = dd_ + a.get_dd();
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator -()  const {
      PARCEL_MAX_FLOTING_TYPE T = -T_;
      ddtype nega_one(-1.0,0.0);
      ddtype dd = dd_ *nega_one;
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator -(scalar a)  const {
      PARCEL_MAX_FLOTING_TYPE T = T_  - a.get_T();
      ddtype dd = dd_ - a.get_dd();
      scalar b;b.set_Tdd(T,dd);
      return b;
    }
    inline const scalar operator *(scalar a)  const {
      PARCEL_MAX_FLOTING_TYPE T = T_ * a.get_T();
      ddtype dd = dd_ * a.get_dd();
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

    inline const scalar operator /(scalar a)  const {
      PARCEL_MAX_FLOTING_TYPE T = T_ / a.get_T();
      ddtype dd = dd_ / a.get_dd();
      scalar b;b.set_Tdd(T,dd);
      return b;
    }

#else

    inline scalar &operator =(double a) {
      T_  = a;
      dd_.set_dd(a,0.0);
      return *this;
    }
    inline scalar &operator =(float a) {
      T_  = a;
      dd_.set_dd(a,0.0);
      return *this;
    }
    inline scalar &operator =(sycl::half a) {
      T_  = a;
      dd_.set_dd(a,0.0);
      return *this;
    }
    inline scalar &operator =(ddtype a) {
      T_  = a.get_T();
      dd_ = a;
      return *this;
    }

    inline scalar &operator +(scalar a) {
      T_  = T_ + a.get_T();
      dd_ = dd_ + a.get_dd();
      return *this;
    }

    inline scalar &operator -() {
      T_  = -T_;
      ddtype dd(-1.0,0.0);
      dd_ = dd_ * dd;
      return *this;
    }

    inline scalar &operator -(scalar a) {
      T_  = T_  - a.get_T();
      dd_ = dd_ - a.get_dd();
      return *this;
    }
    inline scalar &operator *(scalar a) {
      T_  = T_  * a.get_T();
      dd_ = dd_ * a.get_dd();
      return *this;
    }
    inline scalar &operator /(scalar a) {
      T_  = T_  / a.get_T();
      dd_ = dd_ / a.get_dd();
      return *this;
    }
#endif

  };

  //  class scalar_util : public scalar_blas{
  class scalar_util {
    ddmath ddmath_;
  public :     

    scalar add(scalar a,scalar b){
      scalar ret;
      PARCEL_MAX_FLOTING_TYPE a_ = a.get_T();
      PARCEL_MAX_FLOTING_TYPE b_ = b.get_T();

      ddtype a_dd_  = a.get_dd();
      ddtype b_dd_  = b.get_dd();      
      ddtype ret_dd = a_dd_ + b_dd_;

      ret.set_T(a_ + b_);
      ret.set_dd(ret_dd.get_hi(),ret_dd.get_lo());

      return ret;
    }

    scalar div(scalar a,scalar b){
      scalar ret;
      PARCEL_MAX_FLOTING_TYPE a_ = a.get_T();
      PARCEL_MAX_FLOTING_TYPE b_ = b.get_T();

      ddtype a_dd_  = a.get_dd();
      ddtype b_dd_  = b.get_dd();      
      ddtype ret_dd = a_dd_ / b_dd_;

      ret.set_T(a_ / b_);
      ret.set_dd(ret_dd.get_hi(),ret_dd.get_lo());

      return ret;
    }

    scalar mul(scalar a,scalar b){
      scalar ret;
      PARCEL_MAX_FLOTING_TYPE a_ = a.get_T();
      PARCEL_MAX_FLOTING_TYPE b_ = b.get_T();

      ddtype a_dd_  = a.get_dd();
      ddtype b_dd_  = b.get_dd();      
      ddtype ret_dd = a_dd_*b_dd_;

      ret.set_T(a_ * b_);
      ret.set_dd(ret_dd.get_hi(),ret_dd.get_lo());

      return ret;
    }

    scalar abs(scalar a){
      scalar ret;

      PARCEL_MAX_FLOTING_TYPE a_ = a.get_T();
      ddtype a_dd_  = a.get_dd();

      ret.set_T( std::abs( (double)a_) );
      ret.set_dd( std::abs( a_dd_.get_hi() ), std::abs( a_dd_.get_lo() ) );
      
      return ret;
    }
    
    scalar sqrt(scalar a){
      scalar ret;

      PARCEL_MAX_FLOTING_TYPE a_ = a.get_T();
      ddtype a_dd_  = a.get_dd();

      ddtype ret_dd = ddmath_.sqrt(a_dd_);

      //      ret.set_T( std::sqrt(a_) );
      // long doubleはsqrt未対応?
      ret.set_T( std::sqrt( (double)a_) );
      ret.set_dd( ret_dd.get_hi(),ret_dd.get_lo() );

      return ret;
    }

  };

  // ---------
  // ddtype div(ddtype a,ddtype b){ 
  //   ddmath dd;    
  //   return dd.div(a,b);
  // }
  // double      div(double      a, double      b){return a/b;} 
  // float       div(float       a, float       b){return a/b;}
  // long double div(long double a, long double b){return a/b;}
  // sycl::half  div(sycl::half  a, sycl::half  b){return a/b;}

  std::tuple<PARCEL_MAX_FLOTING_TYPE,ddtype>div(
						PARCEL_MAX_FLOTING_TYPE a_T, 
						PARCEL_MAX_FLOTING_TYPE b_T,
						ddtype a_dd,ddtype b_dd
						){
    ddmath dd;    
    return {a_T/b_T,a_dd/b_dd};
  }

  // ---------
  ddtype mul(ddtype a,ddtype b){ 
    ddmath dd;    
    return a*b;
  }
  
  double      mul(double      a, double      b){return a*b;} 
  float       mul(float       a, float       b){return a*b;}
  long double mul(long double a, long double b){return a*b;}
  sycl::half  mul(sycl::half  a, sycl::half  b){return a*b;}

  // ---------
  ddtype sqrt(ddtype a){ 
    ddmath dd;    
    return dd.sqrt(a);
  }
  double      sqrt(double      a){return std::sqrt(a);} 
  float       sqrt(float       a){return std::sqrt(a);}
  long double sqrt(long double a){return std::sqrt(a);}
  sycl::half  sqrt(sycl::half  a){return std::sqrt(a);}
  // ---------

  
  class scalar_blas : public scalar_util{
    //    scalar_util scalar_util_;
  public :

    // コレスキー分解計算

    void potrf(
	       std::vector< parcel::scalar > &L,
	       PARCEL_INT N,
	       PARCEL_INT ldl
	       ){
      for (int i = 0; i < N; ++i) {
	for (int j = 0; j <= i; ++j) {
	  parcel::scalar sum = L[i+j*ldl];
	  for (int k = 0; k < j; ++k) {
	    sum = sum - L[i+k*ldl] * L[j+k*ldl];
	  }
	  if (i == j) {
#if 0	    
	    if (sum.get_T() <= 0.0) {
	      throw std::runtime_error("Matrix is not positive definite.");
	    }
	    L[i+j*ldl] = sqrt(sum);
#else	    

#if 0
	    if (sum.get_T() <= 0.0) {
	      std::cout<<" potrf : "<<sum.get_T()<<std::endl;
	    }
#endif	    
	    L[i+j*ldl] = sqrt(abs(sum));
#endif	    
	  } else {
	    L[i+j*ldl] = sum / L[j+j*ldl];
	  }
	}
      }
      // for (int i = 0; i < N; ++i) {
      // 	for (int j = i; j < N; ++j) {
      // 	  L[i+j*ldl] = L[j+i*ldl];
      // 	}
      // }
      return ;
    }

    // 三角行列の逆行列計算
    void trtri(
	       std::string uplo,
	       std::string diag,
	       std::vector< parcel::scalar > &A,
	       PARCEL_INT N,
	       PARCEL_INT lda
	       ){
      // 単純な掃き出し法で実装
      // 上三角行列のみ対応
      // まずはwork配列に書き出して最後に戻す 
      std::vector<parcel::scalar> Ainv(N*lda);
      for(int i = 0;i<N*lda;i++){
	Ainv[i] = 0.0;
      }
      for(int i = 0;i<N;i++){
	Ainv[i+i*lda] = 1.0;
      }

      if(uplo == "U"){
#if 1
	for(int i = N-1;i>=0;i--){
	  parcel::scalar Aii = A[i+i*lda];
	  for(int j = i;j<N;j++){
	    for(int k = 0;k<i;k++){
	      parcel::scalar Aki  = A[k+i*lda];
	      parcel::scalar Aval = Aki / Aii;
	      parcel::scalar Akj  = Ainv[k+j*lda];
	      parcel::scalar Aij  = Ainv[i+j*lda];
	      Ainv[k+j*lda] = Akj - Aij * Aval;
	    }
	  }
	  for(int j = i;j<N;j++){
	    for(int k = 0;k<i;k++){
	      parcel::scalar Aki  = A[k+i*lda];
	      parcel::scalar Aval = Aki / Aii;
	      parcel::scalar Akj = A[k+j*lda];
	      parcel::scalar Aij = A[i+j*lda];
	      A[k+j*lda]  = Akj - Aij*Aval;
	    }
	    parcel::scalar Aij = A[i+j*lda];
	    A[i+j*lda] = Aij / Aii;
	    parcel::scalar Ainvij = Ainv[i+j*lda];
	    Ainv[i+j*lda] = Ainvij / Aii;
	  }
	}

	for(int j = 0;j<N;j++){
	  int i = 0;
	  parcel::scalar Aii = A[i+i*lda];
	  parcel::scalar Aij = A[i+j*lda];
	  parcel::scalar Ainvij = Ainv[i+j*lda]; 
	  A[i+j*lda]    = Aij / Aii;
	  Ainv[i+j*lda] = Ainvij / Aii;
	}

#else
	for(int i = N-1;i>=0;i--){
	  parcel::scalar Aii = A[i+i*lda];
	  for(int j = i;j<N;j++){
	    for(int k = 0;k<i;k++){
	      parcel::scalar Aki  = A[k+i*lda];
	      parcel::scalar Aval = Aki.get_T() / Aii.get_T();
	      Ainv[k+j*lda] = Ainv[k+j*lda].get_T() - Ainv[i+j*lda].get_T()*Aval.get_T();
	    }
	  }
	  for(int j = i;j<N;j++){
	    for(int k = 0;k<i;k++){
	      parcel::scalar Aki  = A[k+i*lda];
	      parcel::scalar Aval = Aki.get_T() / Aii.get_T();
	      A[k+j*lda]  = A[k+j*lda].get_T() - A[i+j*lda].get_T()*Aval.get_T();
	    }
	    A[i+j*lda] =    A[i+j*lda].get_T() / Aii.get_T();
	    Ainv[i+j*lda] = Ainv[i+j*lda].get_T() / Aii.get_T();
	  }
	}

	for(int j = 0;j<N;j++){
	  int i = 0;
	  parcel::scalar Aii = A[i+i*lda];
	  A[i+j*lda] =    A[i+j*lda].get_T() / Aii.get_T();
	  Ainv[i+j*lda] = Ainv[i+j*lda].get_T() / Aii.get_T();
	}
#endif
      }


#if 0
      // 元の行列が単位行列になることを確認
      for(int jj = 0;jj<N;jj++){
	for(int ii = 0;ii<N;ii++){
	  std::cout<<A[jj+ii*N]<<", ";
	}
	std::cout<<" | ";
	for(int ii = 0;ii<N;ii++){
	  std::cout<<Ainv[jj+ii*N]<<", ";
	}
	std::cout<<std::endl;
      }
#endif

#if 1
      for(int i = 0;i<N*lda;i++){
	A[i] = Ainv[i];
      }
#endif

#if 0
      if(uplo == "L"){
	std::cout<<"Lower Triangular Matrix"<<std::endl;
      }
      if(diag == "N"){
	std::cout<<"Diag is non unit"<<std::endl;
      }
      if(diag == "U"){
	std::cout<<"Diag is unit"<<std::endl;
      }
#endif

      return ;

    }
    
    // C[m,n] = alpha * A[m,k] * B[k,n] + beta * C[m,n]
    void gemm(
	      PARCEL_INT m,
	      PARCEL_INT n,
	      PARCEL_INT k,
	      std::vector< parcel::scalar > &alpha_,
	      std::vector< parcel::scalar > &beta_,
	      std::vector< parcel::scalar > &A_,PARCEL_INT ldA,
	      std::vector< parcel::scalar > &B_,PARCEL_INT ldB,
	      std::vector< parcel::scalar > &C_,PARCEL_INT ldC ){
      
      parcel::scalar alpha = alpha_[0];
      parcel::scalar beta =  beta_[0];
      
      for(PARCEL_INT i =0;i<n;i++){
	for(PARCEL_INT j =0;j<m;j++){
	  parcel::scalar AB = 0.0;
	  for(PARCEL_INT l =0;l<k;l++){
	    AB = AB + A_[j+l*ldA]*B_[l+i*ldB];
	  }
	  C_[j+i*ldC] = alpha*AB + beta*C_[j+i*ldC];
	}	
      }    
      
      return ;
      
    }
    
  };

}

#endif
