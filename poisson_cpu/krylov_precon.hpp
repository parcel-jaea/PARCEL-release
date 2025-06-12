#ifndef KRYLOV_PRECON_HPP_
#define KRYLOV_PRECON_HPP_

#include <iostream>
#include "mat.hpp"
#include "vec.hpp"

template<class T>
class krylov_precon{

  vec<T> vec_;

  crs_mat<T> crsA_;

  std::vector<T> Diag_;
  std::vector<T> x_;
  std::vector<T> b_;
  std::vector<T> work_;

  std::vector<T> A;
  std::vector<PARCEL_INT> crsRow_ptr;
  std::vector<PARCEL_INT> crsCol;

public:
  krylov_precon(){}
  ~krylov_precon(){}

  void init(crs_mat<T> &crsA){
    PARCEL_INT nRow = crsA.getValue_nRow();
    crsA_ = crsA;
    Diag_ = crsA.Diag_ptr();
    work_.resize(nRow,0.0);
    A = crsA_.A_ptr();
    crsRow_ptr = crsA_.crsRow_ptr();
    crsCol     = crsA_.crsCol_ptr();
  }

  void DiagScaling(
		   std::vector<T> &D,
		   std::vector<T> &x,
		   std::vector<T> &y
	    ){

#pragma omp parallel for
    for(auto i = 0; i < y.size(); i++){
      T Dinv = (D[i] == 0.0 ? 1.0 : 1.0/D[i]);
      y[i] = Dinv*x[i];
    }

  }

  void jacobi(
	    std::vector<T> &x,
	    std::vector<T> &b
	    ){

    const PARCEL_INT nRow = crsA_.getValue_nRow();
    
#pragma omp parallel for
    for(auto i = 0; i < nRow; i++){
      T tmp = 0.0;
      for(auto j = crsRow_ptr[i]; j < crsRow_ptr[i+1]; j++){
	auto col = crsCol[j];
	tmp = tmp + A[j]*x[col];
      }
      T Diag_inv = (Diag_[i] == 0.0 ? 1.0 : 1.0/Diag_[i]);
      work_[i] = x[i] + (b[i] - tmp)*Diag_inv;
    }

#pragma omp parallel for
    for(auto i = 0; i < nRow; i++){
       x[i] = work_[i];
    }

  }

  void jacobi_method(
		     std::vector<T> &x,
		     std::vector<T> &b,
		     PARCEL_INT itrmax
		     ){
    // zero initialize
    vec_.axpy(0.0,0.0,x,x);
    for(auto itr = 0;itr < itrmax;itr++){
#if 1
      jacobi(x,b);
#else
      crsA_.SpMV(x,work_);
      vec_.axpy(1.0,-1.0,b,work_);          
      DiagScaling(Diag_,work_,work_);
      vec_.axpy(1.0,1.0,work_,x);
#endif
    }
  }

  void Symmetric_GaussSeidel(
	    std::vector<T> &x,
	    std::vector<T> &b
	    ){

    PARCEL_INT nRow = crsA_.getValue_nRow();

    for(auto i = 0; i < nRow; i++){
      T tmp = 0.0;
      for(auto j = crsRow_ptr[i]; j < crsRow_ptr[i+1]; j++){
	auto col = crsCol[j];
        if(i==col){continue;}
	tmp = tmp + A[j]*x[col];
      }
      T Diag_inv = (Diag_[i] == 0.0 ? 1.0 : 1.0/Diag_[i]);
      x[i] = (b[i] - tmp)*Diag_inv;
    }

    for(auto i = nRow-1; i >= 0; i--){
      T tmp = 0.0;
      for(auto j = crsRow_ptr[i]; j < crsRow_ptr[i+1]; j++){
	auto col = crsCol[j];
        if(i==col){continue;}
	tmp = tmp + A[j]*x[col];
      }
      T Diag_inv = (Diag_[i] == 0.0 ? 1.0 : 1.0/Diag_[i]);
      x[i] = (b[i] - tmp)*Diag_inv;
    }
  }

  void Symmetric_GaussSeidel_method(
		     std::vector<T> &x,
		     std::vector<T> &b,
		     PARCEL_INT itrmax
		     ){
    // zero initialize
    vec_.axpy(0.0,0.0,x,x);
    for(auto itr = 0;itr < itrmax/2;itr++){
      Symmetric_GaussSeidel(x,b);
    }
  }

};

#endif
