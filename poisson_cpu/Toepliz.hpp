#ifndef TOEPLIZ_HPP_
#define TOEPLIZ_HPP_

#include <cstdint>
#include <cmath>
#include "mat.hpp"

template<class T>
class toepliz_equation{

public :

  PARCEL_INT get_nnz(PARCEL_INT n){
    return n + n-1 + n+2;
  }

  void make_matrix(
		   PARCEL_INT n,
		   double gamma,
		   crs_mat<T> &crsA){
    PARCEL_INT nnz = 0;
    double one  = 1.0;
    double two  = 2.0;
    if(n<3){
      std::cout<<" n is too small. n = "<<n<<std::endl;
      return ;
    }

    crsA.setValue_crsRow_ptr(0,0);
    PARCEL_INT i;
    for(i = 0;i<n-1;i++){
      crsA.setValue_crsRow_ptr(nnz,i);
      switch(i){
      case 0:
	crsA.setValue_matA(two,nnz);crsA.setValue_crsCol(i  ,nnz);nnz=nnz+1;
	crsA.setValue_matA(one,nnz);crsA.setValue_crsCol(i+1,nnz);nnz=nnz+1;
	break;
      case 1:
	crsA.setValue_matA(two,nnz);crsA.setValue_crsCol(i  ,nnz);nnz=nnz+1;
	crsA.setValue_matA(one,nnz);crsA.setValue_crsCol(i+1,nnz);nnz=nnz+1;
	break;
      default:
	crsA.setValue_matA(gamma,nnz);crsA.setValue_crsCol(i-2,nnz);nnz=nnz+1;
	crsA.setValue_matA(two  ,nnz);crsA.setValue_crsCol(i  ,nnz);nnz=nnz+1;
	crsA.setValue_matA(one  ,nnz);crsA.setValue_crsCol(i+1,nnz);nnz=nnz+1;
	break;
      }
    }
    i = n-1;
    crsA.setValue_crsRow_ptr(nnz,i);
    crsA.setValue_matA(gamma,nnz);crsA.setValue_crsCol(i-2,nnz);nnz=nnz+1;
    crsA.setValue_matA(two  ,nnz);crsA.setValue_crsCol(i  ,nnz);nnz=nnz+1;
    crsA.setValue_crsRow_ptr(nnz,n);

    return ;
  }

  void make_right_vector(
			 crs_mat<T> &crsA,
			 std::vector<T> &b
			 ){
    PARCEL_INT n = crsA.getValue_nRow();
    std::vector<T> x(n,1.0);
    crsA.SpMV(x,b);
  }

  void check_solution(
		      std::vector<T> &x
		      ){
    PARCEL_INT n = x.size();
    //    std::cout<<"error  : = "<< n <<std::endl;

    T max_error = 0.0;
    T max_exact = 0.0;
    for(PARCEL_INT i = 0; i < n; i++){
      T exact = 1.0;
      T error = std::abs(x[i] - exact);
      // std::cout<<"error  : = "<<error<<std::endl;
      if(max_error < error){
	max_error = error;
	max_exact = exact;
      }
    }
    std::cout<<"max(|numerical-exact|)                = "<<max_error          <<std::endl;
    std::cout<<"max(|numerical-exact|) / max(|exact|) = "<<max_error/max_exact<<std::endl;
  }

};

#endif
