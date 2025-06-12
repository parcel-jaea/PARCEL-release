#ifndef VEC_HPP_
#define VEC_HPP_

#include <iostream>
#include <cmath>
#include <vector>

template<class T>
class vec{
  
public:

  void axpy(
	    T alpha,
	    T beta,
	    std::vector<T> &x,
	    std::vector<T> &y){   
#pragma omp parallel for
    for(auto i = 0; i < y.size(); i++){
      y[i] = alpha * x[i] + beta*y[i];
    }
  }

  T dot(
	std::vector<T> &x,
	std::vector<T> &y){   
    T tmp = 0.0;
#pragma omp parallel for reduction(+:tmp)
    for(auto i = 0; i < y.size(); i++){
      tmp = tmp + x[i] * y[i];
    }
    return tmp;
  }

  T norm(
	 std::vector<T> &x,
	 std::vector<T> &y){   
    T tmp = dot(x,y);
    // tmp = std::sqrt(tmp);
    // return tmp;
    // long double で動作しない?
    return std::sqrt(tmp);
  }

};

#endif
