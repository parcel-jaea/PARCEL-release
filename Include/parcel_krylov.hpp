#ifndef PARCEL_KRYLOV_HPP_
#define PARCEL_KRYLOV_HPP_

#ifdef RESIDUAL_REPLACE
#define TRUE_RESIDUAL 1
#endif

#include <vector>
#include <type_traits>
#include "ddlib.hpp"
#include "util_sycl.hpp"
#include "parcel.hpp"
#include "parcel_precon.hpp"
#include "parcel_type.hpp"
#include "parcel_blas.hpp"
#include "parcel_lapack.hpp"
#include "parcel_util.hpp"

#include "parcel_time.hpp"

// #define PARCEL_TYPE double
//    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();
//    std::chrono::high_resolution_clock::time_point end   = std::chrono::high_resolution_clock::now();
//    std::chrono::microseconds elapsed_time = std::chrono::duration_cast<std::chrono::microseconds>(end - begin);
//    etime = etime + elapsed_time.count();

namespace parcel{
  namespace krylov{

    class krylov_param : public precon_param{
      double tol_ = 1.0e-4;
      PARCEL_INT itrmax_ = 100;
    public :
      krylov_param(){}
      ~krylov_param(){}
      void set_tolerance(double tol){
	tol_ = tol;
      }
      void set_max_iteration(PARCEL_INT itrmax){
	itrmax_ = itrmax;
      }
      double get_tolerance(){
	return tol_;
      }
      PARCEL_INT get_max_iteration(){
	return itrmax_;
      }
    };

  }

}

#include "parcel_krylov_cg.hpp"
#include "parcel_krylov_bicgstab.hpp"
#include "parcel_krylov_gmres.hpp"
#include "parcel_krylov_cagmres.hpp"
#include "parcel_krylov_choleskyqr_cagmres.hpp"


#endif

