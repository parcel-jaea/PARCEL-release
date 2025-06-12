#ifndef PARCEL_TIME_KRYLOV_HPP_
#define PARCEL_TIME_KRYLOV_HPP_

#include <chrono>

namespace parcel{
  class timer{
    std::chrono::high_resolution_clock::time_point begin;
    double elapsed_time;
  public :
    timer(){ 
      elapsed_time = 0.0;
    }
    ~timer(){}

    void reset(){
      elapsed_time = 0.0;
    }

    void start(){
      begin = std::chrono::high_resolution_clock::now();
    }

    void end(){
      std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
      std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin);
      elapsed_time = elapsed_time + elapsed.count();
      //      elapsed_time = elapsed_time + 1.0;
    }

    double get(){
      return elapsed_time*1.0e-6;
    }

  };

#if 0
  class krylov_timer{
    parcel::timer spmv;
    parcel::timer precon;
    parcel::timer dot;
  public :
    krylov_timer(){}
    ~krylov_timer(){}
    void reset(){
      spmv.reset();
      precon.reset();
      dot.reset();
    }
  }
#endif

}

#endif
