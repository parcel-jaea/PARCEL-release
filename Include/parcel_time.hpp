// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_TIME_KRYLOV_HPP_
#define PARCEL_TIME_KRYLOV_HPP_

#include <chrono>

namespace parcel{
  class timer{
    std::chrono::high_resolution_clock::time_point begin;
    double elapsed_time;
    // 呼び出し回数。1 回あたりの時間から到達帯域を逆算するために使う。
    long long call_count;
  public :
    timer(){ 
      elapsed_time = 0.0;
      call_count   = 0;
    }
    ~timer(){}

    void reset(){
      elapsed_time = 0.0;
      call_count   = 0;
    }

    void start(){
      begin = std::chrono::high_resolution_clock::now();
    }

    void end(){
      std::chrono::high_resolution_clock::time_point end_time = std::chrono::high_resolution_clock::now();
      std::chrono::microseconds elapsed = std::chrono::duration_cast<std::chrono::microseconds>(end_time - begin);
      elapsed_time = elapsed_time + elapsed.count();
      call_count   = call_count + 1;
    }

    double get(){
      return elapsed_time*1.0e-6;
    }

    long long get_count(){
      return call_count;
    }

  };


}

#endif
