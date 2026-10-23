// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef UTIL_SYCL_HPP_
#define UTIL_SYCL_HPP_

#ifdef HIPSYCL_SYCL_HPP
using namespace cl;
#endif

#include <CL/sycl.hpp>
#include <cstdlib>
#include <new>
#include <string>
#include <mutex>
#include <unordered_map>

namespace util_sycl{

  // デバイスメモリの確保に失敗したことを表す例外。
  //   sycl::malloc_device は失敗時に nullptr を返すので、malloc_device_checked で検査してこれを投げる。
  //   std::bad_alloc として捕まえられ、what() に要求バイト数と確保した場所を書く。
  //   注意: 確保に失敗するのは多くの場合一部のランクだけなので、他のランクは次の集団通信で待つ。
  //   例外を捕まえて続行せず、MPI_Abort などで全体を止めること。
  class device_bad_alloc : public std::bad_alloc {
    std::string msg_;
  public:
    explicit device_bad_alloc(const std::string &m) : msg_(m) {}
    const char *what() const noexcept override { return msg_.c_str(); }
  };

  // PARCEL が確保したデバイスメモリの量を数える (環境変数 PARCEL_MEM_REPORT があるときだけ)。
  //   nvidia-smi などはソルバの外でしか読めず (作業域が解放された後の値になる)、装置によっては使えないため、
  //   確保 (malloc_device_checked と内積の作業域) と解放 (free_tracked) で現在量とピークを追う。
  //   数えるのは PARCEL が USM で確保したものだけで、SYCL ランタイム自身の確保や sycl::buffer は入らない。
  //   CPU バックエンドでは「デバイス」もホストメモリなので、ホストの RSS のうち PARCEL の配列の分に当たる。
  struct device_mem_tracker_{
    std::mutex m;
    std::unordered_map<void *, size_t> sz;
    size_t cur = 0, peak = 0;
  };
  inline bool device_mem_tracking(){
    static const bool on = (std::getenv("PARCEL_MEM_REPORT") != nullptr);
    return on;
  }
  inline device_mem_tracker_ &device_mem_tracker(){ static device_mem_tracker_ t; return t; }
  inline void track_device_alloc(void *p, size_t bytes, const char *where = nullptr){
    (void)where;
    if(!device_mem_tracking() || p == nullptr) return;
    auto &t = device_mem_tracker(); std::lock_guard<std::mutex> g(t.m);
    t.sz[p] = bytes; t.cur += bytes;
    if(t.cur > t.peak) t.peak = t.cur;
  }
  inline void track_device_free(void *p){
    if(!device_mem_tracking() || p == nullptr) return;
    auto &t = device_mem_tracker(); std::lock_guard<std::mutex> g(t.m);
    auto it = t.sz.find(p);
    if(it != t.sz.end()){ t.cur -= it->second; t.sz.erase(it); }
  }
  inline size_t device_mem_current_bytes(){ auto &t = device_mem_tracker(); std::lock_guard<std::mutex> g(t.m); return t.cur; }
  inline size_t device_mem_peak_bytes(){ auto &t = device_mem_tracker(); std::lock_guard<std::mutex> g(t.m); return t.peak; }
  // 記録してから解放する。sycl::free の代わりに使う (記録に無いポインタはそのまま解放する)。
  inline void free_tracked(void *p, sycl::queue &q){ track_device_free(p); sycl::free(p, q); }
  inline void free_tracked(void *p, const sycl::context &c){ track_device_free(p); sycl::free(p, c); }

  template<class T>
  T *malloc_device_checked(size_t count, sycl::queue &q, const char *where){
    T *p = sycl::malloc_device<T>(count, q);
    track_device_alloc(p, count * sizeof(T), where);
    if(p == nullptr && count > 0){
      throw device_bad_alloc(std::string("PARCEL: device memory allocation failed in ") + where + ": "
                             + std::to_string((unsigned long long)(count * sizeof(T))) + " bytes ("
                             + std::to_string((unsigned long long)count) + " x "
                             + std::to_string((unsigned long long)sizeof(T)) + ")");
    }
    return p;
  }

  template<class T>
  class vector{
    T* mem_;
    size_t n_;
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};
    bool owner_ = true;
  public:

     void shallowcp(T* mem, size_t n, sycl::queue* q){
       mem_ = mem;
       n_ = n;
       q_ = q;
       owner_ = false;
     }

     vector(T* mem, size_t n, sycl::queue* q){
       shallowcp(mem,n,q);
     }

    // 長さ 0 のとき、size() は指定どおり 0 を返し、ポインタを有効に保つため
    //   確保だけは 1 要素以上にする。
    static size_t alloc_count(size_t n){ return n > 0 ? n : 1; }

    vector(size_t n, sycl::queue* q){
      q_ = q;
      n_ = n;
      // 確保しただけのデバイスメモリは未初期化。PARCEL のカーネルは
      // y = alpha*x + beta*y を beta=0 でも y を読む形で計算するため、
      // 未初期化の y が NaN/Inf を含むと 0*NaN = NaN で結果が壊れる。
      // 確保時にゼロで埋めておく。
      mem_ = util_sycl::malloc_device_checked<T>(alloc_count(n),q_[0],"util_sycl::vector");
      q_[0].memset(mem_, 0, sizeof(T) * alloc_count(n)).wait();
    }
    vector(){
      q_ = &q_default;
      n_ = 1;
      mem_ = util_sycl::malloc_device_checked<T>(1,q_[0],"util_sycl::vector");
      q_[0].memset(mem_, 0, sizeof(T) * (size_t)(1)).wait();
    }
    ~vector(){
      q_[0].wait();
      if(owner_ == true){
	util_sycl::free_tracked(mem_, q_[0]);
      }
      q_[0].wait();
    }

    // 他の配列の領域を共有する (所有しない view にする)。自分が確保していた領域は解放する。
    //   Jacobi 前処理が行列 A の行ポインタ・列番号を複製せずに使うためのもの。
    void share(T* mem, size_t n, sycl::queue* q){
      q_[0].wait();
      if(owner_ == true){ util_sycl::free_tracked(mem_, q_[0]); }
      shallowcp(mem, n, q);
    }

    void resize(size_t n, sycl::queue* q){
      q_[0].wait();
      // 所有しない view (shallowcp / share) は解放しない (他の配列の領域なので)
      if(owner_ == true){ util_sycl::free_tracked(mem_, q_[0]); }
      owner_ = true;
      mem_ = nullptr;   // 次の確保で投げたときに二重解放しない
      q_ = q;
      n_ = n;   // 0 は 0 のまま (上のコンストラクタと同じ)
      mem_ = util_sycl::malloc_device_checked<T>(alloc_count(n),q_[0],"util_sycl::vector");
      q_[0].memset(mem_, 0, sizeof(T) * alloc_count(n)).wait();
    }

    void resize(size_t n){
      if(owner_ == true){ util_sycl::free_tracked(mem_, q_[0]); }   // view は解放しない
      owner_ = true;
      mem_ = nullptr;   // 次の確保で投げたときに二重解放しない
      n_ = n;   // 0 は 0 のまま
      mem_ = util_sycl::malloc_device_checked<T>(alloc_count(n),q_[0],"util_sycl::vector");
      q_[0].memset(mem_, 0, sizeof(T) * alloc_count(n)).wait();
    }

    sycl::queue *queue(){
      return q_;
    }

    T *data(){
      return mem_;
    }

    T *at(size_t i){
      return &mem_[i];
    }

    size_t size(){
      return n_;
    }

    void setValue(T *mem){
      const size_t bytes = n_ * sizeof(T);
      q_[0].memcpy(mem_, mem, bytes);
      q_[0].wait();
    }

    void getValue(T *mem){
      const size_t bytes = n_ * sizeof(T);
      q_[0].memcpy(mem, mem_, bytes);
      q_[0].wait();
    }

    template<class TT>
    void getValue(TT *mem,size_t n){
      const size_t bytes = n * sizeof(TT);
      q_[0].memcpy(mem, mem_, bytes);
      q_[0].wait();
    }
    template<class TT>
    void setValue(TT *mem,size_t n){
      const size_t bytes = n * sizeof(TT);
      q_[0].memcpy(mem_, mem, bytes);
      q_[0].wait();
    }

    void setVec(sycl::queue* q, T* mem, size_t n){
      util_sycl::free_tracked(mem_, q_[0]);
      mem_ = mem;
      n_ = n;
      q_ = q;
      return ;
    }

  };

  // nd_range カーネル内のワークグループ同期 (item.barrier の代わりに使う)。
  //   AdaptiveCpp 24.10 の OpenMP (accelerated CPU, --acpp-targets=omp) では、バリアを含む
  //   nd_range カーネルから到達する関数がすべて noexcept 扱いになる。nd_item::barrier はホスト側で
  //   std::function<void()> を呼ぶため、翻訳単位全体で std::function 経由の例外 (PARCEL の設定関数の
  //   std::invalid_argument など) が catch に届かず std::terminate で落ちる。
  //   そこで accelerated CPU のホスト側だけ、CBS (ループ分割) が分割点と認識する注釈
  //   hipsycl_barrier を付けた自前の関数を呼び、std::function を経由しないようにする。
  //   分割後は呼び出しごと取り除かれるので、同期の意味は nd_item::barrier と同じ
  //   (本体が実行されるのは分割に失敗した場合だけで、そのときは停止する)。
  //   デバイス側 (CUDA など) と、それ以外の SYCL 実装では item.barrier を呼ぶ。
#if defined(__clang__) && defined(__ACPP_USE_ACCELERATED_CPU__) && defined(__acpp_if_target_host) \
    && !defined(PARCEL_USE_ITEM_BARRIER)
  [[clang::annotate("hipsycl_barrier")]] __attribute__((noinline))
  inline void cbs_split_barrier() noexcept {
    std::abort();
  }
#define PARCEL_CBS_SPLIT_BARRIER 1
#endif

  template<int Dim>
  inline void group_barrier(const sycl::nd_item<Dim> &item,
                            sycl::access::fence_space space = sycl::access::fence_space::local_space){
#if defined(PARCEL_CBS_SPLIT_BARRIER)
    __acpp_if_target_device( item.barrier(space); )
    __acpp_if_target_host( (void)item; (void)space; cbs_split_barrier(); )
#else
    item.barrier(space);
#endif
  }

}

#endif
