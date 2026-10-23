// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef PARCEL_HPP_
#define PARCEL_HPP_

#include "parcel_nblock.hpp"

#define PARCEL_MAX_FLOTING_TYPE double
#if PARCEL_MPI
#define PARCEL_MPI_MAXFLOTING_TYPE MPI_DOUBLE
#endif

#include <type_traits>
#include <vector>
#include <limits>
#include <stdexcept>
#include <string>
#include <algorithm>
#include "parcel_sycl.hpp"
#include "parcel_scalar.hpp"

namespace parcel{

  using ddtype = ddlib::ddtype;
  const size_t CRS_MATRIX = 1;
  const size_t DIA_MATRIX = 2;

  // -------------------------------
  template<class T>
  class floatingtype{
    util_sycl::vector<T> vec;
  public:
    
    floatingtype(){}
    ~floatingtype(){}

    void resize(size_t nnz, sycl::queue *q){
      vec.resize(nnz,q);
      return ;
    }

    sycl::queue *queue(){
      return vec.queue();
    }

    void setValue( std::vector< parcel::scalar > &mem_){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_[i].get_T();
      }
      vec.setValue(vec_x);
    }


    void setValue(parcel::scalar mem_){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_.get_T();
      }
      vec.setValue(vec_x);
    }

    template<class Tdata>
    void setValue(Tdata *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem[i];
      }
      vec.setValue(vec_x);
    }
    void setValue(parcel::scalar *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem[i].get_T();
      }
      vec.setValue(vec_x);
    }

    template<class Tdata>
    void setValue(Tdata *mem_hi,Tdata *mem_lo){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      for(size_t i = 0;i<n;i++){
	vec_x[i] = mem_hi[i] + mem_lo[i];
      }
      vec.setValue(vec_x);
    }

    void getValue( std::vector< parcel::scalar > &mem ){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      vec.getValue(vec_x);
      for(size_t i = 0;i<n;i++){
	PARCEL_MAX_FLOTING_TYPE vec_x_ = vec_x[i];
	mem[i].set_Tdd(vec_x_);
      }
    }
    
    template<class Tdata>
    void getValue(Tdata *mem){
      size_t n = vec.size();
      std::vector<T> vec_x_(n,0.0);
      T *vec_x = vec_x_.data();
      vec.getValue(vec_x);
      for(size_t i = 0;i<n;i++){
	mem[i] = vec_x[i];
      }
    }

    void setValue(ddlib::ddtype mem_){
      T  mem = mem_.get_T<T>();
      vec.setValue(&mem);
    }

    void setValue(T *mem){
      vec.setValue(mem);
    }

    void getValue(T *mem){
      vec.getValue(mem);
    }

    void getValue_hi(T *mem){
      vec.getValue(mem);
    }
    void getValue_lo(T *mem){
      mem = nullptr;
    }
    
    void getValue_d_hi(double *mem){
      std::vector<T> mem_T(vec.size());
      getValue(mem_T.data());
      for(PARCEL_INT i=0;i<vec.size();i++){
	  mem[i] = mem_T[i];
      }
    }
    
    void getValue_d_lo(double *mem){
      mem = nullptr;
    }

    T *data(){
      return vec.data();
    }

    T *data_hi(){
      return vec.data();
    }

    T *data_lo(){
      return nullptr;
    }

    size_t size(){
      return  vec.size();
    }

    void shallowcp(T* mem, size_t n, sycl::queue* q){
      vec.shallowcp(mem,n,q);
      return ;
    }

    void shallowcp_hi(T* mem, size_t n, sycl::queue* q){
      vec.shallowcp(mem,n,q);
      return ;
    }

    void shallowcp_lo(T* mem, size_t n, sycl::queue* q){
      mem = nullptr;      
      return ;
    }

  };

  class doubledouble{
    util_sycl::vector<double> vec_hi;
    util_sycl::vector<double> vec_lo;
  public:

    doubledouble(){}
    ~doubledouble(){}

    void resize(size_t nnz, sycl::queue *q){
      vec_hi.resize(nnz,q);
      vec_lo.resize(nnz,q);
      return ;
    }

    sycl::queue *queue(){
      return vec_hi.queue();
    }

    size_t size(){
      return  vec_hi.size();
    }

    template<class T>
    void setValue_hi(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_(n,0.0);
      double *vec = vec_.data();
      for(size_t i = 0;i<n;i++){
	vec[i] = mem[i];
      }
      vec_hi.setValue(vec);
    }

    template<class T>
    void setValue_lo(T *mem){
      size_t n = vec_lo.size();
      std::vector<double> vec_(n,0.0);
      double *vec = vec_.data();
      for(size_t i = 0;i<n;i++){
	vec[i] = mem[i];
      }
      vec_lo.setValue(vec);
    }

    void setValue(double *mem_hi,double *mem_lo){
      vec_hi.setValue(mem_hi);
      vec_lo.setValue(mem_lo);
    }

    void setValue(parcel::scalar mem_){
      double mem_hi = mem_.get_dd_hi();
      double mem_lo = mem_.get_dd_lo();

      size_t n = vec_lo.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();

      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem_hi;
      }
      vec_hi.setValue(vec_data_hi);

      for(size_t i = 0;i<n;i++){
	vec_data_lo[i] = mem_lo;
      }
      vec_lo.setValue(vec_data_lo);

    }

    void setValue( std::vector< parcel::scalar > &mem_){

      size_t n = vec_lo.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();

      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem_[i].get_dd_hi();
      }
      vec_hi.setValue(vec_data_hi);

      for(size_t i = 0;i<n;i++){
	vec_data_lo[i] = mem_[i].get_dd_lo();
      }
      vec_lo.setValue(vec_data_lo);

    }

    template<class T>
    void setValue(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();
      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem[i];
	vec_data_lo[i] = 0.0;
      }
      vec_hi.setValue(vec_data_hi);
      vec_lo.setValue(vec_data_lo);
    }
    void setValue(parcel::scalar *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_hi_(n,0.0);
      std::vector<double> vec_lo_(n,0.0);
      double *vec_data_hi = vec_hi_.data();
      double *vec_data_lo = vec_lo_.data();
      for(size_t i = 0;i<n;i++){
	vec_data_hi[i] = mem[i].get_dd_hi();
	vec_data_lo[i] = mem[i].get_dd_lo();
      }
      vec_hi.setValue(vec_data_hi);
      vec_lo.setValue(vec_data_lo);
    }

    void setValue_hi(double *mem){
      vec_hi.setValue(mem);
    }

    void setValue_lo(double *mem){
      vec_lo.setValue(mem);
    }

    // hi / lo を分けて取り出す。setValue_hi / setValue_lo と対称。
    //   parcel::vector<T>::getValue(T*,T*) と getValue(double*,double*) から
    //   呼ばれる。これらが無いと、double-double をデータ精度に指定した
    //   ソルバ (pgmres, ks_arnoldi) が実体化できない。
    template<class T>
    void getValue_hi(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> vec_(n,0.0);
      vec_hi.getValue(vec_.data());
      for(size_t i = 0;i<n;i++){
	mem[i] = vec_[i];
      }
    }

    template<class T>
    void getValue_lo(T *mem){
      size_t n = vec_lo.size();
      std::vector<double> vec_(n,0.0);
      vec_lo.getValue(vec_.data());
      for(size_t i = 0;i<n;i++){
	mem[i] = vec_[i];
      }
    }

    void getValue_d_hi(double *mem){
      vec_hi.getValue(mem);
    }

    void getValue_d_lo(double *mem){
      vec_lo.getValue(mem);
    }

    template<class T>
    void getValue(T *mem){
      size_t n = vec_hi.size();
      std::vector<double> mem_hi_(n,0.0);
      std::vector<double> mem_lo_(n,0.0);
      double *mem_hi = mem_hi_.data();
      double *mem_lo = mem_lo_.data();

      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);

      for(size_t i = 0;i<n;i++){
	mem[i] = mem_hi[i] + mem_lo[i];
      }
    }

    void getValue(  std::vector< parcel::scalar > &mem ){
      size_t n = vec_hi.size();
      std::vector<double> mem_hi_(n,0.0);
      std::vector<double> mem_lo_(n,0.0);
      double *mem_hi = mem_hi_.data();
      double *mem_lo = mem_lo_.data();

      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);

      for(size_t i = 0;i<n;i++){
	mem[i].set_T(  mem_hi[i] + mem_lo[i] );
	mem[i].set_dd( mem_hi[i] , mem_lo[i] );
      }
    }

    void getValue_hi(double *mem){
      vec_hi.getValue(mem);
    }
    void getValue_lo(double *mem){
      vec_lo.getValue(mem);
    }
    void getValue(double *mem_hi,double *mem_lo){
      vec_hi.getValue(mem_hi);
      vec_lo.getValue(mem_lo);
    }
    void getValue( std::vector<double> &mem_hi, std::vector<double> &mem_lo){
      vec_hi.getValue(mem_hi.data());
      vec_lo.getValue(mem_lo.data());
    }

    double *data(){
      return vec_hi.data();
    }
    double *data_hi(){
      return vec_hi.data();
    }
    double *data_lo(){
      return vec_lo.data();
    }

    void shallowcp_hi(double* mem_hi, size_t n, sycl::queue* q){
      vec_hi.shallowcp(mem_hi,n,q);
      return ;
    }
    void shallowcp_lo(double* mem_lo, size_t n, sycl::queue* q){
      vec_lo.shallowcp(mem_lo,n,q);
      return ;
    }
    void shallowcp(double* mem_hi ,double* mem_lo, size_t n, sycl::queue* q){
      vec_hi.shallowcp(mem_hi,n,q);
      vec_lo.shallowcp(mem_lo,n,q);
	return ;
    }

    void shallowcp(double* mem_hi ,size_t n, sycl::queue* q){
      shallowcp_hi(mem_hi,n,q);
      return ;
    }

  };


// 精度を省略したら倍精度 (parcel::vector<> v(n, &q))。parcel_double は後で定義されるので同じ型を直接書く
template<class parceltype = parcel::floatingtype<double> >
  class vector{
    parceltype mem_;
    size_t n_;
  public:
    void resize(size_t n, sycl::queue *q){
      n_ = n;
      mem_.resize(n,q);
    }

    vector(){}

    vector(size_t n, sycl::queue* q){
      resize(n,q);
    }
    ~vector(){}

    sycl::queue *queue(){
      return mem_.queue();
    }

    template<class T>
    void setValue(T *mem){
      mem_.setValue(mem);
    }
  
    template<class T>
    void setValue(std::vector<T> &mem){
      mem_.setValue(mem.data());
    }

    void setValue(parcel::scalar mem){
      mem_.setValue(mem);
    }

    void setValue( std::vector< parcel::scalar >  &mem){
      mem_.setValue(mem);
    }

    void setValue(double *mem_hi,double *mem_lo){
      mem_.setValue(mem_hi,mem_lo);
    }

    void setValue(std::vector<double> &mem_hi,std::vector<double> &mem_lo){
      mem_.setValue(mem_hi.data(),mem_lo.data());
    }

    void setValue_hi(double *mem_hi){
      mem_.setValue_hi(mem_hi);
    }
    void setValue_lo(double *mem_lo){
      mem_.setValue_lo(mem_lo);
    }

    template<class T>
    void getValue(T *mem){
      mem_.getValue(mem);
    }

    void getValue( std::vector< parcel::scalar >  &mem){
      mem_.getValue(mem);
    }

    template<class T>
    void getValue_hi(T *mem_hi){
      mem_.getValue_hi(mem_hi);
    }
    template<class T>
    void getValue_lo(T *mem_lo){
      mem_.getValue_lo(mem_lo);
    }
    
  template<class T>
  void getValue_hi( std::vector<T> &mem_hi){
    mem_.getValue_hi(mem_hi.data());
  }
  template<class T>
  void getValue_lo( std::vector<T> &mem_lo){
    mem_.getValue_lo(mem_lo.data());
  }

    void getValue_hi(double *mem_hi){
      mem_.getValue_hi(mem_hi);
    }
    void getValue_lo(double *mem_lo){
      mem_.getValue_lo(mem_lo);
    }

    template<class T>
    void getValue(T *mem_hi,T *mem_lo){
      mem_.getValue_hi(mem_hi);
      mem_.getValue_lo(mem_lo);
    }
  
    void getValue(double *mem_hi,double *mem_lo){
      mem_.getValue_d_hi(mem_hi);
      mem_.getValue_d_lo(mem_lo);
    }
  
    size_t size(){
      return mem_.size();
    }

    parceltype *mem(){
      return &mem_;
    }

    auto *data(){
      return mem_.data();
    }
    auto *data_hi(){
      return mem_.data_hi();
    }
    auto *data_lo(){
      return mem_.data_lo();
    }

  };
}


#include "parcel_type.hpp"
#include "parcel_util.hpp"

namespace parcel{
  
template<class parceltype>
class parallel_vector{
  
  parcel::vector_util vec_util_;
  parcel::vector<parceltype> vec_add_;
  parcel::vector<parceltype> vec_;
  PARCEL_INT n_;
  PARCEL_INT nadd_;
  PARCEL_INT num_;
  PARCEL_INT ld_;
  
public :

  //   pad: 列の先頭間隔 ld に足す詰め物 [要素]。既定 0 で ld = n + nadd。
  //   Krylov 基底 (num > 1) でキャッシュセットの衝突を避けるときに PARCEL_BASIS_PAD を渡す。
  //   列 id の局所部分・halo 込み部分の位置は ld*id から始まるので、詰め物は halo の後ろに入るだけ。
  void resize(
	      PARCEL_INT n, PARCEL_INT nadd, PARCEL_INT num, sycl::queue *q, PARCEL_INT pad = 0
	      ){
    // 全体長 ld*num は桁あふれしないよう size_t で掛ける。ld 自体は PARCEL_INT で持つので、
    //   n+nadd(+pad) が収まらなければ止める (-DPARCEL_INT32 でビルドしたときだけ起こりうる)。
    if( (int64_t)n + (int64_t)nadd + (int64_t)pad > (int64_t)std::numeric_limits<PARCEL_INT>::max() ){
      throw std::overflow_error("parcel::parallel_vector: n + nadd exceeds PARCEL_INT range (32-bit build); rebuild without -DPARCEL_INT32 (64-bit PARCEL_INT is the default)");
    }
    n_    = n;
    nadd_ = nadd;
    num_  = num;
    ld_   = n + nadd + pad;
    vec_add_.resize((size_t)ld_*(size_t)num_,q);
  }
  
  // 列の先頭間隔。num_ > 1 の配列を列方向に触る側は必ずこれを使う。
  PARCEL_INT ld() const { return ld_; }

  parallel_vector(){
    sycl::queue q_default{sycl::default_selector{}};
    resize(1,1,1,&q_default);
  }
  
  parallel_vector(
		  PARCEL_INT n, PARCEL_INT nadd, PARCEL_INT num, sycl::queue *q, PARCEL_INT pad = 0
		  ){
    resize(n,nadd,num,q,pad);
  }
  
  ~parallel_vector(){}
  
  parcel::vector<parceltype> &vec(PARCEL_INT id){
    vec_util_.setptr(n_,vec_add_,vec_,(size_t)ld_*id);
    return vec_;
  }

  parcel::vector<parceltype> &vec_add(){
    return vec_add_;
  }
  
  parcel::vector<parceltype> &vec_add(PARCEL_INT id){
    //   列 id の halo 込みの view (長さ n + nadd)。詰め物 (pad) があるとき ld_ は n + nadd より長いが、
    //   SpMV の入力として渡す view は局所 + halo の長さに揃える。
    vec_util_.setptr((size_t)n_ + (size_t)nadd_,vec_add_,vec_,(size_t)ld_*id);
    return vec_;
  }

  parcel::vector<parceltype> &vec_add(PARCEL_INT id, PARCEL_INT num){
    vec_util_.setptr((size_t)ld_*num,vec_add_,vec_,(size_t)ld_*id);
    return vec_;
  }
  
};
  
}

#if PARCEL_MPI
#include "./parcel_mpi/parcel_mpi.hpp"
#else
namespace parcel{
  // MPI なしのビルド (-DPARCEL_MPI=0) のコミュニケータ。
  //   1 プロセスで動くので、ランク番号は常に 0、プロセス数は常に 1。
  //   MPI ありのビルドと同じ書き方 (comm.mpi_rank()、set_crsMat(&comm, nstart, nend, ...)) で
  //   使えるように置く。MPI ありの comm(MPI_Comm, sycl::queue*) に当たるのは comm(sycl::queue*)。
  //   MPI の型を返す mpi_comm() と、通信の関数 (allreduce など) は無い。
  class comm {
    sycl::queue *q_ = nullptr;
  public :
    comm(){}
    explicit comm(sycl::queue *q) : q_(q) {}
    ~comm(){}
    void setQueue(sycl::queue *q){ q_ = q; }
    sycl::queue *queue(){ return q_; }
    int mpi_rank(){ return 0; }
    int mpi_size(){ return 1; }
    int rank(){ return 0; }
    int size(){ return 1; }
    void mpi_barrier(){}
    void barrier(){}
  };
}
#endif

namespace parcel{
  // double-double の set_crsMat で、hi/lo を組にして set_localmatrix の並べ替えに通すための型
  //   (set_localmatrix は値を Ttype で 1 要素ずつ移すだけなので、組でよい)
  struct parcel_ddpair{
    double hi = 0.0;
    double lo = 0.0;
  };

  template<class parceltype>
  class crsMat{
    vector<parceltype> mat_;
    // デバイス上の行ポインタと列番号はランク内の添字型 PARCEL_LINT (既定 32 ビット)。
    //   列番号は set_localmatrix で局所化した後の番号 (自ランクの行 0..n_local-1、袖 n_local..)。
    //   変換と範囲の検査は matrix::set_crsMat (check_lint_range) で行う。
    util_sycl::vector<PARCEL_LINT> rowptr_;
    util_sycl::vector<PARCEL_LINT> crsCol_;
    size_t n_; // ローカル行数 (確保の都合で 0 行でも 1 になる)
    size_t nlocal_ = 0; // 呼び出し側が指定したローカル行数 (0 行もそのまま)
    size_t nnz_; // 非ゼロ要素数
    size_t maxCol_; // 最大列数
    
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};

#if PARCEL_MPI

    // MPI並列分散行列
    size_t gn_; // グローバル行数
    PARCEL_INT nstart_; // 開始行数
    PARCEL_INT nend_; // 終了行数
    
    // set_comm を通らない場合に comm() が nullptr を返すよう初期化する。
    parcel::comm *comm_ = nullptr;

    // 各プロセスの担当範囲
    std::vector<PARCEL_INT> nstart_list_;
    std::vector<PARCEL_INT> nend_list_;
    
    // 各プロセスへの送信データ数
    std::vector<PARCEL_INT> nsend_list_;
    // 各プロセスへの送信列番号リスト(グローバル列番号)
    std::vector< std::vector<PARCEL_INT> > nsend_gcol_list_;
    
    // 各プロセスからの受信データ数
    std::vector<PARCEL_INT> nrecv_list_;
    // 各プロセスから受け取った値を置く袖領域の先頭 (局所番号 nblock からの位置)。
    //   袖の列は set_localmatrix で大域列番号の昇順に並べてあるので、送り元の担当範囲 (nstart) の
    //   昇順に積み上げる (ランク番号の順ではない。行ブロックがランク番号の順に並んでいない分割でも正しい)。
    std::vector<PARCEL_INT> recv_ptr_;
    // 各プロセスからの受信列番号リスト(グローバル列番号)
    std::vector< std::vector<PARCEL_INT> > nrecv_gcol_list_;

    // グローバル列番号
    std::vector<PARCEL_INT> gcol_list_;

    // ------------------------------------------------------------
    // SYCL用変数
    std::vector< util_sycl::vector<PARCEL_MAX_FLOTING_TYPE> > send_comm_buff_;
    std::vector< util_sycl::vector<PARCEL_MAX_FLOTING_TYPE> > recv_comm_buff_;

    std::vector< std::vector<PARCEL_MAX_FLOTING_TYPE> > send_comm_hostbuff_;
    std::vector< std::vector<PARCEL_MAX_FLOTING_TYPE> > recv_comm_hostbuff_;
    
    // 各プロセスへ送る値の自ランク内の番号 (局所番号 = 大域列番号 - nstart) のリスト (デバイス側)。
    //   ランク内の型 PARCEL_LINT で置く (set_comm_list)。
    std::vector< util_sycl::vector<PARCEL_LINT> > nsend_gcol_parcel_list_;
    
    // 各プロセスからの受信列番号リスト(グローバル列番号。デバイス側には置くが、カーネルでは使っていない)
    std::vector< util_sycl::vector<PARCEL_INT> > nrecv_gcol_parcel_list_;
    
#endif
    
  public:

#include "mixed_precision_comm_crs.hpp"
    
    void set_queue(sycl::queue* q){
      q_ = q;
      return ;    
    }

#if PARCEL_MPI

    template<class Tvecx>
    void SpMVcomm(
		  sycl::queue *que_,Tvecx *x
		  ){

      MPI_Comm mpi_comm = comm_->mpi_comm();
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);

      PARCEL_INT nstart = nstart_list_[myrank];
      PARCEL_INT nend   = nend_list_[myrank];
      PARCEL_INT nblock = nend - nstart + 1;
      
      que_[0].wait();
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  Tvecx *send_comm_buff = (Tvecx *) send_comm_buff_[irank].data();
	  SpMV_comm_pack< Tvecx >( que_, (PARCEL_LINT)nsend_list_[irank], nsend_gcol_parcel_list_[irank].data(), x, send_comm_buff );
	}
      }
      que_[0].wait();
      
      // --------------------------------------------------
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  send_comm_buff_[irank].getValue(send_comm_hostbuff_[irank].data());
	}
      }

      std::vector<MPI_Request> send_req(nprocess);
      std::vector<MPI_Request> recv_req(nprocess);      
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){
	  int nsend = sizeof(Tvecx) * nsend_list_[irank];
	  if(nsend_list_[irank]>0){
	    MPI_Isend( send_comm_hostbuff_[irank].data(), nsend, MPI_BYTE,irank,irank,mpi_comm,&send_req[irank]); 
	  }
	  if(nrecv_list_[irank]>0){
	    int nrecv = sizeof(Tvecx) * nrecv_list_[irank];
	    MPI_Irecv( recv_comm_hostbuff_[irank].data(), nrecv, MPI_BYTE,irank,myrank,mpi_comm,&recv_req[irank]); 
	  }	  
	}
      }

      MPI_Status istat;
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // send_req は nsend>0, recv_req は nrecv>0 のときだけ MPI_Isend/MPI_Irecv で
	  // 初期化される. 条件を取り違えると, 送受信数が非対称なランク対で未初期化の
	  // MPI_Request を wait して segmentation fault になる.
	  if(nsend_list_[irank]>0){
	    MPI_Wait(&send_req[irank],&istat);
	  }
	  if(nrecv_list_[irank]>0){
	    MPI_Wait(&recv_req[irank],&istat);
	  }
	}
      }
	              
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nrecv_list_[irank]>0) ){
	  recv_comm_buff_[irank].setValue(recv_comm_hostbuff_[irank].data());
	}
      }
      // --------------------------------------------------
      
      que_[0].wait();
      // 置き場所は set_comm_list で求めた recv_ptr_ (送り元の担当範囲の昇順。上の宣言を参照)
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nrecv_list_[irank]>0) ){
	  Tvecx *recv_comm_buff = (Tvecx *) recv_comm_buff_[irank].data();
	  // 袖の位置は局所番号 (n_local + nadd が PARCEL_LINT に収まることは set_crsMat で確認済み)
	  SpMV_comm_unpack< Tvecx >( que_, (PARCEL_LINT)nblock, (PARCEL_LINT)nrecv_list_[irank], (PARCEL_LINT)recv_ptr_[irank], x, recv_comm_buff );
	}
      }      
      que_[0].wait();

      return ;
      
    }
    
    template<class Ttype>
    void set_localmatrix(
			 PARCEL_INT nstart,PARCEL_INT nend,
			 PARCEL_INT *rowptr,
			 PARCEL_INT *crsCol,
			 Ttype *mat_mem
			 ){
      MPI_Comm mpi_comm = comm_->mpi_comm();
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);
      
      PARCEL_INT nblock = nend - nstart + 1;
      PARCEL_INT nodiag_ncol = 0;
      for(PARCEL_INT j=0;j<nblock;j++){
	for(PARCEL_INT i=rowptr[j];i<rowptr[j+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    nodiag_ncol  = nodiag_ncol + 1;
	  }
	}
      }
      
      std::vector<PARCEL_INT> globalcol_list(nodiag_ncol);
      PARCEL_INT iptr = 0;
      for(PARCEL_INT j=0;j<nblock;j++){
	for(PARCEL_INT i=rowptr[j];i<rowptr[j+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    globalcol_list[iptr] = crsCol[i];
	    iptr  = iptr + 1;
	  }
	}
      }

      std::sort(globalcol_list.begin(), globalcol_list.end());
      globalcol_list.erase(
			   std::unique(globalcol_list.begin(), globalcol_list.end()),
			   globalcol_list.end());
  
      std::vector<PARCEL_INT> work_int(nblock);
      std::vector<Ttype> work_type(nblock);
      
      for(PARCEL_INT irow=0;irow<nblock;irow++){

	PARCEL_INT row_ncol = rowptr[irow+1] - rowptr[irow];
	if(row_ncol>work_type.size()){
	  work_int.resize(row_ncol);
	  work_type.resize(row_ncol);
	}

	PARCEL_INT icol_ptr = 0;
	for(PARCEL_INT i = rowptr[irow];i<rowptr[irow+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (nstart<=icol) && (icol<=nend) ){
	    work_int[icol_ptr]  = crsCol[i] - nstart;
	    work_type[icol_ptr] = mat_mem[i];
	    icol_ptr = icol_ptr + 1;
	  }
	}

	PARCEL_INT igcol0 = 0;
	for(PARCEL_INT i=rowptr[irow];i<rowptr[irow+1];i++){
	  PARCEL_INT icol = crsCol[i];
	  if( (icol<nstart) || (nend<icol) ){
	    work_int[icol_ptr] = -1;
	    for(PARCEL_INT igcol=igcol0;igcol<globalcol_list.size();igcol++){
	      if( globalcol_list[ igcol ] == crsCol[i]){
		work_int[icol_ptr] = nblock + igcol ;
		igcol0 = igcol;
		break ;
	      }
	    }
	    work_type[icol_ptr] = mat_mem[i];
	    icol_ptr = icol_ptr + 1;
	  }
	}

	icol_ptr = 0;
	for(PARCEL_INT i=rowptr[irow];i<rowptr[irow+1];i++){
	  mat_mem[i] = work_type[icol_ptr];
	  crsCol[i]  = work_int[icol_ptr];
	  icol_ptr   = icol_ptr + 1;
	}

      }

      
      gcol_list_.resize(globalcol_list.size());
      for(PARCEL_INT i=0;i<globalcol_list.size();i++){
	gcol_list_[i] = globalcol_list[i];
      }

      
      return ;
      
    }

    
    void set_comm_list(){
      
      MPI_Comm mpi_comm = comm_->mpi_comm();
      PARCEL_INT nstart = nstart_;
      PARCEL_INT nend   = nend_;
      
      int nprocess;
      MPI_Comm_size(mpi_comm,&nprocess);
      int myrank;
      MPI_Comm_rank(mpi_comm,&myrank);

      // 各プロセスの担当範囲情報を交換
      std::vector<PARCEL_INT> sbuf(nprocess);
      nstart_list_.resize(nprocess);
      nend_list_.resize(nprocess);
      for(PARCEL_INT i=0;i<nprocess;i++){sbuf[i] = nstart;}
      MPI_Alltoall(
		   sbuf.data(),1,MPI_PARCEL_INT,
		   nstart_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);
      for(PARCEL_INT i=0;i<nprocess;i++){sbuf[i] = nend;}
      MPI_Alltoall(
		   sbuf.data(),1,MPI_PARCEL_INT,
		   nend_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);
      
      // 各プロセスで必要な送信データ数情報を交換
      nsend_list_.resize(nprocess,0);
      nrecv_list_.resize(nprocess,0);
      for(PARCEL_INT i=0;i<gcol_list_.size();i++){
	PARCEL_INT icol = gcol_list_[i];
	for(PARCEL_INT irank=0;irank<nprocess;irank++){
	  if( (nstart_list_[irank]<=icol) && (icol<=nend_list_[irank]) ){
	    nrecv_list_[irank] = nrecv_list_[irank] +1;
	  }
	}
      }  
      MPI_Alltoall(
		   nrecv_list_.data(),1,MPI_PARCEL_INT,
		   nsend_list_.data(),1,MPI_PARCEL_INT,
		   mpi_comm);

      // 袖通信の個数の上限を確かめる。SpMVcomm は 1 ランク対あたり sizeof(値) * 個数 のバイト数を、
      //   set_comm_list は列番号の個数を MPI の int で渡すので、1 ランク対の袖が int (バイト数) を超えると
      //   桁あふれして黙って壊れる。64 ビットの PARCEL_INT で局所行数が大きいときに届きうるので止める
      //   (全ランクでそろえて投げる。1 ランクだけ投げると他が次の集団通信で止まるため)。
      {
	const long long lim = (long long)std::numeric_limits<int>::max()
	  / (long long)std::max(sizeof(PARCEL_MAX_FLOTING_TYPE), sizeof(PARCEL_INT));
	int bad = 0;
	for(int r = 0; r < nprocess; r++){
	  if((long long)nsend_list_[r] > lim || (long long)nrecv_list_[r] > lim) bad = 1;
	}
	int bad_any = 0;
	MPI_Allreduce(&bad, &bad_any, 1, MPI_INT, MPI_MAX, mpi_comm);
	if(bad_any){
	  throw std::overflow_error("parcel::crsMat::set_comm_list: halo exchange between a pair of ranks exceeds the MPI int count (in bytes); use more MPI ranks");
	}
      }

      // 受信した値を置く袖領域の位置 (送り元の担当範囲 nstart の昇順に積み上げる。recv_ptr_ の宣言を参照)
      {
	std::vector<int> order(nprocess);
	for(int r = 0; r < nprocess; r++) order[r] = r;
	std::sort(order.begin(), order.end(),
		  [&](int a, int b){ return nstart_list_[a] < nstart_list_[b]; });
	recv_ptr_.assign(nprocess, 0);
	PARCEL_INT acc = 0;
	for(int k = 0; k < nprocess; k++){
	  const int r = order[k];
	  recv_ptr_[r] = acc;
	  if(r != myrank) acc += nrecv_list_[r];
	}
      }

      // 送信してほしい列番号のベクトルのリストを交換
      nrecv_gcol_list_.resize(nprocess);
      nsend_gcol_list_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if(irank != myrank){
	  PARCEL_INT recv_nbuff = nrecv_list_[irank];
	  if( recv_nbuff > 0 ){
	    nrecv_gcol_list_[irank].resize(recv_nbuff);
	  }
	  PARCEL_INT send_nbuff = nsend_list_[irank];
	  if( send_nbuff > 0 ){
	    nsend_gcol_list_[irank].resize(send_nbuff);
	  }
	}
      }

      std::vector<PARCEL_INT> gcol_list_ptr(nprocess,0);
      for(PARCEL_INT i=0;i<gcol_list_.size();i++){
	PARCEL_INT icol = gcol_list_[i];
	for(PARCEL_INT irank=0;irank<nprocess;irank++){
	  if( (nstart_list_[irank]<=icol) && (icol<=nend_list_[irank]) ){
	    PARCEL_INT list_ptr = gcol_list_ptr[irank];	    
	    nrecv_gcol_list_[irank][list_ptr] = icol;
	    gcol_list_ptr[irank] = gcol_list_ptr[irank] + 1;
	  }
	}
      }

      std::vector<MPI_Request> send_req(nprocess);
      std::vector<MPI_Request> recv_req(nprocess);      
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // send
	  if(nrecv_list_[irank]>0){	    
	    MPI_Isend( nrecv_gcol_list_[irank].data(), nrecv_list_[irank],MPI_PARCEL_INT,irank,myrank,mpi_comm,&send_req[irank]); 
	  }
	  // recv
	  if(nsend_list_[irank]>0){
	    MPI_Irecv( nsend_gcol_list_[irank].data(), nsend_list_[irank],MPI_PARCEL_INT,irank,irank,mpi_comm,&recv_req[irank]); 
	  }	  
	}
      }
      
      MPI_Status istat;
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( irank != myrank){	  
	  // wait
	  if(nrecv_list_[irank]>0){
	    MPI_Wait(&send_req[irank],&istat);
	  }
	  if(nsend_list_[irank]>0){
	    MPI_Wait(&recv_req[irank],&istat);
	  }
	}
      }
      MPI_Barrier(mpi_comm);
      
      // host用MPIバッファ
      send_comm_hostbuff_.resize(nprocess);
      recv_comm_hostbuff_.resize(nprocess);

      // SYCL用MPIバッファ (行列のキューで確保する。set_comm_list() の前に set_queue() されている)
      send_comm_buff_.resize(nprocess);
      recv_comm_buff_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank]>0) ){
	  send_comm_buff_[irank].resize(nsend_list_[irank], q_);
	  send_comm_hostbuff_[irank].resize(nsend_list_[irank]);
	}
	if( (myrank != irank) && (nrecv_list_[irank]>0) ){
	  recv_comm_buff_[irank].resize(nrecv_list_[irank], q_);
	  recv_comm_hostbuff_[irank].resize(nrecv_list_[irank]);
	}
      }
      
      nsend_gcol_parcel_list_.resize(nprocess);
      nrecv_gcol_parcel_list_.resize(nprocess);
      for(PARCEL_INT irank=0;irank<nprocess;irank++){
	if( (myrank != irank) && (nsend_list_[irank] > 0) ){
	  // 大域列番号から自ランク内の番号に直して PARCEL_LINT で置く
	  //   (送る列は自ランクの担当範囲 [nstart, nend] にあるので 0 .. n_local-1 に収まる)
	  std::vector<PARCEL_LINT> lcol(nsend_list_[irank]);
	  for(size_t k = 0; k < lcol.size(); k++){
	    lcol[k] = (PARCEL_LINT)(nsend_gcol_list_[irank][k] - nstart);
	  }
	  nsend_gcol_parcel_list_[irank].resize( nsend_list_[irank], q_ );
	  nsend_gcol_parcel_list_[irank].setValue( lcol.data() );
	}
	if( (myrank != irank) && (nrecv_list_[irank] > 0) ){
	  nrecv_gcol_parcel_list_[irank].resize( nrecv_list_[irank], q_ );
	  nrecv_gcol_parcel_list_[irank].setValue( nrecv_gcol_list_[irank].data() );
	}
      }
      
      
      return ;
      
    }
    
    // 袖の数 (他ランクから受け取る値の数)。set_localmatrix の後で使える
    //   (set_comm_list の前に PARCEL_LINT の範囲を検査するため)
    size_t n_halo() const {
      return gcol_list_.size();
    }

    PARCEL_INT get_send_size(){
      PARCEL_INT nvec_size = 0;
      for(PARCEL_INT i=0;i<nsend_list_.size();i++){
	if( nsend_list_[i] > 0){ 
	  nvec_size = nvec_size + nsend_list_[i];
	}
      }
      return nvec_size;
    }

    PARCEL_INT get_recv_size(){
      PARCEL_INT nvec_size = 0;
      for(PARCEL_INT i=0;i<nrecv_list_.size();i++){
	if( nrecv_list_[i] > 0){ 
	  nvec_size = nvec_size + nrecv_list_[i];
	}
      }
      return nvec_size;
    }
    
    void set_comm(parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend){
      comm_ = comm;
      nstart_ = nstart;
      nend_   = nend;
      size_t nlocal = nend - nstart + 1;

      int myrank   = comm[0].mpi_rank();
      int nprocess = comm[0].mpi_size();
      
  

      
      return ;      
      
    }
    
    parcel::comm *comm(){
      return comm_;
    }    
    
#endif
    
    void resize(size_t n,size_t nnz, sycl::queue* q){
      q_ = q;
      nlocal_ = n;
      n_   = n>1? n:1;
      nnz_ = nnz>1? nnz:1;
      mat_.resize(nnz_ , q_);
      rowptr_.resize(n_+1 , q_);
      crsCol_.resize(nnz_ , q_);
      maxCol_ = 0;
    }

    // 値の配列だけを確保し、行ポインタ・列番号は別の CRS 行列 src と共有する。
    //   src の非零の並びと同じ並びで値を持つ行列 (Jacobi 前処理の (D^-1)(L+U) を対角 0 で A の形に置いたもの)
    //   に使う。src はこの行列より長く生きていなければならない。
    //   alloc_values=false のときは値の配列も 1 要素だけにする (値を持たない Jacobi 前処理)。
    template<class Tsrc>
    void resize_share_structure(size_t n, size_t nnz, sycl::queue* q, crsMat<Tsrc> &src, bool alloc_values = true){
      q_ = q;
      nlocal_ = n;
      n_   = n>1? n:1;
      nnz_ = nnz>1? nnz:1;
      mat_.resize(alloc_values ? nnz_ : 1 , q_);
      rowptr_.share(src.rowptr_data(), n_+1, q_);
      crsCol_.share(src.crsCol_data(), nnz_, q_);
      maxCol_ = 0;
    }

    crsMat(){}
    crsMat(size_t n, size_t nnz,sycl::queue* q){
       resize(n,nnz,q);
    }
    ~crsMat(){
    }

    sycl::queue *queue(){
      return q_;
    }

    auto mat_data(){
      return mat_.data();
    }

    vector<parceltype> &mat(){
      return mat_;
    }

    auto mat_data_hi(){
      return mat_.data_hi();
    }

    auto mat_data_lo(){
      return mat_.data_lo();
    }

    template<class Ttype>
    void mat_setValue(Ttype *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem_hi,double *mem_lo){
      mat_.setValue(mem_hi,mem_lo);
    }

    void mat_setValue_hi(double *mem){
      mat_.setValue_hi(mem);
    }
    void mat_setValue_lo(double *mem){
      mat_.setValue_lo(mem);
    }

    template<class Ttype>
    void mat_getValue(Ttype *mem){
      mat_.getValue(mem);
    }

    void mat_getValue(double *mem){
      mat_.getValue(mem);
    }

    void mat_getValue(double *mem_hi,double *mem_lo){
      mat_.getValue(mem_hi,mem_lo);
    }

    void mat_getValue_hi(double *mem){
      mat_.getValue_hi(mem);
    }
    void mat_getValue_lo(double *mem){
      mat_.getValue_lo(mem);
    }

    // 行ポインタ・列番号 (局所番号) の型はランク内の添字型 PARCEL_LINT
    util_sycl::vector<PARCEL_LINT> *rowptr(){
      return &rowptr_;
    }

    PARCEL_LINT *rowptr_data(){
      return rowptr_.data();
    }

    util_sycl::vector<PARCEL_LINT> *crsCol(){
      return &crsCol_;
    }

    PARCEL_LINT *crsCol_data(){
      return crsCol_.data();
    }

    void rowptr_setValue(PARCEL_LINT *mem){
      rowptr_.setValue(mem);
    }
    void crsCol_setValue(PARCEL_LINT *mem){
      crsCol_.setValue(mem);
      size_t maxCol = 0;
    }

    void rowptr_getValue(PARCEL_LINT *mem){
      rowptr_.getValue(mem);
    }
    void crsCol_getValue(PARCEL_LINT *mem){
      crsCol_.getValue(mem);
      size_t maxCol = 0;
    }

    void set_maxCol(){

      q_[0].wait();
      size_t MAX   = n_;
#if _A64FX_HIPSYCL_
      const size_t nblock      = parcel::nblock_runtime();
      const size_t nlocalblock = 8192;
#else
      const size_t nblock      = n_;
      const size_t nlocalblock = 1;
#endif

      // 行ポインタはランク内の添字型 PARCEL_LINT。行あたりの最大列数も同じ型で持つ
      util_sycl::vector<PARCEL_LINT> nmaxCol_all_;
      nmaxCol_all_.resize(nblock,q_);   // 行列のキューのカーネルが書くので同じキューで確保する
      PARCEL_LINT* nmaxCol_all = nmaxCol_all_.data();
      sycl::buffer<PARCEL_LINT> buff(nmaxCol_all,nblock);
      PARCEL_LINT* Rowptr = rowptr_.data();
      const size_t N = n_;
      q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
	  size_t nmaxCol = 0;
	  for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	    const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	    for(size_t i=iii;i<imax;i++){
	      size_t nCol = Rowptr[i+1] - Rowptr[i];
	      nmaxCol = nmaxCol < nCol ? nCol :nmaxCol;
	    }
	  }
	  nmaxCol_all[ii] = (PARCEL_LINT)nmaxCol;   // 行の非零数 <= nnz_local (PARCEL_LINT に収まる)
	});
      q_[0].wait();
      std::vector<PARCEL_LINT> nmaxCol_all_host(nblock,0);
      nmaxCol_all_.getValue(nmaxCol_all_host.data(),nblock);
      size_t nmaxCol = 0;
      for(size_t i=0;i<nblock;i++){
	size_t nCol = nmaxCol_all_host[i];
	nmaxCol = nmaxCol < nCol ? nCol :nmaxCol;
      }
      maxCol_ = nmaxCol;
    }

    size_t maxCol(){
      return maxCol_;
    }

    size_t nnz(){
      return nnz_;
    }

    size_t n(){
      return n_;
    }

    // 呼び出し側が指定したローカル行数 (0 行のランクでも 0 を返す)
    size_t n_local() const {
      return nlocal_;
    }

  };

  template<class parceltype>
  class diaMat{
    vector<parceltype> mat_;
    util_sycl::vector<PARCEL_INT> offset_;
    PARCEL_INT max_up_offset_   ;
    PARCEL_INT max_down_offset_ ;
    size_t n_;
    size_t nnd_;
    sycl::queue *q_;
    sycl::queue q_default{sycl::default_selector{}};
  public:

    void set_queue(sycl::queue* q){
      q_ = q;
      return ;    
    }

    void resize(size_t n,size_t nnd, sycl::queue *q){
      q_   = q;
      n_   = n>1? n:1;
      nnd_ = nnd>1? nnd:1;

      mat_.resize(n_*nnd_ , q_);
      offset_.resize(nnd_ ,q_);
    }

    diaMat(){}
    diaMat(size_t n, size_t nnd,sycl::queue* q){
       resize(n,nnd,q);
    }
    ~diaMat(){
    }

    sycl::queue *queue(){
      return q_;
    }

    template<class Ttype>
    void mat_setValue(Ttype *mem){
      mat_.setValue(mem);
    }

    void mat_setValue(double *mem_hi,double *mem_lo){
      mat_.setValue(mem_hi,mem_lo);
    }

    void mat_setValue_hi(double *mem){
      mat_.setValue_hi(mem);
    }
    void mat_setValue_lo(double *mem){
      mat_.setValue_lo(mem);
    }

    vector<parceltype> &mat(){
      return mat_;
    }

    auto mat_data(){
      return mat_.data();
    }

    auto mat_data_hi(){
      return mat_.data_hi();
    }

    auto mat_data_lo(){
      return mat_.data_lo();
    }

    PARCEL_INT *offset_data(){
      return offset_.data();
    }

    void offset_setValue(PARCEL_INT *mem){
      PARCEL_INT max_up_offset  = 0;
      PARCEL_INT max_down_offset= 0;
      for(size_t i = 0; i<nnd_;i++){
	max_up_offset   = max_up_offset   > mem[i] ? mem[i] : max_up_offset;
	max_down_offset = max_down_offset < mem[i] ? mem[i] : max_down_offset;
      }
      max_up_offset_   = max_up_offset;
      max_down_offset_ = max_down_offset;
      offset_.setValue(mem);
    }

    util_sycl::vector<PARCEL_INT> &offset(){
      return offset_;
    };

    size_t nnd(){
      return nnd_;
    }

    size_t n(){
      return n_;
    }

    void set_max_up_offset(PARCEL_INT max_up_offset){
      max_up_offset_ = max_up_offset;
      return ;
    }
    void set_max_down_offset(PARCEL_INT max_down_offset){
      max_down_offset_ = max_down_offset;
      return ;
    }

    PARCEL_INT max_up_offset(){
      return max_up_offset_;
    }
    PARCEL_INT max_down_offset(){
      return max_down_offset_;
    }

  };

  // キューが in-order か確かめる。
  //   PARCEL のカーネル (SpMV・axpy・dot・前処理・ハロ詰め) は sycl::event も depends_on も
  //   使わずに 1 本のキューへ順に投入し、前のカーネルが書いたベクトルを次のカーネルが読む。
  //   SYCL 2020 の既定 (out-of-order) キューでは、accessor を使わない USM カーネル同士の
  //   実行順は規格上保証されないので、in-order キューが要る。
  //     sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}} で作ること
  //     (デバイスの選び方は既定構築の sycl::queue q; と同じ)。
  //   検査を外すときは -DPARCEL_REQUIRE_IN_ORDER=0。
  //   呼ぶ場所: 全ソルバの solver() (pcg / pbicgstab / pgmres / pcagmres (Q の作り方によらず) /
  //     power / inverse_power / ks_arnoldi) と parcel::matrix::set_crsMat (2 種)。
  //     違反は std::invalid_argument。キューは全ランクで同じ作り方をする前提なので、
  //     全ランクが同じ分岐を取る (集団通信の前に投げる)。
#ifndef PARCEL_REQUIRE_IN_ORDER
#define PARCEL_REQUIRE_IN_ORDER 1
#endif
  inline void require_in_order(sycl::queue *q, const char *who){
#if PARCEL_REQUIRE_IN_ORDER
    if(q == nullptr){
      throw std::invalid_argument(std::string("parcel::") + who
                                  + ": no sycl::queue is set (pass an in-order queue, e.g. sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}})");
    }
    if(!q->is_in_order()){
      throw std::invalid_argument(std::string("parcel::") + who
                                  + ": the sycl::queue must be in-order; construct it as sycl::queue q{sycl::default_selector_v, sycl::property::queue::in_order{}}"
                                  + " (PARCEL enqueues dependent kernels without events; build with -DPARCEL_REQUIRE_IN_ORDER=0 to skip this check)");
    }
#else
    (void)q; (void)who;
#endif
  }

  // 精度を省略したら倍精度 (parcel::matrix<> A(&q))
  template<class parceltype = parcel_double>
  class matrix{
    // 既定構築でも未設定 (-1) と nullptr に初期化する
    size_t matrix_type_ = (size_t)-1;
    crsMat<parceltype> crsMat_;
    diaMat<parceltype> diaMat_;
    sycl::queue* q_ = nullptr;
#if !PARCEL_MPI
    // MPI なしのビルドで set_crsMat(comm, nstart, nend, ...) に渡された comm
    parcel::comm *comm_ = nullptr;
#endif
  public:

    matrix(sycl::queue* q){
      q_ = q;
      matrix_type_ = -1;
    }
    matrix(){}
    ~matrix(){}

    // ローカル行数。ソルバはベクトル長からではなくここから取る。
    //   CRS : set_crsMat に渡した n (MPI では nend-nstart+1 に等しい)
    //   DIA : diaMat の行数 (DIA は MPI 分割に対応していないので全体の行数)
    size_t n_local(){
      if(matrix_type_==CRS_MATRIX){ return crsMat_.n_local(); }
      if(matrix_type_==DIA_MATRIX){ return diaMat_.n(); }
      throw std::runtime_error("parcel::matrix::n_local: matrix is not set (call set_crsMat / set_diaMat first)");
    }

    crsMat<parceltype> *crsMat_data(){
      return &crsMat_;
    }

#if PARCEL_MPI
    
    // コミュニケータを持たない行列 (DIA 形式、未設定、set_comm を通っていない CRS)
    //   では例外を投げる。
    parcel::comm *comm(){
      if(matrix_type_==CRS_MATRIX && crsMat_.comm() != nullptr){
	return crsMat_.comm();
      }
      throw std::runtime_error("parcel::matrix::comm: matrix has no MPI communicator (use set_crsMat(comm,nstart,nend,...); DIA matrices are not supported with PARCEL_MPI)");
    }
    
    PARCEL_INT get_recv_size(){
      if(matrix_type_==CRS_MATRIX && crsMat_.comm() != nullptr){
	return crsMat_.get_recv_size();
      }
      throw std::runtime_error("parcel::matrix::get_recv_size: matrix has no MPI communicator (use set_crsMat(comm,nstart,nend,...))");
    }
    
#else
    
    // MPI なしのビルド。MPI ありと同じ形で使えるように置く。
    //   comm() は set_crsMat(comm, nstart, nend, ...) で渡した comm を返す。comm を渡さない
    //   set_crsMat・DIA 形式・未設定の行列では、MPI ありと同じく例外を投げる。
    //   get_recv_size() は袖が無いので 0 (CRS を設定していなければ例外)。
    parcel::comm *comm(){
      if(matrix_type_==CRS_MATRIX && comm_ != nullptr){
	return comm_;
      }
      throw std::runtime_error("parcel::matrix::comm: matrix has no communicator (use set_crsMat(comm,nstart,nend,...))");
    }
    
    PARCEL_INT get_recv_size(){
      if(matrix_type_==CRS_MATRIX){
	return 0;
      }
      throw std::runtime_error("parcel::matrix::get_recv_size: CRS matrix is not set (use set_crsMat)");
    }
    
#endif    
    
    void resize(sycl::queue* q,size_t n,size_t nnz_nnd,size_t matrix_type){
      set_queue(q);
      matrix_type_ = matrix_type;
      if(matrix_type_==CRS_MATRIX){
	crsMat_.resize(n,nnz_nnd,q_);
      }
      if(matrix_type_==DIA_MATRIX){
	diaMat_.resize(n,nnz_nnd,q_);
      }
      return ;
    }

    void resize(size_t n,size_t nnz_nnd,size_t matrix_type){
      matrix_type_ = matrix_type;
      if(matrix_type_==CRS_MATRIX){
	crsMat_.resize(n,nnz_nnd,q_);
      }
      if(matrix_type_==DIA_MATRIX){
	diaMat_.resize(n,nnz_nnd,q_);
      }
      return ;
    }

    // 入力の行数・非零数が大域の型 PARCEL_INT に収まるか。収まらないと rowptr が桁あふれして
    //   黙って壊れるので止める (-DPARCEL_INT32 でビルドしたときだけ起こりうる)。
    //   デバイス上の型 PARCEL_LINT の範囲は、局所化の後に check_lint_range で調べる。
    static void check_int_range(size_t n, size_t nnz){
      const size_t lim = (size_t)std::numeric_limits<PARCEL_INT>::max();
      if(n > lim || nnz > lim){
	throw std::overflow_error("parcel::matrix::set_crsMat: n or nnz exceeds PARCEL_INT range (32-bit build); rebuild without -DPARCEL_INT32 (64-bit PARCEL_INT is the default)");
      }
    }

    // 入力 CRS と行ブロック分割の検査。
    //   分割に隙間・重複があると袖通信の受信バッファが未初期化のまま使われて黙って誤った結果になり、
    //   rowptr の不正は配列の範囲外アクセスになるため、事前に調べる。検査する内容:
    //     (1) n == nend - nstart + 1 かつ n >= 1
    //         0 行のランクは扱えない (crsMat が長さを 1 に切り上げて利用者の配列を読むため、
    //         範囲外・null の読み出しになる)。C API も n_local < 1 を PARCEL_ERR_ARG で止める。
    //     (2) rowptr[0] == 0、rowptr が単調非減少、rowptr[n] == nnz
    //     (3) 全ランクの行ブロックが [0, 全体の行数) を隙間・重複なく覆う
    //         (ランク番号の順に並んでいなくてよい。ライブラリは範囲を交換して使う)
    //     (4) 列番号が 0 以上、全体の行数未満
    //   各ランクの判定を集団通信で集めてから、全ランクが同じ std::invalid_argument を投げる
    //   (1 ランクだけ投げると他のランクが次の集団通信で止まるため)。
    //   費用は非零要素の走査 1 回と MPI_Allgather 2 回。-DPARCEL_NO_CRS_CHECK=1 で外せる。
#ifndef PARCEL_NO_CRS_CHECK
#define PARCEL_NO_CRS_CHECK 0
#endif
    // 各ランク内の検査 (1)(2)。問題があれば説明を返す (無ければ空)
    static std::string check_crs_local(long long nstart, long long nend, size_t n, size_t nnz,
				       const PARCEL_INT *rowptr){
      const long long nl = (long long)n;
      if(nend - nstart + 1 != nl){
	return "n (" + std::to_string(nl) + ") != nend - nstart + 1 (nstart = " + std::to_string(nstart)
	  + ", nend = " + std::to_string(nend) + ")";
      }
      if(nl < 1){
	return "rank has no rows (n = 0); each rank must own at least one row";
      }
      if(nstart < 0){
	return "nstart (" + std::to_string(nstart) + ") < 0";
      }
      if(rowptr == nullptr) return "rowptr is null";
      if(rowptr[0] != 0){
	return "rowptr[0] (" + std::to_string((long long)rowptr[0]) + ") != 0";
      }
      for(long long i = 0; i < nl; i++){
	if(rowptr[i+1] < rowptr[i]){
	  return "rowptr decreases at local row " + std::to_string(i) + " (rowptr[" + std::to_string(i) + "] = "
	    + std::to_string((long long)rowptr[i]) + ", rowptr[" + std::to_string(i+1) + "] = "
	    + std::to_string((long long)rowptr[i+1]) + ")";
	}
      }
      if((long long)rowptr[nl] != (long long)nnz){
	return "rowptr[n] (" + std::to_string((long long)rowptr[nl]) + ") != nnz (" + std::to_string((long long)nnz) + ")";
      }
      return "";
    }
    // 列番号の検査 (4)。rowptr が正しいことを確かめてから呼ぶ
    static std::string check_crs_cols(size_t n, const PARCEL_INT *rowptr, const PARCEL_INT *col,
				      long long n_global){
      for(long long i = 0; i < (long long)n; i++){
	for(long long k = rowptr[i]; k < (long long)rowptr[i+1]; k++){
	  const long long c = (long long)col[k];
	  if(c < 0 || c >= n_global){
	    return "column index " + std::to_string(c) + " at local row " + std::to_string(i)
	      + " is out of range [0, " + std::to_string(n_global) + ")";
	  }
	}
      }
      return "";
    }
#if PARCEL_MPI
    // 全ランクで検査する。問題があれば全ランクが同じ例外を投げる
    static void check_crs_input(parcel::comm *comm, PARCEL_INT nstart, PARCEL_INT nend, size_t n, size_t nnz,
				const PARCEL_INT *rowptr, const PARCEL_INT *col){
#if !PARCEL_NO_CRS_CHECK
      if(comm == nullptr){
	throw std::invalid_argument("parcel::matrix::set_crsMat: comm is null");
      }
      MPI_Comm mc = comm->mpi_comm();
      int np = 1, me = 0;
      MPI_Comm_size(mc, &np);
      MPI_Comm_rank(mc, &me);
      // 集める: (nstart, n, 各ランク内の検査で問題があったか)
      std::string err = check_crs_local((long long)nstart, (long long)nend, n, nnz, rowptr);
      long long mine[3] = { (long long)nstart, (long long)n, err.empty() ? 0LL : 1LL };
      std::vector<long long> all(3 * (size_t)np);
      MPI_Allgather(mine, 3, MPI_LONG_LONG, all.data(), 3, MPI_LONG_LONG, mc);
      std::string msg;
      int nbad = 0;
      for(int r = 0; r < np; r++) if(all[3*r+2]) nbad++;
      // 行ブロックの被覆 (3)。0 行のランクは (1) で止まるので、ここには来ない
      long long n_global = 0;
      std::string cover;
      {
	std::vector<std::pair<long long,long long> > blk;   // (開始行, 行数)
	for(int r = 0; r < np; r++){
	  n_global += all[3*r+1];
	  if(all[3*r+1] > 0) blk.push_back(std::make_pair(all[3*r], all[3*r+1]));
	}
	std::sort(blk.begin(), blk.end());
	long long next = 0;
	for(size_t b = 0; b < blk.size() && cover.empty(); b++){
	  if(blk[b].first > next){
	    cover = "rows " + std::to_string(next) + " .. " + std::to_string(blk[b].first - 1)
	      + " are not owned by any rank";
	  } else if(blk[b].first < next){
	    cover = "row " + std::to_string(blk[b].first) + " is owned by more than one rank";
	  }
	  next = std::max(next, blk[b].first + blk[b].second);
	}
      }
      // 列番号 (4)。(1)(2)(3) が通ったときだけ見る (n_global が確定しないため)
      int col_bad = 0;
      std::string col_err;
      if(nbad == 0 && cover.empty()){
	col_err = check_crs_cols(n, rowptr, col, n_global);
	int mine_bad = col_err.empty() ? 0 : 1;
	MPI_Allreduce(&mine_bad, &col_bad, 1, MPI_INT, MPI_SUM, mc);
      }
      if(nbad == 0 && cover.empty() && col_bad == 0) return;
      // 例外の文言: 共通部分 (全ランク同じ) + 自ランクの詳細
      msg = "parcel::matrix::set_crsMat: invalid input (rank " + std::to_string(me) + " of " + std::to_string(np) + "): ";
      if(nbad > 0){
	std::string ranks;
	for(int r = 0; r < np; r++) if(all[3*r+2]) ranks += (ranks.empty() ? "" : ",") + std::to_string(r);
	msg += "local CRS error on rank(s) " + ranks;
	if(!err.empty()) msg += "; this rank: " + err;
      } else if(!cover.empty()){
	msg += "row block distribution: " + cover + " (n_global = " + std::to_string(n_global) + ")";
      } else {
	msg += std::to_string(col_bad) + " rank(s) have column indices out of range";
	if(!col_err.empty()) msg += "; this rank: " + col_err;
      }
      msg += ". Define PARCEL_NO_CRS_CHECK=1 to skip this check.";
      throw std::invalid_argument(msg);
#else
      (void)comm; (void)nstart; (void)nend; (void)n; (void)nnz; (void)rowptr; (void)col;
#endif
    }
#else
    static void check_crs_input(size_t n, size_t nnz, const PARCEL_INT *rowptr, const PARCEL_INT *col){
#if !PARCEL_NO_CRS_CHECK
      std::string err = check_crs_local(0, (long long)n - 1, n, nnz, rowptr);
      if(err.empty()) err = check_crs_cols(n, rowptr, col, (long long)n);
      if(!err.empty()){
	throw std::invalid_argument("parcel::matrix::set_crsMat: invalid input: " + err
				    + ". Define PARCEL_NO_CRS_CHECK=1 to skip this check.");
      }
#else
      (void)n; (void)nnz; (void)rowptr; (void)col;
#endif
    }
#endif

    // ランク内の添字型 PARCEL_LINT の範囲の検査。
    //   set_localmatrix で列番号を局所番号にした後、デバイスへ送る前に呼ぶ。
    //   収まらなければならないもの (1 ランクごと):
    //     非零数 nnz_local (= rowptr[n_local]。行ポインタの最大値)
    //     n_local + nadd   (局所列番号の最大値 + 1。nadd は袖の数 = 他ランクから受け取る値の数)
    //   上限は PARCEL_LINT の最大値 (既定 32 ビットで 2^31-1)。試験用に -DPARCEL_LINT_TEST_LIMIT=<値>
    //   で下げられる (PARCEL_LINT の最大値より大きい値は最大値に切り詰める)。
    //   MPI では各ランクの判定を MPI_Allgather で集め、1 ランクでも超えれば全ランクが同じ
    //   std::overflow_error を投げる (1 ランクだけ投げると他のランクが次の集団通信で止まるため)。
    static long long lint_limit(){
      const long long lmax = (long long)std::numeric_limits<PARCEL_LINT>::max();
#ifdef PARCEL_LINT_TEST_LIMIT
      const long long t = (long long)(PARCEL_LINT_TEST_LIMIT);
      return t < lmax ? t : lmax;
#else
      return lmax;
#endif
    }
    static std::string lint_range_hint(){
#if PARCEL_LINT_BITS == 32 && PARCEL_INT_BITS == 64
      return "rebuild with -DPARCEL_LOCAL_INT64 (64-bit rank-local indices) or use more MPI ranks";
#elif PARCEL_LINT_BITS == 32
      return "rebuild without -DPARCEL_INT32 and with -DPARCEL_LOCAL_INT64 (64-bit rank-local indices), or use more MPI ranks";
#else
      return "use more MPI ranks";
#endif
    }
#if PARCEL_MPI
    static void check_lint_range(parcel::comm *comm, size_t n_local, size_t nnz_local, size_t nadd){
      MPI_Comm mc = comm->mpi_comm();
      int np = 1;
      MPI_Comm_size(mc, &np);
      const long long lim = lint_limit();
      const long long nl  = (long long)n_local + (long long)nadd;
      const long long nz  = (long long)nnz_local;
      long long mine[3] = { (nz > lim || nl > lim) ? 1LL : 0LL, nz, nl };
      std::vector<long long> all(3 * (size_t)np);
      MPI_Allgather(mine, 3, MPI_LONG_LONG, all.data(), 3, MPI_LONG_LONG, mc);
      std::string ranks;
      int nbad = 0;
      for(int r = 0; r < np; r++){
	if(all[3*r]){
	  nbad++;
	  ranks += (ranks.empty() ? "" : ",") + std::to_string(r)
	    + " (nnz_local = " + std::to_string(all[3*r+1])
	    + ", n_local + nadd = " + std::to_string(all[3*r+2]) + ")";
	}
      }
      if(nbad == 0) return;
      throw std::overflow_error("parcel::matrix::set_crsMat: rank-local CRS index exceeds PARCEL_LINT range (limit "
				+ std::to_string(lim) + ", PARCEL_LINT is " + std::to_string((int)PARCEL_LINT_BITS)
				+ "-bit) on rank(s) " + ranks + " of " + std::to_string(np)
				+ "; " + lint_range_hint());
    }
#else
    static void check_lint_range(size_t n_local, size_t nnz_local, size_t nadd){
      const long long lim = lint_limit();
      const long long nl  = (long long)n_local + (long long)nadd;
      const long long nz  = (long long)nnz_local;
      if(nz > lim || nl > lim){
	throw std::overflow_error("parcel::matrix::set_crsMat: rank-local CRS index exceeds PARCEL_LINT range (limit "
				  + std::to_string(lim) + ", PARCEL_LINT is " + std::to_string((int)PARCEL_LINT_BITS)
				  + "-bit): nnz_local = " + std::to_string(nz) + ", n_local + nadd = " + std::to_string(nl)
				  + "; " + lint_range_hint());
      }
    }
#endif
    // PARCEL_INT の配列を PARCEL_LINT の配列にする。check_lint_range を通した後に使う。
    //   両者が同じ型 (-DPARCEL_INT32 または -DPARCEL_LOCAL_INT64) なら複写せず元の配列を指す。
    struct lint_array{
      std::vector<PARCEL_LINT> buf;
      PARCEL_LINT *ptr = nullptr;
      lint_array(PARCEL_INT *src, size_t len){
	if(std::is_same<PARCEL_LINT, PARCEL_INT>::value){
	  ptr = reinterpret_cast<PARCEL_LINT *>(src);
	}else{
	  // 1 要素多めに 0 で確保する (crsMat は 0 行・0 非零でも長さ 1 以上で確保し、その長さ分を読むため)
	  buf.assign(len + 1, (PARCEL_LINT)0);
	  for(size_t i = 0; i < len; i++) buf[i] = (PARCEL_LINT)src[i];
	  ptr = buf.data();
	}
      }
      PARCEL_LINT *data(){ return ptr; }
    };

    // CRS 行列を設定する。入力の rowptr_mem / crsCol_mem は大域の型 PARCEL_INT (列番号は大域列番号)。
    //   局所化の後、範囲を検査して (check_lint_range) ランク内の型 PARCEL_LINT に直してからデバイスへ送る。
    //   注意: 呼び出し側の crsCol_mem と値の配列は局所番号の並びに書き換わる。
    template<class Ttype>
    void set_crsMat(
#if PARCEL_MPI
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
#endif		    
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    Ttype *mat_mem){
      require_in_order(q_, "matrix::set_crsMat");   // in-order キューが前提 (上の require_in_order 参照)
      check_int_range(n, nnz);
#if PARCEL_MPI
      check_crs_input(comm, nstart, nend, n, nnz, rowptr_mem, crsCol_mem);
#else
      check_crs_input(n, nnz, rowptr_mem, crsCol_mem);
#endif
#if PARCEL_MPI
      
      // MPIコミュニケータの設定 
      crsMat_.set_comm(comm,nstart,nend);

      // 分散行列用にCRS行列を変換する
      crsMat_.set_localmatrix(
			      nstart,nend,
			      rowptr_mem,
			      crsCol_mem,
			      mat_mem
			      );
      // 局所化した CRS がランク内の型 PARCEL_LINT に収まるかを全ランクで検査する
      //   (通信テーブルを作る前に。収まらなければ全ランクが同じ例外を投げる)
      check_lint_range(comm, n, nnz, crsMat_.n_halo());
      
      // 通信テーブルを作成
      crsMat_.set_queue(q_);   // 通信用の作業域を行列のキューで確保する (set_comm_list の中)
      crsMat_.set_comm_list();
#else
      check_lint_range(n, nnz, 0);
#endif
      
      crsMat_.resize(n,nnz,q_);
      crsMat_.mat_setValue(mat_mem);
      {
	// ランク内の型 PARCEL_LINT に直して送る (同じ型なら複写しない)
	lint_array rp_l(rowptr_mem, n+1);
	crsMat_.rowptr_setValue(rp_l.data());
      }
      {
	lint_array col_l(crsCol_mem, nnz);
	crsMat_.crsCol_setValue(col_l.data());
      }
      // 最大列数の取得
      crsMat_.set_maxCol();
      matrix_type_ = CRS_MATRIX;
      return ;
    }
    
    void set_crsMat(
#if PARCEL_MPI
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
#endif		    
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    double *mat_mem_hi,double *mat_mem_lo
		    ){
      require_in_order(q_, "matrix::set_crsMat");   // in-order キューが前提
      check_int_range(n, nnz);
#if PARCEL_MPI
      check_crs_input(comm, nstart, nend, n, nnz, rowptr_mem, crsCol_mem);
#else
      check_crs_input(n, nnz, rowptr_mem, crsCol_mem);
#endif
#if PARCEL_MPI
      // テンプレート版 set_crsMat と同じく
      //   (1) set_comm (2) set_localmatrix (3) set_comm_list の順に行う。
      //   set_localmatrix は値の配列を列番号と一緒に並べ替えるので、hi/lo を組にした
      //   一時配列で並べ替え、結果を mat_mem_hi / mat_mem_lo に書き戻す
      //   (テンプレート版と同じく、呼び出し側の crsCol_mem と値の配列は書き換わる)。
      crsMat_.set_comm(comm,nstart,nend);
      {
	std::vector<parcel_ddpair> mat_pair(nnz);
	for(size_t i = 0; i < nnz; i++){
	  mat_pair[i].hi = mat_mem_hi[i];
	  mat_pair[i].lo = mat_mem_lo[i];
	}
	crsMat_.set_localmatrix(
				nstart,nend,
				rowptr_mem,
				crsCol_mem,
				mat_pair.data()
				);
	for(size_t i = 0; i < nnz; i++){
	  mat_mem_hi[i] = mat_pair[i].hi;
	  mat_mem_lo[i] = mat_pair[i].lo;
	}
      }
      // 局所化した CRS がランク内の型 PARCEL_LINT に収まるかを全ランクで検査する
      check_lint_range(comm, n, nnz, crsMat_.n_halo());
      crsMat_.set_queue(q_);   // 通信用の作業域を行列のキューで確保する (set_comm_list の中)
      crsMat_.set_comm_list();
#else
      check_lint_range(n, nnz, 0);
#endif
      crsMat_.resize(n,nnz,q_);
      crsMat_.mat_setValue(mat_mem_hi,mat_mem_lo);
      {
	// ランク内の型 PARCEL_LINT に直して送る (同じ型なら複写しない)
	lint_array rp_l(rowptr_mem, n+1);
	crsMat_.rowptr_setValue(rp_l.data());
      }
      {
	lint_array col_l(crsCol_mem, nnz);
	crsMat_.crsCol_setValue(col_l.data());
      }
      crsMat_.set_maxCol();
      matrix_type_ = CRS_MATRIX;
      return ;
    }

#if !PARCEL_MPI
    // MPI なしのビルドでも MPI ありと同じ形 (comm, nstart, nend を先頭に渡す) で呼べるようにする。
    //   1 プロセスが全体を持つので、n = nend - nstart + 1 かつ nstart = 0 でなければ
    //   ならない (違えば std::invalid_argument。PARCEL_NO_CRS_CHECK によらず調べる)。
    //   検査の後は comm を取らない set_crsMat と同じ (列番号・値の配列の扱い、残りの入力検査も同じ)。
    static void check_nompi_block(parcel::comm *comm, PARCEL_INT nstart, PARCEL_INT nend, size_t n){
      if(comm == nullptr){
	throw std::invalid_argument("parcel::matrix::set_crsMat: comm is null");
      }
      if((long long)nend - (long long)nstart + 1 != (long long)n){
	throw std::invalid_argument("parcel::matrix::set_crsMat: invalid input: n (" + std::to_string((long long)n)
				    + ") != nend - nstart + 1 (nstart = " + std::to_string((long long)nstart)
				    + ", nend = " + std::to_string((long long)nend) + ")");
      }
      if(nstart != 0){
	throw std::invalid_argument("parcel::matrix::set_crsMat: built without MPI (PARCEL_MPI=0), so the single process must own all rows"
				    " (nstart = 0, nend = n - 1); got nstart = " + std::to_string((long long)nstart)
				    + ", nend = " + std::to_string((long long)nend));
      }
    }

    template<class Ttype>
    void set_crsMat(
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    Ttype *mat_mem){
      check_nompi_block(comm, nstart, nend, n);
      set_crsMat<Ttype>(n, nnz, rowptr_mem, crsCol_mem, mat_mem);
      comm_ = comm;
      return ;
    }

    void set_crsMat(
		    parcel::comm *comm,PARCEL_INT nstart,PARCEL_INT nend,
		    size_t n,size_t nnz, 
		    PARCEL_INT *rowptr_mem,
		    PARCEL_INT *crsCol_mem,
		    double *mat_mem_hi,double *mat_mem_lo
		    ){
      check_nompi_block(comm, nstart, nend, n);
      set_crsMat(n, nnz, rowptr_mem, crsCol_mem, mat_mem_hi, mat_mem_lo);
      comm_ = comm;
      return ;
    }
#endif

    // -------
    // DIA Matrix
    diaMat<parceltype> *diaMat_data(){
      return &diaMat_;
    }
    template<class Ttype>
    void set_diaMat(
		    size_t n,size_t nnd, 
		    PARCEL_INT *offset_mem,
		    Ttype *mat_mem){
      diaMat_.resize(n,nnd,q_);
      diaMat_.mat_setValue(mat_mem);
      diaMat_.offset_setValue(offset_mem);
      matrix_type_ = DIA_MATRIX;
      return ;
    }

    void set_diaMat(
		    size_t n,size_t nnd, 
		    PARCEL_INT *offset_mem,
		    double *mat_mem_hi,double *mat_mem_lo){
      diaMat_.resize(n,nnd,q_);
      diaMat_.mat_setValue(mat_mem_hi,mat_mem_lo);
      diaMat_.offset_setValue(offset_mem);
      matrix_type_ = DIA_MATRIX;
      return ;
    }

    void set_queue(sycl::queue* q){
      q_ = q;
      return ;
    }

    sycl::queue *queue(){
      return q_;
    }

    size_t matrix_type(){
      return matrix_type_;
    }

  };



}



#endif
