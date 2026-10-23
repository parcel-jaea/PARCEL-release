// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MAT_HPP_
#define MAT_HPP_

#include <iostream>
#include <cstdint>
#include <vector>

template<class T>
class crs_mat{
  std::vector<T> A_;
  std::vector<T> Diag_;
  std::vector<PARCEL_INT> crsRow_ptr_;
  std::vector<PARCEL_INT> crsCol_;
  PARCEL_INT nRow_;
  PARCEL_INT nnz_;
public:
  void resize(PARCEL_INT n,PARCEL_INT nnz){
    nRow_ = n;
    nnz_  = nnz;
    A_.resize(nnz,0.0);
    Diag_.resize(n,0.0);
    crsRow_ptr_.resize(n+1,0);
    crsCol_.resize(nnz,0);
  }
  crs_mat(){}
  crs_mat(PARCEL_INT n,PARCEL_INT nnz){
    resize(n,nnz);
  };
  ~crs_mat(){}

  void reset(){
    // std::vector は resize で縮めても capacity を返さない。
    // 呼び出し側は全体行列を作ってから自分の担当行だけ切り出しており、
    // reset 後は参照しない。ここで実際に解放しないと、全 MPI ランクが
    // 全体行列 (1 点あたり 96 B) を最後まで抱えたままになる。
    // サイズ 1 の一時オブジェクトと交換して確保済み領域を手放す。
    std::vector<T>(1,0.0).swap(A_);
    std::vector<T>(1,0.0).swap(Diag_);
    std::vector<PARCEL_INT>(1,0).swap(crsRow_ptr_);
    std::vector<PARCEL_INT>(1,0).swap(crsCol_);
  }

  void setValue_nnz(PARCEL_INT nnz){
    nnz_ = nnz;
  }
  void setValue_nRow(PARCEL_INT n){
    nRow_ = n;
  }
  void setValue_matA(T val,PARCEL_INT id){
    A_[id] = val;
  }
  void setValue_matDiag(T val,PARCEL_INT id){
    Diag_[id] = val;
  }
  void setValue_crsRow_ptr(PARCEL_INT ival,PARCEL_INT id){
    crsRow_ptr_[id] = ival;
  }
  void setValue_crsCol(PARCEL_INT ival,PARCEL_INT id){
    crsCol_[id] = ival;
  }
  PARCEL_INT getValue_nRow(){
    return nRow_;
  }

  void SpMV(
	    std::vector<T> &x,
	    std::vector<T> &y
	    ){
#pragma omp parallel for
    for(auto i = 0; i < nRow_; i++){
      T tmp = 0.0;
      for(auto j = crsRow_ptr_[i]; j < crsRow_ptr_[i+1]; j++){
	auto col = crsCol_[j];
	tmp = tmp + A_[j]*x[col];
      }
      y[i] = tmp;
    }
  }

  
  void show(
	    PARCEL_INT nstart,
	    PARCEL_INT nend	    
	    ){
    for(auto i = nstart; i <= nend; i++){
      for(auto j = crsRow_ptr_[i]; j < crsRow_ptr_[i+1]; j++){
	auto col = crsCol_[j];
	std::cout<<" crs : "<<i<<", "<<col<<" : "<<A_[j]<<std::endl;
      }
    }
  }

  std::vector<T> A_ptr(){
    return A_;
  }
  std::vector<T> Diag_ptr(){
    return Diag_;
  }
  std::vector<PARCEL_INT> crsRow_ptr(){
    return crsRow_ptr_;
  }
  std::vector<PARCEL_INT> crsCol_ptr(){
    return crsCol_;
  }

  T *A_data(){
    return A_.data();
  }
  PARCEL_INT *crsRowptr_data(){
    return crsRow_ptr_.data();
  }
  PARCEL_INT *crsCol_data(){
    return crsCol_.data();
  }

};

#endif
