// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_MAX_A64FX_HPP_
#define MIXED_PRECISON_MAX_A64FX_HPP_

#include "parcel_nblock.hpp"

#include <vector>


template<class TdataX,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx
	 ){

#if PARCEL_BLOCKED_DOT
  const size_t nblock    = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)) ;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
      
  // ブロックごとの最大値はホストから直接読める領域に置き、dot と同じ作業域を使い回す
  //   (CPU の OpenMP バックエンド専用の経路)。
  double *out = dot_work_<double>(dot_work_host_, nblock, true);

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out[ii] = 0.0;
      double sum_local[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tret x_ = x[i];
	  x_ = std::abs(x_);
	  sum_local[i-iii] = sum_local[i-iii] < x_ ? x_ : sum_local[i-iii] ;
	}
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] < sum_local[i] ? sum_local[i] : out[ii];
      }
						  
    });
  q_[0].wait();
  const double *result = out;   // ブロックの番号 i の順に比べる

  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret < result[i] ? result[i] : ret ;
  }
      
      
  if(ret == 0.0)ret = 1.0;

  return ret;
  
  
  if(ret == 0.0)ret = 1.0;

  return ret;

}

template<class TdataX,class TdataY,class Tret>
Tret max(
	 PARCEL_INT N,
	 TdataX *x,
	 PARCEL_INT ldx,
	 TdataY *y,
	 PARCEL_INT ldy){


#if PARCEL_BLOCKED_DOT
  const size_t nblock    = parcel::nblock_runtime();
  const size_t nlocalblock = 4096 * (8/sizeof(TdataX)) ;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
#endif
      
  // ブロックごとの最大値はホストから直接読める領域に置き、dot と同じ作業域を使い回す
  //   (CPU の OpenMP バックエンド専用の経路)。
  double *out = dot_work_<double>(dot_work_host_, nblock, true);

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out[ii] = 0.0;
      double sum_local[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  Tret x_  = x[i];
	  Tret y_  = y[i];
	  Tret xy_ = x_*y_; 
	  xy_ = std::abs(xy_);
	  sum_local[i-iii] = sum_local[i-iii] < xy_ ? xy_ : sum_local[i-iii] ;
	}
      }
      for(size_t i=0;i<nlocalblock;i++){
	double ddadd_var1;						    
	out[ii] = out[ii] < sum_local[i] ? sum_local[i] : out[ii];
      }
						  
    });
  q_[0].wait();
  const double *result = out;   // ブロックの番号 i の順に比べる

  double ret = 0.0;
  for(size_t i=0;i<nblock;i++){
    ret = ret < result[i] ? result[i] : ret ;
  }
      
      
  if(ret == 0.0)ret = 1.0;

  return ret;
  
  
  if(ret == 0.0)ret = 1.0;

  return ret;



}


#endif

