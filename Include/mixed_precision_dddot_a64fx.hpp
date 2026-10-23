// SPDX-License-Identifier: LGPL-3.0-only
// Copyright (c) 日本原子力研究開発機構 システム計算科学センター
#ifndef MIXED_PRECISON_DDDOT_A64FX_HPP_
#define MIXED_PRECISON_DDDOT_A64FX_HPP_

#include <vector>

#include "parcel_nblock.hpp"

ddtype dddot(
	    PARCEL_INT N,
	    double *x_hi,double *x_lo,PARCEL_INT ldx,
	    double *y_hi,double *y_lo,PARCEL_INT ldy,
	    double *work // 作業領域 : (num_work_groups + 1) * 2
	    ){
  // デバイス間同期なし
#if PARCEL_BLOCKED_DOT
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 1024;
  const size_t nlocalblock_half = nlocalblock/2;
#else
   const size_t nblock      = N;
   const size_t nlocalblock = 1;
   const size_t nlocalblock_half = 0;
#endif

  // ブロックごとの部分和 (hi / lo) はホストから直接読める領域に置き、使い回す。
  //   このブロック版は CPU (OpenMP バックエンド) でしか使わない。先頭 nblock 個が hi、続きが lo。
  double *out_hi = dot_work_<double>(dot_work_host_, 2*nblock, true);
  double *out_lo = out_hi + nblock;

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out_hi[ii] = 0.0;
      out_lo[ii] = 0.0;

      double sum_local_hi[nlocalblock];
      double sum_local_lo[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local_hi[i] = 0.0;
	sum_local_lo[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double ddmul_var1;
	  double xy_hi,xy_lo;
	  ddmul_calc1(x_hi[i]   ,x_lo[i]   ,y_hi[i],y_lo[i],xy_hi     ,xy_lo);
	  ddadd_calc1(sum_local_hi[i-iii],sum_local_lo[i-iii],xy_hi  ,xy_lo  ,sum_local_hi[i-iii],sum_local_lo[i-iii]);
	}						    
      }

      double sum_local1_hi = 0.0;
      double sum_local1_lo = 0.0;
      double sum_local2_hi = 0.0;
      double sum_local2_lo = 0.0;
      if(nlocalblock_half!=0){
	for(size_t i=0;i<nlocalblock_half;i++){
	  double ddadd_var1;
	  ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local_hi[i],sum_local_lo[i],sum_local1_hi,sum_local1_lo);
	  double ddadd_var2;						    
	  ddadd_calc2(sum_local2_hi,sum_local2_lo,sum_local_hi[i+nlocalblock_half],sum_local_lo[i+nlocalblock_half],sum_local2_hi,sum_local2_lo);	
	}
	double ddadd_var1;
	ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local2_hi,sum_local2_lo,out_hi[ii],out_lo[ii]);
      }else{
	for(size_t i=0;i<nlocalblock;i++){
	  double ddadd_var1;
	  ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local_hi[i],sum_local_lo[i],sum_local1_hi,sum_local1_lo);
	}
	double ddadd_var1;
	ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local2_hi,sum_local2_lo,out_hi[ii],out_lo[ii]);
      }

    });
  q_[0].wait();
  // ブロックの番号 i の順に足す
  const double *result_hi = out_hi;
  const double *result_lo = out_lo;

  double ret_hi = 0.0;
  double ret_lo = 0.0;
  for(size_t i=0;i<nblock;i++){
    double ddadd_var1;
    ddadd_calc1(ret_hi,ret_lo,result_hi[i],result_lo[i],ret_hi,ret_lo);
  }


  ddtype ret(ret_hi,ret_lo);
      
  return ret;      

}

template<class TdataX,class TdataY>
ddtype dddot(
	    PARCEL_INT N,
	    TdataX *x_hi,PARCEL_INT ldx,
	    TdataY *y_hi,PARCEL_INT ldy,
	    double *work // 作業領域 : (num_work_groups + 1) * 2
	    ){
  // デバイス間同期なし

#if PARCEL_BLOCKED_DOT
  const size_t nblock      = parcel::nblock_runtime();
  const size_t nlocalblock = 1024;
  const size_t nlocalblock_half = nlocalblock/2;
#else
  const size_t nblock      = N;
  const size_t nlocalblock = 1;
  const size_t nlocalblock_half = 0;
#endif

  // ブロックごとの部分和 (hi / lo) はホストから直接読める領域に置き、使い回す。
  //   このブロック版は CPU (OpenMP バックエンド) でしか使わない。先頭 nblock 個が hi、続きが lo。
  double *out_hi = dot_work_<double>(dot_work_host_, 2*nblock, true);
  double *out_lo = out_hi + nblock;

  q_[0].parallel_for(sycl::range<1>{nblock},[=](sycl::id<1> ii){
      out_hi[ii] = 0.0;
      out_lo[ii] = 0.0;

      double sum_local_hi[nlocalblock];
      double sum_local_lo[nlocalblock];
      for(size_t i=0;i<nlocalblock;i++){
	sum_local_hi[i] = 0.0;
	sum_local_lo[i] = 0.0;
      }
      for(size_t iii=ii*nlocalblock;iii<N;iii=iii+nblock*nlocalblock){
	const size_t imax = (iii+nlocalblock) > N ? N : (iii+nlocalblock);
	for(size_t i=iii;i<imax;i++){
	  double ddadd_var1;
	  double ddmul_var1;
	  double xy_hi,xy_lo;
	  double x_lo = 0.0;
	  double y_lo = 0.0;
	  ddmul_calc1(x_hi[i],x_lo,y_hi[i],y_lo,xy_hi,xy_lo);
	  ddadd_calc1(sum_local_hi[i-iii],sum_local_lo[i-iii],xy_hi,xy_lo,sum_local_hi[i-iii],sum_local_lo[i-iii]);
	}						    
      }

      double sum_local1_hi = 0.0;
      double sum_local1_lo = 0.0;
      double sum_local2_hi = 0.0;
      double sum_local2_lo = 0.0;
      if(nlocalblock_half!=0){
	for(size_t i=0;i<nlocalblock_half;i++){
	  double ddadd_var1;
	  ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local_hi[i],sum_local_lo[i],sum_local1_hi,sum_local1_lo);
	  double ddadd_var2;						    
	  ddadd_calc2(sum_local2_hi,sum_local2_lo,sum_local_hi[i+nlocalblock_half],sum_local_lo[i+nlocalblock_half],sum_local2_hi,sum_local2_lo);	
	}
	double ddadd_var1;
	ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local2_hi,sum_local2_lo,out_hi[ii],out_lo[ii]);
      }else{
	for(size_t i=0;i<nlocalblock;i++){
	  double ddadd_var1;
	  ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local_hi[i],sum_local_lo[i],sum_local1_hi,sum_local1_lo);
	}
	double ddadd_var1;
	ddadd_calc1(sum_local1_hi,sum_local1_lo,sum_local2_hi,sum_local2_lo,out_hi[ii],out_lo[ii]);
      }

    });
  q_[0].wait();
  // ブロックの番号 i の順に足す
  const double *result_hi = out_hi;
  const double *result_lo = out_lo;

  double ret_hi = 0.0;
  double ret_lo = 0.0;
  for(size_t i=0;i<nblock;i++){
    double ddadd_var1;
    ddadd_calc1(ret_hi,ret_lo,result_hi[i],result_lo[i],ret_hi,ret_lo);
  }


  ddtype ret(ret_hi,ret_lo);
      
  return ret;      

}

#endif

